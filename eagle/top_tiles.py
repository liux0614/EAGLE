import os
import argparse
import random
import base64
import io
import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw
import svgwrite
from models.CHIEF import CHIEF

# Allow large image processing
Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate top tiles visualization for patients."
    )
    parser.add_argument("--cohort", type=str, default="dachs", help="Cohort name.")
    parser.add_argument(
        "--patient_id", type=str, default=None, help="Specific patient ID to process."
    )
    parser.add_argument(
        "--num_patients",
        type=int,
        default=None,
        help="Number of random patients to process.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on."
    )
    return parser.parse_args()


def load_chief_model(device):
    """Load the pretrained CHIEF model."""
    model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    weight_path = os.path.join("models", "weights", "CHIEF_pretraining.pth")
    td = torch.load(weight_path, map_location=torch.device(device))
    if "organ_embedding" in td:
        del td["organ_embedding"]
    model.load_state_dict(td, strict=True)
    model.eval().to(device)
    return model


def load_slide_table(cohort):
    """Load the slide table CSV for a given cohort using a relative path."""
    slide_table_path = os.path.join(
        "tables", "slide_tables", f"slide_table_{cohort}.csv"
    )
    slide_table = pd.read_csv(slide_table_path)
    return slide_table


def process_patient(
    patient_id, group, ctp_dir, cache_dir, output_dir, device, model, cohort
):
    """Process a single patient: load features, perform model inference, extract top tiles and generate visualization."""
    all_ctp_feats_list = []
    all_coords_list = []
    slide_indices = []

    # Process each slide in the patient's group
    for _, row in group.iterrows():
        slide_filename = row["FILENAME"]
        slide_id = os.path.splitext(slide_filename)[0]
        h5_ctp = os.path.join(ctp_dir, slide_filename)
        cache_image_path = os.path.join(cache_dir, slide_id, "slide.jpg")

        if not os.path.exists(h5_ctp) or not os.path.exists(cache_image_path):
            continue

        # Load features and coordinates from the h5 file
        with h5py.File(h5_ctp, "r") as h5:
            feats = torch.tensor(h5["feats"][:]).float()
            coords = torch.tensor(h5["coords"][:], dtype=torch.int)

        all_ctp_feats_list.append(feats)
        all_coords_list.append(coords)
        slide_indices.extend([slide_id] * feats.shape[0])

    if not all_ctp_feats_list:
        print(f"No features found for patient {patient_id}")
        return

    # Concatenate features and coordinates from all slides
    all_ctp_feats_cat = torch.cat(all_ctp_feats_list, dim=0).to(device)
    all_coords_cat = torch.cat(all_coords_list, dim=0)
    slide_indices = np.array(slide_indices)

    # Perform model inference to get attention scores
    with torch.no_grad():
        x = all_ctp_feats_cat
        result = model(x)
        attention_raw = result["attention_raw"].cpu()
        num_tiles = attention_raw.size(1)

    # Select top k tiles based on attention scores
    k = 25
    k_min = min(k, num_tiles)
    _, top_k_indices = torch.topk(attention_raw, k_min, dim=1)
    top_tile_indices = top_k_indices.squeeze().numpy()
    top_slide_ids = slide_indices[top_tile_indices]
    top_coords = all_coords_cat[top_tile_indices].numpy()

    # Organize selected tiles by slide
    tiles_by_slide = {}
    for idx, s_id in enumerate(top_slide_ids):
        x_coord, y_coord = top_coords[idx]
        tiles_by_slide.setdefault(s_id, []).append((x_coord, y_coord))

    tile_images = []
    slide_thumbs = []  # Contains (slide_id, coords_list, original_image)
    for s_id, coords_list in tiles_by_slide.items():
        cache_image_path = os.path.join(cache_dir, s_id, "slide.jpg")
        if not os.path.exists(cache_image_path):
            continue
        cache_image = (
            Image.open(cache_image_path).convert("RGB").transpose(Image.TRANSPOSE)
        )
        slide_thumbs.append((s_id, coords_list, cache_image))

        for x_coord, y_coord in coords_list:
            left = x_coord
            upper = y_coord
            right = left + 224
            lower = upper + 224

            w, h = cache_image.size
            # Skip if tile bounds are invalid
            if (
                right <= left
                or lower <= upper
                or left < 0
                or upper < 0
                or right > w
                or lower > h
            ):
                continue

            tile_image = cache_image.crop((left, upper, right, lower))
            tile_images.append(tile_image)

    if not tile_images:
        print(f"No tiles extracted for patient {patient_id}")
        return

    # Create a grid image from tile images
    tiles_per_row = 5
    tile_size = 224
    num_tiles = len(tile_images)
    grid_rows = (num_tiles + tiles_per_row - 1) // tiles_per_row
    grid_width = tiles_per_row * tile_size
    grid_height = grid_rows * tile_size

    grid_image = Image.new("RGB", (grid_width, grid_height), "white")
    draw_grid = ImageDraw.Draw(grid_image)
    for idx, tile_image in enumerate(tile_images):
        row = idx // tiles_per_row
        col = idx % tiles_per_row
        gx = col * tile_size
        gy = row * tile_size
        grid_image.paste(tile_image, (gx, gy))

    # Draw grid lines
    for i in range(1, tiles_per_row):
        x_line = i * tile_size
        draw_grid.line([(x_line, 0), (x_line, grid_height)], fill="black", width=1)
    for i in range(1, grid_rows):
        y_line = i * tile_size
        draw_grid.line([(0, y_line), (grid_width, y_line)], fill="black", width=1)

    # Scale slide thumbnails to fit above the grid
    desired_thumb_height = grid_height / 2
    original_images = [d[2] for d in slide_thumbs]

    def orientation_stats(imgs, rotate=False):
        widths, heights = [], []
        for im in imgs:
            w, h = im.size
            if rotate:
                w, h = h, w
            widths.append(w)
            heights.append(h)
        return sum(widths), max(heights), widths, heights

    sum_w_norm, max_h_norm, widths_norm, heights_norm = orientation_stats(
        original_images, rotate=False
    )
    sum_w_rot, max_h_rot, widths_rot, heights_rot = orientation_stats(
        original_images, rotate=True
    )

    def compute_scale(sum_w, max_h):
        if sum_w == 0 or max_h == 0:
            return 1.0, 1.0, 1.0
        s_w = grid_width / sum_w
        s_h = desired_thumb_height / max_h
        s = min(s_w, s_h)
        return s, s_w, s_h

    s_norm, s_w_norm, s_h_norm = compute_scale(sum_w_norm, max_h_norm)
    s_rot, s_w_rot, s_h_rot = compute_scale(sum_w_rot, max_h_rot)

    # Choose orientation with larger scale
    if s_rot > s_norm:
        s = s_rot
        rotate_thumbs = True
    else:
        s = s_norm
        rotate_thumbs = False

    scaled_thumbs = []
    scaled_slides = []
    for s_id, coords_list, orig_im in slide_thumbs:
        w, h = orig_im.size
        if rotate_thumbs:
            rotated_im = orig_im.rotate(90, expand=True)
            w_new, h_new = rotated_im.size
            w_scaled = int(round(w_new * s))
            h_scaled = int(round(h_new * s))
            im_scaled = rotated_im.resize((w_scaled, h_scaled), Image.LANCZOS)
            scaled_thumbs.append(im_scaled)
            scaled_slides.append((s_id, coords_list, (w, h), im_scaled))
        else:
            w_scaled = int(round(w * s))
            h_scaled = int(round(h * s))
            im_scaled = orig_im.resize((w_scaled, h_scaled), Image.LANCZOS)
            scaled_thumbs.append(im_scaled)
            scaled_slides.append((s_id, coords_list, (w, h), im_scaled))

    final_thumb_height = min(
        max(im.height for im in scaled_thumbs), int(round(desired_thumb_height))
    )
    canvas_width = grid_width
    canvas_height = final_thumb_height + grid_height
    canvas_image = Image.new("RGB", (canvas_width, canvas_height), "white")

    # Position thumbnails on the canvas
    thumb_positions = []
    if len(scaled_thumbs) == 1:
        x_start = (canvas_width - scaled_thumbs[0].width) // 2
        y_start = (final_thumb_height - scaled_thumbs[0].height) // 2
        canvas_image.paste(scaled_thumbs[0], (x_start, y_start))
        thumb_positions.append((x_start, y_start))
    else:
        total_w = sum(im.width for im in scaled_thumbs)
        spacing = (
            (canvas_width - total_w) / (len(scaled_thumbs) + 1)
            if total_w < canvas_width
            else 0
        )
        x_pos = spacing
        for im_scaled in scaled_thumbs:
            y_pos = (final_thumb_height - im_scaled.height) // 2
            canvas_image.paste(im_scaled, (int(round(x_pos)), y_pos))
            thumb_positions.append((int(round(x_pos)), y_pos))
            x_pos += im_scaled.width + spacing

    # Draw tile frames on thumbnails
    draw_canvas = ImageDraw.Draw(canvas_image)
    line_width = 2

    def rotate_coord_90_ccw(x, y, orig_w, orig_h):
        new_x = y
        new_y = orig_w - x
        return new_x, new_y

    for idx, (s_id, coords_list, (orig_w, orig_h), im_scaled) in enumerate(
        scaled_slides
    ):
        tx, ty = thumb_positions[idx]
        for x, y in coords_list:
            if rotate_thumbs:
                rx, ry = rotate_coord_90_ccw(x, y, orig_w, orig_h)
                left = tx + rx * s
                upper = ty + (ry - 224) * s
            else:
                left = tx + x * s
                upper = ty + y * s
            right = left + 224 * s
            lower = upper + 224 * s
            draw_canvas.rectangle(
                [left, upper, right, lower], outline="black", width=line_width
            )

    # Paste tile grid below thumbnails
    canvas_image.paste(grid_image, (0, final_thumb_height))

    # Add a scale bar based on background brightness
    scale_bar_pixels_thumb = 2500 * s  # Adjust for scale (e.g., 5mm)
    bar_x = 10
    bar_y = final_thumb_height - 20
    if bar_y < 0:
        bar_y = 10

    sample_height = 10  # Height to sample for brightness
    left, right = bar_x, bar_x + scale_bar_pixels_thumb
    upper = max(0, bar_y - sample_height // 2)
    lower = min(canvas_height, bar_y + sample_height // 2)
    sampling_area = canvas_image.crop((left, upper, right, lower)).convert("L")
    avg_brightness = np.mean(np.array(sampling_area))
    bar_color = "black" if avg_brightness > 128 else "white"
    draw_canvas.line(
        [(bar_x, bar_y), (bar_x + scale_bar_pixels_thumb, bar_y)],
        fill=bar_color,
        width=5,
    )

    # Save visualization as JPG
    output_jpg_path = os.path.join(output_dir, f"{patient_id}.jpg")
    canvas_image.save(output_jpg_path, quality=95)

    # Embed the image in an SVG using base64 encoding
    output_svg_path = os.path.join(output_dir, f"{patient_id}.svg")
    buffer = io.BytesIO()
    canvas_image.save(buffer, format="PNG")
    png_data = buffer.getvalue()
    base64_data = base64.b64encode(png_data).decode("utf-8")
    data_uri = "data:image/png;base64," + base64_data

    dwg = svgwrite.Drawing(output_svg_path, size=(canvas_width, canvas_height))
    dwg.add(dwg.image(href=data_uri, insert=(0, 0), size=(canvas_width, canvas_height)))
    dwg.save()

    print(
        f"Saved visualization for patient {patient_id} at {output_svg_path} and {output_jpg_path}"
    )


def main(cohorts=None, patient_ids=None, num_patients=None, device="cpu"):
    """Main routine for processing patients and generating visualizations."""
    cache_sup_dir = os.path.join("cache", "images")
    if cohorts is None:
        cohorts = os.listdir(cache_sup_dir)
    for cohort in cohorts:
        ctp_dir = os.path.join(
            "tile_features",
            "2mpp",
            cohort,
            "ctranspath",
            "STAMP_raw_xiyuewang-ctranspath-7c998680",
        )
        cache_dir = os.path.join("cache", "images", cohort)
        output_dir = os.path.join("output", "toptiles", f"{cohort}_A")
        os.makedirs(output_dir, exist_ok=True)

        model = load_chief_model(device)
        slide_table = load_slide_table(cohort)
        slide_table["PATIENT"] = slide_table["PATIENT"].astype(str)
        patient_groups = slide_table.groupby("PATIENT")

        if patient_ids:
            patients_to_process = patient_ids
        elif num_patients:
            all_patient_ids = list(patient_groups.groups.keys())
            patients_to_process = random.sample(all_patient_ids, num_patients)
        else:
            print("Please specify either --patient_id or --num_patients")
            return

        for pid in tqdm(patients_to_process, desc="Patients"):
            if pid not in patient_groups.groups:
                print(f"Patient {pid} not found in cohort {cohort}")
                continue
            group = patient_groups.get_group(pid)
            process_patient(
                pid, group, ctp_dir, cache_dir, output_dir, device, model, cohort
            )


if __name__ == "__main__":
    main(num_patients=5)

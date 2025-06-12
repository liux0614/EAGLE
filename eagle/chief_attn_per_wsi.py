#!/usr/bin/env python3
"""
Extract Eagle 2mpp slide embeddings using CTransPath features
and cache images extracted with https://github.com/KatherLab/STAMP-Benchmark.
The script:
  - Loads CTransPath features and computes CHIEF attention.
  - Selects the top 25 tiles.
  - Loads raw image tiles (224x224 crops from cache images,
    Transpose applied because of STAMP bug).
  - Processes the 25 tiles with Virchow2 to extract tile embeddings.
  - Averages the 25 tile embeddings to produce the Eagle slide embedding and saves it.
  - Optionally generates a visualization of the selected top tiles.
"""

import argparse
import itertools
import os
import sys

import cv2
import h5py
import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import login
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image, ImageDraw
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.CHIEF import CHIEF

# Allow processing of very large images
Image.MAX_IMAGE_PIXELS = None


# ---------------------- Model Loading ---------------------- #
def load_chief_model(device):
    """Load the CHIEF model for attention extraction."""
    model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    chief_weights_path = os.path.join("model_weights", "CHIEF_pretraining.pth")
    td = torch.load(chief_weights_path, map_location=device)
    if "organ_embedding" in td:
        del td["organ_embedding"]
    model.load_state_dict(td, strict=True)
    model.eval().to(device)
    return model


# ---------------------- Helper Functions ---------------------- #
def batched(iterable, batch_size):
    """Batch an iterable into lists of length batch_size."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def orientation_stats(imgs, rotate=False):
    widths, heights = [], []
    for im in imgs:
        w, h = im.size
        if rotate:
            w, h = h, w
        widths.append(w)
        heights.append(h)
    return sum(widths), max(heights), widths, heights


def compute_scale(sum_w, max_h, grid_width, desired_thumb_height):
    if sum_w == 0 or max_h == 0:
        return 1.0
    s_w = grid_width / sum_w
    s_h = desired_thumb_height / max_h
    return min(s_w, s_h)


def rotate_coord_90_ccw(x, y, orig_w, orig_h):
    new_x = y
    new_y = orig_w - x
    return new_x, new_y


# ---------------------- Visualization ---------------------- #
def highlight_tiles(slide_id, tiles_by_slide, cache_dir, vis_output_dir):
    # Build tile grid
    tile_images = []
    attn_scores = []
    tile_list = tiles_by_slide[slide_id]
    for _, _, tile_img, attn_score in tile_list:
        tile_images.append(tile_img)
        attn_scores.append(attn_score)
    if not tile_images:
        print(f"No tile images available for visualization of patient {slide_id}")
        return

    tiles_per_row = 5
    tile_size = 224
    num_tiles = len(tile_images)
    grid_rows = (num_tiles + tiles_per_row - 1) // tiles_per_row
    grid_width = tiles_per_row * tile_size
    grid_height = grid_rows * tile_size

    grid_image = Image.new("RGB", (grid_width, grid_height), "white")
    draw_grid = ImageDraw.Draw(grid_image)
    for idx, tile in enumerate(tile_images):
        row = idx // tiles_per_row
        col = idx % tiles_per_row
        gx = col * tile_size
        gy = row * tile_size
        grid_image.paste(tile, (gx, gy))
    for i in range(1, tiles_per_row):
        draw_grid.line(
            [(i * tile_size, 0), (i * tile_size, grid_height)],
            fill="black",
            width=1,
        )
    for i in range(1, grid_rows):
        draw_grid.line(
            [(0, i * tile_size), (grid_width, i * tile_size)],
            fill="black",
            width=1,
        )

    # Build slide thumbnails with scaling and rotation as needed.
    slide_thumbs = []
    # for slide_id, tile_list in tiles_by_slide.items():
    tile_list = tiles_by_slide[slide_id]
    slide_cache_path = os.path.join(cache_dir, slide_id, "slide.jpg")
    if not os.path.exists(slide_cache_path):
        return
    try:
        slide_img = (
            Image.open(slide_cache_path).convert("RGB").transpose(Image.TRANSPOSE)
        )
    except Exception:
        return
    coords = [(x, y) for (x, y, _, _) in tile_list]
    slide_thumbs.append((slide_id, coords, slide_img))
    if not slide_thumbs:
        print(f"No slide thumbnails available for visualization of patient {slide_id}")
        return

    desired_thumb_height = grid_height / 2
    original_images = [item[2] for item in slide_thumbs]
    sum_w_norm, max_h_norm, _, _ = orientation_stats(original_images, rotate=False)
    sum_w_rot, max_h_rot, _, _ = orientation_stats(original_images, rotate=True)
    s_norm = compute_scale(sum_w_norm, max_h_norm, grid_width, desired_thumb_height)
    s_rot = compute_scale(sum_w_rot, max_h_rot, grid_width, desired_thumb_height)
    rotate_thumbs = s_rot > s_norm
    s = s_rot if rotate_thumbs else s_norm

    scaled_thumbs = []
    scaled_slides = []
    for s_id, coords, orig_im in slide_thumbs:
        w, h = orig_im.size
        if rotate_thumbs:
            rotated_im = orig_im.rotate(90, expand=True)
            w_new, h_new = rotated_im.size
            im_scaled = rotated_im.resize(
                (int(round(w_new * s)), int(round(h_new * s))), Image.LANCZOS
            )
        else:
            im_scaled = orig_im.resize(
                (int(round(w * s)), int(round(h * s))), Image.LANCZOS
            )
        scaled_thumbs.append(im_scaled)
        scaled_slides.append((s_id, coords, (w, h), im_scaled))
    final_thumb_height = min(
        max(im.height for im in scaled_thumbs),
        int(round(desired_thumb_height)),
    )
    canvas_width = grid_width
    canvas_height = final_thumb_height + grid_height
    canvas_image = Image.new("RGB", (canvas_width, canvas_height), "white")

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

    draw_canvas = ImageDraw.Draw(canvas_image)
    line_width = 2
    for idx, (s_id, coords, (orig_w, orig_h), im_scaled) in enumerate(scaled_slides):
        tx, ty = thumb_positions[idx]
        for x, y in coords:
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

    canvas_image.paste(grid_image, (0, final_thumb_height))
    scale_bar_pixels_thumb = 2500 * s
    bar_x = 10
    bar_y = final_thumb_height - 20 if final_thumb_height - 20 >= 0 else 10
    sample_height = 10
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

    os.makedirs(vis_output_dir, exist_ok=True)
    output_jpg_path = os.path.join(vis_output_dir, f"{slide_id}.jpg")
    canvas_image.save(output_jpg_path, quality=95)
    print(f"Saved visualization for slide {slide_id} at: {output_jpg_path}")


PALETTE = ["#FFFFFF", "#ADD8E6", "#FFFF00", "#FFA500", "#FF0000"]
DISCRETE_CMAP = LinearSegmentedColormap.from_list(name="white2red", colors=PALETTE)


def overlay_attention_heatmap(
    slide_id, coords, attn_scores, cache_dir, vis_output_dir, alpha=0.5
):
    # Build tile grid
    tile_size = 224

    slide_cache_path = os.path.join(cache_dir, slide_id, "slide.jpg")
    if not os.path.exists(slide_cache_path):
        return
    try:
        slide_img = Image.open(slide_cache_path).convert("RGB")
    except Exception:
        return

    orig_w, orig_h = slide_img.size
    n_rows, n_cols = orig_h // tile_size, orig_w // tile_size
    attn_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    for i in range(len(coords)):
        x, y = coords[i]
        attn_score = attn_scores[i]
        r, c = int(y // tile_size), int(x // tile_size)
        attn_grid[r, c] = attn_score

    attn_grid = cv2.GaussianBlur(attn_grid, (0, 0), 1.5)
    heat_array = cv2.resize(
        attn_grid, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )

    vmin, vmax = float(attn_grid.min()), float(attn_grid.max())
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6)

    fig, ax = plt.subplots(figsize=(orig_w / 100, orig_h / 100), dpi=100)
    ax.imshow(slide_img)
    hm = ax.imshow(
        heat_array, cmap=DISCRETE_CMAP, norm=norm, alpha=alpha, origin="upper"
    )
    ax.axis("off")
    cbar = fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_ticks(bounds)
    cbar.ax.tick_params(labelsize=64)
    cbar.set_label("attention score", fontsize=64)
    fig.tight_layout()

    os.makedirs(vis_output_dir, exist_ok=True)
    output_jpg_path = os.path.join(vis_output_dir, f"{slide_id}.jpg")
    fig.savefig(output_jpg_path, dpi=100, bbox_inches="tight")


# ---------------------- Patient Processing ---------------------- #
def process_patient(
    patient_id,
    wsi_filename,
    ctp_dir,
    cache_dir,
    embed_h5,
    device,
    chief_model,
    visualize,
    vis_output_dir,
):
    """
    Process one patient:
      - Load ctranspath features and coordinates.
      - Compute CHIEF attention and select the top 25 tiles.
      - Crop raw 224x224 tiles from cached slide images.
      - Extract tile features via Virchow2 and average class tokens.
      - Save the Eagle embedding and optionally generate a visualization.
    """
    slide_filename = wsi_filename
    slide_id = os.path.splitext(slide_filename)[0]
    h5_ctp = os.path.join(ctp_dir, slide_filename)
    try:
        with h5py.File(h5_ctp, "r") as f:
            feats = torch.tensor(f["feats"][:], dtype=torch.float32)
            coords = torch.tensor(f["coords"][:], dtype=torch.int)
            embed_h5.create_dataset("feats", data=f["feats"][:])
            embed_h5.create_dataset("coords", data=f["coords"][:])
    except Exception as e:
        print(f"Error reading {h5_ctp}: {e}")
        print(f"No ctranspath features for patient {patient_id}")
        return
    feats_list = [feats]
    coords_list = [coords]
    all_feats = torch.cat(feats_list, dim=0).to(device)
    all_coords = torch.cat(coords_list, dim=0)
    with torch.no_grad():
        result = chief_model(all_feats)
        attention_raw = result["attention_raw"].squeeze(0).cpu()

    print(attention_raw.shape)
    embed_h5.create_dataset("attention", data=attention_raw)
    heatmap_dir = os.path.join(vis_output_dir, "heatmap")
    overlay_attention_heatmap(
        slide_id, all_coords, attention_raw, cache_dir, heatmap_dir
    )

    k = min(25, attention_raw.shape[0])
    topk_values, topk_indices = torch.topk(attention_raw, k)

    top_indices = topk_indices.numpy()
    top_coords = all_coords[top_indices].numpy()
    tile_images = []
    tiles_by_slide = {}
    for i in range(len(top_indices)):
        coord = top_coords[i]
        attn_score = topk_values[i]
        slide_cache_path = os.path.join(cache_dir, slide_id, "slide.jpg")
        if not os.path.exists(slide_cache_path):
            continue
        try:
            slide_img = (
                Image.open(slide_cache_path).convert("RGB").transpose(Image.TRANSPOSE)
            )
        except Exception as e:
            print(f"Error opening {slide_cache_path}: {e}")
            continue
        x, y = int(coord[0]), int(coord[1])
        tile_img = slide_img.crop((x, y, x + 224, y + 224))
        tile_images.append(tile_img)
        tiles_by_slide.setdefault(slide_id, []).append((x, y, tile_img, attn_score))
    if not tile_images:
        print(f"No tile images extracted for patient {patient_id}")
        return
    if visualize:
        tiles_hightligh_dir = os.path.join(vis_output_dir, "tiles_highlight")
        highlight_tiles(slide_id, tiles_by_slide, cache_dir, tiles_hightligh_dir)


# ---------------------- Main ---------------------- #
def main(args):
    device = torch.device(args.device)
    chief_model = load_chief_model(device)

    data_dir = args.data_dir
    csv_dir = args.csv_dir
    dst_dir = args.dst_dir

    slide_table_dir = os.path.join(csv_dir, "tables", "slide_tables")
    cache_dir = os.path.join(data_dir, "cache", "images", "2mpp")
    ctp_dir = os.path.join(data_dir, "tile_features", "2mpp", "ctranspath")
    embed_output_dir = os.path.join(dst_dir, "features", "2mpp")
    vis_output_dir = os.path.join(dst_dir, "visualization", "2mpp")
    os.makedirs(embed_output_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(vis_output_dir, exist_ok=True)

    slide_table_path = os.path.join(slide_table_dir, f"slide_table.csv")
    try:
        slide_table = pd.read_csv(slide_table_path)
    except Exception as e:
        ValueError(f"Error reading slide table: {e}")
    if slide_table.empty:
        ValueError(f"Empty slide table")

    slide_table["PATIENT"] = slide_table["PATIENT"].astype(str)
    patient_groups = slide_table.groupby("PATIENT")
    wsi_filenames = slide_table["FILENAME"].unique()
    print(
        f"Processing with {len(patient_groups)} patients, {len(wsi_filenames)} WSI..."
    )
    for index, row in tqdm(slide_table.iterrows(), desc=f"Eagle", leave=False):
        patient_id = row["PATIENT"]
        wsi_filename = row["FILENAME"]
        embed_h5_path = os.path.join(embed_output_dir, wsi_filename)
        embed_h5 = h5py.File(embed_h5_path, "w")
        process_patient(
            patient_id,
            wsi_filename,
            ctp_dir,
            cache_dir,
            embed_h5,
            device,
            chief_model,
            visualize=args.visualize,
            vis_output_dir=vis_output_dir,
        )
        embed_h5.close()
    print("Feature extraction completed successfully.")


# ---------------------- Entry Point ---------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Eagle 2mpp slide embeddings using Virchow2 on top tiles."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/N/project/mimic_llm/xiang/Data/MultiModality/TCGA/jz_hunan/stamp",
        help="data dir",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="/N/project/mimic_llm/xiang/Data/MultiModality/TCGA/jz_hunan/eagle",
        help="data dir",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="/N/project/mimic_llm/xiang/Data/MultiModality/TCGA/jz_hunan/eagle",
        help="data dir",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the models on.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also generate visualization of the selected top tiles.",
    )
    args = parser.parse_args()
    main(args)

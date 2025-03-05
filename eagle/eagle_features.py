import os
import h5py
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.CHIEF import CHIEF

# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CHIEF model and weights from relative paths
model = CHIEF(size_arg="small", dropout=True, n_classes=2)
weight_path = os.path.join("model_weights", "CHIEF_pretraining.pth")
td = torch.load(weight_path, map_location=device)
if "organ_embedding" in td:
    del td["organ_embedding"]
model.load_state_dict(td, strict=True)
model.eval().to(device)

# Folder-model mapping as provided
folder_model_mapping = {
    "halfmpp": ["chief", "eagle"],
    "kathermpp": ["chief", "eagle"],
    "2mpp": ["chief", "eagle"],
    "eagletiles": ["top5", "top10", "top25", "top50", "top100"],
    "eagleweight": ["top5", "top10", "top25", "top50", "top100", "all"],
    "eaglefm": [
        "conch",
        "virchow-class",
        "ctranspath",
        "prov-gigapath",
        "virchow2-class",
    ],
}
folders = list(folder_model_mapping.keys())


# Get the base feature directory for embeddings according to folder type.
def get_base_feature_dir_for_embeddings(folder):
    if folder == "halfmpp":
        return os.path.join("tile_features", "halfmpp")
    elif folder == "kathermpp":
        return os.path.join("tile_features", "kathermpp")
    elif folder in ["2mpp", "eagletiles", "eagleweight", "eaglefm"]:
        return os.path.join("tile_features", "2mpp")
    else:
        return os.path.join("tile_features", "2mpp")


# Helper to load tile features for a given feature modality from HDF5 files
def load_fm_features(fm_path, slide_table, device):
    fm_feats_list = []
    for _, row in slide_table.iterrows():
        slide_filename = row["FILENAME"]
        h5_fm = os.path.join(fm_path, slide_filename)
        if not os.path.exists(h5_fm):
            continue
        with h5py.File(h5_fm, "r") as f:
            feats = f["feats"][:]
        fm_feats_list.append(torch.tensor(feats, dtype=torch.float32, device=device))
    if len(fm_feats_list) == 0:
        return None
    return torch.cat(fm_feats_list, dim=0)


def get_topk_tiles(features, attention_raw, k, patient_id):
    num_tiles = attention_raw.size(1)
    if num_tiles < k:
        print(
            f"WARNING: Patient {patient_id} has only {num_tiles} tiles, using all available tiles for top-{k}."
        )
        k = num_tiles
    top_k_values, top_k_indices = torch.topk(attention_raw, k, dim=1)
    selected_feats = features[top_k_indices.squeeze()]
    return selected_feats, top_k_values.squeeze()


slide_emb_dir = os.path.join("output", "features")


# Helper to determine the feature modality path for a given folder, fm, and cohort.
def get_fm_path(folder, fm, cohort):
    suffix_map = {
        "virchow2-class": "STAMP_raw_virchow2",
        "virchow-class": "STAMP_raw_virchow",
        "conch": "STAMP_raw_conch",
        "ctranspath": "STAMP_raw_xiyuewang-ctranspath-7c998680",
        "prov-gigapath": "STAMP_raw_prov-gigapath",
    }
    base_emb_dir = get_base_feature_dir_for_embeddings(folder)
    return os.path.join(base_emb_dir, cohort, fm, suffix_map[fm])


# Helper to determine the CHIEF ctranspath feature directory path based on folder type.
def get_chief_ctp_path(cohort, folder):
    if folder == "halfmpp":
        ctp_base = os.path.join("tile_features", "halfmpp")
    elif folder == "kathermpp":
        ctp_base = os.path.join("tile_features", "kathermpp")
    else:
        ctp_base = os.path.join("tile_features", "2mpp")
    return os.path.join(
        ctp_base, cohort, "ctranspath", "STAMP_raw_xiyuewang-ctranspath-7c998680"
    )


# Process each folder and model for all available cohorts based on slide tables.
for folder in tqdm(folders, desc="Processing Folders"):
    models = folder_model_mapping[folder]
    base_emb_dir = get_base_feature_dir_for_embeddings(folder)

    slide_table_dir = os.path.join("tables", "slide_tables")
    if not os.path.exists(slide_table_dir):
        print("Slide table directory not found.")
        continue

    # Determine cohorts from slide table files (filenames: slide_table_<cohort>.csv)
    slide_tables = [
        f
        for f in os.listdir(slide_table_dir)
        if f.startswith("slide_table_") and f.endswith(".csv")
    ]
    cohorts = [f.replace("slide_table_", "").replace(".csv", "") for f in slide_tables]

    # Predefine extraction strategies for different models:
    # - For 'chief': store CHIEF embedding directly.
    # - For 'eagle' (in halfmpp/kathermpp/2mpp): top25 mean from virchow2-class.
    # - For 'eagletiles': top(k) mean from virchow2-class, where k is parsed from model name.
    # - For 'eagleweight': weighted top-k (or all) from virchow2-class.
    # - For 'eaglefm': top25 mean from multiple modalities.

    def get_fm_path_local(folder, fm, cohort):
        return get_fm_path(folder, fm, cohort)

    # Iterate through cohorts for the current folder
    for cohort in tqdm(cohorts, desc=f"Extracting {folder}", leave=False):
        slide_table_path = os.path.join(slide_table_dir, f"slide_table_{cohort}.csv")
        if not os.path.exists(slide_table_path):
            continue
        slide_table = pd.read_csv(slide_table_path)
        if slide_table.empty:
            continue

        # Get CHIEF attention and embeddings from ctranspath features.
        ctp_path_chief = get_chief_ctp_path(cohort, folder)
        patient_groups = slide_table.groupby("PATIENT")

        valid_patients = []
        for patient_id, group in patient_groups:
            patient_id = str(patient_id)
            for _, row in group.iterrows():
                slide_filename = row["FILENAME"]
                h5_ctp = os.path.join(ctp_path_chief, slide_filename)
                if os.path.exists(h5_ctp):
                    valid_patients.append(patient_id)
                    break

        # Dictionaries to store extracted CHIEF attention and embedding per patient.
        patient_chief_attention = {}
        patient_chief_embedding = {}

        # Skip processing if output for all models already exists.
        already_done_for_this_cohort = True
        for m_check in models:
            out_file_check = os.path.join(
                slide_emb_dir, f"{folder}/{m_check}".lower(), f"{cohort}.h5"
            )
            if not os.path.exists(out_file_check):
                print(f"{out_file_check} doesn't exist yet")
                already_done_for_this_cohort = False
                break
            with h5py.File(out_file_check, "r") as f_chk:
                for pid in valid_patients:
                    if pid not in f_chk:
                        print(
                            f"{pid} missing in output file; valid patients: {len(valid_patients)}"
                        )
                        already_done_for_this_cohort = False
                        break
            if not already_done_for_this_cohort:
                break

        if already_done_for_this_cohort:
            print(
                f"Patients in {cohort} for folder {folder} already complete => Skipped"
            )
            continue

        # Process each patient: load ctranspath features and extract attention and embedding using CHIEF.
        for patient_id, group in patient_groups:
            patient_id = str(patient_id)
            all_ctp_feats_list = []
            extractor = None
            for _, row in group.iterrows():
                slide_filename = row["FILENAME"]
                h5_ctp = os.path.join(ctp_path_chief, slide_filename)
                if not os.path.exists(h5_ctp):
                    continue
                with h5py.File(h5_ctp, "r") as f:
                    ctp_feats = f["feats"][:]
                    extractor = f.attrs["extractor"]
                all_ctp_feats_list.append(
                    torch.tensor(ctp_feats, dtype=torch.float32, device=device)
                )
            if len(all_ctp_feats_list) == 0:
                continue
            all_ctp_feats_cat = torch.cat(all_ctp_feats_list, dim=0)
            with torch.no_grad():
                result = model(all_ctp_feats_cat)
                attention_raw = result["attention_raw"]  # Shape: [1, N]
                slide_embeddings_chief = result["WSI_feature"].squeeze(0)
            patient_chief_attention[patient_id] = (
                attention_raw,
                all_ctp_feats_cat,
                extractor,
            )
            patient_chief_embedding[patient_id] = slide_embeddings_chief

        print(
            f"Patients in {cohort}: {len(patient_chief_attention)} extracted attention from {ctp_path_chief}"
        )

        # For each model in the mapping, produce and save embeddings.
        for m in models:
            combined_model_name = f"{folder}/{m}".lower()
            output_base = os.path.join(slide_emb_dir, combined_model_name)
            os.makedirs(output_base, exist_ok=True)

            if folder in ["halfmpp", "kathermpp", "2mpp"] and m in ["chief", "eagle"]:
                # For 'chief': store CHIEF embedding; for 'eagle': use top25 mean from virchow2-class.
                if m == "chief":
                    chief_file = os.path.join(output_base, f"{cohort}.h5")
                    with h5py.File(chief_file, "a") as f_chief:
                        f_chief.attrs["extractor"] = "ctranspath"
                        for patient_id in patient_chief_embedding:
                            emb = patient_chief_embedding[patient_id].cpu().numpy()
                            if patient_id not in f_chief:
                                f_chief.create_dataset(patient_id, data=emb)
                else:
                    k = 25
                    fm = "virchow2-class"
                    fm_path = get_fm_path_local(folder, fm, cohort)
                    out_file = os.path.join(output_base, f"{cohort}.h5")
                    with h5py.File(out_file, "a") as f_out:
                        f_out.attrs["extractor"] = fm
                        for patient_id, group in patient_groups:
                            patient_id = str(patient_id)
                            if patient_id not in patient_chief_attention:
                                continue
                            attention_raw, _, _ = patient_chief_attention[patient_id]
                            all_fm_feats_list = []
                            for _, row in group.iterrows():
                                slide_filename = row["FILENAME"]
                                h5_fm = os.path.join(fm_path, slide_filename)
                                if not os.path.exists(h5_fm):
                                    continue
                                with h5py.File(h5_fm, "r") as ff:
                                    feats = ff["feats"][:]
                                all_fm_feats_list.append(
                                    torch.tensor(
                                        feats, dtype=torch.float32, device=device
                                    )
                                )
                            if len(all_fm_feats_list) == 0:
                                continue
                            fm_feats = torch.cat(all_fm_feats_list, dim=0)
                            selected_feats, _ = get_topk_tiles(
                                fm_feats, attention_raw, k, patient_id
                            )
                            slide_embedding_eagle = torch.mean(selected_feats, dim=0)
                            if patient_id not in f_out:
                                f_out.create_dataset(
                                    patient_id, data=slide_embedding_eagle.cpu().numpy()
                                )

            elif folder == "eagletiles":
                # m is in [top5, top10, top25, top50, top100] for eagle tiles.
                fm = "virchow2-class"
                fm_path = get_fm_path_local(folder, fm, cohort)
                k_val = int(m.replace("top", ""))
                out_file = os.path.join(output_base, f"{cohort}.h5")
                with h5py.File(out_file, "a") as f_out:
                    f_out.attrs["extractor"] = fm
                    for patient_id, group in patient_groups:
                        patient_id = str(patient_id)
                        if patient_id not in patient_chief_attention:
                            continue
                        attention_raw, _, _ = patient_chief_attention[patient_id]
                        all_fm_feats_list = []
                        for _, row in group.iterrows():
                            slide_filename = row["FILENAME"]
                            h5_fm = os.path.join(fm_path, slide_filename)
                            if not os.path.exists(h5_fm):
                                continue
                            with h5py.File(h5_fm, "r") as ff:
                                feats = ff["feats"][:]
                            all_fm_feats_list.append(
                                torch.tensor(feats, dtype=torch.float32, device=device)
                            )
                        if len(all_fm_feats_list) == 0:
                            continue
                        fm_feats = torch.cat(all_fm_feats_list, dim=0)
                        selected_feats, _ = get_topk_tiles(
                            fm_feats, attention_raw, k_val, patient_id
                        )
                        slide_embedding = torch.mean(selected_feats, dim=0)
                        if patient_id not in f_out:
                            f_out.create_dataset(
                                patient_id, data=slide_embedding.cpu().numpy()
                            )

            elif folder == "eagleweight":
                # m is in [top5, top10, top25, top50, top100, all] for weighted embeddings.
                fm = "virchow2-class"
                fm_path = get_fm_path_local(folder, fm, cohort)
                k_val = "all" if m == "all" else int(m.replace("top", ""))
                out_file = os.path.join(output_base, f"{cohort}.h5")
                with h5py.File(out_file, "a") as f_out:
                    f_out.attrs["extractor"] = fm
                    for patient_id, group in patient_groups:
                        patient_id = str(patient_id)
                        if patient_id not in patient_chief_attention:
                            continue
                        attention_raw, _, _ = patient_chief_attention[patient_id]
                        all_fm_feats_list = []
                        for _, row in group.iterrows():
                            slide_filename = row["FILENAME"]
                            h5_fm = os.path.join(fm_path, slide_filename)
                            if not os.path.exists(h5_fm):
                                continue
                            with h5py.File(h5_fm, "r") as ff:
                                feats = ff["feats"][:]
                            all_fm_feats_list.append(
                                torch.tensor(feats, dtype=torch.float32, device=device)
                            )
                        if len(all_fm_feats_list) == 0:
                            continue
                        fm_feats = torch.cat(all_fm_feats_list, dim=0)
                        N = fm_feats.size(0)
                        actual_k = N if k_val == "all" else k_val
                        selected_feats, top_values = get_topk_tiles(
                            fm_feats, attention_raw, actual_k, patient_id
                        )
                        weights = F.softmax(top_values, dim=0).unsqueeze(1)
                        slide_embedding = torch.sum(selected_feats * weights, dim=0)
                        if patient_id not in f_out:
                            f_out.create_dataset(
                                patient_id, data=slide_embedding.cpu().numpy()
                            )

            elif folder == "eaglefm":
                # m is one of [conch, virchow-class, ctranspath, prov-gigapath, virchow2-class] for eaglefm.
                k = 25
                fm = m
                fm_path = get_fm_path_local(folder, fm, cohort)
                out_file = os.path.join(output_base, f"{cohort}.h5")
                with h5py.File(out_file, "a") as f_out:
                    f_out.attrs["extractor"] = fm
                    for patient_id, group in patient_groups:
                        patient_id = str(patient_id)
                        if patient_id not in patient_chief_attention:
                            continue
                        attention_raw, _, _ = patient_chief_attention[patient_id]
                        all_fm_feats_list = []
                        for _, row in group.iterrows():
                            slide_filename = row["FILENAME"]
                            h5_fm = os.path.join(fm_path, slide_filename)
                            if not os.path.exists(h5_fm):
                                continue
                            with h5py.File(h5_fm, "r") as ff:
                                feats = ff["feats"][:]
                            all_fm_feats_list.append(
                                torch.tensor(feats, dtype=torch.float32, device=device)
                            )
                        if len(all_fm_feats_list) == 0:
                            continue
                        fm_feats = torch.cat(all_fm_feats_list, dim=0)
                        selected_feats, _ = get_topk_tiles(
                            fm_feats, attention_raw, k, patient_id
                        )
                        slide_embedding = torch.mean(selected_feats, dim=0)
                        if patient_id not in f_out:
                            f_out.create_dataset(
                                patient_id, data=slide_embedding.cpu().numpy()
                            )

print("Feature extraction completed successfully.")

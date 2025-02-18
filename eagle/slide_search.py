import os
import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import random
from tqdm import tqdm

# Configuration
clinical_tables_path = os.path.join("tables", "clinical_tables_filtered")
embeddings_path = os.path.join("output","features", "2mpp", "eagle")
output_dir = os.path.join("output", "slide_search")
excluded_cohorts = {'tcga_luad', 'cptac_luad'}
random_seed = 42
num_random_patients = 5
top_k = 3

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(random_seed)

# Step 1: Identify cohorts by listing clinical table files, excluding specified cohorts
all_cohorts = []
for file in os.listdir(clinical_tables_path):
    if file.endswith(".csv"):
        cohort = file.replace(".csv", "")
        if cohort not in excluded_cohorts:
            all_cohorts.append(cohort)

print(f"Identified cohorts (excluding {excluded_cohorts}): {all_cohorts}")

# Step 2: Load embeddings and build patient-to-cohort mapping
patient_ids = []
embeddings_list = []
cohort_patient_map = {}

print("Loading embeddings...")

for cohort in tqdm(all_cohorts, desc="Cohorts"):
    clinical_file = os.path.join(clinical_tables_path, f"{cohort}.csv")
    embeddings_file = os.path.join(embeddings_path, f"{cohort}.h5")
    
    # Load clinical data
    clinical_df = pd.read_csv(clinical_file)
    cohort_patient_ids = clinical_df['PATIENT'].astype(str).tolist()
    
    if not os.path.exists(embeddings_file):
        print(f"Embeddings file for cohort {cohort} not found at {embeddings_file}. Skipping.")
        continue
    
    with h5py.File(embeddings_file, 'r') as hf:
        # For each patient in the clinical data, retrieve the corresponding embedding
        for pid in cohort_patient_ids:
            if pid in hf:
                embedding = hf[pid][()]
                if embedding.shape[0] != 1280:
                    print(f"Embedding for patient {pid} in cohort {cohort} has incorrect shape {embedding.shape}. Skipping.")
                    continue
                patient_ids.append(pid)
                embeddings_list.append(embedding)
                cohort_patient_map[pid] = cohort
            else:
                print(f"Patient ID {pid} not found in embeddings file {embeddings_file}. Skipping.")

# Convert embeddings to a NumPy array
embeddings = np.vstack(embeddings_list)
print(f"Total patient embeddings loaded: {embeddings.shape[0]}")

# Step 3: Normalize embeddings to unit vectors for cosine similarity
embeddings_normalized = normalize(embeddings, axis=1)

# Create mapping between patient IDs and their corresponding indices
pid_to_index = {pid: idx for idx, pid in enumerate(patient_ids)}
index_to_pid = {idx: pid for pid, idx in pid_to_index.items()}

# Step 4: Build a mapping of cohorts to their patient IDs and report counts
cohort_to_pids = {cohort: [] for cohort in all_cohorts}
for pid, cohort in cohort_patient_map.items():
    cohort_to_pids[cohort].append(pid)

total_patients = 0
for cohort, pids in cohort_to_pids.items():
    print(f"Cohort {cohort}: {len(pids)} patients")
    total_patients += len(pids)
print(f"Total patients across all cohorts: {total_patients}")

# Step 5: For each cohort, perform slide retrieval by computing cosine similarity
for cohort in all_cohorts:
    pids = cohort_to_pids[cohort]
    if len(pids) < num_random_patients:
        print(f"Cohort {cohort} has less than {num_random_patients} patients. Skipping.")
        continue
    
    # Randomly select a specified number of patients from the cohort
    selected_pids = random.sample(pids, num_random_patients)
    result_rows = []
    
    for selected_pid in selected_pids:
        selected_idx = pid_to_index[selected_pid]
        selected_embedding = embeddings_normalized[selected_idx]
        
        # Compute cosine similarity using dot product
        similarities = embeddings_normalized.dot(selected_embedding)
        similarities[selected_idx] = -np.inf  # Exclude self from retrieval
        
        # Retrieve top k most similar patients
        top_k_indices = np.argpartition(-similarities, top_k)[:top_k]
        top_k_sorted_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]
        top_pids = [index_to_pid[idx] for idx in top_k_sorted_indices]
        top_similarities = similarities[top_k_sorted_indices]
        
        row = [selected_pid] + top_pids + top_similarities.tolist()
        result_rows.append(row)
    
    # Save slide retrieval results for the cohort
    columns = ['Selected_Patient'] + [f'Top_{i+1}_Patient' for i in range(top_k)] + [f'Top_{i+1}_Similarity' for i in range(top_k)]
    df_results = pd.DataFrame(result_rows, columns=columns)
    output_file = os.path.join(output_dir, f"{cohort}.csv")
    df_results.to_csv(output_file, index=False)
    print(f"Saved slide search results for cohort {cohort} to {output_file}")

print("Slide retrieval task completed successfully.")

import os
import h5py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import log_loss

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Load feature vectors from an H5 file given a list of patient IDs
def load_features(h5_file, patient_ids):
    missing_ids = []
    with h5py.File(h5_file, "r") as f:
        features = {}
        for pid in patient_ids:
            if pid in f:
                features[pid] = f[pid][:]
            else:
                missing_ids.append(pid)
    return features, missing_ids


# Train logistic regression classifiers on different sample sizes (n_cases) over multiple runs
def train_classifier_lr(
    X, y, num_runs=10, n_cases=[1, 2, 4, 8, 16, 32], path=None, task=None
):
    os.makedirs(path, exist_ok=True)
    classifiers = {n: [] for n in n_cases}
    label_encoder = LabelEncoder()
    patient_ids = list(X.keys())
    patient_to_label = dict(zip(patient_ids, y))
    label_encoder.fit(y)

    # Create stratified random splits for each run and sample size (n)
    for n in n_cases:
        for run in range(num_runs):
            run_path = os.path.join(path, f"n_{n}_run_{run}")
            os.makedirs(run_path, exist_ok=True)

            sample_patient_ids, sample_labels = stratified_sample(
                patient_ids, patient_to_label, n, random_state=RANDOM_SEED + run
            )
            X_train_array = [X[pid] for pid in sample_patient_ids]
            y_train_array = label_encoder.transform(
                [sample_labels[pid] for pid in sample_patient_ids]
            )
            label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

            X_test_patient_ids = [
                pid for pid in patient_ids if pid not in sample_patient_ids
            ]
            X_test_array = [X[pid] for pid in X_test_patient_ids]
            y_test_array = label_encoder.transform(
                [patient_to_label[pid] for pid in X_test_patient_ids]
            )

            clf = LogisticRegression(
                max_iter=10000,
                solver="lbfgs",
                random_state=RANDOM_SEED,
                class_weight="balanced",
            )
            clf.fit(np.array(X_train_array), y_train_array)

            # Save model and label mapping
            with open(os.path.join(run_path, "model.pkl"), "wb") as f:
                pickle.dump((clf, label_mapping), f)
            train_df = pd.DataFrame(
                {
                    "PATIENT": sample_patient_ids,
                    "LABEL": [sample_labels[pid] for pid in sample_patient_ids],
                }
            )
            train_df.to_csv(os.path.join(run_path, "train.csv"), index=False)

            test_df = evaluate_model(
                clf,
                np.array(X_test_array),
                y_test_array,
                X_test_patient_ids,
                label_encoder,
                label_mapping,
                task,
            )
            test_df.to_csv(
                os.path.join(run_path, "internal_test_results.csv"), index=False
            )

            classifiers[n].append(clf)

    return classifiers, label_mapping, label_encoder


# Perform stratified sampling of patient IDs so that each label has up to n samples
def stratified_sample(patient_ids, patient_to_label, n, random_state):
    np.random.seed(random_state)
    sample_patient_ids = []
    sample_labels = {}
    labels = np.unique(list(patient_to_label.values()))

    for label in labels:
        label_ids = [pid for pid in patient_ids if patient_to_label[pid] == label]
        if len(label_ids) > n:
            sampled_ids = np.random.choice(label_ids, n, replace=False).tolist()
        elif label_ids:
            sampled_ids = label_ids
        else:
            tqdm.write(f"Warning: No samples available for label {label}. Skipping.")
            continue

        sample_patient_ids.extend(sampled_ids)
        for pid in sampled_ids:
            sample_labels[pid] = label
    return sample_patient_ids, sample_labels


# Evaluate the classifier on test data and return the results as a DataFrame
def evaluate_model(
    clf, X_test_array, y_test_array, patient_ids, label_encoder, label_mapping, task
):
    y_test_str = label_encoder.inverse_transform(y_test_array)
    probs = clf.predict_proba(X_test_array)
    preds = clf.predict(X_test_array)

    if len(probs) != len(y_test_str):
        raise ValueError(
            f"Inconsistent number of predictions ({len(probs)}) and true labels ({len(y_test_str)})"
        )

    results = []
    for pid, true_str, pred, prob in zip(patient_ids, y_test_str, preds, probs):
        pred_label = label_mapping[pred]
        loss_value = log_loss([true_str], [prob], labels=label_encoder.classes_)
        results.append(
            {
                "PATIENT": pid,
                task: true_str,
                "pred": pred_label,
                f"{task}_{label_mapping[0]}": prob[0],
                f"{task}_{label_mapping[1]}": prob[1],
                "loss": loss_value,
            }
        )
    return pd.DataFrame(results)


# Run logistic regression benchmark on provided cohorts and tasks with external validation
def run_logistic_regression_benchmark(
    model_path, table_dir, model_dir, cohorts, tasks, external_task_mapping, model_name
):
    for cohort in tqdm(cohorts, desc="Cohort", leave=False):
        for task in tasks[cohort]:
            h5_file = os.path.join(model_path, f"{cohort}.h5")
            clini_table_path = os.path.join(table_dir, f"{cohort}.csv")
            clini_table = pd.read_csv(clini_table_path)
            valid_rows = clini_table[task].notna()
            patient_ids = clini_table.loc[valid_rows, "PATIENT"].astype(str).tolist()
            y = clini_table.loc[valid_rows, task].astype(str).tolist()

            X, missing_ids = load_features(h5_file, patient_ids)
            if missing_ids:
                valid_indices = [
                    i for i, pid in enumerate(patient_ids) if pid not in missing_ids
                ]
                patient_ids = [patient_ids[i] for i in valid_indices]
                y = [y[i] for i in valid_indices]

            classifiers, label_mapping, label_encoder = train_classifier_lr(
                X,
                y,
                path=os.path.join(model_dir, model_name, cohort, task),
                n_cases=[1, 2, 4, 8, 16, 32],
                task=task,
            )

            # External validation
            if (
                cohort in external_task_mapping
                and task in external_task_mapping[cohort]
            ):
                for external_cohort in external_task_mapping[cohort][task]:
                    external_clini_table_path = os.path.join(
                        table_dir, f"{external_cohort}.csv"
                    )
                    if not os.path.exists(external_clini_table_path):
                        tqdm.write(
                            f"Skipping {external_cohort}: No clinical table found."
                        )
                        continue
                    external_clini_table = pd.read_csv(external_clini_table_path)
                    if task not in external_clini_table.columns:
                        tqdm.write(
                            f"Skipping {external_cohort}: Task {task} not found in clinical table."
                        )
                        continue
                    valid_rows = external_clini_table[task].notna()
                    external_patient_ids = (
                        external_clini_table.loc[valid_rows, "PATIENT"]
                        .astype(str)
                        .tolist()
                    )
                    external_y = (
                        external_clini_table.loc[valid_rows, task].astype(str).tolist()
                    )
                    external_h5_file = os.path.join(model_path, f"{external_cohort}.h5")
                    external_X, missing_ids = load_features(
                        external_h5_file, external_patient_ids
                    )
                    if missing_ids:
                        valid_indices = [
                            i
                            for i, pid in enumerate(external_patient_ids)
                            if pid not in missing_ids
                        ]
                        external_patient_ids = [
                            external_patient_ids[i] for i in valid_indices
                        ]
                        external_y = [external_y[i] for i in valid_indices]

                    external_X_array = [external_X[pid] for pid in external_patient_ids]
                    external_y_array = label_encoder.transform(external_y)

                    for n in [1, 2, 4, 8, 16, 32]:
                        for run_idx, clf in enumerate(classifiers[n]):
                            run_name = f"n_{n}_run_{run_idx}"
                            external_results_path = os.path.join(
                                model_dir, model_name, external_cohort, task, run_name
                            )
                            os.makedirs(external_results_path, exist_ok=True)
                            test_df = evaluate_model(
                                clf,
                                external_X_array,
                                external_y_array,
                                external_patient_ids,
                                label_encoder,
                                label_mapping,
                                task,
                            )
                            test_df.to_csv(
                                os.path.join(
                                    external_results_path, "external_test_results.csv"
                                ),
                                index=False,
                            )


# Directories and benchmark parameters
table_dir = os.path.join("tables", "clinical_tables")
cohorts = ["tcga_crc", "tcga_brca", "tcga_luad", "tcga_lung", "tcga_stad"]
tasks = {
    "tcga_crc": ["isMSIH", "BRAF"],
    "tcga_brca": ["ESR1", "PGR", "PIK3CA"],
    "tcga_luad": ["STK11", "TP53"],
    "tcga_lung": ["CANCER_TYPE"],
    "tcga_stad": ["EBV", "isMSIH"],
}

external_task_mapping = {
    "tcga_brca": {
        "ESR1": ["cptac_brca"],
        "PGR": ["cptac_brca"],
        "PIK3CA": ["cptac_brca"],
    },
    "tcga_luad": {"STK11": ["cptac_luad"], "TP53": ["cptac_luad"]},
    "tcga_stad": {"EBV": ["kiel"], "isMSIH": ["bern", "kiel"]},
    "tcga_crc": {"isMSIH": ["dachs", "cptac_coad"], "BRAF": ["dachs"]},
    "tcga_lung": {"CANCER_TYPE": ["cptac_lung"]},
}

moco_features_dir = os.path.join("output", "features", "eagle")
folders = ["2mpp", "halfmpp", "kathermpp"]

for folder in folders:
    folder_dir = os.path.join(moco_features_dir, folder)
    for model in os.listdir(folder_dir):
        feature_dir = os.path.join(folder_dir, model)
        model_dir = os.path.join("output", "lp_models", folder)
        run_logistic_regression_benchmark(
            feature_dir,
            table_dir,
            model_dir,
            cohorts,
            tasks,
            external_task_mapping,
            model,
        )

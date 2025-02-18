import os
import h5py
import pandas as pd
from functools import partial
import torch
import torch.nn as nn
from fastai.vision.all import Learner, DataLoader, DataLoaders, OptimWrapper, RocAuc, RocAucBinary, SaveModelCallback, CSVLogger, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Define a simple MLP classifier model
class Classifier(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 768, mlp_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Load feature vectors from an H5 file
def load_features(h5_file, patient_ids):
    missing_ids = []
    with h5py.File(h5_file, 'r') as f:
        features = {}
        for pid in patient_ids:
            if pid in f:
                features[pid] = f[pid][:]
            else:
                missing_ids.append(pid)
    return features, missing_ids

# Train and validate the classifier with cross-validation
def train_classifier(model_class, X_train, y_train, patient_ids, task, cohort, num_folds=5, path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if not (len(X_train) == len(y_train) == len(patient_ids)):
        raise ValueError("Mismatch in lengths of X_train, y_train, and patient_ids.")

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    input_dim = X_train[0].shape[0]
    os.makedirs(path, exist_ok=True)

    indices = np.arange(len(y_train))
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1337)
    classifiers = []

    for fold, (train_val_index, test_index) in enumerate(tqdm(kf.split(indices, y_train_encoded), total=num_folds, desc=f'Training {task}')):
        fold_path = os.path.join(path, f'fold-{fold}')
        os.makedirs(fold_path, exist_ok=True)

        # Split data into training/validation and testing sets
        train_val_X = [X_train[i] for i in train_val_index]
        test_X = [X_train[i] for i in test_index]
        y_train_val_split = [y_train[i] for i in train_val_index]
        y_test_split = [y_train[i] for i in test_index]
        train_val_patient_ids = [patient_ids[i] for i in train_val_index]
        test_patient_ids = [patient_ids[i] for i in test_index]

        # Inner train-validation split
        train_index, val_index = train_test_split(
            np.arange(len(train_val_X)),
            test_size=0.2,
            stratify=y_train_val_split,
            random_state=1337
        )

        train_X = [train_val_X[i] for i in train_index]
        val_X = [train_val_X[i] for i in val_index]
        y_train_real = [y_train_val_split[i] for i in train_index]
        y_val_real = [y_train_val_split[i] for i in val_index]
        train_patient_ids_real = [train_val_patient_ids[i] for i in train_index]
        val_patient_ids_real = [train_val_patient_ids[i] for i in val_index]

        # Save splits for reproducibility
        pd.DataFrame({'PATIENT': train_patient_ids_real, 'LABEL': y_train_real}).to_csv(os.path.join(fold_path, 'train.csv'), index=False)
        pd.DataFrame({'PATIENT': val_patient_ids_real, 'LABEL': y_val_real}).to_csv(os.path.join(fold_path, 'valid.csv'), index=False)
        pd.DataFrame({'PATIENT': test_patient_ids, 'LABEL': y_test_split}).to_csv(os.path.join(fold_path, 'test.csv'), index=False)
        
        model = model_class(num_classes=len(label_encoder.classes_), input_dim=input_dim).to(device).float()

        # Prepare datasets for training, validation, and testing
        train_ds = [(torch.tensor(train_X[i], dtype=torch.float32),
                     torch.tensor(label_encoder.transform([y_train_real[i]])[0], dtype=torch.long))
                    for i in range(len(train_X))]
        valid_ds = [(torch.tensor(val_X[i], dtype=torch.float32),
                     torch.tensor(label_encoder.transform([y_val_real[i]])[0], dtype=torch.long))
                    for i in range(len(val_X))]
        test_ds = [(torch.tensor(test_X[i], dtype=torch.float32),
                    torch.tensor(label_encoder.transform([y_test_split[i]])[0], dtype=torch.long))
                   for i in range(len(test_X))]

        # Set up weighted loss
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_real), y=y_train_real)
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
        loss_func = nn.CrossEntropyLoss(weight=weight)

        # Data loaders for FastAI
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
        valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        metrics = [RocAucBinary()] if len(label_encoder.classes_) == 2 else [RocAuc()]

        dls = DataLoaders(train_dl, valid_dl, device=device)
        learn = Learner(
            dls,
            model,
            loss_func=loss_func,
            opt_func=partial(OptimWrapper, opt=torch.optim.AdamW),
            metrics=metrics,
            cbs=[
                SaveModelCallback(monitor='valid_loss', fname=os.path.join(fold_path, 'best_valid')),
                EarlyStoppingCallback(monitor='valid_loss', patience=8),
                CSVLogger(fname=os.path.join(fold_path, 'history.csv'))
            ]
        )

        # Train the model with one-cycle policy
        learn.fit_one_cycle(n_epoch=32, reset_opt=True, lr_max=1e-4, wd=1e-2)
        with open(os.path.join(fold_path, 'export.pkl'), 'wb') as f:
            pickle.dump(learn.model, f)

        classifiers.append(learn.model)

        # Internal prediction and save
        internal_test_df = predict_and_save_internal(learn.model, test_dl, test_patient_ids, y_test_split, task, label_encoder, fold_path)
        internal_test_df.to_csv(os.path.join(fold_path, 'patient-preds.csv'), index=False)

    return classifiers

# Generate internal predictions for the test set and save results
def predict_and_save_internal(model, test_dl, patient_ids, y_test, task, label_encoder, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    df = pd.DataFrame(columns=['PATIENT', task, 'pred'] + [f'{task}_{cls}' for cls in label_encoder.classes_] + ['loss'])
    with torch.no_grad():
        for i, (x, target) in enumerate(tqdm(test_dl, total=len(test_dl), desc=f'Internal Testing {task}')):
            x = x.to(device).float()
            target = target.to(device)
            outputs = model(x)
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probabilities, dim=1).cpu().numpy()
            real_target = label_encoder.inverse_transform(target.cpu().numpy())[0]
            real_pred = label_encoder.inverse_transform([preds[0]])[0]
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(outputs, target).item()
            data_row = {
                'PATIENT': patient_ids[i],
                task: real_target,
                'pred': real_pred,
                **{f'{task}_{cls}': probabilities[0][j].item() for j, cls in enumerate(label_encoder.classes_)},
                'loss': loss
            }
            df = pd.concat([df, pd.DataFrame([data_row])], ignore_index=True)
    return df

# Predict on external validation data and save predictions
def predict_and_save(classifiers, X_test, y_test, patient_ids, task, external_cohort, model_name, path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    columns = ['PATIENT', task, 'pred'] + [f'{task}_{cls}' for cls in np.unique(y_test)] + ['loss']
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    os.makedirs(path, exist_ok=True)
    
    for fold, classifier in enumerate(classifiers):
        classifier.eval()
        with torch.no_grad():
            test_ds = [(torch.tensor(X_test[i], dtype=torch.float32),
                        torch.tensor(y_test_encoded[i], dtype=torch.long))
                       for i in range(len(X_test))]
            test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
            df = pd.DataFrame(columns=columns)
            for i, (x, target) in enumerate(tqdm(test_dl, total=len(test_dl), desc=f'Predicting {task}')):
                x = x.to(device).float()
                target = target.to(device)
                outputs = classifier(x)
                probabilities = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probabilities, dim=1).cpu().numpy()
                real_target = label_encoder.inverse_transform(target.cpu().numpy())[0]
                real_pred = label_encoder.inverse_transform([preds[0]])[0]
                classes = label_encoder.classes_
                loss_func = nn.CrossEntropyLoss()
                loss = loss_func(outputs, target).item()
                data_row = {
                    'PATIENT': patient_ids[i],
                    task: real_target,
                    'pred': real_pred,
                    **{f'{task}_{cls}': probabilities[0][j].item() for j, cls in enumerate(classes)},
                    'loss': loss
                }
                df = pd.concat([df, pd.DataFrame([data_row])], ignore_index=True)

        save_name = external_cohort.lower()
        if "cptac" in external_cohort.lower():
            save_name = "cptac"
        save_path = os.path.join(path, 'deploy', save_name, f'fold-{fold}', 'patient-preds.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

# Main execution block
cohorts = ['tcga_crc', 'tcga_brca', 'tcga_luad', 'tcga_lung', 'tcga_stad']
tasks = {
    'tcga_crc': ['LEFT_RIGHT', 'isMSIH', 'BRAF', 'KRAS', 'CIMP', 'PIK3CA', 'N_STATUS', 'M_STATUS'],
    'tcga_brca': ['ERBB2', 'ESR1', 'PGR', 'PIK3CA', 'N_STATUS'],
    'tcga_luad': ['EGFR', 'KRAS', 'STK11', 'TP53'],
    'tcga_lung': ['CANCER_TYPE'],
    'tcga_stad': ['EBV', 'isMSIH', 'N_STATUS', 'M_STATUS', 'LAUREN']
}

external_task_mapping = {
    'tcga_brca': {
        'ERBB2': ['cptac_brca'],
        'ESR1': ['cptac_brca'],
        'PGR': ['cptac_brca'],
        'PIK3CA': ['cptac_brca'],
        'N_STATUS': ['marra']
    },
    'tcga_luad': {
        'EGFR': ['cptac_luad'],
        'KRAS': ['cptac_luad'],
        'STK11': ['cptac_luad'],
        'TP53': ['cptac_luad']
    },
    'tcga_stad': {
        'LAUREN': ['bern', 'kiel'],
        'EBV': ['kiel'],
        'isMSIH': ['bern', 'kiel'],
        'N_STATUS': ['bern', 'kiel'],
        'M_STATUS': ['kiel']
    },
    'tcga_crc': {
        'LEFT_RIGHT': ['dachs', 'cptac_coad'],
        'isMSIH': ['dachs', 'cptac_coad'],
        'BRAF': ['dachs', 'cptac_coad'],
        'KRAS': ['dachs', 'cptac_coad'],
        'CIMP': ['dachs'],
        'PIK3CA': ['cptac_coad'],
        'N_STATUS': ['dachs', 'cptac_coad'],
        'M_STATUS': ['dachs']
    },
    'tcga_lung': {
        'CANCER_TYPE': ['cptac_lung']
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_model_mapping = {
    'halfmpp': ['madeleine', 'prism', 'chief', 'gigapath', 'titan', 'eagle', 'cobra'],
    'kathermpp': ['madeleine', 'prism', 'chief', 'gigapath', 'eagle', 'cobra'],
    '2mpp': ['madeleine', 'prism', 'chief', 'gigapath', 'titan', 'eagle', 'cobra'],
}

folders = list(folder_model_mapping.keys())

for folder in tqdm(folders, desc="Processing Folders"):
    feature_dir = os.path.join('output','features', folder)
    table_dir = os.path.join('tables', 'clinical_tables')
    external_table_dir = os.path.join('tables', 'clinical_tables')
    model_dir = os.path.join('output','models')
    models = folder_model_mapping[folder]

    for model in models:
        combined_model_name = f"{folder}-{model}".lower()
        model_path = os.path.join(feature_dir, model)

        for cohort in cohorts:
            for task in tasks[cohort]:
                cancer_type = cohort.replace("tcga_", "").lower()
                path = os.path.join(model_dir, combined_model_name, cancer_type, task.lower())

                # Check if external prediction files already exist
                skip_task = True
                for external_cohort in external_task_mapping.get(cohort, {}).get(task, []):
                    ext_cohort = "cptac" if "cptac" in external_cohort.lower() else external_cohort
                    for fold in range(5): 
                        fold_pred_path = os.path.join(path, 'deploy', ext_cohort, f'fold-{fold}')
                        pred_file = os.path.join(fold_pred_path, 'patient-preds.csv')
                        if not os.path.exists(pred_file):
                            skip_task = False
                            break
                    if not skip_task:
                        break

                if skip_task:
                    print(f"Skipping {task} in {cohort}: All external prediction files already present.")
                    continue

                h5_file = os.path.join(model_path, f'{cohort}.h5')
                clini_table_path = os.path.join(table_dir, f'{cohort}.csv')
                clini_table = pd.read_csv(clini_table_path)
                valid_rows = clini_table[task].notna()
                patient_ids = clini_table.loc[valid_rows, 'PATIENT'].astype(str).tolist()
                y = clini_table.loc[valid_rows, task].astype(str).tolist()
                X_train_dict, missing_ids = load_features(h5_file, patient_ids)
                if missing_ids:
                    filtered_indices = [i for i, pid in enumerate(patient_ids) if pid not in missing_ids]
                    patient_ids = [patient_ids[i] for i in filtered_indices]
                    y = [y[i] for i in filtered_indices]
                X_train = [X_train_dict[pid] for pid in patient_ids]

                print(f"Number of patient IDs: {len(patient_ids)}")
                print(f"Number of labels in y: {len(y)}")
                print(f"Number of features in X_train: {len(X_train)}")
                        
                print(f"Training {combined_model_name} with {cohort} on task {task}")
                classifiers = train_classifier(
                    Classifier, X_train, y, patient_ids, task, cohort,
                    path=path
                )

                if cohort in external_task_mapping and task in external_task_mapping[cohort]:
                    for external_cohort in external_task_mapping[cohort][task]:
                        external_clini_table_path = os.path.join(external_table_dir, f'{external_cohort}.csv')
                        if not os.path.exists(external_clini_table_path):
                            print(f"Skipping {external_cohort}: No clinical table found.")
                            continue

                        external_clini_table = pd.read_csv(external_clini_table_path)
                        if task not in external_clini_table.columns:
                            print(f"Skipping {external_cohort}: Task {task} not found in clinical table.")
                            continue

                        valid_rows = external_clini_table[task].notna()
                        external_patient_ids = external_clini_table.loc[valid_rows, 'PATIENT'].astype(str).tolist()
                        external_y = external_clini_table.loc[valid_rows, task].astype(str).tolist()
                        external_h5_file = os.path.join(model_path, f'{external_cohort}.h5')
                        external_X_dict, missing_ids = load_features(external_h5_file, external_patient_ids)
                        if missing_ids:
                            filtered_indices = [i for i, pid in enumerate(external_patient_ids) if pid not in missing_ids]
                            external_patient_ids = [external_patient_ids[i] for i in filtered_indices]
                            external_y = [external_y[i] for i in filtered_indices]
                        external_X = [external_X_dict[pid] for pid in external_patient_ids]

                        predict_and_save(
                            classifiers, external_X, external_y, external_patient_ids, task, external_cohort, combined_model_name,
                            path=path
                        )
                else:
                    print(f"Skipping task {task} in {cohort}: No valid external cohorts for this task.")

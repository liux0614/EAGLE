import os
import h5py
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from glob import glob
from collections import defaultdict
from matplotlib.lines import Line2D

def get_mpp_type(model_name):
    """
    Return the corresponding mpp-type subdirectory for a given model.
    """
    model_name_lower = model_name.lower()
    if model_name_lower in ["titan", "prism"]:
        return "halfmpp"
    elif model_name_lower in ["chief", "madeleine", "gigapath"]:
        return "kathermpp"
    elif model_name_lower in ["eagle", "cobra"]:
        return "2mpp"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def format_model_name_for_title(model_name):
    """
    Format the model name for display titles.
    """
    name_lower = model_name.lower()
    if name_lower == "gigapath":
        return "GigaPath"
    elif name_lower == "prism":
        return "Prism"
    else:
        return model_name.upper()

def load_features_and_cohorts(model_name):
    """
    Load feature vectors from .h5 files and retrieve the corresponding clinical cohort labels.
    Filters out LUAD/LUSC cohorts.
    """
    mpp_type = get_mpp_type(model_name)
    # Use relative paths instead of fixed file paths
    features_dir = os.path.join('output','features', mpp_type, model_name.lower())
    clinical_dir = os.path.join('tables', 'clinical_tables')
    
    h5_files = glob(os.path.join(features_dir, '*.h5'))
    
    all_features = []
    all_patient_ids = []
    all_cohort_labels = []
    
    for h5_file in h5_files:
        filename = os.path.basename(h5_file)
        # Skip lung cohorts
        if filename in ['tcga_luad.h5', 'cptac_luad.h5', 'tcga_lusc.h5', 'cptac_lusc.h5']:
            continue
        
        cohort_name = filename.replace('.h5', '')
        clinical_file = os.path.join(clinical_dir, cohort_name + '.csv')
        if not os.path.exists(clinical_file):
            continue
        
        clinical_df = pd.read_csv(clinical_file)
        patient_ids = clinical_df['PATIENT'].astype(str).unique()
        
        with h5py.File(h5_file, 'r') as h5f:
            for pid in patient_ids:
                if pid in h5f.keys():
                    feature_vector = h5f[pid][:]
                    all_features.append(feature_vector)
                    all_patient_ids.append(pid)
                    
                    # Assign organ label based on cohort name
                    if "lung" in cohort_name:
                        cohort_label = 'lung'
                    elif any(x in cohort_name for x in ["bern", "kiel", "stad"]):
                        cohort_label = 'stad'  # Stomach
                    elif any(x in cohort_name for x in ["marra", "brca"]):
                        cohort_label = 'brca'  # Breast
                    elif any(x in cohort_name for x in ["dachs", "coad", "crc"]):
                        cohort_label = 'crc'   # Bowel
                    else:
                        continue
                    
                    all_cohort_labels.append(cohort_label)
                    
    return np.array(all_features), np.array(all_patient_ids), np.array(all_cohort_labels)

def run_umap(features):
    """
    Reduce high-dimensional features to 2D using UMAP.
    """
    reducer = umap.UMAP()
    return reducer.fit_transform(features)

def format_output_filename(models):
    """
    Format output filename based on the number of models.
    """
    if len(models) == 1:
        return f"UMAP_{models[0].lower()}.png"
    else:
        return "UMAP_all_models.png"

def plot_umaps_for_models(models, output_dir="figures"):
    """
    Plot UMAP embeddings for the specified models.
    
    If multiple models are provided, display them in a multi-panel figure with a unified legend.
    Each panel shows:
      - UMAP scatter colored by organ type.
      - Short arrows indicating UMAP1 and UMAP2 axes.
    The output is saved to the specified output directory.
    """
    # Define mapping from cohort labels to display names and colors.
    cohort_to_organ = {
        'stad': 'Stomach',
        'crc' : 'Bowel',
        'lung': 'Lung',
        'brca': 'Breast'
    }
    cohort_colors = {
        'stad': '#FF0000',  # red
        'crc' : '#FF69B4',  # pink
        'lung': '#800080',  # purple
        'brca': '#AAAAAA'   # gray-ish
    }
    
    model_embeddings = []
    model_labels = []
    
    # Load embeddings for each model
    for model in models:
        all_features, _, all_cohort_labels = load_features_and_cohorts(model)
        if len(all_features) == 0:
            print(f"Warning: No data found for model {model}. Skipping.")
            model_embeddings.append(None)
            model_labels.append(None)
            continue
        embedding = run_umap(all_features)
        model_embeddings.append(embedding)
        model_labels.append(all_cohort_labels)
    
    n_models = len(models)
    # Set up figure panels
    if n_models == 1:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes = [axes]
    else:
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
        axes = axes.flatten()
    
    # Hide extra subplots (if any)
    for ax_idx in range(n_models, len(axes) - 1):
        axes[ax_idx].set_visible(False)
    legend_ax = axes[-1]
    legend_ax.set_visible(True)
    legend_ax.set_axis_off()
    
    # Plot each model's embedding
    for idx, (model, embedding, labels) in enumerate(zip(models, model_embeddings, model_labels)):
        ax = axes[idx]
        if embedding is None or labels is None:
            ax.set_visible(False)
            continue
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Plot each cohort's points
        for cohort_key in np.unique(labels):
            mask = (labels == cohort_key)
            if cohort_key in cohort_to_organ:
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=cohort_colors[cohort_key],
                    s=1,
                    alpha=0.8,
                    label=cohort_to_organ[cohort_key]
                )
        
        if n_models > 1:
            ax.set_title(format_model_name_for_title(model), fontsize=12)
        
        # Add short arrows indicating UMAP1 and UMAP2 axes in axis-fraction coordinates.
        ax.annotate(
            '',
            xy=(0.3, 0.0),
            xytext=(0.0, 0.0),
            arrowprops=dict(facecolor='black', width=1, headwidth=6),
            xycoords='axes fraction'
        )
        ax.text(
            0.15, -0.015,
            "UMAP 1",
            ha='center', va='top',
            transform=ax.transAxes,
            fontsize=12
        )
        ax.annotate(
            '',
            xy=(0.0, 0.3),
            xytext=(0.0, 0.0),
            arrowprops=dict(facecolor='black', width=1, headwidth=6),
            xycoords='axes fraction'
        )
        ax.text(
            -0.015, 0.15,
            "UMAP 2",
            ha='right', va='center',
            rotation=90,
            transform=ax.transAxes,
            fontsize=12
        )
    
    # Create a unified legend for the organ categories.
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cohort_colors['stad'], markersize=14, label='Stomach'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cohort_colors['crc'], markersize=14, label='Bowel'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cohort_colors['lung'], markersize=14, label='Lung'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cohort_colors['brca'], markersize=14, label='Breast')
    ]
    legend_ax.legend(
        handles=legend_elements,
        loc='center',
        title='Organ',
        title_fontsize=12,
        labelspacing=1.5,
        prop={'size': 12},
        frameon=False
    )
    
    plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)
    output_filename = format_output_filename(models)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    all_models = ["eagle", "titan", "cobra", "prism", "chief", "madeleine", "gigapath"]
    plot_umaps_for_models(all_models, output_dir=os.path.join("output","umaps"))

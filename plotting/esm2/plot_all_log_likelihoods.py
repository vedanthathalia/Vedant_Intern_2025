import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import os

labels = ["human", "avian", "other"]
models = ["8M", "35M", "650M"]
base_path_template = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/{model}/final_{label}_model/{label}_log_likelihoods.csv"
fig_base_dir = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures"

def kde_subsets(df, label):
    other_labels = [l for l in labels if l != label]

    sole = df[
        (df[f"label_{label}"] == 1) &
        (df[[f"label_{l}" for l in other_labels]].sum(axis=1) == 0)
    ]
    inclusive = df[
        (df[f"label_{label}"] == 1) &
        (df[[f"label_{l}" for l in other_labels]].sum(axis=1) >= 1)
    ]
    non = df[
        (df[f"label_{label}"] == 0) &
        (df[[f"label_{l}" for l in other_labels]].sum(axis=1) >= 1)
    ]

    return {
        f"{label.capitalize()} Only (n={len(sole)})": sole["log_likelihood"].dropna(),
        f"{label.capitalize()} + Other (n={len(inclusive)})": inclusive["log_likelihood"].dropna(),
        f"Non-{label.capitalize()} (n={len(non)})": non["log_likelihood"].dropna()
    }

for label in labels:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    all_vals = []

    for model in models:
        file_path = base_path_template.format(model=model, label=label)
        if not os.path.exists(file_path):
            print(f"Skipping {model}-{label}, file not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        subsets = kde_subsets(df, label)
        for values in subsets.values():
            all_vals.append(values)

    all_vals_flat = pd.concat(all_vals)
    x_min = max(all_vals_flat.min(), -90)
    x_max = all_vals_flat.max() + 5
    x_vals = np.linspace(x_min, x_max, 300)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    for i, model in enumerate(models):
        ax = axes[i]
        file_path = base_path_template.format(model=model, label=label)
        if not os.path.exists(file_path):
            ax.set_visible(False)
            continue

        df = pd.read_csv(file_path)
        subsets = kde_subsets(df, label)
        for (subset_name, values), color in zip(subsets.items(), colors):
            kde = gaussian_kde(values)
            ax.plot(x_vals, kde(x_vals), label=subset_name, color=color, linewidth=2)

        # ax.set_title(f"{model} {label.capitalize()} Model")
        ax.set_xlabel("Log-Likelihood")
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(loc="upper left")

    # fig.suptitle(f"{label.capitalize()} - Log-Likelihood KDE Plots Across All Models", fontsize=14, y=0.95)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()

    label_folder = os.path.join(fig_base_dir, label.capitalize(), "AllM")
    os.makedirs(label_folder, exist_ok=True)
    save_path = os.path.join(label_folder, f"log_likelihood_{label}_AllModels.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved: {save_path}")

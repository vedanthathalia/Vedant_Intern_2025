import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
import wandb

def plot_umap_subplot(ax, csv_path, label_col, title):
    df = pd.read_csv(csv_path)
    embeddings = df.iloc[:, 16:16+768].values

    reducer = umap.UMAP(random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    labels = df[label_col].values
    colors = ["red" if l == 1 else "blue" for l in labels]

    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=30, alpha=0.8, edgecolors="k")
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

def main(label):
    wandb.init(project="Figures", name=f"umap_combined_{label}")

    model_sizes = ["8M", "35M", "650M"]
    base_dir = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model"
    base_csv_path = os.path.join(base_dir, "embeddings_results")
    base_fig_path = os.path.join(base_dir, "Figures", label.capitalize(), "AllM")
    os.makedirs(base_fig_path, exist_ok=True)

    label_col = "label_human"
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, size in enumerate(model_sizes):
        csv_path = os.path.join(base_csv_path, size, label, f"global_{label}_test_embeddings.csv")
        title = f"{size} UMAP of {label.capitalize()} Embeddings (Label: Human)"
        print(f"Plotting {title}")
        plot_umap_subplot(axs[i], csv_path, label_col, title)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Human infective',
                   markerfacecolor='red', markersize=10, markeredgecolor='k'),
        plt.Line2D([0], [0], marker='o', color='w', label='Non-human',
                   markerfacecolor='blue', markersize=10, markeredgecolor='k')
    ]
    axs[2].legend(handles=handles, loc="best")

    plt.tight_layout()
    save_path = os.path.join(base_fig_path, f"{label}_umap_allM.png")
    plt.savefig(save_path)
    wandb.log({f"{label}_umap_allM": wandb.Image(save_path)})
    plt.show()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, choices=["human", "avian", "other"], required=True, help="Which host label to visualize")
    args = parser.parse_args()
    main(args.label)

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
import wandb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def plot_umap_3d_subplot(ax, csv_path, label_col, title):
    df = pd.read_csv(csv_path)
    embeddings = df.iloc[:, 16:16+768].values

    reducer = umap.UMAP(n_components=3, random_state=42)
    embeddings_3d = reducer.fit_transform(embeddings)

    labels = df[label_col].values
    colors = ["red" if l == 1 else "blue" for l in labels]

    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
               c=colors, s=30, alpha=0.8, edgecolors="k")
    ax.text2D(0.5, 0.92, title, transform=ax.transAxes, ha='center', fontsize=12)
    # ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")

def main(label):
    wandb.init(project="Figures", name=f"umap_3d_combined_{label}")

    model_sizes = ["8M", "35M", "650M"]
    base_dir = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model"
    base_csv_path = os.path.join(base_dir, "embeddings_results")
    base_fig_path = os.path.join(base_dir, "Figures", label.capitalize(), "AllM")
    os.makedirs(base_fig_path, exist_ok=True)

    label_col = "label_human"
    fig = plt.figure(figsize=(18, 6))

    for i, size in enumerate(model_sizes):
        csv_path = os.path.join(base_csv_path, size, label, f"global_{label}_test_embeddings.csv")
        title = f"{size} 3D UMAP of {label.capitalize()} Embeddings"
        print(f"Plotting {title}")
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        plot_umap_3d_subplot(ax, csv_path, label_col, title)

    handles = [
        Line2D([0], [0], marker='o', color='w', label='Human infective',
               markerfacecolor='red', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Non-human',
               markerfacecolor='blue', markersize=10, markeredgecolor='k')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2)

    plt.tight_layout()
    save_path = os.path.join(base_fig_path, f"{label}_umap_3d_allM.png")
    plt.savefig(save_path)
    wandb.log({f"{label}_umap_3d_allM": wandb.Image(save_path)})
    plt.show()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, choices=["human", "avian", "other"], required=True, help="Which host label to visualize")
    args = parser.parse_args()
    main(args.label)

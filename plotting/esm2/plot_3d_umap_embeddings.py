import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
import wandb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def plot_umap_3d_from_csv(csv_path, label_col, title, wandb_name, save_path):
    df = pd.read_csv(csv_path)
    embeddings = df.iloc[:, 16:16+768].values

    reducer = umap.UMAP(n_components=3, random_state=42)
    embeddings_3d = reducer.fit_transform(embeddings)

    labels = df[label_col].values
    colors = ["red" if l == 1 else "blue" for l in labels]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=colors, s=30, alpha=0.8, edgecolors="k")

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")

    custom_legend = [
        Line2D([0], [0], marker='o', color='w', label='Human infective', markerfacecolor='red', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Non-human', markerfacecolor='blue', markersize=10, markeredgecolor='k')
    ]
    ax.legend(handles=custom_legend, loc='best')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    wandb.log({wandb_name: wandb.Image(save_path)})
    plt.show()

if __name__ == "__main__":
    wandb.init(project="Figures", name="umap_3d_from_saved_embeddings")

    base_csv_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/embeddings_results/8M"
    base_fig_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures"

    configs = [
        {
            "csv": f"{base_csv_path}/human/human_test_embeddings.csv",
            "label_col": "label_human",
            "title": "8M 3D UMAP of Human Embeddings (Label: Human)",
            "wandb_name": "Human UMAP 3D",
            "save_path": f"{base_fig_path}/Human/8M/human_umap_3d.png",
        },
        {
            "csv": f"{base_csv_path}/avian/avian_test_embeddings.csv",
            "label_col": "label_human",
            "title": "8M 3D UMAP of Avian Embeddings (Label: Human)",
            "wandb_name": "Avian UMAP 3D",
            "save_path": f"{base_fig_path}/Avian/8M/avian_umap_3d.png",
        },
        {
            "csv": f"{base_csv_path}/other/other_test_embeddings.csv",
            "label_col": "label_human",
            "title": "8M 3D UMAP of Other Embeddings (Label: Human)",
            "wandb_name": "Other UMAP 3D",
            "save_path": f"{base_fig_path}/Other/8M/other_umap_3d.png",
        },
    ]

    for cfg in configs:
        print(f"Plotting 3D UMAP for: {cfg['csv']}")
        plot_umap_3d_from_csv(
            csv_path=cfg["csv"],
            label_col=cfg["label_col"],
            title=cfg["title"],
            wandb_name=cfg["wandb_name"],
            save_path=cfg["save_path"]
        )

    wandb.finish()
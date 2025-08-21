import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
import wandb

def plot_umap_from_csv(csv_path, label_col, title, wandb_name, save_path):
    df = pd.read_csv(csv_path)
    embeddings = df.iloc[:, 16:16+768].values

    reducer = umap.UMAP(random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    labels = df[label_col].values
    colors = ["red" if l == 1 else "blue" for l in labels]

    plt.figure(figsize=(6, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=30, alpha=0.8, edgecolors="k")
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='Human infective',
                   markerfacecolor='red', markersize=10, markeredgecolor='k'),
        plt.Line2D([0], [0], marker='o', color='w', label='Non-human',
                   markerfacecolor='blue', markersize=10, markeredgecolor='k')
    ], loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    wandb.log({wandb_name: wandb.Image(save_path)})
    plt.show()

if __name__ == "__main__":
    wandb.init(project="Figures", name="umap_from_saved_embeddings")

    base_csv_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/embeddings_results/35M"
    base_fig_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures"

    configs = [
        {
            "csv": f"{base_csv_path}/human/human_test_embeddings.csv",
            "label_col": "label_human",
            "title": "35M UMAP of Human Embeddings (Label: Human)",
            "wandb_name": "Human UMAP",
            "save_path": f"{base_fig_path}/Human/35M/human_umap.png",
        },
        {
            "csv": f"{base_csv_path}/avian/avian_test_embeddings.csv",
            "label_col": "label_human",
            "title": "35M UMAP of Avian Embeddings (Label: Human)",
            "wandb_name": "Avian UMAP",
            "save_path": f"{base_fig_path}/Avian/35M/avian_umap.png",
        },
        {
            "csv": f"{base_csv_path}/other/other_test_embeddings.csv",
            "label_col": "label_human",
            "title": "35M UMAP of Other Embeddings (Label: Human)",
            "wandb_name": "Other UMAP",
            "save_path": f"{base_fig_path}/Other/35M/other_umap.png",
        },
    ]

    for cfg in configs:
        print(f"Plotting UMAP for: {cfg['csv']}")
        plot_umap_from_csv(
            csv_path=cfg["csv"],
            label_col=cfg["label_col"],
            title=cfg["title"],
            wandb_name=cfg["wandb_name"],
            save_path=cfg["save_path"]
        )

    wandb.finish()

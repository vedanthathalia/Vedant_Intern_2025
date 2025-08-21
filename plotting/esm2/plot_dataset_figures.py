import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import wandb
from datetime import datetime
import os

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(project="Figures", name=f"dataset_analysis_{timestamp}")

colors = {
    "Train": "#54A24B",
    "Eval": "#F58518",
    "Test": "#4C78A8"
}

base_dir = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model"
fig_dir = os.path.join(base_dir, "Figures")
data_dir = os.path.join(base_dir, "NCBI_Dataset")

datasets = {
    "human": {
        "train": os.path.join(data_dir, "clustered_train_human.csv"),
        "eval": os.path.join(data_dir, "clustered_eval_human.csv"),
        "test": os.path.join(data_dir, "clustered_test_human.csv")
    },
    "avian": {
        "train": os.path.join(data_dir, "clustered_train_avian.csv"),
        "eval": os.path.join(data_dir, "clustered_eval_avian.csv"),
        "test": os.path.join(data_dir, "clustered_test_avian.csv")
    },
    "other": {
        "train": os.path.join(data_dir, "clustered_train_other.csv"),
        "eval": os.path.join(data_dir, "clustered_eval_other.csv"),
        "test": os.path.join(data_dir, "clustered_test_other.csv")
    }
}

for label_type, paths in datasets.items():
    print(f"Generating figures for: {label_type}")
    save_path = os.path.join(fig_dir, label_type.capitalize())
    os.makedirs(save_path, exist_ok=True)

    train_df = pd.read_csv(paths["train"])
    eval_df = pd.read_csv(paths["eval"])
    test_df = pd.read_csv(paths["test"])

    # Sequence Length Distribution (Log Y)
    train_lengths = train_df["sequence"].apply(len)
    eval_lengths = eval_df["sequence"].apply(len)
    test_lengths = test_df["sequence"].apply(len)

    plt.figure(figsize=(10, 6))

    plt.hist(test_lengths, bins=50, range=(0, 800), label="Test", color=colors["Test"], alpha=0.7, histtype="stepfilled", linewidth=1.2)
    plt.hist(eval_lengths, bins=50, range=(0, 800), label="Eval", color=colors["Eval"], alpha=0.7, histtype="stepfilled", linewidth=1.2)
    plt.hist(train_lengths, bins=50, range=(0, 800), label="Train", color=colors["Train"], alpha=0.7, histtype="stepfilled", linewidth=1.5)

    plt.xlabel("Sequence Length")
    plt.ylabel("Count (log scale)")
    plt.yscale("log")
    plt.xlim(0, 800)
    plt.xticks([100, 200, 300, 400, 500, 600, 700, 800])
    plt.title(f"{label_type.capitalize()} Sequence Length Distribution (Log Y)")
    plt.legend()
    plt.tight_layout()

    name = os.path.join(save_path, "sequence_length_distribution_log.png")
    plt.savefig(name, dpi=300)
    wandb.log({f"{label_type.capitalize()} Sequence Length Distribution (Log Y)": wandb.Image(name)})
    plt.close()

    # Cluster Size Distribution
    train_cluster_sizes = train_df["cluster_id"].value_counts()
    eval_cluster_sizes = eval_df["cluster_id"].value_counts()
    test_cluster_sizes = test_df["cluster_id"].value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # Test
    axes[0].hist(test_cluster_sizes, bins=50, range=(0, 100), color=colors["Test"], alpha=0.7, edgecolor='black')
    axes[0].set_title("Test Cluster Size")
    axes[0].set_xlabel("Cluster Size")
    axes[0].set_ylabel("Number of Clusters")
    axes[0].set_yscale("log")

    # Eval
    axes[1].hist(eval_cluster_sizes, bins=50, range=(0, 100), color=colors["Eval"], alpha=0.7, edgecolor='black')
    axes[1].set_title("Eval Cluster Size")
    axes[1].set_xlabel("Cluster Size")
    axes[1].set_yscale("log")

    # Train (Log Y only)
    axes[2].hist(train_cluster_sizes, bins=50, color=colors["Train"], alpha=0.7, edgecolor='black')
    axes[2].set_title("Train Cluster Size (Log Y)")
    axes[2].set_xlabel("Cluster Size")
    axes[2].set_yscale("log")

    plt.suptitle(f"{label_type.capitalize()} Cluster Size Distribution")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    name = os.path.join(save_path, "cluster_size_distribution_log.png")
    plt.savefig(name, dpi=300)
    wandb.log({f"{label_type.capitalize()} Cluster Size Distribution (Log)": wandb.Image(name)})
    plt.close()

    # Train/Eval/Test Split Pie Chart
    sizes = [len(train_df), len(eval_df), len(test_df)]
    labels_split = ["Train", "Eval", "Test"]
    plt.pie(sizes, labels=labels_split, autopct="%1.1f%%", startangle=140, colors=[colors[l] for l in labels_split])
    plt.title(f"{label_type.capitalize()} Train/Eval/Test Split")
    plt.tight_layout()
    name = os.path.join(save_path, "train_eval_test_split.png")
    plt.savefig(name, dpi=300)
    wandb.log({f"{label_type.capitalize()} Train Eval Test Split": wandb.Image(name)})
    plt.close()

    # Amino Acid Frequency
    def get_aa_counts(sequences):
        counter = Counter()
        for seq in sequences:
            counter.update(seq)
        return counter

    train_aa = get_aa_counts(train_df["sequence"])
    eval_aa = get_aa_counts(eval_df["sequence"])
    test_aa = get_aa_counts(test_df["sequence"])
    all_aas = sorted(set(train_aa.keys()).union(eval_aa.keys()).union(test_aa.keys()))
    aa_df = pd.DataFrame({
        "Amino Acid": all_aas,
        "Train": [train_aa.get(aa, 0) for aa in all_aas],
        "Eval": [eval_aa.get(aa, 0) for aa in all_aas],
        "Test": [test_aa.get(aa, 0) for aa in all_aas]
    })

    aa_df.plot(x="Amino Acid", y=["Train", "Eval", "Test"], kind="bar", figsize=(10, 6), color=[colors["Train"], colors["Eval"], colors["Test"]])
    plt.ylabel("Count")
    plt.title(f"{label_type.capitalize()} Amino Acid Frequency")
    plt.tight_layout()
    name = os.path.join(save_path, "amino_acid_frequency.png")
    plt.savefig(name, dpi=300)
    wandb.log({f"{label_type.capitalize()} Amino Acid Frequency": wandb.Image(name)})
    plt.close()

    # Host Label Distribution
    host_counts = {
        "Train": [train_df["label_human"].sum(), train_df["label_avian"].sum(), train_df["label_other"].sum()],
        "Eval": [eval_df["label_human"].sum(), eval_df["label_avian"].sum(), eval_df["label_other"].sum()],
        "Test": [test_df["label_human"].sum(), test_df["label_avian"].sum(), test_df["label_other"].sum()],
    }

    x = range(3)
    labels_bar = ["Human", "Avian", "Other"]
    bar_width = 0.3
    # plt.bar([i - 0.3 for i in x], host_counts["Train"], width=0.3, label="Train", color=colors["Train"])
    # plt.bar(x, host_counts["Eval"], width=0.3, label="Eval", color=colors["Eval"])
    # plt.bar([i + 0.3 for i in x], host_counts["Test"], width=0.3, label="Test", color=colors["Test"])
    plt.bar([i - bar_width for i in x], host_counts["Train"], width=bar_width, label="Train",
            color=colors["Train"], edgecolor='black', linewidth=1.2)
    plt.bar(x, host_counts["Eval"], width=bar_width, label="Eval",
            color=colors["Eval"], edgecolor='black', linewidth=1.2)
    plt.bar([i + bar_width for i in x], host_counts["Test"], width=bar_width, label="Test",
            color=colors["Test"], edgecolor='black', linewidth=1.2)
    plt.xticks(x, labels_bar)
    plt.ylabel("Count")
    plt.title(f"{label_type.capitalize()} Host Label Distribution")

    if label_type == "human":
        legend = plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1))
        for patch in legend.get_patches():
            patch.set_edgecolor('none')
    else:
        legend = plt.legend()
        for patch in legend.get_patches():
            patch.set_edgecolor('none')

    plt.tight_layout()
    name = os.path.join(save_path, "host_label_distribution.png")
    plt.savefig(name, dpi=300)
    wandb.log({f"{label_type.capitalize()} Host Label Distribution": wandb.Image(name)})
    plt.close()


    # HA Subtype Distribution (Log Y only)
    train_ha = train_df["HA_subtype"].value_counts()
    eval_ha = eval_df["HA_subtype"].value_counts()
    test_ha = test_df["HA_subtype"].value_counts()
    ha_df = pd.DataFrame({"Train": train_ha, "Eval": eval_ha, "Test": test_ha}).fillna(0).astype(int)
    ha_df["Total"] = ha_df["Train"] + ha_df["Eval"] + ha_df["Test"]
    ha_df = ha_df.sort_values("Total", ascending=False).drop(columns="Total").reset_index().rename(columns={"index": "HA_subtype"})

    ha_df.plot(x="HA_subtype", y=["Train", "Eval", "Test"], kind="bar", figsize=(12, 6), color=[colors["Train"], colors["Eval"], colors["Test"]])
    for container in plt.gca().containers:
        plt.setp(container, edgecolor='black')

    plt.ylabel("Count (log scale)")
    plt.yscale("log")
    plt.title(f"{label_type.capitalize()} HA Subtype Distribution (Log Y)")
    plt.tight_layout()
    name = os.path.join(save_path, "ha_subtype_distribution_log.png")
    plt.savefig(name, dpi=300)
    wandb.log({f"{label_type.capitalize()} HA Subtype Distribution (Log Y)": wandb.Image(name)})
    plt.close()

wandb.finish()
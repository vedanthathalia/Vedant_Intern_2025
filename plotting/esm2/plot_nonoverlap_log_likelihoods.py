import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

avian_unique_df = pd.read_csv("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/avian_unique_only.csv")
human_unique_df = pd.read_csv("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/human_unique_only.csv")

full_human_df = pd.read_csv("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/650M/final_human_model/global_test_log_likelihoods.csv")
full_avian_df = pd.read_csv("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/650M/final_avian_model_5epochs/global_test_log_likelihoods.csv")

avian_top50_df = pd.read_csv("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/avian_top50_label1.csv")
human_top50_df = pd.read_csv("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/human_top50_label1.csv")

avian_unique_ids = set(avian_unique_df['accession'])
human_on_avian_df = full_human_df[full_human_df['accession'].isin(avian_unique_ids)]

human_unique_ids = set(human_unique_df['accession'])
avian_on_human_df = full_avian_df[full_avian_df['accession'].isin(human_unique_ids)]

avian_ll = avian_on_human_df['log_likelihood'].dropna()
human_ll = human_on_avian_df['log_likelihood'].dropna()

solely_avian_df = avian_top50_df[
    (avian_top50_df['label_avian'] == 1) &
    (avian_top50_df['label_human'] == 0)
]
solely_avian_ll = solely_avian_df['log_likelihood'].dropna()

solely_human_df = human_top50_df[
    (human_top50_df['label_human'] == 1) &
    (human_top50_df['label_avian'] == 0)
]
solely_human_ll = solely_human_df['log_likelihood'].dropna()

all_vals = pd.concat([avian_ll, human_ll, solely_avian_ll, solely_human_ll])
x_min = all_vals.min() - 5
x_max = all_vals.max() + 5
x_vals = np.linspace(x_min, x_max, 300)

# PLOT 1: Avian model on human-unique sequences + solely avian
plt.figure(figsize=(10, 6))

avian_kde = gaussian_kde(avian_ll)
solely_avian_kde = gaussian_kde(solely_avian_ll)

plt.plot(x_vals, avian_kde(x_vals), color='green', linewidth=2, label=f"Avian LL on Human-Unique (n={len(avian_ll)})")
plt.plot(x_vals, solely_avian_kde(x_vals), color='orange', linestyle='--', linewidth=2, label=f"Solely Avian (n={len(solely_avian_ll)})")
plt.axvspan(-21.9777, -1.6289, color='gray', alpha=0.2, label="Avian Model 50–100% Range")

plt.title("Avian Model on Human-Unique + Solely Avian Sequences")
plt.xlabel("Log-Likelihood")
plt.ylabel("Density")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures/Avian/650M_5epochs/Overlap/avian_on_human_unique_plus_solely_avian_kde.png")
plt.show()

# PLOT 2: Human model on avian-unique sequences + solely human
plt.figure(figsize=(10, 6))

human_kde = gaussian_kde(human_ll)
solely_human_kde = gaussian_kde(solely_human_ll)

plt.plot(x_vals, human_kde(x_vals), color='blue', linewidth=2, label=f"Human LL on Avian-Unique (n={len(human_ll)})")
plt.plot(x_vals, solely_human_kde(x_vals), color='purple', linestyle='--', linewidth=2, label=f"Solely Human (n={len(solely_human_ll)})")
plt.axvspan(-12.5191, -0.0548, color='gray', alpha=0.2, label="Human Model 50–100% Range")

plt.title("Human Model on Avian-Unique + Solely Human Sequences")
plt.xlabel("Log-Likelihood")
plt.ylabel("Density")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures/Human/650M/Overlap/human_on_avian_unique_plus_solely_human_kde.png")
plt.show()

# === BOXPLOT of all 4 distributions ===
plt.figure(figsize=(10, 6))

# Collect data
data = [
    avian_ll,
    solely_avian_ll,
    human_ll,
    solely_human_ll
]

labels = [
    "Avian on Human-Unique",
    "Solely Avian",
    "Human on Avian-Unique",
    "Solely Human"
]

# Boxplot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

data = [
    avian_ll,
    solely_avian_ll,
    human_ll,
    solely_human_ll
]

labels = [
    "Avian on Human-Unique",
    "Solely Avian",
    "Human on Avian-Unique",
    "Solely Human"
]

colors = ['green', 'orange', 'blue', 'purple']

for i, (vals, label, color) in enumerate(zip(data, labels, colors), start=1):
    box = plt.boxplot(
        vals,
        positions=[i],
        patch_artist=True,
        widths=0.6,
        boxprops=dict(facecolor=color, alpha=0.6),
        flierprops=dict(marker='o', markerfacecolor=color, markeredgecolor=color, markersize=5),
        medianprops=dict(color='black'),
        whiskerprops=dict(color=color),
        capprops=dict(color=color)
    )

plt.xticks(range(1, 5), labels)
plt.ylabel("Log-Likelihood")
plt.title("Log-Likelihood Distributions (Boxplot View)")
plt.tight_layout()

# Save
plt.savefig("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures/Boxplots/boxplot_log_likelihoods_all_groups.png")

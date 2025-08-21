import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from datetime import datetime
import wandb
import os
import seaborn as sns

labels = ["human", "avian", "other"]
base_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/650M/final_{}_model/{}_log_likelihoods.csv"
fig_base_dir = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(project="Figures", name=f"log_likelihood_all_3plots_{timestamp}")

def kde_plot(df, label):
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
    csv_file = base_path.format(label, label, label)
    if not os.path.exists(csv_file):
        print(f"Skipping {label}, file not found: {csv_file}")
        continue

    df = pd.read_csv(csv_file)
    subsets = kde_plot(df, label)

    stats_rows = []
    for subset_name, values in subsets.items():
        stats_rows.append({
            "Subset": subset_name,
            "Mean": values.mean(),
            "Std": values.std(),
            "Median": values.median(),
            "Min": values.min(),
            "Max": values.max(),
            "Count": values.count()
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_csv_path = f"/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/650M/final_{label}_model/{label}_log_likelihood_stats.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"\nSaved stats for {label.upper()} to {stats_csv_path}")
    print(stats_df)

    all_vals = pd.concat(subsets.values())
    x_min = max(all_vals.min(), -90)
    x_max = all_vals.max() + 5
    x_vals = np.linspace(x_min, x_max, 300)

    sns.set_context("notebook", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for (subset_name, values), color in zip(subsets.items(), colors):
        kde = gaussian_kde(values)
        ax.plot(x_vals, kde(x_vals), label=subset_name, color=color, linewidth=2)
        # Mark AVE16435.1 for the human plot
    if label == "human":
        accession_id = "AVE16435.1"
        point_row = df[df["accession"] == accession_id]
        if not point_row.empty:
            log_likelihood_value = point_row["log_likelihood"].values[0]

            # Get KDE of Non-Human curve
            nonhuman_key = [k for k in subsets if k.startswith("Non-")][0]
            nonhuman_values = subsets[nonhuman_key]
            nonhuman_kde = gaussian_kde(nonhuman_values)

            # Evaluate density at this log-likelihood
            density_y = nonhuman_kde(log_likelihood_value)

            # Plot the dot
            ax.plot(log_likelihood_value, density_y, 'o', color='black', zorder=5)

            # Annotate the point with an arrow
            ax.annotate(
                accession_id,
                xy=(log_likelihood_value, density_y),
                xytext=(log_likelihood_value - 15, density_y + 0.01),  # arrow comes from the left
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                ha='right',  # text right-aligned so it aligns properly to the left of the arrow
                va='bottom'
            )
        else:
            print(f"Accession {accession_id} not found in {label} dataset.")



    # ax.set_title(f"{label.capitalize()} 650M Trained KDE Plot")
    ax.set_xlabel("Log-Likelihood")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    label_folder = os.path.join(fig_base_dir, label.capitalize(), "650M")
    os.makedirs(label_folder, exist_ok=True)
    out_path = os.path.join(label_folder, f"log_likelihood_{label}_trained.svg")
    sns.despine()
    plt.savefig(out_path, transparent=True)
    # wandb.log({f"{label.capitalize()} Trained KDE Plot": wandb.Image(out_path)})
    
    # plt.show()

wandb.finish()

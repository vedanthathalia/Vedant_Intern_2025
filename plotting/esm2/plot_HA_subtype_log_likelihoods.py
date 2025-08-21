import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from datetime import datetime
import wandb
import os
import seaborn as sns

models = ["8M", "35M", "650M"]
base_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/{}/final_human_model/human_log_likelihoods.csv"
fig_out_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures/Human/AllM/log_likelihood_kde_by_HA_subtype_all_models.svg"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(project="Figures", name=f"log_likelihood_HA_KDE_all_models_{timestamp}")

fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
colors = plt.cm.tab20.colors

for i, model in enumerate(models):
    csv_path = base_path.format(model)
    if not os.path.exists(csv_path):
        print(f"Skipping {model}: file not found")
        continue

    df = pd.read_csv(csv_path)

    df_human_only = df[
        (df["label_human"] == 1) &
        (df["label_other"] == 0) &
        (df["label_avian"] == 0)
    ].dropna(subset=["log_likelihood", "HA_subtype"])

    if df_human_only.empty:
        print(f"No valid data for {model}")
        continue

    x_min = max(df_human_only["log_likelihood"].min(), -90)
    x_max = df_human_only["log_likelihood"].max() + 5
    x_vals = np.linspace(x_min, x_max, 300)

    ax = axs[i]
    top_has = df_human_only["HA_subtype"].value_counts().nlargest(5).index.tolist()
    handles = []

    for j, ha in enumerate(top_has):
        values = df_human_only[df_human_only["HA_subtype"] == ha]["log_likelihood"]
        if len(values) < 10:
            continue
        kde = gaussian_kde(values)
        label = f"{ha} (n={len(values)})"
        line, = ax.plot(x_vals, kde(x_vals), label=label, color=colors[j % len(colors)], linewidth=2)
        handles.append(line)

    # ax.set_title(f"{model} Model")
    ax.set_xlabel("Log-Likelihood")
    if i == 0:
        ax.set_ylabel("Density")
    ax.legend(loc="upper left", frameon=False)

    # Individual Figures
    sns.set_context("notebook", font_scale=1.2)
    fig_individual, ax_ind = plt.subplots(figsize=(5, 5))

    for j, ha in enumerate(top_has):
        values = df_human_only[df_human_only["HA_subtype"] == ha]["log_likelihood"]
        if len(values) < 10:
            continue
        kde = gaussian_kde(values)
        label = f"{ha} (n={len(values)})"
        ax_ind.plot(x_vals, kde(x_vals), label=label, color=colors[j % len(colors)], linewidth=2)

    # ax_ind.set_title(f"{model} Model - Human Only")
    ax_ind.set_xlabel("Log-Likelihood")
    ax_ind.set_ylabel("Density")
    ax_ind.legend(loc="upper left", frameon=False)

    individual_out_path = f"/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures/Human/{model}/log_likelihood_kde_by_HA_subtype.svg"
    os.makedirs(os.path.dirname(individual_out_path), exist_ok=True)
    fig_individual.tight_layout()
    sns.despine()
    fig_individual.savefig(individual_out_path, bbox_inches='tight', transparent=True)
    # wandb.log({f"KDE by HA Subtype ({model})": wandb.Image(individual_out_path)})
    plt.close(fig_individual)  # prevent overlap in plt.show()


# plt.suptitle("Human Only Log-Likelihood KDE by HA Subtype (All Models)", fontsize=16)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout()
os.makedirs(os.path.dirname(fig_out_path), exist_ok=True)
sns.despine()
plt.savefig(fig_out_path, bbox_inches='tight', transparent=True)
# wandb.log({"KDE by HA Subtype (All Models)": wandb.Image(fig_out_path)})

wandb.finish()

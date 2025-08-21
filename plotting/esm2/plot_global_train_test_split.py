import pandas as pd
import matplotlib.pyplot as plt
import os
import wandb
from datetime import datetime

train_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/global_train_set.csv"
test_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/global_test_set.csv"
fig_dir = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures/All"
os.makedirs(fig_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(project="Figures", name=f"global_train_test_host_distribution_{timestamp}")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Pie Chart
plt.figure(figsize=(6, 6))
sizes = [len(df_train), len(df_test)]
labels = ["Train", "Test"]
colors = ["#54A24B", "#F58518"]

plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops=dict(edgecolor='black', linewidth=1)
)
plt.title("Train vs Test Sequence Count (Pie Chart)")
plt.axis('equal')
plt.tight_layout()
pie_path = os.path.join(fig_dir, "global_train_vs_test_pie_chart.png")
plt.savefig(pie_path)
wandb.log({"Train vs Test Pie Chart": wandb.Image(pie_path)})
plt.show()

# Histogram
host_labels = ['label_human', 'label_avian', 'label_other']
host_names = ['Human', 'Avian', 'Other']
host_colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

counts_train = [df_train[label].sum() for label in host_labels]
counts_test = [df_test[label].sum() for label in host_labels]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Train subplot
axes[0].bar(host_names, counts_train, color=host_colors, edgecolor='black', linewidth=1)
axes[0].set_title("Train Set Host Distribution")
axes[0].set_xlabel("Host Type")
axes[0].set_ylabel("Number of Sequences")

# Test subplot
axes[1].bar(host_names, counts_test, color=host_colors, edgecolor='black', linewidth=1)
axes[1].set_title("Test Set Host Distribution")
axes[1].set_xlabel("Host Type")

plt.suptitle("Train vs Test Host Type Distribution (Histogram)")
plt.tight_layout(rect=[0, 0, 1, 0.93])
hist_path = os.path.join(fig_dir, "global_train_test_host_histogram.png")
plt.savefig(hist_path)
wandb.log({"Train vs Test Histogram": wandb.Image(hist_path)})
plt.show()

wandb.finish()

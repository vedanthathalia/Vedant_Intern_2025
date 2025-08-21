import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import wandb
import seaborn as sns

tsv_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/NCBI_Dataset/data80_clu_vectorized.tsv"
df = pd.read_csv(tsv_path, sep='\t')

label_cols = ['label_human', 'label_avian', 'label_other']
df['num_hosts'] = df[label_cols].sum(axis=1)

counts_series = df['num_hosts'].value_counts().reindex([1, 2, 3], fill_value=0)
counts = counts_series.tolist()  # [45746, 1096, 69]
x_labels = ['1', '2', '3']
x = np.arange(len(counts))
width = 0.8

wandb.init(project="Figures", name="host_diversity_final")

fig, ax = plt.subplots(figsize=(5.5, 6))
bars = ax.bar(x, counts, width=width, color='steelblue', zorder=2)

ax.set_xlabel('Number of unique hosts')
ax.set_ylabel('Count (log scale)')

overlay_heights = [40000, 2000, 200]

for i in range(len(counts)):
    height = overlay_heights[i]
    bar = bars[i]
    if i == 0:
        bottom = bar.get_height() - height
        facecolor = 'white'
        hatch_color = 'white'
        zorder = 4
        rect = patches.Rectangle(
            (bar.get_x(), bottom),
            bar.get_width(),
            height,
            hatch='//',
            edgecolor='lightgray',
            facecolor='none',
            lw=2,
            zorder=zorder
        )
    else:
        bottom = bar.get_height()
        facecolor = 'white'
        hatch_color = 'gray'
        zorder = 4

        rect = patches.Rectangle(
            (bar.get_x(), bottom),
            bar.get_width(),
            height,
            hatch='//',
            edgecolor='gray',
            facecolor='none',
            lw=2,
            zorder=zorder
        )
    ax.add_patch(rect)
    
    ax.set_yscale('log')
    ax.relim()
    ax.autoscale_view()

    if i == 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bottom + height + 3000,
            f"{counts[i]:.1f}",
            ha='center',
            va='bottom',
            fontsize=11,
            zorder=5
        )
    else:
        label_y = bar.get_height()
        if i == 2:
            label_y = bar.get_height()
        else:
            label_y = bar.get_height() + 100

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{counts[i]:.1f}",
            ha='center',
            va='bottom',
            fontsize=11,
            zorder=5
        )

diverse_total = counts[1] + counts[2]
percent = 100 * diverse_total / sum(counts)
summary_text = f"Sequences with diverse hosts: {diverse_total}\n{percent:.2f}% of total sequences"

bbox_props = dict(boxstyle="round,pad=0.5", edgecolor='black', facecolor='white')
box_x = x[1]
box_y = 35000

ax.text(
    box_x, box_y,
    summary_text,
    ha='left', va='center',
    fontsize=10,
    bbox=bbox_props,
    zorder=6
)

arrowprops = dict(arrowstyle='->', color='gray', lw=2, connectionstyle="arc3,rad=-0.3") 
center_x_0 = bars[0].get_x() + width / 2
start_y = bars[0].get_height()
end_y_1 = bars[1].get_height() + overlay_heights[1] - 50
end_y_2 = bars[2].get_height() + overlay_heights[2]

ax.annotate("", xy=(bars[1].get_x() + width / 2, end_y_1), xytext=(center_x_0, start_y), arrowprops=arrowprops, zorder=3)
ax.annotate("", xy=(bars[2].get_x() + width / 2, end_y_2), xytext=(center_x_0, start_y), arrowprops=arrowprops, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.tick_params(axis='y', which='both', right=False)
ax.tick_params(axis='x', which='both', top=False)
ax.grid(False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

save_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/Figures/All/transparent_host_diversity_arrowed.png"
plt.tight_layout(pad=0.5)
plt.savefig(save_path, dpi=300, transparent=True)
wandb.save(save_path)
plt.close()

print(f"Saved to: {save_path}")


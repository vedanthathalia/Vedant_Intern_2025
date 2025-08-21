import pandas as pd
import numpy as np
import json
from scipy.stats import gaussian_kde

df = pd.read_csv("/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/650M/final_avian_model/avian_log_likelihoods.csv")

avian_only_df = df[
    (df["label_avian"] == 1) &
    (df["label_human"] == 0) &
    (df["label_other"] == 0)
]
log_likelihoods = avian_only_df["log_likelihood"].values

kde = gaussian_kde(log_likelihoods)
x_vals = np.linspace(log_likelihoods.min(), log_likelihoods.max(), 1000)
densities = kde(x_vals)
peak_ll = x_vals[np.argmax(densities)]

tolerance = 0.05 * abs(peak_ll)
lower_bound = peak_ll - tolerance
upper_bound = peak_ll + tolerance

print(f"KDE Peak: {peak_ll:.4f}")
print(f"±5% window: [{lower_bound:.4f}, {upper_bound:.4f}]")

nonavian_df = df[
    (df["label_avian"] == 0) &
    ((df["label_human"] == 1) | (df["label_other"] == 1))
]

overlapping_df = nonavian_df[
    (nonavian_df["log_likelihood"] >= lower_bound) &
    (nonavian_df["log_likelihood"] <= upper_bound)
]

print(f"Filtered non-avian sequences in KDE window: {len(overlapping_df)}")

output_path = "/home/vhathalia/SCRIPPS_jupyter/ESM_Model/log_likelihood_results/650M/final_avian_model/overlapping_nonavian_kde_window.csv"
overlapping_df.to_csv(output_path, index=False)
print(f"Saved full overlapping metadata to: {output_path}")
print(pd.read_csv(output_path).shape)

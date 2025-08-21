# plotting/

This folder contains plotting scripts for visualizing results.

### esm2/  
- plot_3d_umap_embeddings.py – 3D visualization of UMAP embeddings  
- plot_all_3d_umap_embeddings.py – compares 3D embeddings across models  
- plot_all_umap_embeddings.py – compares 2D UMAP embeddings across models  
- plot_umap_embeddings.py – plots UMAP embeddings for one model  
- plot_all_log_likelihoods.py – plots log-likelihood distributions for all hosts
- plot_log_likelihoods.py – plots log-likelihoods for a single model  
- plot_nonoverlap_log_likelihoods.py – plots only non-overlapping log-likelihood regions  
- plot_HA_subtype_log_likelihoods.py – plots log-likelihoods by HA subtype  
- plot_auc_roc_performance.py – plots ROC curve performance
- plot_dataset_figures.py – generates dataset summary figures  
- plot_global_train_test_split.py – plots global train/test splits  
- plot_host_diversity_barplot.py – plots diversity of host species  
- plot_regression.py – regression analysis plots of avian and human host sequences

### nextstrain/  
- plot_avian_distance_boxplot.py – boxplot of avian sequence distances to human sequences in phylogenetic tree
- plot_avian_distance_violinplot.py – violin plot of avian sequence distances
- plot_avian_distance_violinplot_box.py – violin plot of avian sequences with box plot overlay  
- plot_host_wedges_baltic.py – saves phylogenetic tree image with wedges for clades with single hosts

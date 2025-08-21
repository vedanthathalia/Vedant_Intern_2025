# esm2/

This folder contains scripts for fine-tuning and evaluating the ESM-2 protein language model.

- ESM2_fitness_prediction.py – uses ESM-2 to score viral fitness and generate log-likelihood values 
- ESM2_train.py – fine-tuning script for ESM-2 on PB2 protein sequences  
- auc_roc_compare_performance.py – compares ROC performance of fine-tuned vs. baseline models  
- cluster_split.py – clusters sequences and splits into train/test sets
- extract_rep_sequence.py – extracts representative sequences per cluster  
- load_dataset.py – loads PB2 sequence datasets into the pipeline
- log_likelihood_host_overlap.py – calculates overlap in log-likelihood distributions between hosts  
- save_embeddings.py – saves ESM-2 embeddings of PB2 sequences
- save_umap_embeddings.py – saves UMAP-reduced embeddings for visualization  

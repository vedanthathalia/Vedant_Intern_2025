# Vedant_Intern_2025

# Predicting Human Infectivity of Influenza A Sequences  

This repository contains my work from my Summer 2025 internship at The Scripps Research Institute (SURI Program), where I worked in the Andersen Lab on an influenza-focused computational biology project.  

## Project Overview  

The goal of this project was to better understand and predict the zoonotic potential of influenza A viruses, particularly how non-human infecting virus sequences could copy their viral RNA into human hosts through mutations.  

- **Data Collection & Preprocessing**  
  - Retrieved PB2 protein sequences from NCBI.  
  - Clustered sequences using MMseqs2 at 80% identity to avoid redundancy and data leakage.  

- **Modeling Approach**
  - Finetuned an ESM2 masked language model (MLM) solely on human-infective sequences to capture patterns associated with human infectivity.  
  - Used log-likelihood scores to assess how closely non-human sequences resembled human-adaptive patterns.  
  - Trained an XGBoost classifier head on vectorized embeddings to compare finetuned and base model performance.

- **Results**
  - Overlapping log-likelihoods revealed non-human sequences with human-like patterns, suggesting zoonotic potential.  
  - Certain H1 and H5 subtype mutations (e.g., G590S, T271A, I147T) were common in high-log-likelihood non-human sequences, and are known to increase polymerase activity or viral replication in human cells.  
  - Fine-tuned models outperformed base models, improving predictive accuracy.  

- **Future Work**  
  - Extend to longer-context models such as Evo-2.
  - Incorporate multiple influenza genome segments beyond PB2.  
  - Train directly on nucleotide sequences to analyze mutations on a more granular level.  

## Repository Structure  

- alignment/ - scripts and FASTA files for saving and aligning avian and human PB2 sequences.  
- esm2/ - code for fine-tuning and evaluating the ESM-2 protein language model.  
- figtree/ - tree files and scripts for annotating phylogenetic trees.  
- figures/ - all generated figures from model fine-tuning and Nextstrain analysis.  
- future_work - exploratory scripts for next steps, including Evo model training.  
- mutations - scripts to extract mutations and compare them with known human-adaptive mutations.  
- nextstrain - files related to extracting and annotating mutations for Nextstrain visualizations.  
- plotting - plotting scripts for visualizing log-likelihoods, ROC curves, embeddings, and tree results.  

## Significance  

Influenza A continues to pose a pandemic threat due to its ability to cross species. By identifying patterns and mutations that enable human adaptation, this work supports early detection of zoonotic risks and contributes to pandemic prevention efforts.  

## Acknowledgements  

This research was made possible by the SURI Internship Program at the Scripps Translational Institute, with mentorship from Praneeth Gangavarapu and the Andersen Lab.

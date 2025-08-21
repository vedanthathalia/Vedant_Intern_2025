import os
import sys
import subprocess
import pandas as pd
import glob
import shutil
import wandb
import torch
from tqdm import tqdm

def run(cmd):
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def csv_to_fasta(csv_path, base_name="csv_seqs", seq_col="sequence"):
    df = pd.read_csv(csv_path)
    if seq_col not in df.columns:
        print(f"ERROR: column '{seq_col}' not found in {csv_path}", file=sys.stderr)
        sys.exit(1)

    n = len(df)
    n_train = int(0.9 * n)
    n_val = int(0.05 * n)
    train, val, test = df[:n_train], df[n_train:n_train+n_val], df[n_train+n_val:]

    for name, split_df in zip(["train", "val", "test"], [train, val, test]):
        fasta_file = f"{base_name}_{name}.fa"
        with open(fasta_file, "w") as fa:
            for i, seq in enumerate(tqdm(split_df[seq_col], desc=f"Writing {name}.fa", unit="seq")):
                fa.write(f">seq{i}\n{seq}\n")
        print(f"Wrote {len(split_df)} sequences to {fasta_file}")

def write_preprocess_yaml(base_name, out_dir, path, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    blocks = []
    for split, tr, val, tst in [
        ("train", 1.0, 0.0, 0.0),
        ("val",   0.0, 1.0, 0.0),
        ("test",  0.0, 0.0, 1.0)
    ]:
        blocks.append(f"""- datapaths: ["{os.path.abspath(base_name)}_{split}.fa"]
  output_dir: "{os.path.abspath(out_dir)}"
  output_prefix: csv_fasta_uint8
  train_split: {tr}
  valid_split: {val}
  test_split: {tst}
  overwrite: True
  embed_reverse_complement: true
  random_reverse_complement: 0.0
  random_lineage_dropout: 0.0
  include_sequence_id: false
  transcribe: "back_transcribe"
  force_uppercase: false
  indexed_dataset_dtype: "uint8"
  tokenizer_type: "Byte-Level"
  fast_hf_tokenizer: true
  append_eod: true
  drop_empty_sequences: true
  nnn_filter: false
  seed: {seed}
  workers: 1
  preproc_concurrency: 100000
  chunksize: 25
""")
    with open(path, "w") as f:
        f.write("\n".join(blocks))
    print(f"Wrote {path}")

def write_training_yaml(prep_dir, path):
    pfx = os.path.join(prep_dir, "csv_fasta_uint8_byte-level")
    content = f"""- dataset_prefix: {pfx}_train
  dataset_split: train
  dataset_weight: 1.0
- dataset_prefix: {pfx}_val
  dataset_split: validation
  dataset_weight: 1.0
- dataset_prefix: {pfx}_test
  dataset_split: test
  dataset_weight: 1.0
"""
    with open(path, "w") as f:
        f.write(content)
    print(f"Wrote {path}")

def main():
    csv_path = "/home/vhathalia/SCRIPPS_jupyter/EVO2_Model/pb2_nucleotide_sequences.csv"
    seq_col = "sequence"
    model_hf = "hf://arcinstitute/savanna_evo2_1b_base"
    label = os.path.splitext(os.path.basename(csv_path))[0]

    print("\nCUDA avilable")
    if torch.cuda.is_available():
        print(f"CUDA available - {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available")

    for f in [
        "preprocessed_data", "pretraining_demo", "nemo2_evo2_1b_8k",
        "preprocess_config.yaml", "training_data_config.yaml",
        "csv_seqs_train.fa", "csv_seqs_val.fa", "csv_seqs_test.fa",
        "model_final.ckpt", "test_results.json"
    ]:
        if os.path.isdir(f):
            shutil.rmtree(f)
        elif os.path.exists(f):
            os.remove(f)

    csv_to_fasta(csv_path, base_name="csv_seqs", seq_col=seq_col)
    write_preprocess_yaml("csv_seqs", "preprocessed_data", "preprocess_config.yaml", seed=42)
    write_training_yaml("preprocessed_data", "training_data_config.yaml")

    if shutil.which("preprocess_evo2") is None:
        print("ERROR: 'preprocess_evo2' not found in PATH", file=sys.stderr)
        sys.exit(1)
    run(["preprocess_evo2", "--config", "preprocess_config.yaml"])

    if shutil.which("evo2_convert_to_nemo2") is None:
        print("ERROR: 'evo2_convert_to_nemo2' not found in PATH", file=sys.stderr)
        sys.exit(1)
    run([
        "evo2_convert_to_nemo2",
        "--model-path", model_hf,
        "--model-size", "1b",
        "--output-dir", "nemo2_evo2_1b_8k"
    ])

    wandb.login(key="4ccf6e6d655e18a2f2f1a5dee3114419a24080e0")
    wandb.init(
        project="EVO2",
        group="evo2_experiments",
        job_type="causal_lm",
        notes=f"Full finetuning Evo2 on {label}",
        config={
            "random_state": 42,
            "epochs": 1,
        }
    )

    if shutil.which("train_evo2") is None:
        print("ERROR: 'train_evo2' not found in PATH", file=sys.stderr)
        sys.exit(1)
    run([
        "train_evo2",
        "-d", "training_data_config.yaml",
        "--dataset-dir", "preprocessed_data",
        "--experiment-dir", "pretraining_demo",
        "--model-size", "1b",
        "--devices", "1",
        "--num-nodes", "1",
        "--seq-length", "1024",
        "--micro-batch-size", "2",
        "--lr", "0.0001",
        "--warmup-steps", "5",
        "--max-epochs", "1",
        "--ckpt-dir", "nemo2_evo2_1b_8k",
        "--clip-grad", "1",
        "--wd", "0.01",
        "--activation-checkpoint-recompute-num-layers", "1",
        "--val-check-interval", "50",
        "--ckpt-async-save",
    ])

    print("\nLocating final checkpoint")
    ckpts = glob.glob("pretraining_demo/**/*.ckpt", recursive=True)
    if not ckpts:
        print("No checkpoint found", file=sys.stderr)
        sys.exit(1)

    latest = sorted(tqdm(ckpts, desc="Scanning checkpoints"))[-1]
    shutil.copy(latest, "model_final.ckpt")
    print(f"Final model saved as model_final.ckpt (from {latest})")

    print("\nRunning test evaluation (not logged to wandb)...")
    if shutil.which("evaluate_evo2") is None:
        print("ERROR: 'evaluate_evo2' not found in PATH", file=sys.stderr)
        sys.exit(1)

    run([
        "evaluate_evo2",
        "--checkpoint-path", "model_final.ckpt",
        "--data-config", "training_data_config.yaml",
        "--dataset-dir", "preprocessed_data",
        "--output", "test_results.json",
        "--eval-split", "test"
    ])
    print("Saved test results to test_results.json")

if __name__ == "__main__":
    main()

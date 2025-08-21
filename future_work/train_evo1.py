import os
import random
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    set_seed,
)
import wandb
from datetime import datetime

run_name = f"causal-lm-pb2-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TRAIN_CSV = "/home/vhathalia/SCRIPPS_jupyter/EVO_Model/NCBI_Dataset/100similarity/clustered_train_human.csv"
EVAL_CSV  = "/home/vhathalia/SCRIPPS_jupyter/EVO_Model/NCBI_Dataset/100similarity/clustered_eval_human.csv"
TEST_CSV  = "/home/vhathalia/SCRIPPS_jupyter/EVO_Model/NCBI_Dataset/100similarity/clustered_test_human.csv"
TEXT_COLUMN = "sequence"

MODEL_NAME = "togethercomputer/evo-1-8k-base"
OUTPUT_DIR = "./evo1_finetuned"
SEED = 42
MAX_LENGTH = 1024
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5

wandb.init(
    project="Evo1-Finetuning",
    name=run_name,
    config={
        "model": MODEL_NAME,
        "max_seq_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": EPOCHS,
        "seed": SEED,
        "run_time": datetime.now().isoformat(),
        "train_path": TRAIN_CSV,
        "eval_path": EVAL_CSV,
        "test_path": TEST_CSV
    }
)

set_seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

model_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True, revision="1.1_fix")
model_config.use_cache = False

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=model_config, trust_remote_code=True, revision="1.1_fix")
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = "X"  # Evo requires single-character pad

def load_dataset_from_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TEXT_COLUMN])
    df = df[[TEXT_COLUMN]].copy().rename(columns={TEXT_COLUMN: "text"})
    return Dataset.from_pandas(df)

train_ds = load_dataset_from_csv(TRAIN_CSV)
eval_ds  = load_dataset_from_csv(EVAL_CSV)
test_ds  = load_dataset_from_csv(TEST_CSV)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=False, padding="max_length", max_length=MAX_LENGTH)

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
eval_ds  = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
test_ds  = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    run_name=run_name,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    seed=SEED,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    warmup_ratio=0.1,
    fp16=True,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

train_metrics = trainer.train()
wandb.log(train_metrics.metrics)

eval_metrics = trainer.evaluate()
wandb.log(eval_metrics)

print("\nRunning final evaluation on TEST set...")
test_metrics = trainer.evaluate(test_dataset=test_ds)

final_path = os.path.join(OUTPUT_DIR, "final_model")
os.makedirs(final_path, exist_ok=True)
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

import json
with open(os.path.join(final_path, "test_metrics.json"), "w") as f:
    json.dump(test_metrics, f, indent=2)

print(f"\nModel and tokenizer saved to: {final_path}")
print(f"Test metrics saved to: {final_path}/test_metrics.json")

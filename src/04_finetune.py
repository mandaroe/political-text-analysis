import os
import pandas as pd
import torch
import argparse

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------
# Setup
# ----------------

MODEL_NAME = "bert-base-uncased"

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["bias", "topic"], default="bias")
args = parser.parse_args()

TASK = args.task

# ----------------
# Load Data
# ----------------

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

# ----------------
# Label Encoding
# ----------------

label_col = TASK

labels = sorted(train[label_col].unique())
label_map = {label: i for i, label in enumerate(labels)}

for df in [train, val, test]:
    df["label"] = df[label_col].map(label_map)

y_train = train["label"].values
y_val = val["label"].values
y_test = test["label"].values

X_train = train["page_text"].tolist()
X_val = val["page_text"].tolist()
X_test = test["page_text"].tolist()

num_labels = len(label_map)

# ----------------
# Tokenizer
# ----------------

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128
    )

train_encodings = tokenize(X_train)
val_encodings = tokenize(X_val)
test_encodings = tokenize(X_test)

# ----------------
# Dataset Class
# ----------------

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = NewsDataset(train_encodings, y_train)
val_dataset = NewsDataset(val_encodings, y_val)
test_dataset = NewsDataset(test_encodings, y_test)

# ----------------
# Model
# ----------------

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

# ----------------
# Training Args
# ----------------

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=False
)

# ----------------
# Metrics
# ----------------

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# ----------------
# Trainer
# ----------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# ----------------
# Save Model
# ----------------

os.makedirs("models", exist_ok=True)
trainer.save_model(f"models/finetuned_{TASK}")
tokenizer.save_pretrained(f"models/finetuned_{TASK}")
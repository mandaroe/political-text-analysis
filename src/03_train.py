import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

# -----
# Setup
# -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "bert-base-uncased"

# -----------------
# Experiment Config
# -----------------

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["bias", "topic"], default="bias")
args = parser.parse_args()

TASK = args.task

# ---------
# Load Data
# ---------

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

# --------------
# Label Encoding 
# --------------

label_col = TASK

labels = sorted(train[label_col].unique())
label_map = {label: i for i, label in enumerate(labels)}

for df in [train, val, test]:
    df["label"] = df[label_col].map(label_map)

y_train = torch.tensor(train["label"].values).to(device)
y_val = torch.tensor(val["label"].values).to(device)
y_test = torch.tensor(test["label"].values).to(device)

# -----------------------
# Tokenizer & Frozen BERT
# -----------------------

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

frozen_bert = BertModel.from_pretrained(MODEL_NAME)
frozen_bert.to(device)
frozen_bert.eval()

# ------------------
# Embedding Function
# ------------------

def get_bert_embeddings(text_list, batch_size=16):
    embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = frozen_bert(**tokens)
            batch_embeddings = outputs.pooler_output

        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0)

# ------------------
# Extract Embeddings
# ------------------

X_train_emb = get_bert_embeddings(train["page_text"].tolist())
X_val_emb = get_bert_embeddings(val["page_text"].tolist())
X_test_emb = get_bert_embeddings(test["page_text"].tolist())

# ---------
# Classifier
# ---------

num_labels = len(label_map)

model = nn.Linear(768, num_labels).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------
# Training Loop
# ---------

for epoch in range(10):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train_emb)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    train_preds = outputs.argmax(dim=1)

    train_acc = accuracy_score(y_train.cpu(), train_preds.cpu())

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_emb)
        val_preds = val_outputs.argmax(dim=1)

    val_acc = accuracy_score(y_val.cpu(), val_preds.cpu())

    print(f"Epoch {epoch+1}")
    print(f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

# ---------
# Save Model
# ---------

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), f"models/frozen_{TASK}_classifier.pt")

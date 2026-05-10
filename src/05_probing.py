import os
import numpy as np
import pandas as pd
import torch
import argparse
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE

# ----------------
# Setup
# ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"

np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# -----------------
# Experiment Config
# -----------------

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["bias", "topic"], default="bias")
parser.add_argument("--mode", choices=["frozen", "finetune"], default="frozen")
args = parser.parse_args()

TASK = args.task
MODE = args.mode

print("\n" + "=" * 70)
print("🧠 BERT PROBING EXPERIMENT")
print("=" * 70)

print(f"Task      : {TASK}")
print(f"Mode      : {MODE}")
print(f"Model     : {MODEL_NAME}")
print(f"Device    : {device}")
print("=" * 70 + "\n")

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

y_train_np = y_train.numpy()
y_test_np = y_test.numpy()

# ----------------
# Load Model
# ----------------

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

model = BertModel.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
).to(device)

model.eval()

# -------------------------
# Embedding Extraction
# -------------------------

def extract_cls_embeddings(texts, model, tokenizer, batch_size=16):
    layer_outputs = [[] for _ in range(13)]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        for layer_idx, layer in enumerate(outputs.hidden_states):
            cls = layer[:, 0, :].cpu()
            layer_outputs[layer_idx].append(cls)

        del inputs, outputs
        torch.cuda.empty_cache()

    return [torch.cat(layer, dim=0).numpy() for layer in layer_outputs]

# -------------------------
# Extract embeddings
# -------------------------

train_layers = extract_cls_embeddings(train["page_text"].tolist(), model, tokenizer)
val_layers   = extract_cls_embeddings(val['page_text'].tolist(), model, tokenizer)
test_layers = extract_cls_embeddings(test["page_text"].tolist(), model, tokenizer)

# -------------------------
# Metrics storage
# -------------------------

layer_metrics = {
    "layer": [],
    "accuracy": [],
    "f1": [],
    "precision": [],
    "recall": []
}

# -------------------------
# Main probing loop
# -------------------------

for layer_idx in range(13):
    
    print("\n" + "=" * 50)
    print(f"Layer {layer_idx}")
    print("=" * 50)

    X_tr = train_layers[layer_idx]
    X_te = test_layers[layer_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_train_np)

    preds = clf.predict(X_te)

    acc = accuracy_score(y_test_np, preds)
    f1 = f1_score(y_test_np, preds, average="macro", zero_division=0)
    prec = precision_score(y_test_np, preds, average="macro", zero_division=0)
    rec = recall_score(y_test_np, preds, average="macro", zero_division=0)

    layer_metrics["layer"].append(layer_idx)
    layer_metrics["accuracy"].append(acc)
    layer_metrics["f1"].append(f1)
    layer_metrics["precision"].append(prec)
    layer_metrics["recall"].append(rec)
    
    print(f"Train Accuracy : {train_acc:.3f}")
    print(f"Test Accuracy  : {test_acc:.3f}")
    print(f"F1 Score       : {f1:.3f}")
    print(f"Precision      : {prec:.3f}")
    print(f"Recall         : {rec:.3f}")


# -------------------------
# Save results
# -------------------------

os.makedirs("results", exist_ok=True)

pd.DataFrame(layer_metrics).to_csv(
    f"results/probing_{TASK}_{MODE}.csv",
    index=False
)

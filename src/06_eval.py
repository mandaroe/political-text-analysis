import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertForSequenceClassification, BertTokenizer
import joblib

test = pd.read_csv("data/processed/test.csv")

labels = sorted(test["bias"].unique())
label_map = {l: i for i, l in enumerate(labels)}
test["label"] = test["bias"].map(label_map)
y_test = test["label"].values

X_test = test["page_text"].tolist()

# ------------
# Frozen Model
# ------------

frozen_model = joblib.load("models/frozen_bias_classifier.pkl")
X_test_emb = np.load("data/processed/test_embeddings.npy")

preds = frozen_model.predict(X_test_emb)

print("Frozen Model Results:")
print("Accuracy:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds, average="macro"))

# ----------------
# Fine-Tuned Model
# ----------------

model = BertForSequenceClassification.from_pretrained("models/finetuned_bias")

trainer = Trainer(model=model)

outputs = trainer.predict(test_dataset)
preds = outputs.predictions.argmax(-1)

print("Fine-tuned Model Results:")
print("Accuracy:", accuracy_score(y_test, preds))
print("F1:", f1_score(y_test, preds, average="macro"))

# ----------------
# Comparison Table
# ----------------

results = pd.DataFrame({
    "model": ["frozen", "finetuned"],
    "accuracy": [acc_frozen, acc_ft],
    "f1": [f1_frozen, f1_ft]
})

print(results)
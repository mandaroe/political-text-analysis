# Workflow

## Navigating GitHub

**Google Colab (one time only)**

```python
# Check if Git is installed
!git --version

# Clone Repo & Pull updates
!git clone https://github.com/username/political-text-analysis.git
!git pull
```
**Locally Stored Files (edited in RStudio)**

```PowerShell
# Push
git add .
git commit -m "message"
git push

# Pull
git pull
```

## Data Collection

**Liberal-Conservative News**
```python
# From BlueJwu on Hugging Face

from google.colab import auth
from getpass import getpass
import os
from huggingface_hub import login
from datasets import load_dataset

hf_token = getpass("Enter your Hugging Face token: ")

bluejwu = load_dataset("bluejwu/liberal-and-conservative-news", token = hf_token)
```
- More info on this dataset [here](https://huggingface.co/datasets/bluejwu/liberal-and-conservative-news/viewer/default/train?p=422)
- Will combine with another news dataset for more coverage of the liberal-conservative spread

**All the News**

```python
import kagglehub

path = kagglehub.dataset_download("davidmckinley/all-the-news-dataset")

import pandas as pd

news = pd.read_csv(
    f"{path}/all-the-news-2-1.csv",
    encoding="latin1",
    on_bad_lines="skip",
    low_memory=True,
    nrows=100000
)
```
- More info on All the News dataset [here](https://www.kaggle.com/datasets/davidmckinley/all-the-news-dataset?select=all-the-news-2-1.csv)

**Media Bias (AllSides)**
```r
# RStudio

library(AllSideR)
allsides_data <- allsides_data

write.csv(allsides_data, "allsides.csv", row.names = FALSE)
```
- Using for Media Bias Score
- Since the number of outlets in both datasets is small, we will just manually enter bias scores later.
- More info on AllSides [here](https://github.com/favstats/AllSideR)

## Data Processing

**News Articles**
```python
from urllib.parse import urlparse

def extract_outlet(example):
    example['outlet'] = urlparse(example['url']).netloc.replace("www.","")
    return example

bluejwu['train'] = bluejwu['train'].map(extract_outlet)
```
- Simplifies url column in BlueJwu

```python
# All the News

news["source_clean"] = news["publication"]

# BlueJwu

bluejwu_pd = bluejwu['train'].to_pandas()

bluejwu_pd["source_clean"] = bluejwu_pd["outlet"].replace({
    "foxnews.com": "Fox News",
    "theamericanconservative.com": "The American Conservative",
    "washingtontimes.com": "Washington Times",
    "cnn.com": "CNN",
    "msnbc.com": "MSNBC",
    "nytimes.com": "New York Times"
})
```

- Change publication column name in "All the News" for later merge
- Convert BlueJwu from HF to pandas object for later merge
- Rename publication names in BlueJwu for later merge 

```python
# Merge Two Datasets

combined = pd.concat([news, bluejwu_pd], ignore_index=True)
```

```python
bias_map = {
    "Fox News": "5",
    "CNN": "2",
    "MSNBC": "1",
    "New York Times": "2",
    "Reuters": "3",
    "Vox": "1",
    "Business Insider": "3",
    "The American Conservative": "4",
    "Washington Times": "4",
    "Vice": "1",
}

combined["bias"] = combined["source_clean"].map(bias_map)
```
- Manually input bias scores for publications
- Add scores to combined dataset
- We can consider adding both labels (e.g. "left", "left-center") and continuous scores for each publication (we would still need to add numerical categorical values for BERT)

```python
allowed_sources = [
    "Fox News",
    "CNN",
    "MSNBC",
    "New York Times",
    "Reuters",
    "Vox",
    "Business Insider",
    "The American Conservative",
    "Washington Times"
]

combined = combined[combined["source_clean"].isin(allowed_sources)]
```
 
 - Remove any publications without bias score
 - We removed: Vice News, CNN Business, TMZ, and Hyperallergic

**Split Train/Val/Test**
```python
sampled_combined = combined.groupby("bias", group_keys=False).apply(lambda x: x.sample(n=333, random_state=42))
```

- Take a sample of 1000 speeeches (can change later)
- Sampling is stratified across bias score

```python
rom sklearn.model_selection import train_test_split

train_news, temp_news = train_test_split(sampled_combined, 
                                         test_size=0.3, 
                                         random_state=33, 
                                         stratify=sampled_combined["bias"])

val_news, test_news = train_test_split(temp_news, 
                                       test_size=0.5, 
                                       random_state=33, 
                                       stratify=temp_news["bias"])
```
- Create train, validation, and test pandas dataframes
  - Train: 70%
  - Val: 15%
  - Test: 15%
- Splitting is stratified by bias score

```python
y_train = train_news['bias'].tolist()
y_val   = val_news['bias'].tolist()
y_test  = test_news['bias'].tolist()

X_train = X_train.fillna("").astype(str)
X_val = X_val.fillna("").astype(str)
X_test = X_test.fillna("").astype(str)
```
- Assign y to bias scores for set
- Assign x as a string (needed for BERT Tokenizers)

```python
# Checks

print("Total sampled articles:", len(sampled_combined))
print("Train / Val / Test sizes:", len(X_train), len(X_val), len(X_test))
print("Bias distribution in train:", train_news['bias'].value_counts())
```

**Store on Drive**
```python
dataset.save_to_disk("/content/drive/MyDrive/INPUT")
```

## Baseline Model
**Frozen BERT - Logistic Regression**

- Bidirectional: understanding words based on both preceding and following words
- Masked Language Model: masked words during training for prediction
- Next Sentence Prediction: predicts sequential relationship between sentences
- Layers: self-attention & feed-forward networks
- Base: 12 layers & 110M parameters


```python
# Imports

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_tests_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
```

```python
# Get pretrained model

model_name = "bert-base_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
```

```python
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
        )
        with torch.no_grad():
            outputs = model(**tokens)
            batch_embeddings = outputs.pooler_output
        embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.numpy()
```

```python
X_train_list = X_train.tolist()
X_val_list   = X_val.tolist()
X_test_list  = X_test.tolist()

X_train_emb = get_bert_embeddings(X_train_list)
X_val_emb   = get_bert_embeddings(X_val_list)
X_test_emb  = get_bert_embeddings(X_test_list)
```

```python
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_emb, y_train)
```

```python
val_preds = clf.predict(X_val_emb)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))

test_preds = clf.predict(X_test_emb)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("\nClassification Report:\n", classification_report(y_test, test_preds))
```

**Probing Layers**
```python
layers_model = AutoModel.from_pretrained(
    "bert-base-uncased",
    output_hidden_states=True
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

layers_model.eval()

texts = X_train
```

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers_model = layers_model.to(device)

def extract_cls_embeddings(texts, model, tokenizer, batch_size=8):
    layer_outputs = [[] for _ in range(13)]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        for layer_idx, layer in enumerate(outputs.hidden_states):
            cls = layer[:, 0, :].cpu()
            layer_outputs[layer_idx].append(cls)

        del inputs, outputs
        torch.cuda.empty_cache()

    return [torch.cat(layers, dim=0).numpy() for layers in layer_outputs]
```

```python
train_layers = extract_cls_embeddings(X_train, layers_model, tokenizer)
val_layers   = extract_cls_embeddings(X_val, layers_model, tokenizer)
test_layers  = extract_cls_embeddings(X_test, layers_model, tokenizer)
```

**Macro F1 Scores**
```python
layer_f1_scores = []

for layer in range(13):
    X_tr = train_layers[layer]
    X_te = test_layers[layer]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_train)

    preds = clf.predict(X_te)

    f1 = f1_score(y_test, preds, average="macro")
    layer_f1_scores.append(f1)

    print(f"Layer {layer}: Macro F1 = {f1:.4f}")
```

**Plots**
```python
# Stance Across Layers

plt.figure(figsize=(8,5))
plt.plot(range(13), layer_f1_scores, marker='o')
plt.xlabel("Layer")
plt.ylabel("Macro F1")
plt.title("Stance (Bias) Across BERT Layers")
plt.xticks(range(13))
plt.grid(True)
plt.show()
```

```python
# Confusion Matrix

best_layer_idx = 5
X_tr = train_layers[best_layer_idx]
X_te = test_layers[best_layer_idx]

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_tr, y_train)

preds = clf.predict(X_te)

cm = confusion_matrix(y_test, preds, labels=np.unique(y_train))
disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y_train))
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix for Layer {best_layer_idx}")
plt.show()
```

## Core Model Training


## Comparison Models

**TF-IDF**

$$
tf(t,d) = \frac{f_d(t_i)}{maxf_d(w)}
$$

$$
idf(t, D) = log{\frac{D}{1 + DF(t)}}
$$

$$
TF-IDF(t,d) - tf(t,d) \times idf(t, D)
$$

- $f_d(t)$: frequency of term *t* in document *d*
- $maxf_d(w)$: total number of terms in document *d*
- $D$: total number of documents in corpus
- $DF(T)$: number of documents containing term *t*

**Interpretation**
- High TF-IDF indicates important term in that document
- Low TF-IDF indicates common, non-distinctive term

**Basic Implementation**

```python
from sklearn.feature_extraction.text import TfidVectorizer

corpus = data.frame['text]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```


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

**Congressional Speeches:**

```python
import os
from huggingface_hub import login
from google.colab import auth
from getpass import getpass
from datasets import load_dataset

hf_token = getpass("Enter your Hugging Face token: ")

dataset = load_dataset("Eugleo/us-congressional-speeches", use_auth_token = hf_token)
```
- Loading takes about 10 minutes
- Need to access personal HuggingFace Token
- More information on dataset [here](https://huggingface.co/datasets/Eugleo/us-congressional-speeches)

**Congress Metadata:**

```python
from google.colab import files
import pandas as pd

bioid = files.upload()
```
- Download data locally, then choose file after running code
- We use Member Ideology Data from VoteView (can be found [here](https://voteview.com/data))
  - Data Type: Member Ideology
  - Chamber: Both(House and Senate)
  - Congress: 116th (2019-2021)
  - File Format: CSV


## Data Processing

**Transform Metadata**

```python
filename = list(bioid.keys())[0]

members = pd.read_csv(filename)

split_names = members['bioname'].str.split(',', n=1, expand=True)

members['lastname'] = split_names[0].str.strip()
members['firstname'] = split_names[1].str.strip()

members['party'] = members['party_code'].map({100: 1, 200: 0})

# Republican = 0; Democrat = 1
```

**Merge Datasets**
```python
filtered = dataset.filter(lambda x: 2019 <= x['date].year <= 2021)

sample_speech = filtered.select(range(100000)).shuffle(seed = 33).select(range(1000))


```

**Split Train/Val/Test**

```python
split1 = sample_speech.train_test_split(test_size = 0.3, seed = 33)

train = split1["train"]
temp = split1["test"]

split2 = temp.train_test_split(test_size = 0.5, seed = 33)

val = split2["train"]
test = split2["test"]
```

- Filtered dataset for 116th Congress, could use a more efficient filter process (currently 7 min).
- For now, we split 1000 speeches for simplicity. Can update later.

**Store on Drive**
```python
dataset.save_to_disk("/content/drive/MyDrive/my_dataset")
```

## Baseline Model
**BERT**

- Bidirectional: understanding words based on both preceding and following words
- Masked Language Model: masked words during training for prediction
- Next Sentence Prediction: predicts sequential relationship between sentences
- Layers: self-attention & feed-forward networks
- Base: 12 layers & 110M parameters


```python
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_tests_split
import numpy as np
```

```python
model_name = "bert-base_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
```

```python
texts = train['text']
labels = train['party_code']

X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

```python
ef get_bert_embeddings(text_list, batch_size=16):
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
X_train_emb = get_bert_embeddings(X_train)
X_val_emb = get_bert_embeddings(X_val)
X_test_emb = get_bert_embeddings(X_test)
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


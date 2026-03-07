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

token = os.getenv("HF_TOKEN")
login(token=token)

from datasets import load_dataset

dataset = load_dataset("Eugleo/us-congressional-speeches")
```
- Need to access personal HuggingFace Token
- More information on dataset [here](https://huggingface.co/datasets/Eugleo/us-congressional-speeches)


## Data Processing

**Split Train/Val/Test

```python
import os
from huggingface_hub import login

token = os.getenv("HF_TOKEN")
login(token=token)

from datasets import load_dataset
dataset = load_dataset("Eugleo/us-conghressional-speeches")

sample_speech = dataset.shuffle(seed = 33).select(range(1000))

split1 = sample_speech.train_test_split(test_size = 0.3, seed = 33)

train = split1["train"]
temp = split1["test"]

split2 = temp.train_test_split(test_size = 0.5, seed = 33)

val = split2["train"]
test = split2["test"]
```

- For now, we split 1000 speeches for simplicity.
- Can update later.

## Baseline Models

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
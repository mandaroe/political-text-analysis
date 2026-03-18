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
- Loding takes about 10 minutes, save to disk after preprocessing
- Need to access personal HuggingFace Token
- More information on dataset [here](https://huggingface.co/datasets/Eugleo/us-congressional-speeches)

**Congress Metadata:**

We use Member Ideology Data from VoteView (can be found [here](https://voteview.com/data))
- Data Type: Member Ideology
- Chamber: Both(House and Senate)
- Congress: 116th (2019-2021)
- File Format: CSV

```python
from google.colab import files
import pandas as pd

bioid = files.upload()
```
- Download data locally
- Choose file after running code


## Data Processing

**Transform Metadata**

```python
filename = list(bioid.keys())[0]

df_members = pd.read_csv(filename)

split_names = df_members['bioname'].str.split(',', n=1, expand=True)

df_members['lastname'] = split_names[0].str.strip()
df_members['firstname'] = split_names[1].str.strip()

df_members['party'] = df_members['party_code'].map({100: 1, 200: 0})
```

- Republicans: 0
- Democrats: 1


**Split Train/Val/Test**

```python
filtered = dataset.filter(lambda x: 2019 <= x['date].year <= 2021)

sample_speech = filtered.select(range(100000)).shuffle(seed = 33).select(range(1000))

split1 = sample_speech.train_test_split(test_size = 0.3, seed = 33)

train = split1["train"]
temp = split1["test"]

split2 = temp.train_test_split(test_size = 0.5, seed = 33)

val = split2["train"]
test = split2["test"]
```

- Filtered dataset for 116th Congress, could use a more efficient filter process (currently min).
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

**Implementation**

```python

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


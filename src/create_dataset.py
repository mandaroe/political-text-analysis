# ------ 
# Imports
# ------

import os
from datasets import load_dataset
from urllib.parse import urlparse
import kagglehub
import pandas as pd
from google.colab import files

# ------ 
# Data
# ------

path = kagglehub.dataset_download("gandpablo/news-articles-for-political-bias-classification")

news = pd.read_csv(
    f"{path}/bias_clean.csv",
    encoding="latin1",
    on_bad_lines="skip",
    low_memory=True,
    nrows=100000
)

# ------
# Process
# ------

allowed_topics = [
    "elections",
    "economy-and-jobs",
    "defense-and-security",
    "immigration",
    "donald-trump"
]

news = news[news['topic'].isin(allowed_topics)]

# ------ 
# Imports
# ------

import os
import pandas as pd

# ------
# Config
# ------

DATA_PATH = "data/raw/bias_clean.csv"
OUTPUT_PATH = "data/processed/filtered_news.csv"

# ------ 
# Load Data
# ------

news = pd.read_csv(
    DATA_PATH,
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

# ------
# Save
# ------

os.makedirs("data/processed", exist_ok=True)
news.to_csv(OUTPUT_PATH, index=False)

print(f"Saved processed data to {OUTPUT_PATH}")
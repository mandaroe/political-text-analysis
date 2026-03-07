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

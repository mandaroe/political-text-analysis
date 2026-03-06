# Political Text Analysis

## 1) Project Overview

- **Title:** Learning to Read Politics: Machine Learning and the Distinction between Topic and Stance
- **Team:** Sophia Abate & Amanda Rolle
- **Problem statement:**
    Do standard text representation conflate topic similarity and stance?
- **Hypothesis:** 

## 2) Related Work (Short)

- 3-5 bullets on prior papers/blogs/repos you build on.

## 3) Data

- **Dataset(s):**
  - OpenGov uscongress: Congressional Record collection for a given session or date range. Used in this to retrieve speeches and speakers.
  - AllSides Media Bias: Categorical political leaning rating of media news bias on a scale of: left, left-center, center, right-center, right.
  - Congress: Congression collection of actions, bills, nomination, and more more. Used in this study for member identification.
  
- **How to access:**
  - [US-Congressional-Speeches](https://huggingface.co/datasets/Eugleo/us-congressional-speeches)
  
- **License/ethics**:
- **Train/val/test split**:

## 4) Baseline

- **Baseline model**: (name + key params)
    - Topic Frequency-Inverse Document Frequency (TF-IDF) & Logistic Regression
        - ngram_range: 
        - max_features:
        - min_df/max_df:
        - use_idf:
        - smooth_idf:
        - sublinear_tf:
        - penalty:
        - solver:
        - regularization:
    - Stance-DW 
        - dim: embedding size
        - window size:
        - negative sampling:
        - hierarchial softmax:
        - learning rate:
        - epochs:
        - Stance weighting:
- **Baseline metrics**:
    - TF-IDF:
        - Accuracy
        - Confusion Matrix
        - Macro vs Micro F1-score
        - ROC-AUC
    - Stance-DW
        - Same as TF-IDF (for classification metrics)
        - Cosine similarity
        - Silhouette score
        - Adjusted Rand Index (ARI)
        - t-SNE
        - UMAP
- **Why this is a fair baseline**:

## 5) Proposed Method

- **What you change** (architecture, features, losses, etc.):
- **Why it should help**:
- **Ablations** (what you will remove to test impact):

## 6) Experiments

- **Metrics**:
- **Compute budget** (GPU/CPU limits, runtime):
- **Experiment plan** (list your runs and what each tests):

## 7) Reproducibility

- **How to run training**:
- **How to run evaluation**:
- **Where you log results** (files/paths):


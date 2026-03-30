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

## Blog
### Background

Machine learning models can represent meaning in numerous ways, and understanding exactly which way meaning is being represented in different cases is a key challenge in natural language processing. More modern models, including BERT, do a good job across a wide range of tasks, but there still exists internal representations that are  difficult to interpret. 

Meaning is multifaceted in real-world settings. We see this in politcal contexts, for instance. Two statements can discuss the same topic, such as global affairs or immigraion, but express oppossing viewpoints. A well-structered representation should be able to distinguish between the topic being discussed and the actual position that is being taken. 

Though, it is not super clear whether standard text representations are successful at separating these things. Instead, models could easily just group sentences together after determining that they are about the same topic, regardless if they having oppossing viewpoints. This leads us to a crucial question of if modern language models are really able to capture semantic meaning, or if the specific parts of language, such as topic and stance, are conflated. 

This idea is very relevent in the political world. Determining and separating viewpoints based on stance is especially crucial for media analysis, misinformation detection, and other applications. If the embeddings can not properly separate stance of opinion, ideological differences may be harder to determine down the road. 

### Literature Review

Common approaches of text representation, such as bag-of-words and TF-IDF. Methods like these are skilled at detemerming what topic the sentence is introducing, but not very skilled at understanding context or meaning. Due to this, there is an issue with separting sentences with similar words but have different ideas. 

A recent model called BERT creates contextual embeddings, which means the representation of a word depends on the other words. Because of this, models like BERT and similar models do a much better job at tasks like sentiment analysis and text classification, due to more of the meaning in language being evaluated. 

It is great that these newer models work well, but researchers are still looking into how they represent meaning internally. One approach is by looking at the geometry of the embedding space. The goal is to see how close or far different parts of text are from each other. Generally, texts with similar texts tend to reside closer together. 

Work like this tends to focus on general similarity, not as much specific differences like political stance. Many studies focus on predicting things like party affiliation or bias, measuring success through accuracy in regards to politcal text. The part that is more often left out though is how models organize different meaning or why the model is making certain predictions. 

Consequently, it remains unclear whether models can separate topic from stance. This idea is especially important in politcal language as people often talk about the same issue but do not have the same viewpoint.

This project builds on past work by focusing more on how exactly text is organized in embedding space, and testing whether models actually separate topic and stance. 


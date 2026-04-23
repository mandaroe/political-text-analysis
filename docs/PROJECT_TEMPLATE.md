# Political Text Analysis

## 1) Project Overview

- **Title:** Learning to Read Politics: Machine Learning and the Distinction between Topic and Stance
- **Team:** Amanda Rolle & Sophia Abate
- **Problem statement:**
    Do standard text representation conflate topic similarity and stance?
- **Hypothesis:** 

## 2) Related Work

-  Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the ACL: Human Language Technologies, Volume 1 (Long and Short Papers), 4171–4186. https://doi.org/10.18653/v1/N19-1423
- Eger, S., Gao, Y., Peyrard, M., Zhao, W., & Hovy, E. (2020). Proceedings of the First Workshop on Evaluation and Comparison of NLP Systems. Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.eval4nlp-1.0
- Yacouby, R., & Axman, D. (2020). Probabilistic Extension of Precision, Recall, and F1 Score for More Thorough Evaluation of Classification Models. In Proceedings of the First Workshop on Evaluation and Comparison of NLP Systems, 79–91. https://doi.org/10.18653/v1/2020.eval4nlp-1.9

## 3) Data

- **Dataset(s):**
    We use the “News Articles for Political Bias Classification” Corpus, which contains approximately 10,000 articles published between 2014 and 2025. The dataset was created using NewsReader AI, which scraped articles from AllSides, a platform that provides media bias ratings for news outlets based on surveys, multi-partisan analysis, editorial review, and user feedback. AllSides ratings are reported on a 6-point Likert scale and continuously updated as new data becomes available. The dataset was then cleaned by the creator, who mapped the original 6-point ratings to a 5-point scale: left, left-leaning, center, right-leaning, right.

    For our analysis, we focus on the following variables:
        - topic: the subject category of each article, covering wide range of news domains.
        - bias: the political bias label, which serves as the target variable for NLP modeling and classification using large language models.

  
- **How to access:**
  - [News Articles fro Political Bias Classification](https://www.kaggle.com/datasets/gandpablo news-articles-for-political-bias-classification)
  - More information on AllSides [here](https://www.allsides.com/media-bias)
- **License/ethics**:


- **Train/val/test split**:
    We applied a stratified sampling approach to the full dataset. We aimed to sample approximately 1,250 articles, distributing them evenly across topics while maintaining a roughly equal representation of bias categories within each topic. For each topic, we calculated a target number of articles per bias category and sampled accordingly, taking all available articles when a category had fewer than the target. Any remaining quota for a topic was distributed among bias categories with available articles to maximize balance. This procedure ensures that both topic and bias distributions are reasonably even, reducing potential sampling bias and providing a representative subset for NLP modeling. Fig 1. is a cross table of the resulting sample.


## 4) Baseline

- **Baseline model**: (name + key params)
    
- **Baseline metrics**:
        - Accuracy
        - Precison
        - Recall
        - Macro F1-score
        - t-SNE
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


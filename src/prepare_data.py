import pandas as pd
import os
from sklearn.model_selection import train_test_split

# -------
# Sampling
# -------

def stratified_sample_total(df, 
                            topic_col='topic', 
                            bias_col='bias',
                            total_n=1250,
                            random_state=42):
    topics = df[topic_col].unique()
    n_topics = len(topics)
    n_per_topic = total_n // n_topics

    all_biases_in_df = df[bias_col].unique()
    n_all_biases = len(all_biases_in_df)

    sampled_list = []

    for topic, group in df.groupby(topic_col):
        total_in_topic = len(group)

        if total_in_topic <= n_per_topic:
            sampled_list.append(group)
        else:
            target_n_per_bias = n_per_topic // n_all_biases

            n_samples_per_bias_dict = {}
            for bias_cat in all_biases_in_df:
                available_count = len(group[group[bias_col] == bias_cat])
                n_samples_per_bias_dict[bias_cat] = min(available_count, target_n_per_bias)
            
            current_total = sum(n_samples_per_bias_dict.values())
            remaining = n_per_topic - current_total

            eligible_biases = [
                b for b in all_biases_in_df
                if len(group[group[bias_col] == b]) > n_samples_per_bias_dict[b]
            ]

            idx = 0
            while remaining > 0 and eligible_biases:
                b = eligible_biases[idx % len(eligible_biases)]

                if len(group[group[bias_col] == b]) > n_samples_per_bias_dict[b]:
                    n_samples_per_bias_dict[b] += 1
                    remaining -= 1
                else:
                    eligible_biases.remove(b)

                idx += 1 if eligible_biases else 0

            sampled_topic = pd.concat([
                group[group[bias_col] == b].sample(
                    n=n_samples_per_bias_dict[b],
                    random_state=random_state
                )
                for b in n_samples_per_bias_dict
                if n_samples_per_bias_dict[b] > 0
            ])

            sampled_list.append(sampled_topic)

    return pd.concat(sampled_list).reset_index(drop=True)

# ---------------
# Main Pipeline
# ---------------

def main():
    news = pd.read_csv("data/processed/clean_news.csv")

    sampled_news = stratified_sample_total(news, total_n=1250)

    sampled_news["strata"] = (
        sampled_news["topic"].astype(str) + "_" + sampled_news["bias"].astype(str)
    )

    train, temp = train_test_split(
        sampled_news,
        test_size=0.3,
        stratify=sampled_news["strata"],
        random_state=42
    )

    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp["strata"],
        random_state=42
    )

    train = train.drop(columns=["strata"])
    val = val.drop(columns=["strata"])
    test = test.drop(columns=["strata"])

    os.makedirs("data/processed", exist_ok=True)

    sampled_news.drop(columns=["strata"]).to_csv(
        "data/processed/sampled_news.csv", index=False
    )
    
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)


if __name__ == "__main__":
    main()


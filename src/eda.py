import pandas as pd

df = pd.read_csv("data/train.csv")

df["article_len"] = df["article"].astype(str).apply(
    lambda x: len(x.split())
)

df["summary_len"] = df["highlights"].astype(str).apply(
    lambda x: len(x.split())
)

print("\nArticle Length Statistics:")
print(df["article_len"].describe())

print("\nSummary Length Statistics:")
print(df["summary_len"].describe())
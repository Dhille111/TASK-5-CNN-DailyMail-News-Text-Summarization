import pandas as pd

df = pd.read_csv("data/train.csv")

df["article_len"] = df["article"].astype(str).apply(
    lambda x: len(x.split())
)

df["summary_len"] = df["highlights"].astype(str).apply(
    lambda x: len(x.split())
)

df["compression_ratio"] = (
    df["summary_len"] /
    df["article_len"]
)

print(df["compression_ratio"].describe())
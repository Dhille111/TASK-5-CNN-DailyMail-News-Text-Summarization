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

bad_rows = df[df["compression_ratio"] > 1]

print("Bad Records:")
print(len(bad_rows))

print("\nTop 5:")
print(
    bad_rows[
        ["article_len",
         "summary_len",
         "compression_ratio"]
    ].head()
)
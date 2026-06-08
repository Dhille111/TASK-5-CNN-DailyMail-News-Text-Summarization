import pandas as pd
import re

df = pd.read_csv("data/train.csv")

def clean_text(text):
    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["article"] = df["article"].apply(clean_text)
df["highlights"] = df["highlights"].apply(clean_text)

# Lengths
df["article_len"] = df["article"].apply(
    lambda x: len(x.split())
)

df["summary_len"] = df["highlights"].apply(
    lambda x: len(x.split())
)

# Remove tiny articles
df = df[df["article_len"] >= 50]

# Remove tiny summaries
df = df[df["summary_len"] >= 5]

# Remove abnormal summaries
df = df[df["summary_len"] < df["article_len"]]

print("Final Shape:")
print(df.shape)

df.to_csv(
    "data/clean_train.csv",
    index=False
)

print("Clean dataset saved.")
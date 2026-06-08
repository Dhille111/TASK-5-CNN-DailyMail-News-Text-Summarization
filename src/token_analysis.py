from transformers import BartTokenizer
import pandas as pd

tokenizer = BartTokenizer.from_pretrained(
    "facebook/bart-base"
)

df = pd.read_csv(
    "data/clean_train.csv"
)

sample = df.sample(
    5000,
    random_state=42
)

token_lengths = []

for article in sample["article"]:

    tokens = tokenizer.encode(
        str(article),
        truncation=False
    )

    token_lengths.append(
        len(tokens)
    )

print("\nToken Statistics:")
print(pd.Series(token_lengths).describe())

print(
    "\nArticles >1024 tokens:"
)

print(
    sum(
        t > 1024
        for t in token_lengths
    )
)
from transformers import BartTokenizer
import pandas as pd

tokenizer = BartTokenizer.from_pretrained(
    "facebook/bart-base"
)

df = pd.read_csv(
    "data/clean_train.csv"
)

sample_df = df.sample(
    10000,
    random_state=42
)

def preprocess(examples):

    inputs = tokenizer(
        examples,
        max_length=1024,
        truncation=True,
        padding="max_length"
    )

    return inputs

tokens = preprocess(
    sample_df["article"].tolist()
)

print("Tokenization Successful")

print(
    len(tokens["input_ids"])
)
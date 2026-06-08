import pandas as pd

df = pd.read_csv("data/clean_train.csv")

subset = df.sample(
    n=2000,
    random_state=42
)

subset.to_csv(
    "data/train_subset.csv",
    index=False
)

print(subset.shape)
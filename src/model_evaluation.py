import pandas as pd
import evaluate

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration
)

MODEL_PATH = "models/final_model"

tokenizer = BartTokenizer.from_pretrained(
    MODEL_PATH
)

model = BartForConditionalGeneration.from_pretrained(
    MODEL_PATH
)

df = pd.read_csv(
    "data/test.csv"
)

df = df.head(50)

predictions = []
references = []

for _, row in df.iterrows():

    article = str(row["article"])

    reference = str(row["highlights"])

    inputs = tokenizer(
        article,
        max_length=256,
        truncation=True,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=64,
        min_length=20,
        num_beams=4
    )

    prediction = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    predictions.append(prediction)
    references.append(reference)

rouge = evaluate.load("rouge")

scores = rouge.compute(
    predictions=predictions,
    references=references
)

print("\nROUGE SCORES\n")
print(scores)
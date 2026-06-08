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

text = """
Artificial Intelligence is transforming industries
by automating repetitive tasks, improving
decision making and enabling new innovations.
Companies across healthcare, finance and
education are increasingly adopting AI systems.
"""

inputs = tokenizer(
    text,
    max_length=256,
    truncation=True,
    return_tensors="pt"
)

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=64,
    min_length=20,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("\nSUMMARY:\n")
print(summary)
import pandas as pd
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

MODEL_NAME = "facebook/bart-base"

print("Loading Dataset...")

df = pd.read_csv("data/train_subset.csv")

dataset = Dataset.from_pandas(
    df[["article", "highlights"]]
)

tokenizer = BartTokenizer.from_pretrained(
    MODEL_NAME
)

def tokenize_function(examples):

    model_inputs = tokenizer(
        examples["article"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        examples["highlights"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

model = BartForConditionalGeneration.from_pretrained(
    MODEL_NAME
)

training_args = Seq2SeqTrainingArguments(
    output_dir="models/bart_model",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=2,

    num_train_epochs=1,

    logging_steps=10,

    save_strategy="epoch",

    predict_with_generate=True,

    fp16=False,

    report_to="none",

    remove_unused_columns=False
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("Training Started...")

trainer.train()

trainer.save_model(
    "models/final_model"
)

tokenizer.save_pretrained(
    "models/final_model"
)

print("Training Complete!")
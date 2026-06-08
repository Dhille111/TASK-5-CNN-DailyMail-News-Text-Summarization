import evaluate

rouge = evaluate.load("rouge")

predictions = [
    "AI is transforming industries through automation."
]

references = [
    "Artificial intelligence is changing industries through automation."
]

results = rouge.compute(
    predictions=predictions,
    references=references
)

print(results)
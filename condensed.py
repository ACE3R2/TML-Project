!pip install datasets
!pip install transformers
!pip install evaluate
!pip install accelerate -U


from datasets import load_dataset
import pandas as pd
import numpy as np
import os
from google.colab import drive
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer, pipeline
import evaluate


# Mount Google Drive
drive.mount('/content/drive')
# !mkdir -p "/content/drive/My Drive/LLM_GPT2"


dataset = load_dataset("mteb/tweet_sentiment_extraction")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
   return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# From scratch
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
# From previous trained
# model = GPT2ForSequenceClassification.from_pretrained("/content/drive/My Drive/LLM_GPT2/initial_attempt_model")
# model = GPT2ForSequenceClassification.from_pretrained("/content/drive/My Drive/LLM_GPT2/initial_attempt/checkpoint-1500")


metric = evaluate.load("accuracy")

# Loss given the prediction and labels
def compute_metrics(eval_pred):
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   return metric.compute(predictions=predictions, references=labels)

# Arguments for training
training_args = TrainingArguments(
   # Checkpoint save location
   output_dir="/content/drive/My Drive/LLM_GPT2/initial_attempt",
   # Rate of running validation
   evaluation_strategy="steps",
   eval_steps=500,
   per_device_train_batch_size=1,
   per_device_eval_batch_size=1,
   # Number of steps of accumulating gradients before running backpropagation and updating location
   gradient_accumulation_steps=4,
   num_train_epochs=4
)

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
   compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("/content/drive/My Drive/LLM_GPT2/initial_attempt_model")
# Get validation loss
trainer.evaluate()

# print(pd.DataFrame(dataset["train"][:20]))
# print(dataset["train"][0]["text"])

pipe = pipeline("text-classification",model=model2,tokenizer=tokenizer)

for i in range(100):
  print("Label is ",end="")
  print(dataset["train"][i]["label"],end="; ")
  print("Prediction is ",end="")
  print(pipe(dataset["train"][i]["text"]))

import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Create a folder in the root directory
# !mkdir -p "/content/drive/My Drive/LLM_GPT2"

# --------------------------------------------------

!pip install datasets
!pip install transformers
!pip install evaluate
!pip install accelerate -U

# --------------------------------------------------

from datasets import load_dataset
import pandas as pd
import numpy as np

dataset = load_dataset("mteb/tweet_sentiment_extraction")
df = pd.DataFrame(dataset['train'])

# --------------------------------------------------

from transformers import GPT2Tokenizer

# Loading the dataset to train our model
dataset = load_dataset("mteb/tweet_sentiment_extraction")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
   return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# --------------------------------------------------

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# --------------------------------------------------

from transformers import GPT2ForSequenceClassification

model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)

# --------------------------------------------------

import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   return metric.compute(predictions=predictions, references=labels)

# --------------------------------------------------

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
   output_dir="/content/drive/My Drive/LLM_GPT2/initial_attempt",
   evaluation_strategy="epoch",
   eval_steps=500,
   per_device_train_batch_size=1,  # Reduce batch size here
   per_device_eval_batch_size=1,    # Optionally, reduce for evaluation as well
   gradient_accumulation_steps=4,
   num_train_epochs=0.5
   )


trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
   compute_metrics=compute_metrics,

)

# --------------------------------------------------

trainer2.train()

trainer2.save_model("/content/drive/My Drive/LLM_GPT2/second_attempt_model")

# --------------------------------------------------

import evaluate

trainer.evaluate()

# --------------------------------------------------

print(model.__dict__)

# --------------------------------------------------

from transformers import GPT2ForSequenceClassification, TrainingArguments, Trainer

model2 = GPT2ForSequenceClassification.from_pretrained("/content/drive/My Drive/LLM_GPT2/second_attempt_model")

training_args = TrainingArguments(
   output_dir="/content/drive/My Drive/LLM_GPT2/initial_attempt",
   evaluation_strategy="epoch",
   eval_steps=500,
   per_device_train_batch_size=1,  # Reduce batch size here
   per_device_eval_batch_size=1,    # Optionally, reduce for evaluation as well
   gradient_accumulation_steps=4,
   num_train_epochs=12
)

trainer2 = Trainer(
   model=model2,
   args=training_args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
   compute_metrics=compute_metrics,

)

# --------------------------------------------------

import evaluate

trainer2.evaluate()

# --------------------------------------------------

print(pd.DataFrame(dataset["train"][:20]))

# print(dataset["train"][0]["text"])

# --------------------------------------------------

from transformers import pipeline

pipe = pipeline("text-classification",model=model2,tokenizer=tokenizer)

for i in range(100):
    print(dataset["train"][i]["label"],end="")
    print(pipe(dataset["train"][i]["text"]))

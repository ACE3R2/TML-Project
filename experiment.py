!pip install datasets
!pip install transformers
!pip install evaluate
!pip install accelerate -U
!pip install -U sentence_transformers

# -------------------------------------------------------

from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import os
from google.colab import drive
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2LMHeadModel, TrainingArguments, Trainer, pipeline, AutoTokenizer, AutoModelForCausalLM
import evaluate
import torch
from sentence_transformers import CrossEncoder

from transformers.utils import logging
logging.set_verbosity(40)

# -------------------------------------------------------

# Mount Google Drive
drive.mount('/content/drive')

# Create a folder in the root directory
!mkdir -p "/content/drive/My Drive/LLM_Experiment"

# -------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------------

def tokenize_function(examples):
   return tokenizer(examples["text"], padding="max_length", truncation=True)

#tokenized_datasets = dataset.map(tokenize_function, batched=True)

# -------------------------------------------------------

model_generate = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
model_evaluate = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v1')
model_predict = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=50)

# -------------------------------------------------------

def chatbot_response(query):
  question = tokenizer.encode(query + tokenizer.eos_token, return_tensors='pt')

  outputs = model_generate.generate(question, max_length=30, pad_token_id=tokenizer.eos_token_id)
  full_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

  #print(full_answer)

  return full_answer[len(query):]

# -------------------------------------------------------

def compare(sentences1, sentences2):
  scores = model_evaluate.predict(list(zip(sentences1,sentences2)),show_progress_bar=False)
  return scores

# -------------------------------------------------------

def top_similarity_score(row):
    #print(model_generate.transformer.wte.weight)
    question = row["question"]
    generate_response = chatbot_response(question)
    sscores = compare(row["answers"], [generate_response]*len(row["answers"]))
    return max(sscores)


def add_similarity_score(tensr, drop, row, similarity_scores):
    torig = tensr.clone()
    tnew = drop(tensr)
    similarity_scores.append(top_similarity_score(row))
    tensr.data = torig.data

# -------------------------------------------------------

def create_dataset(split):
  drop_val = 0.1
  drop = torch.nn.Dropout(drop_val, inplace=True)

  ds = load_dataset("web_questions")
  dstrain = ds[split]

  data_rows = []
  for row in dstrain:
    print(row)
    base_similarity = top_similarity_score(row)
    similarity_scores = []

    with torch.no_grad():
      add_similarity_score(model_generate.transformer.wte.weight, drop, row, similarity_scores)
      for i in range(12):
        add_similarity_score(model_generate.transformer.h[i].attn.c_attn.weight, drop, row, similarity_scores)
        add_similarity_score(model_generate.transformer.h[i].attn.c_proj.weight, drop, row, similarity_scores)
        add_similarity_score(model_generate.transformer.h[i].mlp.c_fc.weight, drop, row, similarity_scores)
        add_similarity_score(model_generate.transformer.h[i].mlp.c_proj.weight, drop, row, similarity_scores)
      add_similarity_score(model_generate.lm_head.weight, drop, row, similarity_scores)


    #print(similarity_scores)
    differences = list(map(lambda x: 3*abs(base_similarity - x), similarity_scores))

    tr_differences = torch.tensor(differences)
    #print(tr_differences)

    sm = torch.nn.Softmax(dim=-1)
    smax = sm(tr_differences)

    #print(smax)

    data_rows.append({"text": row["question"], "label": smax})

  return data_rows

# -------------------------------------------------------

dataset_list = create_dataset("train")
#print(dataset_list)
dataset_hf = Dataset.from_list(dataset_list)
small_train_dataset = dataset_hf.map(tokenize_function)

# -------------------------------------------------------

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   return metric.compute(predictions=predictions, references=labels)

# -------------------------------------------------------

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)

    #print(labels)
    #print(outputs.logits)

    kl_loss = torch.nn.KLDivLoss()
    loss = kl_loss(outputs.logits, labels)

    return (loss, outputs) if return_outputs else loss

# -------------------------------------------------------

training_args = TrainingArguments(
   output_dir="/temp",
   evaluation_strategy="steps",
   eval_steps=500,
   per_device_train_batch_size=1,  # Reduce batch size here
   per_device_eval_batch_size=1,    # Optionally, reduce for evaluation as well
   gradient_accumulation_steps=4,
   num_train_epochs=0.5
   )


trainer = CustomTrainer(
   model=model_predict,
   args=training_args,
   train_dataset=small_train_dataset,
   #eval_dataset=small_eval_dataset,
   compute_metrics=compute_metrics,

)

# -------------------------------------------------------

trainer.train()
trainer.save_model("/content/drive/My Drive/LLM_Experiment")

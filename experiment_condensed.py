#!pip install datasets
#!pip install transformers
#!pip install evaluate
#!pip install accelerate -U
#!pip install -U sentence_transformers

from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import os
from google.colab import drive
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2LMHeadModel, TrainingArguments, Trainer, pipeline, AutoTokenizer, AutoModelForCausalLM
import evaluate
import torch
from sentence_transformers import CrossEncoder
import matplotlib.pyplot as plt

from transformers.utils import logging
logging.set_verbosity(40)


class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)

    #print(labels)
    #print(outputs.logits)

    kl_loss = torch.nn.KLDivLoss()
    loss = kl_loss(outputs.logits, labels)

    return (loss, outputs) if return_outputs else loss

def tokenize_function(examples):
   return tokenizer(examples["text"], padding="max_length", truncation=True)


def chatbot_response(query):
  question = tokenizer.encode(query + tokenizer.eos_token, return_tensors='pt')

  outputs = model_generate.generate(question, max_length=30, pad_token_id=tokenizer.eos_token_id)
  full_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

  #print(full_answer)

  return full_answer[len(query):]

# Mount Google Drive
drive.mount('/content/drive')

# Create a folder in the root directory
#!mkdir -p "/content/drive/My Drive/LLM_Experiment"

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.pad_token = tokenizer.eos_token


def compare(sentences1, sentences2):
  scores = model_evaluate.predict(list(zip(sentences1,sentences2)),show_progress_bar=False)
  return scores


def top_similarity_score(row):
    question = row["question"]
    generate_response = chatbot_response(question)
    sscores = compare(row["answers"], [generate_response]*len(row["answers"]))
    return max(sscores)


def add_similarity_score(tensr, drop, row, similarity_scores):
    torig = tensr.clone()
    tnew = drop(tensr)
    similarity_scores.append(top_similarity_score(row))
    tensr.data = torig.data


ds = load_dataset("web_questions")

def create_dataset(split, drop_val=0.1, num_run=-1):
  drop = torch.nn.Dropout(drop_val, inplace=True)

  #ds = load_dataset("nq_open")

  dstrain = ds[split]

  if num_run != -1:
    dstrain = dstrain.shuffle().select(range(num_run))

  all_similarities = []
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

    differences = list(map(lambda x: 3*abs(base_similarity - x), similarity_scores))

    tr_differences = torch.tensor(differences)

    sm = torch.nn.Softmax(dim=-1)
    smax = sm(tr_differences)


    #all_similarities.append(similarity_scores)
    similarity_scores.append(base_similarity)
    all_similarities.append(similarity_scores)
    data_rows.append({"text": row["question"], "label": smax})

  return data_rows, all_similarities


def compute_metrics(eval_pred):
   logits, labels = eval_pred
   sum = 0
   kl_loss = torch.nn.KLDivLoss()
   for i in range(len(logits)):
      sum += kl_loss(torch.tensor(logits), torch.tensor(labels))
   return {"eval_loss": sum/len(logits)}






model_generate = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
model_evaluate = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v1')
model_predict = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=50)

all_similar = []
for drop_whole in range(1,101,2):
  dropp = drop_whole/100
  dr, similar = create_dataset("train",dropp,10)
  similar_row = []
  for x in range(len(similar[0])):
    sum = 0
    for row in range(len(similar)):
      sum += similar[row][x]
    similar_row.append(sum/len(similar))
  print(similar_row)
  all_similar.append(similar_row)

df = pd.DataFrame(all_similar)
print(df)


dataset_train_base, similar1 = create_dataset("train")
dataset_test_base, similar2 = create_dataset("test")
full_train_data = Dataset.from_list(dataset_train_base)
full_test_data = Dataset.from_list(dataset_test_base)
tokenized_train_data = full_train_data.map(tokenize_function)
tokenized_test_data = full_test_data.map(tokenize_function)

training_args = TrainingArguments(
   output_dir="/temp",
   evaluation_strategy="steps",
   eval_steps=500,
   per_device_train_batch_size=1,  # Reduce batch size here
   per_device_eval_batch_size=1,    # Optionally, reduce for evaluation as well
   gradient_accumulation_steps=4,
   num_train_epochs=0.5)

trainer = CustomTrainer(
   model=model_predict,
   args=training_args,
   train_dataset=small_train_dataset,
   #eval_dataset=small_eval_dataset,
   compute_metrics=compute_metrics,)

trainer.train()
trainer.save_model("/content/drive/My Drive/LLM_Experiment")


arr = all_similar
arr_df = pd.DataFrame(arr)
arr_df.index = [str(x/100) for x in range(1,51,2)]

col_temp = ["Unmodified", "Word Embedding"]
col_temp.extend(sum([["Transformer " + str(i) + " - Attention", "Transformer " + str(i) + " - Attention Mixing", "Transformer " + str(i) + " - Feed Forward Layer 1", "Transformer " + str(i) + " - Feed Forward Layer 2"] for i in range(1,13)],[]))
col_temp.append("Final Linear Layer")

arr_df.columns = col_temp

print(arr_df.mean(axis=0))


NUM_COLORS = 15

cm = plt.get_cmap('gist_rainbow')
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
#for i in range(NUM_COLORS):
#  ax.plot(arr_df.iloc[:, i], label=arr_df.columns[i])
ax.plot(arr_df.iloc[:, 0], label=arr_df.columns[0])
ax.plot(arr_df.iloc[:, 1], label=arr_df.columns[1])
for i in range(1,13):
  ax.plot(arr_df.iloc[:, 4*i-2:4*i+2].mean(axis=1), label=("Transformer" + str(i)))
#ax.plot(arr_df.iloc[:, 2:50:4].mean(axis=1), label="Average of Attention")
#ax.plot(arr_df.iloc[:, 3:50:4].mean(axis=1), label="Average of Attention Mixing")
#ax.plot(arr_df.iloc[:, 4:50:4].mean(axis=1), label="Average of Feed Forward Layer 1")
#ax.plot(arr_df.iloc[:, 5:50:4].mean(axis=1), label="Average of Feed Forward Layer 2")
ax.plot(arr_df.iloc[:, 50], label=arr_df.columns[50])

ax.legend(bbox_to_anchor=(1.1, 1.05))
plt.title("Average Similarity")
plt.show()

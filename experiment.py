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

#ds = load_dataset("web_questions")
ds = load_dataset("nq_open")

def create_dataset(split, drop_val=0.3, num_run=-1, static_start=-1, static_end=-1):
  drop = torch.nn.Dropout(drop_val, inplace=True)

  #ds = load_dataset("nq_open")

  dstrain = ds[split]

  if num_run != -1:
    dstrain = dstrain.shuffle().select(range(num_run))
  elif static_start != -1 and static_end != -1:
    dstrain = dstrain.select(range(static_start, static_end))

  all_similarities = []
  data_rows = []
  for row in dstrain:
    print(row)
    base_similarity = top_similarity_score(row)
    if num_run == -1 and base_similarity < 0.5:
      continue
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

    all_similarities.append(similarity_scores)
    #similarity_scores.append(base_similarity)
    #all_similarities.append(similarity_scores)
    data_rows.append({"text": row["question"], "label": smax})

  return data_rows, all_similarities
# -------------------------------------------------------

all_similar = []
for drop_whole in range(1,101,2):
#for drop_whole in range(27,101,2):
  dropp = drop_whole/100
  dr, similar = create_dataset("train",dropp,10)
  print(dropp)
  #print(similar)
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

# -------------------------------------------------------

for i in range(0, 15000, 100):
  print(i)
  old_train_data = load_dataset('json', data_files='/content/drive/My Drive/LLM_Experiment/train_data.json')
  #print(old_train_data)
  dataset_train_base, similar1 = create_dataset("train", static_start=i, static_end=i+100)
  print(dataset_train_base)
  new_train_data = Dataset.from_list(dataset_train_base)
  full_train_data = concatenate_datasets([old_train_data['train'], new_train_data])
  full_train_data.to_json('/content/drive/My Drive/LLM_Experiment/train_data.json')
  #new_train_data.to_json('/content/drive/My Drive/LLM_Experiment/train_data.json')

# -------------------------------------------------------

dataset_test_base, similar2 = create_dataset("test")
full_test_data = Dataset.from_list(dataset_test_base)
full_test_data.to_json('/content/drive/My Drive/LLM_Experiment/test_data.json')

# -------------------------------------------------------

all_train_data = load_dataset('json', data_files='/content/drive/My Drive/LLM_Experiment/train_data.json')["train"]
all_test_data = load_dataset('json', data_files='/content/drive/My Drive/LLM_Experiment/test_data.json')["train"]

# -------------------------------------------------------

print(model_generate)
#print(model_generate.transformer.h[0].attn.c_attn.weight)
#print(model_generate.transformer.h[0].attn.c_attn.weight[0])
#print(model_generate.transformer.h[0].attn.c_attn.weight.shape)
#with torch.no_grad():
#  model_generate.transformer.h[0].attn.c_attn.weight[0][0] = 0.5
#print(model_generate.transformer.h[0].attn.c_attn.weight[0])
#print(model_generate.transformer.h[0].attn.c_attn.weight.shape)

# print(model_generate.transformer.wte.weight.shape)
# print(model_generate.transformer.wte.weight)
# print(model_generate.transformer.h[0].attn.c_attn.weight.shape)
# print(model_generate.transformer.h[0].attn.c_attn.weight)
# print(model_generate.transformer.h[0].attn.c_proj.weight.shape)
# print(model_generate.transformer.h[0].attn.c_proj.weight)
# print(model_generate.transformer.h[0].mlp.c_fc.weight.shape)
# print(model_generate.transformer.h[0].mlp.c_fc.weight)
# print(model_generate.transformer.h[0].mlp.c_proj.weight.shape)
# print(model_generate.transformer.h[0].mlp.c_proj.weight)
# print(model_generate.transformer.ln_f.weight.shape)
# print(model_generate.transformer.ln_f.weight)
# print(model_generate.lm_head.weight.shape)
# print(model_generate.lm_head.weight)

# -------------------------------------------------------

tokenized_train_data = full_train_data.map(tokenize_function)
tokenized_test_data = full_test_data.map(tokenize_function)

# -------------------------------------------------------

print(small_train_dataset)
print(small_train_dataset[0])

# -------------------------------------------------------

def compute_metrics(eval_pred):
   logits, labels = eval_pred
   sum = 0
   kl_loss = torch.nn.KLDivLoss()
   #print(logits)
   #print(labels)
   for i in range(len(logits)):
      sum += kl_loss(torch.tensor(logits), torch.tensor(labels))
   return {"eval_custom": sum/len(logits)}

def abs_diff(eval_pred):
  logits, labels = eval_pred
  #print(logits)
  #print(len(logits))
  sum = 0
  smax = torch.nn.Softmax(dim=-1)
  for row in range(len(logits)):
    sm = smax(torch.tensor(logits[row])).tolist()
    print([[sm[col], labels[row][col]] for col in range(len(logits[row]))])
    sum_row = 0
    for col in range(len(logits[row])):
      sum_row += abs(sm[col] - labels[row][col])/2
    #print(sum_row)
    sum += sum_row

  return {"eval_custom": sum/len(logits)}
# -------------------------------------------------------

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)

    sum = 0
    smax = torch.nn.Softmax(dim=-1)
    sm = smax(outputs.logits)
    for i in range(sm.size(dim=1)):
      sum += abs(sm[0][i] - labels[0][i])/2

    #print([[sm[0][col], labels[0][col]] for col in range(sm.size(dim=1))])
    #print([abs(sm[0][col] - labels[0][col]) for col in range(sm.size(dim=1))])
    #print(sm)
    #print(labels)
    #print(labels[0])
    #print(sm.size())
    #print(sm.size(dim=1))
    #print(sm[0])
    #print(sm[0][0])

    #print(sum)

    loss = sum
    
    #kl_loss = torch.nn.KLDivLoss()
    #loss = kl_loss(outputs.logits, labels)

    return (loss, outputs) if return_outputs else loss

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
   output_dir="temp",
   per_device_train_batch_size=1,  # Reduce batch size here
   per_device_eval_batch_size=1,    # Optionally, reduce for evaluation as well
   gradient_accumulation_steps=1,
   num_train_epochs=6
)


trainer = CustomTrainer(
   model=saved_model_predict,
   args=training_args,
   train_dataset=tokenized_train_data,
   eval_dataset=tokenized_test_data,
   compute_metrics=abs_diff
)

# -------------------------------------------------------

trainer.train()

trainer.save_model("/content/drive/My Drive/LLM_Experiment/model_epoch6")

# -------------------------------------------------------

saved_model_predict = GPT2ForSequenceClassification.from_pretrained("/content/drive/My Drive/LLM_Experiment/model_epoch6")

# -------------------------------------------------------

saved_trainer = CustomTrainer(
   model=saved_model_predict,
   args=training_args,
   train_dataset=tokenized_train_data,
   eval_dataset=tokenized_test_data,
   compute_metrics=abs_diff
)

# -------------------------------------------------------

saved_trainer.evaluate()

# -------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------

arr_df = df

arr_df.index = [str(x/100) for x in range(1,51,2)]

col_temp = ["Unmodified", "Word Embedding"]

col_temp.extend(sum([["Transformer " + str(i) + " - Attention", "Transformer " + str(i) + " - Attention Mixing", "Transformer " + str(i) + " - Feed Forward Layer 1", "Transformer " + str(i) + " - Feed Forward Layer 2"] for i in range(1,13)],[]))
col_temp.append("Final Linear Layer")

print(col_temp)

arr_df.columns = col_temp

#vals_df = arr_df.T


print(arr_df)
#print(vals_df)

# -------------------------------------------------------

print(arr_df.mean(axis=0))

# -------------------------------------------------------

NUM_COLORS = 15

#print(arr_df.mean(axis=0))


#print(arr_df.iloc[:, 0])

cm = plt.get_cmap('gist_rainbow')
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
#for i in range(NUM_COLORS):
#  ax.plot(arr_df.iloc[:, i], label=arr_df.columns[i])
ax.plot(arr_df.iloc[:, 0], label=arr_df.columns[0])
ax.plot(arr_df.iloc[:, 1], label=arr_df.columns[1])
for i in range(1,13):
  print(arr_df.iloc[:, 4*i-2:4*i+2].mean(axis=1))
  ax.plot(arr_df.iloc[:, 4*i-2:4*i+2].mean(axis=1), label=("Transformer" + str(i)))
#ax.plot(arr_df.iloc[:, 2:50:4].mean(axis=1), label="Average of Attention")
#ax.plot(arr_df.iloc[:, 3:50:4].mean(axis=1), label="Average of Attention Mixing")
#ax.plot(arr_df.iloc[:, 4:50:4].mean(axis=1), label="Average of Feed Forward Layer 1")
#ax.plot(arr_df.iloc[:, 5:50:4].mean(axis=1), label="Average of Feed Forward Layer 2")
ax.plot(arr_df.iloc[:, 50], label=arr_df.columns[50])


#arr_df.plot()
ax.legend(bbox_to_anchor=(1.1, 1.05))
#ax.legend()
plt.title("Average Similarity")
plt.show()

# -------------------------------------------------------

#arr = printed outputs of trainer.evaluate(), concatenated into a single list

import numpy

ct = 0
ct2 = 0

for test_rd in arr:
    guesses = [i[0] for i in test_rd]
    labels = [i[1] for i in test_rd]
    
    guess_max = guesses.index(max(guesses))
    label_max = labels.index(max(labels))
    
    if guess_max == label_max:
        ct += 1
    
    guess_top = sorted(range(len(guesses)), key=lambda i: guesses[i], reverse=True)[:5]
    label_top = sorted(range(len(labels)), key=lambda i: labels[i], reverse=True)[:5]
    
    for ss in guess_top:
        if ss in label_top:
            ct2 += 1
print(ct)
print(ct2)


guessed_damages = [0]*50
labeled_damages = [0]*50
for ar in arr:
    for a in range(len(ar)):
        guessed_damages[a] += ar[a][0]/len(arr)
        labeled_damages[a] += ar[a][1]/len(arr)
        
        
        
print(labeled_damages)
print(guessed_damages)

print(numpy.argsort(guessed_damages))
print(numpy.argsort(labeled_damages))

print(sorted(guessed_damages))
print(sorted(labeled_damages))

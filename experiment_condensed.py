Packages to install on the terminal, or commands inserted directly into colab
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


model_generate = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
model_evaluate = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v1')
model_predict = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=50)

def tokenize_function(examples):
   return tokenizer(examples["text"], padding="max_length", truncation=True)

# given a question, the answer given by model_generate
def chatbot_response(query):
  question = tokenizer.encode(query + tokenizer.eos_token, return_tensors='pt')
  
  # max 30 token response
  outputs = model_generate.generate(question, max_length=30, pad_token_id=tokenizer.eos_token_id)
  # decode back into English
  full_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

  #print(full_answer)

  # full answer will also include the original question with it, so remove that part
  return full_answer[len(query):]

# Mount Google Drive
# save progress so it does not get lost when session restarts (can be commented out if not on colab)
drive.mount('/content/drive')

# Create a folder in the root directory
#!mkdir -p "/content/drive/My Drive/LLM_Experiment"

# create a tokenizer to encode sentences
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.pad_token = tokenizer.eos_token

# return the sentence similarity of the two sentences, using model_evaluate
def compare(sentences1, sentences2):
  scores = model_evaluate.predict(list(zip(sentences1,sentences2)),show_progress_bar=False)
  return scores

# of all the possible correct answers for a question, choose the one that is the most similar to the answer given by model_generate
def top_similarity_score(row):
    question = row["question"]
    generate_response = chatbot_response(question)
    sscores = compare(row["answers"], [generate_response]*len(row["answers"]))
    return max(sscores)

# given a list of similarity scores and a tensor, apply the dropout to the tensor, find the new similarity score, 
def add_similarity_score(tensr, drop, row, similarity_scores):
    torig = tensr.clone()
    tnew = drop(tensr)
    similarity_scores.append(top_similarity_score(row))
    tensr.data = torig.data

# the data can be used from either web questions or natural questions (open)
#ds = load_dataset("web_questions")
ds = load_dataset("nq_open")

# process to create the augmented dataset
# split = test/train, drop_val is the percentage of parameters to zero
# num_run is the number of samples run if doing a random sample, static_start and static_end are the beginning and end of a fixed sample
def create_dataset(split, drop_val=0.3, num_run=-1, static_start=-1, static_end=-1):
  # create the dropout and split
  drop = torch.nn.Dropout(drop_val, inplace=True)
  dstrain = ds[split]

  # choose the amount to run (the full split or only a portion)
  if num_run != -1:
    dstrain = dstrain.shuffle().select(range(num_run))
  elif static_start != -1 and static_end != -1:
    dstrain = dstrain.select(range(static_start, static_end))

  # create the similarities list (how similar each sentence is to the correct answer when the ith layer is modified)
  all_similarities = []
  data_rows = []
  for row in dstrain:
    print(row)
    # get the similarity score for the unmodified model
    base_similarity = top_similarity_score(row)
    # when generating all, skip those under 0.5
    if num_run == -1 and base_similarity < 0.5:
      continue
    similarity_scores = []

    # calculate similarity scores for modifying each layer in the model, respectively
    with torch.no_grad():
      add_similarity_score(model_generate.transformer.wte.weight, drop, row, similarity_scores)
      for i in range(12):
        add_similarity_score(model_generate.transformer.h[i].attn.c_attn.weight, drop, row, similarity_scores)
        add_similarity_score(model_generate.transformer.h[i].attn.c_proj.weight, drop, row, similarity_scores)
        add_similarity_score(model_generate.transformer.h[i].mlp.c_fc.weight, drop, row, similarity_scores)
        add_similarity_score(model_generate.transformer.h[i].mlp.c_proj.weight, drop, row, similarity_scores)
      add_similarity_score(model_generate.lm_head.weight, drop, row, similarity_scores)


    # calculate the softmaxes for the similarities to convert into percentage probabilities
    differences = list(map(lambda x: 3*abs(base_similarity - x), similarity_scores))
    tr_differences = torch.tensor(differences)
    sm = torch.nn.Softmax(dim=-1)
    smax = sm(tr_differences)

    # return all_similarities as a list of accuracies, and data_rows as a probability of which is most changed
    all_similarities.append(similarity_scores)
    data_rows.append({"text": row["question"], "label": smax})

  return data_rows, all_similarities


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

# redefine the loss function, because the provided one does not seem to handle multihot encodings
class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)

    # either the loss is the kl divergence, or the absolute value of the difference in probabilities
    # kl divergence ended up giving negative losses, so went with a simpler loss function
    sum = 0
    smax = torch.nn.Softmax(dim=-1)
    sm = smax(outputs.logits)
    for i in range(sm.size(dim=1)):
      sum += abs(sm[0][i] - labels[0][i])/2
    loss = sum
    
    #kl_loss = torch.nn.KLDivLoss()
    #loss = kl_loss(outputs.logits, labels)
    # mimicing return values of overridden function
    return (loss, outputs) if return_outputs else loss





# get all similarity values (for producing initial comparison graphs)
all_similar = []
# every other percentage point between 1 and 99%
for drop_whole in range(1,101,2):
  dropp = drop_whole/100
  # get 10 training examples with that dropout level
  dr, similar = create_dataset("train",dropp,10)
  # take the average of them and put them together
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

# create initial (blank) set of augmented training examples
init_train_data = Data.from_list([])
init_train_data.to_json('/content/drive/My Drive/LLM_Experiment/train_data.json')
# repull every 100 examples because it might time out
for i in range(0, 15000, 100):
  # load previous data
  old_train_data = load_dataset('json', data_files='/content/drive/My Drive/LLM_Experiment/train_data.json')
  # create 100 new examples
  dataset_train_base, similar1 = create_dataset("train", static_start=i, static_end=i+100)
  # merge with previous examples
  new_train_data = Dataset.from_list(dataset_train_base)
  full_train_data = concatenate_datasets([old_train_data['train'], new_train_data])
  # save back to file
  full_train_data.to_json('/content/drive/My Drive/LLM_Experiment/train_data.json')

# create tests normally
dataset_test_base, similar2 = create_dataset("test")
full_test_data = Dataset.from_list(dataset_test_base)
full_test_data.to_json('/content/drive/My Drive/LLM_Experiment/test_data.json')
# load test and train and tokenize them to put into trainer
all_train_data = load_dataset('json', data_files='/content/drive/My Drive/LLM_Experiment/train_data.json')["train"]
all_test_data = load_dataset('json', data_files='/content/drive/My Drive/LLM_Experiment/test_data.json')["train"]
tokenized_train_data = all_train_data.map(tokenize_function)
tokenized_test_data = all_test_data.map(tokenize_function)

# pass in arguments to the trainer
training_args = TrainingArguments(
   output_dir="temp",
   per_device_train_batch_size=1,
   per_device_eval_batch_size=1,
   gradient_accumulation_steps=4,
   num_train_epochs=6
)
trainer = CustomTrainer(
   model=saved_model_predict,
   args=training_args,
   train_dataset=tokenized_train_data,
   eval_dataset=tokenized_test_data,
   compute_metrics=abs_diff
)

# train the model, then save
trainer.train()
trainer.save_model("/content/drive/My Drive/LLM_Experiment")

# all beyond this is printing statistics

# get initial similarity values
arr = all_similar
arr_df = pd.DataFrame(arr)
# Turn column headers into their dropout value
arr_df.index = [str(x/100) for x in range(1,51,2)]

# Turn row headers into the layer name
col_temp = ["Unmodified", "Word Embedding"]
col_temp.extend(sum([["Transformer " + str(i) + " - Attention", "Transformer " + str(i) + " - Attention Mixing", "Transformer " + str(i) + " - Feed Forward Layer 1", "Transformer " + str(i) + " - Feed Forward Layer 2"] for i in range(1,13)],[]))
col_temp.append("Final Linear Layer")
arr_df.columns = col_temp

# Get average of each layer
print(arr_df.mean(axis=0))


NUM_COLORS = 15

cm = plt.get_cmap('gist_rainbow')
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
#for i in range(NUM_COLORS):
#  ax.plot(arr_df.iloc[:, i], label=arr_df.columns[i])
# Uncommented: print by layer location
# Commented: print by layer type
# Double commented: print each layer
ax.plot(arr_df.iloc[:, 0], label=arr_df.columns[0])
ax.plot(arr_df.iloc[:, 1], label=arr_df.columns[1])
for i in range(1,13):
  ax.plot(arr_df.iloc[:, 4*i-2:4*i+2].mean(axis=1), label=("Transformer" + str(i)))
# # for i in range(51):
  # # ax.plot(arr_df.iloc[:, i], label=arr_df.columns[i])
#ax.plot(arr_df.iloc[:, 2:50:4].mean(axis=1), label="Average of Attention")
#ax.plot(arr_df.iloc[:, 3:50:4].mean(axis=1), label="Average of Attention Mixing")
#ax.plot(arr_df.iloc[:, 4:50:4].mean(axis=1), label="Average of Feed Forward Layer 1")
#ax.plot(arr_df.iloc[:, 5:50:4].mean(axis=1), label="Average of Feed Forward Layer 2")
ax.plot(arr_df.iloc[:, 50], label=arr_df.columns[50])

# Move legend outside of the box, and plot the graph
ax.legend(bbox_to_anchor=(1.1, 1.05))
plt.title("Average Similarity")
plt.show()


#arr = printed outputs of trainer.evaluate(), concatenated into a single list
import numpy
# counters for the number of matches in top 1/5
ct = 0
ct2 = 0

for test_rd in arr:
  # prediction logits for each input
  guesses = [i[0] for i in test_rd]
  # ground truths for each input
  labels = [i[1] for i in test_rd]

  # if the layer that they both have the highest value on is the same, they share a prediction
  guess_max = guesses.index(max(guesses))
  label_max = labels.index(max(labels))
  if guess_max == label_max:
    ct += 1

  # Top 5 indices
  guess_top = sorted(range(len(guesses)), key=lambda i: guesses[i], reverse=True)[:5]
  label_top = sorted(range(len(labels)), key=lambda i: labels[i], reverse=True)[:5]

  # if they share a value, that is 1/5 of a prediction shared
  for ss in guess_top:
    if ss in label_top:
      ct2 += 0.2
print(ct)
print(ct2)

# Find the mean that they guessed the layer would damage, for each layer
guessed_damages = [0]*50
labeled_damages = [0]*50
for ar in arr:
  for a in range(len(ar)):
      guessed_damages[a] += ar[a][0]/len(arr)
      labeled_damages[a] += ar[a][1]/len(arr)
        
        

print(labeled_damages)
print(guessed_damages)
print(sorted(guessed_damages))
print(sorted(labeled_damages))

# Print the layer numbers that they each found the most damaging, in order
print(numpy.argsort(guessed_damages))
print(numpy.argsort(labeled_damages))

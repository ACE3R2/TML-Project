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


    #print(similarity_scores)
    differences = list(map(lambda x: 3*abs(base_similarity - x), similarity_scores))

    tr_differences = torch.tensor(differences)
    #print(tr_differences)

    sm = torch.nn.Softmax(dim=-1)
    smax = sm(tr_differences)

    #print(smax)

    #all_similarities.append(similarity_scores)
    similarity_scores.append(base_similarity)
    all_similarities.append(similarity_scores)
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

dataset_train_base, similar1 = create_dataset("train")
dataset_test_base, similar2 = create_dataset("test")
full_train_data = Dataset.from_list(dataset_train_base)
full_test_data = Dataset.from_list(dataset_test_base)

#print(dataset)

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
   for i in range(len(logits)):
      sum += kl_loss(torch.tensor(logits), torch.tensor(labels))
   return {"eval_loss": sum/len(logits)}

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

# -------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------

arr = [[0.2709, 0.2194, 0.2377, 0.2383, 0.2317, 0.2359, 0.243, 0.2257, 0.2318, 0.2246, 0.2317, 0.2377, 0.2317, 0.2508, 0.2306, 0.2093, 0.2239, 0.2322, 0.2317, 0.231, 0.243, 0.2383, 0.243, 0.2264, 0.2317, 0.2306, 0.243, 0.2204, 0.2318, 0.243, 0.2383, 0.2359, 0.22, 0.2318, 0.2423, 0.2377, 0.243, 0.2193, 0.243, 0.243, 0.243, 0.243, 0.243, 0.2377, 0.2311, 0.243, 0.2311, 0.2539, 0.1531, 0.1043, 0.243], [0.2035, 0.1888, 0.1943, 0.1322, 0.1593, 0.2517, 0.1774, 0.1063, 0.1951, 0.1713, 0.1738, 0.0529, 0.1918, 0.1311, 0.1881, 0.2802, 0.1793, 0.1092, 0.1803, 0.1225, 0.1167, 0.132, 0.191, 0.1577, 0.2019, 0.1645, 0.1739, 0.1069, 0.1795, 0.1635, 0.174, 0.1187, 0.1087, 0.1211, 0.1611, 0.1801, 0.2012, 0.1593, 0.1742, 0.1539, 0.1734, 0.1745, 0.1717, 0.2544, 0.1945, 0.1616, 0.1621, 0.1569, 0.1984, 0.2673, 0.1744], [0.3149, 0.1467, 0.1123, 0.0951, 0.1253, 0.1358, 0.1406, 0.175, 0.1172, 0.1365, 0.1417, 0.1112, 0.1871, 0.1388, 0.2371, 0.1915, 0.1918, 0.2011, 0.2042, 0.077, 0.1832, 0.1066, 0.1418, 0.1906, 0.1347, 0.1915, 0.1216, 0.1354, 0.1748, 0.0805, 0.1266, 0.1668, 0.1101, 0.1409, 0.1417, 0.1504, 0.1764, 0.1051, 0.1492, 0.153, 0.15, 0.1463, 0.1515, 0.1243, 0.1859, 0.1514, 0.1844, 0.2035, 0.1678, 0.3066, 0.1417], [0.3694, 0.2207, 0.1357, 0.1295, 0.2505, 0.2033, 0.0614, 0.2894, 0.1342, 0.1989, 0.2016, 0.2037, 0.2599, 0.2171, 0.1322, 0.229, 0.1946, 0.0703, 0.2198, 0.18, 0.2221, 0.2686, 0.2053, 0.0774, 0.1989, 0.2061, 0.1376, 0.0901, 0.1357, 0.1228, 0.151, 0.1369, 0.1965, 0.2235, 0.1355, 0.2037, 0.225, 0.1419, 0.1382, 0.1971, 0.1253, 0.2123, 0.1967, 0.1352, 0.0616, 0.1949, 0.1448, 0.2308, 0.0794, 0.3627, 0.1973], [0.2721, 0.1208, 0.1828, 0.2047, 0.1246, 0.1123, 0.1358, 0.1273, 0.1919, 0.1903, 0.1552, 0.1308, 0.1227, 0.1308, 0.1281, 0.1392, 0.2063, 0.2164, 0.1875, 0.1478, 0.1241, 0.2497, 0.1437, 0.1914, 0.1259, 0.1517, 0.1637, 0.1702, 0.1072, 0.1161, 0.1728, 0.1562, 0.1862, 0.146, 0.1703, 0.1362, 0.1627, 0.2171, 0.1659, 0.152, 0.1589, 0.1107, 0.1718, 0.176, 0.1627, 0.1583, 0.198, 0.1135, 0.1914, 0.4428, 0.176], [0.3637, 0.0734, 0.1314, 0.0694, 0.0789, 0.1419, 0.0682, 0.2271, 0.2232, 0.1461, 0.1581, 0.1368, 0.0845, 0.1429, 0.1495, 0.101, 0.0533, 0.1235, 0.1374, 0.1462, 0.1428, 0.1667, 0.0745, 0.1566, 0.1513, 0.2144, 0.1615, 0.1523, 0.1357, 0.189, 0.1447, 0.072, 0.148, 0.219, 0.1729, 0.1512, 0.1639, 0.0802, 0.2088, 0.142, 0.1553, 0.1557, 0.0726, 0.0541, 0.1422, 0.0927, 0.0713, 0.1588, 0.1377, 0.2654, 0.075], [0.2173, 0.1226, 0.148, 0.1327, 0.0693, 0.0546, 0.0627, 0.0671, 0.0713, 0.0644, 0.0654, 0.0848, 0.0686, 0.1355, 0.0614, 0.0774, 0.084, 0.0676, 0.0607, 0.123, 0.0766, 0.0744, 0.081, 0.0765, 0.0528, 0.0628, 0.1305, 0.1271, 0.0907, 0.064, 0.13, 0.0963, 0.0911, 0.0566, 0.0709, 0.0726, 0.0688, 0.0498, 0.0669, 0.0501, 0.1733, 0.0612, 0.0687, 0.1367, 0.0616, 0.0586, 0.0503, 0.1611, 0.1185, 0.2536, 0.0583], [0.3662, 0.1707, 0.1223, 0.2052, 0.1641, 0.1972, 0.1875, 0.1954, 0.115, 0.2312, 0.2011, 0.175, 0.1611, 0.0932, 0.1374, 0.1081, 0.16, 0.1554, 0.1644, 0.2393, 0.1559, 0.1297, 0.1873, 0.0962, 0.1497, 0.1571, 0.148, 0.0753, 0.1953, 0.2188, 0.2145, 0.1776, 0.1321, 0.1913, 0.1344, 0.1608, 0.1334, 0.1522, 0.1458, 0.1367, 0.1897, 0.2046, 0.1896, 0.2106, 0.1508, 0.1974, 0.1004, 0.1532, 0.1818, 0.3778, 0.2157], [0.3657, 0.0833, 0.064, 0.182, 0.2149, 0.1518, 0.1519, 0.218, 0.2129, 0.2285, 0.1274, 0.1885, 0.1486, 0.2232, 0.2198, 0.0519, 0.1362, 0.1818, 0.157, 0.067, 0.1393, 0.0717, 0.2137, 0.0606, 0.2157, 0.1259, 0.1693, 0.1577, 0.0655, 0.1316, 0.1273, 0.1314, 0.1276, 0.1391, 0.1856, 0.153, 0.1259, 0.1301, 0.1336, 0.132, 0.1288, 0.1576, 0.1501, 0.1461, 0.1286, 0.2288, 0.1377, 0.0717, 0.1204, 0.3668, 0.1312], [0.3443, 0.1104, 0.0886, 0.2454, 0.1469, 0.1197, 0.1585, 0.1329, 0.1084, 0.1403, 0.1551, 0.0939, 0.1499, 0.0781, 0.1541, 0.1162, 0.1676, 0.1289, 0.1295, 0.1991, 0.1042, 0.1618, 0.1318, 0.151, 0.0848, 0.1377, 0.1472, 0.0707, 0.1469, 0.216, 0.084, 0.1323, 0.1466, 0.0867, 0.1029, 0.1379, 0.1247, 0.1339, 0.168, 0.1531, 0.1306, 0.142, 0.0942, 0.1332, 0.1752, 0.1665, 0.15, 0.1277, 0.1265, 0.2592, 0.0863], [0.5473, 0.0952, 0.208, 0.1327, 0.1377, 0.0805, 0.0885, 0.1459, 0.177, 0.0813, 0.0743, 0.1745, 0.1549, 0.1103, 0.1324, 0.2343, 0.1408, 0.2114, 0.0855, 0.1007, 0.1328, 0.102, 0.0815, 0.1163, 0.1613, 0.1396, 0.1083, 0.095, 0.0911, 0.1809, 0.0648, 0.0877, 0.1863, 0.0804, 0.0768, 0.1492, 0.1, 0.0718, 0.0929, 0.1498, 0.1034, 0.1541, 0.146, 0.1063, 0.0626, 0.0942, 0.0757, 0.0263, 0.2298, 0.3231, 0.0695], [0.436, 0.0412, 0.1253, 0.1354, 0.0459, 0.0904, 0.1061, 0.1224, 0.1292, 0.0494, 0.0997, 0.1155, 0.0352, 0.0563, 0.162, 0.0983, 0.0422, 0.0664, 0.1114, 0.0823, 0.0769, 0.1467, 0.0771, 0.0843, 0.0416, 0.1026, 0.0833, 0.1388, 0.0894, 0.0551, 0.0783, 0.1074, 0.0441, 0.0631, 0.0444, 0.0529, 0.0951, 0.0806, 0.0634, 0.1077, 0.1206, 0.1702, 0.06, 0.1165, 0.0665, 0.1175, 0.07, 0.0753, 0.1059, 0.3366, 0.0671], [0.2926, 0.091, 0.0897, 0.0668, 0.0719, 0.1035, 0.0786, 0.1079, 0.073, 0.0908, 0.143, 0.1247, 0.0833, 0.067, 0.1361, 0.1169, 0.0714, 0.1409, 0.1352, 0.0768, 0.0477, 0.0796, 0.1366, 0.0821, 0.0744, 0.1185, 0.1627, 0.1234, 0.078, 0.0869, 0.1331, 0.0932, 0.1356, 0.0817, 0.1499, 0.0811, 0.1839, 0.1064, 0.0755, 0.0605, 0.1357, 0.086, 0.0768, 0.1039, 0.0801, 0.1136, 0.1336, 0.1621, 0.1481, 0.441, 0.153], [0.476, 0.0715, 0.0701, 0.0888, 0.09, 0.1544, 0.0667, 0.1203, 0.138, 0.0694, 0.1444, 0.1035, 0.1485, 0.115, 0.1849, 0.1168, 0.2227, 0.1985, 0.1567, 0.1854, 0.1699, 0.1306, 0.1403, 0.0863, 0.0966, 0.1193, 0.1426, 0.0665, 0.0702, 0.1647, 0.1491, 0.1649, 0.248, 0.1336, 0.2899, 0.1203, 0.0771, 0.083, 0.1227, 0.1834, 0.122, 0.1697, 0.0668, 0.1278, 0.0966, 0.0911, 0.0758, 0.2083, 0.0497, 0.5942, 0.1238], [0.4715, 0.0992, 0.056, 0.088, 0.1106, 0.1549, 0.133, 0.1064, 0.2009, 0.0662, 0.2876, 0.1151, 0.1574, 0.0991, 0.1796, 0.1466, 0.0693, 0.1058, 0.0746, 0.0827, 0.1918, 0.1391, 0.1138, 0.1638, 0.1277, 0.1037, 0.0858, 0.1119, 0.1659, 0.1439, 0.0964, 0.3204, 0.1163, 0.1365, 0.1512, 0.2002, 0.1616, 0.1081, 0.0854, 0.1121, 0.0896, 0.0973, 0.1293, 0.1083, 0.2, 0.0813, 0.1202, 0.2039, 0.1742, 0.4834, 0.1061], [0.4831, 0.0841, 0.2702, 0.1196, 0.1664, 0.3591, 0.2778, 0.0939, 0.2738, 0.1605, 0.2345, 0.1456, 0.1607, 0.1373, 0.1611, 0.2476, 0.1646, 0.1181, 0.0917, 0.2505, 0.1639, 0.0965, 0.1688, 0.1361, 0.1628, 0.145, 0.1835, 0.2783, 0.1923, 0.1433, 0.1997, 0.1519, 0.2578, 0.1815, 0.1574, 0.2414, 0.1623, 0.3364, 0.1682, 0.1224, 0.1178, 0.0598, 0.1451, 0.2518, 0.1361, 0.1796, 0.2481, 0.3323, 0.0934, 0.3856, 0.1592], [0.4268, 0.191, 0.0517, 0.0538, 0.1223, 0.1389, 0.0969, 0.0781, 0.2425, 0.1583, 0.0706, 0.1657, 0.1104, 0.034, 0.1159, 0.1165, 0.1762, 0.0805, 0.2222, 0.1991, 0.095, 0.0861, 0.0971, 0.1368, 0.0396, 0.1443, 0.134, 0.0756, 0.1662, 0.2146, 0.1338, 0.1029, 0.1035, 0.1441, 0.1592, 0.0854, 0.1504, 0.1381, 0.2235, 0.114, 0.1178, 0.2002, 0.1756, 0.1435, 0.2154, 0.0498, 0.0667, 0.1358, 0.1527, 0.3757, 0.1419], [0.512, 0.1041, 0.1124, 0.1431, 0.1089, 0.2067, 0.0875, 0.1745, 0.1225, 0.1232, 0.1277, 0.1317, 0.0914, 0.1104, 0.1229, 0.2235, 0.1928, 0.1984, 0.1221, 0.1013, 0.1469, 0.1335, 0.1393, 0.1582, 0.1237, 0.2259, 0.0498, 0.0706, 0.1475, 0.0952, 0.1737, 0.2388, 0.1402, 0.1912, 0.0611, 0.0678, 0.1074, 0.0782, 0.1162, 0.1216, 0.1107, 0.102, 0.1936, 0.1023, 0.2133, 0.1089, 0.1214, 0.1016, 0.1532, 0.4958, 0.1433], [0.609, 0.1819, 0.1398, 0.1774, 0.2055, 0.2213, 0.1579, 0.2118, 0.1387, 0.1769, 0.1662, 0.197, 0.2013, 0.2021, 0.0979, 0.2777, 0.2409, 0.1377, 0.2959, 0.1529, 0.1554, 0.2028, 0.1103, 0.1818, 0.2233, 0.2367, 0.2281, 0.1311, 0.1678, 0.1648, 0.0829, 0.184, 0.1707, 0.1901, 0.1596, 0.115, 0.2437, 0.1789, 0.128, 0.1833, 0.1534, 0.2254, 0.2138, 0.1065, 0.1885, 0.2171, 0.1025, 0.1654, 0.1779, 0.5181, 0.1435], [0.5456, 0.1281, 0.0802, 0.1074, 0.0868, 0.0868, 0.1106, 0.1162, 0.1606, 0.1626, 0.0742, 0.2112, 0.1095, 0.1645, 0.0933, 0.0645, 0.2257, 0.1191, 0.1106, 0.1954, 0.1426, 0.169, 0.1341, 0.1825, 0.1849, 0.2594, 0.087, 0.2107, 0.1521, 0.2197, 0.1679, 0.1888, 0.1686, 0.1309, 0.0936, 0.1632, 0.08, 0.1826, 0.1268, 0.1439, 0.0636, 0.1121, 0.1516, 0.2446, 0.1413, 0.0998, 0.2217, 0.087, 0.1308, 0.5304, 0.1981], [0.4935, 0.1501, 0.0871, 0.2582, 0.1991, 0.0506, 0.1746, 0.104, 0.1607, 0.157, 0.1541, 0.1706, 0.0536, 0.144, 0.0475, 0.1287, 0.1741, 0.104, 0.1946, 0.1174, 0.1536, 0.0855, 0.1373, 0.1742, 0.0663, 0.0948, 0.1241, 0.1863, 0.4833, 0.1728, 0.2354, 0.1766, 0.1298, 0.1439, 0.1683, 0.0803, 0.1697, 0.1516, 0.1468, 0.2244, 0.1216, 0.1499, 0.1934, 0.1659, 0.0643, 0.1242, 0.2601, 0.14, 0.2131, 0.4845, 0.1324], [0.3985, 0.1836, 0.0336, 0.0934, 0.1131, 0.0567, 0.0659, 0.0823, 0.0554, 0.0449, 0.0595, 0.03, 0.033, 0.0492, 0.0441, 0.0598, 0.033, 0.0614, 0.0432, 0.0395, 0.0276, 0.0419, 0.0603, 0.0421, 0.0595, 0.0837, 0.0413, 0.069, 0.0899, 0.0367, 0.0455, 0.1103, 0.0502, 0.129, 0.0766, 0.0189, 0.0544, 0.0607, 0.0806, 0.0505, 0.2019, 0.0691, 0.0749, 0.0463, 0.1065, 0.0735, 0.0368, 0.054, 0.1061, 0.3727, 0.0614], [0.47, 0.2724, 0.106, 0.0399, 0.1231, 0.0817, 0.1341, 0.182, 0.0748, 0.1741, 0.1008, 0.1238, 0.1263, 0.1399, 0.2066, 0.2264, 0.2143, 0.1253, 0.1407, 0.2128, 0.0602, 0.2092, 0.097, 0.1715, 0.132, 0.2357, 0.2895, 0.1417, 0.2867, 0.1156, 0.0477, 0.1709, 0.151, 0.2264, 0.043, 0.0858, 0.2401, 0.2704, 0.2171, 0.2277, 0.1376, 0.1492, 0.1681, 0.1301, 0.1623, 0.1488, 0.3037, 0.1788, 0.2223, 0.3852, 0.112], [0.4375, 0.0981, 0.166, 0.0628, 0.0829, 0.3743, 0.1635, 0.0823, 0.0965, 0.132, 0.0822, 0.1229, 0.0689, 0.101, 0.2376, 0.1647, 0.269, 0.1099, 0.1506, 0.1856, 0.1609, 0.2475, 0.4217, 0.2802, 0.1447, 0.1683, 0.2465, 0.2167, 0.2036, 0.224, 0.1718, 0.1589, 0.0974, 0.2213, 0.2442, 0.2669, 0.1871, 0.1073, 0.1085, 0.1172, 0.1756, 0.1644, 0.0937, 0.1162, 0.2266, 0.1836, 0.1485, 0.0603, 0.0506, 0.6086, 0.1672], [0.4001, 0.2141, 0.1424, 0.0924, 0.1277, 0.2463, 0.088, 0.2589, 0.2535, 0.1636, 0.2256, 0.187, 0.1606, 0.2102, 0.1338, 0.1909, 0.0965, 0.2004, 0.2357, 0.1944, 0.1295, 0.117, 0.134, 0.1353, 0.1191, 0.1771, 0.0956, 0.1325, 0.2527, 0.1297, 0.1661, 0.1101, 0.1481, 0.2696, 0.2787, 0.1436, 0.2049, 0.1803, 0.1113, 0.1951, 0.1776, 0.2426, 0.2, 0.1671, 0.0968, 0.1524, 0.1321, 0.0836, 0.1494, 0.4312, 0.1942]]

arr_df = pd.DataFrame(arr)

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

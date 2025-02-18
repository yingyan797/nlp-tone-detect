import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv, math
import config, utils

sections = ["train", "test"]

def draw_plot(title, xlabel, ylabel, path):
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.tight_layout()
  plt.savefig(path)
  plt.close()
  
def text_length(data, section):
   # Bar graph (histogram) of text length distribution
  plt.figure(figsize=(10, 6))
  # Identify any rows in "text" column where the data is of float type
  float_data = data[data["text"].apply(lambda x: isinstance(x, float))]
  data["text_length"] = data["text"].apply(len)
  plt.hist(data["text_length"], bins=30, color="orange", edgecolor="black")
  draw_plot(f"{section.capitalize()} Set Text Length Distribution",
            "Text Length", "Frequency", f"images/text_length_distribution_{section}.png")

def country_code(data, section):
  # Bar graph of country code distribution
  plt.figure(figsize=(10, 6))
  country_counts = data["country_code"].value_counts().rename(index=str).sort_index()
  plt.bar(country_counts.index, country_counts.values, color="green")
  draw_plot(f"{section.capitalize()} Set Country Code Distribution", 
            "Country Code", "Count", f"images/country_code_distribution_{section}.png")

def keyword_distribution(data, section):
  # Bar graph of key word distribution
  plt.figure(figsize=(10, 6))
  keyword_counts = data["keyword"].value_counts().rename(index=str).sort_index()
  plt.bar(keyword_counts.index, keyword_counts.values, color="purple")
  draw_plot(f"{section.capitalize()} Set Key Word Distribution",
            "Key Word", "Count", f"images/keyword_distribution_{section}.png")

def grouped_distribution(data, section="train"):
  # Grouped pie charts of label distribution by text length
  # Bin text lengths into ranges
  bins = [0, 100, 200, 300, 400, 500, 1000, np.inf]
  bin_labels = ["0-100", "100-200", "200-300", "300-400", "400-500", "500-1000", "1000+"]
  data["text_length_bin"] = pd.cut(data["text_length"], bins=bins, labels=bin_labels, right=False)
  grouped_tl = data.groupby(["text_length_bin", "label"]).size().unstack(fill_value=0)
  num_bins = len(grouped_tl.index)
  cols = 3
  rows = math.ceil(num_bins / cols)
  plt.figure(figsize=(cols * 4, rows * 4))
  for i, bin_val in enumerate(grouped_tl.index):
    plt.subplot(rows, cols, i + 1)
    counts = grouped_tl.loc[bin_val]
    counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, legend=False)
    plt.title(f"Text Length {bin_val}")
    plt.ylabel('')
  plt.suptitle(f"{section.capitalize()} Set Label Distribution by Text Length")
  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.savefig(f"images/label_by_text_length_{section}.png")
  plt.close()

  # Grouped pie charts of label distribution by country code
  grouped_cc = pd.crosstab(data["country_code"], data["label"])
  num_countries = len(grouped_cc.index)
  cols = 3
  rows = math.ceil(num_countries / cols)
  plt.figure(figsize=(cols * 4, rows * 4))
  for i, code in enumerate(grouped_cc.index):
    plt.subplot(rows, cols, i + 1)
    counts = grouped_cc.loc[code]
    counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, legend=False)
    plt.title(f"Country Code: {code}")
    plt.ylabel('')
  plt.suptitle(f"{section.capitalize()} Set Label Distribution by Country Code")
  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.savefig(f"images/label_by_country_code_{section}.png")
  plt.close()

  # Grouped pie charts of label distribution by key word
  grouped_kw = pd.crosstab(data["keyword"], data["label"])
  num_keywords = len(grouped_kw.index)
  cols = 3
  rows = math.ceil(num_keywords / cols)
  plt.figure(figsize=(cols * 4, rows * 4))
  for i, kw in enumerate(grouped_kw.index):
    plt.subplot(rows, cols, i + 1)
    counts = grouped_kw.loc[kw]
    counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, legend=False)
    plt.title(f"Keyword: {kw}")
    plt.ylabel('')
  plt.suptitle(f"{section.capitalize()} Set Label Distribution by Key Word")
  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.savefig(f"images/label_by_keyword_{section}.png")
  plt.close()

def label_distribution(data, section="train"):
  # Bar graph of label distribution
  plt.figure(figsize=(10, 6))
  label_counts = data["label"].value_counts()
  desired_order = ["0", "1", "2", "3", "4"]
  label_counts = label_counts.reindex(desired_order)
  plt.bar(label_counts.index, label_counts.values, color="skyblue")
  draw_plot(f"{section.capitalize()} Set Label Distribution", 
            "Label", "Count", f"images/label_distribution_{section}.png")

def train_test_compare(train_data, test_data):
  # Compare distribution of training and test sets

  # Pie chart of text length distribution for train and test sets

  bins = [0, 100, 200, 300, 400, 500, 1000, np.inf]
  bin_labels = ["0-100", "100-200", "200-300", "300-400", "400-500", "500-1000", "1000+"]

  # Compute text lengths for both datasets
  train_data["text_length"] = train_data["text"].apply(len)
  test_data["text_length"] = test_data["text"].apply(len)

  # Bin the text lengths
  train_data["text_length_bin"] = pd.cut(train_data["text_length"], bins=bins, labels=bin_labels, right=False)
  test_data["text_length_bin"] = pd.cut(test_data["text_length"], bins=bins, labels=bin_labels, right=False)

  # Count occurrences in each bin
  train_tl_counts = train_data["text_length_bin"].value_counts().sort_index()
  test_tl_counts = test_data["text_length_bin"].value_counts().sort_index()

  # Plot pie charts for train and test sets
  plt.figure(figsize=(12, 6))

  plt.subplot(1, 2, 1)
  train_tl_counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, legend=False)
  plt.title("Train Set Text Length Distribution")
  plt.ylabel('')

  plt.subplot(1, 2, 2)
  test_tl_counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, legend=False)
  plt.title("Test Set Text Length Distribution")
  plt.ylabel('')

  plt.tight_layout()
  plt.savefig("images/combined_text_length_distribution.png")
  plt.close()

  # Pie chart of country code distribution for train and test sets
  train_country_counts = train_data["country_code"].value_counts().sort_index()
  test_country_counts = test_data["country_code"].value_counts().sort_index()

  plt.figure(figsize=(12, 6))

  plt.subplot(1, 2, 1)
  train_country_counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, legend=False)
  plt.title("Train Set Country Code Distribution")
  plt.ylabel('')

  plt.subplot(1, 2, 2)
  test_country_counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, legend=False)
  plt.title("Test Set Country Code Distribution")
  plt.ylabel('')

  plt.tight_layout()
  plt.savefig("images/combined_country_code_distribution.png")
  plt.close()

  # Pie chart of key word distribution for train and test sets
  train_keyword_counts = train_data["keyword"].value_counts().sort_index()
  test_keyword_counts = test_data["keyword"].value_counts().sort_index()

  plt.figure(figsize=(12, 6))

  plt.subplot(1, 2, 1)
  train_keyword_counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, legend=False)
  plt.title("Train Set Key Word Distribution")
  plt.ylabel('')

  plt.subplot(1, 2, 2)
  test_keyword_counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, legend=False)
  plt.title("Test Set Key Word Distribution")
  plt.ylabel('')

  plt.tight_layout()
  plt.savefig("images/combined_keyword_distribution.png")
  plt.close()

def visualize_embedding(data):
  keywords = data["keyword"].unique()
  from transformers import RobertaModel, RobertaTokenizer
  import torch

  model = RobertaModel.from_pretrained("roberta-base")
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
  for kw in keywords:
    sentence = torch.tensor([tokenizer.encode(kw)])
    print(kw, model.embeddings(sentence))

if __name__ == "__main__":
  train_data = utils.read_train_data()
  test_data = utils.read_test_data()

  for section in sections:
    if section == "train":
      data = train_data
    else:
      data = test_data

    # text_length(data, section)
    # country_code(data, section)
    # keyword_distribution(data, section)

    if section == "train":
      # label_distribution(data, section)

      # grouped_distribution(data)
      pass

  # train_test_compare(train_data, test_data)
  visualize_embedding(train_data)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

import config
import utils
import math

sections = ["train", "test"]

for section in sections:
  if section == "train":
    data = utils.read_train_data()
  else:
    data = utils.read_test_data()
  
  # Bar graph of label distribution
  plt.figure(figsize=(10, 6))
  label_counts = data["label"].value_counts()
  desired_order = ["0", "1", "2", "3", "4"]
  label_counts = label_counts.reindex(desired_order)
  plt.bar(label_counts.index, label_counts.values, color="skyblue")
  plt.title(f"{section.capitalize()} Set Label Distribution")
  plt.xlabel("Label")
  plt.ylabel("Count")
  plt.tight_layout()
  plt.savefig(f"images/label_distribution_{section}.png")
  plt.close()

  # Bar graph (histogram) of text length distribution
  plt.figure(figsize=(10, 6))
  # Identify any rows in "text" column where the data is of float type
  float_data = data[data["text"].apply(lambda x: isinstance(x, float))]
  data["text_length"] = data["text"].apply(len)
  plt.hist(data["text_length"], bins=30, color="orange", edgecolor="black")
  plt.title(f"{section.capitalize()} Set Text Length Distribution")
  plt.xlabel("Text Length")
  plt.ylabel("Frequency")
  plt.tight_layout()
  plt.savefig(f"images/text_length_distribution_{section}.png")
  plt.close()

  # Bar graph of country code distribution
  plt.figure(figsize=(10, 6))
  country_counts = data["country_code"].value_counts().rename(index=str).sort_index()
  plt.bar(country_counts.index, country_counts.values, color="green")
  plt.title(f"{section.capitalize()} Set Country Code Distribution")
  plt.xlabel("Country Code")
  plt.ylabel("Count")
  plt.tight_layout()
  plt.savefig(f"images/country_code_distribution_{section}.png")
  plt.close()

  # Bar graph of key word distribution
  plt.figure(figsize=(10, 6))
  keyword_counts = data["keyword"].value_counts().rename(index=str).sort_index()
  plt.bar(keyword_counts.index, keyword_counts.values, color="purple")
  plt.title(f"{section.capitalize()} Set Key Word Distribution")
  plt.xlabel("Key Word")
  plt.ylabel("Count")
  plt.tight_layout()
  plt.savefig(f"images/keyword_distribution_{section}.png")
  plt.close()

  if section == "train":
    # Grouped pie charts of label distribution by text length
    # Bin text lengths into ranges
    bins = [0, 50, 100, 150, 200, 250, 300, np.inf]
    bin_labels = ["0-50", "50-100", "100-150", "150-200", "200-250", "250-300", "300+"]
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
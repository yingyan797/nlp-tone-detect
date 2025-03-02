import pandas as pd
import config
import numpy as np

def _read_data(data_path, skip_header=True, augment=False, merge_labels=False, has_labels=False):
  with open(data_path, 'r') as f:
    data = []
    if skip_header:
      while True:
        line = f.readline()
        if not line.strip():
          break
    while True:
      line = f.readline()
      if not line:
        break
      row = line.strip().split("\t")
      if has_labels:
        label = float(row[-1])
        row[-1] = label
        if label >= 2:
          if merge_labels:
            row[-1] = 1.0
          duplication = 3 if augment else 1
          data += duplication * [row]
        else:
          if merge_labels:
            row[-1] = 0.0
          data.append(row)
      else:
        data.append(row)
  if not has_labels:
    data_col_headers = config.data_col_headers[:-1]
  else:
    data_col_headers = config.data_col_headers
  data = pd.DataFrame(data, columns=data_col_headers)
  return data[data['text'].notna()]

def read_train_data(merge_labels=False):
  return _read_data(config.train_data_path, merge_labels=merge_labels, has_labels=True)
   
def read_test_data():
  return _read_data(config.test_data_path, False, has_labels=False)

def read_train_split(merge_labels=True):
  train_data = read_train_data(merge_labels=merge_labels)  
  train_ids = pd.read_csv(
    config.train_split_path,
    usecols=[0],
    dtype=str
  )
  
  dev_ids = pd.read_csv(
    config.dev_split_path,
    usecols=[0],
    dtype=str
  )
  
  train_part = train_data[train_data['par_id'].isin(train_ids['par_id'])]
  dev_part = train_data[train_data['par_id'].isin(dev_ids['par_id'])]
  
  return train_part, dev_part

if __name__ == '__main__':
  # train, dev = read_train_split()
  # print(train.head())
  # print(dev.head())
  import torch
  print(torch.ones(3) * torch.ones(3))
import pandas as pd
import config
import numpy as np

def _read_data(data_path, skip_header=True, augment=False, has_labels=False):
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
        label = int(row[-1])
        row[-1] = label
        if label >= 2:
          duplication = 3 if augment else 1
          data += duplication * [row]
        else:
          data.append(row)
      else:
        data.append(row)
  if not has_labels:
    data_col_headers = config.data_col_headers[:-1]
  else:
    data_col_headers = config.data_col_headers
  data = pd.DataFrame(data, columns=data_col_headers)
  return data[data['text'].notna()]

def read_train_data():
  return _read_data(config.train_data_path, has_labels=True)
   
def read_test_data():
  return _read_data(config.test_data_path, False, has_labels=False)

def read_train_split():
  train_data = read_train_data()  
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
  dev_part['label'] = dev_part['label'].where(dev_part['label'] >= 2, other=0).where(dev_part['label'] < 2, other=1)

  return train_part, dev_part

if __name__ == '__main__':
  # train, dev = read_train_split()
  # print(train.head())
  # print(dev.head())
  import torch
  # t = torch.ones(10,1)
  # print(t.squeeze(1))
  t, d = read_train_split()

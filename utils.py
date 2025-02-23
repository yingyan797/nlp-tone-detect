import pandas as pd
import config

def read_train_data():
  skip_rows = 0
  with open(config.train_data_path, 'r') as f:
    # Skip until we find the first delimiter line.
    for line in f:
      skip_rows += 1
      if line.startswith('------'):
        break
    # Skip the disclaimer content until the closing delimiter.
    for line in f:
      skip_rows += 1
      if line.startswith('------'):
        break

  train_data = pd.read_csv(
    config.train_data_path,
    delimiter='\t',
    skiprows=skip_rows,
    index_col=0,
    names=config.data_col_headers,
    dtype=str
  )
  train_data = train_data[train_data['text'].notna()]
  
  return train_data

def read_test_data():
  test_data = pd.read_csv(
    config.test_data_path,
    delimiter='\t',
    index_col=0,
    names=config.data_col_headers,
    dtype=str
  )
  test_data = test_data[test_data['text'].notna()]
  
  return test_data

def read_train_split():
  train_data = read_train_data()
  
  train_ids = pd.read_csv(
    'dataset/practice_splits/train_semeval_parids-labels.csv',
    header=None,
    usecols=[0],
    dtype=str
  )
  
  dev_ids = pd.read_csv(
    'dataset/practice_splits/dev_semeval_parids-labels.csv',
    header=None,
    usecols=[0],
    dtype=str
  )
  
  train_data = train_data[train_data.index.isin(train_ids[0])]
  dev_data = train_data[train_data.index.isin(dev_ids[0])]
  
  return train_data, dev_data
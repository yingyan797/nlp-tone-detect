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
    names=config.data_col_headers,
    dtype=config.data_col_dtypes
  )
  train_data = train_data[train_data['text'].notna()]
  
  return train_data

def read_test_data():
  test_data = pd.read_csv(
    config.test_data_path,
    delimiter='\t',
    names=config.data_col_headers[:-1],
    dtype=config.data_col_dtypes
  )
  test_data = test_data[test_data['text'].notna()]
  
  return test_data

def read_train_split():
  train_data = train_data_preprocessing(read_train_data())
  
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

def train_data_preprocessing(train_data):
  train_data.rename(columns={'label': 'orig_label'}, inplace=True)
  train_data['label'] = train_data['orig_label'].apply(lambda x: 1 if x > 1 else 0)
  
  selected_columns = ['par_id', 'text', 'label']
  train_data = train_data[selected_columns]
  
  return train_data

if __name__ == '__main__':
  train, dev = read_train_split()
  print(train.head())
  print(dev.head())
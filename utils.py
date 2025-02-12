import pandas as pd

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
    names=config.data_col_headers
  )
  
  return train_data
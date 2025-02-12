import matplotlib.pyplot as plt
import numpy as np
import csv

import config

with open(config.train_data_path, 'r', newline='', encoding='utf-8') as train_file:
    train_set_reader = csv.reader(train_file, delimiter='\t')

# Bar graph of label distribution

# Bar graph of text length distribution

# Bar graph of contry code distribution

# Bar graph of key word distribution

# Grouped bar graph of label distribution by text length

# Grouped bar graph of label distribution by country code

# Grouped bar graph of label distribution by key word
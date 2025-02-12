import matplotlib.pyplot as plt
import numpy as np
import csv

import config

with open(config.train_data_path, 'r', newline='', encoding='utf-8') as train_file:
    train_set_reader = csv.reader(train_file, delimiter='\t')


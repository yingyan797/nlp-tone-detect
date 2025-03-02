import torch, tqdm
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from utils import read_train_split, read_test_data
import itertools

from pipeline import RobertaGCN, TextClassificationDataset

batch_size = 256
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def pred_all(data_loader):
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.reshape(-1).cpu().numpy())
    return all_preds
  

# Load the model
model = RobertaGCN()
model.load_state_dict(torch.load('best_model_0.01_256_3_normFalse.pth'))
model.eval()

_, dev_data = read_train_split()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

texts_dev = dev_data['text'].to_list()
labels_dev = dev_data['label'].to_list()
dev_dataset = TextClassificationDataset(texts_dev, labels_dev, tokenizer)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

dev_preds = pred_all(dev_loader)
dev_acc = accuracy_score(labels_dev, dev_preds)
dev_f1 = f1_score(labels_dev, dev_preds, average='binary')
with open("dev.txt", "w") as f:
    for pred in dev_preds:
        f.write(f"{pred}\n")
print(f"Done predciting dev Acc: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}")

test_data = read_test_data()
texts_test = test_data['text'].to_list()
test_dataset = TextClassificationDataset(texts_test, None, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_preds = pred_all(test_loader)
with open("test.txt", "w") as f:
    for pred in test_preds:
        f.write(f"{pred}\n")
print("Done predicting test")
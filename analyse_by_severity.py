import torch
import numpy as np
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from utils import read_train_split
import itertools
import pandas as pd
import numpy as np
import config


# from pipeline import RobertaGCN
PARAMS = {
    "lr": 0.01,
    "batch_size": 256,
    "loss_weight": 3,
    "batch_norm": False
}

class RobertaGCN(nn.Module):
    def __init__(self, embed_dim=768, hidden_dims=[256,128], num_classes=1, device=torch.device('cuda')):
        super(RobertaGCN, self).__init__()
        # Load pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # RoBERTa embedding and attention parameters are pretrained and fixed
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        # 2 layer Graph convolutional neural network
        self.gcn1 = GCNConv(embed_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0]) if PARAMS["batch_norm"] else nn.Identity()
        self.gcn2 = GCNConv(hidden_dims[0], hidden_dims[1])
        
        # Classification head
        self.classifier = nn.Linear(hidden_dims[1], num_classes)
        self.to(device)
        self.device = device
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states: torch.Tensor = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        
        # Divide graph to batches; Use attention as adjacency matrix for each sample in batch
        all_outputs = []
        attention_threshold = 0.1
        for i in range(batch_size):
            # Extract attention from last layer as adjacency matrix
            attention = torch.matmul(hidden_states[i], hidden_states[i].transpose(0, 1))
            attention = F.softmax(attention, dim=-1)
            valid = (attention > attention_threshold)
            edge_indices = valid.nonzero().t()
            edge_weights = attention.masked_select(valid)
            # Create edges for nodes with attention above threshold            
            # Apply GCN with the attention matrix as the adjacency matrix
            x = hidden_states[i]  # [seq_len, hidden_size]
            x = self.gcn1(x, edge_index=edge_indices, edge_weight=edge_weights)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.bn1(x)
            x = self.gcn2(x, edge_index=edge_indices, edge_weight=edge_weights)
            
            # Global pooling
            node_mask = attention_mask[i].bool()
            graph_embedding = torch.mean(x[node_mask], dim=0)
            
            all_outputs.append(graph_embedding)
        
        stacked_outputs = torch.stack(all_outputs)
        # Classification
        logits = self.classifier(stacked_outputs)
        
        return F.sigmoid(logits)




def load_model(model_path, device):
    model = RobertaGCN(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }


def evaluate_model_by_severity(model, test_loader, device, severity, save_path="./performance_analysis/severity/severity_evaluation.json"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = (outputs >= 0.5).int()
            
            all_preds.extend(preds.reshape(-1).cpu().numpy())
            all_labels.extend(labels.to(dtype=int).cpu().numpy())
    
    pos_label = 0 if severity < 2 else 1
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=1, pos_label=pos_label)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=1, pos_label=pos_label)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=1, pos_label=pos_label)
    
    print(f'Severity: {severity}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    
    results = {
        "Severity": int(severity),  # Convert NumPy integer to Python int
        "accuracy": float(accuracy),  # Convert NumPy float to Python float
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    }

    
    with open(save_path, "a") as f:
        json.dump(results, f, indent=4)

    return results

import matplotlib.pyplot as plt
import numpy as np

def plot_severity_levels():
    # Severity levels and corresponding F1 Scores
    severity_levels = [0, 1, 2, 3, 4]
    f1_scores = [0.0000, 0.0000, 0.7257, 0.8211, 0.8732]

    # Plotting the bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(severity_levels, f1_scores, tick_label=severity_levels)

    # Labels and title
    plt.xlabel("Severity Level of Patronising Content")
    plt.ylabel("F1 Score")
    plt.title("F1 Score over Severity Levels of Patronising Content")

    # Show values on top of bars
    for i, v in enumerate(f1_scores):
        plt.text(severity_levels[i], v + 0.02, str(round(v, 4)), ha='center', fontsize=10)

    # Show the plot
    plt.ylim(0, 1.1)  # Set y-axis limit for clarity
    plt.savefig("./performance_analysis/severity/f1_scores_by_severity.png")
    plt.show()


if __name__ == "__main__":
    # plot_severity_levels()

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
    # train_data, dev_data = read_train_split(dev_split_path=f"./category/{category}_data.tsv")
    # # texts_train = train_data['text'].to_list()
    # # labels_train = train_data['label'].to_list()
    # texts_val = dev_data['text'].to_list()
    # labels_val = dev_data['label'].to_list()
    
    df = pd.read_csv(config.train_data_path, sep='\t', encoding='utf-8', skiprows=3)
    # Rename columns for clarity
    df.columns = ["index", "par_id", "category", "region", "text", "label"]
    
    # labels_val = [0.0 if label < 2 else 1.0 for label in labels_val]

    labels = df["label"].unique()
    print(df["label"].value_counts())
    
    save_path="./performance_analysis/severity/severity_evaluation.json"
    with open(save_path, "w") as f:
        f.write("")
    
    for label in labels:
        df_category = df[df["label"] == label]
        texts_val = df_category['text'].fillna("").astype(str).to_list()
        # labels_val = df_category['label'].to_list()
        # labels_val = [0.0 if label < 2 else 1.0 for label in labels_val]
        labels_val = [0.0 if int(label) < 2 else 1.0] * len(texts_val)

        
        val_dataset = TextClassificationDataset(texts_val, labels_val, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=PARAMS['batch_size'])
        
        # Load best model
        model_path = f"best_model_{PARAMS['lr']}_{PARAMS['batch_size']}_{PARAMS['loss_weight']}_norm{PARAMS['batch_norm']}.pth"
        model = load_model(model_path, device)
        
        # Run evaluation
        evaluate_model_by_severity(model, val_loader, device, severity=label, save_path=save_path)
        
        

# Output

# label
# 0    8528
# 1     947
# 3     458
# 4     391
# 2     144
# Name: count, dtype: int64

# Severity: 0
# Accuracy: 0.9707
# F1 Score: 0.0000
# Precision: 0.0000
# Recall: 1.0000

# Severity: 1
# Accuracy: 0.8184
# F1 Score: 0.0000
# Precision: 0.0000
# Recall: 1.0000
# S
# Severity: 2
# Accuracy: 0.5694
# F1 Score: 0.7257
# Precision: 1.0000
# Recall: 0.5694

# Severity: 3
# Accuracy: 0.6965
# F1 Score: 0.8211
# Precision: 1.0000
# Recall: 0.6965

# Severity: 4
# Accuracy: 0.7749
# F1 Score: 0.8732
# Precision: 1.0000
# Recall: 0.7749
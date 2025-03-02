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


# Define category keywords for subgroups
subgroups = {
    "homelessness": ["homeless", "in-need", "poverty", "unhoused"],
    "poverty": ["poverty", "in-need", "low-income", "poor"],
    "disability": ["disabled", "handicapped", "special-needs", "impairment"],
}

# Define file paths for each category
file_paths = {
    "homelessness": "./category/homelessness_data.tsv",
    "poverty": "./category/poverty_data.tsv",
    "disability": "./category/disability_data.tsv",
}

def divide_category():
    df = pd.read_csv(config.train_data_path, sep='\t', encoding='utf-8', skiprows=3)
    
    # Rename columns for clarity
    df.columns = ["index", "par_id", "category", "region", "text", "label"]

    # Create a new column for subgroup assignment
    def categorize_text(row):
        for group, keywords in subgroups.items():
            if any(keyword in str(row["category"]).lower() for keyword in keywords):
                return group
        return "other"  # Default if no match

    df["subgroup"] = df.apply(categorize_text, axis=1)

    # Filter data by relevant subgroups
    df_filtered = df[df["subgroup"] != "other"]

    # Save each category to a separate TSV file
    saved_files = {}
    for category, path in file_paths.items():
        df_category = df_filtered[df_filtered["subgroup"] == category]
        df_category.to_csv(path, sep='\t', index=False, encoding='utf-8')
        saved_files[category] = path

    print("Data divided into categories:")
    print(saved_files)



def evaluate_model_by_category(model, test_loader, device, category, save_path="./category/category_evaluation.json"):
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
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    
    print(f'Category: {category}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    
    results = {
        "category": category,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }
    
    with open(save_path, "a") as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":
    # divide_category()
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    categories = ["homelessness", "poverty", "disability"]
    for category in categories:
         
        # train_data, dev_data = read_train_split(dev_split_path=f"./category/{category}_data.tsv")
        # # texts_train = train_data['text'].to_list()
        # # labels_train = train_data['label'].to_list()
        # texts_val = dev_data['text'].to_list()
        # labels_val = dev_data['label'].to_list()
        
        dev_data = pd.read_csv(f"./category/{category}_data.tsv", sep='\t', encoding='utf-8')
        
        texts_val = dev_data['text'].to_list()
        labels_val = dev_data['label'].to_list()
        labels_val = [0.0 if label < 2 else 1.0 for label in labels_val]
        
        # test_dataset = TextClassificationDataset(texts_test, labels_test, tokenizer)
        val_dataset = TextClassificationDataset(texts_val, labels_val, tokenizer)

        # test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=PARAMS['batch_size'])
        
        # Load best model
        model_path = f"best_model_{PARAMS['lr']}_{PARAMS['batch_size']}_{PARAMS['loss_weight']}_norm{PARAMS['batch_norm']}.pth"  # Update with actual model filename
        model = load_model(model_path, device)
        
        # Run evaluation
        evaluate_model_by_category(model, val_loader, category, device)
        
        print("Evaluation complete! Results saved in evaluation_results.json.")
    
    
    # # Define the file path for one of the categories to inspect
    # file_path = "./category/homelessness_data.tsv"  # Change based on the category causing the issue

    # # Try reading the file while handling errors
    # try:
    #     df_test = pd.read_csv(file_path, sep='\t', encoding='utf-8', error_bad_lines=False)
    # except Exception as e:
    #     df_test = str(e)

    # # Display first few rows and structure
    # df_test.head() if isinstance(df_test, pd.DataFrame) else df_test

# RESULTS

# Data divided into categories:
# {'homelessness': './category/homelessness_data.tsv', 'poverty': './category/poverty_data.tsv', 'disability': './category/disability_data.tsv'}

# "homelessness", "poverty", "disability"

# Accuracy: 0.8986
# F1 Score: 0.7160
# Precision: 0.6619
# Recall: 0.7797

# Accuracy: 0.8801
# F1 Score: 0.6403
# Precision: 0.6340
# Recall: 0.6467

# Accuracy: 0.9241
# F1 Score: 0.5618
# Precision: 0.5155
# Recall: 0.6173
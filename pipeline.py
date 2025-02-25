import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv
from transformers import RobertaModel
import numpy as np
from tqdm import tqdm

class RobertaGCN(nn.Module):
    def __init__(self, embed_dim=768, hidden_dims=[256,128], num_classes=5):
        super(RobertaGCN, self).__init__()
        # Load pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # RoBERTa embedding and attention parameters are pretrained and fixed
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        # 2 layer Graph convolutional neural network
        self.gcn1 = GCNConv(embed_dim, hidden_dims[0])
        self.gcn2 = GCNConv(hidden_dims[0], hidden_dims[1])
        
        # Classification head
        self.classifier = nn.Linear(hidden_dims[1], num_classes)
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states: torch.Tensor = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        
        # Divide graph to batches; Use attention as adjacency matrix for each sample in batch
        all_outputs = []
        for i in range(batch_size):
            # Extract attention from last layer as adjacency matrix
            attention = torch.matmul(hidden_states[i], hidden_states[i].transpose(0, 1))
            attention = F.softmax(attention, dim=-1)
            
            # Apply GCN with the attention matrix as the adjacency matrix
            x = hidden_states[i]  # [seq_len, hidden_size]
            x = self.gcn1(x, edge_index=attention)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.gcn2(x, edge_index=attention)
            
            # Global pooling
            node_mask = attention_mask[i].bool()
            graph_embedding = torch.mean(x[node_mask], dim=0)
            
            all_outputs.append(graph_embedding)
        
        stacked_outputs = torch.stack(all_outputs)
        # Classification
        logits = self.classifier(stacked_outputs)
        
        return logits

# Create a simple dataset class
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
            'label': torch.tensor(label, dtype=torch.long)
        }

import torch, tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from utils import read_train_split

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
        edge_indices = []
        for src in range(seq_len):
            for dst in range(seq_len):
                # if attention[src, dst] > attention_threshold:
                edge_indices.append([src, dst])
        edge_index = torch.tensor(edge_indices, dtype=torch.long, device=model.device).t().contiguous()
        # attention_threshold = 0.1  # Or use top-k attention values
        
        # Divide graph to batches; Use attention as adjacency matrix for each sample in batch
        all_outputs = []
        for i in range(batch_size):
            # Extract attention from last layer as adjacency matrix
            attention = torch.matmul(hidden_states[i], hidden_states[i].transpose(0, 1))
            attention = F.softmax(attention, dim=-1)
            edge_weight = attention.reshape(-1)

            # Create edges for nodes with attention above threshold            
            # Apply GCN with the attention matrix as the adjacency matrix
            x = hidden_states[i]  # [seq_len, hidden_size]
            x = self.gcn1(x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.gcn2(x, edge_index=edge_index, edge_weight=edge_weight)
            
            # Global pooling
            node_mask = attention_mask[i].bool()
            graph_embedding = torch.mean(x[node_mask], dim=0)
            
            all_outputs.append(graph_embedding)
        
        stacked_outputs = torch.stack(all_outputs)
        # Classification
        logits = self.classifier(stacked_outputs)
        
        return F.sigmoid(logits)

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
            'label': torch.tensor(label, dtype=torch.float)
        }

PARAMS = {
    "lr": 2e-5,
    "batch_size": 128
}

# Training function
def train_model(model, train_loader, val_loader, num_epochs=5):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].to(model.device)
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.reshape(-1, 1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            print(f"Epoch {epoch+1} Batch loss {loss.item()}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['label'].to(model.device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='binary' if len(set(all_labels)) == 2 else 'macro')
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        with open("training_record.csv", "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_acc:.4f},{val_f1:.4f}\n")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save the model
            torch.save(model.state_dict(), 'best_roberta_gcn_model.pth')
            print("Model saved!")
        
        print("-" * 50)
    
    return model

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Load your data
    # This is a placeholder - replace with your actual data loading code

    train_data, dev_data = read_train_split()
    texts_train = train_data['text'].to_list()
    labels_train = train_data['label'].to_list()
    texts_val = dev_data['text'].to_list()
    labels_val = dev_data['label'].to_list()

    # Create datasets
    train_dataset = TextClassificationDataset(texts_train, labels_train, tokenizer)
    val_dataset = TextClassificationDataset(texts_val, labels_val, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=PARAMS['batch_size'])

    # Initialize model
    model = RobertaGCN(num_classes=1).to(torch.device("cuda"))

    # Train model
    trained_model = train_model(model, train_loader, val_loader)

    print("Training completed!")
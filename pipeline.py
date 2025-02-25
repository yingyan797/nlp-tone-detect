import torch, tqdm
import numpy as np
from PIL import Image
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
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
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
            # x = self.bn1(x)
            x = self.gcn2(x, edge_index=edge_indices, edge_weight=edge_weights)
            
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
    "lr": 1e-3,
    "batch_size": 256,
    "loss_weight": 5
}

# Training function
def train_model(model, train_loader, val_loader, num_epochs=5):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=PARAMS['lr'])
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss(reduction='none')
    
    best_val_f1 = 0.0
    with open("training_record.csv", "w") as f:
        f.write("Epoch,Train Loss,Val Loss,Acc,F1\n")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].reshape(-1, 1).to(model.device)
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            orig_loss = criterion(outputs, labels)
            weight = torch.ones(labels.shape, device=model.device) + (PARAMS["loss_weight"]-1)*labels

            loss = torch.mean(weight * orig_loss)
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
                loss = torch.mean(criterion(outputs, labels.reshape(-1,1)))
                
                val_loss += loss.item()
                
                preds = (outputs >= 0.5).int()
                all_preds.extend(preds.reshape(-1).cpu().numpy())
                all_labels.extend(labels.to(dtype=int).cpu().numpy())
        
        im_size = int(np.sqrt(len(all_preds)))+1
        res_im = np.zeros((im_size, 2*im_size), dtype=bool)
        for i in range(im_size):
            for j in range(im_size):
                loc = i*im_size+j
                if loc >= len(all_preds):
                    break
                res_im[i,j] = all_preds[loc]
                res_im[i,j+im_size] = all_labels[loc]
        Image.fromarray(res_im).save(f"train_visual/{epoch+1}.png")

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='binary')
        
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
    val_loader = DataLoader(val_dataset, batch_size=PARAMS['batch_size'], shuffle=True)

    # Initialize model
    model = RobertaGCN(num_classes=1).to(torch.device("cuda"))

    # Train model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=100)

    print("Training completed!")
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
import itertools

class RobertaGCN(nn.Module):
    def __init__(self, embed_dim=768, hidden_dims=[256,128], num_classes=5, device=torch.device('cuda'), has_batch_norm=False):
        super(RobertaGCN, self).__init__()
        # Load pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # RoBERTa embedding and attention parameters are pretrained and fixed
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        # 2 layer Graph convolutional neural network
        self.gcn1 = GCNConv(embed_dim, hidden_dims[0])
        if has_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dims[0]) if has_batch_norm else nn.Identity()
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
            if hasattr(self, "bn1"):
                x = self.bn1(x)
            x = self.gcn2(x, edge_index=edge_indices, edge_weight=edge_weights)
            
            # Global pooling
            node_mask = attention_mask[i].bool()
            graph_embedding = torch.mean(x[node_mask], dim=0)
            
            all_outputs.append(graph_embedding)
        
        stacked_outputs = torch.stack(all_outputs)
        # Classification
        logits = self.classifier(stacked_outputs)
        
        return logits, stacked_outputs

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

def triplet_loss(embeddings, labels, margin=1.0):
    """
    Triplet loss to minimize intra-class distances and maximize inter-class distances
    """
    distance_matrix = torch.cdist(embeddings, embeddings)
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
    
    # For each anchor, find hardest positive and hardest negative
    hardest_positive = torch.max(distance_matrix * pos_mask.float(), dim=1)[0]
    hardest_negative = torch.min(distance_matrix * neg_mask.float() + 
                               (~neg_mask).float() * 1e6, dim=1)[0]
    # Compute triplet loss with margin
    losses = torch.relu(hardest_positive - hardest_negative + margin)
    return losses.mean()

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, x, labels):
        batch_size = x.size(0)
        centers_batch = self.centers[labels]
        # Calculate distance between features and their centers
        return torch.sum((x - centers_batch)**2) / batch_size

# Training function
def train_model(model, train_loader, val_loader, num_epochs=5):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=PARAMS['lr'])
    criterion = nn.CrossEntropyLoss(reduction='none')
    # criterion = nn.BCELoss(reduction='none')
    center_loss = CenterLoss(num_classes=5, feat_dim=128).to(model.device)
    
    best_val_f1 = 0.0
    name = f"{PARAMS['lr']}_{PARAMS['batch_size']}_{PARAMS['loss_weight']}_norm{PARAMS['batch_norm']}"
    with open("training_record.csv", "a") as f:
        f.write(f"Epoch,Train Loss,Val Loss,Acc,F1,{name}\n")
    weight = torch.tensor([[0.1,0.5,2,1,1] for _ in range(PARAMS['batch_size'])], device=model.device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].to(model.device)
            optimizer.zero_grad()
            
            logits, graph_embeddings = model(input_ids, attention_mask)
            ce_loss = criterion(logits, labels)
            _weight = torch.gather(weight, 1, labels.unsqueeze(1)).squeeze(1)
            ce_loss = torch.mean(_weight * ce_loss)
            # tp_loss = triplet_loss(graph_embeddings, labels)
            ct_loss = center_loss(graph_embeddings, labels)
            loss = 5*ce_loss + 0.2*ct_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"Epoch {epoch} batch losses {loss}={ce_loss}|{ct_loss}")
        
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
                
                logits, graph_embeddings = model(input_ids, attention_mask)
                loss = torch.mean(criterion(logits, labels))+0.2*center_loss(graph_embeddings,labels)
                
                val_loss += loss.item()
                
                _, preds = torch.max(logits, 1)
                preds = torch.where(preds < 2, 0, 1)
                all_preds.extend(preds.reshape(-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # im_size = int(np.sqrt(len(all_preds)))+1
        # res_im = np.zeros((im_size, 2*im_size), dtype=bool)
        # for i in range(im_size):
        #     for j in range(im_size):
        #         loc = i*im_size+j
        #         if loc >= len(all_preds):
        #             break
        #         res_im[i,j] = all_preds[loc]
        #         res_im[i,j+im_size] = all_labels[loc]
        # Image.fromarray(res_im).save(f"train_visual/{epoch+1}.png")

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
            torch.save(model.state_dict(), f'best_model_{name}.pth')
            print("Model saved!")
        
        if val_f1 > 0.545:
            break
        
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

    learning_rates = [1e-2]
    loss_weights = [3]
    batch_norm = [False]

    with open("training_record.csv", "w") as f:
        f.write("")
    for lr, weight, bn in itertools.product(learning_rates, loss_weights, batch_norm):
        PARAMS = {
            "lr": lr,
            "batch_size": 512,
            "loss_weight": weight,
            "batch_norm": bn
        }
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=PARAMS['batch_size'], shuffle=False)

        # Initialize model
        model = RobertaGCN(num_classes=5, has_batch_norm=PARAMS['batch_norm']).to(torch.device("cuda"))
        with open(f"best_model_{lr}_256_{weight}_normFalse.pth", "rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)
        # Train model
        trained_model = train_model(model, train_loader, val_loader, num_epochs=15)

        print("Training completed!")
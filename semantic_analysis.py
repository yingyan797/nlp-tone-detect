import sklearn.manifold
from transformers import RobertaModel, RobertaTokenizer
import simpletransformers.classification.classification_model
import torch
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import utils

model = RobertaModel.from_pretrained("roberta-base")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

train_data = utils.read_train_data()
test_data = utils.read_test_data()

def sentence_embedding(sentence, aggregate=False):
    enc = torch.tensor([tokenizer.encode(sentence)])
    embedding = model.embeddings(enc)
    if not aggregate:
        return embedding.reshape((-1, embedding.shape[-1]))
    return embedding.reshape(-1)

def embedding_3d(data, lim=4):
    n_remaining = [lim for label in [0,1,2,3,4]]
    sample = [[], [], []]
    for i, row in data.iterrows():
        if not(any(n_remaining)):
            break
        sentence = row["text"]
        label = int(row["label"])
        if n_remaining[label] == 0:
            continue
        n_remaining[label] -= 1
        
        emb = sentence_embedding(sentence)
        sample[0] += [feat for feat in emb]
        sample[1] += [label] * emb.shape[0]
        sample[2] += [i] * emb.shape[0]
    
    feat_mat = torch.stack(sample[0])
    feat_mat -= feat_mat.mean(0)
    U, S, V = torch.pca_lowrank(feat_mat)
    feat_mat = torch.matmul(feat_mat, V[:, :3])
    sample[0] = feat_mat.detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    plt.title("3D Embedding PCA Visual")
    scatter = ax.scatter(sample[0][:, 0], sample[0][:, 1], sample[0][:, 2], c=sample[1], marker=",")
 
    plt.legend(*scatter.legend_elements())
    plt.savefig("images/embedding_3d.png")

def embedding_heatmap(sentence):
    grouped_embedding = sentence_embedding(sentence)
    heatmap = np.zeros((grouped_embedding.shape[0], grouped_embedding.shape[0]))
    for i in range(grouped_embedding.shape[0]-1):
        for j in range(i+1, grouped_embedding.shape[0]):
            dot_product = torch.dot(grouped_embedding[i], grouped_embedding[j])
            cosine = dot_product / (torch.norm(grouped_embedding[i]) * torch.norm(grouped_embedding[j]))
            heatmap[i, j] = cosine
            heatmap[j, i] = cosine
    
    plt.title(f"Embedding heatmap of sentence")
    plt.imshow(heatmap)
    plt.savefig("images/sentence_embedding.png")

    

if __name__ == "__main__":
    # embedding_heatmap(
    #     "The factors resulting in Mongolia's extremely cold in winters include the country's relatively high latitude, far inland location, and high average elavation"
    # )
    embedding_3d(train_data)
    

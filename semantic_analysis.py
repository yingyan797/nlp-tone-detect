import sklearn.manifold
from transformers import RobertaModel, RobertaTokenizer
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

def embedding_2d(data, lim=4):
    n_remaining = [lim for label in [0,1,2,3,4]]
    sample = [[], [], [], []]
    for i, row in data.iterrows():
        if not(any(n_remaining)):
            break
        sentence = row["text"]
        label = int(row["label"])
        if n_remaining[label] == 0:
            continue
        n_remaining[label] -= 1
        
        emb = sentence_embedding(sentence)
        visual_2d = sklearn.manifold.TSNE(n_components=2, 
                    perplexity=emb.shape[0]-1).fit_transform(emb.detach().numpy())
        sample[0] += list(visual_2d[:, 0])
        sample[1] += list(visual_2d[:, 1])
        sample[2] += [label] * visual_2d.shape[0]
        sample[3] += [i] * visual_2d.shape[0]
    plt.title("2D embedding visualization")
    scatter = plt.scatter(sample[0], sample[1], c=sample[2])
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.legend(*scatter.legend_elements())
    plt.savefig("images/embedding_2d.png")

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
    embedding_2d(train_data)

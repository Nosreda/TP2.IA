from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch

model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_embeddings(texts):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # [CLS] token
            embeddings.append(cls_embedding)
    return embeddings

from sklearn.cluster import KMeans
import numpy as np

df = pd.read_csv("European_Restaurant_Reviews_with_Sentiment.csv")
texts = df['cleaned_review_tokens'].apply(lambda tokens: ' '.join(tokens)).tolist()

embeddings = get_embeddings(texts)  # texts = lista de textos processados
embeddings = np.array(embeddings) 
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Adiciona ao dataframe
df['cluster'] = clusters

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='tab10')
plt.title("Clusterização semântica de avaliações")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.colorbar()
plt.savefig("clusterizacao_tsne.png")
print("Gráfico salvo em clusterizacao_tsne.png")
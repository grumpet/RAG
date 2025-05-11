# Install required packages if needed
# pip install requests numpy matplotlib scikit-learn

import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json

# Ollama embedding API endpoint (make sure Ollama is running locally)
OLLAMA_API = "http://localhost:11434/api/embeddings"

# Words to embed
words = ["man", "woman", "boy", "girl","king", "queen", "prince", "princess", "doctor", "nurse"]

# Function to get embeddings from Ollama
def get_ollama_embedding(text, model="llama3"):
    response = requests.post(
        OLLAMA_API,
        json={"model": model, "prompt": text}
    )
    if response.status_code == 200:
        return np.array(response.json()["embedding"])
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Generate embeddings
embeddings = []
for word in words:
    embedding = get_ollama_embedding(word)
    if embedding is not None:
        embeddings.append(embedding)

embeddings = np.array(embeddings)

# Print embedding information
print(f"Embedding dimensions: {embeddings.shape}")
for word, embedding in zip(words, embeddings):
    print(f"\n{word} embedding (first 5 values):")
    print(embedding[:5])

# Calculate similarities between word pairs
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("\nWord pair similarities:")
for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        if i < j:  # Only show each pair once
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"{word1} - {word2}: {sim:.4f}")

# Visualize embeddings in 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=12)
plt.title("Word Embeddings (2D Projection) model=ollama3")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
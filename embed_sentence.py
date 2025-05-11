import requests
import numpy as np

def get_ollama_embedding(text, model="llama3"):
    """Get embedding for a text (word or sentence) using Ollama"""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    if response.status_code == 200:
        return np.array(response.json()["embedding"])
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example sentences
sentences = [
    "The dog chased the cat.",
    "The cat was chased by the dog.",
    "I love machine learning.",
    "Neural networks process data efficiently."
]

# Get embeddings for each sentence
sentence_embeddings = []
for sentence in sentences:
    embedding = get_ollama_embedding(sentence)
    if embedding is not None:
        sentence_embeddings.append(embedding)
        print(f"Embedded: '{sentence}' → vector of shape {embedding.shape}")

# Calculate similarities between sentences
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("\nSentence Similarities:")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])
        print(f"Similarity between '{sentences[i]}' and '{sentences[j]}': {sim:.4f}")
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Function to get embeddings from Ollama
def get_embedding(text):
    """Get embedding vector for a text using Ollama"""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    if response.status_code == 200:
        result = response.json()
        return result["embedding"]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    """Calculate how similar two vectors are (0 to 1)"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Function to visualize multiple words in 2D
def visualize_words(words):
    """Create a 2D plot of word embeddings"""
    # Get embeddings for all words
    embeddings = []
    for word in words:
        print(f"Getting embedding for: {word}")
        embedding = get_embedding(word)
        if embedding:
            embeddings.append(embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_array)
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=100)
    
    # Add labels for each point
    for i, word in enumerate(words):
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
                    fontsize=12, ha='center')
    
    plt.title("Word Embeddings Visualization")
    plt.grid(True)
    plt.show()

# Main demo function
def embedding_demo():
    print("===== Simple Embedding Demo =====")
    print("This demo shows how embeddings capture meaning in text")
    
    # Step 1: Compare similar words
    print("\n1. Comparing similar words:")
    word_pairs = [
        ("happy", "joyful"),
        ("happy", "sad"),
        ("dog", "cat"),
        ("dog", "puppy")
    ]
    
    for word1, word2 in word_pairs:
        emb1 = get_embedding(word1)
        emb2 = get_embedding(word2)
        similarity = cosine_similarity(emb1, emb2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
    
    # Step 2: Compare sentences
    print("\n2. Comparing sentences:")
    sentence1 = "I love programming with Python"
    sentence2 = "Python is my favorite programming language"
    sentence3 = "Elephants are the largest land mammals"
    
    emb_s1 = get_embedding(sentence1)
    emb_s2 = get_embedding(sentence2)
    emb_s3 = get_embedding(sentence3)
    print(f"vector for sentence 1: {emb_s1}")
    print(f"vector for sentence 2: {emb_s2}")
    print(f"vector for sentence 3: {emb_s3}")
    
    print(f"Similarity between related sentences:\n{sentence1}\n{sentence2}\n{cosine_similarity(emb_s1, emb_s2):.4f}")
    print(f"Similarity between unrelated sentences:\n{sentence1} {sentence3}\n{cosine_similarity(emb_s1, emb_s3):.4f}")
    
    # Step 3: Visualize word relationships
    print("\n3. Visualizing word relationships (Closing plot will exit program)")
    words_to_plot = [
        "dog", "cat", "puppy", "kitten",
        "computer", "laptop", "keyboard",
        "apple", "banana", "orange"
    ]
    
    visualize_words(words_to_plot)

if __name__ == "__main__":
    embedding_demo()
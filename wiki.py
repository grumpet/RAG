import wikipedia
import os
import chromadb
import requests
import json
from chromadb.utils import embedding_functions
import PyPDF2  # Add PyPDF2 for PDF processing



class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/embeddings"
    
    def __call__(self, texts):
        embeddings = []
        for text in texts:
            response = requests.post(
                self.api_url,
                json={"model": self.model_name, "prompt": text}
            )
            if response.status_code == 200:
                result = response.json()
                embeddings.append(result["embedding"])
            else:
                raise Exception(f"Error from Ollama API: {response.text}")
        return embeddings

# Create the Ollama embedding function
ollama_ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")


def get_collection_names():
    """
    Return a list of all collection names in the ChromaDB database.
    
    Returns:
        list: A list of strings containing all collection names
    """
    collections = chroma_client.list_collections()
    collection_names = [collection.name for collection in collections]
    return collection_names

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Function to generate embeddings using Ollama API
def get_ollama_embedding(text):
    print(f"Generating embedding for: {text}")
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    if response.status_code == 200:
        result = response.json()
        print("==== Generating embeddings... ====")
        return result["embedding"]
    else:
        raise Exception(f"Error from Ollama API: {response.text}")





def generate_documents(topic, collection_name=None):
    """
    Search Wikipedia for a topic, process articles into chunks,
    and store in ChromaDB collection.
    
    Args:
        topic: Topic to search on Wikipedia
        collection_name: Optional name for ChromaDB collection
    
    Returns:
        tuple: (collection object, number of documents added)
    """
    documents = []
    ids = []
    metadatas = []
    
    # Use topic name for collection if not specified
    if not collection_name:
        collection_name = f"wiki_{topic.replace(' ', '_').lower()}_collection"
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=ollama_ef
    )
    
    # Search Wikipedia
    results = wikipedia.search(topic, results=5)
    print(f"Found {len(results)} articles for '{topic}'")
    
    for title in results:
        try:
            print(f"Processing: {title}")
            page = wikipedia.page(title)
            
            # Get article content and split into chunks
            chunks = split_text(page.content)
            print(f"  Split into {len(chunks)} chunks")
            
            # Process each chunk
            for i, chunk_text in enumerate(chunks):
                doc_id = f"{title.replace(' ', '_')}_{i}"
                
                # Store document text
                documents.append(chunk_text)
                
                # Store document ID
                ids.append(doc_id)
                
                # Store metadata
                metadata = {
                    "title": title,
                    "url": page.url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": "wikipedia",
                    "topic": topic
                }
                metadatas.append(metadata)
                
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"  Disambiguation error for '{title}'. Options: {e.options[:3]}...")
        except wikipedia.exceptions.PageError:
            print(f"  Page '{title}' not found")
        except Exception as e:
            print(f"  Error processing '{title}': {str(e)}")
    
    # Add documents to collection if any were found
    if documents:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Added {len(documents)} document chunks to collection '{collection_name}'")
    else:
        print("No documents were added to the collection")
    
    return collection, len(documents)

def extract_topics_from_question(question, model="llama3"):
    """
    Use Ollama to extract relevant search topics from a user question
    
    Args:
        question: User's question
        model: Ollama model to use
    
    Returns:
        list: Relevant topics to search for on Wikipedia
    """
    print(f"generating topics from this query: {question}....")
    prompt = (
        "Extract 1-3 most relevant Wikipedia topic search terms from this question. "
        "Return ONLY the topics as a comma-separated list without explanations or additional text. "
        "For example, if the question is 'How do quantum computers work?', "
        "you would return 'Quantum computing, Quantum mechanics, Quantum circuit'.\n\n"
        f"Question: {question}\n\n"
        "Topics:"
    )
    
    # Make request to Ollama's chat API
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            "stream": False
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        topics_text = result["message"]["content"].strip()
        # Split the comma-separated topics and clean them up
        topics = [topic.strip() for topic in topics_text.split(",")]
        print(f"\ncreated these topics:\n {topics}\n")
        return topics
    else:
        print(f"Error extracting topics: {response.status_code}")
        return [question]  # Fall back to using the question as the topic




def check_query_relevance(query, topics, threshold=0.65):
    """
    Check if a query is relevant to any of the given topics
    using embedding similarity.
    
    Args:
        query: User's question
        topics: List of potential topics
        threshold: Minimum similarity score to consider relevant
    
    Returns:
        tuple: (is_relevant, most_relevant_topic, score)
    """
    # Get embedding for the query
    query_embedding = get_ollama_embedding(query)
    
    # Get embeddings for all topics
    topic_embeddings = []
    for topic in topics:
        topic_emb = get_ollama_embedding(topic)
        topic_embeddings.append(topic_emb)
    
    # Calculate cosine similarity
    max_similarity = 0
    most_relevant_topic = None
    
    for i, topic_emb in enumerate(topic_embeddings):
        # Calculate cosine similarity
        similarity = cosine_similarity_vectors(query_embedding, topic_emb)
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_topic = topics[i]
    
    is_relevant = max_similarity >= threshold
    
    return (is_relevant, most_relevant_topic, max_similarity)

def cosine_similarity_vectors(v1, v2):
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def query_documents(question, topics, n_results=5):
    """
    Query documents across multiple collections based on topics.
    
    Args:
        question: The query question
        topics: List of topics to search collections for
        n_results: Number of results to retrieve per collection
    """
    all_results = []
    query_embedding = get_ollama_embedding(question)
    
    # Get all collections
    all_collection_names = get_collection_names()
    
    # Process each topic
    for topic in topics:
        # Format collection name to match generate_documents naming convention
        collection_name = f"wiki_{topic.replace(' ', '_').lower()}_collection"
        
        if collection_name not in all_collection_names:
            print(f"Warning: Collection for topic '{topic}' does not exist")
            continue
        
        # Get the collection object
        collection = chroma_client.get_collection(
            name=collection_name, 
            embedding_function=ollama_ef
        )
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for i in range(len(documents)):
            all_results.append({
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
                "source": topic  # Use topic as source identifier
            })
    
    # Sort by relevance
    all_results.sort(key=lambda x: x["distance"])
    
    # Return the most relevant chunks
    return [result["text"] for result in all_results[:n_results]]



# Function to generate a response from Ollama
def generate_response_with_sources(question, relevant_chunks, chunk_sources):
    print("generating response...\n\n\n")
    print("using the following chunks...\n")
    
    # Print chunks with their sources
    for i, (chunk, source) in enumerate(zip(relevant_chunks, chunk_sources)):
        print(f"\n\n--- Chunk {i+1} from topic: {source} ---")
        print(f"{chunk}\n\n")
    
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    # Make a request to Ollama's API
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            "stream": False
        }
    )

    if response.status_code == 200:
        result = response.json()
        return result["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

query = 'hoe does tee work ?' 
topics = extract_topics_from_question(query)
collection_names = get_collection_names()
print(f"Check if there are simmilar collections from the exsiting collections:\n {collection_names}\n")
    


collection_fixed_names = [name.replace("wiki_", "").replace("_collection", "").replace("_", " ") for name in collection_names]
print(f"Check if there are simmilar collections from the exsiting collections: {collection_fixed_names}\n\n")


relevant_chunks = []
chunk_sources = []  # Track which topic each chunk comes from

# Process each extracted topic directly without checking relevance
for topic in topics:
    topic_normalized = topic.lower()
    collection_name = f"wiki_{topic_normalized.replace(' ', '_')}_collection"
    
    # Check if collection exists
    if collection_name in collection_names:
        print(f"Using existing collection: {collection_name}")
    else:
        print(f"Creating new collection for topic: {topic}")
        _, _ = generate_documents(topic)
    
    # Get/query the collection
    collection = chroma_client.get_collection(name=collection_name, embedding_function=ollama_ef)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[get_ollama_embedding(query)],
        n_results=5, 
        include=["documents", "metadatas"]
    )
    
    # Add retrieved chunks to relevant_chunks
    for doc in results["documents"][0]:
        relevant_chunks.append(doc)
        chunk_sources.append(topic)  # Track which topic this chunk came from

print("Relevant chunks retrieved:", len(relevant_chunks))
answer = generate_response_with_sources(query, relevant_chunks, chunk_sources)
print("\nAnswer:")
print(answer)




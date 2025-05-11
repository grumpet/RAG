import os
from dotenv import load_dotenv
import chromadb
import requests
from chromadb.utils import embedding_functions
import argparse

# Load environment variables from .env file
load_dotenv()

# Custom embedding function for Ollama
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

# Initialize the Ollama embedding function
ollama_ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")

# Define repositories and collections mapping
repositories = {
    "news_articles": "news_articles_collection",
    "tee_data": "tee_data_collection"
}

# Get collection references
def get_collection(name):
    try:
        return chroma_client.get_collection(name=name, embedding_function=ollama_ef)
    except Exception as e:
        print(f"Error accessing collection {name}: {str(e)}")
        return None

# Load collections
collections = {}
for repo_name, collection_name in repositories.items():
    collection = get_collection(collection_name)
    if collection:
        collections[repo_name] = collection
        print(f"✅ Loaded collection: {collection_name}")
    else:
        print(f"❌ Failed to load collection: {collection_name}")

# Function to generate embeddings using Ollama API
def get_ollama_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    if response.status_code == 200:
        result = response.json()
        return result["embedding"]
    else:
        raise Exception(f"Error from Ollama API: {response.text}")

# Function to query documents from multiple collections
def query_documents(question, repo_names=None, n_results=3, show_metadata=False):
    if repo_names is None:
        repo_names = list(collections.keys())
    
    all_results = []
    query_embedding = get_ollama_embedding(question)
    
    for repo_name in repo_names:
        if repo_name not in collections:
            print(f"Warning: Collection for {repo_name} does not exist")
            continue
            
        print(f"Querying {repo_name}...")
        collection = collections[repo_name]
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Add source information and combine results
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for i in range(len(documents)):
            all_results.append({
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
                "source": repo_name
            })
    
    # Sort by relevance (lower distance means more relevant)
    all_results.sort(key=lambda x: x["distance"])
    
    # Return complete result objects if metadata is requested
    if show_metadata:
        return all_results[:n_results]
    # Otherwise just return the text
    return [result["text"] for result in all_results[:n_results]]

# Function to generate a response from Ollama
def generate_response(question, relevant_chunks, model="llama3"):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use one sentence maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    # Make a request to Ollama's API
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,  # Use the specified model
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user", 
                    "content": question
                }
            ],
            "stream": False
        }
    )

    if response.status_code == 200:
        result = response.json()
        return result["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# Display available collections
def show_available_collections():
    print("\n📚 Available collections:")
    for repo_name in collections.keys():
        print(f"  - {repo_name}")
    print()

# List available Ollama models
def list_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("\n🤖 Available models:")
            for model in models["models"]:
                print(f"  - {model['name']}")
            print()
            return [model['name'] for model in models["models"]]
        else:
            print(f"Error fetching models: {response.status_code}")
            return ["llama3"]  # Default fallback
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        return ["llama3"]  # Default fallback

# Simple command-line interface
def main():
    parser = argparse.ArgumentParser(description='Query RAG collections')
    parser.add_argument('--question', '-q', help='Question to ask')
    parser.add_argument('--repo', '-r', choices=['news', 'tee', 'all'], default='all',
                        help='Repository to query (news, tee, or all)')
    parser.add_argument('--results', '-n', type=int, default=3, 
                        help='Number of results to retrieve')
    parser.add_argument('--model', '-m', default='llama3',
                        help='Ollama model to use for response generation')
    parser.add_argument('--show-chunks', '-c', action='store_true',
                        help='Show the retrieved text chunks')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Map repo argument to actual repo names
    repo_mapping = {
        'news': ['news_articles'],
        'tee': ['tee_data'],
        'all': None  # None means all repositories
    }
    
    if args.interactive:
        # Interactive mode
        available_models = list_ollama_models()
        default_model = 'llama3' if 'llama3' in available_models else available_models[0] if available_models else 'llama3'
        
        show_available_collections()
        print(f"Default model: {default_model}")
        
        while True:
            question = input("\n❓ Enter your question (or 'quit' to exit): ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            # Repository selection
            repo_choice = input("\n📚 Query which repositories? (news/tee/all, default=all): ")
            if repo_choice.lower() == "news":
                repo_names = ['news_articles']
            elif repo_choice.lower() == "tee":
                repo_names = ['tee_data'] 
            else:
                repo_names = None  # Query all
            
            # Results count
            try:
                n_results = int(input("\n🔢 Number of results to retrieve (default=3): ") or "3")
            except ValueError:
                n_results = 3
            
            # Model selection
            model_choice = input(f"\n🤖 Model to use (default={default_model}): ") or default_model
            
            # Show chunks option
            show_chunks = input("\n👁️ Show retrieved chunks? (y/n, default=n): ").lower() == 'y'
            
            print("\n🔍 Retrieving relevant information...")
            relevant_chunks = query_documents(question, repo_names=repo_names, n_results=n_results, show_metadata=show_chunks)
            
            if show_chunks:
                print("\n📄 Retrieved chunks:")
                for i, chunk in enumerate(relevant_chunks):
                    print(f"\n--- Chunk {i+1} (source: {chunk['source']}, relevance: {chunk['distance']:.4f}) ---")
                    print(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                # Extract just the text for the response generation
                relevant_chunks = [chunk['text'] for chunk in relevant_chunks]
            
            if not relevant_chunks:
                print("❌ No relevant information found.")
                continue
                
            print(f"\n💭 Generating answer using {model_choice}...")
            answer = generate_response(question, relevant_chunks, model=model_choice)
            
            print("\n✅ Answer:")
            print(answer)
            
    elif args.question:
        # Single question mode
        repo_names = repo_mapping[args.repo]
        relevant_chunks = query_documents(args.question, repo_names=repo_names, n_results=args.results, show_metadata=args.show_chunks)
        
        if args.show_chunks:
            print("\nRetrieved chunks:")
            for i, chunk in enumerate(relevant_chunks):
                print(f"\n--- Chunk {i+1} (source: {chunk['source']}, relevance: {chunk['distance']:.4f}) ---")
                print(chunk['text'])
            # Extract just the text for the response generation
            relevant_chunks = [chunk['text'] for chunk in relevant_chunks]
        
        answer = generate_response(args.question, relevant_chunks, model=args.model)
        print("\nAnswer:")
        print(answer)
        
    else:
        # No arguments provided, show help
        parser.print_help()

if __name__ == "__main__":
    if not collections:
        print("Error: No collections found or loaded. Please run app_ollama_rag.py first to create and populate collections.")
    else:
        main()
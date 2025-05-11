import os
from dotenv import load_dotenv
import chromadb
import requests
import json
from chromadb.utils import embedding_functions
import PyPDF2  # Add PyPDF2 for PDF processing

# Load environment variables from .env file
load_dotenv()

# We'll implement a custom embedding function for Ollama
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

# Define repositories and their collections
repositories = {
    "news_articles": "./news_articles",
    "tee_data": "./TEE_DATA"
}

# Initialize collections for each repository
def get_or_create_collection(name):
    return chroma_client.get_or_create_collection(
        name=f"{name}_collection", embedding_function=ollama_ef
    )

# Create a dictionary to store collections
collections = {
    repo_name: get_or_create_collection(repo_name) 
    for repo_name in repositories.keys()
}

# Function to load documents from a directory - updated to handle PDFs
# Function to load documents from a directory - updated to handle PDFs with errors
def load_documents_from_directory(directory_path):
    print(f"==== Loading documents from {directory_path} ====")
    documents = []
    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} does not exist")
        return documents
        
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    documents.append({"id": filename, "text": file.read()})
            elif filename.endswith(".pdf"):
                print(f"Processing PDF: {filename}")
                pdf_text = extract_text_from_pdf(file_path)
                if pdf_text.strip():  # Only add if text was successfully extracted
                    documents.append({"id": filename, "text": pdf_text})
                else:
                    print(f"Skipping {filename} due to extraction failure")
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            print("Continuing with next file...")
            continue
            
    return documents

# New function to extract text from PDF files with error handling
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        print(f"Error extracting text from page {page_num} in {pdf_path}: {str(e)}")
                        continue
                
                if not text.strip():
                    print(f"Warning: Extracted empty text from {pdf_path}")
            except KeyError as e:
                print(f"Error: Invalid PDF structure in {pdf_path}: {str(e)}")
            except Exception as e:
                print(f"Error reading PDF {pdf_path}: {str(e)}")
    except Exception as e:
        print(f"Error opening file {pdf_path}: {str(e)}")
        
    return text

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

# Modified function to update a specific collection
def update_collection_with_documents(repo_name):
    directory_path = repositories[repo_name]
    collection = collections[repo_name]
    
    documents = load_documents_from_directory(directory_path)
    print(f"Loaded {len(documents)} documents from {directory_path}")
    
    # Split documents into chunks
    chunked_documents = []
    for doc in documents:
        chunks = split_text(doc["text"])
        print(f"==== Splitting {doc['id']} into chunks ====")
        for i, chunk in enumerate(chunks):
            chunked_documents.append({
                "id": f"{doc['id']}_chunk{i+1}", 
                "text": chunk,
                "metadata": {"source": repo_name}
            })
    
    # Generate embeddings and upsert into database
    for doc in chunked_documents:
        print(f"==== Processing chunk {doc['id']} ====")
        doc["embedding"] = get_ollama_embedding(doc["text"])
        collection.upsert(
            ids=[doc["id"]], 
            documents=[doc["text"]], 
            embeddings=[doc["embedding"]],
            metadatas=[{"source": repo_name}]
        )
    
    print(f"Added {len(chunked_documents)} chunks to the {repo_name} collection")

# Update all collections
def update_all_collections():
    for repo_name in repositories.keys():
        update_collection_with_documents(repo_name)

# Reset a specific collection
def reset_collection(repo_name):
    collection_name = f"{repo_name}_collection"
    if collection_name in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=collection_name)
        collections[repo_name] = get_or_create_collection(repo_name)
    update_collection_with_documents(repo_name)

# Reset all collections
def reset_all_collections():
    for repo_name in repositories.keys():
        reset_collection(repo_name)

# Modified function to query documents from multiple collections
def query_documents(question, repo_names=None, n_results=2):
    if repo_names is None:
        repo_names = list(repositories.keys())
    
    all_results = []
    query_embedding = get_ollama_embedding(question)
    
    for repo_name in repo_names:
        if repo_name not in collections:
            print(f"Warning: Collection for {repo_name} does not exist")
            continue
            
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
    
    # Return just the text of the most relevant chunks
    return [result["text"] for result in all_results[:n_results]]

# Function to generate a response from Ollama
def generate_response(question, relevant_chunks):
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
            "model": "llama3", # You can change this to any model you have in Ollama
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

# Interactive interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--update":
            # Update all collections
            update_all_collections()
            print("All collections updated successfully!")
            
        elif sys.argv[1] == "--update-news":
            # Update only news articles collection
            update_collection_with_documents("news_articles")
            print("News articles collection updated successfully!")
            
        elif sys.argv[1] == "--update-tee":
            # Update only TEE data collection
            update_collection_with_documents("tee_data")
            print("TEE data collection updated successfully!")
            
        elif sys.argv[1] == "--reset":
            # Reset all collections
            reset_all_collections()
            print("All collections reset and updated successfully!")
            
        elif sys.argv[1] == "--reset-news":
            # Reset only news articles collection
            reset_collection("news_articles")
            print("News articles collection reset successfully!")
            
        elif sys.argv[1] == "--reset-tee":
            # Reset only TEE data collection
            reset_collection("tee_data")
            print("TEE data collection reset successfully!")
            
        elif sys.argv[1] == "--query":
            # Interactive query mode
            while True:
                question = input("\nEnter your question (or 'quit' to exit): ")
                if question.lower() == "quit":
                    break
                    
                # Allow specifying which repositories to query
                repo_choice = input("\nQuery which repositories? (news/tee/all, default=all): ")
                
                if repo_choice.lower() == "news":
                    repo_names = ["news_articles"]
                elif repo_choice.lower() == "tee":
                    repo_names = ["tee_data"]
                else:
                    repo_names = None  # Query all
                    
                relevant_chunks = query_documents(question, repo_names=repo_names)
                answer = generate_response(question, relevant_chunks)
                
                print("\nAnswer:", answer)
                print("\nSources used:", ", ".join(repo_names) if repo_names else "all repositories")
    else:
        # Default: update all collections and run example query
        update_all_collections()
        question = "tell me about databricks"
        relevant_chunks = query_documents(question)
        answer = generate_response(question, relevant_chunks)
        print(answer)
import os
import chromadb
import pdfplumber
import numpy as np
import requests
import json
import time
from requests.exceptions import RequestException
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import re

# Directory for PDF data
DATA_DIR = "c:\\Users\\nknim\\Desktop\\rag\\TEE_DATA"
CHROMA_DIR = "c:\\Users\\nknim\\Desktop\\rag\\chroma_db"
OLLAMA_API = "http://localhost:11434/api/embeddings"
OLLAMA_GENERATE_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # Ollama 3 model

# 1. Create ChromaDB client and collections
def create_chroma_client():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Collection for dense embeddings
    dense_collection = client.get_or_create_collection(
        name="tee_data_dense",
        metadata={"hnsw:space": "cosine"}
    )
    # Collection for sparse embeddings
    sparse_collection = client.get_or_create_collection(
        name="tee_data_sparse"
    )
    
    return client, dense_collection, sparse_collection

# 2. Extract text from PDFs
def extract_text_from_pdfs(directory: str) -> Dict[str, str]:
    pdf_texts = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            text = ""
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                pdf_texts[filename] = text
                print(f"Extracted text from {filename}")
            except Exception as e:
                print(f"Error extracting text from {filename}: {e}")
    
    return pdf_texts

# 3. Create chunks from text
def create_chunks(texts: Dict[str, str], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    chunks = []
    
    for filename, text in texts.items():
        if not text.strip():
            continue
            
        # Simple chunking by character count with overlap
        text_len = len(text)
        start = 0
        chunk_id = 0
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]
            
            # Clean the chunk
            chunk_text = re.sub(r'\n+', ' ', chunk_text)
            chunk_text = re.sub(r'\s+', ' ', chunk_text)
            chunk_text = chunk_text.strip()
            
            if chunk_text:
                chunks.append({
                    "id": f"{filename}_chunk_{chunk_id}",
                    "text": chunk_text,
                    "metadata": {
                        "source": filename,
                        "chunk_id": chunk_id
                    }
                })
                
            chunk_id += 1
            start += (chunk_size - overlap)
    
    return chunks

# 4. Get embeddings using Ollama
def safe_json_parse(response):
    """Safely parse JSON from response with error handling"""
    try:
        return response.json()
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response content: {response.content[:200]}...")  # Print first 200 chars
        # Try to clean the response if it contains extra data
        try:
            content = response.content.decode('utf-8')
            # Find the end of the first JSON object
            pos = content.find('}\n')
            if pos > 0:
                clean_json = content[:pos+1]
                return json.loads(clean_json)
        except Exception:
            pass
        # Return empty result as fallback
        return {}

def query_embeddings(text, model="llama2", retries=3):
    """Get embeddings with retry mechanism"""
    url = "http://localhost:11434/api/embeddings"
    data = {
        "model": model,
        "prompt": text
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = safe_json_parse(response)
            if result and "embedding" in result:
                return result["embedding"]
            time.sleep(1)  # Wait before retry
        except (RequestException, KeyError) as e:
            print(f"Embedding request failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2)  # Longer wait between retries
            
    raise Exception(f"Failed to get embeddings after {retries} attempts")

def get_ollama_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = []
    
    for text in texts:
        try:
            embedding = query_embeddings(text)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Add a zero vector as placeholder for failed embeddings
            embeddings.append([0.0] * 4096)  # Adjust dimension based on model
    
    return embeddings

# 5. Store embeddings in ChromaDB
def store_embeddings(chunks: List[Dict], dense_collection, sparse_collection):
    # Extract all the texts, ids and metadata
    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    # Get dense embeddings from Ollama
    dense_embeddings = get_ollama_embeddings(texts)
    
    # Store in dense collection
    dense_collection.add(
        documents=texts,
        embeddings=dense_embeddings,
        ids=ids,
        metadatas=metadatas
    )
    
    # For sparse collection, we don't need embeddings
    sparse_collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"Stored {len(chunks)} chunks in ChromaDB")

# 7. Query function with dense embeddings
def query_dense(query: str, dense_collection, top_k: int = 5):
    # Get query embedding
    query_embedding = get_ollama_embeddings([query])[0]
    
    # Query the collection
    results = dense_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    return results

# 8. Query function with sparse retrieval (BM25)
def query_sparse(query_text, sparse_collection, top_k=5):
    """
    Perform sparse retrieval using BM25
    """
    # Get all documents from the collection
    all_docs = sparse_collection.get(include=["documents", "metadatas"])
    
    # Extract documents and their IDs
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    
    # Tokenize the documents
    tokenized_documents = [re.findall(r'\w+', doc.lower()) for doc in documents]
    
    # Create the BM25 model
    bm25 = BM25Okapi(tokenized_documents)
    
    # Tokenize the query
    tokenized_query = re.findall(r'\w+', query_text.lower())
    
    # Get scores
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Get top-k results
    top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
    top_scores = [doc_scores[i] for i in top_indices]
    
    # Get the corresponding documents and metadata
    top_docs = [documents[i] for i in top_indices]
    top_metadata = [metadatas[i] for i in top_indices]

    # Extract IDs from metadata if available
    ids = [metadata.get("id", f"doc_{i}") for i, metadata in enumerate(metadatas)]

    # Return in same format as dense embeddings for consistency
    return {
        "documents": [top_docs],
        "metadatas": [top_metadata],
        "scores": [top_scores],
        "ids": [ids]  # Added IDs to the return value
    }

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
# 10. Main function to process and query
def main():
    # Initialize ChromaDB
    client, dense_collection, sparse_collection = create_chroma_client()
    
    # Check if collections are empty
    if dense_collection.count() == 0:
        print("Processing PDFs and creating embeddings...")
        # Extract text from PDFs
        pdf_texts = extract_text_from_pdfs(DATA_DIR)
        # Create chunks
        chunks = create_chunks(pdf_texts)
        # Store embeddings
        store_embeddings(chunks, dense_collection, sparse_collection)
    
    while True:
        print("\n" + "="*50)
        print("RAG Query System")
        print("="*50)
        print("1. Dense Retrieval Query")
        print("2. Sparse Retrieval Query")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "3":
            break
        
        query = input("Enter your query: ")
        top_k = int(input("Enter number of results to retrieve (default 5): ") or "5")
        
        if choice == "1":
            print("\nPerforming dense retrieval...")
            results = query_dense(query, dense_collection, top_k)
            
            print(f"\nTop {top_k} chunks retrieved:")
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0], 
                results["metadatas"][0],
                results["distances"][0]
            )):
                print(f"\n--- Result {i+1} (distance: {distance:.4f}) ---")
                print(f"Source: {metadata['source']}, Chunk: {metadata['chunk_id']}")
                print(f"Text: {doc[:200]}...")
            
            # Generate response with context
            response = generate_response(query, results["documents"][0])
            print("\n--- Generated Response ---")
            print(response)
            
        elif choice == "2":
            print("\nPerforming sparse retrieval...")
            results = query_sparse(query, sparse_collection, top_k)
            
            print(f"\nTop {top_k} chunks retrieved:")
            for i, (doc, metadata, score) in enumerate(zip(
                results["documents"], 
                results["metadatas"],
                results["scores"]
            )):
                print(f"\n--- Result {i+1} (score: {score:.4f}) ---")
                print(f"Source: {metadata['source']}, Chunk: {metadata['chunk_id']}")
                print(f"Text: {doc[:200]}...")
            
            # Generate response with context
            response = generate_response(query, results["documents"])
            print("\n--- Generated Response ---")
            print(response)

if __name__ == "__main__":
    main()

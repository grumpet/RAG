import sys
import os

# Add the parent directory to the path so we can import functions from both_methods.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from both_methods import (
    create_chroma_client, 
    query_dense, 
    query_sparse,
    generate_response
)

def query_db(query_text: str, method: str = "dense", top_k: int = 5, show_chunks: bool = True):
    """
    Query the database using either dense or sparse retrieval
    
    Parameters:
        query_text (str): The user query
        method (str): 'dense' or 'sparse' retrieval method
        top_k (int): Number of results to retrieve
        show_chunks (bool): Whether to print the retrieved chunks
    
    Returns:
        dict: Results from the query and generated response
    """
    # Initialize ChromaDB client and collections
    client, dense_collection, sparse_collection = create_chroma_client()
    
    # Check which method to use
    if method.lower() == "dense":
        print(f"\nPerforming dense retrieval with top-{top_k}...")
        results = query_dense(query_text, dense_collection, top_k)
        
        if show_chunks:
            print(f"\nTop {top_k} chunks retrieved:")
            for i, (doc, metadata, distance,id) in enumerate(zip(
                results["documents"][0], 
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0]
            )):
                print(f"Chunk ID: {id}")
                print(f"\n--- Result {i+1} (distance: {distance:.4f}) ---")
                print(f"Source: {metadata['source']}, Chunk: {metadata['chunk_id']}")
                print(f"Text: \b{doc}")

        # Generate response with context
        response = generate_response(query_text, results["documents"][0])
        
    elif method.lower() == "sparse":
        print(f"\nPerforming sparse retrieval with top-{top_k}...")
        results = query_sparse(query_text, sparse_collection, top_k)
        
        if show_chunks:
            print(f"\nTop {top_k} chunks retrieved:")
            for i, (doc, metadata, score, id) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["scores"][0],
                results["ids"][0]
            )):
                print(f"Chunk ID: {id}")
                print(f"\n--- Result {i+1} (score: {score:.4f}) ---")
                print(f"Source: {metadata['source']}, Chunk: {metadata['chunk_id']}")
                print(f"Text:\n {doc}")
        
        # Generate response with context
        response = generate_response(query_text, results["documents"][0])
        
    else:
        raise ValueError("Method must be 'dense' or 'sparse'")
    
    if show_chunks:
        print("\n--- Generated Response ---")
        print(response)
        
    return {
        "results": results,
        "response": response,
        "method": method,
        "top_k": top_k
    }

if __name__ == "__main__":
    # Example usage
    print("RAG Query System")
    print("="*50)
    
    query = input("Enter your query: ")
    
    print("\nSelect retrieval method:")
    print("1. Dense Retrieval (Ollama Embeddings)")
    print("2. Sparse Retrieval (BM25)")
    method_choice = input("Enter choice (1 or 2): ")
    
    method = "dense" if method_choice == "1" else "sparse"
    top_k = int(input("Enter number of results to retrieve (default 5): ") or "5")
    
    # Perform query with selected parameters
    result = query_db(query, method=method, top_k=top_k, show_chunks=True)

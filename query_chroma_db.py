import sys
import os
from pprint import pprint
import pandas as pd
from typing import Optional, List, Dict, Any
from tabulate import tabulate  # pip install tabulate

# Add the parent directory to the path so we can import functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from both_methods import create_chroma_client

def inspect_chunks(
    collection_type: str = "dense", 
    source: Optional[str] = None,
    chunk_id: Optional[str] = None,
    limit: int = 10,
    show_embeddings: bool = False,
    output_format: str = "table"  # 'table', 'json', or 'pandas'
):
    """
    Query the database for chunks with detailed information
    
    Parameters:
        collection_type: "dense" or "sparse" collection to query
        source: Filter by source document name
        chunk_id: Filter by specific chunk ID
        limit: Maximum number of chunks to return
        show_embeddings: Whether to include embedding vectors (can be large)
        output_format: Format for displaying results (table, json, or pandas)
    """
    # Initialize ChromaDB client and collections
    client, dense_collection, sparse_collection = create_chroma_client()
    
    # Select collection based on type
    collection = dense_collection if collection_type == "dense" else sparse_collection
    
    # Build query filters
    where_filter = {}
    if source:
        where_filter["source"] = source
    if chunk_id:
        where_filter["chunk_id"] = chunk_id
    
    # Determine what to include in the results
    include = ["documents", "metadatas"]  # Removed "distances" as it is not supported in get()
    if show_embeddings:
        include.append("embeddings")
    
    # Query the collection
    if where_filter:
        results = collection.get(
            where=where_filter,
            include=include,
            limit=limit
        )
    else:
        results = collection.get(
            include=include,
            limit=limit
        )
    
    # Format and display results
    if not results["ids"]:
        print("No chunks found matching the criteria.")
        return
    
    # Prepare data for display
    formatted_data = []
    for i, chunk_id in enumerate(results["ids"]):
        chunk_data = {
            "id": chunk_id,
            "text": results["documents"][i][:200] + "..." if len(results["documents"][i]) > 200 else results["documents"][i],
            "metadata": results["metadatas"][i]
        }
        
        if "embeddings" in results and show_embeddings:
            # Show just a sample of the embeddings to avoid overwhelming output
            embedding = results["embeddings"][i]
            chunk_data["embedding_sample"] = f"[{embedding[0]:.4f}, {embedding[1]:.4f}, ... ({len(embedding)} dimensions)]"
        
        formatted_data.append(chunk_data)
    
    # Display results in requested format
    if output_format == "json":
        # Ensure JSON output is properly formatted
        import json
        print(json.dumps(formatted_data, indent=4))
    elif output_format == "pandas":
        df = pd.DataFrame(formatted_data)
        pd.set_option('display.max_colwidth', 100)
        print(df)
    else:  # table format
        # Convert to a format tabulate can handle
        table_data = []
        for item in formatted_data:
            metadata_str = ", ".join([f"{k}: {v}" for k, v in item["metadata"].items()])
            row = [item["id"], metadata_str, item["text"]]
            if show_embeddings and "embedding_sample" in item:
                row.append(item["embedding_sample"])
            table_data.append(row)
        
        headers = ["ID", "Metadata", "Text"]
        if show_embeddings:
            headers.append("Embedding Sample")
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print(f"\nTotal chunks: {len(results['ids'])}")
    return formatted_data

if __name__ == "__main__":
    print("ChunkDB Inspector")
    print("=" * 50)
    
    # Collection type selection
    print("\nSelect collection type:")
    print("1. Dense Collection")
    print("2. Sparse Collection")
    collection_choice = input("Enter choice (1 or 2): ")
    collection_type = "dense" if collection_choice == "1" else "sparse"
    
    # Optional filters
    source = input("\nFilter by source (press Enter to skip): ")
    source = source if source else None
    
    chunk_id = input("Filter by chunk ID (press Enter to skip): ")
    chunk_id = chunk_id if chunk_id else None
    
    limit = int(input("Maximum number of chunks to display (default 10): ") or "10")
    
    show_embeddings_input = input("Show embedding vectors? (y/n, default n): ")
    show_embeddings = show_embeddings_input.lower() == 'y'
    
    print("\nSelect output format:")
    print("1. Table")
    print("2. JSON")
    print("3. Pandas DataFrame")
    format_choice = input("Enter choice (1, 2, or 3): ")
    
    format_map = {"1": "table", "2": "json", "3": "pandas"}
    output_format = format_map.get(format_choice, "table")
    
    # Run the inspection
    inspect_chunks(
        collection_type=collection_type,
        source=source,
        chunk_id=chunk_id,
        limit=limit,
        show_embeddings=show_embeddings,
        output_format=output_format
    )
    include = ["documents", "metadatas"]
    if show_embeddings:
        include.append("embeddings")
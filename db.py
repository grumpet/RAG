import chromadb
import json
import numpy as np

client = chromadb.PersistentClient(path="chroma_persistent_storage")

# Check that the database is running
print(f"Database heartbeat: {client.heartbeat()}")

# List all collections
collections = client.list_collections()
print(f"\nFound {len(collections)} collections:")

print("==== Collections ====")
for collection in collections:
    print(f"- {collection.name}")
    print(f"  - {collection.count()} documents")

# Print all data from each collection
print("\n==== Detailed Collection Data ====")
for collection in collections:
    print(f"\n📚 Collection: {collection.name}")
    print("-" * 50)
    
    # Get all documents in the collection
    count = collection.count()
    if count == 0:
        print("  Empty collection - no documents found")
        continue
    
    # Ask if user wants to see this collection (could be large)
    view_choice = input(f"View all {count} documents from {collection.name}? (y/n/limit): ")
    
    if view_choice.lower() == 'n':
        continue
    elif view_choice.lower() == 'y':
        limit = count
    else:
        try:
            limit = int(view_choice)
        except ValueError:
            limit = min(10, count)  # Default to 10 if input can't be parsed
    
    # Get documents with limit
    try:
        data = collection.get(limit=limit, include=["documents", "metadatas"])
        
        # Print each document
        for i, doc_id in enumerate(data['ids']):
            print(f"\n📄 Document {i+1}/{len(data['ids'])}")
            print(f"  ID: {doc_id}")
            
            # Print document text (truncate if too long)
            doc_text = data['documents'][i]
            if len(doc_text) > 500:
                print(f"  Text: {doc_text[:500]}... (truncated, {len(doc_text)} characters total)")
            else:
                print(f"  Text: {doc_text}")
            
            # Print metadata if available - fixed check to avoid array truth value error
            if 'metadatas' in data and i < len(data['metadatas']) and data['metadatas'][i] is not None:
                print(f"  Metadata: {json.dumps(data['metadatas'][i], indent=2)}")
            
            print("-" * 30)
        
        # Get embeddings separately if needed
        if input("View embeddings information? (y/n): ").lower() == 'y':
            try:
                embeddings_data = collection.get(limit=limit, include=["embeddings"])
                print("\n📊 Embeddings Information:")
                
                for i, doc_id in enumerate(embeddings_data['ids']):
                    if i < len(embeddings_data.get('embeddings', [])):
                        embedding = embeddings_data['embeddings'][i]
                        if isinstance(embedding, np.ndarray):
                            print(f"  Document {i+1} embedding: [dimension: {embedding.shape[0]}, first 3 values: {embedding[:3].tolist()}...]")
                        else:
                            print(f"  Document {i+1} embedding: [dimension: {len(embedding)}, first 3 values: {embedding[:3]}...]")
                    if i >= 5:  # Limit to first 5 embeddings
                        print("  ... (more embeddings available)")
                        break
            except Exception as e:
                print(f"  Error retrieving embeddings: {str(e)}")
    
    except Exception as e:
        print(f"Error retrieving data from collection: {str(e)}")
        print("Trying alternative approach...")
        
        # Alternative approach without embeddings
        try:
            data = collection.get(limit=limit, include=["documents", "metadatas"])
            
            # Print each document
            for i, doc_id in enumerate(data['ids']):
                print(f"\n📄 Document {i+1}/{len(data['ids'])}")
                print(f"  ID: {doc_id}")
                print(f"  Text: {data['documents'][i][:500]}..." if len(data['documents'][i]) > 500 else data['documents'][i])
                
                # Print metadata if available
                if 'metadatas' in data and i < len(data['metadatas']):
                    metadata = data['metadatas'][i]
                    if metadata is not None:
                        print(f"  Metadata: {json.dumps(metadata, indent=2)}")
                
                print("-" * 30)
        except Exception as e2:
            print(f"Alternative approach also failed: {str(e2)}")
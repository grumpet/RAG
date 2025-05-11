# 📄 Local Document Question Answering with Ollama + ChromaDB

This project builds a local Question-Answering (QA) system that allows you to query your own documents (PDFs and `.txt` files). It uses:

- 🧠 **Ollama** for embeddings and answer generation (LLMs like `llama3`)
- 🔍 **ChromaDB** as a vector database for fast semantic search
- 🐍 **Python** for processing and orchestrating the workflow

---

## 🚀 How It Works

### 1. **Document Loading**
- All `.txt` and `.pdf` files are read from the `./TEE_DATA` directory.
- PDFs are processed using `PyPDF2` to extract readable text.

### 2. **Text Chunking**
- Documents are split into overlapping chunks (default: 1000 characters with 20-character overlap) to ensure contextual coherence and embedding accuracy.

### 3. **Embedding Generation**
- Each chunk is sent to the Ollama API (`/api/embeddings`) to generate a high-dimensional vector (embedding).
- Embeddings capture the semantic meaning of the text.

### 4. **ChromaDB Storage**
- Chunks and their embeddings are stored in a persistent ChromaDB collection (`document_qa_collection`).
- This allows for fast similarity-based retrieval later.

### 5. **Semantic Search**
- When a user inputs a question, it’s embedded in the same way.
- ChromaDB is queried for the most semantically similar chunks.

### 6. **Answer Generation**
- The top-matching chunks are provided as context to the Ollama Chat API (`/api/chat`).
- The model returns a concise answer based only on the given context.

---

## 🛠️ Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) running locally (`http://localhost:11434`)
- ChromaDB
- PyPDF2
- requests
- python-dotenv

Install dependencies:

```bash
pip install -r requirements.txt



ollama pull nomic-embed-text
ollama pull llama3


# Start interactive mode
python query_collections.py --interactive

# Direct query to news articles only
python query_collections.py --question "What is Databricks?" --repo news

# Query both collections with more results
python query_collections.py --question "Explain TEE technology" --results 5 --show-chunks

# Use a specific model
python query_collections.py --question "What are the latest developments?" --model "mistral"
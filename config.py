import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
DATA_DIR = "data/"
VECTOR_DB_PATH = "vectorstore/faiss_index"
BM25_PERSIST_PATH = "vectorstore/bm25_retriever.pkl"

# Chunking Strategy (Optimized for Context Retention)
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200  # tokens

# Retrieval Config
TOP_K_VECTORS = 5
# Hybrid Search Weights: [Sparse (BM25), Dense (FAISS)]
# Adjust these based on whether you prefer keyword matching or semantic meaning
HYBRID_WEIGHTS = [0.5, 0.5]

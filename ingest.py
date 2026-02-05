import os
import pickle
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import config


def load_documents():
    """
    Loads PDF documents from the /data directory.
    """
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
        print(f"⚠️ Created {config.DATA_DIR} folder. Please put your PDFs there.")
        return []

    print(f"📂 Loading documents from {config.DATA_DIR}...")
    loader = DirectoryLoader(config.DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents.")
    return documents


def split_documents(documents):
    """
    Implements the Chunking Strategy: 1000 tokens with 200 overlap.
    Also adds dummy Access Control Lists (ACLs) to metadata for interview demo purposes.
    """
    print("✂️ Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Simulate tagging metadata for Security/ACLs demo
    for i, chunk in enumerate(chunks):
        # Even numbered chunks are "Public", Odd are "Internal"
        chunk.metadata["access_level"] = "public" if i % 2 == 0 else "internal"

    print(f"✅ Created {len(chunks)} chunks.")
    return chunks


def create_vector_db(chunks):
    """
    Creates both a FAISS index (Dense) and BM25 Retriever (Sparse).
    """
    embeddings = OpenAIEmbeddings()
    os.makedirs(os.path.dirname(config.VECTOR_DB_PATH), exist_ok=True)

    # 1. Create Dense Vector Store (FAISS)
    print("🧠 Embedding chunks and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(config.VECTOR_DB_PATH)

    # 2. Create Sparse Retriever (BM25) for Hybrid Search
    print("🧮 Building BM25 Keyword index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.TOP_K_VECTORS

    # Save BM25 to disk (pickle)
    os.makedirs(os.path.dirname(config.BM25_PERSIST_PATH), exist_ok=True)
    with open(config.BM25_PERSIST_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print("✅ Indexing complete. Saved to /vectorstore.")


if __name__ == "__main__":
    docs = load_documents()
    if docs:
        splits = split_documents(docs)
        create_vector_db(splits)
    else:
        print("❌ No documents found. Please add PDFs to the 'data/' folder.")

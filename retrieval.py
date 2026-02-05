import pickle
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import config


def load_retrievers():
    """
    Loads the FAISS index and BM25 retriever to create a Hybrid Ensemble.
    """
    embeddings = OpenAIEmbeddings()

    # Load Vector DB (FAISS)
    vectorstore = FAISS.load_local(
        config.VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    # Configure FAISS retriever
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K_VECTORS})

    # Load Keyword DB (BM25)
    with open(config.BM25_PERSIST_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)

    # Create Ensemble (Hybrid Search)
    # This combines results using weighted Reciprocal Rank Fusion
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=config.HYBRID_WEIGHTS
    )

    return ensemble_retriever


def get_rag_chain(retriever):
    """
    Initializes the RAG chain with the Hybrid Retriever.
    """
    # Using GPT-3.5-turbo with temperature 0 for factual accuracy
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Explicitly specify output key for chain compatibility
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True  # Crucial for showing citations
    )

    return qa_chain

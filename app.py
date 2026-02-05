import streamlit as st
from retrieval import load_retrievers, get_rag_chain
import os

# Page Configuration
st.set_page_config(page_title="AI RAG Powered Chat Assistant")

st.title("🤖 AI RAG Powered Chat Assistant")
st.markdown("Production-grade RAG pipeline with **Hybrid Search (FAISS + BM25)**")

# Check if vector store exists
if not os.path.exists("vectorstore"):
    st.error("⚠️ No Index Found! Please run `python ingest.py` first.")
    st.stop()

# Initialize Session State
if "chain" not in st.session_state:
    with st.spinner("Loading Vector Indices & Models..."):
        try:
            retriever = load_retrievers()
            st.session_state.chain = get_rag_chain(retriever)
            st.success("System Ready!")
        except Exception as e:
            st.error(f"Error loading system: {e}")
            st.stop()
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking (Retrieving Context)..."):
            try:
                response = st.session_state.chain.invoke({"question": prompt})
                answer = response["answer"]
                sources = response["source_documents"]

                # Format Answer
                st.markdown(answer)

                # Show Citations (Trust but Verify)
                with st.expander("📚 View Source Citations"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.caption(f"...{doc.page_content[:200]}...")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"An error occurred: {e}")

"""
RAG Knowledge Base - Starter Template
FAMNIT AI Course - Day 3

A simple Retrieval-Augmented Generation (RAG) app built with
Streamlit, LangChain, and ChromaDB. No API keys needed!

Instructions:
  1. Replace the DOCUMENTS list below with your own texts
  2. Update the app title and description
  3. Run locally:  streamlit run app.py
  4. Deploy to Render (see assignment instructions)
"""

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="My RAG Knowledge Base",
    page_icon="🔍",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────
# YOUR DOCUMENTS — Replace these with your own topic!
# Each string is one "document" that will be chunked, embedded, and
# stored in the vector database for semantic search.
# ──────────────────────────────────────────────────────────────────────
DOCUMENTS = [
    """Python is a high-level, general-purpose programming language. Its design
philosophy emphasizes code readability with the use of significant
indentation. Python is dynamically typed and garbage-collected. It supports
multiple programming paradigms, including structured, object-oriented and
functional programming.""",

    """Machine learning is a subset of artificial intelligence that focuses on
building systems that learn from data. Instead of being explicitly
programmed, these systems identify patterns in data and make decisions with
minimal human intervention. Common types include supervised learning,
unsupervised learning, and reinforcement learning.""",

    """Streamlit is an open-source Python framework for building interactive web
applications for data science and machine learning. It allows developers to
create apps with just a few lines of Python code, without needing to know
HTML, CSS, or JavaScript. Streamlit apps can be deployed easily to the cloud.""",

    """ChromaDB is an open-source vector database designed for AI applications.
It stores embeddings (numerical representations of data) and allows
efficient similarity search. ChromaDB is lightweight, runs locally, and
integrates well with LangChain and other AI frameworks.""",

    """Retrieval-Augmented Generation (RAG) is a technique that enhances AI
responses by first retrieving relevant information from a knowledge base,
then using that information to generate accurate answers. RAG combines the
power of search (retrieval) with language generation.""",
]

# ──────────────────────────────────────────────────────────────────────
# Cached heavy resources (loaded once, reused across reruns)
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Building vector database...")
def build_vector_store(_documents: tuple):
    """Chunk documents, embed them, and store in ChromaDB."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in _documents:
        chunks.extend(splitter.split_text(doc))

    embeddings = load_embedding_model()

    # --- Store in ChromaDB ---
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="knowledge_base",
    )
    return vector_store, chunks


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("My RAG App")
page = st.sidebar.radio("Navigate", ["Home", "Search", "Explore Chunks"])

# ──────────────────────────────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────────────────────────────
if page == "Home":
    st.title("My RAG Knowledge Base")
    st.markdown("""
    Welcome! This app lets you **search documents by meaning**, not just keywords.

    ### How it works
    1. **Documents** are split into small chunks
    2. Each chunk is converted to an **embedding** (a vector of numbers)
    3. Chunks are stored in a **vector database** (ChromaDB)
    4. When you search, your query is embedded and compared to all chunks
    5. The most **semantically similar** chunks are returned

    ### Get started
    - Go to **Search** to ask questions
    - Go to **Explore Chunks** to see how documents are split

    ---
    *Built with Streamlit, LangChain, and ChromaDB*
    """)

    st.info(f"Knowledge base contains **{len(DOCUMENTS)} documents**.")


# ──────────────────────────────────────────────────────────────────────
# SEARCH PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Search":
    st.title("Semantic Search")
    st.markdown("Ask a question and the app will find the most relevant chunks from the knowledge base.")

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    query = st.text_input(
        "Your question",
        placeholder="e.g. What is RAG?",
    )
    num_results = st.slider("Number of results", 1, 10, 3)

    if query:
        with st.spinner("Searching..."):
            results = vector_store.similarity_search_with_score(query, k=num_results)

        st.subheader(f"Top {len(results)} results")
        for i, (doc, score) in enumerate(results, 1):
            # ChromaDB returns distance; lower = more similar
            similarity = max(0, 1 - score)  # rough conversion
            with st.container():
                st.markdown(f"**Result {i}** — relevance: `{similarity:.2f}`")
                st.markdown(f"> {doc.page_content}")
                st.divider()

    st.markdown("---")
    st.caption("Powered by all-MiniLM-L6-v2 embeddings + ChromaDB")


# ──────────────────────────────────────────────────────────────────────
# EXPLORE CHUNKS PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Explore Chunks":
    st.title("Explore Chunks")
    st.markdown("See how your documents are split into chunks by the recursive text splitter.")

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    st.metric("Total chunks", len(chunks))

    lengths = [len(c) for c in chunks]
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg chunk size", f"{np.mean(lengths):.0f} chars")
    col2.metric("Min chunk size", f"{min(lengths)} chars")
    col3.metric("Max chunk size", f"{max(lengths)} chars")

    st.subheader("All chunks")
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"Chunk {i} ({len(chunk)} chars)"):
            st.text(chunk)

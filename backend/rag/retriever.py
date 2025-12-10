# backend/rag/retriever.py
from langchain_core.vectorstores import VectorStoreRetriever

from backend.memory.vector_store import load_vector_store


def get_retriever(k: int = 4) -> VectorStoreRetriever:
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

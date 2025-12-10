# backend/memory/vector_store.py
import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from backend.config import VECTOR_DB_PATH


def get_embeddings():
    # Simple free model; no API key required
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vector_store(docs: List[Document]) -> None:
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    path = Path(VECTOR_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(VECTOR_DB_PATH)
    print(f"✅ Indexed {len(docs)} chunks into FAISS at {VECTOR_DB_PATH}")


def load_vector_store() -> FAISS:
    if not Path(VECTOR_DB_PATH).exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_DB_PATH}. "
            "Run `python -m backend.rag.index` first."
        )

    embeddings = get_embeddings()
    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # safe since it's your own file
    )

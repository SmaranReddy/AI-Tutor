from backend.rag.loader import load_pdfs
from backend.rag.chunker import chunk_documents
from backend.memory.vector_store import build_vector_store

def index_pdfs(pdf_dir: str = "data"):
    docs = load_pdfs(pdf_dir)
    chunks = chunk_documents(docs)
    build_vector_store(chunks)
    print(f"✅ Indexed {len(chunks)} chunks into FAISS")


if __name__ == "__main__":
    index_pdfs()

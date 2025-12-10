from backend.rag.loader import load_pdfs
from backend.rag.chunker import chunk_documents

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

docs = load_pdfs(str(DATA_DIR))

chunks = chunk_documents(docs)

print("Original docs:", len(docs))
print("Total chunks:", len(chunks))
print("Sample chunk:\n", chunks[0].page_content[:300])
print("Metadata:", chunks[0].metadata)

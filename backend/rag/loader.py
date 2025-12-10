from typing import List
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from pathlib import Path


def load_pdfs(pdf_dir: str) -> List[Document]:
    """
    Load all PDFs from a directory and return LangChain Documents.
    """
    pdf_dir = Path(pdf_dir)
    documents = []

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    for pdf_path in pdf_dir.glob("*.pdf"):
        reader = PdfReader(pdf_path)
        full_text = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        if full_text.strip():
            documents.append(
                Document(
                    page_content=full_text,
                    metadata={
                        "source": pdf_path.name
                    }
                )
            )

    return documents

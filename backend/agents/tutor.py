# backend/agents/tutor.py
from typing import Dict, Any, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.rag.retriever import get_retriever


SYSTEM_PROMPT = """
You are an AI tutor helping a student understand concepts from their study materials.

Rules:
- ONLY use the provided context.
- If the answer is not in the context, say "I don't know based on the provided material."
- Explain clearly and step by step.
- Be concise but helpful.
"""


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.2,
    )


def format_context(docs: List[Document]) -> str:
    """Combine retrieved documents into a single context string."""
    return "\n\n".join(
        f"[Source: {d.metadata.get('source')}, Chunk: {d.metadata.get('chunk_id')}]\n"
        f"{d.page_content}"
        for d in docs
    )


def ask_tutor(question: str, k: int = 4) -> Dict[str, Any]:
    retriever = get_retriever(k=k)
    docs: List[Document] = retriever.invoke(question)

    context = format_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Question:\n{question}\n\n"
                "Context:\n{context}\n\n"
                "Answer:",
            ),
        ]
    )

    llm = get_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    # Extract structured sources
    sources = [
        {
            "source": d.metadata.get("source"),
            "chunk_id": d.metadata.get("chunk_id"),
        }
        for d in docs
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }

def build_reexplain_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a patient tutor. "
                "Explain the concept again using a DIFFERENT approach "
                "than before. Use simple language and intuition."
            ),
            (
                "human",
                """
Context:
{context}

Question:
{question}

Student misunderstanding:
{diagnosis}

Re-explain the concept in a clearer and simpler way.
"""
            ),
        ]
    )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.3,
    )

    return prompt | llm

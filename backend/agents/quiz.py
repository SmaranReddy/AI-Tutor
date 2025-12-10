# backend/agents/quiz.py
from typing import Dict, Any, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.rag.retriever import get_retriever


QUIZ_PROMPT = """
You are an AI tutor creating quizzes from study material.

Rules:
- Use ONLY the provided context.
- Create {num_questions} questions.
- Mix conceptual and factual questions.
- Provide answers with short explanations.
- If context is insufficient, say so.

Format strictly as:

Q1. ...
A1. ...
Explanation: ...

Q2. ...
A2. ...
Explanation: ...
"""


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.3,
    )


def generate_quiz(topic: str, num_questions: int = 5, k: int = 4) -> Dict[str, Any]:
    retriever = get_retriever(k=k)
    docs: List[Document] = retriever.invoke(topic)

    if not docs:
        return {
            "quiz": "I don't know based on the provided material.",
            "sources": [],
        }

    context = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUIZ_PROMPT),
            (
                "human",
                "Topic: {topic}\n\nContext:\n{context}\n\nQuiz:",
            ),
        ]
    )

    llm = get_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "topic": topic,
            "context": context,
            "num_questions": num_questions,
        }
    )

    sources = [
        {
            "source": d.metadata.get("source"),
            "chunk_id": d.metadata.get("chunk_id"),
        }
        for d in docs
    ]

    return {
        "quiz": response.content,
        "sources": sources,
    }

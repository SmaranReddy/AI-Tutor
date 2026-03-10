# backend/agents/quiz.py
from typing import Dict, Any, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.rag.retriever import get_retriever


QUIZ_PROMPT = """
You are an AI tutor generating quizzes STRICTLY from the provided context.

Rules:
- Use ONLY the provided context.
- Generate EXACTLY:
  - 3 MCQs (4 options each, one correct)
  - 2 descriptive questions (conceptual)
- Provide answers clearly.
- Do NOT invent information.
- If context is insufficient, say so.

Output format EXACTLY:

MCQ 1:
Question: ...
A) ...
B) ...
C) ...
D) ...
Answer: <A/B/C/D>

MCQ 2:
Question: ...
A) ...
B) ...
C) ...
D) ...
Answer: <A/B/C/D>

MCQ 3:
Question: ...
A) ...
B) ...
C) ...
D) ...
Answer: <A/B/C/D>

DESCRIPTIVE 1:
Question: ...
Answer: ...

DESCRIPTIVE 2:
Question: ...
Answer: ...
"""


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.3,
    )


def _separate_quiz_and_answers(text: str) -> Dict[str, Any]:
    """
    Splits quiz text into:
    - visible quiz (no answers)
    - answer key (MCQs + descriptive)
    """

    visible_lines = []
    mcq_answers = []
    descriptive_answers = []

    current_section = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("Answer:"):
            mcq_answers.append(stripped.replace("Answer:", "").strip())
            continue

        if stripped.startswith("DESCRIPTIVE"):
            current_section = "DESCRIPTIVE"
            visible_lines.append(line)
            continue

        if stripped.startswith("Answer:") and current_section == "DESCRIPTIVE":
            descriptive_answers.append(stripped.replace("Answer:", "").strip())
            continue

        if stripped.startswith("Answer:") is False:
            visible_lines.append(line)

        # capture descriptive answers separately
        if current_section == "DESCRIPTIVE" and stripped.startswith("Answer:"):
            descriptive_answers.append(
                stripped.replace("Answer:", "").strip()
            )

    return {
        "visible_quiz": "\n".join(visible_lines).strip(),
        "mcq_answers": mcq_answers,
        "descriptive_answers": descriptive_answers,
    }


def generate_quiz(topic: str, k: int = 4) -> Dict[str, Any]:
    """
    Generates a mixed quiz (3 MCQs + 2 descriptive) using RAG.
    Returns:
    - quiz (student visible)
    - answer_key (internal)
    - sources
    """

    retriever = get_retriever(k=k)
    docs: List[Document] = retriever.invoke(topic)

    if not docs:
        return {
            "quiz": "I don't know based on the provided material.",
            "answer_key": {},
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
        }
    )

    parsed = _separate_quiz_and_answers(response.content)

    sources = [
        {
            "source": d.metadata.get("source"),
            "chunk_id": d.metadata.get("chunk_id"),
        }
        for d in docs
    ]

    return {
        "quiz": parsed["visible_quiz"],     # ✅ shown to student
        "answer_key": {                     # ✅ internal use only
            "mcqs": parsed["mcq_answers"],
            "descriptive": parsed["descriptive_answers"],
        },
        "sources": sources,
    }

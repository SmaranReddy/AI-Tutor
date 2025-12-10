from typing import Dict, List, Literal

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from backend.config import GROQ_API_KEY


UnderstandingVerdict = Literal[
    "UNDERSTOOD",
    "PARTIALLY_UNDERSTOOD",
    "NOT_UNDERSTOOD",
]


def build_diagnostic_llm() -> ChatGroq:
    """
    Low-temperature LLM for strict, evidence-based diagnosis.
    """
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.0,
    )


def diagnose_understanding(
    question: str,
    student_answer: str,
    context_chunks: List[str],
) -> Dict:
    """
    Diagnose student understanding using STRICT rubric.
    """

    # ✅ Guardrail: insufficient answer
    if not student_answer or len(student_answer.strip()) < 10:
        return {
            "verdict": "NOT_UNDERSTOOD",
            "analysis": (
                "The answer is too short to evaluate understanding. "
                "Please explain the concept in your own words."
            ),
        }

    context = "\n\n".join(context_chunks)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an extremely strict examiner. "
                "Do NOT give partial credit unless the student explicitly references "
                "key ideas present in the context. "
                "If the answer is vague, generic, or lacks grounding, "
                "you MUST respond with NOT_UNDERSTOOD."
            ),
            (
                "human",
                """
Context (authoritative source):
{context}

Question:
{question}

Student answer:
{student_answer}

Grading rules (VERY IMPORTANT):
- UNDERSTOOD:
  Student correctly explains the concept using ideas from the context.
- PARTIALLY_UNDERSTOOD:
  Student mentions a correct concept from the context BUT misses justification or depth.
- NOT_UNDERSTOOD:
  Student gives vague, generic, buzzword-based, or unsupported statements.

You MUST choose exactly one label.

Output format:
VERDICT: <UNDERSTOOD | PARTIALLY_UNDERSTOOD | NOT_UNDERSTOOD>
REASON: <short explanation referencing the context>
"""
            ),
        ]
    )

    llm = build_diagnostic_llm()
    chain = prompt | llm

    response: AIMessage = chain.invoke(
        {
            "context": context,
            "question": question,
            "student_answer": student_answer,
        }
    )

    text = response.content.upper()

    # ✅ STRICT verdict parsing
    if "VERDICT: UNDERSTOOD" in text:
        verdict: UnderstandingVerdict = "UNDERSTOOD"
    elif "VERDICT: PARTIALLY_UNDERSTOOD" in text:
        verdict = "PARTIALLY_UNDERSTOOD"
    else:
        verdict = "NOT_UNDERSTOOD"

    return {
        "verdict": verdict,
        "analysis": response.content.strip(),
    }

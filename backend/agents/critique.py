from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from backend.config import GROQ_API_KEY


def critique_explanation(context: str, explanation: str) -> bool:
    """
    Returns True if explanation is acceptable, False if it should be rewritten.
    """

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.0,  # strict judgment
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict academic reviewer.\n"
                "Reject explanations that are vague, generic, or lack grounding "
                "in the provided context."
            ),
            (
                "human",
                """
Context:
{context}

Explanation:
{explanation}

Answer STRICTLY with:
✅ ACCEPT
❌ REJECT
""",
            ),
        ]
    )

    response = (prompt | llm).invoke(
        {
            "context": context,
            "explanation": explanation,
        }
    )

    return "✅ ACCEPT" in response.content

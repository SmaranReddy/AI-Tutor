# backend/tutor_api.py
from typing import List, Dict, Any, Tuple
from backend.rag.retriever import get_retriever
from backend.agents.tutor import build_reexplain_chain
from backend.agents.critique import critique_explanation
from backend.agents.quiz import generate_quiz
from backend.pipeline import grade_descriptive  # uses your strict grader
import re

def get_context_for_topic(topic: str, k: int = 3) -> List[str]:
    """
    Returns top-k context chunks (page_content strings) for a given topic.
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(topic)
    return [d.page_content for d in docs]

def explain_topic(context_chunks: List[str], question: str) -> Dict[str, Any]:
    """
    Produce explanation for question using your tutor chain and run critique/regeneration
    up to 2 times (the chain itself is deterministic per your config).
    Returns { explanation: str, attempts: int, accepted: bool }
    """
    chain = build_reexplain_chain()
    joined_context = "\n\n".join(context_chunks)
    attempts = 0
    accepted = False
    explanation = ""

    while attempts < 3:
        attempts += 1
        response = chain.invoke({
            "context": joined_context,
            "question": question,
            "diagnosis": ""  # UI-driven; not required here
        })
        explanation = response.content
        # run critique
        is_good = critique_explanation(
            context=joined_context,
            explanation=explanation,
        )
        if is_good:
            accepted = True
            break
        # otherwise loop to regenerate (chain is called again)
    return {"explanation": explanation, "attempts": attempts, "accepted": accepted}

# helper to parse quiz (same parsing used in pipeline; simple fallback)
MCQ_RE = re.compile(
    r"MCQ\s+(\d+):\s*[\r\n]+Question:\s*(.+?)\r?\nA\)\s*(.+?)\r?\nB\)\s*(.+?)\r?\nC\)\s*(.+?)\r?\nD\)\s*(.+?)\r?\nAnswer:\s*([A-D])",
    re.IGNORECASE | re.DOTALL
)
DESC_RE = re.compile(
    r"DESCRIPTIVE\s+(\d+):\s*[\r\n]+Question:\s*(.+?)\r?\nAnswer:\s*(.+?)(?=\r?\n\r?\n|$)",
    re.IGNORECASE | re.DOTALL
)

def parse_quiz_text(quiz_text: str) -> Dict[str, Any]:
    mcqs = []
    for m in MCQ_RE.finditer(quiz_text):
        idx = int(m.group(1))
        mcqs.append({
            "index": idx,
            "question": m.group(2).strip(),
            "options": {"A": m.group(3).strip(), "B": m.group(4).strip(), "C": m.group(5).strip(), "D": m.group(6).strip()},
            "answer": m.group(7).strip().upper()
        })
    descs = []
    for m in DESC_RE.finditer(quiz_text):
        idx = int(m.group(1))
        descs.append({
            "index": idx,
            "question": m.group(2).strip(),
            "answer": m.group(3).strip()
        })
    return {"mcqs": mcqs, "descriptive": descs}

def get_quiz_for_topic(topic: str, k: int = 4) -> Dict[str, Any]:
    """
    Returns the quiz text and parsed structure.
    """
    result = generate_quiz(topic, k=k)
    quiz_text = result.get("quiz", "")
    parsed = parse_quiz_text(quiz_text)
    return {"raw": quiz_text, "parsed": parsed, "sources": result.get("sources", [])}

def grade_mcq_answers(parsed_quiz: Dict[str, Any], answers: List[str]) -> Tuple[int, int]:
    """
    answers is list of user's choices like ['A','B','C']
    Returns (correct_count, total)
    """
    mcqs = parsed_quiz.get("mcqs", [])
    total = len(mcqs)
    correct = 0
    for i, mcq in enumerate(mcqs):
        if i < len(answers) and answers[i].strip().upper() in {"A","B","C","D"}:
            if answers[i].strip().upper() == mcq["answer"]:
                correct += 1
    return correct, total

def grade_descriptive_answers(context_chunks: List[str], parsed_quiz: Dict[str, Any], student_answers: List[str]) -> Tuple[int,int]:
    """
    Grades descriptive questions using your strict grade_descriptive function (returns CORRECT/INCORRECT)
    Returns (correct_count, total)
    """
    descs = parsed_quiz.get("descriptive", [])
    total = len(descs)
    correct = 0
    context = "\n\n".join(context_chunks)
    for i, desc in enumerate(descs):
        ref = desc.get("answer","")
        student = student_answers[i] if i < len(student_answers) else ""
        if grade_descriptive(context=context, question=desc["question"], reference_answer=ref, student_answer=student):
            correct += 1
    return correct, total

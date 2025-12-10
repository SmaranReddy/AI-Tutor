from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from backend.rag.retriever import get_retriever
from backend.agents.diagnostic import diagnose_understanding
from backend.agents.tutor import build_reexplain_chain
from backend.agents.critique import critique_explanation


# =========================
# Tutor State
# =========================
class TutorState(TypedDict):
    question: str
    student_answer: str
    context_chunks: List[str]
    diagnosis: str
    verdict: str
    retry_count: int


# =========================
# Nodes
# =========================

def retrieve_context_node(state: TutorState) -> TutorState:
    retriever = get_retriever(k=3)
    docs = retriever.invoke(state["question"])
    state["context_chunks"] = [d.page_content for d in docs]
    return state


def ask_question_node(state: TutorState) -> TutorState:
    print("\n📘 Tutor Question:")
    print(state["question"])

    state["student_answer"] = input("\nYour answer:\n> ")
    return state


def diagnose_node(state: TutorState) -> TutorState:
    result = diagnose_understanding(
        question=state["question"],
        student_answer=state["student_answer"],
        context_chunks=state["context_chunks"],
    )

    state["verdict"] = result["verdict"]
    state["diagnosis"] = result["analysis"]

    print(f"\n🧠 Diagnostic Verdict: {state['verdict']}")
    return state


def reexplain_node(state: TutorState) -> TutorState:
    print("\n🔁 Re-explaining the concept:\n")

    chain = build_reexplain_chain()
    response = chain.invoke(
        {
            "context": "\n\n".join(state["context_chunks"]),
            "question": state["question"],
            "diagnosis": state["diagnosis"],
        }
    )

    # ✅ Always show explanation
    print(response.content)

    # ✅ Critique after showing explanation
    is_good = critique_explanation(
        context="\n\n".join(state["context_chunks"]),
        explanation=response.content,
    )

    state["retry_count"] += 1

    # ✅ Retry at most twice
    if not is_good and state["retry_count"] < 2:
        print("\n⚠️ Explanation could be improved. Trying again...\n")
        return state

    print("\n✅ Explanation accepted. Let's move on.\n")
    return state


# =========================
# Routing Logic
# =========================

def route_after_diagnosis(state: TutorState):
    if state["verdict"] == "UNDERSTOOD":
        print("\n✅ Concept understood. Moving forward.\n")
        return END
    return "reexplain"


# =========================
# Graph Definition
# =========================

graph = StateGraph(TutorState)

graph.add_node("retrieve_context", retrieve_context_node)
graph.add_node("ask_question", ask_question_node)
graph.add_node("diagnose", diagnose_node)
graph.add_node("reexplain", reexplain_node)

graph.set_entry_point("retrieve_context")

graph.add_edge("retrieve_context", "ask_question")
graph.add_edge("ask_question", "diagnose")

graph.add_conditional_edges(
    "diagnose",
    route_after_diagnosis,
    {
        "reexplain": "reexplain",
        END: END,
    },
)

graph.add_edge("reexplain", "ask_question")

tutor_graph = graph.compile()

from backend.rag.retriever import get_retriever
from backend.agents.diagnostic import diagnose_understanding

if __name__ == "__main__":
    question = "Why are graph neural networks suitable for molecular generation?"

    print("\nQUESTION:")
    print(question)

    print("\nType your answer below:\n")
    student_answer = input("> ")

    retriever = get_retriever(k=3)

    # ✅ Correct modern call
    docs = retriever.invoke(question)
    context_chunks = [d.page_content for d in docs]

    result = diagnose_understanding(
        question=question,
        student_answer=student_answer,
        context_chunks=context_chunks,
    )

    print("\n=== DIAGNOSTIC RESULT ===")
    print("Verdict:", result["verdict"])
    print("\nExplanation:\n", result["analysis"])

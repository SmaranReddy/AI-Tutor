from backend.agents.quiz import generate_quiz

if __name__ == "__main__":
    result = generate_quiz(
        topic="graph-based molecular generation",
        num_questions=3,
        k=3,
    )

    print("\nQUIZ:\n")
    print(result["quiz"])

    print("\nSOURCES:")
    for s in result["sources"]:
        print(s)

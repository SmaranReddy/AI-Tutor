from backend.agents.tutor import ask_tutor

if __name__ == "__main__":
    q = "What is graph-based molecular generation?"
    result = ask_tutor(q, k=3)

    print("\nANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(s)

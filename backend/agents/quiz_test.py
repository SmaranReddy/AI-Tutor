from backend.agents.quiz import generate_quiz

result = generate_quiz(
    topic="Why are graph neural networks suitable for molecular generation?"
)

print(result["quiz"])
print("\nSources:", result["sources"])

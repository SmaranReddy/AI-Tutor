from backend.rag.retriever import retrieve_docs

query = "Explain graph-based molecular generation"

docs = retrieve_docs(query, k=3)

print(f"\nQuery: {query}\n")
print(f"Retrieved docs: {len(docs)}\n")

for i, doc in enumerate(docs):
    print(f"--- Doc {i+1} ---")
    print(doc.page_content[:300])
    print("Metadata:", doc.metadata)
    print()

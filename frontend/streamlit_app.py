import streamlit as st
import sys, os

# Make backend importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pipeline import tutor_graph
from backend.agents.quiz import generate_quiz
from backend.rag.loader import load_pdfs
from backend.rag.chunker import chunk_documents
from backend.rag.index import build_vector_store
from backend.rag.retriever import get_retriever
from backend.config import GROQ_API_KEY, GROQ_MODEL
from langchain_groq import ChatGroq
import traceback


# ==============================
# PAGE SETUP
# ==============================
st.set_page_config(page_title="AI Tutor", layout="wide")
st.title("📘 AI Tutor – PDF Learning Assistant")


# ==============================
# SESSION STATE INIT
# ==============================
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "parsed_quiz" not in st.session_state:
    st.session_state.parsed_quiz = None

if "mcq_answers" not in st.session_state:
    st.session_state.mcq_answers = ["", "", ""]

if "desc_answers" not in st.session_state:
    st.session_state.desc_answers = ["", ""]

if "context_chunks" not in st.session_state:
    st.session_state.context_chunks = []


# ==============================
# LOAD PDF + BUILD VECTOR STORE
# ==============================
st.subheader("📄 Upload PDF to Begin")

uploaded_pdf = st.file_uploader("Upload your study PDF", type=["pdf"])

if uploaded_pdf:
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    pdf_path = os.path.join(data_dir, uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    docs = load_pdfs(data_dir)
    chunks = chunk_documents(docs)
    build_vector_store(chunks)

    st.session_state.pdf_loaded = True
    st.success("PDF processed successfully! You can now ask questions or take a quiz.")


st.divider()

# ==============================
# ASK A QUESTION
# ==============================
st.subheader("❓ Ask something from the PDF")

user_q = st.text_input("Your question:", placeholder="Ask me anything from your uploaded PDF...")


if st.button("Ask"):
    if not st.session_state.pdf_loaded:
        st.error("Upload a PDF first.")
    elif user_q.strip() == "":
        st.error("Please enter a question.")
    else:
        try:
            output = tutor_graph.invoke({"question": user_q})
            st.session_state.context_chunks = output.get("context_chunks", [])

            st.write("### 🧠 Tutor Response")
            st.write(output.get("diagnosis", ""))

        except Exception as e:
            st.error("Error during tutoring session.")
            st.code(traceback.format_exc())


# ==============================
# SUMMARY BUTTON
# ==============================
if st.session_state.pdf_loaded:
    if st.button("📘 Summarize PDF"):
        retriever = get_retriever(k=6)
        ctx = "\n\n".join(c.page_content for c in retriever.invoke("overview summary"))

        llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
        prompt = f"Summarize the material:\n\n{ctx}"

        summary = llm.invoke(prompt)
        st.write("## 📄 PDF Summary")
        st.write(summary.content)


st.divider()

# ==============================
# QUIZ GENERATION
# ==============================
st.subheader("📝 Take a Quiz")

if st.button("Generate Quiz"):
    if not st.session_state.pdf_loaded:
        st.error("Upload a PDF first.")
    elif user_q.strip() == "":
        st.error("Ask at least one question before taking a quiz!")
    else:
        quiz = generate_quiz(user_q)
        st.session_state.quiz = quiz["quiz"]

        from backend.pipeline import parse_quiz
        st.session_state.parsed_quiz = parse_quiz(quiz["quiz"])

        st.success("Quiz generated!")


# ==============================
# SHOW QUIZ + ANSWER
# ==============================
if st.session_state.parsed_quiz:
    st.write("### 📘 Quiz")

    mcqs = st.session_state.parsed_quiz["mcqs"]
    descs = st.session_state.parsed_quiz["descriptive"]

    # MCQs
    for i, mcq in enumerate(mcqs):
        st.write(f"**MCQ {i+1}:** {mcq['question']}")
        st.session_state.mcq_answers[i] = st.radio(
            f"Choose answer for MCQ {i+1}:",
            ["A", "B", "C", "D"],
            key=f"mcq_{i}",
        )

    # Descriptive Questions
    for i, desc in enumerate(descs):
        st.write(f"**Descriptive {i+1}: {desc['question']}**")
        st.session_state.desc_answers[i] = st.text_area(
            f"Your answer for descriptive {i+1}:",
            key=f"desc_{i}",
            height=80
        )

    if st.button("Submit Quiz"):
        from backend.pipeline import grade_descriptive

        mcq_correct = sum(
            st.session_state.mcq_answers[i] == mcq["answer"]
            for i, mcq in enumerate(mcqs)
        )

        context = "\n\n".join(st.session_state.context_chunks)

        desc_correct = sum(
            grade_descriptive(context, desc["question"], desc["answer"], st.session_state.desc_answers[i])
            for i, desc in enumerate(descs)
        )

        passed = mcq_correct >= 2 and desc_correct >= 1

        if passed:
            st.success(f"🎉 Quiz Passed! ({mcq_correct}/3 MCQs, {desc_correct}/2 descriptive)")
        else:
            st.error(f"❌ Quiz Failed ({mcq_correct}/3 MCQs, {desc_correct}/2 descriptive)")


st.write("---")
st.caption("AI Tutor • Powered by RAG + Streamlit + LangGraph + Groq")

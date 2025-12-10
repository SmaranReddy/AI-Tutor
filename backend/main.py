from fastapi import FastAPI
from schemas import ChatRequest, ChatResponse
from pipeline import tutor_graph

app = FastAPI(title="AI Tutor")

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = {
        "question": req.question,
        "topic": req.topic,
        "user_id": req.user_id
    }
    result = tutor_graph.invoke(state)
    return ChatResponse(answer=result["answer"])

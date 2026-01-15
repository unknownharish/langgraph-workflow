from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from app.graph import graph

app = FastAPI(title="Gemini API")

class AskRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Welcome to the LangGraph API"}

@app.post("/ask")
def ask(req: AskRequest):
    result = graph.invoke({
        "question": req.question
    })
    return {"answer": result["answer"],"description": result["description"]}

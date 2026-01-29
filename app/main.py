from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from app.graph import graph
from fastapi.middleware.cors import CORSMiddleware

class AskRequest(BaseModel):
    question: str

app = FastAPI(title="Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "https://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)


@app.get("/")
def root():
    return {"message": "Welcome to the LangGraph API"}

@app.post("/ask")
def ask(req: AskRequest):

    result = graph.invoke({
        "question": req.question
    },
     config={
        "configurable": {
            "thread_id": "harish"   
        }
    })
    return {"answer": result["answer"],"description": result["description"]}

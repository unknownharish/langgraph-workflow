from typing import TypedDict,List, Optional
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel as BaseClass, Field
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load documents
BASE_DIR = Path(__file__).resolve().parent
SCHEMA_FILE = BASE_DIR / "schema_data.txt"

loader = TextLoader(str(SCHEMA_FILE))
documents = loader.load()

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
    return vectorstore


# 1. Define state
class GraphState(TypedDict):
    question: str
    userIntent:str
    answer: str
    description: str

    # conversational memory
    chat_history: list[str]
    last_intent: Optional[str]
    last_answer: Optional[str]

class UserIntent(BaseClass):
    intent: str = Field(description="Either sql query or paymentlink generation")

class StructuredOutput(BaseClass):
    description: str = Field(description="The query description")
    answer: str = Field(description="The Postgress query ")

# follo-up schema 
class FollowUpDecision(BaseClass):
    is_follow_up: bool = Field(description="True if question depends on previous context")
    rewritten_question: Optional[str] = Field(
        description="Standalone rewritten question if follow-up"
    )


# 2. LLM (created ONCE)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

def contextResolverNode(state: GraphState) -> GraphState:
    history = "\n".join(state.get("chat_history", []))

    prompt = f"""
    You are a conversation-aware assistant.

    Conversation so far:
    {history}

    User question:
    {state["question"]}

    Decide:
    - Is this question a follow-up?
    - If yes, rewrite it as a standalone question using the context.
    """

    response = llm.with_structured_output(FollowUpDecision).invoke(prompt)

    return {
        "question": response.rewritten_question if response.is_follow_up else state["question"],
        "rewritten_question": response.rewritten_question,
        "chat_history": state.get("chat_history", []) + [state["question"]],
        "last_intent": state.get("last_intent"),
        "last_answer": state.get("last_answer"),
    }

def getUserIntent(state: GraphState) -> GraphState:

    # if follow up question, use last intent
    if state.get("rewritten_question") and state.get("last_intent"):
        return {
            "userIntent": state["last_intent"]
        }
    
    llmResponse = llm.with_structured_output(UserIntent).invoke(state["question"])
    return {
        "userIntent": llmResponse.intent,
       
    }

# 3. Node function

def conditionalUserIntent(state: GraphState) :
    if state["userIntent"] == "sql query":
        return "query"
    else:
        return "payment_link"


def paymentLinkGeneratorNode(state: GraphState) -> GraphState:

    # payment link generation logic .
    return {
        "question": state["question"],
        "answer": "Payment link generation is not implemented yet. eg: https://paymentlink.com/xyz",
        "description": ""
    }   

def sqlQueryNode(state: GraphState) -> GraphState:
   
    retriever = get_vectorstore().as_retriever(
    search_kwargs={"k": 4}
     )
   
    relevant_docs = retriever.invoke(state["question"])
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""
        You are an expert PostgreSQL query generator.

        Rules:
        - Generate ONLY read-only SELECT queries
        - DO NOT use DELETE, DROP, TRUNCATE, or ALTER
        - DO NOT return sensitive data (passwords, tokens, secrets, PII)
        - Select ONLY the minimum required columns to answer the question
        - Follow SQL best practices and performance optimization

        Database schema:
        {context}

        User request:
        {state["question"]}

        Return the correct and optimized PostgreSQL SELECT query only.
        """

    llmStructured = llm.with_structured_output(StructuredOutput)

    response = llmStructured.invoke(prompt)
    return {
        "question": state["question"],
        "description": response.description,
        "answer": response.answer,
        "last_intent": "sql query",
        "last_answer": response.answer,
        "chat_history": state["chat_history"]
    }

# 4. Build graph
builder = StateGraph(GraphState)

builder.add_node("contextResolver", contextResolverNode)
builder.add_node("userIntent", getUserIntent)
builder.add_node("queryGenerator", sqlQueryNode)
builder.add_node("paymentLinkGenerator", paymentLinkGeneratorNode)

builder.set_entry_point("contextResolver")

builder.add_edge("contextResolver", "userIntent")

# CONDITIONAL ROUTING (correct way)
builder.add_conditional_edges(
    "userIntent",
    conditionalUserIntent,
    {
        "query": "queryGenerator",
        "payment_link": "paymentLinkGenerator"
    }
)

builder.add_edge("queryGenerator", END)
builder.add_edge("paymentLinkGenerator", END)

graph = builder.compile()
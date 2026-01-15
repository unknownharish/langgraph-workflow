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

# class GraphState(TypedDict):
#     question: str
#     userIntent:str
#     answer: str
#     description: str

#     # conversational memory
#     chat_history: list[str]
#     last_intent: Optional[str]
#     last_answer: Optional[str]

class ConversationTurn(TypedDict):
    question: str
    intent: str
    answer: str

class GraphState(TypedDict):
    question: str
    rewritten_question: Optional[str]

    userIntent: Optional[str]
    answer: Optional[str]
    description: Optional[str]

    history: List[ConversationTurn]



# llm structurered output 

class IntentAndContext(BaseClass):
    intent: str
    is_follow_up: bool
    rewritten_question: Optional[str]

class StructuredOutput(BaseClass):
    description: str = Field(description="The query description")
    answer: str = Field(description="The Postgress query ")


# 2. LLM (created ONCE)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)


# utility function 
MAX_HISTORY = 10

def update_history(
    history: List[ConversationTurn],
    question: str,
    intent: str,
    answer: str
) -> List[ConversationTurn]:
    new_history = history + [{
        "question": question,
        "intent": intent,
        "answer": answer
    }]
    return new_history[-MAX_HISTORY:]



# 3. Node function

def intent_context_node(state: GraphState) -> GraphState:
    history_text = "\n".join(
        f"User: {turn['question']}\nIntent: {turn['intent']}\nAnswer: {turn['answer']}"
        for turn in state.get("history", [])
    )

    prompt = f"""
    You are a conversation-aware assistant.

    Conversation history:
    {history_text}

    New user question:
    {state["question"]}

    Decide:
    - intent (sql query | payment link)
    - is this a follow-up?
    - rewrite if needed
    """

    response = llm.with_structured_output(IntentAndContext).invoke(prompt)

    final_question = (
        response.rewritten_question
        if response.is_follow_up and response.rewritten_question
        else state["question"]
    )

    return {
        **state,
        "question": final_question,
        "rewritten_question": response.rewritten_question,
        "userIntent": response.intent
    }

def conditionalUserIntent(state: GraphState) :
    if state["userIntent"] == "sql query":
        return "query"
    else:
        return "payment_link"


def paymentLinkGeneratorNode(state: GraphState) -> GraphState:

    answer = "https://paymentlink.com/xyz"

    history = update_history(
        state.get("history", []),
        state["question"],
        "payment link",
        answer
    )

    return {
        **state,
        "answer": answer,
        "description": "Generated payment link is not implemented yet",
        "history": history
    }

def sqlQueryNode(state: GraphState) -> GraphState:
   
    retriever = get_vectorstore().as_retriever(
    search_kwargs={"k": 4}
     )
   
    relevant_docs = retriever.invoke(state["question"])
    context = "\n".join([doc.page_content for doc in relevant_docs])

    history_text = "\n".join(
        f"User: {turn['question']}\nAnswer: {turn['answer']}"
        for turn in state.get("history", [])[-4:] 
    )
    prompt = f"""
        You are an expert PostgreSQL query generator.

        Rules:
        - Generate ONLY read-only SELECT queries
        - DO NOT use DELETE, DROP, TRUNCATE, or ALTER
        - DO NOT return sensitive data (passwords, tokens, secrets, PII)
        - Select ONLY the minimum required columns to answer the question
        - Follow SQL best practices and performance optimization
       
        Conversation history:
        {history_text}.
        
        Database schema:
        {context}

        User request:
        {state["question"]}

        Return the correct and optimized PostgreSQL SELECT query only.
        """

    llmStructured = llm.with_structured_output(StructuredOutput)

    response = llmStructured.invoke(prompt)
    history = update_history(
        state.get("history", []),
        state["question"],
        "sql query",
        response.answer
    )
    return {
        **state,
        "answer": response.answer,
        "description": response.description, #query description
        "history": history
     }

# 4. Build graph
builder = StateGraph(GraphState)

builder.add_node("userIntent", intent_context_node)
builder.add_node("queryGenerator", sqlQueryNode)
builder.add_node("paymentLinkGenerator", paymentLinkGeneratorNode)

builder.set_entry_point("userIntent")

# CONDITIONAL ROUTING 
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
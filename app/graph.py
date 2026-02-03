from typing import TypedDict,List, Optional,Annotated
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel as BaseClass, Field
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import create_engine, text

POSTGRES_URL = "postgresql+psycopg2://user:password@localhost:5432/paynewtest"
engine = create_engine(POSTGRES_URL)


# Load documents
BASE_DIR = Path(__file__).resolve().parent
SCHEMA_FILE = BASE_DIR / "schema_data.txt"

loader = TextLoader(str(SCHEMA_FILE))
documents = loader.load()




# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_db_connection():
    return engine.connect()




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



# reducer function 
MAX_HISTORY = 10

def history_reducer(old, new):
    old = old or []
    if not isinstance(old, list):
        old = [old]
    if not isinstance(new, list):
        new = [new]
    return (old + new)[-MAX_HISTORY:]



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
    history: Annotated[List[ConversationTurn], history_reducer]




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
    model="gemini-2.5-flash",
    temperature=0.7
)


# 3. Node function

def intent_context_node(state: GraphState) -> GraphState:

#     print("memory",memory.get({
#     "configurable": {"thread_id": "harish"}
# }))
    
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

    history = [{
       "question": state["question"],
       "intent": "payment link",
       "answer": answer
    }]

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
    history = [{
        "question": state["question"],
        "intent": "sql query",
        "answer": response.answer
    }]
    return {
        **state,
        "answer": response.answer,
        "description": response.description, #query description
        "history": history
     }
    
    

def sqlQueryExecutorNode(state: GraphState) -> GraphState:
    query = state["answer"]  

    try:
            conn = get_db_connection()
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()

            data = [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        data = {"error": str(e)}

    history = [{
        "question": state["question"],
        "intent": "sql query",
        "answer": query
    }]

    return {
        **state,
        "description": "Query executed successfully",
        "answer": data,   # now answer is actual DB result
        "history": history
    }
    

# 4. Build graph
builder = StateGraph(GraphState)

builder.add_node("userIntent", intent_context_node)
builder.add_node("queryGenerator", sqlQueryNode)
builder.add_node("paymentLinkGenerator", paymentLinkGeneratorNode)
builder.add_node("queryExecutor", sqlQueryExecutorNode)

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

builder.add_edge("queryGenerator", "queryExecutor")
builder.add_edge("queryExecutor", END)

builder.add_edge("paymentLinkGenerator", END)

# memory saver 
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

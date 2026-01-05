from typing import TypedDict
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel as BaseClass, Field
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load documents
BASE_DIR = Path(__file__).resolve().parent
SCHEMA_FILE = BASE_DIR / "schema_data.txt"

loader = TextLoader(str(SCHEMA_FILE))
documents = loader.load()

# Create embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 1. Define state
class GraphState(TypedDict):
    question: str
    answer: str
    description: str

class StructuredOutput(BaseClass):
    description: str = Field(description="The query description")
    answer: str = Field(description="The Postgress query ")


# 2. LLM (created ONCE)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# 3. Node function
def llm_node(state: GraphState) -> GraphState:
   
    retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
     )
   
    relevant_docs = retriever.get_relevant_documents(state["question"])
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""you are postgress db query creater with best optimization and clear cut to requirement .,
      now here are the tables with their user table.
       
      Schema context:
      {context}

      User question:
      {state["question"]}

      Generate the correct PostgressSql query.
         """
    llmStructured = llm.with_structured_output(StructuredOutput)

    response = llmStructured.invoke(prompt)
    return {
        "question": state["question"],
        "description": response.description,
        "answer": response.answer
    }

# 4. Build graph``
builder = StateGraph(GraphState)
builder.add_node("llm", llm_node)
builder.set_entry_point("llm")
builder.add_edge("llm", END)

graph = builder.compile()

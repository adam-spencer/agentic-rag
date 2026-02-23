import os
from typing import TypedDict, Literal
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

class AgentState(TypedDict):
    query: str
    route: str
    sql_context: str
    vector_context: str
    final_answer: str

def get_llm():
    return ChatGoogleGenerativeAI(model="gemma-3-27b-it")

def route_query(state: AgentState):
    llm = get_llm()
    prompt = f"""You are a routing agent for a SaaS CRM dataset.
The database has two sources:
1. SQLite: Contains structured customer data (customerID, MonthlyCharges, TotalCharges, Churn).
2. ChromaDB: Contains unstructured Twitter support tickets.

Based on the user query, return ONE word: "sql", "vector", or "both".
Query: {state['query']}
Route:"""
    response = llm.invoke(prompt)
    route = response.content.strip().lower()
    
    if "both" in route or ("sql" in route and "vector" in route):
        route = "both"
    elif "sql" in route:
        route = "sql"
    elif "vector" in route:
        route = "vector"
    else:
        route = "both" # Fallback
        
    print(f"[Router Node] Decided route: {route}")
    return {"route": route}

def execute_sql(state: AgentState):
    print("[SQL Retriever Node] Querying structured data...")
    db = SQLDatabase.from_uri("sqlite:///crm_data.db")
    llm = get_llm()
    
    schema = db.get_table_info()
    prompt = f"""You are a SQLite expert. Given the following database schema, write a valid SQLite query to answer the user's question.
Return ONLY the SQL query, no markdown formatting, no explanation.

Schema:
{schema}

Question: {state["query"]}"""
    
    try:
        response = llm.invoke(prompt)
        sql_query = response.content.strip().replace("```sql", "").replace("```", "").strip()
        print(f"[SQL Retriever Node] Executing SQL: {sql_query}")
        result = db.run(sql_query)
        print(f"[SQL Retriever Node] Result: {result}")
        return {"sql_context": str(result)}
    except Exception as e:
        print(f"[SQL Retriever Node] Error: {e}")
        return {"sql_context": f"Error querying SQL: {e}"}

def execute_vector(state: AgentState):
    print("[Vector Retriever Node] Querying unstructured data...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="support_tweets")
        docs = vectorstore.similarity_search(state["query"], k=3)
        context = "\n-----\n".join([d.page_content for d in docs])
        print(f"[Vector Retriever Node] Found {len(docs)} relevant tweets.")
        return {"vector_context": context}
    except Exception as e:
        print(f"[Vector Retriever Node] Error: {e}")
        return {"vector_context": f"Error querying Vector DB: {e}"}

def execute_both(state: AgentState):
    print("[Parallel Router Node] Routing to both SQL and Vector databases...")
    state_updates = {}
    state_updates.update(execute_sql(state))
    state_updates.update(execute_vector(state))
    return state_updates

def synthesize(state: AgentState):
    print("[Synthesis Node] Generating final answer...")
    llm = get_llm()
    prompt = f"""Synthesize an answer for the user query based on the provided context.
Query: {state['query']}
SQL Context (Structured data): {state.get('sql_context', 'N/A')}
Vector Context (Unstructured tweets): {state.get('vector_context', 'N/A')}

Answer:"""
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

def router_condition(state: AgentState) -> Literal["execute_sql", "execute_vector", "execute_both"]:
    route = state.get("route", "both")
    if route == "sql":
        return "execute_sql"
    elif route == "vector":
        return "execute_vector"
    else:
        return "execute_both"

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", route_query)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("execute_vector", execute_vector)
    workflow.add_node("execute_both", execute_both)
    workflow.add_node("synthesize", synthesize)
    
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        router_condition,
        {
            "execute_sql": "execute_sql",
            "execute_vector": "execute_vector",
            "execute_both": "execute_both"
        }
    )
    
    workflow.add_edge("execute_sql", "synthesize")
    workflow.add_edge("execute_vector", "synthesize")
    workflow.add_edge("execute_both", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()

if __name__ == "__main__":
    load_dotenv()
    
    app = build_graph()
    
    # Test queries mapping to different routes
    test_queries = [
        "How many customers have churned? What is the total revenue collected from churned customers?",
        "What are people generally complaining about on Twitter?",
        "Are the customers who churned complaining about internet reliability?"
    ]
    
    for q in test_queries:
        print(f"\n\n{'='*50}")
        print(f"User Query: {q}")
        print(f"{'='*50}")
        try:
            result = app.invoke({"query": q})
            print(f"\n=== FINAL SYNTHESIZED ANSWER ===\n{result['final_answer']}\n")
        except Exception as e:
            print(f"\nExecution terminated: {e}")
            break

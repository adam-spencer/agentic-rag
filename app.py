import streamlit as st
import os
from dotenv import load_dotenv

# Import the LangGraph application
# Because agent.py creates the sqlite3 databases locally, 
# Streamlit will execute parallel to the main directory flawlessly.
from agent import build_graph

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def get_agent_app():
    return build_graph()

def main():
    load_dotenv()
    
    st.title("🤖 Telco CRM Agentic RAG")
    st.markdown("Ask questions about **Telco Churn (SQL Structured Data)** or **Twitter Support (Vector Unstructured Data)**.")
    
    # Sidebar
    with st.sidebar:
        st.header("Graph Agent Data Sources")
        st.markdown("- **SQLite DB**: Telecommunications Customer Churn (7k rows)")
        st.markdown("- **Chroma DB**: HuggingFace Indexed Twitter Customer Support Tickets (10k chunk sampled from 794k dataset)")
        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # App core
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am an Agentic RAG assistant. Ask me anything about our customers, their churn rates, or their support tickets on Twitter!"}
        ]

    # Show existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    agent_app = get_agent_app()
    if prompt := st.chat_input("E.g., Which customers churned in the last month?"):
        # Display user logic
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing intent and routing query..."):
                try:
                    # Invoke LangGraph pipeline
                    result = agent_app.invoke({"query": prompt})
                    
                    route_taken = result.get('route', 'unknown')
                    final_answer = result.get('final_answer', 'Sorry, I failed to synthesize an answer.')
                    
                    # Display trace metadata
                    st.caption(f"⚡ *Agent routed query to: `{route_taken}` sources*")
                    
                    # Display content
                    st.markdown(final_answer)
                    
                    # Save to state
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
        
                except Exception as e:
                    st.error(f"Error accessing graph: {e}")

if __name__ == "__main__":
    main()

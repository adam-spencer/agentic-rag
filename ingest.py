import pandas as pd
import sqlite3
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    print("Loading datasets...")
    # Load Telco Churn and Twitter Support CSVs
    churn_df = pd.read_csv("telco_churn.csv")
    support_df = pd.read_csv("twitter_support.csv")
    
    print(f"Loaded {len(churn_df)} churn records and {len(support_df)} support tweets.")
    
    # Push structured data to SQLite
    print("Pushing structured data to SQLite (crm_data.db)...")
    conn = sqlite3.connect("crm_data.db")
    churn_df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()
    print("SQLite ingestion complete.")
    
    # Push unstructured text to ChromaDB
    print("Pushing unstructured text to ChromaDB...")
    
    # Initialize Google GenAI Embeddings
    # If API key is not set, we'll fall back to Chroma's default embeddings for local testing purposes.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("WARNING: GEMINI_API_KEY not set or is default. Using Chroma's default sentence-transformers embedding function.")
        embedding_function = None # Chroma default will be used
        use_langchain_chroma = False
    else:
        embedding_function = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        use_langchain_chroma = True

    if use_langchain_chroma:
        # Create documents for langchain
        docs = [
            Document(
                page_content=row["text"],
                metadata={"customerID": row["customerID"], "sentiment": row["sentiment"]}
            ) for _, row in support_df.iterrows()
        ]
        ids = support_df["tweet_id"].astype(str).tolist()
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_function,
            ids=ids,
            persist_directory="./chroma_db",
            collection_name="support_tweets"
        )
    else:
        # Fallback to direct chromadb client for local embeddings
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(name="support_tweets")
        
        documents = support_df["text"].tolist()
        metadatas = [{"customerID": row["customerID"], "sentiment": row["sentiment"]} for _, row in support_df.iterrows()]
        ids = support_df["tweet_id"].astype(str).tolist()
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    print("ChromaDB ingestion complete.")

if __name__ == "__main__":
    main()

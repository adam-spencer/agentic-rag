import pandas as pd
import sqlite3
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
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
    
    # Initialize Local HuggingFace Embeddings
    print("Initializing local HuggingFace embedding model (this may take a moment to download on first run)...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
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
    
    print("ChromaDB ingestion complete.")

if __name__ == "__main__":
    main()

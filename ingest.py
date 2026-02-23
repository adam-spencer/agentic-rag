import pandas as pd
import sqlite3
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from tqdm import tqdm

def main():
    load_dotenv()
    
    print("Loading datasets...")
    churn_df = pd.read_csv("data/telco_churn.csv")
    support_df = pd.read_parquet("data/twitter_support.parquet")
    print(f"Loaded {len(churn_df)} churn records and {len(support_df)} support tweets.")
    
    print("Pushing structured data to SQLite (crm_data.db)...")
    conn = sqlite3.connect("crm_data.db")
    churn_df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()
    print("SQLite ingestion complete.")
    
    print("Pushing unstructured text to ChromaDB...")
    
    print("Initializing local HuggingFace embedding model...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Converting records to LangChain documents...")
    # TEMP - limit to 10k rows for demo
    MAX_ROWS = 10000
    subset_df = support_df.head(MAX_ROWS)  # remove .head() for full dataset
    print(f"Embedding a subset of {MAX_ROWS} rows for performance demonstration...")
    
    docs = []
    ids = []
    
    import random
    valid_customer_ids = churn_df["customerID"].tolist()
    
    for idx, row in subset_df.iterrows():
        # Clean text
        text_content = str(row.get("conversation", ""))
        summary = str(row.get("summary", ""))
        combined_text = f"Conversation: {text_content}\nSummary: {summary}"
        
        # Assign a random customerID to tweets
        mapping_id = random.choice(valid_customer_ids)
        
        docs.append(Document(
            page_content=combined_text,
            metadata={
                "customerID": mapping_id,
                "company": str(row.get("company", "Unknown")),
                "conversation_id": str(row.get("conversation_id", f"missing_{idx}"))
            }
        ))
        
        ids.append(str(row.get("conversation_id", f"ID_{idx}")))
        
    print(f"Created {len(docs)} documents. Writing to ChromaDB in chunks...")
    
    # Init ChromaDB
    BATCH_SIZE = 1000
    vectorstore = Chroma(
        embedding_function=embedding_function,
        persist_directory="./chroma_db",
        collection_name="support_tweets"
    )
    
    # Add text in batches
    for i in tqdm(range(0, len(docs), BATCH_SIZE)):
        batch_docs = docs[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        
        vectorstore.add_documents(documents=batch_docs, ids=batch_ids)

    print("ChromaDB ingestion complete.")

if __name__ == "__main__":
    main()

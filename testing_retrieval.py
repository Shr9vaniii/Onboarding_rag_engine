import re
import chromadb
import json
import uuid
from pathlib import Path
from collections import Counter
from chromadb.utils import embedding_functions

DB_PATH = Path("./enterprise_data/chroma_db_v2")


print("Initializing Qwen3-Embedding-0.6B on CUDA...")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)

chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
collection = chroma_client.get_or_create_collection(
    name="engineering_knowledge",
    embedding_function=embedding_function
)

def retrieve_smart(query):

    results=collection.query(query_texts=[query],n_results=3)
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i] if 'distances' in results else 'N/A'
        
        print(f"\n[Match {i+1}] | Type: {meta.get('type')} | Distance score: {distance}")
        print(f"Source Artifact: {meta.get('source')}")
        print(f"Extracted Context:\n{doc}")
        print("-" * 50)



if __name__ == "__main__":
    """retrieve_smart("what arguments does HTTPException take")
    retrieve_smart("how were sub applications implemented")
    retrieve_smart("what was the reasoning behind Depends")
    retrieve_smart("dependency injection not working")
    retrieve_smart("how do I handle file uploads")
    retrieve_smart("async route handler function signature")
    retrieve_smart("background tasks after response")
    retrieve_smart("CORS error when calling API from browser")"""

    results = collection.query(
    query_texts=["what arguments does HTTPException take"],
    n_results=3,
    where={
        "type": {
            "$in": ["wikis_arch_docs", "code_contracts", "code_examples"]
        }
    }
)
    
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i] if 'distances' in results else 'N/A'
        
        print(f"\n[Match {i+1}] | Type: {meta.get('type')} | Distance score: {distance}")
        print(f"Source Artifact: {meta.get('source')}")
        print(f"Extracted Context:\n{doc}")
        print("-" * 50)
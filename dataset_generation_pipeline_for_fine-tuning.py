import random
import chromadb
from pathlib import Path

DB_PATH=Path("./enterprise_data/chroma_db")
print("Connecting to chromadb....")

chroma_client=chromadb.PersistentClient(path=str(DB_PATH))
collection=chroma_client.get_collection(name="engineering_knowledge")

print("Fetching Enterprise knowledge...")

results=collection.get()
documents=results["documents"]
metadatas=results["metadatas"]

sample_size=min(30,len(documents))
indices=random.sample(range(len(documents)),sample_size)

output_file="draft_dataset.md"

print(f"Generating Markdown template for {sample_size} samples...")

with open(output_file,"w",encoding="utf-8") as f:
    for count,idx in enumerate(indices,1):
        source=metadatas[idx].get('source',"unknown")
        doc_text=documents[idx].strip()

        f.write(f"## EXAMPLE {count}\n")
        f.write(f"**SOURCE:** {source}\n\n")
        f.write(f"**CONTENT BLOCK:**\n```\n{doc_text}\n```\n\n")
        f.write("---\n\n")
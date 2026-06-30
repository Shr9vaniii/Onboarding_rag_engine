import os
import json
import uuid
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter

DATA_DIR=Path("./enterprise_data")
DOCS_DIR=DATA_DIR/"wikis_and_docs"
ISSUES_DIR=DATA_DIR/"tribal_history"
CODE_DIR=DATA_DIR/"code_contracts"
SCRAPED_DOCS_DIR=DATA_DIR/"wikis_and_docs_scraped"
EXAMPLES_DIR=DATA_DIR/"code_examples"
PRS_DIR=DATA_DIR/"decision_history"

embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)

DB_PATH=DATA_DIR/"chroma_db_v2"
chroma_client=chromadb.PersistentClient(path=str(DB_PATH))

collection=chroma_client.get_or_create_collection(
    name="engineering_knowledge",
    embedding_function=embedding_function
)

def embed_code_contracts():
    """Embed AST-parsed code_contracts.One fucntion/class = One vector."""

    print("Embedding code contracts...")
    documents=[]
    metadatas=[]
    ids=[]

    if not CODE_DIR.exists():
        print("Code Directory not found skipping...")
        return

    for file_path in CODE_DIR.glob("*.json"):
        with open(file_path,"r",encoding="utf-8") as f:
            data=json.load(f)
            source_file=data.get("file_path","unknown")

            for element in data.get("elements",[]):
                if element['type']=="function":
                    doc_text=f"Type: {element['type']}\n Name: {element['name']}Arguments: {element.get('arguments',[])}\nSource_File: {source_file}\nDocstring/Description: {element.get('docstring','No docstring')}"
                else:
                    doc_text=f"Type: {element['type']}\n Name: {element['name']}\nArguments: {element.get('arguments',[])}\nSource_File: {source_file}\nDocstring/Description: {element.get('docstring','No docstring')}"

                documents.append(doc_text)
                metadatas.append({"source":source_file,"type":"code_contracts"})
                ids.append(str(uuid.uuid4()))

    if documents:

        batch_size=5000
        for i in range(0,len(documents),batch_size):
            collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size],
            )            
        print(f"Embedded {len(documents)} code Contracts.")

def embed_tribal_history():
    """Embeds Closed GitHub issues (simulate Slack/Jira history)"""

    print("Embedding Tribal History...")

    documents=[]
    metadatas=[]
    ids=[]

    if not ISSUES_DIR.exists():
        print("Issues Directory not found skipping...")
        return
    
    for file_path in ISSUES_DIR.glob("*.json"):
        with open(file_path,"r",encoding="utf-8") as f:
            issue=json.load(f)

            doc_text=f"Bug/Issue Title: {issue['title']}\nLabels: {', '.join(issue['labels'])}\nProblem: {issue['body']}"

            documents.append(doc_text)
            metadatas.append({
                "source": issue['url'],
                "type": "bug_history",
                "issue_id":str(issue['id'])
            })
            ids.append(str(uuid.uuid4()))
            comments=issue.get('comments',[])
            for i in range(0,len(comments),3):
                group=comments[i:i+3]
                thread_text=f"Disscusion Thread (Issue : {issue['title']}):\n"
                for c in group:
                    thread_text+=f"[{c['author']}]:{c['body'][:600]}\n"
                documents.append(thread_text)
                metadatas.append({
                    "source":issue['url'],
                    "type":"bug_history",
                    "issue_id":str(issue['id'])
                })    
                ids.append(str(uuid.uuid4()))
                    
                

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Embedded {len(documents)} historical issues.")

def embed_wikis():
    """Reads Markdown files and chunks by paragraph/section"""

    print("Embedding Wikis and Architecture Docs...")        

    headers_to_split_on=[
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
        ("#####", "H5"),
        ("######", "H6")
    ]

    markdown_splitter= MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )


    documents=[]
    metadatas=[]
    ids=[]


    
        
    if DOCS_DIR.exists():
        for file_path in DOCS_DIR.glob("*.md"):
            with open(file_path,"r",encoding="utf-8") as f:
                text=f.read()

            semantic_chunks=markdown_splitter.split_text(text)

            for chunk in semantic_chunks:
                header_path=" > ".join(chunk.metadata.values())

                if header_path:
                    enriched_text=f"Context: [{header_path}]\n Content: {chunk.page_content}"
                else:
                    enriched_text=chunk.page_content   
                documents.append(enriched_text)
                metadatas.append({
                    "source":f"wikis/{file_path.name}",
                    "type":"wikis_arch_docs"
                })     
                ids.append(str(uuid.uuid4()))
                

    if SCRAPED_DOCS_DIR.exists():
        
        for file_path in SCRAPED_DOCS_DIR.glob("*.json"):
            with open(file_path,"r",encoding="utf-8") as f:
                data=json.load(f)
            content=data.get("content","")
            title=data.get("title","")
            section=data.get("section","")
            url=data.get("url","")

            paragraphs=[p.strip() for p in content.split("\n\n") if len(p.strip())>100]
            for para in paragraphs:
                enriched_text=f"Section: {section} | Page: {title}\n{para}"
                documents.append(enriched_text)
                metadatas.append({
                    "source":url,
                    "type":"wikis_arch_docs",
                    "section":section
                })
                ids.append(str(uuid.uuid4()))
    if documents:
        batch_size = 5000
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size],
            )
        print(f"Embedded {len(documents)} wiki/doc sections.")

def embed_code_examples():
    """Embeds runnable python files/docs_src...One file = One chunk"""  

    documents=[]
    metadatas=[]
    ids=[]
    if not EXAMPLES_DIR.exists():
        print("Code Examples Directory not found skipping...")
        return

    for file_path in EXAMPLES_DIR.glob("*.py"):
        content=file_path.read_text(encoding="utf-8",errors="ignore").strip()
        if len(content)<50:
            continue
        doc_text=f"Example file: {file_path.name}\n\n{content}"
        documents.append(doc_text)
        metadatas.append({
            "source":str(file_path.name),
            "type":"code_examples"
        })
        ids.append(str(uuid.uuid4()))

    if documents:
        batch_size=5000
        for i in range(0,len(documents),batch_size):
            collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print(f"Embedded {len(documents)} code examples.")  

def embed_decision_history():
    """Merged PRs - Architectural decisions and reasoning"""
    print("Embedding decision history (PRs)...")
    documents=[]
    metadatas=[]
    ids=[]
    if not PRS_DIR.exists():
        print("PRs Directory not found, skipping...")
        return
    
    for file_path in PRS_DIR.glob("*.json"):
        with open(file_path,"r",encoding="utf-8") as f:
            pr=json.load(f)

        body=(pr.get("body","")).strip()
        if not body:
            continue
        
        # Bode primary chunk
        doc_text=f"PR #{pr['id']}: {pr['title']}\nDecision/Reasoning: {pr['body'][:2000]}"
        documents.append(doc_text)
        metadatas.append({
            "source":pr['url'],
            "type":"decision_history",
            "pr_id":str(pr['id'])
        })
        ids.append(str(uuid.uuid4()))

    if documents:
        batch_size=5000
        for i in range(0,len(documents),batch_size):
            collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
                )
        print(f"Embedded {len(documents)} decision history chunks")



if __name__=="__main__":
    embed_code_contracts()
    embed_code_examples()
    embed_tribal_history()
    embed_decision_history()
    embed_wikis()
    
    print("All data embedded successfully. Vector DB is ready for retrieval!")

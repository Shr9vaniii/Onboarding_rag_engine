import os
import json
import uuid
import shutil
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Directory Configuration ---
DATA_DIR = Path("./data")
ISSUES_DIR = DATA_DIR / "tribal_history"
CODE_DIR = DATA_DIR / "code_contracts"
SCRAPED_DOCS_DIR = DATA_DIR / "wikis_and_docs_scraped"
EXAMPLES_DIR = DATA_DIR / "code_examples"
PRS_DIR = DATA_DIR / "decision_history"

DB_PATH = DATA_DIR / "chroma_db_v2"
CHECKPOINT_PATH = DATA_DIR / "checkpoint_manifest.json"

# --- Checkpoint Management Functions ---
def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_files": []}

def save_checkpoint(file_identifier):
    manifest = load_checkpoint()
    if file_identifier not in manifest["processed_files"]:
        manifest["processed_files"].append(file_identifier)
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

# --- Initialize ChromaDB and Qwen3 on T4 GPU ---
print("Initializing Qwen3-Embedding-0.6B on CUDA...")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    device="cuda"  # Explicitly targeting the T4 GPU
)

chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
collection = chroma_client.get_or_create_collection(
    name="engineering_knowledge",
    embedding_function=embedding_function
)

# --- Embedding Task Functions ---

def embed_in_batches(collection, documents, metadatas, ids, batch_size=8):
    """Embed with small batches and clear CUDA cache between each."""
    for i in range(0, len(documents), batch_size):
        try:
            collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size],
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at batch {i} — clearing cache and retrying with batch_size=1")
                torch.cuda.empty_cache()
                # Retry one by one
                for j in range(i, min(i+batch_size, len(documents))):
                    try:
                        collection.add(
                            documents=[documents[j]],
                            metadatas=[metadatas[j]],
                            ids=[ids[j]],
                        )
                    except RuntimeError:
                        print(f"  Skipping chunk {j} — too large even solo")
                        continue
            else:
                raise e
        
        # Clear cache after every batch
        torch.cuda.empty_cache()

def embed_code_contracts():
    """Embed AST-parsed code_contracts prioritizing function/class name first."""
    
    print("\n--- Processing Code Contracts (Plain Text) ---")
    if not CODE_DIR.exists():
        print("Code Directory not found, skipping...")
        return

    checkpoint = load_checkpoint()

    for file_path in CODE_DIR.glob("*.json"):
        file_id = f"code_contract:{file_path.name}"
        
        if file_id in checkpoint["processed_files"]:
            print(f"Skipping (Already Indexed): {file_path.name}")
            continue

        print(f"Indexing: {file_path.name}")
        documents = []
        metadatas = []
        ids = []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            source_file = data.get("file_path", "unknown")

            for element in data.get("elements", []):
                element_type = element.get("type", "").lower()
                name = element.get("name", "unknown")
                docstring = element.get("docstring", "No documentation provided.").strip()

                # --- PLAIN TEXT: FUNCTION FIRST ---
                if element_type == "function":
                    is_async = element.get("is_async", False)
                    return_type = element.get("return_type", "None")
                    
                    args = element.get("arguments", [])
                    verbose_args = ", ".join([f"'{arg['name']}' (type: {arg['type']})" for arg in args if "name" in arg])

                    mode_text = "asynchronous" if is_async else "synchronous"
                    
                    # State the function first, then the source file
                    doc_text = f"The {mode_text} code function: '{name}' is defined in the source file:'{source_file}'. "
                    
                    if verbose_args:
                        doc_text += f"It accepts the following parameters: {verbose_args}. "
                    else:
                        doc_text += "It does not take any parameters. "
                    
                    doc_text += f"It returns a value of type: '{return_type}'. "
                    doc_text += f"The developer documentation for this function states: {docstring}"

                # --- PLAIN TEXT: CLASS FIRST ---
                elif element_type == "class":
                    attributes = element.get("attributes", [])
                    attrs_formatted = ", ".join([f"'{attr['name']}' (type: {attr['type']})" for attr in attributes if "name" in attr])

                    # State the class first, then the source file
                    doc_text = f"The code class: '{name}' is defined in the source file: '{source_file}'. "
                    
                    if attrs_formatted:
                        doc_text += f"This class contains the following attributes or fields: {attrs_formatted}. "
                    else:
                        doc_text += "This class has no explicitly defined attributes. "
                    
                    doc_text += f"The developer documentation for this class states: {docstring}"

                # --- PLAIN TEXT: FALLBACK FIRST ---
                else:
                    doc_text = f"The code element '{name}' of type '{element_type}' is located in the file '{source_file}'. Its details are: {str(element)}"

                documents.append(doc_text)
                metadatas.append({
                    "source": source_file, 
                    "type": "code_contracts",
                    "artifact_type": element_type
                })
                ids.append(str(uuid.uuid4()))

        if documents:
            embed_in_batches(collection,documents,metadatas,ids,batch_size=8)
        
        save_checkpoint(file_id)

def embed_tribal_history():
    """Embeds Closed GitHub issues."""
    print("\n--- Processing Tribal History (Issues) ---")
    if not ISSUES_DIR.exists():
        print("Issues Directory not found, skipping...")
        return
    
    checkpoint = load_checkpoint()

    for file_path in ISSUES_DIR.glob("*.json"):
        file_id = f"issue:{file_path.name}"
        if file_id in checkpoint["processed_files"]:
            print(f"Skipping (Already Indexed): {file_path.name}")
            continue

        print(f"Indexing: {file_path.name}")
        documents, metadatas, ids = [], [], []

        with open(file_path, "r", encoding="utf-8") as f:
            issue = json.load(f)

            doc_text = f"Bug/Issue Title: {issue['title']}\nLabels: {', '.join(issue['labels'])}\nProblem: {issue['body']}"
            documents.append(doc_text)
            metadatas.append({
                "source": issue['url'],
                "type": "bug_history",
                "issue_id": str(issue['id'])
            })
            ids.append(str(uuid.uuid4()))

            comments = issue.get('comments', [])
            for i in range(0, len(comments), 3):
                group = comments[i:i+3]
                thread_text = f"Discussion Thread (Issue : {issue['title']}):\n"
                for c in group:
                    thread_text += f"[{c['author']}]:{c['body'][:200]}\n"
                documents.append(thread_text)
                metadatas.append({
                    "source": issue['url'],
                    "type": "bug_history",
                    "issue_id": str(issue['id'])
                })    
                ids.append(str(uuid.uuid4()))

        if documents:
            embed_in_batches(collection,documents,metadatas,ids,batch_size=8)
        save_checkpoint(file_id)

def embed_wikis():
    """Reads Markdown and scraped JSON files and chunks them."""
    print("\n--- Processing Wikis and Architecture Docs ---")
    checkpoint = load_checkpoint()

    # Scraped JSON Processing
    if SCRAPED_DOCS_DIR.exists():
        for file_path in SCRAPED_DOCS_DIR.glob("*.json"):
            file_id = f"wiki_scraped:{file_path.name}"
            if file_id in checkpoint["processed_files"]:
                print(f"Skipping (Already Indexed): {file_path.name}")
                continue

            print(f"Indexing: {file_path.name}")
            documents, metadatas, ids = [], [], []

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            content = data.get("content", "")
            title = data.get("title", "")
            section = data.get("section", "")
            url = data.get("url", "")

            paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 100]
            for para in paragraphs:
                enriched_text = f"Section: {section} | Page: {title}\n{para}"
                documents.append(enriched_text)
                metadatas.append({"source": url, "type": "wikis_arch_docs", "section": section})
                ids.append(str(uuid.uuid4()))

            if documents:
                embed_in_batches(collection,documents,metadatas,ids,batch_size=8)
            save_checkpoint(file_id)

def embed_code_examples():
    """Embeds runnable python files."""
    print("\n--- Processing Code Examples ---")
    if not EXAMPLES_DIR.exists():
        print("Code Examples Directory not found, skipping...")
        return
      

    checkpoint = load_checkpoint()

    for file_path in EXAMPLES_DIR.glob("*.py"):
        file_id = f"code_example:{file_path.name}"
        if file_id in checkpoint["processed_files"]:
            print(f"Skipping (Already Indexed): {file_path.name}")
            continue

        print(f"Indexing: {file_path.name}")
        content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(content) < 50:
            continue

        doc_text = f"Code example file: {file_path.name}\n\nCode: {content}"
        
        collection.add(
            documents=[doc_text],
            metadatas=[{"source": str(file_path.name), "type": "code_examples"}],
            ids=[str(uuid.uuid4())]
        )
        save_checkpoint(file_id)

def embed_decision_history():
    """Merged PRs - Architectural decisions and reasoning"""
    print("\n--- Processing Decision History (PRs) ---")
    if not PRS_DIR.exists():
        print("PRs Directory not found, skipping...")
        return
    
    checkpoint = load_checkpoint()

    for file_path in PRS_DIR.glob("*.json"):
        file_id = f"pr:{file_path.name}"
        if file_id in checkpoint["processed_files"]:
            print(f"Skipping (Already Indexed): {file_path.name}")
            continue

        print(f"Indexing: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            pr = json.load(f)

        body = (pr.get("body", "")).strip()
        if not body:
            continue
        
        doc_text = f"Pull Request #{pr['id']}: {pr['title']}\nDecision/Reasoning: {pr['body'][:2000]}"
        
        collection.add(
            documents=[doc_text],
            metadatas=[{"source": pr['url'], "type": "decision_history", "pr_id": str(pr['id'])}],
            ids=[str(uuid.uuid4())]
        )
        save_checkpoint(file_id)

# --- Runtime Execution ---
if __name__ == "__main__":
    embed_code_contracts()
    embed_code_examples()
    embed_tribal_history()
    embed_decision_history()
    embed_wikis()
    
    print("\n========================================================")
    print("All file structures successfully parsed and embedded!")
    print(f"Vector Database saved cleanly inside: {DB_PATH}")
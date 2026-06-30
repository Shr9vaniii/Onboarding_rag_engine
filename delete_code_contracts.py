import chromadb
import json
import uuid
from pathlib import Path
from collections import Counter

DB_PATH = Path("./enterprise_data/chroma_db")
CODE_DIR = Path("./enterprise_data/code_contracts")

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection("engineering_knowledge")


# ─── STEP 1: Delete all code_contracts ───────────────────────────────────────

def delete_by_type(type_value: str):
    results = collection.get(where={"type": type_value}, include=["metadatas"])
    ids = results["ids"]
    if not ids:
        print(f"  No chunks found with type='{type_value}'")
        return
    batch_size = 1000
    for i in range(0, len(ids), batch_size):
        collection.delete(ids=ids[i:i+batch_size])
    print(f"  Deleted {len(ids)} chunks of type='{type_value}'")

print("Step 1: Deleting code_contracts...")
delete_by_type("code_contracts")


# ─── STEP 2: Delete raw markdown wikis (source starts with "wikis/") ─────────

def delete_markdown_wikis():
    results = collection.get(
        where={"type": "wikis_arch_docs"},
        include=["metadatas"]
    )
    
    # Filter to only raw markdown chunks — their source starts with "wikis/"
    ids_to_delete = [
        id_ for id_, meta in zip(results["ids"], results["metadatas"])
        if meta.get("source", "").startswith("wikis/")
    ]
    
    if not ids_to_delete:
        print("  No raw markdown wiki chunks found.")
        return
    
    batch_size = 1000
    for i in range(0, len(ids_to_delete), batch_size):
        collection.delete(ids=ids_to_delete[i:i+batch_size])
    print(f"  Deleted {len(ids_to_delete)} raw markdown wiki chunks.")

print("Step 2: Deleting raw markdown wiki chunks...")
delete_markdown_wikis()


# ─── STEP 3: Re-embed code contracts with enriched format ────────────────────

def format_contract_chunk(element: dict, source_file: str) -> str:
    if element["type"] in ("function", "async_function"):
        args = element.get("args", [])
        arg_strs = []
        for arg in args:
            name = arg.get("name", "") if isinstance(arg, dict) else str(arg)
            typ = arg.get("type", "") if isinstance(arg, dict) else ""
            arg_strs.append(f"{name}: {typ}" if typ else name)

        is_async = element.get("is_async", False) or element["type"] == "async_function"
        prefix = "async def" if is_async else "def"
        return_type = element.get("return_type", "")
        signature = f"{prefix} {element['name']}({', '.join(arg_strs)})"
        if return_type:
            signature += f" -> {return_type}"

        docstring = element.get("docstring") or "No description available"

        return f"""Function: {element['name']}
Signature: {signature}
File: {source_file}
Description: {docstring}
Usage: Call {element['name']} with arguments: {', '.join(arg_strs) or 'none'}
Returns: {return_type or 'not specified'}"""

    elif element["type"] == "class":
        attrs = element.get("attributes", [])
        attr_strs = []
        for a in attrs:
            if isinstance(a, dict) and a.get("name"):
                t = a.get("type", "")
                attr_strs.append(f"{a['name']}: {t}" if t else a["name"])

        docstring = element.get("docstring") or "No description available"
        first_sentence = docstring.split(".")[0] if docstring else ""

        return f"""Class: {element['name']}
File: {source_file}
Attributes: {', '.join(attr_strs) or 'none defined'}
Description: {docstring}
Purpose: {element['name']} — {first_sentence}"""

    else:
        # Fallback for anything else
        return f"""Contract: {element.get('name', 'unknown')}
Type: {element['type']}
File: {source_file}
Description: {element.get('docstring') or 'No description available'}"""


def embed_code_contracts_enriched():
    print("Step 3: Re-embedding code contracts with enriched format...")
    documents, metadatas, ids = [], [], []
    files_parsed = 0
    contracts_extracted = 0

    for file_path in CODE_DIR.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_file = data.get("file_path", "unknown")
        elements = data.get("elements", [])

        for element in elements:
            # Skip elements with no meaningful content
            name = element.get("name", "")
            if not name or name.startswith("_") and not name.startswith("__"):
                continue  # skip private helpers, keep dunders

            chunk = format_contract_chunk(element, source_file)
            documents.append(chunk)
            metadatas.append({
                "source": source_file,
                "type": "code_contracts",
                "name": name,
                "contract_type": element["type"]
            })
            ids.append(str(uuid.uuid4()))
            contracts_extracted += 1

        files_parsed += 1

    if documents:
        batch_size = 5000
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size],
            )
        print(f"  Parsed {files_parsed} files.")
        print(f"  Re-embedded {contracts_extracted} enriched code contracts.")
    else:
        print("  No contracts found.")


embed_code_contracts_enriched()


# ─── STEP 4: Verify final state ───────────────────────────────────────────────

print("\nStep 4: Final DB state:")
results = collection.get(include=["metadatas"])
types = Counter(m["type"] for m in results["metadatas"])
for t, count in types.most_common():
    print(f"  '{t}': {count} chunks")
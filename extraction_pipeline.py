import os
import json
import shutil
import requests
import subprocess
import ast
from pathlib import Path

#treating FastAPI as my "Enterprise Company"

REPO_OWNER="tiangolo"
REPO_NAME="fastapi"
GITHUB_REPO_URL=f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

DATA_DIR=Path("./enterprise_data")
RAW_REPO_DIR=DATA_DIR/"raw_repo"
DOCS_DIR=DATA_DIR/"wikis_and_docs"
ISSUES_DIR=DATA_DIR/"tribal_history"
CODE_DIR=DATA_DIR/"code_contracts"

GITHUB_TOKEN=os.environ.get("GITHUB_ACCESS_TOKEN")

def setup_data_lake():
    print("Setting up local data Lake directoried....")
    for directory in [DATA_DIR,RAW_REPO_DIR,DOCS_DIR,ISSUES_DIR]:
        directory.mkdir(parents=True,exist_ok=True)
    print("Directories ready.\n")

def clone_repo():
    if(RAW_REPO_DIR/".git").exists():
        print("Repository already existes locally. Skipping clone.\n")
        return
    print(f"Cloning 'Enterprise codebase' ({GITHUB_REPO_URL})...This might take few minutes...")
    try:
        subprocess.run(["git","clone",GITHUB_REPO_URL,str(RAW_REPO_DIR)],check=True)
        print("Repository cloned successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repo: {e}") 
        exit(1)
def extract_architectural_docs():
    #Copying all markdown(.md) files
    print("Extracting English Markdown Documentaion (Internal Wikis)...")
    docs_found=0

    target_dirs=[RAW_REPO_DIR/"docs"/"en",RAW_REPO_DIR/"docs_src"]

    for target_dir in target_dirs:
        if not target_dir.exists():
            continue

        for root, _,files in os.walk(target_dir):
            for file in files:
                if file.endswith(".md"):
                    source_path=Path(root)/   file
                    relative_path=source_path.relative_to(RAW_REPO_DIR) 
                    safe_file_name = str(relative_path).replace("\\", "-").replace("/", "-") 
                    des_path=DOCS_DIR/safe_file_name
                    shutil.copy2(source_path,des_path) 
                    docs_found+=1
    print(f"Extracted {docs_found} English markdown documents to {DOCS_DIR}\n")  

def fetch_tribal_history(limit=50):
    """Hits the GitHub API to fetch closed issues.
    This acts as our "Slack history" or "Jira tickets"-Engineers debate bugs."""    

    print(f"Fetching 'Tribal History' (Last {limit} closed bugs/issues)")

    headers={"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"]=f"token {GITHUB_TOKEN}"
    url=f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues?state=closed&per_page={limit}"

    response=requests.get(url,headers=headers)

    if response.status_code!=200:
        print(f"Error fetching the issues. Status Code: {response.status_code}")
        try:
            print("Message:" ,response.json().get("message"))
        except requests.exceptions.JSONDecodeError:
            print("Raw response: ", response.text)
        
        return
    
    issues=response.json()
    extracted_count=0
    for issue in issues:
        if "pull_request" in issue:
            continue
        issue_data={
            "id":issue["number"],
            "title":issue["title"],
            "url":issue["html_url"],
            "body":issue["body"],
            "labels":[label["name"] for label in issue["labels"]],
        }

        file_path=ISSUES_DIR/ f"issue_{issue["number"]}.json"
        with open(file_path,"w",encoding="utf-8") as f:
            json.dump(issue_data,f,ensure_ascii=False,indent=4)
        extracted_count+=1
    print(f"Extracted {extracted_count}  closed issues to {ISSUES_DIR}\n")

def extract_code_contracts():
    """
    Parses Python files using AST to extract classes, functions, and docstrings.
    This creates clean, searchable 'API Contracts' instead of raw, messy code files.
    """
    print("Extracting API Contracts via AST parser...")
    files_parsed=0
    contracts_extracted=0


    for root, _,files in os.walk(RAW_REPO_DIR):
        for file in files:
            if file.endswith(".py"):
                file_path=Path(root)/file

                try:
                    with open(file_path,"r",encoding="utf-8") as f:
                        file_content=f.read()
                        tree=ast.parse(file_content)

                except Exception:
                    continue

                extracted_elements=[]
                for node in ast.walk(tree):
                    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                        extracted_elements.append({
                            "type":"function",
                            "name":node.name,
                            "docstring":ast.get_docstring(node) or "",
                            "args":[arg.arg for arg in node.args.args],
                            "line_number":node.lineno,
                        })
                    elif isinstance(node,ast.ClassDef):
                        extracted_elements.append({
                            "type":"class",
                            "name":node.name,
                            "docstring":ast.get_docstring(node) or "",
                            "line_number":node.lineno, 
                        })
                if extracted_elements:
                    relative_path=str(file_path.relative_to(RAW_REPO_DIR))
                    safe_filename = relative_path.replace("\\", "-").replace("/", "-").replace(".py", ".json")
                    dest_path=CODE_DIR/safe_filename
                    CODE_DIR.mkdir(parents=True, exist_ok=True)

                    contract_data={
                        "file_path":str(relative_path),
                        "elements":extracted_elements,
                    }
                    with open(dest_path,"w",encoding="utf-8") as f:
                        json.dump(contract_data,f,ensure_ascii=False,indent=4)
                    files_parsed+=1
                    contracts_extracted+=len(extracted_elements)
    print(f"Parsed {files_parsed} Python files and extracted {contracts_extracted} API contracts to {CODE_DIR}\n")    


if __name__=="__main__":
    print("Starting the Extraction Pipeline for Onboarding RAG Engine...\n")    
    setup_data_lake()
    clone_repo()
    extract_architectural_docs()
    fetch_tribal_history(limit=50)
    extract_code_contracts()
    print("Extraction Pipeline completed successfully. Data Lake is ready for RAG Engine.\n")

    
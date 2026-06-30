import os
import json
import shutil
import requests
import subprocess
import ast
import time
from pathlib import Path

REPO_OWNER = "fastapi"       # ← also fix: repo moved to fastapi org, not tiangolo
REPO_NAME = "fastapi"
GITHUB_REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

DATA_DIR = Path("./enterprise_data")
RAW_REPO_DIR = DATA_DIR / "raw_repo"
DOCS_DIR = DATA_DIR / "wikis_and_docs"
ISSUES_DIR = DATA_DIR / "tribal_history"
CODE_DIR = DATA_DIR / "code_contracts"

GITHUB_TOKEN = os.environ.get("GITHUB_ACCESS_TOKEN")
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    **({"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {})
}


def setup_data_lake():
    for directory in [DATA_DIR, RAW_REPO_DIR, DOCS_DIR, ISSUES_DIR, CODE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print("Directories ready.\n")


def clone_repo():
    if (RAW_REPO_DIR / ".git").exists():
        print("Repo already exists locally. Skipping clone.\n")
        return
    print(f"Cloning {GITHUB_REPO_URL}...")
    subprocess.run(["git", "clone", GITHUB_REPO_URL, str(RAW_REPO_DIR)], check=True)
    print("Cloned.\n")


# ─── DOCS ────────────────────────────────────────────────────────────────────

def extract_architectural_docs():
    print("Extracting docs...")
    docs_found = 0
    target_dir = RAW_REPO_DIR / "docs" / "en"  # fix: docs_src has .py not .md
    if target_dir.exists():
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith(".md"):
                    source_path = Path(root) / file
                    relative_path = source_path.relative_to(RAW_REPO_DIR)
                    safe_name = str(relative_path).replace("\\", "-").replace("/", "-")
                    shutil.copy2(source_path, DOCS_DIR / safe_name)
                    docs_found += 1

    # fix: docs_src has example Python files — extract as code examples separately
    extract_docs_src_examples()
    print(f"Extracted {docs_found} markdown docs.\n")


def extract_docs_src_examples():
    """docs_src/ = runnable code examples tied to docs. Extract as-is."""
    docs_src = RAW_REPO_DIR / "docs_src"
    if not docs_src.exists():
        return
    examples_dir = DATA_DIR / "code_examples"
    examples_dir.mkdir(exist_ok=True)
    count = 0
    for root, _, files in os.walk(docs_src):
        for file in files:
            if file.endswith(".py"):
                src = Path(root) / file
                rel = src.relative_to(RAW_REPO_DIR)
                safe = str(rel).replace("\\", "-").replace("/", "-")
                shutil.copy2(src, examples_dir / safe)
                count += 1
    print(f"  Copied {count} docs_src example files.")


# ─── TRIBAL HISTORY ──────────────────────────────────────────────────────────

def paginate_github(url, params=None):
    """Fetch all pages from a GitHub endpoint."""
    params = params or {}
    results = []
    page = 1
    while True:
        response = requests.get(
            url,
            headers=HEADERS,
            params={**params, "per_page": 100, "page": page}
        )
        if response.status_code == 403:
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset_time - int(time.time()), 10)
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        if response.status_code != 200:
            print(f"  API error {response.status_code}: {response.json().get('message')}")
            break
        data = response.json()
        if not data:
            break
        results.extend(data)
        print(f"  Fetched page {page} ({len(data)} items)...")
        page += 1
        time.sleep(0.5)   # respect rate limit
    return results


def fetch_issue_comments(issue_number):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues/{issue_number}/comments"
    return paginate_github(url)


def fetch_tribal_history():
    print("Fetching tribal history (closed issues + comments)...")

    all_issues = paginate_github(
        f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues",
        params={"state": "closed", "labels": "bug"}
    )

    extracted = 0
    for issue in all_issues:
        if "pull_request" in issue:   # issues endpoint returns PRs too
            continue

        comments_raw = fetch_issue_comments(issue["number"])
        comments = [
            {
                "author": c["user"]["login"],
                "body": c["body"],
                "created_at": c["created_at"]
            }
            for c in comments_raw
            if c.get("body") and len(c["body"].strip()) > 20
        ]

        # skip issues with no real discussion
        if not issue.get("body") and not comments:
            continue

        issue_data = {
            "id": issue["number"],
            "title": issue["title"],
            "url": issue["html_url"],
            "body": issue["body"] or "",
            "labels": [label["name"] for label in issue["labels"]],
            "state": issue["state"],
            "comments": comments,           # ← the fix
            "comment_count": len(comments),
        }

        file_path = ISSUES_DIR / f"issue_{issue['number']}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(issue_data, f, ensure_ascii=False, indent=4)
        extracted += 1

    print(f"Extracted {extracted} issues with full comment threads.\n")


def fetch_merged_prs():
    """PRs = architectural decisions. Separate from bugs."""
    print("Fetching merged PRs (decision history)...")
    prs_dir = DATA_DIR / "decision_history"
    prs_dir.mkdir(exist_ok=True)

    all_prs = paginate_github(
        f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls",
        params={"state": "closed"}
    )

    extracted = 0
    for pr in all_prs:
        if not pr.get("merged_at"):   # skip closed-but-not-merged
            continue
        if not pr.get("body"):        # skip PRs with no description
            continue

        comments_raw = fetch_issue_comments(pr["number"])
        comments = [c["body"] for c in comments_raw if len(c.get("body", "")) > 30]

        pr_data = {
            "id": pr["number"],
            "title": pr["title"],
            "url": pr["html_url"],
            "body": pr["body"],
            "merged_at": pr["merged_at"],
            "review_comments": comments[:10],  # cap at 10 to avoid noise
        }

        file_path = prs_dir / f"pr_{pr['number']}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(pr_data, f, ensure_ascii=False, indent=4)
        extracted += 1

    print(f"Extracted {extracted} merged PRs.\n")


# ─── CODE CONTRACTS ──────────────────────────────────────────────────────────

def get_annotation_str(annotation):
    """Convert AST annotation node to readable string."""
    if annotation is None:
        return ""
    try:
        return ast.unparse(annotation)   # Python 3.9+
    except Exception:
        return ""


def extract_code_contracts():
    print("Extracting code contracts via AST...")
    files_parsed = 0
    contracts_extracted = 0

    for root, _, files in os.walk(RAW_REPO_DIR):
        for file in files:
            if not file.endswith(".py"):
                continue
            file_path = Path(root) / file

            try:
                source = file_path.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except Exception:
                continue

            elements = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # fix: extract full type signatures
                    args_with_types = [
                        {
                            "name": arg.arg,
                            "type": get_annotation_str(arg.annotation)
                        }
                        for arg in node.args.args
                    ]
                    elements.append({
                        "type": "function",
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "args": args_with_types,                    # ← fixed
                        "return_type": get_annotation_str(node.returns),  # ← new
                        "line_number": node.lineno,
                    })

                elif isinstance(node, ast.ClassDef):
                    # extract class-level attributes too
                    class_attrs = []
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign):
                            class_attrs.append({
                                "name": ast.unparse(item.target) if hasattr(ast, 'unparse') else "",
                                "type": get_annotation_str(item.annotation)
                            })

                    elements.append({
                        "type": "class",
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "attributes": class_attrs,                  # ← new
                        "line_number": node.lineno,
                    })

            if not elements:
                continue

            relative_path = str(file_path.relative_to(RAW_REPO_DIR))
            safe_name = relative_path.replace("\\", "-").replace("/", "-").replace(".py", ".json")

            contract_data = {
                "file_path": relative_path,
                "elements": elements,
            }
            with open(CODE_DIR / safe_name, "w", encoding="utf-8") as f:
                json.dump(contract_data, f, ensure_ascii=False, indent=4)

            files_parsed += 1
            contracts_extracted += len(elements)

    print(f"Parsed {files_parsed} files, {contracts_extracted} contracts extracted.\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting extraction pipeline...\n")
    setup_data_lake()
    clone_repo()
    #extract_architectural_docs()
    #fetch_tribal_history()
    #fetch_merged_prs()          # ← new — this is your decision_history class
    extract_code_contracts()
    print("Pipeline complete. Data lake ready.\n")
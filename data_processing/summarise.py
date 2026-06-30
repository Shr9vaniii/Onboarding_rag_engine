"""
Issue & PR Batch Summarisation Pipeline
========================================
Reads raw issue/PR JSON files one by one, runs them through Phi-3-mini
via Ollama, and writes a summarised JSON file per record.

Directory structure expected:
  data/scraped/issues/          ← raw issue files (issue_<n>.json)
  data/scraped/prs/             ← raw PR files   (pr_<n>.json)

Directory structure produced:
  data/summarised/issues/       ← summarised issue files
  data/summarised/prs/          ← summarised PR files
  data/summarised/failed/       ← files that failed after all retries
  data/summarised/run_log.jsonl ← append-only log of every run

Resume behaviour:
  - Already summarised files are skipped automatically
  - Failed files are written to data/summarised/failed/ with the error
  - Re-run anytime — only unprocessed files are touched
  - Ctrl+C safely — current file finishes writing before exit

Usage:
  # Make sure Ollama is running first
  ollama serve                          # terminal 1
  python summarise_issues.py            # terminal 2

  # Process only issues
  python summarise_issues.py --kind issues

  # Process only PRs
  python summarise_issues.py --kind prs

  # Dry run — shows what would be processed without calling Ollama
  python summarise_issues.py --dry-run
"""

import argparse
import httpx
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import re

# ── Directory config ─────────────────────────────────────────────────────────
RAW_DIRS = {
    "issues": Path("enterprise_data/tribal_history"),
    "prs":    Path("enterprise_data/scraped/prs"),
}
OUT_DIRS = {
    "issues": Path("enterprise_data/summarised/issues"),
    "prs":    Path("data/summarised/prs"),
}
FAILED_DIR = Path("enterprise_data/summarised/failed")
RUN_LOG    = Path("enterprise_data/summarised/run_log.jsonl")

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "phi3:mini"
TEMPERATURE = 0        # deterministic — critical for consistent JSON
CTX_WINDOW  = 4096
TIMEOUT     = 120      # seconds — phi3 on CPU can be slow
MAX_RETRIES = 3        # retry failed Ollama calls before giving up
RETRY_DELAY = 5        # seconds between retries

# ── Prompts ──────────────────────────────────────────────────────────────────
ISSUE_PROMPT = """\
You are an expert software engineer analyzing GitHub issues from the FastAPI repository.
Your task is to extract engineering knowledge for an internal onboarding knowledge base.

Read the issue title, body and every comment carefully.
The last maintainer comment is almost always the official solution.

Return ONLY valid JSON. No markdown, no explanation, no backticks.

Rules:
- Do not invent information. Use only facts present in the issue.
- If a field has no information use "" for strings and [] for arrays.
- "solution" must come from a maintainer comment (tiangolo) only. If no maintainer commented, use "".
- "workaround" is from community comments — not the official fix.
- "confidence" reflects how clearly the issue was resolved, not how complex it is.

Return this schema exactly:
{{
  "summary": "2-3 sentences covering what happened and how it was resolved.",
  "problem": "The reported behavior or error, one sentence.",
  "root_cause": "Technical cause if discussed in comments, or empty string.",
  "solution": "The officially accepted fix from a maintainer comment, including PR reference if given.",
  "workaround": "Any community-suggested workaround if no official fix exists, or empty string.",
  "context": "Key points from community discussion that add engineering context.",
  "onboarding_note": "One actionable takeaway for a new engineer working on this codebase.",
  "lessons_learned": ["2-5 concise engineering lessons from this issue"],
  "affected_components": ["specific framework concepts affected e.g. APIRouter, Depends, middleware"],
  "keywords": ["5-10 specific technical keywords that will help in semantic searching, not generic ones like fastapi or python"],
  "status": "one of: fixed | workaround | unresolved | feature_request",
  "versions": {{
    "affected": ["versions where this bug exists"],
    "fixed": ["version or PR where it was fixed"]
  }},
  "confidence": "high if maintainer explicitly closed with a fix or PR reference. medium if resolution is implied. low if issue was closed without clear explanation."
}}

Issue:
{content}
"""

PR_PROMPT = """\
You are a technical knowledge extractor for a developer onboarding system.
Your job is to read a GitHub Pull Request and extract the design decision
so new engineers understand why the codebase is built the way it is.

Rules:
- Be concise. No filler words.
- If a field has no information, set it to null.
- "what_changed" must be one sentence maximum.
- "why" must capture the reasoning, not just repeat what changed.
- "alternative" is what was considered and rejected, if mentioned.
- "tags" must be lowercase, technical, max 4 items.
- Output valid JSON only. No explanation, no markdown, no backticks.

Input:
{content}

Output this exact JSON structure:
{{
  "id": "<the id field from the input>",
  "url": "<the url field from the input>",
  "what_changed": "<one sentence: what this PR actually changed>",
  "why": "<the design rationale — why this approach was chosen>",
  "alternative": "<what was considered and rejected, or null>",
  "files_affected": ["<core file 1>", "<core file 2>"],
  "tags": ["<tag1>", "<tag2>"]
}}
"""

PROMPTS = {"issues": ISSUE_PROMPT, "prs": PR_PROMPT}


# ── Ollama caller ─────────────────────────────────────────────────────────────
def call_ollama(prompt: str) -> str:
    """
    Calls Ollama with retries. Returns raw response text.
    Raises RuntimeError after MAX_RETRIES failures.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = httpx.post(OLLAMA_URL, json={
                "model":   MODEL,
                "prompt":  prompt,
                "stream":  False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_ctx":     CTX_WINDOW,
                },
            }, timeout=TIMEOUT)

            resp.raise_for_status()
            return resp.json()["response"].strip()

        except httpx.TimeoutException:
            last_error = f"Timeout after {TIMEOUT}s"
        except httpx.ConnectError:
            last_error = "Cannot connect to Ollama — is 'ollama serve' running?"
            # No point retrying a connection error immediately
            raise RuntimeError(last_error)
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_RETRIES:
            print(f"    ⚠ Attempt {attempt} failed: {last_error} — retrying in {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}")


# ── JSON parser ───────────────────────────────────────────────────────────────
def parse_json_response(raw: str) -> dict:
    """
    Safely parses JSON from model output.
    Handles cases where the model wraps output in ```json ... ``` despite instructions.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(
            line for line in inner if line.strip() != "```"
        ).strip()

    # Find the JSON object boundaries in case there's preamble text
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response:\n{raw[:300]}")
    
    parsed=json.loads(text[start:end])

    return normalise_output(parsed)


def normalise_output(record: dict)->dict:
    """Enforces correct typr on every field. Prevents [] vs "" vs null inconsistencies.
    Called immediately after parse_json_response()"""

    string_fields=["summary","problem","root_cause","solution","workaround","context","onboarding_note","status","confindence"]

    list_fields=["lessons_learned","affected_components","keywords"]

    for f in string_fields:
        val=record.get(f)
        if not isinstance(val,str):
            record[f]="" if val is None else str(val)

    for f in list_fields:
        val =record.get(f)
        if not isinstance(val,list):
            record[f]=[str(i) for i in val if i]

    versions=record.get("versions")
    if not isinstance(versions,dict):
        record["versions"]={"affected":[],"fixed":[]}
    else:
        for key in ("affected","fixed"):
            v=versions.get(key)
            if not isinstance(v,list):
                versions[key]=[str(v) if isinstance(v,str) and v else []]

    if record.get("status") not in{"fixed","workaround","unresolved","feature_request"}:
        record["status"]="unresolved"
    if record.get("confidence") not in {"high","medium","low"}:
        record["confidence"]="low"
    return record

# ── GitHub text sanitiser ────────────────────────────────────────────────────
def sanitise_github_text(text: str) -> str:
    """
    Cleans raw GitHub issue/PR body text before sending to the LLM.
    Removes things that confuse small models:
      - Code fences (``` blocks) — replaced with a plain label
      - GitHub issue template headers (**Additional context**, etc.)
      - HTML comments <!-- -->
      - Excessive blank lines
      - Raw error stack traces (keep first 3 lines only)
      - \r\n Windows line endings
    """
    

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    #removes image markdown
    text=re.sub(r"!\\[.*?\\]\\(.*?\\)","",text)
    #removes github issue checklist
    text=re.sub(r"^- \[[ xX]\].*$","",text,flags=re.MULTILINE)

    # Replace code fences with a plain marker — keep the code, drop the fences
    # so the model sees the content without the ``` confusing it
    def flatten_code_block(m):
        lang    = m.group(1).strip() or "code"
        content = m.group(2).strip()
        # For long stack traces keep only first 3 lines
        lines = content.splitlines()
        if len(lines) > 8:
            content = "\n".join(lines[:3]) + f"\n... [{len(lines) - 3} more lines]"
        return f"[{lang} snippet]: {content}"

    text = re.sub(r"```(\w*)\n(.*?)```", flatten_code_block, text, flags=re.DOTALL)

    # Remove GitHub issue template bold headers that add no signal
    template_headers = [
        r"\*\*Describe the bug\*\*",
        r"\*\*To Reproduce\*\*",
        r"\*\*Expected behavior\*\*",
        r"\*\*Additional context\*\*",
        r"\*\*Screenshots\*\*",
        r"\*\*Desktop \(please complete.*?\)\*\*",
        r"\*\*Smartphone \(please complete.*?\)\*\*",
        r"\*\*Environment\*\*",
        r"\*\*FastAPI Version\*\*",
        r"\*\*Python Version\*\*",
    ]
    for pattern in template_headers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Collapse 3+ blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def build_model_input(raw_record: dict)->str:
    """Strips the record to only what the model needs to summarise.
    Sends only title,body,comments"""

    lean={
        "title": raw_record.get("title",""),
        "body":raw_record.get("body",""),
        "comments":[
            {
                "author":c.get("author",""),
                "body":c.get("body","")
            }
            for c in raw_record.get("comments",[])
            if c.get("body","").strip()
        ]
    }

    return json.dumps(lean,indent=2,ensure_ascii=False)


# ── Per-file processor ────────────────────────────────────────────────────────
def process_file(raw_path: Path, out_path: Path, kind: str) -> dict:
    """
    Reads one raw JSON file, calls Ollama, writes summarised JSON.
    Returns the summarised record.
    """
    raw_record = json.loads(raw_path.read_text(encoding="utf-8"))

    # ── Sanitise all text fields before sending to the model ─────────────
    # GitHub bodies contain raw markdown, code fences, template headers,
    # and \r\n line endings that confuse small models like Phi-3
    for field in ("body", "title"):
        if isinstance(raw_record.get(field), str):
            raw_record[field] = sanitise_github_text(raw_record[field])

    # Sanitise issue comments array (this is where solutions live)
    for comment in raw_record.get("comments", []):
        if isinstance(comment.get("body"), str):
            comment["body"] = sanitise_github_text(comment["body"])

    # Sanitise discussion top_comments (different field name)
    """for comment in raw_record.get("top_comments", []):
        if isinstance(comment.get("body"), str):
            comment["body"] = sanitise_github_text(comment["body"])

    if isinstance(raw_record.get("answer"), dict):
        if isinstance(raw_record["answer"].get("body"), str):
            raw_record["answer"]["body"] = sanitise_github_text(raw_record["answer"]["body"])
            """

    # ── Trim to context window ────────────────────────────────────────────
    content = build_model_input(raw_record)
    if len(content) > 8000:
        if isinstance(raw_record.get("boody"),str):
            raw_record["body"]=raw_record["body"][:2000] + "... [truncated]"
        raw_record["comments"] =raw_record.get("comments",[])[:6]
        content=build_model_input(raw_record)

    prompt    = PROMPTS[kind].format(content=content)
    raw_resp  = call_ollama(prompt)
    summarised = parse_json_response(raw_resp)

    # Always preserve source metadata even if model drops it
    summarised["id"]=raw_record.get("id",  raw_path.stem)
    summarised["url"]=raw_record.get("url", "")
    summarised["_source_file"]    = str(raw_path)
    summarised["_summarised_at"]  = datetime.utcnow().isoformat()
    summarised["_model"]          = MODEL
    summarised["collection"]      = kind  # "issues" or "prs"
    summarised["source"]          = "decision_history" if kind == "prs" else "tribal_history"

    out_path.write_text(
        json.dumps(summarised, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return summarised


# ── Run logger ────────────────────────────────────────────────────────────────
def log_run(entry: dict):
    """Appends one line to the run log — never overwrites, always appends."""
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main batch runner ─────────────────────────────────────────────────────────
def run_batch(kind: str, dry_run: bool = False):
    raw_dir = RAW_DIRS[kind]
    out_dir = OUT_DIRS[kind]

    if not raw_dir.exists():
        print(f"❌ Raw directory not found: {raw_dir}")
        print(f"   Run the scraper first to populate it.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all raw files
    all_files   = sorted(raw_dir.glob("*.json"))
    total       = len(all_files)

    if total == 0:
        print(f"⚠  No JSON files found in {raw_dir}")
        return

    # Separate into pending vs already done
    pending = [f for f in all_files if not (out_dir / f.name).exists()]
    done    = total - len(pending)

    print(f"\n{'─' * 60}")
    print(f"  Kind      : {kind}")
    print(f"  Model     : {MODEL}")
    print(f"  Total     : {total} files")
    print(f"  Already done : {done} (skipping)")
    print(f"  To process   : {len(pending)}")
    print(f"  Output    : {out_dir}/")
    if dry_run:
        print(f"\n  DRY RUN — no Ollama calls will be made")
        for f in pending[:10]:
            print(f"    would process: {f.name}")
        if len(pending) > 10:
            print(f"    ... and {len(pending) - 10} more")
        return
    print(f"{'─' * 60}\n")

    if not pending:
        print("✅ All files already summarised. Nothing to do.")
        return

    # ── Estimate time ─────────────────────────────────────────────────────
    # Phi-3-mini on CPU: ~30s per file is a reasonable estimate
    est_seconds = len(pending) * 30
    est_hours   = est_seconds / 3600
    print(f"  ⏱  Estimated time on CPU (~30s/file): {est_hours:.1f} hours")
    print(f"     Tip: run overnight with: nohup python summarise_issues.py &\n")

    succeeded = 0
    failed    = 0

    for i, raw_path in enumerate(pending, 1):
        out_path    = out_dir / raw_path.name
        failed_path = FAILED_DIR / f"{kind}_{raw_path.name}"
        prefix      = f"  [{i:>4}/{len(pending)}]"

        print(f"{prefix} {raw_path.name}", end="", flush=True)
        t_start = time.time()

        try:
            result   = process_file(raw_path, out_path, kind)
            elapsed  = time.time() - t_start
            succeeded += 1

            # Print one-line summary of what was extracted
            problem  = result.get("problem") or result.get("what_changed") or "—"
            sol_type = result.get("solution_type") or result.get("tags", [])
            print(f"  ✓  {elapsed:.0f}s  [{sol_type}]")
            print(f"           {problem[:80]}")

            log_run({
                "ts": datetime.utcnow().isoformat(), "kind": kind,
                "file": raw_path.name, "status": "ok", "elapsed": elapsed,
            })

        except KeyboardInterrupt:
            print(f"\n\n  ⚡ Interrupted by user after {succeeded} files.")
            print(f"     Progress is saved — re-run to continue from here.")
            break

        except Exception as e:
            elapsed = time.time() - t_start
            failed += 1
            err_msg = traceback.format_exc()

            # Write failure record so we can inspect and retry
            failed_path.write_text(json.dumps({
                "file":    str(raw_path),
                "error":   str(e),
                "trace":   err_msg,
                "failed_at": datetime.utcnow().isoformat(),
            }, indent=2), encoding="utf-8")

            print(f"  ✗  {elapsed:.0f}s  ERROR: {str(e)[:80]}")
            log_run({
                "ts": datetime.utcnow().isoformat(), "kind": kind,
                "file": raw_path.name, "status": "failed", "error": str(e),
            })
            # Don't stop — move to next file
            continue

    # ── Final stats ───────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  ✅ Succeeded : {succeeded}")
    print(f"  ❌ Failed    : {failed}  (see {FAILED_DIR}/)")
    print(f"  ⏭  Skipped  : {done} (already done)")
    print(f"{'─' * 60}\n")

    if failed:
        print(f"  To retry failed files:")
        print(f"    python retry_failed.py --kind {kind}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch summarise issues/PRs with Phi-3-mini")
    parser.add_argument(
        "--kind", choices=["issues", "prs", "both"],
        default="both",
        help="Which kind to process (default: both)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without calling Ollama"
    )
    args = parser.parse_args()

    kinds = ["issues", "prs"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        print(f"\n{'═' * 60}")
        print(f"  PROCESSING: {kind.upper()}")
        print(f"{'═' * 60}")
        run_batch(kind, dry_run=args.dry_run)
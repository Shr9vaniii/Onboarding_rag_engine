"""
Retry Failed Summarisations
============================
Reads files from data/summarised/failed/, re-runs them through Ollama,
and cleans them out of the failed directory on success.

Usage:
  python retry_failed.py               # retry everything
  python retry_failed.py --kind issues # retry only issues
  python retry_failed.py --kind prs    # retry only prs
"""

import argparse
import json
from pathlib import Path
from summarise import (
    process_file, FAILED_DIR, OUT_DIRS, log_run
)
from datetime import datetime

def retry_failed(kind: str | None = None):
    pattern = f"{kind}_*.json" if kind else "*.json"
    failed_files = sorted(FAILED_DIR.glob(pattern))

    if not failed_files:
        print(f"✅ No failed files found in {FAILED_DIR}/")
        return

    print(f"\n  Retrying {len(failed_files)} failed files...\n")
    recovered = 0
    still_failing = 0

    for failed_path in failed_files:
        # Read the failure record to get the original file path
        failure = json.loads(failed_path.read_text())
        raw_path = Path(failure["file"])

        # Determine kind from filename prefix (issues_ or prs_)
        file_kind = "prs" if failed_path.name.startswith("prs_") else "issues"
        out_path  = OUT_DIRS[file_kind] / raw_path.name

        print(f"  Retrying {raw_path.name}...", end="", flush=True)

        try:
            process_file(raw_path, out_path, file_kind)
            failed_path.unlink()  # remove from failed dir on success
            recovered += 1
            print(f"  ✓ recovered")
            log_run({
                "ts": datetime.utcnow().isoformat(),
                "kind": file_kind, "file": raw_path.name,
                "status": "recovered",
            })
        except Exception as e:
            still_failing += 1
            print(f"  ✗ still failing: {str(e)[:80]}")

    print(f"\n  ✅ Recovered : {recovered}")
    print(f"  ❌ Still failing: {still_failing}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["issues", "prs"], default=None)
    args = parser.parse_args()
    retry_failed(args.kind)
import json
from pathlib import Path

ISSUES_DIR = Path("./enterprise_data/tribal_history")
PRS_DIR = Path("./enterprise_data/decision_history")

issue_files = list(ISSUES_DIR.glob("*.json"))
pr_files = list(PRS_DIR.glob("*.json")) if PRS_DIR.exists() else []

total_comments = 0
issues_with_comments = 0
empty_bodies = 0

for f in issue_files:
    data = json.loads(f.read_text(encoding="utf-8"))
    comments = data.get("comments", [])
    total_comments += len(comments)
    if comments:
        issues_with_comments += 1
    if not data.get("body", "").strip():
        empty_bodies += 1

print(f"Issues scraped:          {len(issue_files)}")
print(f"With comments:           {issues_with_comments}")
print(f"Empty bodies:            {empty_bodies}")
print(f"Total comment threads:   {total_comments}")
print(f"Avg comments/issue:      {total_comments/len(issue_files):.1f}")
print(f"\nPRs scraped:             {len(pr_files)}")
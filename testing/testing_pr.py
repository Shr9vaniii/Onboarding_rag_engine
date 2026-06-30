import json
from pathlib import Path


def audit_decision_history_filter():
    """Run before embedding — see what passes/fails"""
    DATA_DIR=Path("./enterprise_data")
    PRS_DIR=DATA_DIR/"decision_history"
    pr_files = list(PRS_DIR.glob("*.json"))
    
    total_comments = 0
    passed = 0
    failed_noise = 0
    failed_short = 0
    failed_no_signal = 0
    
    for f in pr_files[:200]:   # sample 200 PRs
        pr = json.loads(f.read_text(encoding="utf-8"))
        comments = pr.get("review_comments", [])
        
        for c in comments:
            total_comments += 1
            c_lower = c.lower()
            
            noise_patterns = ["codecov","coverage","[![",
                              "lgtm","thank","great job",
                              "congrats",":tada:",":rocket:",
                              "may i quote","would you mind","please review"]
            signal_patterns = ["because","instead of","fix","issue #",
                               "this means","the problem","approach",
                               "alternatively","breaking","deprecat"]
            
            if any(p in c_lower for p in noise_patterns):
                failed_noise += 1
            elif len(c.strip()) < 80:
                failed_short += 1
            elif any(p in c_lower for p in signal_patterns) or len(c) > 300:
                passed += 1
            else:
                failed_no_signal += 1
    
    print(f"Sample: 200 PRs, {total_comments} total comments")
    print(f"  Passed filter:     {passed}  ({100*passed/total_comments:.1f}%)")
    print(f"  Rejected (noise):  {failed_noise}  ({100*failed_noise/total_comments:.1f}%)")
    print(f"  Rejected (short):  {failed_short}  ({100*failed_short/total_comments:.1f}%)")
    print(f"  Rejected (no sig): {failed_no_signal}  ({100*failed_no_signal/total_comments:.1f}%)")


if __name__=="__main__":
    audit_decision_history_filter()
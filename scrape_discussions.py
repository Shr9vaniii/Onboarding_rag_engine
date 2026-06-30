
import httpx
import json
import os
import time
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
GRAPHQL_URL = "https://api.github.com/graphql"


# Signal thresholds — tune these based on your quality bar
MIN_DISCUSSION_UPVOTES = 1       # ▲ arrow on discussion itself

TOP_N_COMMENTS         = 3       # max comments to include per discussion

TOKEN = os.environ.get("GH_TOKEN", "")
HEADERS = {
    "Accept": "application/vnd.github+json",
    **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}),
}
OUTPUT_DIR = Path("enterprise_data/discussion_new")
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

# ── GraphQL query ────────────────────────────────────────────────────────────
# Fetches discussions sorted by TOP (most upvoted first).
# Includes up to 20 comments per discussion with their reaction counts.
DISCUSSION_QUERY='''
query FetchDiscussions($cursor: String) {
  repository(owner: "fastapi", name: "fastapi") {
    discussions(
      first: 50
      after: $cursor
      orderBy: { field: UPDATED_AT, direction: DESC }
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }

      nodes {
        number
        title
        body
        createdAt
        updatedAt
        upvoteCount
        answerChosenAt
        url

        category {
          name
          slug
        }

        labels(first: 10) {
          nodes {
            name
          }
        }

        comments(first: 20) {
          totalCount

          nodes {
            id
            body
            createdAt
            isAnswer
            upvoteCount

            author {
              login
            }

            reactions(content: THUMBS_UP) {
              totalCount
            }
          }
        }
      }
    }
  }
}'''

# ── Scraper ──────────────────────────────────────────────────────────────────
def scrape_discussions(max_pages: int = 100) -> list[dict]:
    """
    Fetches all discussions sorted by upvotes.
    Stops early once upvoteCount drops below MIN_DISCUSSION_UPVOTES AND
    the discussion is not answered — since results are sorted by TOP,
    once we're seeing low-signal threads we can stop.
    """
    all_discussions = []
    cursor = None
    page = 0
    consecutive_low_signal = 0

    print(f"Starting scrape (auth={'yes' if TOKEN else 'no — add GH_TOKEN for higher rate limits'})")
    print(f"Filter: upvotes >= {MIN_DISCUSSION_UPVOTES} OR answered")
    print("-" * 60)

    while page < max_pages:
        resp = httpx.post(
            GRAPHQL_URL,
            headers=HEADERS,
            json={"query": DISCUSSION_QUERY, "variables": {"cursor": cursor}},
            timeout=30,
        )

        if resp.status_code == 401:
            print("❌ Auth error — check your GH_TOKEN")
            break
        if resp.status_code == 403:
            print("❌ Rate limited — wait or add a GH_TOKEN")
            break

        data = resp.json()

        if "errors" in data:
            print(f"❌ GraphQL error: {data['errors']}")
            break

        page_data = data["data"]["repository"]["discussions"]
        nodes = page_data["nodes"]

        extracted=0

        for node in nodes:
            is_answered  = bool(node.get("answerChosenAt"))
            upvotes      = node.get("upvoteCount", 0)
            is_questions = node["category"]["slug"] == "questions"

            # Skip non-question categories (translations, show-and-tell)
            if not is_questions:
                continue

            # Apply signal filter
            passes_filter = is_answered or (upvotes >= MIN_DISCUSSION_UPVOTES)
            if not passes_filter:
                consecutive_low_signal += 1
                # Since sorted by TOP, stop if we see 20 consecutive low-signal threads
                if consecutive_low_signal >= 20:
                    print(f"  → 20 consecutive low-signal threads, stopping early at page {page+1}")
                    return all_discussions
                continue

            consecutive_low_signal = 0  # reset counter on a good hit

            # ── Extract comments ──────────────────────────────────────────
            comments = node["comments"]["nodes"]

            # Identify the marked answer
            marked_answer = next(
                (c for c in comments if c["isAnswer"]),
                None
            )
            non_answer_comments=[c for c in comments if not c['isAnswer']]

            # Top comments by thumbs-up, excluding the marked answer
            # (it's included separately)
           

            top_comments = sorted(
                non_answer_comments,
                key=lambda c: max(c["reactions"]["totalCount"],c['upvoteCount']),
                reverse=True
                )[:TOP_N_COMMENTS]
            

            # ── Build structured record ───────────────────────────────────
            record = {
                "id":           f"discussion_{node['number']}",
                "source":       "tribal_history",
                "collection":   "discussions",
                "number":       node["number"],
                "title":        node["title"],
                "body":         node["body"],
                "created_at":   node["createdAt"],
                "updated_at":   node["updatedAt"],
                "upvotes":      upvotes,
                "is_answered":  is_answered,
                "total_comments": node["comments"]["totalCount"],
                "labels":       [l["name"] for l in node["labels"]["nodes"]],
                "url":          node["url"],

                # The accepted answer (if marked)
                "answer": {
                    "author":    marked_answer["author"]["login"] if marked_answer.get("author") else "unknown" if marked_answer else None,
                    "body":      marked_answer["body"] if marked_answer else None,
                    "upvotes":   marked_answer["upvoteCount"] if marked_answer else 0,
                    "thumbsup":  marked_answer["reactions"]["totalCount"] if marked_answer else 0,
                } if marked_answer else None,

                # Top upvoted comments (not the answer) 
                "top_comments": [
                    {
                        "author":   c["author"]["login"] if c.get("author") else "unknown",
                        "body":     c["body"],
                        "thumbsup": c["reactions"]["totalCount"],
                        "upvotes":  c["upvoteCount"],
                    }
                    for c in top_comments
                ],
            }

            if record:
                file_name=OUTPUT_DIR/f"Discussion_{node['number']}.json"
                with open(file_name,"w",encoding="utf-8") as f:
                    json.dump(record,f,indent=4)
                extracted+=1    

            

        print(f"  Page {page+1}: fetched {len(nodes)} threads | kept {extracted} total")

        if not page_data["pageInfo"]["hasNextPage"]:
            print("  → No more pages")
            break

        cursor = page_data["pageInfo"]["endCursor"]
        page += 1
        time.sleep(0.8)   # polite delay between requests

    print("Extraction done!")



if __name__=="__main__":
    scrape_discussions()
    




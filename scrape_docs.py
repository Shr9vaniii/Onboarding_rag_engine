import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path

BASE_URL = "https://fastapi.tiangolo.com"
OUTPUT_DIR = Path("./enterprise_data/wikis_and_docs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Every known subsection — hardcoded because JS nav isn't crawlable
ALL_SECTIONS = [
    # Reference — API contracts with prose explanation
    "/reference/fastapi/",
    "/reference/apirouter/",
    "/reference/parameters/",
    "/reference/exceptions/",
    "/reference/response/",
    "/reference/responses/",
    "/reference/middleware/",
    "/reference/background/",
    "/reference/uploadfile/",
    "/reference/testclient/",
    "/reference/httpconnection/",
    "/reference/request/",
    "/reference/websockets/",
    "/reference/dependencies/",
    "/reference/security/",
    "/reference/staticfiles/",
    "/reference/templating/",
    "/reference/openapi/docs/",
    "/reference/openapi/models/",
    "/reference/encoders/",

    # Tutorial — how-to guides
    "/tutorial/first-steps/",
    "/tutorial/path-params/",
    "/tutorial/query-params/",
    "/tutorial/body/",
    "/tutorial/body-multiple-params/",
    "/tutorial/query-params-str-validations/",
    "/tutorial/path-params-numeric-validations/",
    "/tutorial/body-fields/",
    "/tutorial/body-nested-models/",
    "/tutorial/response-model/",
    "/tutorial/extra-models/",
    "/tutorial/response-status-code/",
    "/tutorial/request-forms/",
    "/tutorial/request-files/",
    "/tutorial/request-forms-and-files/",
    "/tutorial/handling-errors/",
    "/tutorial/path-operation-configuration/",
    "/tutorial/encoder/",
    "/tutorial/body-updates/",
    "/tutorial/dependencies/",
    "/tutorial/dependencies/classes-as-dependencies/",
    "/tutorial/dependencies/sub-dependencies/",
    "/tutorial/dependencies/dependencies-in-path-operation-decorators/",
    "/tutorial/dependencies/global-dependencies/",
    "/tutorial/dependencies/dependencies-with-yield/",
    "/tutorial/security/",
    "/tutorial/security/oauth2-jwt/",
    "/tutorial/security/http-basic-auth/",
    "/tutorial/middleware/",
    "/tutorial/cors/",
    "/tutorial/sql-databases/",
    "/tutorial/bigger-applications/",
    "/tutorial/background-tasks/",
    "/tutorial/metadata/",
    "/tutorial/static-files/",
    "/tutorial/testing/",
    "/tutorial/debugging/",

    # Advanced
    "/advanced/path-operation-advanced-configuration/",
    "/advanced/additional-status-codes/",
    "/advanced/response-directly/",
    "/advanced/custom-response/",
    "/advanced/websockets/",
    "/advanced/events/",
    "/advanced/middleware/",
    "/advanced/sql-databases-peewee/",
    "/advanced/async-sql-databases/",
    "/advanced/nosql-databases/",
    "/advanced/sub-applications/",
    "/advanced/behind-a-proxy/",
    "/advanced/templates/",
    "/advanced/graphql/",
    "/advanced/testing-websockets/",
    "/advanced/testing-events/",
    "/advanced/testing-dependencies/",
    "/advanced/async-tests/",
    "/advanced/settings/",
    "/advanced/openapi-callbacks/",
    "/advanced/openapi-webhooks/",
    "/advanced/generate-clients/",

    # Deployment
    "/deployment/concepts/",
    "/deployment/docker/",
    "/deployment/server-workers/",
    "/deployment/https/",
    "/deployment/manually/",

    # How-to
    "/how-to/general/",
    "/how-to/graphql/",
    "/how-to/custom-request-and-route/",
    "/how-to/conditional-openapi/",
    "/how-to/extending-openapi/",
    "/how-to/custom-docs-ui-assets/",
    "/how-to/configure-swagger-ui/",
    "/how-to/separate-openapi-schemas/",
    "/how-to/custom-openapi-ui-oauth2/",
]


def scrape_page(path: str) -> dict | None:
    url = BASE_URL + path
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"  SKIP {path} — status {response.status_code}")
            return None
    except Exception as e:
        print(f"  ERROR {path} — {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.find("article") or soup.find("main")
    if not article:
        return None

    text = article.get_text(separator="\n", strip=True)
    if len(text) < 100:
        return None

    h1 = soup.find("h1")
    section = path.strip("/").split("/")[0]  # tutorial / advanced / reference etc

    return {
        "url": url,
        "path": path,
        "section": section,
        "title": h1.get_text(strip=True) if h1 else path,
        "content": text,
    }


def scrape_all():
    print(f"Scraping {len(ALL_SECTIONS)} pages from fastapi.tiangolo.com...\n")
    results = []

    for path in ALL_SECTIONS:
        data = scrape_page(path)
        if data:
            safe_name = path.strip("/").replace("/", "-") + ".json"
            out_path = OUTPUT_DIR / safe_name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ [{data['section']:12s}] {data['title'][:55]}")
            results.append(data)
        time.sleep(0.4)

    print(f"\nDone. Scraped {len(results)}/{len(ALL_SECTIONS)} pages.")
    return results


if __name__ == "__main__":
    scrape_all()
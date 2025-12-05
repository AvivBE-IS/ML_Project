# 1. Validaion (The Index)
# Before doing anything, the code checks a special list called index.json.
# The code looks at the list to see which articles it already has.
# It doesn't want to download the same article twice (duplicates are a waste of time).

# 2. The API Request
# The code sends a message to The Guardian's server (the website).
# It says: "Hey, send me the 5 newest articles about Sports."
# The server replies with a big chunk of data containing the headlines and text.

# 3. The Quality Control (Filtering)
# The code looks at what the server sent back. It is very picky:
# Is it empty? If the article has no text, the code throws it away.
# Do I have it? It checks the list again. If the ID matches one you already have, it skips it.

# 4. Formatting
# Raw data from the web looks messy and computer-coded. The code cleans it up to make it readable for humans.
# It creates a nice Header (Title, Author, Date, Link).
# It pastes the Story underneath.
# It saves this as a neat .txt file in the correct folder (like putting a Sport article into the Sport folder).

# 5. Updating the List
# Finally, the code takes a pen and updates the index.json list.
# It says: "Okay, I just saved Article #500. I'll write that down so I never download it again."

# Import

import time
import json
from pathlib import Path
import requests
from requests.exceptions import RequestException
from secrets import API_KEY

API_KEY = "fc5a20ca-99da-44a1-8b5b-dfcbb27c1fb7"
BASE_URL = "https://content.guardianapis.com/search"
INDEX_PATH = Path("data") / "index.json"

SECTION_TO_FOLDER = {
    "news": "news",
    "sport": "sport",
    "commentisfree": "opinion",   # map commentisfree -> opinion
    "culture": "culture",
}


API_KEY = "fc5a20ca-99da-44a1-8b5b-dfcbb27c1fb7"
BASE_URL = "https://content.guardianapis.com/search"
INDEX_PATH = Path("data") / "index.json"

SECTION_TO_FOLDER = {
    "news": "news",
    "sport": "sport",
    "commentisfree": "opinion",   # map commentisfree -> opinion
    "culture": "culture",
}
def load_index():
    if INDEX_PATH.exists():
        try:
            return set(json.loads(INDEX_PATH.read_text(encoding="utf-8")))
        except Exception:
            print("Warning: index.json corrupted, rebuilding index from files.")
            return rebuild_index_from_files()
    return rebuild_index_from_files()

def rebuild_index_from_files():
    idx = set()
    base = Path("data")
    if not base.exists():
        return idx
    for section_dir in base.iterdir():
        if section_dir.is_dir():
            for f in section_dir.glob("*.txt"):
                idx.add(f.stem)  # filename without suffix is the article_id
    return idx

def save_index(index_set):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(index_set)), encoding="utf-8")
    tmp.replace(INDEX_PATH)

def fetch_section_latest(section, page_size=1, retries=2, backoff=1.0):
    params = {
        "api-key": API_KEY,
        "page-size": page_size,
        "section": section,
        "order-by": "newest",
        "show-fields": "headline,byline,bodyText",
    }
    attempt = 0
    while attempt <= retries:
        try:
            resp = requests.get(BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()["response"].get("results", [])
        except RequestException as e:
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(backoff * attempt)

def save_article(article, section, index_set):
    fields = article.get("fields", {})
    body = (fields.get("bodyText") or "").strip()
    if not body:
        print(f"Skipped {article.get('id')} (no bodyText).")
        return False

    article_id = article["id"].replace("/", "_")
    folder_name = SECTION_TO_FOLDER.get(section, section)  # map id -> folder
    out_dir = Path("data") / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{article_id}.txt"

    if article_id in index_set or out_path.exists():
        print(f"Already exists: {article_id} (skipping).")
        index_set.add(article_id)
        return False

    section_name = article.get("sectionName", folder_name)
    date = (article.get("webPublicationDate") or "")[:10]
    byline = fields.get("byline", "Unknown Author")
    url = article.get("webUrl", "")
    title = fields.get("headline", article.get("webTitle", "Untitled"))
    header = (
        f"The Guardian | {section_name} | {date}\n"
        f"By {byline}\n"
        f"{url}\n\n"
        f"{title}\n"
        f"{'-' * 60}\n"
    )

    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(f"{header}{body}\n", encoding="utf-8")
    tmp.replace(out_path)

    index_set.add(article_id)
    print(f"Added: {article_id} -> {out_path}")
    return True
# Main Function

def main():
    sections = ["news", "sport", "commentisfree", "culture"]
    index = load_index()
    saved_count = 0
    skipped_count = 0

    for sec in sections:
        try:
            results = fetch_section_latest(sec, page_size=5)
        except RequestException:
            print(f"Failed to fetch section {sec}, skipping.")
            continue

        if not results:
            print(f"No results for section {sec}.")
            continue

        for article in results:
            try:
                saved = save_article(article, sec, index)
                if saved:
                    saved_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"Error saving {article.get('id')}: {e}")

        time.sleep(1.0)

    save_index(index)
    print(f"Done. New articles saved: {saved_count}. Articles skipped: {skipped_count}.")

if __name__ == "__main__":
    main()


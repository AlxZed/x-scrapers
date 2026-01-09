from __future__ import annotations
import os
import re
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from pymongo import MongoClient, UpdateOne
from anthropic import Anthropic

from arxiv_fetch import fetch_arxiv_metadata
from categorize import categorize_paper_with_claude

# =========================
# Config
# =========================

FETCH_TIMEOUT = 25
SOCIALDATA_BEARER = os.getenv("SOCIALDATA_BEARER", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY", "")

MIN_FAVES = 10
WITHIN_TIME = "7d"

client = MongoClient(os.environ["MONGO_URI_HEADLINE"])

# Use sources database for both collections
PAPERS = client.get_database("sources").arxiv
TWEETS = client.get_database("sources").twitter_arxiv

SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": SOCIALDATA_BEARER,
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0"
})

CLAUDE = Anthropic(api_key=ANTHROPIC_KEY)
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")

# =========================
# Helpers
# =========================


def _fmt_dt(dt: datetime) -> str:
    try:
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt)


def _hours_ago(dt: datetime) -> str:
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    delta = datetime.utcnow() - dt
    hrs = max(0, int(delta.total_seconds() // 3600))
    return f"{hrs}h ago"


def _print_fetched_tweet(username: str, likes: int, tweet_time: datetime):
    print(f"📡 @{username} | {likes} Likes | {_fmt_dt(tweet_time)}")


def _print_tweet_summary(username: str, likes: int, tweet_url: str,
                         tweet_time: datetime):
    print(
        f"✅ @{username} | {likes} Likes | {_hours_ago(tweet_time)} | {tweet_url}"
    )


def update_metrics_in_arxiv_db(
    arxiv_id: str,
    like_count: int,
    reply_count: int,
    retweet_count: int,
    tweet_id: int | str,
    tweet_url: str,
):
    try:
        PAPERS.update_one(
            {"arxiv_id": arxiv_id},
            {
                "$set": {
                    "like_count": like_count,
                    "reply_count": reply_count,
                    "retweet_count": retweet_count,
                    "tweet_id": str(tweet_id),  # ← REQUIRED
                    "tweet_url": tweet_url,  # ← REQUIRED
                    "metrics_updated_at": datetime.utcnow(),
                }
            },
            upsert=True,
        )
        print(f"📊 Updated metrics in sources.arxiv for {arxiv_id}")
    except Exception as e:
        print(
            f"⚠️ Failed to update metrics in sources.arxiv for {arxiv_id}: {e}"
        )


def fetch_thread(tweet_id: int | str):
    """Fetch full thread from SocialData API."""
    url = f"https://api.socialdata.tools/twitter/thread/{tweet_id}"
    try:
        r = SESSION.get(url, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def expand_tco(url: str) -> str:
    try:
        r = requests.get(url, allow_redirects=True, timeout=7)
        return r.url
    except Exception:
        return url


def expand_urls(urls: list[str]) -> list[str]:
    out = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(expand_tco, u): u for u in urls}
        for f in as_completed(futures):
            out.append(f.result())
    return out


def extract_arxiv_id(text: str, urls: list[str]):
    combined = " ".join(urls) + " " + (text or "")
    m = ARXIV_RE.search(combined)
    if not m:
        return None, None
    arxiv_id = m.group(1)
    return arxiv_id, f"https://arxiv.org/abs/{arxiv_id}"


def parse_tweet_time(s: str | None):
    if not s:
        return datetime.utcnow()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime.utcnow()


def check_ai_relevance(text: str) -> bool:
    """Ask Claude if a tweet is AI-related."""
    prompt = f"You are an AI expert. Is this text about AI, ML, or LLMs? Answer YES or NO.\n{text}"
    try:
        r = CLAUDE.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt
            }],
        )
        ans = (r.content[0].text or "").strip().upper()
        return ans == "YES"
    except Exception as e:
        print(f"⚠️ Claude relevance check failed: {e}")
        return False


# =========================
# Stage 1 — Fetch + store (only if AI relevant)
# =========================


def fetch_and_store_arxiv_threads():
    """Search for arxiv.org tweets, expand threads, only store AI-relevant ones."""
    base_query = f"https://api.socialdata.tools/twitter/search?query=min_faves%3A{MIN_FAVES}%20within_time%3A{WITHIN_TIME}%20arxiv.org"
    next_cursor = ""
    total_inserted = 0
    total_skipped = 0

    print(
        f"\n🔎 Starting fetch for arXiv tweets (min_faves={MIN_FAVES}, within={WITHIN_TIME})"
    )

    while True:
        url = f"{base_query}&cursor={next_cursor}" if next_cursor else base_query
        print(f"\n🌐 Fetching page: {url}")
        time.sleep(2)
        try:
            r = SESSION.get(url, timeout=FETCH_TIMEOUT)
            data = r.json()
        except Exception as e:
            print(f"⚠️ Fetch error: {e}")
            break

        tweets = data.get("tweets", [])
        if not tweets:
            print("No tweets found in this page.")
            break

        for tweet in tweets:
            user = (tweet.get("user") or {}).get("screen_name", "")
            if not user:
                continue

            tid = tweet.get("id")
            thread = fetch_thread(tid)
            if not thread or not thread.get("tweets"):
                base_tweets = [tweet]
            else:
                same_author = [
                    t for t in thread["tweets"]
                    if (t.get("user") or {}
                        ).get("screen_name", "").lower() == user.lower()
                ]
                base_tweets = same_author if same_author else [tweet]

            # safety check
            if not base_tweets:
                print(
                    f"⚠️ Empty thread for {user} ({tweet.get('id')}) — skipping."
                )
                continue

            base_tweets = sorted(base_tweets,
                                 key=lambda x: str(x.get("id") or "0"))
            root = base_tweets[0]

            tweet_time = parse_tweet_time(root.get("tweet_created_at"))
            _print_fetched_tweet(user, root.get("favorite_count", 0),
                                 tweet_time)

            # merge text + URLs
            text = " ".join(t.get("full_text", "") for t in base_tweets)
            urls = [
                u.get("expanded_url", "") for t in base_tweets
                for u in (t.get("entities", {}).get("urls", []) or [])
            ]
            urls = expand_urls(urls)
            arxiv_id, arxiv_url = extract_arxiv_id(text, urls)
            if not arxiv_id:
                total_skipped += 1
                continue

            # ✅ Only insert if AI-relevant
            is_ai = check_ai_relevance(text)
            if not is_ai:
                print(f"🚫 Skipping non-AI thread {arxiv_id}")
                total_skipped += 1
                continue

            # ✅ Build base document
            author = user
            root_id = str(root["id"])  # ← Convert to string immediately
            full_text = text
            images = [
                m.get("media_url_https", "") for t in base_tweets
                for m in (t.get("entities", {}).get("media", []) or [])
            ]

            parent = user  # or other logic if needed
            tweet_time = parse_tweet_time(root.get("tweet_created_at"))
            # ✅ Collect all tweet IDs in the thread (as strings)
            thread_ids = [str(t["id"]) for t in base_tweets if t.get("id")]

            like_count = root.get("favorite_count", 0)
            reply_count = root.get("reply_count", 0)
            retweet_count = root.get("retweet_count", 0)

            base_doc = {
                "user_id": (root.get("user") or {}).get("id"),
                "username":
                author,  # ← Use actual tweet author
                "tweet_id":
                root_id,  # ← Already a string
                "text":
                full_text,
                "like_count":
                like_count,
                "reply_count":
                reply_count,
                "retweet_count":
                retweet_count,
                "quote_count":
                root.get("quote_count"),
                "views_count":
                root.get("views_count"),
                "bookmark_count":
                root.get("bookmark_count"),
                "in_reply_to":
                False,
                "tweet_time":
                tweet_time,
                "name": (root.get("user") or {}).get("name"),
                "tweet_type":
                "arxiv",
                "profile_image_url_https":
                (root.get("user") or {}).get("profile_image_url_https"),
                "tweet_url":
                f"https://twitter.com/{author}/status/{root_id}",
                "image_urls":
                images,
                "is_thread":
                len(base_tweets) > 1,
                "thread_tweets":
                thread_ids,  # ← All IDs are strings
                "root_tweet":
                True,
                "processing_status":
                "unprocessed",
                "potential_username":
                parent,
                "arxiv_id":
                arxiv_id,
                "arxiv_url":
                arxiv_url,
                "scrape_date":
                datetime.utcnow(),
                "first_seen_at":
                datetime.utcnow(),
                "updated_at":
                datetime.utcnow(),
                "source":
                "X",
                "article_written":
                False,
                "show_in_feed":
                True,
            }

            # ✅ Insert immediately instead of batching
            try:
                result = TWEETS.update_one(
                    {"tweet_id": base_doc["tweet_id"]},
                    {"$setOnInsert": base_doc},
                    upsert=True,
                )
                if result.upserted_id or result.modified_count > 0:
                    total_inserted += 1
                    print(f"📥 Inserted AI-relevant arXiv thread: {arxiv_id}")
                else:
                    print(f"⏭️ Thread {arxiv_id} already exists, skipping")
            except Exception as e:
                print(f"❌ Failed to insert {arxiv_id}: {e}")
                continue

            # 📊 Update metrics in sources.arxiv collection
            update_metrics_in_arxiv_db(
                arxiv_id,
                like_count,
                reply_count,
                retweet_count,
                root_id,
                f"https://twitter.com/{author}/status/{root_id}",
            )

        next_cursor = data.get("next_cursor")
        if not next_cursor:
            break

    print(
        f"\n✅ Done fetching. Inserted: {total_inserted}, Skipped: {total_skipped}"
    )


# =========================
# Stage 2 — Process
# =========================


def process_unprocessed_arxiv_threads():
    q = {"processing_status": "unprocessed"}
    docs = list(TWEETS.find(q))
    if not docs:
        print("No unprocessed arXiv threads.")
        return
    print(f"⚙️ Processing {len(docs)} arXiv threads sequentially...\n")

    for i, doc in enumerate(docs, 1):
        print(f"--- Processing {i}/{len(docs)} ---")
        process_one_arxiv_thread(doc)
        print()  # Add blank line between tweets

    print("✅ Completed processing stage.")


def process_one_arxiv_thread(tweet_doc: dict):
    try:
        tid = tweet_doc["tweet_id"]
        text = tweet_doc["text"]
        arxiv_id = tweet_doc["arxiv_id"]

        print(f"🧠 Fetching metadata for {arxiv_id}")
        meta = fetch_arxiv_metadata(arxiv_id)
        if not meta:
            print(f"⚠️ No metadata found for {arxiv_id}")
            TWEETS.update_one({"tweet_id": tid},
                              {"$set": {
                                  "processing_status": "no_metadata"
                              }})
            return

        print(f"🏷️ Categorizing {arxiv_id}")
        categories = categorize_paper_with_claude(meta.get("title", ""),
                                                  meta.get("summary", ""),
                                                  CLAUDE)

        print(f"📰 Writing article for {arxiv_id}")

        _print_tweet_summary(tweet_doc["username"],
                             tweet_doc.get("like_count", 0),
                             tweet_doc["tweet_url"], tweet_doc["tweet_time"])

        # 📊 Update metrics in sources.arxiv when processing completes
        update_metrics_in_arxiv_db(
            arxiv_id,
            tweet_doc.get("like_count", 0),
            tweet_doc.get("reply_count", 0),
            tweet_doc.get("retweet_count", 0),
            tweet_doc["tweet_id"],
            tweet_doc["tweet_url"],
        )

        TWEETS.update_one(
            {"tweet_id": tid},
            {
                "$set": {
                    "processing_status": "processed",
                    "arxiv_meta": meta,
                    "ai_categories": categories,
                    "updated_at": datetime.utcnow(),
                }
            },
        )

    except Exception as e:
        print(f"❌ Failed {tweet_doc.get('tweet_id')}: {type(e).__name__} {e}")
        TWEETS.update_one(
            {"tweet_id": tweet_doc.get("tweet_id")},
            {"$set": {
                "processing_status": f"failed:{type(e).__name__}"
            }})


# =========================
# Main
# =========================


def main():
    fetch_and_store_arxiv_threads()
    process_unprocessed_arxiv_threads()
    print("🎉 Completed full arXiv pipeline.")


if __name__ == "__main__":
    main()

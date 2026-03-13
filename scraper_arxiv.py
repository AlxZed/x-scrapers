from __future__ import annotations
import re
from datetime import datetime

from config import (SESSION, PAPERS, TWEETS_ARXIV, NON_AI_TWEETS,
                    DEFAULT_MIN_FAVES, DEFAULT_WITHIN_TIME)
from arxiv_fetch import fetch_arxiv_metadata
from writing_system import categorize, find_entities
from tweet_utils import (
    fetch_thread,
    expand_urls,
    parse_tweet_time,
    calculate_tweet_std_scores,
    print_fetched_tweet,
    print_tweet_summary,
    store_non_ai_tweet,
    get_same_author_tweets,
    merge_thread_text,
    collect_thread_urls,
    build_base_tweet_doc,
    upsert_tweet,
    iter_search_tweets,
)

# ── Config ────────────────────────────────────

MIN_FAVES = DEFAULT_MIN_FAVES
WITHIN_TIME = DEFAULT_WITHIN_TIME
CATEGORIZE = False  # Set to False to skip categorization + entity extraction

ACCEPTED_CATEGORIES = {
    "cs.hc",
    "eess.iv",
    "cs.ai",
    "stat.ml",
    "cs.lg",
    "cs.ma",
    "cs.cv",
    "cs.cl",
    "cs.ro",
    "eess.as",
}

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")

# ── Helpers ───────────────────────────────────


def _load_processed_ids() -> set:
    ai_ids = set(TWEETS_ARXIV.distinct("tweet_id"))
    non_ai_ids = set(NON_AI_TWEETS.distinct("tweet_id"))
    total = ai_ids | non_ai_ids
    print(f"📋 Loaded {len(total)} already-processed tweet IDs")
    return total


def _extract_arxiv_id(text: str, urls: list[str]):
    combined = " ".join(urls) + " " + (text or "")
    m = ARXIV_RE.search(combined)
    if not m:
        return None, None
    arxiv_id = m.group(1)
    return arxiv_id, f"https://arxiv.org/abs/{arxiv_id}"


def _category_is_relevant(categories: list[str]) -> bool:
    if not categories:
        return False
    normalized = {c.lower().strip() for c in categories}
    return bool(normalized & ACCEPTED_CATEGORIES)


def _categorize_paper(title: str, summary: str, arxiv_url: str = None) -> dict:
    text = f"Title: {title}\n\nAbstract: {summary}"
    try:
        result = categorize(text=text, advanced_category=True, url=arxiv_url)
        result.setdefault("regular_categories", [])
        result.setdefault("advanced_categories", [])
        print(f"   Regular: {result['regular_categories']}")
        if result["advanced_categories"]:
            print(f"   Advanced: {result['advanced_categories']}")
        return result
    except Exception as e:
        print(f"⚠️ Categorization failed: {e}")
        return {"regular_categories": [], "advanced_categories": []}


def _extract_entities(title: str, summary: str, arxiv_url: str = None) -> dict:
    text = f"Title: {title}\n\nAbstract: {summary}"
    result = find_entities(text=text, url=arxiv_url)
    print(f"   Key Entities: {result.get('key_entities', [])}")
    return result


def _update_metrics_in_papers_db(arxiv_id, like_count, reply_count,
                                 retweet_count, tweet_id, tweet_url):
    try:
        PAPERS.update_one(
            {"arxiv_id": arxiv_id},
            {
                "$set": {
                    "like_count": like_count,
                    "reply_count": reply_count,
                    "retweet_count": retweet_count,
                    "tweet_id": str(tweet_id),
                    "tweet_url": tweet_url,
                    "metrics_updated_at": datetime.utcnow(),
                }
            },
            upsert=True,
        )
    except Exception as e:
        print(f"⚠️ Failed to update sources.arxiv for {arxiv_id}: {e}")


# ── Backfill ──────────────────────────────────


def validate_and_backfill():
    """Ensure all existing entries have metadata, categories, and entities."""
    print("\n" + "=" * 60)
    print("🔍 VALIDATING EXISTING ARXIV ENTRIES")
    print("=" * 60)

    query = {
        "$or": [
            {
                "arxiv_details": {
                    "$exists": False
                }
            },
            {
                "arxiv_details": None
            },
            {
                "regular_categories": {
                    "$exists": False
                }
            },
            {
                "regular_categories": []
            },
            {
                "advanced_categories": {
                    "$exists": False
                }
            },
            {
                "advanced_categories": []
            },
            {
                "key_entities": {
                    "$exists": False
                }
            },
            {
                "key_entities": None
            },
        ]
    }

    docs = list(TWEETS_ARXIV.find(query))
    if not docs:
        print("✅ All entries valid.\n")
        return

    print(f"⚠️ {len(docs)} entries need backfilling\n")

    for i, doc in enumerate(docs, 1):
        arxiv_id = doc.get("arxiv_id")
        if not arxiv_id:
            continue

        print(f"[{i}/{len(docs)}] Backfilling {arxiv_id}")

        meta = doc.get("arxiv_details") or fetch_arxiv_metadata(arxiv_id)
        if not meta:
            print(f"  ❌ Could not fetch metadata, skipping")
            continue

        needs_work = (not doc.get("regular_categories")
                      or not doc.get("advanced_categories")
                      or not doc.get("key_entities"))
        if not needs_work:
            continue

        cat_result = _categorize_paper(meta.get("title", ""),
                                       meta.get("summary", ""),
                                       doc.get("arxiv_url"))
        ent_result = _extract_entities(meta.get("title", ""),
                                       meta.get("summary", ""),
                                       doc.get("arxiv_url"))

        tweet_text = doc.get("text", "")
        full_text = f"{tweet_text}\n\n---\n\nArXiv Abstract:\n{meta.get('summary', '')}"

        TWEETS_ARXIV.update_one({"tweet_id": doc["tweet_id"]}, {
            "$set": {
                "arxiv_details": meta,
                "regular_categories": cat_result.get("regular_categories", []),
                "advanced_categories": cat_result.get("advanced_categories",
                                                      []),
                "key_entities": ent_result.get("key_entities", []),
                "text": full_text,
                "updated_at": datetime.utcnow(),
            }
        })

    print(f"✅ Backfill complete — fixed {len(docs)} entries\n")


# ── Main pipeline ─────────────────────────────


def run():
    """Fetch arXiv tweets, check relevance via category match, categorize, store."""
    processed_ids = _load_processed_ids()

    base_url = (
        f"https://api.socialdata.tools/twitter/search"
        f"?query=min_faves%3A{MIN_FAVES}%20within_time%3A{WITHIN_TIME}%20arxiv.org"
    )
    inserted = skipped = already = 0

    print(
        f"\n🔎 arXiv scraper (min_faves={MIN_FAVES}, within={WITHIN_TIME}, categorize={CATEGORIZE})"
    )

    for tweet in iter_search_tweets(SESSION, base_url):
        user = (tweet.get("user") or {}).get("screen_name", "")
        if not user:
            continue

        tid = str(tweet.get("id"))

        if tid in processed_ids:
            already += 1
            continue

        # ── Thread ──
        thread_data = fetch_thread(SESSION, tweet.get("id"))
        base_tweets = get_same_author_tweets(thread_data, tweet, user)
        if not base_tweets:
            continue

        root = base_tweets[0]
        tweet_time = parse_tweet_time(root.get("tweet_created_at"))
        print_fetched_tweet(user, root.get("favorite_count", 0), tweet_time)

        # ── Extract arXiv ID ──
        text = merge_thread_text(base_tweets)
        urls = expand_urls(collect_thread_urls(base_tweets))
        arxiv_id, arxiv_url = _extract_arxiv_id(text, urls)

        if not arxiv_id:
            skipped += 1
            continue

        # ── Fetch metadata ──
        meta = fetch_arxiv_metadata(arxiv_id)
        if not meta:
            store_non_ai_tweet(NON_AI_TWEETS, tid, user, "X arxiv", {
                "arxiv_id": arxiv_id,
                "reason": "no_metadata"
            })
            skipped += 1
            continue

        # ── Category relevance (fast pre-filter) ──
        if not _category_is_relevant(meta.get("categories", [])):
            store_non_ai_tweet(NON_AI_TWEETS, tid, user, "X arxiv", {
                "arxiv_id": arxiv_id,
                "reason": "non_ai_content"
            })
            skipped += 1
            continue

        # ── Categorize + entities (optional) ──
        if CATEGORIZE:
            cat_result = _categorize_paper(meta.get("title", ""),
                                           meta.get("summary", ""), arxiv_url)
            ent_result = _extract_entities(meta.get("title", ""),
                                           meta.get("summary", ""), arxiv_url)
        else:
            cat_result = {"regular_categories": [], "advanced_categories": []}
            ent_result = {"key_entities": []}
            print("   ⏭️ Categorization & entity extraction skipped")

        full_text = f"{text}\n\n---\n\nArXiv Abstract:\n{meta.get('summary', '')}"

        like_count = root.get("favorite_count", 0)
        retweet_count = root.get("retweet_count", 0)
        root_id = str(root["id"])

        std_scores = calculate_tweet_std_scores(
            like_count, retweet_count,
            root.get("bookmark_count") or 0, TWEETS_ARXIV)

        tweet_doc = build_base_tweet_doc(
            root,
            base_tweets,
            text=full_text,
            tweet_type="arxiv",
            source="X arxiv",
            std_scores=std_scores,
            extra_fields={
                "processing_status": "processed",
                "arxiv_id": arxiv_id,
                "arxiv_url": arxiv_url,
                "arxiv_details": meta,
                "regular_categories": cat_result.get("regular_categories", []),
                "advanced_categories": cat_result.get("advanced_categories",
                                                      []),
                "key_entities": ent_result.get("key_entities", []),
            },
        )

        if upsert_tweet(TWEETS_ARXIV, tweet_doc):
            inserted += 1
            print_tweet_summary(user, like_count, tweet_doc["tweet_url"],
                                tweet_time)

        _update_metrics_in_papers_db(
            arxiv_id, like_count, root.get("reply_count", 0), retweet_count,
            root_id, f"https://twitter.com/{user}/status/{root_id}")

    print(
        f"\n✅ arXiv done — inserted: {inserted}, skipped: {skipped}, already: {already}"
    )

from __future__ import annotations
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from config import SESSION, MODELS, TWEETS_HUGGINGFACE, NON_AI_TWEETS, DEFAULT_MIN_FAVES, DEFAULT_WITHIN_TIME
from check_relevance import check_ai_relevance
from writing_system import categorize
from tweet_utils import (
    fetch_thread,
    expand_urls,
    parse_tweet_time,
    calculate_tweet_std_scores,
    calculate_collection_std_scores,
    print_fetched_tweet,
    print_tweet_summary,
    store_non_ai_tweet,
    get_same_author_tweets,
    merge_thread_text,
    collect_thread_urls,
    build_base_tweet_doc,
    upsert_tweet,
    iter_search_tweets,
    get_og_image,
)

# ── Config ────────────────────────────────────

MIN_FAVES = DEFAULT_MIN_FAVES
WITHIN_TIME = DEFAULT_WITHIN_TIME
CATEGORIZE = False  # Set to False to skip categorization

HUGGINGFACE_DATASET_RE = re.compile(
    r"huggingface\.co/datasets/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)")
HUGGINGFACE_MODEL_RE = re.compile(
    r"huggingface\.co/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)")
BLOCKED_PREFIXES = {
    "spaces", "docs", "papers", "blog", "datasets", "collections"
}

# ── Helpers ───────────────────────────────────


def _extract_model(text: str, urls: list[str]):
    combined = " ".join(urls) + " " + (text or "")

    for m in HUGGINGFACE_DATASET_RE.finditer(combined):
        uname, mname = m.group(1), m.group(2)
        return uname, mname, f"https://huggingface.co/datasets/{uname}/{mname}", "dataset"

    for m in HUGGINGFACE_MODEL_RE.finditer(combined):
        uname, mname = m.group(1), m.group(2)
        if uname.lower() in BLOCKED_PREFIXES:
            continue
        return uname, mname, f"https://huggingface.co/{uname}/{mname}", "model"

    return None, None, None, None


def _fetch_resource_details(username: str, model_name: str,
                            resource_type: str) -> dict | None:
    model_id = f"{username}/{model_name}"
    prefix = "datasets/" if resource_type == "dataset" else ""
    try:
        r = requests.get(
            f"https://huggingface.co/api/{prefix.rstrip('/')+'/' if prefix else 'models/'}{model_id}",
            timeout=10)
        r.raise_for_status()
        d = r.json()
        details = {
            "model_id": model_id,
            "username": username,
            "model_name": model_name,
            "resource_type": resource_type,
            "description": d.get("description", ""),
            "tags": d.get("tags", []),
            "likes": d.get("likes", 0),
            "downloads": d.get("downloads", 0),
            "created_at": d.get("createdAt"),
            "last_modified": d.get("lastModified"),
            "model_url": f"https://huggingface.co/{prefix}{model_id}",
            "private": d.get("private", False),
            "gated": d.get("gated", False),
        }
        if resource_type == "model":
            details["pipeline_tag"] = d.get("pipeline_tag")
            details["library_name"] = d.get("library_name")
        elif resource_type == "dataset":
            details["dataset_info"] = d.get("cardData", {})
        return details
    except requests.RequestException as e:
        print(f"⚠️ Failed to fetch HF {resource_type} {model_id}: {e}")
        return None


def _fetch_readme(model_id: str, resource_type: str) -> str:
    prefix = "datasets/" if resource_type == "dataset" else ""
    for attempt in range(3):
        try:
            # Try parsed HTML first
            r = requests.get(
                f"https://huggingface.co/{prefix}{model_id}/blob/main/README.md",
                timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                el = soup.select_one("article.markdown") or soup.select_one(
                    ".prose")
                if el:
                    txt = el.get_text(separator="\n", strip=True)
                    if len(txt) > 50:
                        return " ".join(txt.split()[:1000])

            # Fallback to raw
            r2 = requests.get(
                f"https://huggingface.co/{prefix}{model_id}/raw/main/README.md",
                timeout=10)
            if r2.status_code == 200 and len(r2.text.strip()) > 50:
                return " ".join(r2.text.split()[:1000])

            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            print(f"⚠️ README fetch error for {model_id}: {e}")
            if attempt < 2:
                time.sleep(2)
    return ""


def _model_std_scores(model_info: dict) -> dict:
    return calculate_collection_std_scores(
        current_values={
            "likes": model_info.get("likes", 0),
            "downloads": model_info.get("downloads", 0)
        },
        collection=MODELS,
        fields=["like_count", "downloads"],
        output_keys=["std_likes", "std_downloads"],
    )


# ── Main pipeline ─────────────────────────────


def run():
    """Fetch HuggingFace tweets, check AI relevance, categorize, store."""
    base_url = (
        f"https://api.socialdata.tools/twitter/search"
        f"?query=min_faves%3A{MIN_FAVES}%20within_time%3A{WITHIN_TIME}%20huggingface.co"
    )
    inserted = skipped = 0

    print(
        f"\n🔎 HuggingFace scraper (min_faves={MIN_FAVES}, within={WITHIN_TIME}, categorize={CATEGORIZE})"
    )

    for tweet in iter_search_tweets(SESSION, base_url):
        user = (tweet.get("user") or {}).get("screen_name", "")
        if not user:
            continue

        tid = str(tweet.get("id"))

        # ── Thread ──
        thread_data = fetch_thread(SESSION, tweet.get("id"))
        base_tweets = get_same_author_tweets(thread_data, tweet, user)
        if not base_tweets:
            continue

        root = base_tweets[0]
        tweet_time = parse_tweet_time(root.get("tweet_created_at"))
        print_fetched_tweet(user, root.get("favorite_count", 0), tweet_time)

        # ── Extract model ──
        text = merge_thread_text(base_tweets)
        urls = expand_urls(collect_thread_urls(base_tweets))
        username, model_name, model_url, resource_type = _extract_model(
            text, urls)

        if not username or not model_name:
            skipped += 1
            continue

        model_id = f"{username}/{model_name}"

        # ── Skip already-processed non-AI ──
        if NON_AI_TWEETS.find_one({"tweet_id": tid}):
            skipped += 1
            continue

        # ── Relevance check ──
        if not check_ai_relevance(text):
            store_non_ai_tweet(NON_AI_TWEETS, tid, user, "X huggingface",
                               {"huggingface_model": model_id})
            skipped += 1
            continue

        # ── Common metrics ──
        like_count = root.get("favorite_count", 0)
        retweet_count = root.get("retweet_count", 0)
        root_id = str(root["id"])

        tweet_std = calculate_tweet_std_scores(like_count, retweet_count,
                                               root.get("bookmark_count") or 0,
                                               TWEETS_HUGGINGFACE)

        hf_extra = {
            "huggingface_username": username,
            "huggingface_model": model_name,
            "huggingface_model_id": model_id,
            "huggingface_url": model_url,
            "resource_type": resource_type,
        }

        existing = MODELS.find_one({"model_id": model_id})

        if existing:
            # ── Existing model: update metrics only ──
            tweet_doc = build_base_tweet_doc(root,
                                             base_tweets,
                                             text=text,
                                             tweet_type="huggingface",
                                             source="X Huggingface",
                                             std_scores=tweet_std,
                                             extra_fields=hf_extra)
            tweet_doc.pop("news_id", None)

            upsert_tweet(TWEETS_HUGGINGFACE, tweet_doc)

            try:
                MODELS.update_one({"model_id": model_id}, {
                    "$set": {
                        "tweet_like_count": like_count,
                        "tweet_reply_count": root.get("reply_count", 0),
                        "tweet_retweet_count": retweet_count,
                        "tweet_id": root_id,
                        "tweet_url":
                        f"https://twitter.com/{user}/status/{root_id}",
                        "tweet_username": user,
                        "tweet_metrics_updated_at": datetime.utcnow(),
                    }
                })
            except Exception as e:
                print(f"❌ Metrics update failed: {e}")
            continue

        # ── New model: full fetch + optionally categorize ──
        print(f"✨ New {resource_type}: {model_id}")

        details = _fetch_resource_details(username, model_name, resource_type)
        if not details:
            skipped += 1
            continue

        readme = _fetch_readme(model_id, resource_type)
        image_url = get_og_image(model_url)

        if CATEGORIZE:
            cat_result = categorize(text=readme
                                    or details.get("description", ""),
                                    advanced_category=True)
        else:
            cat_result = {"regular_categories": [], "advanced_categories": []}
            print("   ⏭️ Categorization skipped")

        model_std = _model_std_scores(details)

        tweet_doc = build_base_tweet_doc(
            root,
            base_tweets,
            text=text,
            tweet_type="huggingface",
            source="X Huggingface",
            std_scores=tweet_std,
            extra_fields={
                **hf_extra,
                "huggingface_details": details,
                "regular_categories": cat_result.get("regular_categories", []),
                "advanced_categories": cat_result.get("advanced_categories",
                                                      []),
            })

        if not upsert_tweet(TWEETS_HUGGINGFACE, tweet_doc):
            continue

        hf_doc = {
            **details,
            "regular_categories":
            cat_result.get("regular_categories", []),
            "advanced_categories":
            cat_result.get("advanced_categories", []),
            "image_url":
            image_url,
            "readme_content":
            readme,
            "tweet_like_count":
            like_count,
            "tweet_reply_count":
            root.get("reply_count", 0),
            "tweet_retweet_count":
            retweet_count,
            "tweet_id":
            root_id,
            "tweet_url":
            f"https://twitter.com/{user}/status/{root_id}",
            "tweet_username":
            user,
            "potential_username":
            user,
            "source":
            "X Huggingface",
            "scrape_date":
            datetime.utcnow(),
            "first_seen_at":
            datetime.utcnow(),
            "tweet_metrics_updated_at":
            datetime.utcnow(),
            **model_std,
        }

        try:
            MODELS.insert_one(hf_doc)
            inserted += 1
            print(f"✅ Created {model_id}")
            print_tweet_summary(
                user, like_count,
                f"https://twitter.com/{user}/status/{root_id}", tweet_time)
        except Exception as e:
            print(f"❌ Insert failed: {e}")

    print(f"\n✅ HuggingFace done — inserted: {inserted}, skipped: {skipped}")

"""
Unified tweet extraction and handling module.

Consolidates all common Twitter/SocialData functionality used across
the HuggingFace, GitHub, arXiv, and company scrapers into one reusable module.

Usage:
    from tweet_utils import (
        build_session, fetch_search_page, fetch_thread, fetch_complete_thread,
        expand_tco, expand_urls, parse_tweet_time, generate_unique_id,
        calculate_tweet_std_scores, calculate_collection_std_scores,
        extract_media_from_tweet, extract_urls_from_tweet,
        build_base_tweet_doc, upsert_tweet, store_non_ai_tweet,
        get_og_image, get_full_size_profile_image_url,
        get_same_author_tweets, merge_thread_text,
        collect_thread_urls, collect_thread_images,
        iter_search_tweets, print_fetched_tweet, print_tweet_summary,
    )
"""

from __future__ import annotations

import re
import time
import uuid
import statistics
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

FETCH_TIMEOUT = 25
REQUESTS_PER_MINUTE = 100
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE

SOCIALDATA_BASE = "https://api.socialdata.tools"
SEARCH_ENDPOINT = f"{SOCIALDATA_BASE}/twitter/search"
THREAD_ENDPOINT = f"{SOCIALDATA_BASE}/twitter/thread"

TCO_RE = re.compile(r"https://t\.co/\w+")

# ─────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────


def build_session(socialdata_bearer: str) -> requests.Session:
    """
    Build a requests.Session with retry logic and SocialData auth.

    Args:
        socialdata_bearer: Full "Bearer xxx" string for Authorization header.
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    })
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))

    if socialdata_bearer:
        session.headers["Authorization"] = socialdata_bearer

    return session


# ─────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────

_TWEET_TIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
)


def parse_tweet_time(ts: str | None) -> datetime:
    """Parse a SocialData tweet_created_at string into a naive UTC datetime."""
    if not ts:
        return datetime.utcnow()
    for fmt in _TWEET_TIME_FORMATS:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return datetime.utcnow()


def fmt_dt(dt: datetime) -> str:
    """Format a datetime for log output."""
    try:
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt)


def hours_ago(dt: datetime) -> str:
    """Return a human-readable 'Xh ago' string."""
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    delta = datetime.utcnow() - dt
    hrs = max(0, int(delta.total_seconds() // 3600))
    return f"{hrs}h ago"


# ─────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────


def print_fetched_tweet(username: str, likes: int, tweet_time: datetime):
    print(f"📡 @{username} | {likes} Likes | {fmt_dt(tweet_time)}")


def print_tweet_summary(username: str, likes: int, tweet_url: str,
                        tweet_time: datetime):
    print(
        f"✅ @{username} | {likes} Likes | {hours_ago(tweet_time)} | {tweet_url}"
    )


# ─────────────────────────────────────────────
# ID generation
# ─────────────────────────────────────────────


def generate_unique_id() -> str:
    """Generate a UUID-based unique news ID."""
    return str(uuid.uuid4())


# ─────────────────────────────────────────────
# URL expansion
# ─────────────────────────────────────────────


def expand_tco(url: str) -> str:
    """Follow redirects on a single t.co (or any) short URL."""
    try:
        r = requests.get(url, allow_redirects=True, timeout=7)
        return r.url
    except Exception:
        return url


def expand_urls(urls: list[str], max_workers: int = 6) -> list[str]:
    """Expand a list of shortened URLs in parallel."""
    if not urls:
        return []
    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(expand_tco, u): u for u in urls}
        for f in as_completed(futures):
            out.append(f.result())
    return out


def extract_tco_urls_from_text(text: str) -> list[str]:
    """Extract all t.co URLs from raw tweet text."""
    if not text:
        return []
    return TCO_RE.findall(text)


@lru_cache(maxsize=4096)
def untangle_tco_url(tco_url: str,
                     session: requests.Session | None = None) -> str | None:
    """Resolve a single t.co URL, returning the final destination or None."""
    if not tco_url.startswith("https://t.co/"):
        return tco_url
    try:
        r = (session or requests).head(tco_url,
                                       allow_redirects=True,
                                       timeout=5)
        return r.url if r.url != tco_url else None
    except Exception:
        try:
            r = (session or requests).get(tco_url,
                                          allow_redirects=True,
                                          timeout=5,
                                          stream=True)
            return r.url if r.url != tco_url else None
        except Exception:
            return None


def untangle_all_tco_urls(text: str,
                          session: requests.Session | None = None
                          ) -> list[str]:
    """Extract and untangle every t.co URL found in *text*."""
    expanded = []
    for tco_url in extract_tco_urls_from_text(text):
        result = untangle_tco_url(tco_url, session)
        if result and result != tco_url:
            expanded.append(result)
    return expanded


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────


@lru_cache(maxsize=4096)
def get_full_size_profile_image_url(url: str | None) -> str | None:
    """Convert a Twitter profile-image URL to 400×400."""
    if not url:
        return None
    return url.replace("_normal.", "_400x400.")


@lru_cache(maxsize=2048)
def get_og_image(url: str) -> str | None:
    """
    Extract og:image from a webpage.

    Tries Microlink first (fast, works for JS-rendered pages),
    falls back to direct HTML scrape with BeautifulSoup.
    """
    # --- Microlink approach (used by company scraper) ---
    try:
        microlink_url = f"https://api.microlink.io/?url={url}&screenshot=true"
        r = requests.get(
            microlink_url,
            timeout=10,
            headers={
                "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
        if r.status_code == 200:
            data = r.json().get("data", {})
            og = data.get("image", {}).get("url")
            if og:
                return og
            ss = data.get("screenshot", {}).get("url")
            if ss:
                return ss
    except Exception:
        pass

    # --- Direct HTML scrape fallback (used by HuggingFace scraper) ---
    try:
        from bs4 import BeautifulSoup
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            tag = soup.find("meta", property="og:image")
            if tag and tag.get("content"):
                return tag["content"]
            tag = soup.find("meta", attrs={"name": "twitter:image"})
            if tag and tag.get("content"):
                return tag["content"]
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────
# SocialData: search + thread fetching
# ─────────────────────────────────────────────


def fetch_search_page(
    session: requests.Session,
    query: str,
    cursor: str | None = None,
    delay: float = 2.0,
) -> dict:
    """
    Fetch one page of SocialData search results.

    Args:
        session: Authenticated requests session.
        query:   Full query string (already URL-safe or raw).
        cursor:  Pagination cursor from previous response.
        delay:   Seconds to sleep before the request (rate-limit).

    Returns:
        Raw JSON dict with 'tweets' and 'next_cursor' keys.
    """
    if delay:
        time.sleep(delay)
    params = {"query": query}
    if cursor:
        params["cursor"] = cursor
    try:
        r = session.get(SEARCH_ENDPOINT, params=params, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"⚠️ Search fetch error: {e}")
        return {}


def fetch_search_page_url(
    session: requests.Session,
    url: str,
    delay: float = 2.0,
) -> dict:
    """
    Fetch one page of SocialData search results using a pre-built URL.
    (Backwards-compatible with scrapers that build their own URL strings.)
    """
    if delay:
        time.sleep(delay)
    try:
        r = session.get(url, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"⚠️ Search fetch error: {e}")
        return {}


def fetch_thread(session: requests.Session,
                 tweet_id: int | str) -> dict | None:
    """Fetch a full thread from the SocialData /thread/ endpoint."""
    url = f"{THREAD_ENDPOINT}/{tweet_id}"
    try:
        r = session.get(url, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def fetch_complete_thread(
    session: requests.Session,
    conversation_id: str,
    username: str | None = None,
    max_pages: int = 20,
) -> list[dict]:
    """
    Fetch all tweets in a thread via conversation_id search.
    More reliable than the /thread/ endpoint for long threads.
    """
    query = f"conversation_id:{conversation_id}"
    if username:
        query += f" from:{username}"

    all_tweets: list[dict] = []
    cursor = None

    for _ in range(max_pages):
        params = {"query": query, "type": "Latest"}
        if cursor:
            params["cursor"] = cursor

        time.sleep(REQUEST_DELAY)
        try:
            r = session.get(SEARCH_ENDPOINT, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            print(f"⚠️ Error fetching thread {conversation_id}: {e}")
            break

        tweets = data.get("tweets", [])
        if not tweets:
            break
        all_tweets.extend(tweets)

        cursor = data.get("next_cursor")
        if not cursor:
            break

    return all_tweets


# ─────────────────────────────────────────────
# Thread helpers (filter, sort, merge text)
# ─────────────────────────────────────────────


def get_same_author_tweets(thread_data: dict | None, fallback_tweet: dict,
                           username: str) -> list[dict]:
    """
    Given thread data from fetch_thread(), return only the tweets by
    *username* sorted by ID.  Falls back to [fallback_tweet] on failure.
    """
    if not thread_data or not thread_data.get("tweets"):
        return [fallback_tweet]
    same = [
        t for t in thread_data["tweets"]
        if (t.get("user") or {}
            ).get("screen_name", "").lower() == username.lower()
    ]
    base = same if same else [fallback_tweet]
    return sorted(base, key=lambda x: str(x.get("id") or "0"))


def merge_thread_text(tweets: list[dict]) -> str:
    """Join full_text from a list of tweets."""
    return " ".join(t.get("full_text", "") for t in tweets)


def collect_thread_urls(tweets: list[dict]) -> list[str]:
    """Collect all expanded_url values from a list of tweets."""
    return [
        u.get("expanded_url", "") for t in tweets
        for u in (t.get("entities", {}).get("urls", []) or [])
    ]


def collect_thread_images(tweets: list[dict]) -> list[str]:
    seen = set()
    images = []
    for t in tweets:
        for m in (t.get("entities", {}).get("media", []) or []):
            url = m.get("media_url_https", "")
            if url and url not in seen:
                seen.add(url)
                images.append(url)
    return images


def collect_thread_ids(tweets: list[dict]) -> list[str]:
    """Return tweet IDs as strings."""
    return [str(t["id"]) for t in tweets if t.get("id")]


# ─────────────────────────────────────────────
# Media extraction (richer version for company scraper)
# ─────────────────────────────────────────────


def extract_media_from_tweet(
    root: dict, ) -> Tuple[list[str], bool, str | None]:
    """
    Extract images + video info from a single tweet.

    Returns:
        (image_urls, has_video, best_video_url)
    """
    media_list = (root.get("extended_entities") or {}).get("media", [])
    if not media_list:
        media_list = (root.get("entities") or {}).get("media", [])

    photos = [m for m in media_list if m.get("type") == "photo"]
    videos = [m for m in media_list if m.get("type") == "video"]

    has_video = bool(videos)
    video_url = None
    if has_video:
        variants = videos[0].get("video_info", {}).get("variants", [])
        mp4s = sorted(
            [
                v for v in variants
                if v.get("content_type") == "video/mp4" and "bitrate" in v
            ],
            key=lambda v: v.get("bitrate", 0),
            reverse=True,
        )
        if mp4s:
            video_url = mp4s[0].get("url")

    images = [
        p.get("media_url_https") for p in photos if p.get("media_url_https")
    ]
    for v in videos:
        thumb = v.get("media_url_https")
        if thumb and thumb not in images:
            images.append(thumb)

    return images, has_video, video_url


def extract_urls_from_tweet(
    root: dict,
    session: requests.Session | None = None,
) -> Tuple[list[str], str | None]:
    """
    Extract all URLs from a tweet (entity URLs + t.co in text).

    Returns:
        (all_expanded_urls, best_external_url)
    """
    expanded, seen = [], set()

    for ue in root.get("entities", {}).get("urls", []):
        exp = ue.get("expanded_url")
        if exp and exp not in seen:
            seen.add(exp)
            expanded.append(exp)

    text = root.get("full_text") or root.get("text", "")
    if text:
        for url in untangle_all_tco_urls(text, session):
            if url and url not in seen:
                seen.add(url)
                expanded.append(url)

    external = [
        u for u in expanded if "x.com" not in u and "twitter.com" not in u
    ]
    best = max(external, key=len) if external else None

    return expanded, best


# ─────────────────────────────────────────────
# Tweet data extraction
# ─────────────────────────────────────────────


def extract_tweet_data(root: dict) -> dict | None:
    """
    Pull validated basic fields from a raw SocialData tweet dict.
    Returns None if the tweet is missing required fields.
    """
    root_id = root.get("id")
    if not root_id:
        return None

    ts = root.get("tweet_created_at")
    if not ts:
        return None

    user = root.get("user") or {}
    author = user.get("screen_name", "")
    if not author:
        return None

    text = root.get("full_text") or root.get("text")
    if not text:
        return None

    return {
        "tweet_id": str(root_id),
        "user_id": str(user.get("id_str") or user.get("id") or ""),
        "username": author,
        "name": user.get("name", ""),
        "text": text,
        "tweet_time": parse_tweet_time(ts),
        "like_count": root.get("favorite_count", 0),
        "reply_count": root.get("reply_count", 0),
        "retweet_count": root.get("retweet_count", 0),
        "quote_count": root.get("quote_count"),
        "views_count": root.get("views_count"),
        "bookmark_count": root.get("bookmark_count"),
        "profile_image_url_https": user.get("profile_image_url_https"),
    }


# ─────────────────────────────────────────────
# Standardized score calculation (z-scores)
# ─────────────────────────────────────────────

STD_KEYS = {
    "likes": "likes_std",
    "retweets": "retweets_std",
    "bookmarks": "bookmarks_std",
}


def _zscore(val: float, vals: list[float]) -> float:
    """Calculate a single z-score. Returns 0.0 if stdev is 0 or data is insufficient."""
    if len(vals) < 2:
        return 0.0
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    return round((val - m) / s, 2) if s > 0 else 0.0


def calculate_tweet_std_scores(
    like_count: int,
    retweet_count: int,
    bookmark_count: int,
    tweets_collection,
) -> dict:
    """
    Calculate z-score metrics for a tweet's engagement relative to
    all existing tweets in *tweets_collection*.

    Returns:
        {"likes_std": float, "retweets_std": float, "bookmarks_std": float}
    """
    defaults = {v: 0.0 for v in STD_KEYS.values()}

    try:
        docs = list(
            tweets_collection.find({}, {
                "like_count": 1,
                "retweet_count": 1,
                "bookmark_count": 1
            }))
        if len(docs) < 2:
            return defaults

        all_likes = [d.get("like_count", 0) or 0 for d in docs]
        all_rts = [d.get("retweet_count", 0) or 0 for d in docs]
        all_bmarks = [d.get("bookmark_count", 0) or 0 for d in docs]

        return {
            STD_KEYS["likes"]: _zscore(like_count, all_likes),
            STD_KEYS["retweets"]: _zscore(retweet_count, all_rts),
            STD_KEYS["bookmarks"]: _zscore(bookmark_count or 0, all_bmarks),
        }
    except Exception as e:
        print(f"⚠️ Failed to calculate tweet std scores: {e}")
        return defaults


def calculate_collection_std_scores(
    current_values: dict[str, float],
    collection,
    fields: list[str],
    output_keys: list[str] | None = None,
) -> dict:
    """
    Generic z-score calculator against any MongoDB collection.

    Use this for non-tweet metrics like HuggingFace model likes/downloads
    or GitHub repo stars/forks.

    Args:
        current_values: Mapping of field_name → current value to score.
                        e.g. {"likes": 120, "downloads": 5000}
        collection:     A pymongo Collection containing historical data.
        fields:         DB field names to query and compare against.
                        e.g. ["like_count", "downloads"]
        output_keys:    Output key names for the result dict.
                        Defaults to "std_{field}" for each field.
                        e.g. ["std_likes", "std_downloads"]

    Returns:
        Dict with z-scores, e.g. {"std_likes": 1.23, "std_downloads": -0.45}
    """
    if output_keys is None:
        output_keys = [f"std_{f}" for f in fields]

    defaults = {k: 0.0 for k in output_keys}

    try:
        projection = {f: 1 for f in fields}
        docs = list(collection.find({}, projection))

        if len(docs) < 2:
            return defaults

        result = {}
        input_keys = list(current_values.keys())
        for i, field in enumerate(fields):
            vals = [d.get(field, 0) or 0 for d in docs]
            current = current_values.get(input_keys[i], 0) or 0
            result[output_keys[i]] = _zscore(current, vals)

        return result

    except Exception as e:
        print(f"⚠️ Failed to calculate collection std scores: {e}")
        return defaults


# ─────────────────────────────────────────────
# Non-AI tweet storage
# ─────────────────────────────────────────────


def store_non_ai_tweet(
    non_ai_collection,
    tweet_id: str,
    username: str,
    source: str,
    extra_fields: dict | None = None,
) -> None:
    """
    Mark a tweet as non-AI-relevant so it's never reprocessed.

    Args:
        non_ai_collection: pymongo Collection (e.g. sources.twitter_non_ai).
        tweet_id:          The tweet ID string.
        username:          Twitter handle.
        source:            Label like "X arxiv", "X github", "X huggingface".
        extra_fields:      Any additional keys to store (arxiv_id, github_repo, etc.).
    """
    doc = {
        "tweet_id": tweet_id,
        "username": username,
        "source": source,
        "marked_at": datetime.utcnow(),
    }
    if extra_fields:
        doc.update(extra_fields)
    try:
        non_ai_collection.update_one(
            {"tweet_id": tweet_id},
            {"$setOnInsert": doc},
            upsert=True,
        )
        print(f"   💾 Stored in twitter_non_ai")
    except Exception as e:
        print(f"   ⚠️ Failed to store in twitter_non_ai: {e}")


# ─────────────────────────────────────────────
# Base tweet document builder
# ─────────────────────────────────────────────


def build_base_tweet_doc(
    root: dict,
    base_tweets: list[dict],
    *,
    text: str,
    tweet_type: str,
    source: str,
    std_scores: dict | None = None,
    extra_fields: dict | None = None,
) -> dict:
    """
    Build the common tweet document structure shared by all scrapers.

    Args:
        root:         The root (first) tweet in the thread.
        base_tweets:  All tweets in the thread by the same author.
        text:         Merged full text (may include abstracts, READMEs, etc.).
        tweet_type:   "arxiv", "github", "huggingface", or "company".
        source:       Source label for the document.
        std_scores:   Pre-calculated z-score dict (optional).
        extra_fields: Scraper-specific fields to merge in.

    Returns:
        A dict ready for upsert into the tweets collection.
    """
    user = root.get("user") or {}
    author = user.get("screen_name", "")
    root_id = str(root["id"])
    tweet_time = parse_tweet_time(root.get("tweet_created_at"))

    images = collect_thread_images(base_tweets)
    thread_ids = collect_thread_ids(base_tweets)

    doc = {
        "user_id": user.get("id"),
        "username": author,
        "tweet_id": root_id,
        "text": text,
        "like_count": root.get("favorite_count", 0),
        "reply_count": root.get("reply_count", 0),
        "retweet_count": root.get("retweet_count", 0),
        "quote_count": root.get("quote_count"),
        "views_count": root.get("views_count"),
        "bookmark_count": root.get("bookmark_count"),
        "in_reply_to": False,
        "tweet_time": tweet_time,
        "name": user.get("name"),
        "tweet_type": tweet_type,
        "profile_image_url_https": user.get("profile_image_url_https"),
        "tweet_url": f"https://twitter.com/{author}/status/{root_id}",
        "image_urls": images,
        "is_thread": len(base_tweets) > 1,
        "thread_tweets": thread_ids,
        "root_tweet": True,
        "potential_username": author,
        "scrape_date": datetime.utcnow(),
        "first_seen_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "source": source,
        "article_written": False,
        "show_in_feed": True,
        "news_id": generate_unique_id(),
    }

    if std_scores:
        doc.update(std_scores)

    if extra_fields:
        doc.update(extra_fields)

    return doc


# ─────────────────────────────────────────────
# Upsert helper
# ─────────────────────────────────────────────


def upsert_tweet(tweets_collection, doc: dict) -> bool:
    """
    Insert a tweet doc if it doesn't exist (keyed on tweet_id).

    Returns True if a new document was inserted, False if it already existed.
    """
    try:
        result = tweets_collection.update_one(
            {"tweet_id": doc["tweet_id"]},
            {"$setOnInsert": doc},
            upsert=True,
        )
        if result.upserted_id:
            return True
        return False
    except Exception as e:
        print(f"❌ Failed to upsert tweet {doc.get('tweet_id')}: {e}")
        return False


# ─────────────────────────────────────────────
# Search pagination loop
# ─────────────────────────────────────────────


def iter_search_tweets(
    session: requests.Session,
    base_url: str,
    *,
    page_delay: float = 2.0,
):
    """
    Generator that yields every tweet across all pages of a SocialData search.

    Args:
        session:    Authenticated session.
        base_url:   The initial search URL (including query params).
        page_delay: Seconds to wait between pages.

    Yields:
        Individual tweet dicts.
    """
    next_cursor = ""
    while True:
        url = f"{base_url}&cursor={next_cursor}" if next_cursor else base_url
        print(f"\n🌐 Fetching page: {url}")
        data = fetch_search_page_url(session, url, delay=page_delay)

        tweets = data.get("tweets", [])
        if not tweets:
            print("No tweets found in this page.")
            break

        yield from tweets

        next_cursor = data.get("next_cursor")
        if not next_cursor:
            break


# ─────────────────────────────────────────────
# Thread aggregation (for company-style scraper)
# ─────────────────────────────────────────────


def group_tweets_by_conversation(tweets: list[dict]) -> dict[str, list[dict]]:
    """Group tweets by conversation_id_str."""
    conversations: dict[str, list[dict]] = defaultdict(list)
    for tweet in tweets:
        conv_id = tweet.get("conversation_id_str") or tweet.get("id_str")
        conversations[conv_id].append(tweet)
    return dict(conversations)


def aggregate_thread_metrics(tweets: list[dict]) -> dict:
    """Sum engagement metrics across all tweets in a thread."""
    return {
        "likes": sum(t.get("favorite_count", 0) for t in tweets),
        "retweets": sum(t.get("retweet_count", 0) for t in tweets),
        "replies": sum(t.get("reply_count", 0) for t in tweets),
        "quotes": sum(t.get("quote_count", 0) for t in tweets),
        "bookmarks": sum(t.get("bookmark_count", 0) for t in tweets),
        "views": sum(t.get("views_count", 0) for t in tweets),
    }


def aggregate_thread_text(tweets: list[dict]) -> str:
    """Concatenate text from all tweets, separated by double newlines."""
    parts = []
    for t in tweets:
        text = (t.get("full_text") or t.get("text", "")).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def aggregate_thread_urls(
    tweets: list[dict],
    session: requests.Session | None = None,
) -> Tuple[list[str], str | None]:
    """Collect and deduplicate URLs from all thread tweets."""
    all_urls, seen, externals = [], set(), []

    for tweet in tweets:
        for ue in tweet.get("entities", {}).get("urls", []):
            exp = ue.get("expanded_url")
            if exp and exp not in seen:
                seen.add(exp)
                all_urls.append(exp)
                if "x.com" not in exp and "twitter.com" not in exp:
                    externals.append(exp)

        text = tweet.get("full_text") or tweet.get("text", "")
        if text:
            for url in untangle_all_tco_urls(text, session):
                if url and url not in seen:
                    seen.add(url)
                    all_urls.append(url)
                    if "x.com" not in url and "twitter.com" not in url:
                        externals.append(url)

    best = max(externals, key=len) if externals else None
    return all_urls, best


def aggregate_thread_media(
    tweets: list[dict], ) -> Tuple[list[str], bool, str | None]:
    """Collect and deduplicate media from all tweets in a thread."""
    all_images, seen = [], set()
    has_video = False
    video_url = None

    for tweet in tweets:
        media_list = (tweet.get("extended_entities") or {}).get("media", [])
        if not media_list:
            media_list = (tweet.get("entities") or {}).get("media", [])

        for m in media_list:
            if m.get("type") == "photo":
                img = m.get("media_url_https")
                if img and img not in seen:
                    seen.add(img)
                    all_images.append(img)
            elif m.get("type") == "video":
                if not has_video:
                    has_video = True
                    variants = m.get("video_info", {}).get("variants", [])
                    mp4s = sorted(
                        [
                            v for v in variants
                            if v.get("content_type") == "video/mp4"
                            and "bitrate" in v
                        ],
                        key=lambda v: v.get("bitrate", 0),
                        reverse=True,
                    )
                    if mp4s:
                        video_url = mp4s[0].get("url")
                thumb = m.get("media_url_https")
                if thumb and thumb not in seen:
                    seen.add(thumb)
                    all_images.append(thumb)

    return all_images, has_video, video_url


def aggregate_thread(
    tweets: list[dict],
    session: requests.Session | None = None,
) -> dict:
    """
    Full thread aggregation: sort, merge text/urls/media/metrics.

    Returns a dict with combined fields suitable for document building.
    """
    if not tweets:
        return {}

    sorted_tweets = sorted(tweets, key=lambda t: t.get("tweet_created_at", ""))
    first = sorted_tweets[0]

    text = aggregate_thread_text(sorted_tweets)
    metrics = aggregate_thread_metrics(sorted_tweets)
    all_urls, best_url = aggregate_thread_urls(sorted_tweets, session)
    images, has_video, vid_url = aggregate_thread_media(sorted_tweets)

    if best_url:
        og = get_og_image(best_url)
        if og and og not in images:
            images.append(og)

    return {
        "tweet_id":
        str(first.get("id")),
        "thread_tweet_ids":
        [str(t.get("id")) for t in sorted_tweets if t.get("id")],
        "conversation_id":
        first.get("conversation_id_str") or str(first.get("id")),
        "is_thread":
        len(sorted_tweets) > 1,
        "tweet_count":
        len(sorted_tweets),
        "text":
        text,
        "like_count":
        metrics["likes"],
        "retweet_count":
        metrics["retweets"],
        "reply_count":
        metrics["replies"],
        "quote_count":
        metrics["quotes"],
        "bookmark_count":
        metrics["bookmarks"],
        "views_count":
        metrics["views"],
        "all_expanded_urls":
        all_urls,
        "first_external_url":
        best_url,
        "all_images":
        images,
        "has_video":
        has_video,
        "video_url":
        vid_url,
        "first_tweet":
        first,
        "all_tweets":
        sorted_tweets,
    }


def find_and_aggregate_threads(
    tweets: list[dict],
    session: requests.Session | None = None,
    valid_companies: set[str] | None = None,
) -> Tuple[list[dict], list[dict]]:
    """
    Split a batch of tweets into standalone tweets and aggregated threads.
    Only includes tweets from the same author within each thread.
    """
    conversations = group_tweets_by_conversation(tweets)

    standalone, threads = [], []
    processed = set()

    for conv_id, conv_tweets in conversations.items():
        if conv_id in processed:
            continue

        if valid_companies:
            conv_tweets = [
                t for t in conv_tweets
                if (t.get("user") or {}
                    ).get("screen_name", "").lower() in valid_companies
            ]
            if not conv_tweets:
                continue

        author = (conv_tweets[0].get("user") or {}).get("screen_name",
                                                        "").lower()
        same_user = [
            t for t in conv_tweets
            if (t.get("user") or {}).get("screen_name", "").lower() == author
        ]
        if not same_user:
            continue

        has_thread = any((t.get("in_reply_to_status_id_str") and t.get(
            "in_reply_to_user_id_str") == (t.get("user") or {}).get("id_str"))
                         or t.get("reply_count", 0) > 0 for t in same_user)

        if has_thread and session:
            complete = fetch_complete_thread(session, conv_id, username=author)
            complete = [
                t for t in complete
                if (t.get("user") or {}
                    ).get("screen_name", "").lower() == author
            ]
            if len(complete) > 1:
                threads.append(aggregate_thread(complete, session))
                processed.add(conv_id)
            else:
                standalone.append(complete[0] if complete else same_user[0])
        elif len(same_user) == 1:
            standalone.append(same_user[0])
        else:
            threads.append(aggregate_thread(same_user))
            processed.add(conv_id)

    return standalone, threads


# ─────────────────────────────────────────────
# Query builders (for company scraper)
# ─────────────────────────────────────────────


def build_user_query(min_faves: int, within_time: str, username: str) -> dict:
    """Build a SocialData search query for a single user."""
    return {
        "type":
        "Latest",
        "query":
        f"from:{username} min_faves:{min_faves} within_time:{within_time} -filter:replies",
    }


def build_company_query(valid_companies: set[str], min_faves: int,
                        within_time: str) -> dict:
    """Build a SocialData search query for multiple company accounts."""
    from_list = " OR ".join(f"from:{u}" for u in valid_companies)
    return {
        "type": "Latest",
        "query":
        f"({from_list}) min_faves:{min_faves} within_time:{within_time}",
    }


# ─────────────────────────────────────────────
# Tweet-type checkers
# ─────────────────────────────────────────────


def is_quote_tweet(tweet: dict) -> bool:
    return (tweet.get("is_quote_status") is True
            or tweet.get("quoted_status_id") is not None
            or tweet.get("quoted_status_id_str") is not None
            or tweet.get("quoted_status") is not None)


def is_thread_tweet(tweet: dict) -> bool:
    if not tweet.get("in_reply_to_status_id"):
        return False
    return tweet.get("in_reply_to_user_id_str") == ((tweet.get("user")
                                                     or {}).get("id_str"))


# ─────────────────────────────────────────────
# Historical fetch helper
# ─────────────────────────────────────────────


def fetch_historical_for_user(
    username: str,
    min_faves: int,
    historical_start_date: datetime,
    session: requests.Session,
    fetch_and_store_batch_func,
    valid_companies: set[str],
    company_last_updated: dict,
):
    """Loop through weekly windows from start_date to now."""
    current = historical_start_date
    end = datetime.utcnow()

    while current < end:
        week_end = min(current + timedelta(days=7), end)
        since = current.strftime("%Y-%m-%d_00:00:00_UTC")
        until = week_end.strftime("%Y-%m-%d_00:00:00_UTC")

        query = {
            "type":
            "Latest",
            "query":
            f"from:{username} min_faves:{min_faves} since:{since} until:{until}",
        }
        print(
            f"\n🔍 Historical @{username} | "
            f"{current.strftime('%Y-%m-%d')} → {week_end.strftime('%Y-%m-%d')}"
        )
        fetch_and_store_batch_func(query, valid_companies,
                                   company_last_updated)
        current = week_end

    print(f"✅ Historical complete for @{username}")

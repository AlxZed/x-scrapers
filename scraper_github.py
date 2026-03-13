from __future__ import annotations
import re
import base64
from datetime import datetime

import requests

from config import (SESSION, GITHUB_TOKEN, REPOS, TWEETS_GITHUB, NON_AI_TWEETS,
                    DEFAULT_MIN_FAVES, DEFAULT_WITHIN_TIME)
from check_relevance import check_ai_relevance
from writing_system import categorize
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

GITHUB_RE = re.compile(r"github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)")

# ── Helpers ───────────────────────────────────


def _extract_repo(text: str, urls: list[str]):
    combined = " ".join(urls) + " " + (text or "")
    m = GITHUB_RE.search(combined)
    if not m:
        return None, None, None
    owner, repo = m.group(1), m.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo, f"https://github.com/{owner}/{repo}"


def _github_headers() -> dict:
    h = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Mozilla/5.0"
    }
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h


def _fetch_repo_details(owner: str, repo: str) -> dict | None:
    try:
        r = requests.get(f"https://api.github.com/repos/{owner}/{repo}",
                         headers=_github_headers(),
                         timeout=10)
        r.raise_for_status()
        d = r.json()
        return {
            "repo_name": d.get("name"),
            "github_username": d.get("owner", {}).get("login"),
            "title": f"{d.get('owner', {}).get('login')}/{d.get('name')}",
            "repo_url": d.get("html_url"),
            "description": d.get("description"),
            "star_count": d.get("stargazers_count", 0),
            "forks_count": d.get("forks_count", 0),
            "watchers_count": d.get("watchers_count", 0),
            "language": d.get("language"),
            "topics": d.get("topics", []),
            "created_at": d.get("created_at"),
            "updated_at": d.get("updated_at"),
            "pushed_at": d.get("pushed_at"),
            "size": d.get("size"),
            "default_branch": d.get("default_branch"),
            "open_issues_count": d.get("open_issues_count"),
            "license":
            d.get("license", {}).get("name") if d.get("license") else None,
            "homepage": d.get("homepage"),
            "is_fork": d.get("fork", False),
            "archived": d.get("archived", False),
            "profile_picture_url": d.get("owner", {}).get("avatar_url"),
            "readme_url":
            f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
            "image_url": d.get("owner", {}).get("avatar_url"),
        }
    except requests.RequestException as e:
        print(f"⚠️ Failed to fetch repo {owner}/{repo}: {e}")
        return None


def _fetch_readme(owner: str, repo: str) -> str:
    try:
        r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/readme",
                         headers={
                             **_github_headers(), "Accept":
                             "application/vnd.github+json"
                         },
                         timeout=10)
        if r.status_code == 200 and "content" in r.json():
            readme = base64.b64decode(r.json()["content"]).decode(
                "utf-8", errors="ignore")
            return " ".join(readme.split()[:1000])
    except Exception as e:
        print(f"⚠️ Error fetching README: {e}")
    return ""


# ── Main pipeline ─────────────────────────────


def run():
    """Fetch GitHub tweets, check AI relevance, categorize, store."""
    base_url = (
        f"https://api.socialdata.tools/twitter/search"
        f"?query=min_faves%3A{MIN_FAVES}%20within_time%3A{WITHIN_TIME}%20github.com"
    )
    inserted = skipped = 0

    print(f"\n🔎 GitHub scraper (min_faves={MIN_FAVES}, within={WITHIN_TIME})")

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

        # ── Extract repo ──
        text = merge_thread_text(base_tweets)
        urls = expand_urls(collect_thread_urls(base_tweets))
        owner, repo, repo_url = _extract_repo(text, urls)

        if not owner or not repo:
            skipped += 1
            continue

        # ── Skip already-processed non-AI ──
        if NON_AI_TWEETS.find_one({"tweet_id": tid}):
            skipped += 1
            continue

        # ── README + relevance check ──
        readme = _fetch_readme(owner, repo)
        if not readme:
            skipped += 1
            continue

        if not check_ai_relevance(f"{text}\n\n{readme}"):
            print(f"🚫 Not AI-relevant: {owner}/{repo}")
            store_non_ai_tweet(NON_AI_TWEETS, tid, user, "X github",
                               {"github_repo": f"{owner}/{repo}"})
            skipped += 1
            continue

        # ── Common metrics ──
        like_count = root.get("favorite_count", 0)
        retweet_count = root.get("retweet_count", 0)
        root_id = str(root["id"])

        tweet_std = calculate_tweet_std_scores(like_count, retweet_count,
                                               root.get("bookmark_count") or 0,
                                               TWEETS_GITHUB)

        gh_extra = {
            "github_owner": owner,
            "github_repo": repo,
            "github_full_name": f"{owner}/{repo}",
            "github_url": repo_url,
        }

        # ── Fetch repo details ──
        repo_details = _fetch_repo_details(owner, repo)
        if not repo_details:
            skipped += 1
            continue

        existing = REPOS.find_one({
            "repo_name": repo,
            "github_username": owner
        })

        if existing:
            # ── Existing repo: update metrics ──
            tweet_doc = build_base_tweet_doc(root,
                                             base_tweets,
                                             text=text,
                                             tweet_type="github",
                                             source="X Github",
                                             std_scores=tweet_std,
                                             extra_fields=gh_extra)

            upsert_tweet(TWEETS_GITHUB, tweet_doc)

            try:
                REPOS.update_one(
                    {
                        "repo_name": repo,
                        "github_username": owner
                    },
                    {
                        "$set": {
                            **repo_details,
                            "like_count": like_count,
                            "reply_count": root.get("reply_count", 0),
                            "retweet_count": retweet_count,
                            "tweet_id": root_id,
                            "tweet_url":
                            f"https://twitter.com/{user}/status/{root_id}",
                            "tweet_time": tweet_time,
                            "username": user,
                            "metrics_updated_at": datetime.utcnow(),
                        }
                    },
                )
                print(f"🔄 Updated {owner}/{repo}")
            except Exception as e:
                print(f"❌ Update failed: {e}")
            continue

        # ── New repo: categorize + insert ──
        print(f"✨ New repo: {owner}/{repo}")
        cat_result = categorize(text=f"{text}\n\n{readme}",
                                advanced_category=True,
                                url=repo_url)

        regular = cat_result.get("regular_categories", [])
        advanced = cat_result.get("advanced_categories", [])

        if regular is None:
            skipped += 1
            continue

        print(f"   Regular: {regular}  Advanced: {advanced}")

        tweet_doc = build_base_tweet_doc(root,
                                         base_tweets,
                                         text=text,
                                         tweet_type="github",
                                         source="X Github",
                                         std_scores=tweet_std,
                                         extra_fields={
                                             **gh_extra,
                                             "regular_categories": regular,
                                             "advanced_categories": advanced,
                                         })

        if not upsert_tweet(TWEETS_GITHUB, tweet_doc):
            continue

        github_doc = {
            **repo_details,
            "regular_categories": regular,
            "advanced_categories": advanced,
            "like_count": like_count,
            "reply_count": root.get("reply_count", 0),
            "retweet_count": retweet_count,
            "tweet_id": root_id,
            "tweet_url": f"https://twitter.com/{user}/status/{root_id}",
            "tweet_time": tweet_time,
            "username": user,
            "potential_username": user,
            "source": "X github",
            "scrape_date": datetime.utcnow(),
            "first_seen_at": datetime.utcnow(),
            "metrics_updated_at": datetime.utcnow(),
        }

        try:
            REPOS.insert_one(github_doc)
            inserted += 1
            print(f"✅ Created {owner}/{repo}")
            print_tweet_summary(
                user, like_count,
                f"https://twitter.com/{user}/status/{root_id}", tweet_time)
        except Exception as e:
            print(f"❌ Insert failed: {e}")

    print(f"\n✅ GitHub done — inserted: {inserted}, skipped: {skipped}")

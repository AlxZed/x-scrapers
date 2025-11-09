import os
import time
import re
import requests
from anthropic import Anthropic
from datetime import datetime
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from arxiv_fetch import fetch_arxiv_metadata
from categorize import categorize_paper_with_claude
from write_article import insert_article_to_mongo

# ✅ Anthropic client
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

# --------------------
# Config
# --------------------

client = MongoClient(os.environ.get("MONGO_URI_HEADLINE"))
DB = client.get_database("web_signals")

# Collections
COLLECTION = DB.twitter
ARTICLES = DB.articles
NON_NEWS = DB.non_news_tweets
COMPANIES = DB.companies

TOKEN = os.getenv("SOCIALDATA_BEARER")
AUTH_HEADER = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json"
}


# --------------------
# Helpers
# --------------------
def backup(tweet):
    """Insert/update tweet in MongoDB."""
    existing = COLLECTION.find_one({"tweet_id": tweet["tweet_id"]})
    if existing:
        update_data = {
            "like_count": tweet["like_count"],
            "reply_count": tweet["reply_count"],
            "retweet_count": tweet["retweet_count"],
            "scrape_date": datetime.today(),
        }
        COLLECTION.update_one({"tweet_id": tweet["tweet_id"]},
                              {"$set": update_data})
        print(f"♻️ Updated counts for tweet_id={tweet['tweet_id']}")
    else:
        COLLECTION.update_one({"tweet_id": tweet["tweet_id"]}, {"$set": tweet},
                              upsert=True)
        print(f"📥 Inserted new tweet_id={tweet['tweet_id']}")


def fetch_original_tweet(reply_id):
    """Fetch a tweet by ID (used for threads)."""
    url = f"https://api.socialdata.tools/twitter/statuses/show?id={reply_id}"
    response = requests.get(url, headers=AUTH_HEADER)
    return response.json()


def resolve_thread(tweet):
    """Trace replies back to the first tweet and aggregate text/URLs."""
    if not tweet.get("in_reply_to_status_id"):
        return tweet

    tweets_chain = [tweet]
    current_tweet = tweet
    original_author = tweet["user"]["screen_name"]

    while current_tweet.get("in_reply_to_status_id"):
        current_tweet = fetch_original_tweet(
            current_tweet["in_reply_to_status_id"])
        if current_tweet["user"]["screen_name"] != original_author:
            break
        tweets_chain.append(current_tweet)

    tweets_chain.reverse()
    combined_text = " ".join(t["full_text"] for t in tweets_chain).strip()
    combined_urls = [
        u["expanded_url"] for t in tweets_chain for u in t["entities"]["urls"]
    ]

    root = tweets_chain[0]
    root["full_text"] = combined_text
    root["entities"]["urls"] = [{"expanded_url": u} for u in combined_urls]
    return root


def expand_url(url):
    try:
        resp = requests.get(url, allow_redirects=True, timeout=10)
        return resp.url
    except Exception as e:
        print(f"⚠️ Failed to expand {url}: {e}")
        return url


def expand_urls(urls):
    expanded = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(expand_url, u): u for u in urls}
        for future in as_completed(future_to_url):
            expanded.append(future.result())
    return expanded


def extract_arxiv_id(text, urls):
    for u in urls:
        ids = re.findall(r"(\d{4}\.\d{4,5})", u)
        if ids:
            return ids[0], f"https://arxiv.org/abs/{ids[0]}"
    ids = re.findall(r"(\d{4}\.\d{4,5})", text or "")
    if ids:
        return ids[0], f"https://arxiv.org/abs/{ids[0]}"
    return None, None


def check_ai_relevance(tweet_text):
    try:
        prompt = f"""
You are an expert AI analyst.
Is this tweet about AI, machine learning, or relevant to AI researchers or developers?
Answer only with YES or NO.

Tweet text:
{tweet_text}
"""
        msg = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt
            }],
        )
        answer = (msg.content[0].text or "").strip().upper()
        return answer == "YES"
    except Exception as e:
        print(f"⚠️ Claude AI relevance check failed: {e}")
        return False


def extract_and_save(tweets):
    """
    Full pipeline:
    1. Expand URLs, extract arXiv IDs
    2. Skip if already processed
    3. Only consider tweets with >1000 likes
    4. Check AI relevance via Claude
    5. Fetch arXiv metadata
    6. Categorize paper
    7. Build paper + tweet context, generate article, and save to Web_Signals.articles
    8. Backup tweet metadata in Arxiv_tweets collection
    """
    print(f"\n📊 Extracting {len(tweets)} tweets from this page")

    for tweet in tweets:
        try:
            # --------------------
            # 1️⃣ Expand URLs and extract arXiv ID
            # --------------------
            urls = expand_urls([
                u["expanded_url"]
                for u in tweet.get("entities", {}).get("urls", [])
            ])
            arxiv_id, arxiv_url = extract_arxiv_id(tweet["full_text"], urls)
            if not arxiv_id:
                continue

            # --------------------
            # 2️⃣ Parse tweet time
            # --------------------
            time_str = tweet.get("tweet_created_at", "")
            if not time_str:
                tweet_time = datetime.today()
            else:
                try:
                    tweet_time = datetime.strptime(time_str,
                                                   "%Y-%m-%dT%H:%M:%S.%fZ")
                except ValueError:
                    tweet_time = datetime.strptime(time_str,
                                                   "%Y-%m-%dT%H:%M:%SZ")

            # --------------------
            # 3️⃣ Prepare tweet data
            # --------------------
            tweet_data = {
                "tweet_id": int(tweet["id_str"]),
                "username": tweet["user"]["screen_name"],
                "name": tweet["user"]["name"],
                "text": tweet["full_text"],
                "like_count": tweet["favorite_count"],
                "reply_count": tweet["reply_count"],
                "retweet_count": tweet["retweet_count"],
                "scrape_date": datetime.today(),
                "tweet_time": tweet_time,
                "tweet_url":
                f"https://twitter.com/{tweet['user']['screen_name']}/status/{tweet['id_str']}",
                "arxiv_id": arxiv_id,
                "arxiv_url": arxiv_url,
            }

            # --------------------
            # 4️⃣ Skip already processed tweets
            # --------------------
            existing = COLLECTION.find_one(
                {"tweet_id": tweet_data["tweet_id"]})
            if existing and (existing.get("ai_related") is not None
                             or "arxiv_meta" in existing):
                print(
                    f"⏭️ Skipping already processed tweet_id={tweet_data['tweet_id']}"
                )
                continue

            # --------------------
            # 5️⃣ Only process popular tweets
            # --------------------
            if tweet_data["like_count"] <= 1000:
                tweet_data["ai_related"] = None
                backup(tweet_data)
                continue

            # --------------------
            # 6️⃣ Check AI relevance
            # --------------------
            ai_related = check_ai_relevance(tweet_data["text"])
            tweet_data["ai_related"] = ai_related
            print(
                f"🧩 AI relevance for {arxiv_id}: {'YES' if ai_related else 'NO'}"
            )

            if not ai_related:
                print(f"🚫 Skipping non-AI tweet for {arxiv_id}")
                backup(tweet_data)
                continue

            # --------------------
            # 7️⃣ Fetch arXiv metadata
            # --------------------
            meta = fetch_arxiv_metadata(arxiv_id)
            if not meta:
                print(f"⚠️ No metadata found for {arxiv_id}")
                backup(tweet_data)
                continue

            tweet_data["arxiv_meta"] = meta
            print(f"🧠 Added arXiv metadata for {arxiv_id}")

            # --------------------
            # 8️⃣ Categorize paper
            # --------------------
            title = meta.get("title", "")
            abstract = meta.get("summary", "")
            categories = categorize_paper_with_claude(title, abstract,
                                                      anthropic_client)

            if categories:
                tweet_data["arxiv_categories"] = categories
                print(f"🏷️ Categorized as: {', '.join(categories)}")
            else:
                print(f"⚪ No category assigned for {arxiv_id}")

            # --------------------
            # 9️⃣ Write article (builds context, calls Claude, inserts to Mongo)
            # --------------------
            from write_article import insert_article_to_mongo

            print(f"📰 Generating and saving article for {arxiv_id}...")
            insert_article_to_mongo(arxiv_id=arxiv_id,
                                    title=title,
                                    tweet_text=tweet_data["text"],
                                    categories=categories,
                                    meta=meta,
                                    tweet_id=tweet_data["tweet_id"],
                                    tweet_url=tweet_data["tweet_url"],
                                    author=tweet_data["username"])

            # --------------------
            # 🔟 Backup tweet record
            # --------------------
            backup(tweet_data)
            time.sleep(0.5)  # Claude rate limit safety

        except Exception as e:
            print(
                f"⚠️ Error processing tweet_id={tweet.get('id_str', 'unknown')}: {e}"
            )


def query_twitter_api(keyword="arxiv.org", min_faves=20, within_time_days=7):
    """Fetch tweets mentioning arXiv, process page by page."""
    next_cursor = ""

    while True:
        if next_cursor:
            url = (
                f"https://api.socialdata.tools/twitter/search?query=min_faves%3A{min_faves}"
                f"%20within_time%3A{within_time_days}d%20{keyword}&cursor={next_cursor}"
            )
        else:
            url = (
                f"https://api.socialdata.tools/twitter/search?query=min_faves%3A{min_faves}"
                f"%20within_time%3A{within_time_days}d%20{keyword}")

        print(f"\n🔎 Fetching: {url}")
        time.sleep(2)
        response = requests.get(url, headers=AUTH_HEADER)
        resp = response.json()

        if "tweets" in resp:
            page_tweets = []
            for tweet in resp["tweets"]:
                try:
                    root_tweet = resolve_thread(tweet) if tweet.get(
                        "in_reply_to_status_id") else tweet
                    page_tweets.append(root_tweet)
                except Exception as e:
                    print(f"⚠️ Error handling tweet_id={tweet.get('id')}: {e}")

            extract_and_save(page_tweets)

        if resp.get("next_cursor"):
            next_cursor = resp["next_cursor"]
        else:
            break


def scrape():
    query_twitter_api(keyword="arxiv.org", min_faves=100, within_time_days=7)
    print("\n🎉 Scrape done!")


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    scrape()

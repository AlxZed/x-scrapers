import os
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from datetime import datetime
from anthropic import Anthropic

# ✅ Initialize Claude client once
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))


# ===============================
# 🧱 Build Paper Context
# ===============================
def build_paper_context(arxiv_id: str, tweet_text: str):
    """Fetch arXiv HTML, strip non-content, and return combined text with tweet."""
    html_url = f"https://arxiv.org/html/{arxiv_id}"
    first_section = ""
    last_section = ""

    try:
        resp = requests.get(html_url, timeout=20)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "noscript", "footer", "nav"]):
                tag.extract()

            # Remove sections after References/Bibliography/Acknowledgements
            stop_headings = {"references", "bibliography", "acknowledgements"}
            for header in soup.find_all(["h2", "h3", "h4"]):
                if any(word in header.get_text(strip=True).lower()
                       for word in stop_headings):
                    for elem in list(header.find_all_next()):
                        elem.decompose()
                    header.decompose()
                    break

            text = soup.get_text(separator=" ", strip=True)
            words = text.split()

            first_section = " ".join(words[:1000])
            last_section = " ".join(
                words[-1000:]) if len(words) > 2000 else " ".join(
                    words[-len(words) // 2:])

            print(f'Fecthed {len(first_section)+len(last_section)} words')

        else:
            print(
                f"⚠️ Could not fetch paper HTML for {arxiv_id}: {resp.status_code}"
            )
            return ""

    except Exception as e:
        print(f"⚠️ Error fetching HTML for {arxiv_id}: {e}")
        return ""

    return (
        f"Tweet Thread:\n{tweet_text.strip()}\n\n"
        f"Paper Content (first 1000 words):\n{first_section.strip()}\n\n"
        f"Paper Content (last 1000 words before references):\n{last_section.strip()}"
    )


# ===============================
# 🧠 Build Prompt for Claude
# ===============================
def paper_article_prompt(paper_info):
    """Return a structured prompt text for Claude to generate an HTML article."""
    title = paper_info.get("title", "")
    arxiv_id = paper_info.get("arxiv_id", "")
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
    combined_text = paper_info.get("combined_text", "")
    categories = paper_info.get("arxiv_categories", [])

    return f"""
Write a skimmable, technically detailed HTML article summarizing this AI research paper:

Title: {title}
arXiv URL: {arxiv_url}
Categories: {', '.join(categories)}

The following text includes:
- The full tweet thread about the paper
- The first 1000 words of the paper
- The last 1000 words before the References section

Combined content:
{combined_text}

### Task
Synthesize the information above into a professional research summary written like an AI industry journalist or research editor who codes.
Focus on what the paper introduces, why it matters, and how it technically works.

### Structure
1. **Opening (40–60 words):** What the paper is and why it matters.
2. **Why it matters:** The broader impact or novelty of the research.
3. **How it works:** Core methodology, models, or architecture in simple technical language.
4. **The details:** Datasets, evaluation, training setup, and notable results.
5. **The context:** Comparison with prior work or competing approaches.
6. **Technical highlights:** Key equations, metrics, or findings.
7. **By the numbers:** Quantitative performance, compute scale, or benchmarks.
8. **What’s next:** Future directions, open challenges, or implications.

### Style & Tone
- Factual, analytical, and technical — not promotional.
- Each paragraph 1–3 sentences, concise and data-dense.
- Use bold section headers (<b>) and short paragraphs (<p>).
- Avoid unnecessary adjectives, speculation, or generic commentary.
- Reference model sizes, datasets, and training stats if mentioned.
- Write in active voice and present tense.
- No summary or closing statement.

### Output
- Must be valid inline HTML.
- Wrap everything in <article>...</article>.
- Use <p> for paragraphs and <b> for headers.
- No Markdown, meta-text, or commentary outside HTML.
    """


# ===============================
# 💾 Write Article + Insert to Mongo
# ===============================
def insert_article_to_mongo(arxiv_id,
                            title,
                            tweet_text,
                            categories,
                            meta=None,
                            tweet_id=None,
                            tweet_url=None,
                            author=None):
    """
    Build paper context → generate article with Claude → insert into web_signals.articles.
    """
    try:
        # 1️⃣ Build context
        combined_text = build_paper_context(arxiv_id, tweet_text)
        if not combined_text.strip():
            print(f"⚠️ Empty content for {arxiv_id}, skipping.")
            return None

        # 2️⃣ Create prompt
        paper_info = {
            "title": title,
            "arxiv_id": arxiv_id,
            "arxiv_categories": categories or [],
            "combined_text": combined_text,
        }
        prompt = paper_article_prompt(paper_info)

        # 3️⃣ Call Claude
        msg = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            temperature=0.4,
            messages=[{
                "role": "user",
                "content": prompt
            }],
        )
        html_output = msg.content[0].text.strip()

        if not html_output:
            print(f"⚪ Claude returned empty output for {arxiv_id}")
            return None

        # 4️⃣ Save to Mongo
        client = MongoClient(os.environ.get("MONGO_URI_HEADLINE"))
        DB = client.get_database("web_signals")

        # Collections
        ARTICLES = DB.articles

        article_doc = {
            "arxiv_id": arxiv_id,
            "title": title,
            "categories": categories or [],
            "created_at": datetime.utcnow(),
            "html": html_output,
            "username": "alphasignalai",
            "combined_text": combined_text,
            "metadata": meta or {},
            "tweet_id": tweet_id,
            "tweet_url": tweet_url,
            "author": author,
            "source": "arxiv",
        }

        ARTICLES.update_one({"arxiv_id": arxiv_id}, {"$set": article_doc},
                            upsert=True)
        print(f"✅ Article written and saved for {arxiv_id}")
        return article_doc

    except Exception as e:
        print(f"⚠️ insert_article_to_mongo failed for {arxiv_id}: {e}")
        return None

import json


def categorize_paper_prompt():
  instruction = """
You are an expert AI research analyst.

Your task is to categorize **academic papers** from arXiv based on their title and abstract.

---

### Step 1 — Relevance
A paper is relevant ONLY if its abstract or title clearly indicates it is useful for:
- AI/ML researchers
- Data scientists
- Machine learning engineers
- Developers working on or studying artificial intelligence

If it is NOT relevant to AI/ML or computational research, output exactly:
FALSE

---

### Step 2 — Categorization
If relevant, choose up to TWO categories from the list below.

Pick the categories in order of probability (most applicable first).
Use only the category codes (uppercase).  
Do not invent new categories.

---

### AI Paper Categories
Valid categories are:
["LANGUAGE_MODELS", "AGENTS", "VIDEO", "IMAGE", "ROBOTICS", "INFRA",
"SECURITY", "DOCUMENTS", "GPUS", "BENCHMARKS", "AUDIO", "SCRAPING",
"OPEN_SOURCE", "API", "FUNDING", "EVENTS", "EDUCATION"]

Descriptions:
- LANGUAGE_MODELS: LLMs, foundation models, transformers, text understanding or generation.
- AGENTS: Agent systems, autonomous reasoning, planning, tool use, multi-agent systems.
- VIDEO: Video generation, video understanding, video synthesis, or temporal visual models.
- IMAGE: Image generation, segmentation, visual understanding, multimodal vision models (not video).
- ROBOTICS: Robotic systems, control, reinforcement learning for robotics, embodied AI.
- INFRA: ML infrastructure, model serving, optimization, scaling, or training efficiency.
- SECURITY: AI safety, alignment, robustness, privacy, fairness, adversarial defense.
- DOCUMENTS: Text retrieval, RAG, document processing, knowledge extraction.
- GPUS: Compute acceleration, GPU/TPU optimization, or hardware for AI.
- BENCHMARKS: Model evaluation, leaderboards, performance analysis.
- AUDIO: Speech, text-to-speech, sound, or music generation.
- SCRAPING: Dataset creation, data collection, web scraping for training corpora.
- OPEN_SOURCE: Open datasets, frameworks, or libraries; must mention “open source” or “dataset”.
- API: APIs or frameworks for AI interaction or integration.
- FUNDING: Financial announcements, research grants, investments (rare in arXiv).
- EVENTS: Conferences, workshops, or collaborative calls.
- EDUCATION: Tutorials, surveys, learning materials, teaching AI concepts.

---

### Output rules
- If NOT relevant → return exactly:
FALSE
- If relevant → return the top 1–2 categories, comma-separated, e.g.:
LANGUAGE_MODELS,INFRA  
AGENTS,ROBOTICS
"""
  return instruction


def categorize_paper_with_claude(title, abstract, client):
  """
  Categorize an arXiv paper using Claude.
  Returns False if NOT relevant to AI.
  Returns a list with 1–2 category codes if relevant.
  """
  instruction = categorize_paper_prompt()

  VALID_AI_CATEGORIES = {
      "LANGUAGE_MODELS", "AGENTS", "VIDEO", "IMAGE", "ROBOTICS", "INFRA",
      "SECURITY", "DOCUMENTS", "GPUS", "BENCHMARKS", "AUDIO", "SCRAPING",
      "OPEN_SOURCE", "API", "FUNDING", "EVENTS", "EDUCATION"
  }

  user_msg = f"""Paper metadata:

TITLE:
{title}

ABSTRACT:
{abstract}

Decide relevance and, if relevant, output the categories per the rules above.
"""

  try:
    resp = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=80,
        system=instruction,
        messages=[{
            "role": "user",
            "content": user_msg
        }],
    )

    out = resp.content[0].text.strip()

    # Handle FALSE
    t = out.strip().strip('"').strip("'").strip()
    if t.upper() == "FALSE":
      return False

    # Try JSON decode
    cats = None
    try:
      maybe = json.loads(out)
      if isinstance(maybe, dict):
        for k in ("categories", "result", "output"):
          v = maybe.get(k)
          if isinstance(v, str):
            parts = [p.strip().upper() for p in v.split(",")]
            cats = [p for p in parts if p in VALID_AI_CATEGORIES][:2]
            break
          if isinstance(v, list):
            parts = [str(x).strip().upper() for x in v if isinstance(x, str)]
            cats = [p for p in parts if p in VALID_AI_CATEGORIES][:2]
            break
      elif isinstance(maybe, list):
        parts = [str(x).strip().upper() for x in maybe if isinstance(x, str)]
        cats = [p for p in parts if p in VALID_AI_CATEGORIES][:2]
    except Exception:
      pass

    # Fallback to plain text parsing
    if cats is None:
      parts = [p.strip().upper() for p in out.split(",")]
      cats = [p for p in parts if p in VALID_AI_CATEGORIES][:2]

    if not cats:
      return False

    return cats

  except Exception as e:
    print(f"⚠️ Error calling/parsing Claude paper categorization: {e}")
    return False

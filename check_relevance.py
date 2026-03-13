"""
Unified AI-relevance checker used by all scrapers.
"""
from __future__ import annotations
import time
from config import CLAUDE

MAX_RETRIES = 3
RETRY_DELAY = 2


def check_ai_relevance(text: str, max_retries: int = MAX_RETRIES) -> bool:
    """
    Ask Claude whether *text* discusses AI/ML content relevant to
    researchers, engineers, or data scientists.

    Returns True if relevant, False otherwise.
    Defaults to True on API failure (conservative — don't filter out).
    """
    prompt = f"""You are an AI expert evaluating social media posts for AI/ML relevance.

Determine if this tweet/post is discussing AI/ML MODELS, TOOLS, or RESEARCH that would be BENEFICIAL and RELEVANT to:
- AI researchers (advancing the field)
- AI companies (building AI products/services)
- AI developers (implementing AI systems)
- Data scientists (working with AI/ML)
- ML engineers (training and deploying models)

Consider the tweet relevant if it discusses:
- AI/ML model releases or updates (LLMs, vision models, audio models, etc.)
- Machine learning techniques, algorithms, or architectures
- Large language models, transformers, or foundation models
- AI applications, use cases, or implementations
- Deep learning, neural networks, or AI systems
- Model benchmarks, evaluations, or performance comparisons
- Data processing, feature engineering, or model training methods
- AI safety, alignment, or ethics with practical implications
- Tools, frameworks, or infrastructure for AI/ML development
- Computer vision, NLP, robotics with AI/ML components
- Research papers that advance AI in a meaningful way
- HuggingFace model releases, datasets, or spaces
- GitHub repositories for AI/ML models, training, inference, evaluation, tooling, or infrastructure

Do NOT consider relevant if the tweet:
- Discusses pure mathematics without clear AI application
- Discusses general software engineering without AI focus
- Discusses hardware without AI/ML optimization focus
- Discusses purely theoretical physics or chemistry without AI application
- Is casual conversation, memes, or personal updates without AI content
- Is promotional content without substantial AI/ML discussion
- Is a list or roundup of multiple repos without depth

Answer with ONLY "YES" or "NO".

Tweet/Post:
{text}"""

    for attempt in range(max_retries):
        try:
            r = CLAUDE.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=5,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            ans = (r.content[0].text or "").strip().upper()
            return ans == "YES"
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Relevance check failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"⚠️ Relevance check failed after {max_retries} attempts: {e}")
                return True  # conservative default

    return True
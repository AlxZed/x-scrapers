from __future__ import annotations
import os

from pymongo import MongoClient
from anthropic import Anthropic

import install_shared

install_shared.install_writing_system()

from tweet_utils import build_session

# ── API keys ──────────────────────────────────

SOCIALDATA_BEARER = os.environ.get("SOCIALDATA_BEARER", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# ── Clients ───────────────────────────────────

SESSION = build_session(SOCIALDATA_BEARER)
CLAUDE = Anthropic(api_key=ANTHROPIC_KEY)

# ── MongoDB ───────────────────────────────────

_client = MongoClient(os.environ["MONGO_URI_HEADLINE"])
_sources = _client.get_database("sources")

PAPERS = _sources.arxiv
TWEETS_ARXIV = _sources.twitter_arxiv
TWEETS_GITHUB = _sources.twitter_github
TWEETS_HUGGINGFACE = _sources.twitter_huggingface
NON_AI_TWEETS = _sources.twitter_non_ai
REPOS = _sources.github
MODELS = _sources.huggingface

# ── Search defaults ───────────────────────────

DEFAULT_MIN_FAVES = 50
DEFAULT_WITHIN_TIME = "3d"

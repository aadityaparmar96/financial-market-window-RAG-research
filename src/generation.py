"""
generation.py
-------------
Financial Market RAG System — Generation Module

Responsibility: Given a question and a target time window, retrieve relevant
context using WindowRetreiver, construct a constrained prompt, and call the
Claude API to produce a grounded answer. Also supports a no-RAG baseline
mode for measuring base model knowledge leakage.

This module does NOT:
    - Build or populate collections (that's embeddings.py)
    - Retrieve chunks directly (that's retrieval.py — this module calls it)
    - Score or evaluate answers (that's evaluation.py, built later)

Depends on:
    - retrieval.py (WindowRetreiver, VALID_WINDOWS)
    - An Anthropic API key set as an environment variable (ANTHROPIC_API_KEY)
"""

import os
import logging
from typing import Optional

import anthropic

from retrieval import WindowRetreiver, VALID_WINDOWS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generation")
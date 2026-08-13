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

# Constant prompts and conditons

MODEL_NAME = "claude-haiku-4-5"
MAX_TOKENS = 600

SYSTEM_PROMPT = """You are a financial research assistant analyzing historical economic data.

CRITICAL RULES:
1. Answer ONLY using the context passages provided below.
2. Do NOT use knowledge from your own training data.
3. If the context does not contain enough information to answer the question,
   respond with exactly: INSUFFICIENT CONTEXT
4. Always cite which time period your reasoning draws from.
5. Never guess. Never supplement with outside knowledge."""

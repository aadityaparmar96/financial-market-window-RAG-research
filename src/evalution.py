#evaluation metrics:
#1 point for correct-
#0.5 points for partial
#0 for completely off
#to be compared vs lr 
"""
evaluation.py
-------------
Financial Market RAG System — Evaluation Module

Responsibility: Take raw answers produced by generation.py (baseline + 4 RAG
windows, per question), apply the manual scoring rubric, compute the
improvement-over-baseline metric, and run statistical tests across the
scored results.

This module does NOT:
    - Call the LLM (that's generation.py)
    - Retrieve chunks (that's retrieval.py)
    - Build knowledge bases (that's embeddings.py)

Scoring is manual by design — the researcher reads each answer and assigns
a score using score_answer(). This file provides the structure, storage,
and statistical machinery around that manual judgment; it does not attempt
to automate the scoring decision itself.

On A second note, we might consider having a LLM judge to automatically evaluate the answers.

Scoring is automated via a constrained LLM judge — see JUDGE_SYSTEM_PROMPT
below. The judge is given the verified correct answer explicitly and
instructed to score against it, not against its own general knowledge,
to avoid reintroducing the same leakage problem this study measures in
the generation stage.
"""


import json
import logging
from pathlib import Path
from typing import TypedDict, Literal, Optional
from datetime import datetime
import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluation")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITIONS = ["baseline", "5yr", "10yr", "20yr", "50yr"]
RAG_CONDITIONS = ["5yr", "10yr", "20yr", "50yr"]  # excludes baseline

VALID_SCORES = [0.0, 0.5, 1.0]
QUESTION_TYPES = ["FACT", "TREND"]

JUDGE_MODEL = "claude-sonnet-5"   # not confirmed yet 
JUDGE_MAX_TOKENS = 300
# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class QuestionRecord(TypedDict):
    """
    One entry in your locked eval question set — matches the format we've
    been building question by question (Category A, B, ...).
    """
    id: str                 # e.g. "A1", "B3"
    category: str           # e.g. "monetary_policy_cycle"
    question_type: Literal["FACT", "TREND"]
    question_text: str
    correct_answer: str
    source: str
    is_leakage_probe: bool  # True for Q11/Q12-style obscure-fact questions

class ScoredAnswer(TypedDict):
    """
    One scored answer: a specific question, asked under a specific
    condition (baseline or one of the four RAG windows), with a manually
    assigned score.
    """
    question_id: str
    condition: str           # "baseline", "5yr", "10yr", "20yr", "50yr"
    raw_answer: str          # the full text Claude returned
    score: Optional[float]   # None until manually scored; then 0.0/0.5/1.0
    leaked: Optional[bool]   # only meaningful for leakage-probe questions
    scorer: str              # who assigned this score, e.g. "primary" or "rater2"
    notes: str                # free-text justification, esp. for 0.5 scores
    judge_reasoning: str     # the judge's stated justification, for audit
    scorer: str               # always "llm_judge" now, kept for schema stability




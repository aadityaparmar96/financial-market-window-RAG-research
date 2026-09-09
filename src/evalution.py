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
"""


import json
import logging
from pathlib import Path
from typing import TypedDict, Literal, Optional
from datetime import datetime

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






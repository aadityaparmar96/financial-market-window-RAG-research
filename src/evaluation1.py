"""
evaluation.py
-------------
Financial Market RAG System — Evaluation Module

Responsibility: Take raw answers produced by generation.py (baseline + 4 RAG
windows, per question), score them automatically using an LLM judge against
the locked correct answer, extract comparable numeric predictions where
possible, compute the improvement-over-baseline metric, and run statistical
tests across the scored results.

This module does NOT:
    - Call the generation LLM (that's generation.py)
    - Retrieve chunks (that's retrieval.py)
    - Build knowledge bases (that's embeddings.py)

Scoring has two independent components:
    1. Reason score (0.0 / 0.5 / 1.0) — rubric-based judgment of direction,
       magnitude reasonableness, and precedent quality.
    2. Numeric error score — absolute and relative error between the
       verified correct figure and the model's stated range, computed
       ONLY for questions where both sides contain an actual number.
       Questions without a comparable numeric target report None here,
       which is expected and correct, not a failure.

Both scores are extracted by the same LLM judge call, then validated
against a human-reviewed subsample (see inter-rater reliability notes).
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import TypedDict, Literal, Optional

import numpy as np
from scipy import stats
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
RAG_CONDITIONS = ["5yr", "10yr", "20yr", "50yr"]

VALID_SCORES = [0.0, 0.5, 1.0]
QUESTION_TYPES = ["FACT", "TREND"]

JUDGE_MODEL = "claude-sonnet-5"   # confirm exact model string before locking
JUDGE_MAX_TOKENS = 400


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class QuestionRecord(TypedDict):
    id: str
    category: str
    question_type: Literal["FACT", "TREND"]
    question_text: str
    correct_answer: str
    source: str
    is_leakage_probe: bool


class NumericError(TypedDict):
    actual: float
    predicted_midpoint: float
    predicted_range: list          # [low, high]
    absolute_error: float
    relative_error_pct: Optional[float]
    range_covered_truth: bool
    unit: Optional[str]


class ScoredAnswer(TypedDict):
    question_id: str
    condition: str
    raw_answer: str
    score: Optional[float]              # reason score: 0.0/0.5/1.0, or None
    leaked: Optional[bool]              # only meaningful for leakage probes
    judge_reasoning: str
    numeric_error: Optional[NumericError]  # None if no comparable number
    scorer: str


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a strict, mechanical grader for a research study. You are NOT being asked for your own knowledge or opinion about what actually happened historically.

You will be given:
1. A question that was asked to a test AI system
2. The VERIFIED CORRECT ANSWER (treat this as ground truth — do not second-guess it, do not substitute your own knowledge of history in its place)
3. The test AI's actual answer, structured as DIRECTION / MAGNITUDE / PRECEDENT

Score the test AI's answer using ONLY this rubric:

1.0 — ALL of the following are true:
    (a) DIRECTION matches the verified correct answer
    (b) MAGNITUDE is within a reasonable range of the verified correct answer
    (c) PRECEDENT cites a specific, real, relevant historical reference

0.5 — ONE of the following:
    (a) DIRECTION correct but MAGNITUDE clearly outside a reasonable range
    (b) DIRECTION and MAGNITUDE reasonable, but PRECEDENT vague, missing, or not clearly relevant
    (c) Sound reasoning but a conclusion that only partially matches the verified answer

0.0 — ANY of the following:
    (a) DIRECTION contradicts the verified correct answer
    (b) The answer is "INSUFFICIENT CONTEXT"
    (c) PRECEDENT cites a fabricated or clearly irrelevant event
    (d) States a specific numerical fact that contradicts the verified answer with false confidence

Additionally, extract numeric values if and only if BOTH the verified correct answer AND the test AI's MAGNITUDE field contain an actual number or numeric range (a percentage, a rate, a count of years/months, a basis-point figure). If either side is categorical with no genuine figure (e.g. "hike vs cut," "months vs years" with no specific number), output N/A for all numeric fields. Do not invent or estimate a number that was not actually stated. If the AI gave a single point value rather than a range, use that same value for both LOW and HIGH.

Respond in EXACTLY this format, nothing else:
SCORE: [0.0, 0.5, or 1.0]
REASONING: [one or two sentences explaining which rubric condition applied]
NUMERIC_ACTUAL: [number from the verified answer, or N/A]
NUMERIC_PREDICTED_LOW: [low end of the AI's stated range, or N/A]
NUMERIC_PREDICTED_HIGH: [high end of the AI's stated range, or N/A]
NUMERIC_UNIT: [e.g. "percent", "percentage_points", "years", "basis_points", or N/A]"""


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class AnswerJudge:
    """
    Wraps a Claude client configured specifically for scoring, kept
    separate from AnswerGenerator's client in generation.py so the
    generation and judging roles never share prompt context or state.
    """

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Set it before running "
                "evaluation.py."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info("AnswerJudge initialized (model=%s)", JUDGE_MODEL)

    def score(
        self,
        question: QuestionRecord,
        raw_answer: str,
        condition: str,
    ) -> ScoredAnswer:
        """
        Score one answer. Leakage-probe questions are routed to a plain
        regex check instead of the LLM judge — binary leak/no-leak is
        directly checkable and doesn't need judgment.
        """
        if question["is_leakage_probe"]:
            return self._check_leakage(question, raw_answer, condition)

        judge_message = (
            f"QUESTION: {question['question_text']}\n\n"
            f"VERIFIED CORRECT ANSWER: {question['correct_answer']}\n\n"
            f"TEST AI'S ANSWER:\n{raw_answer}\n\n"
            f"Score this answer using the rubric."
        )

        response = self.client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": judge_message}],
        )

        judge_text = response.content[0].text
        parsed = self._parse_judge_response(judge_text)
        numeric_error = compute_numeric_error(parsed)

        return {
            "question_id": question["id"],
            "condition": condition,
            "raw_answer": raw_answer,
            "score": parsed["score"],
            "leaked": None,
            "judge_reasoning": parsed["reasoning"],
            "numeric_error": numeric_error,
            "scorer": "llm_judge",
        }

    def _parse_judge_response(self, judge_text: str) -> dict:
        def _get(pattern: str) -> Optional[str]:
            m = re.search(pattern, judge_text, re.DOTALL)
            return m.group(1).strip() if m else None

        score_raw = _get(r"SCORE:\s*(0\.0|0\.5|1\.0)")
        reasoning = _get(r"REASONING:\s*(.+?)(?=\nNUMERIC_ACTUAL:|\Z)")
        actual_raw = _get(r"NUMERIC_ACTUAL:\s*(.+)")
        low_raw = _get(r"NUMERIC_PREDICTED_LOW:\s*(.+)")
        high_raw = _get(r"NUMERIC_PREDICTED_HIGH:\s*(.+)")
        unit = _get(r"NUMERIC_UNIT:\s*(.+)")

        def _to_float(s: Optional[str]) -> Optional[float]:
            if not s or s.strip().upper().startswith("N/A"):
                return None
            match = re.search(r"-?\d+\.?\d*", s)
            return float(match.group(0)) if match else None

        if not score_raw:
            logger.error("Could not parse judge score from: %s", judge_text[:200])

        return {
            "score": float(score_raw) if score_raw else None,
            "reasoning": reasoning or "",
            "numeric_actual": _to_float(actual_raw),
            "numeric_predicted_low": _to_float(low_raw),
            "numeric_predicted_high": _to_float(high_raw),
            "numeric_unit": unit if unit and not unit.strip().upper().startswith("N/A") else None,
        }

    def _check_leakage(
        self,
        question: QuestionRecord,
        raw_answer: str,
        condition: str,
    ) -> ScoredAnswer:
        """
        Plain regex check — no LLM call needed for a binary leak/no-leak
        signal on a specific known figure.
        """
        target_match = re.search(r"[\d,]+\.?\d*%?", question["correct_answer"])
        leaked = False
        if target_match:
            leaked = target_match.group(0) in raw_answer

        return {
            "question_id": question["id"],
            "condition": condition,
            "raw_answer": raw_answer,
            "score": None,
            "leaked": leaked,
            "judge_reasoning": (
                f"Leakage check: looked for "
                f"'{target_match.group(0) if target_match else 'N/A'}' — "
                f"{'FOUND (leak)' if leaked else 'not found'}."
            ),
            "numeric_error": None,
            "scorer": "automated_leakage_check",
        }



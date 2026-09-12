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
import os
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

    # ---------------------------------------------------------------------------
    # Judge Instructions
    # ---------------------------------------------------------------------------
    
JUDGE_SYSTEM_PROMPT = """You are a strict, mechanical grader for a research study. You are NOT being asked for your own knowledge or opinion about what actually happened historically.

You will be given:
1. A question that was asked to a test AI system
2. The VERIFIED CORRECT ANSWER (treat this as ground truth — do not second-guess it, do not substitute your own knowledge of history in its place)
3. The test AI's actual answer, structured as DIRECTION / MAGNITUDE / PRECEDENT

Score the test AI's answer using ONLY this rubric:

1.0 — ALL of the following are true:
    (a) DIRECTION matches the verified correct answer
    (b) MAGNITUDE is within a reasonable range of the verified correct answer (not exact — a plausible estimate that would contain or nearly contain the true value)
    (c) PRECEDENT cites a specific, real, relevant historical reference

0.5 — ONE of the following:
    (a) DIRECTION is correct but MAGNITUDE is clearly outside a reasonable range
    (b) DIRECTION is correct and MAGNITUDE is reasonable, but PRECEDENT is vague, missing, or not clearly relevant
    (c) The reasoning process is sound but arrives at a conclusion that only partially matches the verified answer

0.0 — ANY of the following:
    (a) DIRECTION contradicts the verified correct answer
    (b) The answer is "INSUFFICIENT CONTEXT"
    (c) PRECEDENT cites a fabricated or clearly irrelevant event
    (d) The answer states a specific numerical fact that contradicts the verified answer with false confidence

Respond in EXACTLY this format, nothing else:
SCORE: [0.0, 0.5, or 1.0]
REASONING: [one or two sentences explaining which rubric condition applied]"""


#---------------------
# JUDGE FUNCTION:
#---------------------

class AnswerJudge:
    """
    Wraps a Claude client configured specifically for scoring, separate
    from the AnswerGenerator client in generation.py — kept distinct so
    the generation and judging roles never accidentally share prompt
    context or state.
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
        Send one answer to the judge and parse back a ScoredAnswer.

        Leakage-probe questions are NOT scored on the 0/0.5/1.0 rubric —
        they're checked separately for whether the exact figure appears
        in the answer at all, since that's a binary leak/no-leak signal,
        not a graded reasoning quality.
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
        score, reasoning = self._parse_judge_response(judge_text)

        return {
            "question_id": question["id"],
            "condition": condition,
            "raw_answer": raw_answer,
            "score": score,
            "leaked": None,
            "judge_reasoning": reasoning,
            "scorer": "llm_judge",
        }
    def _check_leakage(
        self,
        question: QuestionRecord,
        raw_answer: str,
        condition: str,
    ) -> ScoredAnswer:
        """
        For leakage-probe questions: check whether the verified specific
        figure appears in the answer at all. This is a plain string/regex
        check, not an LLM judgment call — deliberately, since "did this
        exact number appear" doesn't need reasoning, and using the LLM
        judge here would be unnecessary cost and an unnecessary point of
        failure for something checkable directly.
        """
        # Extract the key figure from the correct answer, e.g. "4.2%"
        target_match = re.search(r"[\d,]+\.?\d*%?", question["correct_answer"])
        leaked = False
        if target_match:
            target_value = target_match.group(0)
            leaked = target_value in raw_answer

        return {
            "question_id": question["id"],
            "condition": condition,
            "raw_answer": raw_answer,
            "score": None,  # leakage probes aren't scored on the rubric
            "leaked": leaked,
            "judge_reasoning": (
                f"Leakage check: looked for '{target_match.group(0) if target_match else 'N/A'}' "
                f"in answer — {'FOUND (leak)' if leaked else 'not found'}."
            ),
            "scorer": "automated_leakage_check",
        }
            
# ---------------------------------------------------------------------------
# Loading questions and generation results
# ---------------------------------------------------------------------------

def load_questions(path: str = "data/eval_questions.json") -> list[QuestionRecord]:
    """
    Load the locked eval question set from JSON. This file is hand-edited
    as new questions are verified and added on the daily cadence — this
    function just needs to read whatever is currently there.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Question set not found at {filepath}. "
            f"Create it before running evaluation."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        questions = json.load(f)
    logger.info("Loaded %d questions from %s", len(questions), filepath)
    return questions



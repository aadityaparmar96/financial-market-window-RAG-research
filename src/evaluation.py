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
from generation import MODEL_NAME as JUDGE_MODEL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITIONS = ["baseline", "5yr", "10yr", "20yr", "50yr"]
RAG_CONDITIONS = ["5yr", "10yr", "20yr", "50yr"]

VALID_SCORES = [0.0, 0.5, 1.0]
QUESTION_TYPES = ["FACT", "TREND"]

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

# ---------------------------------------------------------------------------
# Numeric error computation
# ---------------------------------------------------------------------------

def compute_numeric_error(parsed: dict) -> Optional[NumericError]:
    """
    Compute absolute and relative error from a parsed judge response.
    Returns None if the question had no extractable numeric target on
    both sides — expected and correct for many questions, not a failure.
    """
    actual = parsed["numeric_actual"]
    low = parsed["numeric_predicted_low"]
    high = parsed["numeric_predicted_high"]

    if actual is None or low is None or high is None:
        return None

    midpoint = (low + high) / 2
    absolute_error = abs(midpoint - actual)
    relative_error = (absolute_error / abs(actual)) * 100 if actual != 0 else None
    covered = low <= actual <= high

    return {
        "actual": actual,
        "predicted_midpoint": midpoint,
        "predicted_range": [low, high],
        "absolute_error": absolute_error,
        "relative_error_pct": relative_error,
        "range_covered_truth": covered,
        "unit": parsed["numeric_unit"],
    }


# ---------------------------------------------------------------------------
# Loading questions
# ---------------------------------------------------------------------------

def load_questions(path: str = "data/eval_questions.json") -> list[QuestionRecord]:
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Question set not found at {filepath}.")
    with open(filepath, "r", encoding="utf-8") as f:
        questions = json.load(f)
    logger.info("Loaded %d questions from %s", len(questions), filepath)
    return questions


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    questions: list[QuestionRecord],
    generator,          # an AnswerGenerator instance from generation.py
    judge: AnswerJudge,
    output_path: str = "results/scored_results.json",
) -> list[ScoredAnswer]:
    all_scored: list[ScoredAnswer] = []

    for i, question in enumerate(questions, 1):
        logger.info(
            "[%d/%d] Processing %s (%s)",
            i, len(questions), question["id"], question["question_type"]
        )

        generation_results = generator.generate_all_conditions(question["question_text"])

        for condition, result in generation_results.items():
            scored = judge.score(question, result["answer"], condition)
            all_scored.append(scored)

            if scored["leaked"] is not None:
                logger.info("  %s: LEAKAGE — %s",
                            condition, "LEAKED" if scored["leaked"] else "clean")
            else:
                numeric_note = ""
                if scored["numeric_error"]:
                    numeric_note = f" (abs_err={scored['numeric_error']['absolute_error']:.2f})"
                logger.info("  %s: score=%s%s", condition, scored["score"], numeric_note)

    _save_results(all_scored, output_path)
    return all_scored


def _save_results(results: list[ScoredAnswer], output_path: str) -> None:
    filepath = Path(output_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d scored results to %s", len(results), filepath)


# ---------------------------------------------------------------------------
# Improvement-over-baseline (reason score)
# ---------------------------------------------------------------------------

def compute_improvement_scores(
    scored_results: list[ScoredAnswer],
) -> dict[str, dict[str, float]]:
    by_question: dict[str, dict[str, ScoredAnswer]] = {}
    for r in scored_results:
        by_question.setdefault(r["question_id"], {})[r["condition"]] = r

    improvements: dict[str, dict[str, float]] = {}

    for qid, condition_map in by_question.items():
        baseline = condition_map.get("baseline")
        if baseline is None or baseline["score"] is None:
            logger.warning("Skipping %s: no scored baseline.", qid)
            continue

        improvements[qid] = {}
        for window in RAG_CONDITIONS:
            rag_result = condition_map.get(window)
            if rag_result is None or rag_result["score"] is None:
                continue
            improvements[qid][window] = rag_result["score"] - baseline["score"]

    return improvements


# ---------------------------------------------------------------------------
# Numeric error aggregation (numeric score)
# ---------------------------------------------------------------------------

def aggregate_numeric_errors(
    scored_results: list[ScoredAnswer],
) -> dict[str, dict]:
    """
    For each condition, compute mean absolute error, mean relative error,
    and coverage rate across only the questions that had a comparable
    numeric target. Reports the subset size explicitly, since this is
    never computed across your full question set.
    """
    by_condition: dict[str, list[NumericError]] = {c: [] for c in CONDITIONS}

    for r in scored_results:
        if r["numeric_error"] is not None:
            by_condition[r["condition"]].append(r["numeric_error"])

    summary = {}
    for condition, errors in by_condition.items():
        if not errors:
            summary[condition] = {"n": 0}
            continue

        abs_errors = [e["absolute_error"] for e in errors]
        rel_errors = [e["relative_error_pct"] for e in errors if e["relative_error_pct"] is not None]
        coverage = [e["range_covered_truth"] for e in errors]

        summary[condition] = {
            "n": len(errors),
            "mean_absolute_error": float(np.mean(abs_errors)),
            "std_absolute_error": float(np.std(abs_errors, ddof=1)) if len(abs_errors) > 1 else 0.0,
            "mean_relative_error_pct": float(np.mean(rel_errors)) if rel_errors else None,
            "coverage_rate": float(np.mean(coverage)),
        }

    return summary

# ---------------------------------------------------------------------------
# Statistics on the reason-score improvements
# ---------------------------------------------------------------------------

def summarize_improvements(
    improvements: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """
    Mean, std, and 95% confidence interval of improvement-over-baseline
    per window, across all questions that had a valid delta.
    """
    by_window: dict[str, list[float]] = {w: [] for w in RAG_CONDITIONS}
    for qid, deltas in improvements.items():
        for window, delta in deltas.items():
            by_window[window].append(delta)

    summary = {}
    for window, deltas in by_window.items():
        if len(deltas) < 2:
            summary[window] = {"n": len(deltas), "mean": None, "ci_95": None}
            continue
        mean = float(np.mean(deltas))
        sem = stats.sem(deltas)
        ci = stats.t.interval(0.95, len(deltas) - 1, loc=mean, scale=sem)
        summary[window] = {
            "n": len(deltas),
            "mean": mean,
            "std": float(np.std(deltas, ddof=1)),
            "ci_95": [float(ci[0]), float(ci[1])],
        }
    return summary


def cochrans_q_test(
    questions: list[QuestionRecord],
    scored_results: list[ScoredAnswer],
    question_type_filter: Optional[str] = None,
) -> dict:
    """
    Cochran's Q test on whether window condition significantly affects
    the (binarized) reason score, across paired questions.

    Scores are binarized at >= 1.0 = success, < 1.0 = failure, since
    Cochran's Q requires binary outcomes. This is a simplification —
    stated explicitly, since your actual scores are 0/0.5/1.0, not
    strictly binary — worth noting in your methodology as a modeling
    choice, not an oversight.
    """
    qtype_map = {q["id"]: q["question_type"] for q in questions}

    by_question: dict[str, dict[str, float]] = {}
    for r in scored_results:
        if r["score"] is None:
            continue
        if question_type_filter and qtype_map.get(r["question_id"]) != question_type_filter:
            continue
        by_question.setdefault(r["question_id"], {})[r["condition"]] = r["score"]

    # Only keep questions with a complete row across all 5 conditions
    complete = {
        qid: scores for qid, scores in by_question.items()
        if all(c in scores for c in CONDITIONS)
    }

    if len(complete) < 3:
        return {"error": f"Only {len(complete)} complete question rows — too few for Cochran's Q."}

    matrix = np.array([
        [1 if scores[c] >= 1.0 else 0 for c in CONDITIONS]
        for scores in complete.values()
    ])

    k = matrix.shape[1]
    n = matrix.shape[0]
    col_sums = matrix.sum(axis=0)
    row_sums = matrix.sum(axis=1)
    grand_sum = matrix.sum()

    numerator = k * (k - 1) * np.sum((col_sums - grand_sum / k) ** 2)
    denominator = k * grand_sum - np.sum(row_sums ** 2)
    q_stat = numerator / denominator if denominator != 0 else 0.0
    df = k - 1
    p_value = 1 - stats.chi2.cdf(q_stat, df)

    return {
        "n_questions": n,
        "q_statistic": float(q_stat),
        "df": df,
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "question_type_filter": question_type_filter or "ALL",
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run from the project root, after generation.py and retrieval.py are
    working and ANTHROPIC_API_KEY is set:

        python src/evaluation.py
    """
    from generation import AnswerGenerator  # local import, avoids circularity

    questions = load_questions()
    generator = AnswerGenerator()
    judge = AnswerJudge()

    scored = run_evaluation(questions, generator, judge)

    improvements = compute_improvement_scores(scored)
    improvement_summary = summarize_improvements(improvements)
    numeric_summary = aggregate_numeric_errors(scored)
    q_all = cochrans_q_test(questions, scored)
    q_fact = cochrans_q_test(questions, scored, question_type_filter="FACT")
    q_trend = cochrans_q_test(questions, scored, question_type_filter="TREND")

    print(f"\n{'='*60}\nREASON SCORE — Improvement over baseline\n{'='*60}")
    for window, s in improvement_summary.items():
        if s["mean"] is not None:
            print(f"{window:6s} n={s['n']:3d}  mean={s['mean']:+.3f}  "
                  f"95% CI=[{s['ci_95'][0]:+.3f}, {s['ci_95'][1]:+.3f}]")

    print(f"\n{'='*60}\nNUMERIC SCORE — Error by condition\n{'='*60}")
    for cond, s in numeric_summary.items():
        if s["n"] > 0:
            print(f"{cond:10s} n={s['n']:3d}  MAE={s['mean_absolute_error']:.2f}  "
                  f"MAPE={s.get('mean_relative_error_pct', 0):.1f}%  "
                  f"coverage={s['coverage_rate']*100:.0f}%")
        else:
            print(f"{cond:10s} n=0 (no comparable numeric questions)")

    print(f"\n{'='*60}\nCOCHRAN'S Q TEST\n{'='*60}")
    print("ALL:  ", q_all)
    print("FACT: ", q_fact)
    print("TREND:", q_trend)

    with open("results/final_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "improvement_summary": improvement_summary,
            "numeric_summary": numeric_summary,
            "cochrans_q_all": q_all,
            "cochrans_q_fact": q_fact,
            "cochrans_q_trend": q_trend,
        }, f, indent=2, ensure_ascii=False)

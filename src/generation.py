"""
generation.py
-------------
Financial Market RAG System — Generation Module

Depends on:
    - retrieval.py (WindowRetreiver, VALID_WINDOWS)
    - An Anthropic API key set as an environment variable (ANTHROPIC_API_KEY)
"""

import os
import logging
import anthropic

from retrieval import WindowRetreiver, VALID_WINDOWS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generation")

MODEL_NAME = "claude-sonnet-4-5"   # confirm exact current model string before real runs
MAX_TOKENS = 600

REFINED_SYSTEM_PROMPT = """You are a financial research assistant analyzing historical economic data to reason about market conditions.

CRITICAL RULES:
1. Answer ONLY using the context passages provided below. Do NOT use knowledge from your own training data, even if you recognize the scenario described.
2. If the context does not contain enough information to answer meaningfully, respond with exactly: INSUFFICIENT CONTEXT — and briefly state what specific information is missing.
3. Never state a specific numerical fact (a date, percentage, or figure) unless it appears directly in the provided context. If you are estimating or inferring a range rather than reading an exact figure, say so explicitly.

REQUIRED ANSWER STRUCTURE:
Structure every answer in exactly these three labeled parts, in this order:

DIRECTION: State your prediction or conclusion in one sentence — a clear directional call (e.g., "increase," "recover within X months," "yes/no").

MAGNITUDE: State your estimated size or scale of the effect, as a range if you are not citing an exact figure from context (e.g., "roughly 15–25%," "within 1–2 years"). If the question does not require a magnitude, write "N/A" here.

PRECEDENT: Name the specific historical period or data point from the provided context that your reasoning is based on, including its approximate date. If no specific precedent from the context supports your answer, say so explicitly rather than inventing one.

Do not add commentary outside these three labeled sections. Do not hedge with phrases like "it depends" without still committing to a DIRECTION based on the context given."""

BASELINE_SYSTEM_PROMPT = """Answer the following question directly, structuring your answer in exactly these three labeled parts:

DIRECTION: your prediction or conclusion in one sentence.
MAGNITUDE: your estimated size or scale, as a range if uncertain. Write "N/A" if not applicable.
PRECEDENT: any specific historical event or data you are basing your reasoning on, if any.

Do not add commentary outside these three sections."""


class AnswerGenerator:
    """
    Wraps the Anthropic client and a WindowRetreiver, exposing:
    - generate_rag(question, window)  → retrieval-augmented answer
    - generate_baseline(question)     → raw LLM answer, no context at all
    """

    def __init__(self, chromadb_path: str = "./chromadb"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.retriever = WindowRetreiver(chromadb_path=chromadb_path)
        logger.info("AnswerGenerator initialized (model=%s)", MODEL_NAME)

    def generate_rag(
        self,
        question: str,
        window: str,
        n_results: int = 5,
    ) -> dict:
        if window not in VALID_WINDOWS:
            raise ValueError(
                f"Invalid Window '{window}'. Must be one of {VALID_WINDOWS}"
            )

        chunks = self.retriever.retreive(question, window, n_results)
        context = self.retriever.format_context(chunks)

        user_message = (
            f"CONTEXT FROM KNOWLEDGE BASE ({window} window, ending 2015-12-31):\n"
            f"{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"Answer using the required DIRECTION / MAGNITUDE / PRECEDENT structure, "
            f"based strictly on the context above."
        )

        logger.info(
            "Generating RAG answer | window=%s | chunks_retrieved=%d",
            window, len(chunks)
        )

        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=REFINED_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return {
            "answer": response.content[0].text,
            "window": window,
            "retrieved_chunks": chunks,
            "context_used": context,
        }

    def generate_baseline(self, question: str) -> dict:
        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=BASELINE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )

        return {
            "answer": response.content[0].text,
            "window": "baseline",
            "retrieved_chunks": [],
            "context_used": None,
        }

    def generate_all_conditions(self, question: str, n_results: int = 5) -> dict:
        results = {"baseline": self.generate_baseline(question)}
        for window in VALID_WINDOWS:
            results[window] = self.generate_rag(question, window, n_results)
        return results


if __name__ == "__main__":
    generator = AnswerGenerator()

    test_question = (
        "Based on historical patterns, how does the S&P 500 typically "
        "perform in the 12 months following a Federal Reserve interest "
        "rate cut that follows a period of aggressive tightening?"
    )

    print(f"\nTest question: \"{test_question}\"\n")
    print("=" * 70)

    results = generator.generate_all_conditions(test_question)

    for condition, result in results.items():
        print(f"\n[{condition.upper()}]")
        print(f"Chunks retrieved: {len(result['retrieved_chunks'])}")
        print(f"Answer:\n{result['answer'][:500]}")
        print("-" * 70)
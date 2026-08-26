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



# Generator 
"""
    Wraps the Anthropic client and a WindowRetreiver, exposing two modes:

    - generate_rag(question, window)  → retrieval-augmented answer
    - generate_baseline(question)     → raw LLM answer, no context at all

    Both modes are needed because the baseline is how you measure how much
    of the model's answer comes from its own training data rather than
    from the retrieved knowledge base, to measure the leakage
    """

class AnswerGenerator:

    def __init__(self, chromadb_path: str = "./chromadb"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
             raise EnvironmentError(
                 "API KEY is not set"
             )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.retriever = WindowRetreiver(chromadb_path=chromadb_path)
        logger.info("AnswerGenerator initialized (model=%s)", MODEL_NAME)

    def generate_rag(
            self,
            question: str,
            window: str,
            n_results: 5,
    )-> dict:
        if window not in VALID_WINDOWS:
            raise ValueError(
                f"Invalid Window '{window}'. Must be one of '{VALID_WINDOWS}"
            )

        chunks = self.retriever.retreive(question, window, n_results)
        context = self.retriever.format_context(chunks)

        user_message = (
            f"CONTEXT FROM KNOWLEDGE BASE:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"Answer based strictly on the context above."
        )

        logger.info(
            "Generating RAG answer | window=%s | chunks_retrieved=%d",
            window, len(chunks)
        )
        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer_text = response.content[0].text

        return {
            "answer": answer_text,
            "window": window,
            "retrieved_chunks": chunks,
            "context_used": context,
        }

    def generate_baseline(self, question: str) -> dict:
        """Generate an answer without providing retrieved context."""
        user_message = (
            "No knowledge-base context is available. Answer the question using "
            "your own knowledge.\n\n"
            f"QUESTION: {question}"
        )

        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": user_message}],
        )

        return {
            "answer": response.content[0].text,
            "window": "baseline",
            "retrieved_chunks": [],
            "context_used": None,
        }

    def generate_all_conditions(self, question: str, n_results: int=5) -> dict:
        results = {"baseline": self.generate_baseline(question)}

        
        for window in VALID_WINDOWS:
            results[window] = self.generate_rag(question, window, n_results)

        return results

    """
        Convenience method: 

        Returns
        -------
        dict mapping condition name ("baseline", "5yr", "10yr", "20yr", "50yr")
        to the result dict from generate_baseline() or generate_rag().
        """


# CLI smoke test
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
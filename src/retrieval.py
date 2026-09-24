import logging
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrieval")


class RetrievedChunk(TypedDict):
    text: str
    date: str
    source: str
    dataset_type: str
    distance: float


VALID_WINDOWS = ["5yr", "10yr", "20yr", "50yr"]


class WindowRetreiver:
    def __init__(self, chromadb_path: str = "./chromadb"):
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("WindowRetreiver initialized (path=%s)", chromadb_path)

    def retreive(
        self,
        question: str,
        window: str,
        n_results: int = 5,
    ) -> list[RetrievedChunk]:

        if window not in VALID_WINDOWS:
            raise ValueError(
                f"Invalid window '{window}'. Must be one of {VALID_WINDOWS}."
            )

        collection_name = f"finance_{window}"

        try:
            collection = self.client.get_collection(collection_name)
        except Exception as exc:
            logger.error(
                "Collection '%s' not found. Did embeddings.py run "
                "successfully? (%s)", collection_name, exc
            )
            return []

        if collection.count() == 0:
            logger.warning("Collection '%s' is empty.", collection_name)
            return []

        results = collection.query(
            query_texts=[question],
            n_results=min(n_results, collection.count()),
        )

        chunks: list[RetrievedChunk] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append({
                "text": doc,
                "date": meta.get("date", "unknown"),
                "source": meta.get("source", "unknown"),
                "dataset_type": meta.get("dataset_type", "unknown"),
                "distance": float(dist),
            })

        return chunks

    def retrieve_diverse(
        self,
        question: str,
        window: str,
        per_source: int = 2,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top chunks PER SOURCE FILE, rather than top-k across the
        whole mixed collection. This prevents a single large source (e.g.
        the Shiller S&P500 dataset, which has far more rows than FEDFUNDS
        or UNRATE) from crowding out smaller, but potentially more directly
        relevant, sources for a given question.

        Retrieves per_source chunks independently from each dataset source
        present in the collection, then combines and re-sorts them by
        relevance so the strongest matches still lead the context block —
        every source just gets a guaranteed chance to be considered first.
        """
        if window not in VALID_WINDOWS:
            raise ValueError(
                f"Invalid window '{window}'. Must be one of {VALID_WINDOWS}."
            )

        collection_name = f"finance_{window}"

        try:
            collection = self.client.get_collection(collection_name)
        except Exception as exc:
            logger.error(
                "Collection '%s' not found. Did embeddings.py run "
                "successfully? (%s)", collection_name, exc
            )
            return []

        if collection.count() == 0:
            logger.warning("Collection '%s' is empty.", collection_name)
            return []

        sample = collection.get(
            limit=min(collection.count(), 2000),
            include=["metadatas"],
        )
        sources_present = sorted(set(
            m.get("source", "unknown") for m in sample["metadatas"]
        ))

        all_chunks: list[RetrievedChunk] = []
        for source in sources_present:
            results = collection.query(
                query_texts=[question],
                n_results=per_source,
                where={"source": source},
            )
            if not results["documents"][0]:
                continue
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                all_chunks.append({
                    "text": doc,
                    "date": meta.get("date", "unknown"),
                    "source": meta.get("source", "unknown"),
                    "dataset_type": meta.get("dataset_type", "unknown"),
                    "distance": float(dist),
                })

        all_chunks.sort(key=lambda c: c["distance"])

        logger.info(
            "retrieve_diverse | window=%s | sources=%d | total_chunks=%d",
            window, len(sources_present), len(all_chunks)
        )

        return all_chunks

    def retrieve_date_matched(
        self,
        question: str,
        window: str,
        per_source: int = 2,
        anchor_source: str = "S&P500",
    ) -> list[RetrievedChunk]:
        """
        Like retrieve_diverse(), but additionally ensures that for every
        non-anchor chunk retrieved (e.g. a FEDFUNDS reading for a given
        month), the corresponding anchor_source chunk (default: S&P500)
        for that SAME DATE is also included, if it exists in this window.

        This directly addresses a failure mode observed during testing:
        retrieve_diverse() could correctly surface a relevant macro
        indicator (e.g. FEDFUNDS showing a rate decline from 6.4% in
        2000-12 to 1.24% in 2002-12) without also surfacing the S&P 500
        price data for those same dates — leaving the model unable to
        connect a policy move to its actual market outcome, since the
        two pieces of context described different, unrelated months.

        Chunks added purely because they share a date with a retrieved
        chunk (rather than because they matched the query by similarity)
        are given distance=0.0 to mark them as exact date-alignment
        additions, not similarity-ranked results — useful for debugging
        or for later filtering if this behavior needs to be distinguished
        downstream.
        """
        base_chunks = self.retrieve_diverse(question, window, per_source)

        if not base_chunks:
            return base_chunks

        try:
            collection = self.client.get_collection(f"finance_{window}")
        except Exception as exc:
            logger.error(
                "Collection 'finance_%s' not found during date-matching. (%s)",
                window, exc
            )
            return base_chunks

        dates_needing_anchor = set(
            c["date"] for c in base_chunks if c["source"] != anchor_source
        )
        existing_anchor_dates = set(
            c["date"] for c in base_chunks if c["source"] == anchor_source
        )

        supplement_dates = dates_needing_anchor - existing_anchor_dates
        added_count = 0

        for date_str in supplement_dates:
            if date_str == "unknown":
                continue
            try:
                results = collection.get(
                    where={
                        "$and": [
                            {"source": anchor_source},
                            {"date": date_str},
                        ]
                    },
                    limit=1,
                    include=["documents", "metadatas"],
                )
            except Exception as exc:
                logger.warning(
                    "Date-match lookup failed for date=%s source=%s: %s",
                    date_str, anchor_source, exc
                )
                continue

            if results["documents"]:
                base_chunks.append({
                    "text": results["documents"][0],
                    "date": date_str,
                    "source": anchor_source,
                    "dataset_type": results["metadatas"][0].get("dataset_type", "generic"),
                    "distance": 0.0,
                })
                added_count += 1

        logger.info(
            "retrieve_date_matched | window=%s | base_chunks=%d | "
            "date_matched_additions=%d | total=%d",
            window, len(base_chunks) - added_count, added_count, len(base_chunks)
        )

        return base_chunks

    def retrieve_all_windows(
        self,
        question: str,
        n_results: int = 5,
    ) -> dict[str, list[RetrievedChunk]]:
        return {
            window: self.retreive(question, window, n_results)
            for window in VALID_WINDOWS
        }

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "(No relevant context was retrieved for this particular window.)"
        return "\n\n".join(
            f"[{c['date']}]: {c['text']}" for c in chunks
        )


if __name__ == "__main__":
    retriever = WindowRetreiver()

    test_query = "Federal Reserve interest rate cuts following economic shock"

    print("=== Standard retrieval (top-k across mixed collection) ===")
    for w in VALID_WINDOWS:
        try:
            result = retriever.retreive(test_query, w, n_results=1)
            print(f"{w}: OK, {len(result)} result(s)")
        except Exception as e:
            print(f"{w}: FAILED — {e}")

    print("\n=== Diverse retrieval (per-source, guaranteed representation) ===")
    for w in VALID_WINDOWS:
        try:
            result = retriever.retrieve_diverse(test_query, w, per_source=1)
            sources_returned = sorted(set(c["source"] for c in result))
            print(f"{w}: OK, {len(result)} result(s) from sources: {sources_returned}")
        except Exception as e:
            print(f"{w}: FAILED — {e}")

    print("\n=== Date-matched retrieval (diverse + S&P500 date alignment) ===")
    for w in VALID_WINDOWS:
        try:
            result = retriever.retrieve_date_matched(test_query, w, per_source=1)
            sp500_dates = sorted(set(c["date"] for c in result if c["source"] == "S&P500"))
            other_dates = sorted(set(c["date"] for c in result if c["source"] != "S&P500"))
            print(f"{w}: OK, {len(result)} total chunk(s)")
            print(f"    S&P500 dates present: {sp500_dates}")
            print(f"    Other-source dates:   {other_dates}")
        except Exception as e:
            print(f"{w}: FAILED — {e}")
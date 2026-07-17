import logging
from datetime import datetime, timedelta
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
    date : str
    source : str
    dataset_type = str
    distance = float

VALID_WINDOWS = ["5yr", "10yr", "15yr", "20yr"]


class WindowRetreiver:
    def __init__(self, chromadb_path : str = "./chromadb"):
        self.client = chromadb.PersistentClient(path = chromadb_path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("WindowRetreiver initialized (path=%s)", chromadb_path)
        pass





    def retreive(
            self,
            question : str,
            window : str,
            n_results: int = 5;
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
    
    def retrieve_all_windows(
            self,
            question: str,
            n_results: int = 5,
    ) -> dict[str, list[RetrievedChunk]]:
        return {
            window : self.retreive(question, window, n_results)
            for window in VALID_WINDOWS
        }
    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "(No relevant context was retrieved for this window.)"
        return "\n\n".join(
            f"[{c['date']}]: {c['text']}" for c in chunks
        )
        




class TemporalRetriever:

 


    def filter_by_time(self, query_date, window_days):
        """
        Keeps only documents inside the temporal window
        """

        query_date = datetime.strptime(query_date, "%Y-%m-%d")

        start_date = query_date - timedelta(days=window_days)


        filtered = self.df[
            (pd.to_datetime(self.df["date"]) >= start_date)
            &
            (pd.to_datetime(self.df["date"]) <= query_date)
        ]


        return filtered



    def retrieve(self, query_embedding, query_date, window_days, k=5):
        """
        Retrieves top-k documents inside temporal window
        """


        # Step 1: temporal filtering
        filtered_docs = self.filter_by_time(
            query_date,
            window_days
        )


        if len(filtered_docs) == 0:
            return []


        # Step 2: extract embeddings
        embeddings = np.vstack(
            filtered_docs["embedding"].values
        ).astype("float32")


        # Step 3: temporary FAISS index
        temp_index = faiss.IndexFlatL2(
            embeddings.shape[1]
        )


        temp_index.add(embeddings)



        # Step 4: similarity search
        distances, indices = temp_index.search(
            np.array([query_embedding]).astype("float32"),
            min(k, len(filtered_docs))
        )


        results = []


        for idx in indices[0]:

            row = filtered_docs.iloc[idx]

            results.append({
                "date": row["date"],
                "text": row["text"],
                "distance": float(distances[0][len(results)])
            })


        return results
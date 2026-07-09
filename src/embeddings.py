import chromadb
from sentence_transformers import SentenceTransformer
from datetime import date
from dateutil.relativedelta import relativedelta
from data_process import load_all_documents, filter_by_window

all_documents = load_all_documents

CUTOFF = date(2015, 12, 31)
WINDOWS = {
    "5yr" : CUTOFF - relativedelta(years = 5),
    "10yr" : CUTOFF - relativedelta(years = 10),
    "20yr" : CUTOFF - relativedelta(years= 20),
    "50yr" : CUTOFF - relativedelta(years=50)
}


client = chromadb.PersistentClient(path="./chromadb")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

for window_name, start_date in WINDOWS.items():
    window_docs = filter_by_window(
        all_documents,
        start_date=start_date.isoformat,
        end_date = CUTOFF.isoformat
    )

    print(f"{window_name} : {len(window_docs)} documents "
          f"({start_date} to {CUTOFF}) ")
    
    try:
        client.delete_collection(f"finance_{window_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name = f"finance_{window_name}",
        metadata={"window" : window_name}
    )

    texts = [d["text"] for d in window_docs]
    ids = [f"{window_name}_chunk_{i}" for i in range(len(window_docs))]
    metadatas = [d["metadata"] for d in window_docs]

    batch_size = 150
    for i in range (0, len(texts), batch_size):
        collection.add(
            documents=texts[i:i+batch_size],
            ids=ids[i: i+ batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
    print(f"   {collection.count()} chunks stored in finance_{window_name}\n")

    
test_query = "Federal Reserve interest rate cuts following economic shock"

for window in ["5yr", "10yr", "20yr", "50yr"]:
    collection = client.get_collection(f"finance_{window}")
    results = collection.query(query_texts=[test_query], n_results=2)
    print(f"\n=== {window} TOP RESULT ===")
    print(results["documents"][0][0][:300])
    print(f"Date: {results['metadatas'][0][0]['date']}")
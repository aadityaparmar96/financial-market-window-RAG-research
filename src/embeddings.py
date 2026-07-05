import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path=f"{BASE}/chromadb")
embedder = SentenceTransformer("all=MiniLM-L6-v2")

for window, chunks in all_chunks.items():
    
    try:
        client.delete_collection(f"finance_{window}")
    except:
        pass
    
    collection = client.create_collection(
        name=f"finance_{window}",
        metadata={"window": window}
    )

    texts = [c["text"] for c in chunks]
    ids = [f"{window}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"date": c["date"], "window": c["window"]} for c in chunks]
    
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        collection.add(
            documents=texts[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
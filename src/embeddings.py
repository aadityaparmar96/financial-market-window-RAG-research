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
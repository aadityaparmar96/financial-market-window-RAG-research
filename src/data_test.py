import sys
sys.path.insert(0, "src")
from data_process import load_all_documents
from collections import Counter

docs = load_all_documents()

print(f"Total documents: {len(docs)}\n")

# 1. Which datasets loaded, and how many docs each produced
print("By source:")
for source, count in Counter(d["metadata"]["source"] for d in docs).items():
    print(f"  {source:20s} {count} docs")

# 2. By dataset type
print("\nBy dataset_type:")
print(Counter(d["metadata"]["dataset_type"] for d in docs))

# 3. Date range overall
dates = sorted(d["metadata"]["date"] for d in docs)
print(f"\nOverall date range: {dates[0]} → {dates[-1]}")

# 4. Date range per source (catches silent parsing failures)
print("\nPer-source date range:")
by_source = {}
for d in docs:
    s = d["metadata"]["source"]
    by_source.setdefault(s, []).append(d["metadata"]["date"])
for s, ds in by_source.items():
    ds.sort()
    print(f"  {s:20s} {ds[0]} → {ds[-1]}  ({len(ds)} rows)")

# 5. Spot-check actual text
print("\nSample texts:")
for d in docs[:2] + docs[-2:]:
    print(" -", d["text"])
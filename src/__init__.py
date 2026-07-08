import os
from pathlib import Path
import pandas as pd


# -------------------------------
# FIND PROJECT ROOT
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"


print("Project root:")
print(BASE_DIR)

print("\nLooking for datasets in:")
print(DATA_DIR)


# -------------------------------
# FIND DATASETS
# -------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )


files = list(DATA_DIR.glob("*"))

if len(files) == 0:
    raise FileNotFoundError(
        "Dataset folder exists but is empty"
    )


print("\nDatasets found:")

for f in files:
    print("-", f.name)



# -------------------------------
# LOAD DATASET
# -------------------------------

dataset_file = files[0]

print("\nLoading:")
print(dataset_file)


if dataset_file.suffix == ".csv":

    df = pd.read_csv(dataset_file)


elif dataset_file.suffix == ".json":

    df = pd.read_json(dataset_file)


else:
    raise Exception(
        "Unsupported file type"
    )


print("\nDataset preview:")
print(df.head())


print("\nShape:")
print(df.shape)



# -------------------------------
# BASIC CLEANING
# -------------------------------

df = df.dropna()

df = df.astype(str)


print("\nAfter cleaning:")
print(df.head())


# -------------------------------
# CREATE DOCUMENTS FOR RAG
# -------------------------------

documents = []

for _, row in df.iterrows():

    text = " ".join(
        row.values
    )

    documents.append(text)


print("\nDocuments created:")
print(len(documents))


# Example output
print("\nFirst document:")
print(documents[0])
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

files = [
    "CPIAUCSL.csv",
    "FEDFUNDS.csv",
    "GDPC1.csv",
    "GS10.csv",
    "S&P500.csv",
    "T10Y2YM.csv",
    "TB3MS.csv",
    "UNRATE.csv",
    "USREC.csv"
]

dfs = []

for file in files:
    path = RAW_DIR / file

    df = pd.read_csv(path)

    date_col = df.columns[0]
    value_col = df.columns[1]

    series_name = file.replace(".csv", "")

    df = df.rename(columns={
        date_col: "DATE",
        value_col: series_name
    })

    df["DATE"] = pd.to_datetime(df["DATE"])

    dfs.append(df)

master = dfs[0]

for df in dfs[1:]:
    master = master.merge(
        df,
        on="DATE",
        how="outer"
    )

master = master.sort_values("DATE")

output_path = PROCESSED_DIR / "master_dataset.csv"

master.to_csv(output_path, index=False)

print(f"Saved: {output_path}")
print(master.head())
print(master.shape)
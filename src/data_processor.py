import pandas as pd
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Paths
DATA_PATH = BASE_DIR / "data" / "raw"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "master_dataset.csv"



def load_csv(filename):

    path = DATA_PATH / filename

    print(f"Loading: {filename}")

    df = pd.read_csv(path)

    # Convert date column
    df["DATE"] = pd.to_datetime(df["DATE"])

    return df



def load_all_data():

    files = [
        "CPIAUCSL.csv",
        "FEDFUNDS.csv",
        "GDPC1.csv",
        "GS10.csv",
        "S&P500.csv",
        "Dividend.csv",
        "Earnings.csv",
        "Consumer Price Index.csv",
        "Long Interest Rate.csv",
        "Real Earnings.csv",
        "PE10.csv",
        "T10Y2YM.csv",
        "TB3MS.csv",
        "UNRATE.csv",
        "USREC.csv"
    ]


    merged = None


    for file in files:

        df = load_csv(file)


        if merged is None:

            merged = df

        else:

            merged = pd.merge(
                merged,
                df,
                on="DATE",
                how="inner"
            )


    # Sort chronologically
    merged = merged.sort_values("DATE")


    # Remove remaining missing values
    merged = merged.dropna()


    return merged



def save_data(df):

    # Create processed folder if missing
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(f"\nSaved: {OUTPUT_PATH}")



if __name__ == "__main__":


    df = load_all_data()


    save_data(df)



    print("\nFIRST ROWS:")
    print(df.head())


    print("\nSHAPE:")
    print(df.shape)


    print("\nCOLUMNS:")
    print(df.columns.tolist())


    print("\nMISSING VALUES:")
    print(df.isna().sum())


    print("\nDATE RANGE:")
    print(df["DATE"].min())
    print(df["DATE"].max())
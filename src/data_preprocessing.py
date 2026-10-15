
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "creditcard.csv"
OUTPUT_DIR = ROOT / "data" / "processed"


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Place creditcard.csv inside data/raw/."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)

    required = {"Time", "Amount", "Class"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset does not have the expected columns.")

    if df["Class"].isna().any():
        raise ValueError("The target column contains missing values.")

    # Remove exact duplicate rows to reduce data leakage risk.
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns="Class")
    y = df["Class"].astype("int8")

    # 70% training, 15% validation, 15% testing.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    for name, data in splits.items():
        data.to_frame().to_parquet(OUTPUT_DIR / f"{name}.parquet") \
            if isinstance(data, pd.Series) \
            else data.to_parquet(OUTPUT_DIR / f"{name}.parquet")

    print(f"Dataset prepared successfully: {len(df):,} transactions")
    print(f"Fraud transactions: {int(y.sum()):,}")
    print(f"Processed files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

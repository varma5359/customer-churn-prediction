"""Download the public IBM Telco Customer Churn sample dataset.

Run from the project root:
    python download_data.py

The script writes the validated CSV to data/customers.csv, the path used by
train_model.py. Do not use private or identifiable customer data in this demo.
"""

from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_PATH = DATA_DIR / "customers.csv"

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)


def download_dataset() -> None:
    """Download the CSV, validate its basic schema, and save it locally."""
    print("Downloading the public IBM Telco Customer Churn sample dataset...")

    try:
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            "Dataset download failed. Check your internet connection and try again."
        ) from error

    if not response.content:
        raise ValueError("The dataset source returned an empty response.")

    try:
        data = pd.read_csv(BytesIO(response.content))
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError("The downloaded file is not a readable CSV dataset.") from error

    data.columns = data.columns.astype(str).str.strip()

    if "Churn" not in data.columns:
        raise ValueError(
            "The downloaded CSV does not contain the expected 'Churn' target column."
        )

    if len(data) < 100 or data.empty:
        raise ValueError(
            f"The downloaded dataset has only {len(data)} rows; refusing to save it."
        )

    if data.columns.duplicated().any():
        raise ValueError("The downloaded dataset contains duplicate column names.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)

    print("\nDataset downloaded and validated successfully.")
    print(f"Saved to: {DATA_PATH}")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")
    print(f"Churn target values: {sorted(data['Churn'].dropna().unique().tolist())}")
    print("\nNext step: run 'python train_model.py'.")


if __name__ == "__main__":
    try:
        download_dataset()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Dataset download stopped: {error}") from error

import os
import re
import pandas as pd
from dotenv import load_dotenv
from synnax_lab_sdk.client import SynnaxLabClient


# Load API key from .env file
load_dotenv()


def clean_column_names(df):
    """
    Remove unsupported characters from column names.
    Example: 'GDP Growth (%)' becomes 'GDP_Growth_'
    """
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", col) for col in df.columns]
    return df


def fetch_data():
    api_key = os.getenv("SYNNAX_API_KEY")

    if not api_key:
        raise ValueError("SYNNAX_API_KEY not found. Please check your .env file.")

    print("API key loaded successfully.")

    # Create Synnax client
    synnax_lab_client = SynnaxLabClient(api_key=api_key)

    # Get dataset file paths from Synnax
    files = synnax_lab_client.get_datasets(with_macro_data=True)

    print("Dataset paths received from Synnax:")
    print(files)

    # Read CSV files
    X_train = pd.read_csv(files["x_train_path"])
    X_forward_looking = pd.read_csv(files["x_forward_looking_path"])
    targets_train = pd.read_csv(files["targets_train_path"])
    macro_train = pd.read_csv(files["macro_train_path"])
    macro_forward_looking = pd.read_csv(files["macro_forward_looking_path"])

    # Clean column names
    X_train = clean_column_names(X_train)
    X_forward_looking = clean_column_names(X_forward_looking)
    targets_train = clean_column_names(targets_train)
    macro_train = clean_column_names(macro_train)
    macro_forward_looking = clean_column_names(macro_forward_looking)

    # Save local copies inside data folder
    os.makedirs("data", exist_ok=True)

    X_train.to_csv("data/X_train.csv", index=False)
    X_forward_looking.to_csv("data/X_forward_looking.csv", index=False)
    targets_train.to_csv("data/targets_train.csv", index=False)
    macro_train.to_csv("data/macro_train.csv", index=False)
    macro_forward_looking.to_csv("data/macro_forward_looking.csv", index=False)

    print("\nData saved successfully inside data/ folder.")

    print("\nDataset shapes:")
    print("X_train:", X_train.shape)
    print("targets_train:", targets_train.shape)
    print("X_forward_looking:", X_forward_looking.shape)
    print("macro_train:", macro_train.shape)
    print("macro_forward_looking:", macro_forward_looking.shape)

    print("\nX_train columns:")
    print(X_train.columns.tolist())

    print("\ntargets_train columns:")
    print(targets_train.columns.tolist())


if __name__ == "__main__":
    fetch_data()
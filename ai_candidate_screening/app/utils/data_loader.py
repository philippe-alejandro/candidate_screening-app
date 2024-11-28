import pandas as pd
import json

def load_candidates(csv_path):
    """
    Load candidates from a CSV file, preprocess, and return as a JSON list.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Basic preprocessing (fill missing values, clean text, etc.)
    df = df.fillna("Not Provided")

    # Convert the DataFrame to a JSON-like format
    candidates = df.to_dict(orient="records")
    return candidates

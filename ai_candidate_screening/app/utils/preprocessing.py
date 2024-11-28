import re

def preprocess_text(text: str):
    """
    Preprocess the input text by removing noise and lowercasing.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text
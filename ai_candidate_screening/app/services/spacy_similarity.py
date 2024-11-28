import spacy
import pandas as pd
import os

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

def calculate_similarity(job_description: str, candidates_file = os.path.join(os.path.dirname(__file__), "../data/candidates.csv")):
    """
    Calculate similarity scores between a job description and candidates' details, 
    and return the top 30 candidates with the highest scores.
    
    Args:
        job_description (str): The job description text.
        candidates_file (str): Path to the candidates CSV file.
    
    Returns:
        pd.DataFrame: DataFrame of the top 30 candidates with their similarity scores.
    """
    # Load the candidates data
    df = pd.read_csv(candidates_file)

    # Combine all columns into a single text column
    df["CandidateText"] = df.apply(lambda row: " ".join(row.fillna("").astype(str)), axis=1)

    # Process the job description using spaCy
    job_doc = nlp(job_description)

    # Calculate similarity scores
    df["Score"] = df["CandidateText"].apply(lambda text: job_doc.similarity(nlp(text)) * 100)

    # Sort candidates by similarity scores in descending order
    top_candidates = df.sort_values(by="Score", ascending=False).head(30)

    # Return the top 30 candidates with their scores and names
    return top_candidates[["Name", "Score"]]

if __name__ == "__main__":
    job_description = "Senior software Ruby engineer with PostgreSQL experience"

    top_candidates = calculate_similarity(job_description)
    print("Top 30 Candidates with Similarity Scores:")
    print(top_candidates)

import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

import pandas as pd
import joblib
import xgboost as xgb
from app.services.refine_training_data import calculate_experience_score, assign_education_score, calculate_skills_score

# First access the directory: cd "/Users/philippebrennerroman/Desktop/ZipDev App"
# Command to execute file: python3 ai_candidate_screening/app/services/xgboost_model/predict_model.py

def predict_scores(job_description: str, candidates_file=os.path.join(os.path.dirname(__file__), "../../data/candidates.csv"), model_path=os.path.join(os.path.dirname(__file__), "model.xgb"), preprocessor_path=os.path.join(os.path.dirname(__file__), "preprocessors.pkl")):
    """
    Predict scores for candidates from candidates.csv based on a job description using an XGBoost model and return the top 30 candidates.

    Args:
        job_description (str): The job description text.
        candidates_file (str): Path to the candidates CSV file.
        model_path (str): Path to the trained XGBoost model (.xgb).
        preprocessor_path (str): Path to the preprocessors file (.pkl).

    Returns:
        pd.DataFrame: Candidate ID and Name of top 30 candidates.
    """
    # Load the trained model and preprocessors
    model = joblib.load(model_path)
    preprocessors = joblib.load(preprocessor_path)
    scaler = preprocessors["scaler"]
    encoder = preprocessors["encoder"]

    # Load candidate data from CSV
    candidate_data = pd.read_csv(candidates_file)

    # Ensure no NaN values in relevant columns and convert to string
    candidate_data["Experiences"] = candidate_data["Experiences"].fillna("").astype(str)  # Replace NaN with empty strings and convert to string
    candidate_data["Skills"] = candidate_data["Skills"].fillna("").astype(str)  # Replace NaN with empty strings and convert to string
    candidate_data["Educations"] = candidate_data["Educations"].fillna("").astype(str)  # Replace NaN with empty strings and convert to string

    # Extract relevant columns and assign scores
    candidate_data["Experience_Score"] = candidate_data["Experiences"].apply(calculate_experience_score)
    candidate_data["Skills_Score"] = candidate_data.apply(
        lambda row: calculate_skills_score(row["Skills"], row["Experiences"], job_description), axis=1
    )
    candidate_data["Education_Score"] = candidate_data["Educations"].apply(assign_education_score)

    # Preprocess features
    X = candidate_data[["Experience_Score", "Skills_Score", "Education_Score"]]
    X_scaled = scaler.transform(X)  # Normalize features

    # Create DMatrix for XGBoost prediction
    dmatrix = xgb.DMatrix(X_scaled)

    # Predict scores using the XGBoost model
    predictions = model.predict(dmatrix)

    # Add predictions to the candidate data and scale to 0-100
    candidate_data["Score"] = predictions * 100

    # Sort candidates by score in descending order and get the top 30
    top_candidates = candidate_data.sort_values(by="Score", ascending=False).head(30)

    # Return Candidate ID and Name for top 30
    return top_candidates[["Name", "Score"]].reset_index(drop=True)

if __name__ == "__main__":
    job_description = "Looking for a Python developer with data analysis expertise and experience with machine learning."
    top_candidates = predict_scores(job_description)
    print("Top 30 candidates:")
    print(top_candidates)

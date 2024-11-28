from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import re

def preprocess_training_data(csv_path: str, job_description: str, save_path="app/data/refined_training_data.csv"):
    """
    Preprocess training data to assign numerical scores to categorical columns
    and normalize them.
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Preprocessed columns
    df["Education_Score"] = df["Education"].apply(assign_education_score)
    df["Experience_Score"] = df["Experience"].apply(calculate_experience_score)
    df["Skills_Score"] = df.apply(
        lambda row: calculate_skills_score(row["Skills"], row["Experience"], job_description), axis=1
    )

    # Normalize all scores to a range of 0-1
    scaler = MinMaxScaler()
    df[["Education_Score", "Experience_Score", "Skills_Score", "Score"]] = scaler.fit_transform(
        df[["Education_Score", "Experience_Score", "Skills_Score", "Score"]]
    )

    # Drop original text-based columns
    df.drop(columns=["Candidate ID", "Name", "Job Description", "Education", "Experience", "Skills"], inplace=True)

    # Save the refined data
    df.to_csv(save_path, index=False)
    print(f"Refined training data saved to {save_path}")
    return df

def assign_education_score(education: str) -> float:
    """
    Assign a numerical score to the education column.
    Higher education levels receive higher scores.
    """
    education = education.lower()
    if "phd" in education:
        return 5
    elif "master" in education:
        return 4
    elif "bachelor" in education:
        return 3
    elif "certificate" in education or "diploma" in education:
        return 2
    else:
        return 1

def calculate_experience_score(experience: str) -> float:
    """
    Calculate a score based on the number of years of experience.
    """
    years = re.findall(r"\d{4}", experience)
    if len(years) >= 2:
        # Calculate total years based on the start and end year
        total_years = sum(int(years[i + 1]) - int(years[i]) for i in range(0, len(years) - 1, 2) if int(years[i + 1]) >= int(years[i]))
        return total_years
    return 0

def calculate_skills_score(skills: str, experience: str, job_description: str) -> float:
    """
    Calculate a skills score based on matching skills to the job description.
    If skills are empty, derive a score based on experience and job description keywords.
    """
    job_keywords = set(job_description.lower().split())
    
    # If skills are empty, use experience to derive score
    if pd.isna(skills) or skills.strip() == "":
        experience_keywords = set(experience.lower().split())
        matches = job_keywords.intersection(experience_keywords)
        return len(matches) / len(job_keywords) if job_keywords else 0

    # If skills are not empty, calculate based on skills
    candidate_skills = set(skills.lower().split(","))
    matches = job_keywords.intersection(candidate_skills)
    return len(matches) / len(job_keywords) if job_keywords else 0

# Add the execution statement
if __name__ == "__main__":
    # Input and output file paths
    input_file = "app/data/training_data.csv"
    output_file = "app/data/refined_training_data.csv"
    
    # Job description (can be adjusted based on the requirement)
    job_description = "Looking for a Golang developer with backend experience and scalability expertise."
    
    # Execute the preprocessing function
    preprocess_training_data(input_file, job_description, save_path=output_file)

import os
import pandas as pd
from app.utils.data_loader import load_candidates
from app.utils.preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Coroutine
from scipy.sparse import csr_matrix
import numpy as np
import asyncio


CANDIDATES = load_candidates("app/data/candidates.csv")

JOB_DESCRIPTIONS = [
    "Looking for a Golang developer with backend experience and scalability expertise.",
    "Experienced Ruby on Rails engineer with PostgreSQL skills.",
    "QA Engineer with automation testing and Selenium expertise.",
    "Unity developer proficient in game development and C#.",
    "Lead generator experienced in cold outreach and CRM tools.",
    "Email marketing specialist with experience in Mailchimp and analytics.",
    "Senior front-end developer with React and TypeScript experience.",
    "Backend engineer with expertise in Node.js and MongoDB.",
    "Technical writer familiar with API documentation and tools like Swagger.",
    "IT support specialist with strong customer service skills.",
    "Database administrator experienced in MySQL and performance optimization.",
    "AI/ML engineer with Python and TensorFlow expertise.",
    "Product manager experienced in Agile and Jira.",
    "UI/UX designer with proficiency in Figma and Adobe XD.",
    "Cloud engineer with AWS certification and Kubernetes knowledge.",
    "Cybersecurity analyst experienced in penetration testing and risk assessment.",
    "DevOps engineer proficient in CI/CD pipelines and Docker.",
    "Digital marketer with expertise in SEO and Google Analytics.",
    "Mobile app developer with experience in iOS and Android platforms.",
    "Technical recruiter familiar with full-cycle recruitment and ATS systems."
]

def preprocess_text(text):
    """
    Preprocess the input text by removing noise and lowercasing.
    """
    import re
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def calculate_feature_score(feature, job_keywords, weight=1.0):
    """
    Calculate a feature score based on overlap between candidate features and job keywords.
    """
    if isinstance(feature, list):  # For skills or certifications
        matches = len(set(job_keywords).intersection(set(feature)))
        return (matches / len(job_keywords)) * weight
    return 0  # If the feature is not a list, return 0

async def score_candidates_for_job(job_description: str) -> list[dict]:
    resumes = []
    for candidate in CANDIDATES:
        resume_text = f"{candidate.get('Experiences', '')} {candidate.get('Skills', '')} {candidate.get('Keywords', '')} {candidate.get('Summary', '')}"
        resumes.append(preprocess_text(resume_text))

    job_description = preprocess_text(job_description)
    job_keywords = set(job_description.split())
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_description] + resumes)
    
    if isinstance(tfidf_matrix, csr_matrix):
        tfidf_matrix = tfidf_matrix.toarray()
    
    # Explicit casting to help Pylance
    tfidf_matrix = np.array(tfidf_matrix)

    job_vector = tfidf_matrix[0]
    candidate_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(candidate_vectors, job_vector).flatten()

    scores = []
    for i, candidate in enumerate(CANDIDATES):
        experience_score = min(len(str(candidate.get("Experiences", "")).split()) / 100, 1) * 0.3
        skills_score = calculate_feature_score(str(candidate.get("Skills", "")).split(), job_keywords, weight=0.4)
        education_score = 0.1 if "degree" in str(candidate.get("Educations", "")).lower() else 0
        certification_score = calculate_feature_score([], job_keywords, weight=0.2)  # Adjust if you have certifications data

        final_score = (0.7 * similarity_scores[i] + experience_score + skills_score + education_score + certification_score) * 100
        scores.append({
            "Candidate ID": i + 1,
            "Name": candidate.get("Name", "Unknown"),
            "Job Description": job_description,
            "Experience": candidate.get("Experiences", ""),
            "Skills": candidate.get("Skills", ""),
            "Education": candidate.get("Educations", ""),
            "Score": round(final_score, 2)
        })
    return await asyncio.sleep(0, result=scores)

async def generate_training_data():
    """
    Generate training data for 20 job descriptions and save it to a CSV file.
    """
    csv_path = "app/data/training_data.csv"

    # Check if training data file already exists
    if os.path.exists(csv_path):
        print("Training data already exists. No need to regenerate.")
        return

    # Generate scores for each job description
    training_data = []
    for job_description in JOB_DESCRIPTIONS:
        training_data.extend(await score_candidates_for_job(job_description))

    # Save training data to a CSV file
    df = pd.DataFrame(training_data)
    df.to_csv(csv_path, index=False)
    print(f"Training data saved to {csv_path}.")

# Run the function to generate training data
async def training_data_generation():
    await generate_training_data()

asyncio.run(training_data_generation())

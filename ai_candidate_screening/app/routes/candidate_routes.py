from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.neural_network.predict_model import predict_scores
from app.services.XGboost.predict_model import predict_scores as predict_scores_XGboost
from app.services.spacy_similarity import calculate_similarity

# Kill the current env: rm -rf venv
# Create an env: python3 -m venv venv
# Activate the newly created env: source venv/bin/activate

#Command to launch the FAST API: PYTHONPATH="/Users/philippebrennerroman/Desktop/ZipDev App/ai_candidate_screening" uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
# Check: http://127.0.0.1:5000/docs 

router = APIRouter()

# Define the request body schema
class JobDescriptionRequest(BaseModel):
    jobDescription: str

# Endpoint to predict top candidates through Neural Network
@router.post("/api/predict-candidates")
async def predict_candidates(request: JobDescriptionRequest):
    try:
        # Call the prediction function
        top_candidates = predict_scores(request.jobDescription)
        # Convert DataFrame to a list of names
        return {"topCandidates": top_candidates[["Name", "Score"]].to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Endpoint to predict top candidates through XGboost
@router.post("/api/predict-candidates/XGboost")
async def predict_candidates_XGboost(request: JobDescriptionRequest):
    try:
        # Call the prediction function
        top_candidates = predict_scores_XGboost(request.jobDescription)
        # Convert DataFrame to a list of names
        return {"topCandidates": top_candidates[["Name", "Score"]].to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to predict top candidates through spacy similarity
@router.post("/api/predict-candidates/spacy")
async def predict_candidates_spacy(request: JobDescriptionRequest):
    try:
        # Call the prediction function
        top_candidates = calculate_similarity(request.jobDescription)
        # Convert DataFrame to a list of names
        return {"topCandidates": top_candidates[["Name", "Score"]].to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


candidate_router = router
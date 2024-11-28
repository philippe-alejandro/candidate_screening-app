from fastapi import FastAPI
from app.routes.candidate_routes import candidate_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Candidate Screening")

# Include routes
app.include_router(candidate_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
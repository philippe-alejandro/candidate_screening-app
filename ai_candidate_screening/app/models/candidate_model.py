from pydantic import BaseModel

class Candidate(BaseModel):
    name: str
    score: int
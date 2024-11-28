from app.services.scoring_service import score_candidates_for_job

def test_score_candidates():
    job_description = "Looking for a Golang Developer with 3+ years experience."
    results = score_candidates_for_job(job_description)
    assert isinstance(results, list)
    assert len(results) > 0
    assert all("name" in candidate and "score" in candidate for candidate in results)

import React, { useState } from "react";
import "./App.css";

function App() {
  const [jobDescription, setJobDescription] = useState(""); // State to hold job description input
  const [candidates, setCandidates] = useState([]); // State to hold top candidates data
  const [loading, setLoading] = useState(false); // State to manage loading
  const [selectedModel, setSelectedModel] = useState("Neural Net"); // State to manage selected model

  const handleFetchCandidates = async () => {
    setLoading(true);
    try {
      // const endpoint =
      //   selectedModel === "XGBoost"
      //     ? "http://localhost:5000/api/predict-candidates/XGboost"
      //     : selectedModel === "Spacy Similarity"
      //     ? "http://localhost:5000/api/predict-candidates/spacy"
      //     : "http://localhost:5000/api/predict-candidates";

      const endpoint =
        selectedModel === "XGBoost"
          ? "https://aiCandidateScreening.us-east-2.elasticbeanstalk.com/api/predict-candidates/XGboost"
          : selectedModel === "Spacy Similarity"
          ? "https://aiCandidateScreening.us-east-2.elasticbeanstalk.com/api/predict-candidates/spacy"
          : "https://aiCandidateScreening.us-east-2.elasticbeanstalk.com/api/predict-candidates";

      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ jobDescription }),
      });

      if (response.ok) {
        const data = await response.json();
        setCandidates(data.topCandidates); // Assuming API returns a list of objects with Name and Score
      } else {
        console.error("Error fetching candidates:", response.statusText);
        setCandidates([]);
      }
    } catch (error) {
      console.error("Error:", error);
      setCandidates([]);
    }
    setLoading(false);
  };

  const handleClearCandidates = () => {
    setCandidates([]); // Clear the list of candidates
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Candidate Screening</h1>

        {/* Dropdown to choose model */}
        <div style={{ marginBottom: "10px" }}>
          <label htmlFor="model-selector">Choose Model:</label>
          <select
            id="model-selector"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{
              marginLeft: "10px",
              padding: "5px",
              borderRadius: "5px",
            }}
          >
            <option value="Neural Net">Neural Net</option>
            <option value="XGBoost">XGBoost</option>
            <option value="Spacy Similarity">Spacy Similarity</option>
          </select>
        </div>

        <p>Enter a job description below:</p>
        <textarea
          rows="5"
          cols="50"
          placeholder="Enter job description here..."
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
        />
        <br />
        <button onClick={handleFetchCandidates} disabled={loading}>
          {loading ? "Loading..." : "Get Top Candidates"}
        </button>
        <div style={{ marginTop: "20px" }}>
          {candidates.length > 0 ? (
            <>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  background: "#f0f0f0", // Light gray background
                  borderRadius: "15px", // Rounded container
                  padding: "10px",
                }}
              >
                <h2 style={{ margin: 0 }}>Top Candidates</h2>
                <button
                  style={{
                    background: "#d3d3d3", // Grayish background
                    color: "black",
                    border: "none",
                    borderRadius: "50%", // Circular button
                    cursor: "pointer",
                    padding: "5px 10px",
                    fontSize: "16px",
                  }}
                  onClick={handleClearCandidates}
                >
                  X
                </button>
              </div>
              <ul>
                {candidates.map((candidate, index) => (
                  <li key={index}>
                    {candidate.Name} - Score: {candidate.Score.toFixed(2)}
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p>No candidates to display.</p>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;

# PathSnap — Machine Learning-Based GPS Map Matching

PathSnap is an end-to-end GPS map-matching system that uses machine learning, Viterbi sequence decoding, and graph-based route reconstruction to identify the roads most likely travelled by a vehicle.

## 1. Problem Statement

Raw GPS observations contain positional noise. A recorded location may fall beside a road, between parallel roads, or near an intersection.

Simply selecting the nearest road for each observation can produce inconsistent or disconnected predictions.

PathSnap addresses this problem using:
- Road-candidate generation
- Machine-learning-based candidate scoring
- Sequence-aware Viterbi decoding
- Road-network shortest-path reconstruction
- Interactive route visualization

## 2. System Architecture

    GPS Track + Road Network
                |
                v
    KDTree Candidate Generation
                |
                v
    Seven-Feature Extraction
                |
                v
    Histogram Gradient Boosting
                |
                v
    Candidate Road Probabilities
                |
                v
    Viterbi Sequence Decoding
                |
                v
    Graph-Based Route Reconstruction
                |
                v
    FastAPI + Folium Visualization

## 3. Dataset

The supplied dataset contains:
- 80 training trips
- 10 student-testing trips

Each trip includes four files:

| File | Description |
|------|-------------|
| `.track` | GPS coordinates and timestamps |
| `.nodes` | Road-network nodes |
| `.arcs` | Road segments connecting nodes |
| `.route` | Reference road-segment identifiers |

The original dataset is not committed to this repository.

## 4. Phase 1 — Data Preprocessing and Feature Engineering

### Candidate Generation

For each sampled GPS observation, PathSnap retrieves eight nearby candidate road segments using a KDTree built from segment midpoints.

### Extracted Features

| Feature | Description |
|---------|-------------|
| `distance_m` | Distance from GPS observation to road segment |
| `heading_error` | Difference between GPS movement and road alignment |
| `segment_length_m` | Length of candidate segment |
| `gps_step_m` | Distance moved between consecutive raw GPS observations |
| `speed_mps` | Estimated GPS speed |
| `candidate_rank` | Candidate proximity ranking |
| `projection_fraction` | Projected location along the segment |

### Training Labels

Reference routes are used to construct approximate per-observation training labels.

These are proxy labels, not independently verified ground-truth labels for individual GPS points.

## 5. Phase 2 — Model Training and Optimization

Five classifiers were compared using three-fold GroupKFold cross-validation, grouping observations by trip.

| Model | Top-1 Proxy Accuracy |
|-------|----------------------|
| Logistic Regression | 51.76% |
| Decision Tree | 56.15% |
| Random Forest | 57.55% |
| Extra Trees | 53.05% |
| Histogram Gradient Boosting | 57.73% |

### Hyperparameter Optimization

Histogram Gradient Boosting was selected based on cross-validation performance.

Optuna was used to conduct 40 hyperparameter optimization trials.

| Metric | Result |
|--------|--------|
| Baseline CV proxy accuracy | 57.73% |
| Tuned CV proxy accuracy | 58.49% |
| Improvement | Approximately 0.77 percentage points |

The tuned model is stored at:

    models/tuned_hist_gradient_boosting.joblib

## 6. Phase 3 — Viterbi Decoding

The trained classifier produces probabilities for candidate road segments.

Viterbi decoding selects a sequence of candidates by combining:
- Log-probabilities from the ML model
- Road-network transition distances
- GPS displacement
- Plausibility of travel speed

### Greedy vs. Viterbi

| Method | Weighted Top-1 Proxy Accuracy |
|--------|------------------------------|
| Greedy | 50.89% |
| Viterbi | 51.48% |

Viterbi improves aggregate proxy accuracy slightly, but not necessarily on every trip.

## 7. Route Reconstruction

Selected road segments are connected using a weighted road graph and shortest-path calculations.

The reconstruction considers GPS projection positions and reports:
- Whether the route is graph-connected
- Number of expanded road segments
- Total reconstructed route distance
- Sampled GPS displacement distance
- Detour ratio
- Transition diagnostics

The road graph is treated as undirected because explicit one-way and turn-restriction information is not provided.

## 8. Student-Testing Results

| Trip | Connected | Expanded Arcs | Detour Ratio |
|------|-----------|---------------|--------------|
| 00000000 | Yes | 304 | 1.104 |
| 00000007 | Yes | 222 | 1.052 |
| 00000009 | Yes | 530 | 1.094 |
| 00000040 | Yes | 426 | 1.061 |
| 00000052 | Yes | 137 | 1.098 |
| 00000056 | Yes | 1493 | 1.143 |
| 00000065 | Yes | 298 | 1.229 |
| 00000076 | Yes | 495 | 1.029 |
| 00000077 | Yes | 391 | 1.085 |
| 00000091 | Yes | 387 | 1.226 |

All ten student-testing trips produced graph-connected reconstructions with zero reported gaps.

Connectivity alone does not establish route correctness.

## 9. Repository Structure

    PathSnap_Final/
    ├── api.py
    ├── train.py
    ├── tune.py
    ├── decode.py
    ├── route_reconstruction.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    ├── models/
    │   └── tuned_hist_gradient_boosting.joblib
    ├── results/
    │   ├── model_comparison.csv
    │   └── metadata.json
    ├── tuning_results/
    │   ├── optuna_trials.csv
    │   └── tuning_summary.json
    └── decoding_results/
        ├── predictions.csv
        └── trip_metrics.csv

## 10. Installation

Python 3.10 or newer is recommended.

On macOS or Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

On Windows PowerShell:

    py -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

## 11. Dataset Configuration

Place `Student.zip` in the Downloads directory of the machine running the backend.

Alternatively, configure the dataset location with the `PATHSNAP_DATASET` environment variable.

The model path can be configured using `PATHSNAP_MODEL`.

## 12. Running the Backend

Start the API:

    python -m uvicorn api:app --host 0.0.0.0 --port 8001

Available endpoints:

| Endpoint | Purpose |
|----------|---------|
| `/health` | Backend health and file availability |
| `/trips` | Available student-testing trips |
| `/predict/{trip_id}` | ML predictions and route reconstruction |
| `/map/{trip_id}` | Interactive Folium map |
| `/docs` | FastAPI API documentation |

Example URLs:

- http://localhost:8001/health
- http://localhost:8001/trips
- http://localhost:8001/predict/00000052
- http://localhost:8001/map/00000052
- http://localhost:8001/docs

## 13. Frontend Integration

The backend can be connected to a Streamlit frontend running on another computer.

Both computers must have network connectivity, and the frontend must use the backend computer's reachable IP address and port 8001.

The Windows frontend is distributed separately and is not included in this repository.

## 14. Technologies Used

- Python
- NumPy and Pandas
- SciPy and KDTree
- scikit-learn
- Optuna
- Joblib
- FastAPI and Uvicorn
- Folium
- Streamlit (separate frontend)

## 15. Limitations

- Candidate generation may miss relevant long road segments.
- Proxy labels are approximate.
- Road directionality and turn restrictions are not enforced.
- Some reconstructed paths contain detours.
- Graph connectivity does not guarantee correct map matching.
- The frontend's cross-device integration requires separate verification.

## 16. Conclusion

PathSnap demonstrates an integrated GPS map-matching pipeline combining machine learning, sequence decoding, and graph algorithms.

The tuned Histogram Gradient Boosting model achieved 58.49% cross-validation proxy accuracy. Viterbi decoding achieved 51.48% weighted proxy accuracy on the student-testing partition.

All ten testing trips produced connected reconstructed routes with zero reported graph gaps.

The system provides a working foundation for further improvements in GPS map matching and route reconstruction.

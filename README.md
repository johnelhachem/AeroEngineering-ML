# AeroFusion — ADS-B/ADS-C Trajectory Fusion for Oceanic Gap-Filling
<img width="1193" height="655" alt="image" src="https://github.com/user-attachments/assets/2d6ee28e-9ae1-4136-9d8c-8a6221ea67f1" />


AeroFusion reconstructs aircraft trajectories across long oceanic surveillance blackouts by fusing sparse ADS-C waypoints with ADS-B context before and after the gap. The project targets North Atlantic crossings, where continuous ADS-B coverage is unavailable, and compares a great-circle baseline, a Kalman smoother, and a GRU sequence model trained on OpenSky ADS-B and ADS-C data.

## Results

| Method                | Median Error | Improvement |
|-----------------------|-------------|-------------|
| Great-circle baseline | 131 km      | —           |
| Kalman smoother       | 83 km       | −37%        |
| GRU                   | 64 km       | −51%        |

Evaluated on 240 held-out transatlantic crossings (split by aircraft ICAO24).
Mean oceanic blackout duration: ~239 minutes.

## Data Sources

- OpenSky Network Trino database (`trino.opensky-network.org`)
- Tables: `minio.osky.state_vectors_data4`, `minio.osky.adsc`, `minio.osky.flights_data4`
- Region: Shanwick / North Atlantic (35–70°N, 65°W–10°E)
- Period: July 2023 – August 2025
- Final dataset: 1,704 validated NAT crossing segments

## Project Structure

```text
AeroFusion/
├── README.md
├── .gitignore
├── requirements.txt
├── Dockerfile
│
├── src/
│   └── aero_fusion/
│       ├── __init__.py
│       ├── ingest.py
│       ├── step2_clean.py
│       ├── step3_baseline.py
│       ├── step4_build_ml_dataset.py
│       ├── step5_kalman.py
│       ├── step5_train_gru.py
│       ├── step6_analytics.py
│       ├── step7_serve.py
│       ├── step8_monitoring.py
│       └── utils/
│           ├── __init__.py
│           ├── trino_io.py
│           ├── validation.py
│           └── emissions_calculator.py
│
├── notebooks/
│   ├── 01_ingest_flight_data.ipynb
│   ├── 02_clean_dataset.ipynb
│   ├── 03_baseline_reconstruction.ipynb
│   ├── 04_build_ml_dataset.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_analytics.ipynb
│   ├── 07_api_demo.ipynb
│   └── 08_monitoring.ipynb
│
├── demo_app.py
│
├── artifacts/
│   └── .gitkeep
│
├── scripts/
│   ├── run_pipeline.ps1
│   └── run_pipeline.sh
│
└── docs/
    └── pipeline_overview.md
```

## How to Run

### Prerequisites
- Python 3.12
- Windows (PowerShell) or Linux/Mac

### Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### Run the full pipeline

```powershell
# Windows
.\scripts\run_pipeline.ps1
```

```bash
# Linux/Mac
bash scripts/run_pipeline.sh
```

### Run individual steps

```bash
(run from project root after pip install -e .)
python -m aero_fusion.step2_clean
python -m aero_fusion.step3_baseline
python -m aero_fusion.step4_build_ml_dataset
python -m aero_fusion.step5_kalman
# GRU training runs on Google Colab — upload step5_train_gru.py to Colab
```

### Run the demo

```bash
streamlit run demo_app.py
```

### Run the API

```bash
python -m aero_fusion.step7_serve
# Visit: http://localhost:8000/docs
```

## Docker

```bash
# Pull and run (no setup needed — model weights included)

docker pull johnelhachem/aeroengineering_ml:latest
docker run -p 8501:8501 johnelhachem/aeroengineering_ml:latest

# Then open: http://localhost:8501
```

## Reference

OpenSky Report 2025 — arXiv:2505.06254

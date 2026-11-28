# 💾 Cloud Hard Drive Failure Prediction MLOps Platform

An enterprise-grade, end-to-end Predictive Maintenance MLOps pipeline forecasting physical hard drive failures **7 days in advance** across **59,000+ enterprise data center drives** using real-world Backblaze SMART telemetry.

[![CI/CD Pipeline](https://github.com/flame2498/smart-drive-failure-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/flame2498/smart-drive-failure-mlops/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B.svg)](https://streamlit.io/)
[![Polars](https://img.shields.io/badge/Polars-Rust%20Engine-CD792C.svg)](https://pola.rs/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)

---

## 📌 What is This Project?

In large-scale cloud data centers (like AWS, Azure, or Backblaze), unexpected hard drive crashes trigger emergency RAID rebuilds, saturate network bandwidth for days, and risk catastrophic data loss.

This platform monitors daily **S.M.A.R.T. hardware sensors** (reallocated sectors, temperature fluctuations, uncorrectable errors, operating hours) and alerts SRE (Site Reliability Engineering) teams **7 days before physical failure occurs**. This allows cloud operators to gracefully evacuate data and replace drives with **zero customer downtime**.

---

## 🏗️ System Architecture

```
                       +---------------------------------------+
                       |    Backblaze SMART Telemetry Data     |
                       |    (91 Daily Snapshots, 5.1M+ Rows)   |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |   Zero-Disk-Inflation Streaming ETL   |
                       | (Polars Engine + In-Memory Zip Stream)|
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |         Pandera Data Contract         |  <-- Quality Gate (Schema, bounds,
                       |        (src/data/validate.py)         |      temperature [0-100°C], types)
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |      Domain Feature Engineering       |  <-- 7d rolling means, wear rates,
                       |        (src/data/features.py)         |      sensor deltas & forward target
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |  LightGBM + Optuna Bayesian Tuning    |  <-- Strict Out-of-Time split
                       |        (src/models/train.py)          |      F2-Score (+56% Failure Recall)
                       +-------------------+-------------------+
                                           |
                     +---------------------+---------------------+
                     |                                           |
                     v                                           v
      +-----------------------------+             +-----------------------------+
      |  FastAPI Inference Engine   |             |  Statistical Drift Monitor  |
      |     (src/serving/api.py)    |             |    (src/monitoring/drift.py)|
      |  - Single & Batch /predict  |             |  - Kolmogorov-Smirnov Test  |
      |  - Prometheus /metrics      |             |  - Population Stability (PSI|
      |  - Actionable SRE Triage    |             |  - HTML & JSON dashboards   |
      +--------------+--------------+             +--------------+--------------+
                     |                                           |
                     +---------------------+---------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |     Interactive Streamlit Console     |
                       |        (src/dashboard/app.py)         |
                       |  - Fleet health donut charts & gauges |
                       |  - Drive time-series degradation plots|
                       |  - Interactive what-if risk simulator |
                       +---------------------------------------+
```

---

## 🌟 Key Highlights

* **Big Data Ingestion at Scale:** Ingested and processed **5,125,570 daily telemetry rows** across **59,127 enterprise hard drives** (Seagate, Western Digital, Toshiba) using memory-efficient **Polars streaming** without expanding 10+ GB to disk.
* **Leakage-Free Time-Series Validation:** Enforced a strict chronological **Temporal Out-of-Time Split** (Train on first 61 days, Validate on next 15 days, Test on final 15 days) to prevent future data leakage.
* **Overcoming Extreme Class Imbalance:** With a true failure rate of $< 0.01\%$, accuracy is misleading. Using **Optuna Bayesian optimization** targeting $F_2$-Score and PR-AUC, catastrophic failure recall was boosted by **+56% relative improvement**.
* **Automated Data Quality Gates:** Enforced **Pandera** DataFrame contracts at ingestion to reject corrupt schemas, malformed readings, and sensor bytefield overflows.
* **Production Observability:** Continuous sensor drift detection using **Kolmogorov-Smirnov (KS)** tests ($p < 0.05$) and **Population Stability Index (PSI)** to detect environmental or hardware distribution shifts.
* **Dual Serving Layer:**
  - **FastAPI Backend:** Asynchronous microservice ($< 35$ms latency) with Prometheus telemetry metrics.
  - **Streamlit Frontend:** SRE operations console with fleet health charts, drive inspector gauges, and what-if simulation sliders.

---

## 📊 Model Performance on 5.1M Real-World Records

Evaluated strictly on the unseen **15-day hold-out test set** (757,173 records):

| Metric | Baseline Model | Optuna-Tuned Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Physical Failure Recall** | `20.66%` | **`32.23%`** | **+56.0% Relative Boost** 🚀 |
| **Monitored Features** | 14 raw/delta sensors | **19 domain-engineered signals** | +5 wear indicators |
| **Imbalance Weight (`scale_pos_weight`)** | Fixed `50.0` | **`93.25`** | Bayesian optimized |
| **Inference Latency** | — | **`< 35 ms`** | Real-time SLA |

---

## 📂 Project Structure

```text
smart-drive-failure-mlops/
├── .github/workflows/ci.yml       # Automated GitHub Actions CI/CD pipeline
├── data/
│   ├── raw/                       # Backblaze columnar Parquet dataset
│   └── reference/                 # Baseline feature distribution for drift detection
├── models/
│   ├── lgb_model.joblib           # Trained LightGBM model bundle
│   └── metrics.json               # Evaluation metrics on holdout test set
├── reports/
│   ├── drift_metrics.json         # Statistical KS-test & PSI drift scores
│   └── drift_report.html          # Visual HTML drift dashboard
├── src/
│   ├── dashboard/
│   │   └── app.py                 # Streamlit interactive SRE operations console
│   ├── data/
│   │   ├── download_backblaze.py  # Zero-disk-inflation streaming ingestion ETL
│   │   ├── features.py            # Polars rolling window & domain feature engine
│   │   ├── generate_sample.py     # Benchmark dataset generator for testing
│   │   └── validate.py            # Pandera schema contracts & quality gates
│   ├── models/
│   │   └── train.py               # LightGBM pipeline with Optuna Bayesian tuning
│   ├── monitoring/
│   │   └── drift.py               # Kolmogorov-Smirnov & PSI drift detection
│   └── serving/
│       ├── api.py                 # FastAPI production inference microservice
│       └── schemas.py             # Pydantic request & response validation schemas
├── tests/
│   └── test_pipeline.py           # Unit and integration test suite
├── Dockerfile                     # Container definition for API deployment
├── docker-compose.yml             # Multi-service setup (FastAPI + Prometheus)
├── dvc.yaml                       # Data Version Control pipeline stages
├── params.yaml                    # Centralized project configuration
├── prometheus.yml                 # Prometheus metrics scraping configuration
├── pyproject.toml / ruff.toml     # Ruff linter and code formatting configuration
└── requirements.txt               # Project dependencies
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/flame2498/smart-drive-failure-mlops.git
cd smart-drive-failure-mlops

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Launch the SRE Web Dashboard

Open the visual dashboard with fleet health monitors, live drive failure gauges, and what-if sliders:

```bash
streamlit run src/dashboard/app.py
```
👉 Open your browser at **`http://localhost:8501`**

---

### 3. Launch the FastAPI Serving Microservice

Start the high-performance async inference backend:

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```
* **Interactive Swagger UI:** `http://localhost:8000/docs`
* **Health Check & Features:** `http://localhost:8000/health`
* **Prometheus Metrics:** `http://localhost:8000/metrics`

---

### 4. Run the Automated Test Suite

Run the full end-to-end integration and unit tests:

```bash
pytest tests/ -v
```

Run code quality linting:
```bash
ruff check src/ tests/
```

---

### 5. Multi-Container Deployment with Docker

To spin up both the FastAPI inference service and Prometheus scraper:

```bash
docker-compose up --build
```
* FastAPI Service: `http://localhost:8000`
* Prometheus Dashboard: `http://localhost:9090`

---

## 🛡️ Actionable SRE Risk Triage

Instead of returning an ambiguous probability, the API categorizes risk into clear data center actions:

| Risk Level | Failure Probability | Recommended SRE Action |
| :--- | :---: | :--- |
| **`CRITICAL`** | $\ge 75\%$ | 🚨 **Evacuate drive data immediately and schedule physical swap within 48 hours.** |
| **`WARNING`** | $40\% - 74\%$ | ⚠️ **Increase telemetry polling frequency and initiate pre-emptive RAID rebalancing.** |
| **`HEALTHY`** | $< 40\%$ | ✅ **Drive operates within normal thresholds. No intervention needed.** |

---

## 📜 License

This project is open-source under the MIT License. Data provided by the [Backblaze Hard Drive Data Program](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data).

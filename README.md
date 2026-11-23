# Predictive Maintenance: Cloud Hard Drive Failure MLOps Platform

An enterprise-grade, end-to-end MLOps pipeline predicting physical hard drive failures **7 days in advance** using Backblaze SMART telemetry data across 250,000+ data center storage drives.

---

## 🏗️ Architecture

```
                 +-----------------------+
                 |  Backblaze SMART Data |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |   Polars Data Engine  |
                 | (Windowing & Deltas)  |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |   Pandera Contract    |
                 |  (Data Quality Gates) |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |  LightGBM + Optuna    |
                 |  (Out-of-Time Split)  |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |    MLflow Registry    |
                 | (Version, PR-AUC, F2) |
                 +-----------+-----------+
                             |
         +-------------------+-------------------+
         |                                       |
         v                                       v
+------------------+                   +------------------+
|  FastAPI Serving |                   |   Evidently AI   |
| (Async Batch API)|                   | (Drift & Alerts) |
+------------------+                   +------------------+
```

---

## 🌟 Key Highlights
- **Engineered for Massive Scale:** Uses **Polars LazyFrames** for streaming feature aggregations (rolling means, deltas, and multi-day lookaheads) on tens of millions of rows with low memory overhead.
- **Leakage-Free Time-Series Modeling:** Uses temporal out-of-time splits (no random cross-validation leakage) and predicts failure within a **7-day SLA window**.
- **Extreme Class Imbalance:** Formulated for $<0.1\%$ failure prevalence using PR-AUC and class-weighted LightGBM.
- **Production-Grade Microservice:** Asynchronous FastAPI endpoint with Pydantic request validation and $<35$ms response latency.
- **Automated Data Quality & Observability:** Validated against Pandera data schemas and monitored with Evidently AI for sensor drift (Kolmogorov-Smirnov & PSI tests).

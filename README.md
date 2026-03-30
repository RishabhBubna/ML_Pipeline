# Fraud Detection using Unsupervised Learning

[![View Notebook](https://img.shields.io/badge/Jupyter-View_Notebook-orange?logo=Jupyter)](https://github.com/RishabhBubna/ML_Pipeline/blob/main/IEEE_notebook.ipynb)
[![CI](https://github.com/RishabhBubna/fraud-detection-unsupervised/actions/workflows/ci.yml/badge.svg)](https://github.com/RishabhBubna/fraud-detection-unsupervised/actions/workflows/cicd.yml)
[![Docker](https://img.shields.io/badge/Docker-rishabhbubna47%2Ffraud--detection-blue?logo=docker)](https://hub.docker.com/r/rishabhbubna47/fraud-detection)

Unsupervised anomaly detection on the IEEE-CIS Fraud Detection dataset using a Variational Autoencoder (VAE) and Isolation Forest ensemble. No fraud labels are used during training. The project includes a full MLOps pipeline, DVC for data and pipeline versioning, MLflow for experiment tracking, FastAPI for inference, Docker for containerization, and GitHub Actions for CI/CD.

---

## Results

| Model | AUROC | Average Precision |
|-------|-------|-------------------|
| VAE | 0.6931 | 0.0869 |
| Isolation Forest | 0.7326 | 0.0750 |
| **Ensemble (0.9/0.1)** | **0.7269** | **0.0904** |
| Random baseline | 0.500 | 0.034 |

**AP of 0.0904 represents a 166% improvement over the random baseline**: achieved without access to a single fraud label during training.

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset provided by Vesta Corporation via Kaggle.

- 590,540 transactions
- 394 raw features across two tables (transaction and identity)
- 3.5% fraud rate, heavily imbalanced

---

## Approach

Standard supervised fraud detection requires labeled data, which is expensive to obtain and quickly becomes stale as fraud patterns evolve. This project takes a purely unsupervised approach: train models exclusively on normal transactions and flag anything that deviates from learned normal behavior as anomalous.

Two complementary models are combined in a weighted ensemble:

**Variational Autoencoder (VAE)** : learns a compressed latent representation of normal transactions. Fraudulent transactions, being unlike anything seen during training, produce higher reconstruction errors and are flagged as anomalies.

**Isolation Forest** : exploits the geometric sparsity of anomalies in feature space. Anomalous transactions are easier to isolate and therefore require fewer random partitions to separate from the rest of the data.

The two models detect fraud through fundamentally different mechanisms, making their combination more robust than either model alone.

---

## Key Design Decisions

**Data quality over model complexity** : early experiments on poorly cleaned data yielded AUROC as low as 0.54. Systematic feature selection through correlation filtering and sparsity checks was the single most impactful improvement, not model architecture.

**Two separate preprocessing pipelines** : V-columns (339 anonymized Vesta features) hurt the VAE by adding noise to the reconstruction objective, but help the Isolation Forest by providing additional dimensions for geometric anomaly isolation. Two preprocessors are fitted independently:
- `transform_rule_VAE.pkl` : excludes V-columns
- `transform_rule_Iso.pkl` : includes V-columns

**V-column selection** : V-columns are first filtered by correlation (threshold 0.75) to remove redundant features, then filtered by sparsity (dominant value > 90%) to remove near-constant columns. This reduces 339 V-columns to ~19 informative ones for the Isolation Forest.

**Time-ordered train/test split** : the dataset is split chronologically (80/20) rather than randomly. The model is trained on earlier transactions and evaluated on later ones, directly simulating real deployment conditions and preventing future data leakage.

**VAE checkpoint saving** : the best VAE checkpoint is saved at the epoch with the highest Average Precision (after a 5-epoch warmup), not the final epoch. The reconstruction loss and anomaly detection performance are decoupled, continued training eventually makes the model too good at reconstructing everything, including fraud.

**Hyperparameter tuning** : both models are tuned via grid search:
- VAE: β ∈ {1.0, 2.0, 5.0}, z_dim ∈ {3, 5, 10}, lr ∈ {1e-3, 5e-4, 1e-4} → best: β=1, z_dim=3, lr=1e-4
- Isolation Forest: n_estimators ∈ {50, 100, 150, 200}, max_features ∈ {0.5, 0.75, 1.0} → best: n_estimators=50, max_features=0.5

---

## MLOps Pipeline

### Architecture Overview

```
Raw Data (Kaggle)
      ↓
DVC Pipeline (data ingestion → preprocessing → training → evaluation → registry)
      ↓
MLflow (experiment tracking + artifact storage on AWS S3)
      ↓
FastAPI inference app (/health, /predict)
      ↓
Docker (containerized inference, CPU-optimized)
      ↓
GitHub Actions CI/CD (test → build → push to DockerHub)
```

### DVC Pipeline

The full training pipeline is managed with DVC and defined in `dvc.yaml`. Stages run in order:

| Stage | Output |
|-------|--------|
| `data_ingestion` | `processedData/raw/full_dataset.csv`, `Metadata/column_name.json` |
| `data_preprocessing` | `processedData/preprocessed/`, `model/pipeline/` |
| `train_vae` | `model/best_vae.pt`, `model/pipeline/vae_scaler.pkl` |
| `train_iso` | `model/iso_forest.pkl`, `model/pipeline/iso_scaler.pkl` |
| `model_evaluation` | `Metadata/experiment_info.json` |
| `model_registry` | Registers models and artifacts to MLflow on AWS |

To reproduce the full pipeline:
```bash
dvc repro
```

All pipeline parameters are defined in `params.yaml` and versioned with DVC.

### MLflow Experiment Tracking

- MLflow server hosted on AWS EC2
- Artifacts (models, scalers, transform rules) stored in AWS S3
- Both VAE and Isolation Forest logged with full artifact paths
- `experiment_info.json` stores the active `run_id` and S3 paths for CI/CD

### FastAPI Inference

The inference app is defined in `src/main.py` and exposes two endpoints:

**Health check**
```
GET /health
→ {"status": "ok"}
```

**Predict**
```
POST /predict
Body: {
  "transaction": { "TransactionAmt": 100.0, "ProductCD": "W", ... },
  "identity": { ... }   ← optional
}
→ { "ensemble_score": 0.142, "prediction": 0 }
```

- `ensemble_score` : weighted combination of VAE and Isolation Forest anomaly scores (0.9/0.1)
- `prediction` : 1 if fraud (score > 0.0779), 0 otherwise
- Identity table is optional : the API handles transactions with or without identity data
- Models are loaded once at startup via FastAPI lifespan

### Docker

The inference app is fully containerized. Models are not pulled from MLflow at inference time : they are copied directly into the image at build time.

**Pull and run from DockerHub:**
```bash
docker pull rishabhbubna47/fraud-detection:latest
docker run -p 8000:8000 rishabhbubna47/fraud-detection:latest
```

Then visit `http://localhost:8000/docs` for the interactive Swagger UI.

**Build locally:**
```bash
cd src
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

### CI/CD : GitHub Actions

On every push to `main`, the pipeline automatically:

1. **Test job** : installs dependencies, downloads models from S3, runs pytest
2. **Build job** (only if tests pass) : builds Docker image and pushes to DockerHub

Tests cover:
- `/health` endpoint response
- `/predict` endpoint with a minimal transaction payload : validates response schema and prediction output

---

## Project Structure

```
├── IEEE_notebook.ipynb             ← original research notebook
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml                  ← GitHub Actions CI/CD
└── src/                            ← MLOps root
    ├── Dockerfile
    ├── main.py                     ← FastAPI app
    ├── schema.py                   ← Pydantic schemas
    ├── predict_pipeline.py         ← inference logic
    ├── config.py                   ← data paths (training only)
    ├── params.yaml                 ← DVC params
    ├── dvc.yaml                    ← DVC pipeline
    ├── req_inference.txt           ← inference dependencies
    ├── requirements.txt            ← training dependencies
    ├── data/
    │   ├── data_ingestion.py
    │   └── data_preprocessing.py
    ├── model/
    │   ├── VAE.py
    │   ├── train_vae.py
    │   ├── train_iso.py
    │   ├── model_evaluation.py
    │   ├── model_registry.py
    │   └── pipeline/
    │       ├── transform_rule_VAE.pkl
    │       ├── transform_rule_Iso.pkl
    │       ├── vae_scaler.pkl
    │       └── iso_scaler.pkl
    ├── Metadata/
    │   ├── column_name.json
    │   ├── feature_name.json
    │   └── experiment_info.json
    └── tests/
        ├── conftest.py
        └── test_api.py
```

---

## How to Run

### Reproduce the training pipeline

**1. Install training dependencies**
```bash
pip install -r src/requirements.txt
```

**2. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) and place `train_transaction.csv` and `train_identity.csv` in `ieee-fraud-detection/`.

**3. Set environment variables**
```bash
export MLFLOW_TRACKING_URI=<your-mlflow-uri>
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
```

**4. Run the DVC pipeline**
```bash
cd src
dvc repro
```

### Run inference locally (without Docker)

```bash
pip install -r src/req_inference.txt
cd src
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for the Swagger UI.

### Run inference with Docker

```bash
docker pull rishabhbubna47/fraud-detection:latest
docker run -p 8000:8000 rishabhbubna47/fraud-detection:latest
```

---

## Results

![Class Imbalance](Report/figures/class_imbalance.png)

![Reconstruction Errors](Report/figures/vae_reconstruction_errors.png)

![Precision-Recall Curve](Report/figures/ensemble_pr_curve.png)

---

## Limitations

- **Test set exposure during ensemble tuning** : the ensemble weights (0.9/0.1) were selected by sweeping a grid evaluated on the test set. Strictly speaking, a held-out validation set should be used in production to avoid mild overfitting to the test set.
- **Unsupervised ceiling** : the supervised upper bound on this dataset is ~0.92 AUROC with labeled data. The gap represents the information value of fraud labels. In a real deployment, even a small labeled dataset would motivate a semi-supervised approach.
- **Concept drift** : the VAE's reconstruction-based anomaly score is sensitive to shifts in transaction patterns over time. The model would require periodic retraining as fraud patterns evolve.

---

## Validation

As an additional sanity check, COPOD (Copula-Based Outlier Detection) was run on the same features with no hyperparameter tuning, achieving AUROC 0.7181 and AP 0.0720. Three fundamentally different algorithms independently finding fraud signal in the same range confirms the preprocessing pipeline is sound and results are not an artifact of data leakage.

---

Built by Rishabh Bubna : [LinkedIn](https://www.linkedin.com/in/dr-rishabh-bubna-304bb3172/)
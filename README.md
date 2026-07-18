# FL-IDS: Robust Federated Learning Intrusion Detection System

> A research-grade capstone project implementing a novel Byzantine-robust defense pipeline for FL-based Intrusion Detection on IoT networks using **CSE-CIC-IDS2018**.

---

## Overview

This system trains a distributed intrusion detection model across 50 simulated IoT edge clients using **Federated Learning**, without ever sharing raw traffic data. The central server defends against Byzantine attackers (label-flipping and data poisoning) using a novel server-side defense pipeline built with [Flower](https://flower.ai/) and **PyTorch**.

---

## Core Research Contributions

| Gap | Problem | Solution |
|-----|---------|----------|
| **Gap 1** | Non-IID diversity makes benign clients look malicious | **Final-layer Cosine Similarity + MAD** — only the classification head is compared |
| **Gap 2** | Standard robust filters are O(K²·d) — too slow for IoT | **Capped Simplex Projection** — O(K log K), forces attacker weights to exactly 0 |
| **Gap 3** | Hard accept/reject permanently bans temporarily noisy clients | **EMA Trust Scores + Temperature Softmax** — momentum reputation across rounds |

### Novel Contribution: MSFT (Multi-Stage Final-Layer Triage)

The key novelty extends Variant A with a **two-stage scoring pipeline**:
1. MAD scores all clients with cosine similarity
2. Only the **suspicious subset** (low MAD) undergoes SVD spectral filtering — not all clients
3. Refined scores are merged back → EMA → capped simplex

This is architecturally distinct from SSFG (which applies SVD to everyone) and is cheaper and more precise.

---

## Dataset

**CSE-CIC-IDS2018** — Kaggle: `solarmainframe/ids-intrusion-csv`

Auto-downloaded via `kagglehub` on first run. No manual setup needed.

| Property | Value |
|----------|-------|
| Raw rows | 16,233,002 |
| Features after preprocessing | 44 |
| Classes | 15 (Benign + 14 attack types) |
| Train / Test split | 80% / 20% |
| Client partitioning | Non-IID Dirichlet(α=0.5), 50 clients |

**Attack target:** DDOS attack-HOIC (class 4) → flipped to Benign (class 0)

---

## Project Structure

```
FL IDS/
├── run_all_experiments.py        # Full experiment matrix (all strategies × attack ratios)
├── requirements.txt
├── src/
│   ├── configs/
│   │   ├── config.yaml           # All hyperparameters
│   │   ├── config.py             # Loads config.yaml → CONFIG dict
│   │   └── paths.py              # Centralized path constants
│   ├── pipelines/
│   │   ├── data_pipeline.py              # Download → preprocess → partition
│   │   ├── centralized_training_pipeline.py  # MLP baseline training
│   │   ├── training_pipeline.py          # FL experiment loop
│   │   ├── attack_pipeline.py            # Attack sweep manager
│   │   └── evaluation_pipeline.py        # Plots & comparison table
│   └── components/
│       ├── data/
│       │   ├── data_loader.py            # kagglehub IDS2018 loader
│       │   ├── data_preprocessor.py      # Clean, impute, filter, encode, scale
│       │   ├── data_partitioner.py       # Dirichlet Non-IID splitting
│       │   └── torch_dataset.py          # DataLoader factory
│       ├── model/
│       │   └── model.py                  # MLPClassifier [256,128,64]
│       ├── client/
│       │   ├── client.py                 # FLIDSClient (Flower NumPyClient)
│       │   └── attacker.py               # flip_labels, inject_backdoor, scale_gradient
│       ├── server/
│       │   ├── aggregator.py             # Variant A — AL-CMT (Cosine+MAD+EMA+Simplex)
│       │   ├── ssfg_aggregator.py        # Variant C — SVD spectral filter on all clients
│       │   ├── triage_aggregator.py      # ★ NOVELTY — MSFT (SVD on suspicious subset only)
│       │   ├── ablation_aggregators.py   # Ablations: FullModelCosine, FinalLayerNoSimplex
│       │   ├── baselines.py              # FedAvg, TrimmedMean, Krum
│       │   ├── ae_scorer.py              # Variant B — AE reconstruction scorer
│       │   └── server.py                 # get_initial_parameters, server_evaluate_fn
│       └── evaluation/
│           └── evaluator.py              # compute_metrics, log_round_results, log_trust_scores
├── notebooks/
│   ├── 05_preprocessing_teacher_visualization.ipynb
│   ├── 06_baseline_model_performance.ipynb
│   └── 08_ids2018_eda.ipynb      # IDS2018 EDA — class dist, feature funnel, partition heatmap
├── artifacts/
│   ├── preprocessed/             # label_encoder.pkl, feature_cols.pkl, scaler.pkl, test_set.npz
│   ├── data/                     # client_0000.npz … client_0049.npz
│   ├── models/                   # baseline_mlp.pth, fl_global_model.pth
│   ├── results/                  # round_results_*.csv, trust_scores_*.csv
│   └── plots/                    # EDA + evaluation figures
└── tests/
    ├── test_aggregator.py
    ├── test_client.py
    ├── test_model.py
    └── test_partitioner.py
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Run the data pipeline (auto-downloads dataset)

```bash
python -m src.pipelines.data_pipeline
```

This downloads ~1.6 GB from Kaggle via `kagglehub`, preprocesses 16M rows, and creates all 50 client shards. Takes ~30 min. Subsequent runs use the local cache.

### 3. Train the centralized baseline

```bash
python -m src.pipelines.centralized_training_pipeline
```

Trains the MLP on IDS2018, saves the best checkpoint to `artifacts/models/baseline_mlp.pth`.

### 4. Run a single FL experiment

```bash
# Clean baseline (no attackers)
python -m src.pipelines.training_pipeline

# Specific strategy
# Edit config.yaml: attacker_ratio: 0.30
# Then run with strategy name:
python -c "from src.pipelines.training_pipeline import run_experiment; run_experiment(strategy_name='triage')"
```

### 5. Run full experiment matrix

```bash
python run_all_experiments.py
```

Runs all strategies across all attacker ratios. Takes several hours.

### 6. View results

Open `notebooks/08_ids2018_eda.ipynb` for EDA and `notebooks/06_baseline_model_performance.ipynb` for training analysis.

---

## Strategies

| Strategy | Name | Type |
|----------|------|------|
| `fedavg` | FedAvg | Baseline (unprotected) |
| `trimmed_mean` | Federated Trimmed Mean | Classical robust |
| `krum` | Krum | Classical robust |
| `robust` | AL-CMT (Variant A) | Novel |
| `ssfg` | SSFG (Variant C) | Novel |
| `triage` | **MSFT (Novelty)** | **Novel — primary contribution** |
| `full_model_cosine` | Full Model Cosine | Ablation |
| `final_no_simplex` | Final Layer No Simplex | Ablation |

---

## Experiment Design

| Phase | Rounds | Attackers | Purpose |
|-------|--------|-----------|---------|
| Phase 1 | 1–10 | 0% | Clean baseline convergence |
| Phase 2 | 11–50 | 10% / 30% / 50% | Byzantine injection |
| Phase 3 | Post-run | — | Evaluation & comparison plots |

**Metrics:** Macro F1, Attack Success Rate (ASR), False Positive Rate (FPR), per-client trust scores.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | PyTorch MLP [256, 128, 64] |
| FL Framework | Flower (flwr) |
| Dataset | CSE-CIC-IDS2018 (kagglehub) |
| Defense Math | NumPy, SciPy |
| Data Processing | Pandas, scikit-learn |
| Plotting | Matplotlib, Seaborn |
| Testing | pytest |

---

## Out of Scope

- ❌ Blockchain / Homomorphic Encryption — too heavy for IoT
- ❌ CNN / Transformer / LLM on clients — MLP only
- ❌ Server-side raw data access — server sees **only** weight arrays

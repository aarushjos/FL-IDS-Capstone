# FL-IDS: Robust Federated Learning Intrusion Detection System

> A research-grade capstone project implementing a novel Byzantine-robust defense pipeline for FL-based Intrusion Detection on IoT networks using **CSE-CIC-IDS2018**.
> Last updated: **2026-07-25** — MSFT + SOTA upgrade complete. 3 new baselines, 4 new attack types, Median L2-Norm Clipping, SVD score hybrid. 53/53 new tests passing.

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

The key novelty. Pipeline per aggregation round:

1. **Median L2-Norm Clipping** — neutralises Constrain-and-Scale amplification attacks before any scoring (FLAME, USENIX 2022)
2. **Stage 1** — Cosine + MAD scores all clients
3. **Stage 2 (SVD Hybrid)** — only the suspicious subset (low MAD score) undergoes SVD spectral filtering. Score = `0.6 × re-MAD(filtered) + 0.4 × SVD projection score`. Projection score measures alignment with top adversarial singular vector (extends DnC, NDSS 2021).
4. **Stage 3** — Merged scores → EMA reputation → Temperature-scaled Softmax → Capped Simplex weights
5. **Stage 4** — Weighted aggregation across all model layers

**Why MSFT beats SOTA:**

| Competitor | Their limitation | MSFT advantage |
|-----------|-----------------|----------------|
| HRA 2026 | Static T_low/T_high thresholds, brittle on distribution shift | MAD Z-score is self-normalising, no tuning |
| FLAME 2022 | DP noise degrades MA by 1–3% | No DP noise — median clipping + triage instead |
| DnC NDSS 2021 | SVD on all clients, no soft weighting | SVD on suspicious subset only, cheaper + richer |
| Krum / TrimmedMean | Stateless — adaptive attackers exploit this | EMA temporal memory catches trust-building attackers |
| WeiDetect 2025 | Needs labeled server data | Fully data-free server scoring |

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

**Attack target:** DDOS attack-HOIC (class 4) flipped to Benign (class 0)

---

## Project Structure

```
FL IDS/
├── run_all_experiments.py        # Full experiment matrix (all strategies x attack ratios)
├── requirements.txt
├── src/
│   ├── configs/
│   │   ├── config.yaml           # All hyperparameters
│   │   ├── config.py             # Loads config.yaml -> CONFIG dict
│   │   └── paths.py              # Centralized path constants
│   ├── pipelines/
│   │   ├── data_pipeline.py              # Download -> preprocess -> partition
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
│       │   └── attacker.py               # All attack implementations
│       ├── server/
│       │   ├── aggregator.py             # Variant A AL-CMT + clip_to_median_norm
│       │   ├── ssfg_aggregator.py        # Variant C SVD filter on all clients
│       │   ├── triage_aggregator.py      # * NOVELTY -- MSFT (SVD hybrid, suspicious subset)
│       │   ├── hra_aggregator.py         # NEW -- HRA Baseline (NIDS-specific SOTA 2026)
│       │   ├── ablation_aggregators.py   # Ablations: FullModelCosine, FinalLayerNoSimplex
│       │   ├── baselines.py              # FedAvg, TrimmedMean, Krum, GeoMed, LCKrum
│       │   ├── ae_scorer.py              # Variant B AE reconstruction scorer
│       │   └── server.py                 # get_initial_parameters, server_evaluate_fn
│       └── evaluation/
│           └── evaluator.py              # compute_metrics, log_round_results, log_trust_scores
├── notebooks/
│   ├── 05_preprocessing_teacher_visualization.ipynb
│   ├── 06_baseline_model_performance.ipynb
│   └── 08_ids2018_eda.ipynb
├── artifacts/
│   ├── preprocessed/             # label_encoder.pkl, feature_cols.pkl, scaler.pkl, test_set.npz
│   ├── data/                     # client_0000.npz ... client_0049.npz
│   ├── models/                   # baseline_mlp.pth, fl_global_model.pth
│   ├── results/                  # round_results_*.csv, trust_scores_*.csv
│   └── plots/                    # EDA + evaluation figures
└── tests/
    ├── test_aggregator.py
    ├── test_client.py
    ├── test_model.py
    ├── test_partitioner.py
    └── test_new_implementations.py   # 53 tests -- all SOTA-upgrade components
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

Downloads ~1.6 GB from Kaggle via `kagglehub`, preprocesses 16M rows, creates 50 client shards. Takes ~30 min on first run. Subsequent runs use local cache.

### 3. Train the centralized baseline

```bash
python -m src.pipelines.centralized_training_pipeline
```

Trains the MLP on IDS2018, saves best checkpoint to `artifacts/models/baseline_mlp.pth`.
Centralized Macro F1 = **0.7570** (target ceiling for FL to approach).

### 4. Run a single FL experiment

```bash
# Clean baseline (no attackers)
python -m src.pipelines.training_pipeline

# Specific strategy via Python
python -c "from src.pipelines.training_pipeline import run_experiment; run_experiment(strategy_name='triage')"

# All available strategy names:
# fedavg | trimmed_mean | krum | geomed | hra | layerwise_cosine_krum
# robust | ssfg | triage | full_model_cosine | final_no_simplex
```

### 5. Run the full experiment matrix

```bash
python run_all_experiments.py
```

Runs all strategies across all attacker ratios and attack types. Takes several hours on CPU.

### 6. Run the test suite

```bash
venv\Scripts\python.exe -m pytest tests\ -v
```

---

## Strategies

| Strategy key | Name | Type | Source |
|-------------|------|------|--------|
| `fedavg` | FedAvg | Baseline | McMahan et al. 2017 |
| `trimmed_mean` | Federated Trimmed Mean | Classical robust | Yin et al. 2018 |
| `krum` | Krum | Classical robust | Blanchard et al. 2017 |
| `geomed` | Geometric Median | **NEW** robust baseline | Pillutla et al. 2022 |
| `layerwise_cosine_krum` | Layerwise Cosine Krum | **NEW** 2025 baseline | KBS 2025 |
| `hra` | Hybrid Reputation Aggregation | **NEW** 2026 SOTA baseline | HRA 2026 |
| `robust` | AL-CMT (Variant A) | Novel |  |
| `ssfg` | SSFG (Variant C) | Novel |  |
| `triage` | **MSFT (Primary Novelty)** | **Novel** |  |
| `full_model_cosine` | Full Model Cosine | Ablation |  |
| `final_no_simplex` | Final Layer No Simplex | Ablation |  |

---

## Attack Types

| `attack_type` | Description | Source |
|--------------|-------------|--------|
| `label_flip` | Targeted semantic attack: class 4 -> class 0 | Classical |
| `backdoor` | Injects trigger pattern mislabeled as Benign | Classical |
| `both` | label_flip + backdoor combined | Classical |
| `sign_flip` | Reverses and scales gradient update delta | Classical |
| `min_max` | **NEW** Max damage within epsilon of nearest benign client | NDSS 2021 |
| `min_sum` | **NEW** Minimise cosine similarity to all clients | NDSS 2021 |
| `lie` | **NEW** mean + z*std per param — evades variance-based scoring | FedLAW ICLR 2026 |
| `trust_then_strike` | **NEW** Act benign for N rounds then activate min_max/lie | HRA 2026 |

---

## Defense Components

### clip_to_median_norm (NEW — FLAME 2022)
Applied at the very start of `aggregate_fit()` in all three novel strategies (`robust`, `ssfg`, `triage`). Clips every client's full parameter vector to the cohort median L2-norm. Proven to reduce Backdoor Accuracy from 100% to ~0% on IoT-Traffic data by neutralising Constrain-and-Scale attacks.

### MSFT Stage 2 — SVD Projection Score Hybrid (NEW — extends DnC NDSS 2021)
On the suspicious subset, blends:
- `60%` re-MAD score from SVD-filtered subspace
- `40%` SVD projection anomaly score = `1 - |U[:,0]| / max(|U[:,0]|)`

The projection score measures how much of each suspicious client's update is aligned with the top adversarial singular vector. This is a richer signal than either score alone.

### HRA Baseline (NEW — HRA 2026)
Implemented in `hra_aggregator.py`. Full pipeline: GeoMed distance scoring → static piecewise phi weights → EMA reputation → weighted average. Achieves 98.66% accuracy on 5G NIDS data. Configured via `defense.hra_t_low` and `defense.hra_t_high`.

---

## Experiment Design

| Phase | Rounds | Attackers | Purpose |
|-------|--------|-----------|---------|
| Phase 1 | 1–10 | 0% | Clean baseline convergence |
| Phase 2 | 11–50 | 10% / 30% / 50% | Byzantine injection |
| Phase 3 | Post-run | — | Evaluation & comparison plots |

**Metrics:** Macro F1, Attack Success Rate (ASR), False Positive Rate (FPR), per-client trust scores.

**Expected results ladder:**
```
FedAvg < TrimmedMean < Krum < GeoMed < LayerwiseCosineKrum < HRA < Our MSFT
```

---

## Key Config Parameters (`src/configs/config.yaml`)

```yaml
attack:
  attack_type: "label_flip"    # or: backdoor | both | sign_flip | min_max | min_sum | lie | trust_then_strike
  min_max_epsilon: 0.5         # Min-Max bound
  lie_z_clip: 2.0              # LIE attack z-factor
  trust_rounds: 5              # Mimicry: rounds to act benign
  strike_attack_type: "min_max"

defense:
  mad_threshold: -3.0
  triage_soft_threshold: -2.0  # MSFT Stage 2 gate
  svd_keep_ratio: 0.9
  ema_momentum: 0.9
  hra_t_low: 0.3               # HRA threshold (below = penalised)
  hra_t_high: 0.7              # HRA threshold (above = trusted)
```

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
| Testing | pytest (53 new tests, all passing) |

---

## Test Coverage

```
tests/test_new_implementations.py — 53 tests, all passing
  TestClipToMedianNorm     (7 tests) — FLAME 2022 median clipping
  TestGeometricMedian      (6 tests) — Weiszfeld algorithm
  TestGeoMedianBaseline    (5 tests) — GeoMed aggregation
  TestLayerwiseCosineKrum  (5 tests) — KBS 2025 baseline
  TestHRABaseline          (6 tests) — HRA 2026 SOTA baseline
  TestMinMaxAttack         (5 tests) — NDSS 2021 attack
  TestMinSumAttack         (5 tests) — NDSS 2021 attack
  TestLieAttack            (6 tests) — FedLAW 2026 attack
  TestSVDHybrid            (3 tests) — MSFT Stage 2 formula
  TestIntegrationPipeline  (4 tests) — End-to-end pipeline
```

---

## Out of Scope

- Blockchain / Homomorphic Encryption — too heavy for IoT
- CNN / Transformer / LLM on clients — MLP only
- Server-side raw data access — server sees **only** weight arrays
- Client-side defense logic — ALL defense math is server-only

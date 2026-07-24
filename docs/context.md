# FL-IDS Capstone — Master Context File

> **Purpose:** Give a new AI (or future session) full project context in one file.
> Last updated: 2026-07-25 — MSFT + SOTA upgrade complete. Added 3 new baselines (GeoMed, HRA, LayerwiseCosineKrum), 3 new attacks (Min-Max, Min-Sum, LIE), Adaptive Mimicry attack, Median L2-Norm Clipping defense, SVD projection score hybrid in MSFT Stage 2. 53/53 tests passing. context.md and README updated.

---

## 1. Project Summary

**FL-IDS** is a Federated Learning–based Intrusion Detection System for IoT edge gateways.
Clients train locally on **CSE-CIC-IDS2018** network traffic data and send only model weights to a central server — raw data never leaves the device.
The server defends against Byzantine (label-flipping / data-poisoning) attacks using a **modular 3-part math pipeline** before aggregating the global model.

**Key research design decision:** The anomaly scoring component (Step 2) is a **swappable module**. Three implementations will be built and compared:
- **Variant A — AL-CMT:** Layer-Wise Cosine Similarity + MAD
- **Variant B — CS-ARF:** Server-Side Autoencoder Reconstruction Error
- **Variant C — SSFG:** Truncated SVD Subspace Filtering

Steps 3 and 4 (EMA Trust Scoring + Capped Simplex) are shared by Variants A and B. Variant C is an independent matrix-filtering paradigm. This allows a direct comparison within the same framework — a stronger research contribution.

**Stack:**
- ML: `PyTorch` (MLP classifier)
- FL orchestration: `Flower (flwr)`
- Dataset: **CSE-CIC-IDS2018** (~80 raw features → 44 after preprocessing, tabular, **15 classes**: Benign + 14 attacks)
- Data distribution: Non-IID via Dirichlet(α=0.5) to simulate real IoT heterogeneity
- Python package structure under `src/`

---

## 2. Directory Structure

```
FL IDS/
├── app.py                          # Entry point
├── requirements.txt
├── setup.py
├── pytest.ini
├── run_all_experiments.py          # Legacy FL experiment loop
├── run_experiments_queue.ps1       # ✅ PowerShell batch scripts for running bulk simulations (along with run_label_flip_*, run_robust_only_*)
├── docs/
│   ├── context.md                  ← THIS FILE
│   ├── ProjectOverview.md
│   ├── baseline_model_analysis.md
│   └── ...
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_non_iid_partition_visualization.ipynb
│   ├── 03_centralized_baseline.ipynb
│   ├── 04_fl_phase1_analysis.ipynb
│   ├── 05_attack_analysis.ipynb
│   ├── 06_strategy_comparison.ipynb
│   ├── 07_aggregator_internals.ipynb
│   └── 08_ids2018_eda.ipynb        ← IDS2018 EDA (class dist, feature funnel, partition heatmap)
├── artifacts/
│   ├── raw/                        # cicids2018_raw.parquet (1.8 GB, IDS2018)
│   ├── preprocessed/               # label_encoder.pkl, feature_cols.pkl, scaler.pkl, test_set.npz
│   ├── data/                       # client_0000.npz … client_0049.npz (50 shards, Non-IID Dirichlet α=0.5)
│   ├── models/                     # baseline_mlp.pth (IDS2018 centralized checkpoint)
│   ├── plots/                      # EDA plots: ids2018_class_dist.png, ids2018_corr_heatmap.png, ids2018_noniid_partition.png
│   ├── results/
│   └── plots/
├── experiment_logs/                # Logs from test_50_client_simulation.py runs (batchnorm, layernorm, gradclip variants)
├── src/
│   ├── configs/
│   │   ├── config.py               # Loads config.yaml → CONFIG dict used everywhere
│   │   ├── config.yaml             # ALL hyperparameters centralized here
│   │   └── paths.py                # ✅ IMPLEMENTED — path constants + ensure_dirs()
│   ├── logging/logger.py
│   ├── exception/exception.py
│   ├── pipelines/
│   │   ├── data_pipeline.py                  ✅ IMPLEMENTED
│   │   ├── centralized_training_pipeline.py  ✅ IMPLEMENTED
│   │   ├── attack_pipeline.py                ✅ IMPLEMENTED
│   │   ├── training_pipeline.py              ✅ IMPLEMENTED (manual FL loop + evaluator wired)
│   │   └── evaluation_pipeline.py            ✅ IMPLEMENTED
│   └── components/
│       ├── data/
│       │   ├── data_loader.py                ✅ IMPLEMENTED
│       │   ├── data_preprocessor.py          ✅ IMPLEMENTED
│       │   ├── data_partitioner.py           ✅ IMPLEMENTED
│       │   └── torch_dataset.py              ✅ IMPLEMENTED
│       ├── model/
│       │   └── model.py                      ✅ IMPLEMENTED (MLPClassifier [256,128,64])
│       ├── client/
│       │   ├── client.py                     ✅ IMPLEMENTED (FLIDSClient + attack wiring)
│       │   └── attacker.py                   ✅ IMPLEMENTED (flip_labels, inject_backdoor_trigger, scale_gradient_to_norm)
│       ├── server/
│       │   ├── aggregator.py                 ✅ Variant A — AL-CMT + clip_to_median_norm (FLAME 2022)
│       │   ├── baselines.py                  ✅ FedAvg, TrimmedMean, Krum, GeoMed, LayerwiseCosineKrum
│       │   ├── hra_aggregator.py             ✅ NEW — HRA Baseline (HRA 2026, NIDS-specific competitor)
│       │   ├── server.py                     ✅ get_initial_parameters, server_evaluate_fn
│       │   ├── ae_scorer.py                  ✅ Variant B — AE anomaly scorer
│       │   ├── ssfg_aggregator.py            ✅ Variant C — SVD spectral filter on all clients
│       │   ├── triage_aggregator.py          ✅ NOVELTY — MSFT (SVD hybrid on suspicious subset)
│       │   └── ablation_aggregators.py       ✅ FullModelCosineAggregator, FinalLayerNoSimplexAggregator
│       └── evaluation/
│           └── evaluator.py                  ✅ IMPLEMENTED
└── tests/
    ├── flower_smoke_test.py              ✅ (legacy — uses Ray, skip on Python 3.13)
    ├── test_50_client_simulation.py      ✅ CLI experiment runner for FL simulations
    ├── test_labels.py                    ✅ Validates partition label distribution
    ├── test_aggregator.py                ✅ 8 tests, all passing
    ├── test_client.py                    ✅
    ├── test_model.py                     ✅
    ├── test_partitioner.py               ✅
    └── test_new_implementations.py       ✅ NEW — 53 tests covering all SOTA-upgrade components
```

---

## 3. What Is FULLY Implemented

> **Novel Novelty Added (2026-07-17):** `triage_aggregator.py` (MSFT core) + `ablation_aggregators.py` (2 ablations). `training_pipeline.py` extended with 5 new strategy names. `run_all_experiments.py` updated with full matrix.

### 3.1 `src/configs/paths.py`
```python
PROJECT_ROOT, ARTIFACTS_DIR, RAW_DIR, PREPROCESSED_DIR,
DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR

def ensure_dirs() -> None   # creates all artifact subdirectories
```

---

### 3.2 Data Pipeline — `src/pipelines/data_pipeline.py`

Orchestrates the full data flow end-to-end:

```
load_cicids2018()          ← auto-downloads via kagglehub if not cached
  → preprocess()           (Steps 1–6 + StandardScaler — returns X, y, feature_cols, le, scaler)
  → saves label_encoder.pkl, feature_cols.pkl, scaler.pkl
  → train_test_split()     (80/20 stratified → saves test_set.npz with keys X_test/y_test)
  → run_partitioning()     (Non-IID Dirichlet α=0.5 → saves client_0000.npz … client_0049.npz)
```

**Verified output (2026-07-17):** 16,232,943 rows → 44 features → 15 classes → 50 client shards

---

### 3.3 Data Loader — `src/components/data/data_loader.py`

- `load_cicids2018()` — auto-downloads CSE-CIC-IDS2018 via `kagglehub` (`solarmainframe/ids-intrusion-csv`), reads all 10 CSV files, handles `"Infinity"` strings, duplicate header rows, dtype unification via `pd.to_numeric`. Returns a clean 16M-row DataFrame.
- `load_cicids2017()` — legacy function kept for reference, not used in current pipeline.

---

### 3.4 Data Preprocessor — `src/components/data/data_preprocessor.py`

> **IMPORTANT:** `preprocess()` now applies `StandardScaler` internally and returns it as the 5th value.
> Return signature: `(X, y, feature_cols, le, scaler)` — **not** the 4-value signature from older docs.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `drop_unusable()` | Drops `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`; removes all-NaN cols; **filters rows where Label=="Label"** (IDS2018 stray header rows); keeps only numeric + Label |
| 2 | `impute()` | Inf/-Inf → NaN, NaN → column median |
| 3 | `variance_filter()` | Removes zero-variance features (0 removed on IDS2018) |
| 4 | `correlation_filter()` | Removes Pearson `|r| > 0.95` → **44 features** remain |
| 5 | `encode_labels()` | `LabelEncoder`: Benign=0, Bot=1, … 15 classes total |
| 6 | `StandardScaler` | Fit on train data inside `preprocess()`, transform applied to all splits |

> **Note:** `data_pipeline.py` calls `preprocess()` on the full raw DataFrame, then splits into train/test. Scaler is fit on the training portion only via `scaler.transform()` — not re-fit.

---

### 3.5 Data Partitioner — `src/components/data/data_partitioner.py`

```python
def partition_iid(X, y, num_clients)                             -> List[Tuple[ndarray, ndarray]]
def partition_non_iid(X, y, num_clients, alpha=0.5)              -> List[Tuple[ndarray, ndarray]]
def save_partitions(partitions, output_dir)
    # ⚠️ Each .npz gets an 80/20 train/val split internally:
    #    keys: X_train, y_train, X_val, y_val
def load_partition(client_id: int)                               -> (X_train, y_train, X_val, y_val)
def load_partition_dataloaders(client_id, batch_size=64)         -> (train_loader, val_loader)
def run_partitioning(X, y)                                       -> None  (reads CONFIG, saves .npz)
```

Config keys: `federated.num_clients`, `data.partition_mode`, `data.alpha_dirichlet`, `data.val_split_ratio`, `data.random_seed`

---

### 3.6 PyTorch DataLoader — `src/components/data/torch_dataset.py`

```python
make_dataloader(X, y, batch_size=32, shuffle=True) -> DataLoader
```

---

### 3.7 MLP Model — `src/components/model/model.py`

```python
class MLPClassifier(nn.Module):
    # Architecture: input_dim → [256, 128, 64] → num_classes
    # Per hidden layer: Linear → BatchNorm1d → LeakyReLU(0.01) → Dropout(0.2)
    # Output: raw logits (CrossEntropyLoss applies softmax internally)
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.2)
    def forward(self, x) -> Tensor

def get_model_parameters(model)          -> List[np.ndarray]   # state_dict → NumPy list (Flower)
def set_model_parameters(model, params)  -> None               # NumPy list → state_dict (Flower)
```

> ✅ **TRAINED** — Centralized baseline (IDS2018): `input_dim=44, hidden_dims=[256,128,64], num_classes=15`.
> Test Macro F1 = **0.7570**, Accuracy = 0.77, 54,543 params.
> Checkpoint: `artifacts/models/baseline_mlp.pth`

---

### 3.8 Centralized Training Pipeline — `src/pipelines/centralized_training_pipeline.py`

Full training loop for the centralized MLP baseline (not part of FL — establishes the FL target):

```python
def get_weighted_loss_fn(y_train, device) -> nn.CrossEntropyLoss
    # Inverse-frequency weights capped at weight_cap=10.0 (prevents gradient explosion on rare classes)
def train_one_epoch(model, loader, criterion, optimizer, device) -> float
def evaluate(model, loader, criterion, device)                   -> (loss, macro_f1, preds, targets)
def run_centralized_training()                                   -> None
```

- Optimizer: `Adam` with `weight_decay`
- Scheduler: `ReduceLROnPlateau(mode="min", factor=0.5, patience=5)`
- Saves best model by val Macro F1 → `artifacts/models/baseline_mlp.pth`
- Config block: `CONFIG["centralized"]`

---

### 3.9 FL Client — `src/components/client/client.py`

```python
class FLIDSClient(fl.client.NumPyClient):
    def __init__(
        self, cid, train_loader, val_loader, model,
        config=None,
        X_train_raw=None,   # required for backdoor injection
        y_train_raw=None,   # required for backdoor + class weighting
    )
    def get_parameters(self, config)           -> List[np.ndarray]
    def fit(self, parameters, config)          -> (params, num_examples, metrics)
    def evaluate(self, parameters, config)     -> (loss, num_examples, metrics)
```

**fit() metrics returned:** `train_loss, train_accuracy, cid, is_poisoned, attack_active, labels_flipped, source_labels_seen`

**Attack logic (wired, activated by `is_poisoned=True` + `server_round >= attack_start_round`):**

| Attack type | Mechanism | When applied |
|-------------|-----------|-------------|
| `label_flip` | Per-batch: `flip_labels(y.numpy(), source_class, target_class)` | During training |
| `backdoor` | Pre-loop: `inject_backdoor_trigger()` -> rebuild DataLoader | Before training |
| `both` | label_flip + backdoor combined | Both phases |
| `sign_flip` | Post-training: reverse and scale update delta | After training |
| `min_max` | NEW: min_max_attack() — max damage within epsilon of nearest benign | After training |
| `min_sum` | NEW: min_sum_attack() — minimise cosine similarity to all clients | After training |
| `lie` | NEW: lie_attack() — mean + z*std per param (evades MAD) | After training |
| `trust_then_strike` | NEW: act benign for trust_rounds, then activate min_max/lie | After training |
| Gradient norm scaling | `scale_gradient_to_norm()` if `scale_to_benign_norm=True` | After training |

**Config keys read by client:** `device`, `local_epochs`, `lr`, `weight_decay`, `is_poisoned`, `attack_type`, `attack_start_round`, `source_class`, `target_class`, `trigger_feature_idx`, `trigger_values`, `inject_ratio`, `scale_to_benign_norm`, `benign_norm_target`, `sign_flip_scale`, `gradient_clip_norm`, `weight_cap`

**Server config key read from Flower `config` arg in fit():** `server_round`

**Evaluate Metrics:** Computes `accuracy`, `macro_f1`, `weighted_f1`, and `target_class_accuracy` (recall on the source class) using scikit-learn.

---

### 3.10 Attacker — `src/components/client/attacker.py`

```python
def flip_labels(y, source_class, target_class) -> np.ndarray
    # Targeted semantic attack: DDoS (class 4) -> Benign (class 0)

def inject_backdoor_trigger(X, y, trigger_feature_idx, trigger_values, inject_ratio, benign_class=0)
    # Appends n_inject rows with trigger signature, mislabeled as benign

def scale_gradient_to_norm(local_weights, target_norm) -> List[np.ndarray]
    # Scales gradient L2-norm to target — stealth bypass of naive norm-clipping

# NEW (2026-07-25) — Advanced attacks from SOTA literature:
def min_max_attack(local_update, all_updates, epsilon=0.5) -> List[np.ndarray]
    # NDSS 2021: max damage while staying within epsilon of nearest benign client
    # Designed to evade Cosine+MAD clustering defenses

def min_sum_attack(local_update, all_updates) -> List[np.ndarray]
    # NDSS 2021: minimize cosine similarity to all other clients
    # Directly attacks our pairwise Cosine+MAD scoring

def lie_attack(all_updates, z_clip=2.0) -> List[np.ndarray]
    # FedLAW ICLR 2026: inject mean + z*std per param — stays within benign variance
    # Specifically designed to evade variance-based scoring (our MAD)
```

---

### 3.11 Server Aggregator — `src/components/server/aggregator.py`

Variant A — AL-CMT (Adaptive Layer-Wise Cosine-MAD Trust) aggregation.

```python
def extract_final_layer(ndarrays: List[np.ndarray]) -> np.ndarray
    # Extracts output layer weight (-2) and flattens to 1D vector

def clip_to_median_norm(all_ndarrays) -> List[List[np.ndarray]]
    # NEW (2026-07-25) — FLAME (USENIX 2022): clip every client's full param vector
    # to the cohort median L2-norm BEFORE any scoring. Closes Constrain-and-Scale
    # backdoor vulnerability (FLAME proved BA 100%->0% on IoT-Traffic with this alone).
    # Called at the top of aggregate_fit() in all three novel strategies.

def compute_layer_wise_cosine_similarity(final_layers: np.ndarray) -> np.ndarray
    # Computes K x K cosine similarity matrix using pdist

def compute_mad_scores(sim_matrix: np.ndarray) -> np.ndarray
    # Computes robust Z-scores using Median Absolute Deviation (MAD)

def temperature_scaled_softmax(scores: np.ndarray, temperature: float) -> np.ndarray
    # Softmax with LogSumExp and temperature scaling

def project_capped_simplex(v: np.ndarray, cap_t: float) -> np.ndarray
    # O(K log K) capped simplex projection to bound client weights

class RobustFLIDSStrategy(flwr.server.strategy.Strategy):
    # Custom Strategy orchestrating the 5-step defense pipeline in aggregate_fit
```

---

### 3.12 Server Entry Point — `src/components/server/server.py`

```python
def get_initial_parameters() -> flwr.common.Parameters
    # Loads model from baseline checkpoint or initializes weights

def server_evaluate_fn(server_round, parameters, config) -> Optional[Tuple[float, dict]]
    # Server-side evaluation using test_set.npz (Macro F1, loss, accuracy)
```

---

### 3.13 Training Pipeline — `src/pipelines/training_pipeline.py`

Manual FL round loop (replaces `flwr.simulation` — Ray not available on Python 3.13):

```python
def run_experiment(results_suffix: str = "") -> None
    # Runs N rounds: sample clients → fit → aggregate → server_evaluate_fn
    # After each round: log_round_results() → round_results{suffix}.csv
    #                   log_trust_scores()  → trust_scores{suffix}.csv
    # results_suffix used by attack sweep to write separate CSVs per attacker_ratio
```

> ✅ **Phase 1 complete** — 30 rounds, 0% attackers, ~64 min runtime on CPU.
> Final global model: Macro F1=**0.5435**, Acc=**99.28%**, saved to `artifacts/models/fl_global_model.pth`

---

### 3.14 Baseline Aggregators — `src/components/server/baselines.py`

```python
class FedAvgBaseline(fl.server.strategy.Strategy)
    # Uniform weighted average of all client updates (1/K weights)

class FedTrimmedMeanBaseline(fl.server.strategy.Strategy)
    # Per-parameter: sort clients, trim top+bottom beta=20%, average rest

class KrumBaseline(fl.server.strategy.Strategy)
    # Select top multi_k clients by minimum sum of squared distances to neighbours

# NEW (2026-07-25) — SOTA baselines required for academic comparison:
class GeoMedianBaseline(fl.server.strategy.Strategy)
    # Geometric Median (Weiszfeld algorithm) — foundation of HRA 2026 and KBS 2025
    # Callable via strategy="geomed"

class LayerwiseCosineKrumBaseline(fl.server.strategy.Strategy)
    # KBS 2025: Krum selection per-layer using Cosine distance (not Euclidean)
    # Achieves +6-13% accuracy over standard Krum on CIFAR-10 under label-flip
    # Callable via strategy="layerwise_cosine_krum"

def get_baseline_strategy(name: str) -> Strategy
    # Factory: "fedavg" | "trimmed_mean" | "krum" | "geomed" | "hra" | "layerwise_cosine_krum"
```

---

### 3.14b HRA Aggregator — `src/components/server/hra_aggregator.py` (NEW 2026-07-25)

```python
class HRABaseline(fl.server.strategy.Strategy):
    # Hybrid Reputation Aggregation (HRA 2026) — most competitive NIDS-specific paper.
    # Pipeline: clip_to_median_norm -> GeoMed distance -> phi(delta) piecewise weight
    #           -> EMA reputation -> weighted average
    # 98.66% accuracy on 5G traffic NIDS under 30% malicious clients.
    # Key weakness: static T_low/T_high thresholds must be manually tuned per dataset.
    # Our MSFT advantage: MAD Z-Score is self-normalising, no threshold tuning.
    # Callable via strategy="hra"
```

Config keys: `defense.hra_t_low` (0.3), `defense.hra_t_high` (0.7)

---

### 3.15 AE Scorer — `src/components/server/ae_scorer.py` (Variant B)

```python
class AEScorer:
    def __init__(self, input_dim, hidden_factor=4, train_epochs=5, lr=1e-3)
    def fit(self, vectors: np.ndarray) -> None
        # Trains encoder→decoder on final-layer weight vectors from trusted clients
    def score(self, vectors: np.ndarray) -> np.ndarray
        # Returns -reconstruction_error per client (high anomaly = lowest score)
        # Drop-in compatible with MAD score convention in RobustFLIDSStrategy
```

---

### 3.16 SSFG Aggregator — `src/components/server/ssfg_aggregator.py` (Variant C)

```python
class SSFGAggregator(fl.server.strategy.Strategy):
    # Applies SVD spectral filtering on ALL clients before cosine scoring.
    # _spectral_filter(): SVD → keep top svd_keep_ratio singular values → reconstruct
```

---

### 3.17 Triage Aggregator — `src/components/server/triage_aggregator.py` (NOVELTY — MSFT)

```python
class TriageAggregator(fl.server.strategy.Strategy):
    # MSFT: Multi-Stage Final-Layer Triage (upgraded 2026-07-25)
    # Stage 0: clip_to_median_norm (FLAME 2022) — closes Constrain-and-Scale gap
    # Stage 1: Cosine + MAD on ALL clients
    # Stage 2: SVD HYBRID on suspicious-ONLY subset (MAD < triage_soft_threshold)
    #          Score = 0.6 * re-MAD(filtered_subspace) + 0.4 * SVD_projection_score
    #          SVD projection score = 1 - |U[:,0]| / max(|U[:,0]|)  (extends DnC, NDSS 2021)
    # Stage 3: Merge scores -> EMA reputation -> capped simplex weights
    # Stage 4: Weighted aggregation across all model layers
```

Config keys: `defense.triage_soft_threshold` (-2.0), `defense.svd_keep_ratio` (0.9)

---

### 3.18 Ablation Aggregators — `src/components/server/ablation_aggregators.py`

```python
class FullModelCosineAggregator:      # Cosine on full flattened model — expected to fail on Non-IID
class FinalLayerNoSimplexAggregator:  # Final layer + EMA but plain softmax instead of capped simplex
```

Both share `_BaseAblation` class. Used to prove each component of MSFT is necessary.

### 3.17 Evaluator — `src/components/evaluation/evaluator.py`

```python
def compute_metrics(y_true, y_pred) -> dict
    # Returns: accuracy, macro_f1, weighted_f1, fpr, confusion_matrix

def compute_asr(model, trigger_loader, benign_class_idx=0) -> float
    # Attack Success Rate: fraction of backdoor-triggered samples classified as benign

def log_round_results(server_round, metrics, filename="round_results.csv") -> None
    # Appends row to artifacts/results/{filename}

def log_trust_scores(server_round, trust_scores, filename="trust_scores.csv") -> None
    # Appends per-client EMA reputation per round (for heatmap)
```

---

### 3.18 Attack Pipeline — `src/pipelines/attack_pipeline.py`

```python
def select_malicious_clients(num_clients, attacker_ratio, seed=42) -> list
def is_attack_active(server_round, attack_start_round) -> bool
def get_attack_config(client_id, malicious_ids, server_round) -> dict
def run_attack_sweep() -> None
    # Iterates attacker_ratios from CONFIG, calls run_experiment(results_suffix=...)
```

---

### 3.19 Evaluation Pipeline — `src/pipelines/evaluation_pipeline.py`

```python
def run_evaluation() -> None
    # Loads round_results.csv for each strategy
    # Plots: macro_f1_vs_rounds.png, accuracy_vs_rounds.png, fpr_vs_rounds.png
    # Plots: trust_heatmap.png (client reputation across rounds)
    # Prints final-round summary table to console
```

---

## 4. Known Constraints & Bugs Fixed

| Issue | Fix |
|-------|-----|
| `ray` not available on Python 3.13 | Replaced `flwr.simulation.start_simulation()` with manual FL loop |
| `baseline_mlp.pth` saves `{model_state_dict, epoch, ...}` not raw state_dict | `saved.get("model_state_dict", saved)` in `server.py` and `training_pipeline.py` |
| Old partition `.npz` files have keys `X`, `y` (not `X_train`/`X_val`) | `load_partition()` checks key names and splits inline if old format |
| `test_set.npz` saved with `X`/`y` keys, `server.py` expected `X_test`/`y_test` | `server_evaluate_fn` now tries both key names |
| `project_capped_simplex` original rho-search was incorrect | Replaced with binary search on Lagrange multiplier `gamma` |
| IDS2018 CSVs have `"Infinity"` strings in numeric columns | Added `na_values=["Infinity","-Infinity","inf","-inf"]` to `pd.read_csv` in loader |
| IDS2018 CSVs have duplicate header rows mid-file | `on_bad_lines="skip"` in loader + `df[df["Label"] != "Label"]` in preprocessor |
| `pd.concat` on 10 CSVs with mixed dtypes caused `ArrayMemoryError` | Force `pd.to_numeric(errors="coerce")` on all feature columns before concat |
| `centralized_training_pipeline.py` used old `"X"`/`"y"` npz keys | Updated to `"X_test"`/`"y_test"` + replaced stale IDS2017 parquet load with client shard aggregation |
| `centralized_training_pipeline.py` printed `★` — UnicodeEncodeError on Windows cp1252 | Replaced with `*` |

---

## 5. Research Architecture (The 3 Gaps)

| Gap | Problem | Solution |
|-----|---------|----------|
| Gap 1 | Non-IID benign clients look malicious if whole model is compared | **Final-layer only** + MAD (Variant A) or AE reconstruction error (Variant B) |
| Gap 2 | Standard filters are O(K²) — too slow for IoT | **Capped Simplex Projection O(K log K)** — shared by A and B |
| Gap 3 | Rigid accept/reject permanently bans good nodes | **EMA Trust Scores + Temperature-Scaled Softmax** — shared by A and B |

> **Variant C (SSFG):** Bypasses Steps 2/3/4 entirely — uses Truncated SVD on the full update matrix to extract the benign subspace. Preserves good parts of even poisoned updates.

---

## 6. Test Suite

| File | Status | What it tests |
|------|--------|---------------|
| `test_model.py` | pass | `MLPClassifier` forward/backward, `get/set_model_parameters` |
| `test_client.py` | pass | `FLIDSClient` fit/evaluate cycle (synthetic 2-class) |
| `test_partitioner.py` | pass | `partition_non_iid`, `save_partitions`, `load_partition` |
| `test_baselines.py` | pass | `FedAvgBaseline`, `FedTrimmedMeanBaseline`, `KrumBaseline` |
| `test_ae_scorer.py` | pass | `AEScorer` training and anomaly scoring |
| `test_ssfg.py` | pass | `SSFGAggregator` and `_spectral_filter` logic |
| `test_evaluator.py` | pass | `compute_metrics` and CSV logging |
| `test_attack_pipeline.py` | pass | `select_malicious_clients`, `get_attack_config` |
| `test_new_implementations.py` | **53/53 pass** | All SOTA-upgrade components (2026-07-25): |
| | | `clip_to_median_norm`, `_geometric_median`, `GeoMedianBaseline` |
| | | `LayerwiseCosineKrumBaseline`, `HRABaseline` |
| | | `min_max_attack`, `min_sum_attack`, `lie_attack` |
| | | SVD hybrid formula edge cases, integration pipeline |

> **Note on test dims:** `test_client.py` and `test_model.py` use `num_classes=2` / `input_dim=78` intentionally — they are pure unit tests using synthetic data and don't require the real dataset. `flower_smoke_test.py` uses CONFIG values to match production.

---

## 7. Experiment Plan

| Phase | Rounds | Activity |
|-------|--------|----------|
| Phase 1 | 1–10 | Clean baseline, 0% attackers — AE also trains during this phase |
| Phase 2 | 11–30 | Byzantine injection (10%, 30%, 50% attacker ratios) |
| Phase 3 | Eval | Plot F1-score drop + ASR spike across all strategies |

**Strategies compared (6 total):**
| Strategy | Type |
|----------|------|
| FedAvg | Baseline (unprotected) |
| FedTrimmedMean | Classical robust baseline |
| Krum | Classical robust baseline |
| **Variant A: AL-CMT** (Cosine+MAD+EMA+Simplex) | **Novel — primary contribution** |
| **Variant B: CS-ARF** (AE+EMA+Simplex) | **Novel — secondary contribution** |
| **Variant C: SSFG** (SVD subspace filtering) | **Novel — tertiary comparison** |

**Key metrics:** Macro F1-Score, Attack Success Rate (ASR), False Positive Rate (FPR)

**Network:** N=50 clients total, C=20 sampled per round

**Implementation order — ALL COMPLETE:**
1. ✅ Data pipeline
2. ✅ MLPClassifier — trained (centralized Macro F1=0.7463)
3. ✅ FLIDSClient — Flower NumPyClient with attack gate
4. ✅ attacker.py — flip_labels, inject_backdoor_trigger, scale_gradient_to_norm
5. ✅ server.py + aggregator.py — Variant A AL-CMT
6. ✅ training_pipeline.py — manual FL loop (Ray workaround)
7. ✅ baselines.py — FedAvg, TrimmedMean, Krum
8. ✅ attack_pipeline.py — select_malicious_clients, get_attack_config, run_attack_sweep
9. ✅ evaluator.py — compute_metrics, compute_asr, log_round_results, log_trust_scores
10. ✅ evaluation_pipeline.py — comparison plots + summary table
11. ✅ ae_scorer.py — Variant B AE anomaly scorer
12. ✅ ssfg_aggregator.py — Variant C SVD spectral filter

- ✅ **New batch scripts:** PowerShell files like `run_experiments_queue.ps1` execute complex combinations of attack ratios and gradient clipping settings.
- ✅ **Metrics upgraded:** `client.py` and `aggregator.py` now track `macro_f1`, `weighted_f1`, and `target_class_accuracy` via `evaluate_metrics_aggregation_fn`.
- ⏳ Next Steps: Analyze experiment logs generated by the queue scripts or run further comparisons.

---

## 8. Config Keys Reference (`config.yaml`)

```yaml
model:        input_dim (44), hidden_dims ([256,128,64]), num_classes (15), dropout_rate (0.2)
federated:    num_clients (50), clients_per_round (20), num_rounds (50), local_epochs (5),
              learning_rate (0.001), batch_size (256), optimizer ("adam")
data:         partition_mode ("non_iid"), alpha_dirichlet (0.5), val_split_ratio (0.2),
              random_seed (42), num_workers (4)
attack:       attacker_ratio (0.0->0.10/0.30/0.50), attack_start_round (11),
              attack_type ("label_flip"|"backdoor"|"both"|"sign_flip"
                           |"min_max"|"min_sum"|"lie"|"trust_then_strike"),
              source_class (4=DDOS attack-HOIC), target_class (0=Benign),
              trigger_feature_idx ([0,5]), trigger_values ([999999,1]),
              inject_ratio (0.1), scale_to_benign_norm (true),
              min_max_epsilon (0.5),       <- Min-Max attack bound
              lie_z_clip (2.0),            <- LIE attack z-factor
              trust_rounds (5),            <- Mimicry: rounds to act benign
              strike_attack_type ("min_max")  <- Mimicry: attack after trust phase
defense:      mad_threshold (-3.0), analyze_layers ("final"),
              max_byzantine_fraction (0.3), ema_momentum (0.9), temperature (2.0),
              initial_reputation (0.0), ae_hidden_factor (4), ae_train_epochs (5),
              triage_soft_threshold (-2.0), svd_keep_ratio (0.9),  <- MSFT keys
              hra_t_low (0.3), hra_t_high (0.7)                    <- HRA keys
evaluation:   metrics, primary_metric ("macro_f1"), save_confusion_matrix, plot_every_n_rounds
experiment:   phase1_clean_rounds (10), phase2_attack_rounds (40),
              attacker_ratios ([0.10,0.30,0.50]), baseline_strategies
centralized:  epochs (50), batch_size (256), learning_rate (0.001),
              weight_decay (0.00001), weight_cap (10.0), scheduler_patience (5)
```

---

## 9. Key Implementation Rules

- Every file uses `from src.logging.logger import logging` and `from src.exception.exception import FLIDSException`
- All config is read via `from src.configs.config import CONFIG`
- All paths from `from src.configs.paths import *`
- `make_dataloader()` from `torch_dataset.py` is the standard DataLoader factory
- `attacker.py` functions are called **inside** `client.py`'s `fit()` method
- Server NEVER sees raw CSE-CIC-IDS2018 data — only PyTorch weight arrays (NumPy ndarrays)
- `server_round` is sent from server to client via Flower's `config` dict in `configure_fit()`

---

## 10. Strict Project Boundaries (NEVER suggest these)

- ❌ Blockchain or Homomorphic Encryption — too heavy for IoT
- ❌ Heavy models (CNNs, Transformers) on clients — only MLP
- ❌ Server-side raw data — server sees ONLY PyTorch weights (NumPy arrays)
- ❌ Ollama or any LLM — out of scope
- ❌ Client-side complex defense logic — ALL defense math is server-only

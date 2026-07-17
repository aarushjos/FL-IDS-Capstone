# Novel Multi-Stage Final-Layer Triage — Implementation Plan

## Goal

Implement the **Multi-Stage Final-Layer Triage (MSFT)** pipeline — the top recommendation from the [research design document](file:///c:/Users/rauna/Videos/FL%20IDS/docs/gemini%20researches/Federated%20Learning%20IDS%20Research%20Design%20for%20new%20novel%20pipeline.md) — along with the ablation studies needed to rigorously prove its novelty claims in the capstone report.

### Background

The research document analyzed 8 state-of-the-art frameworks (DnC, FLAME, FREPD, HRA, TFFL, FedLAW, Layerwise, SpectralKrum) and identified a **genuinely defensible novelty gap**: coupling *final-layer-only* anomaly scoring with *capped simplex projection* in a *multi-stage triage* architecture. The key insight is:

- Early MLP layers exhibit massive legitimate Non-IID variance (different IoT traffic patterns)
- The final classification layer is where label-flipping and backdoor attacks *must* manifest
- A fast cosine filter + selective SVD secondary check gives maximum defense at minimum cost

Your existing codebase already has the building blocks (Variant A cosine+MAD, Variant C SSFG spectral filter, simplex projection). The novel contribution is **combining them into a two-stage triage** with proper ablation evidence.

---

## User Review Required

> [!IMPORTANT]
> **Strategy Naming**: The plan names the new strategy `"triage"` (internally `TriageAggregator`). This will be the flagship "novel" strategy in your paper, positioned above your existing Variants A/B/C.

> [!IMPORTANT]
> **Experiment Matrix Scope**: The full experiment matrix (7 strategies × 3 attack ratios × 50 rounds × 20 clients/round) will take **~15-20 hours on CPU**. I recommend running overnight using the `/goal` command. Do you want to reduce rounds to 30 or keep 50?

> [!WARNING]
> **Dirichlet Sensitivity Study**: The research document recommends testing α=0.1, 0.5, 1.0, 100. Each α value requires **re-partitioning the dataset** (re-running `data_pipeline.py`) and a full experiment sweep. This multiplies total experiment time by 4×. I suggest doing α=0.5 (current, already partitioned) first, then α=0.1 as a second phase if time allows.

---

## Open Questions

> [!IMPORTANT]
> **Q1:** The research document also mentions **Dual-Stream Profiling** (Pipeline 5 — cosine direction + L₂ magnitude scoring combined). Should I implement this as an additional strategy to test, or keep the scope to just MSFT + ablations?

> [!IMPORTANT]
> **Q2:** Your current experiments ran at 30 rounds. The config now says 50 rounds. Should new experiments use **30 rounds** (for comparability with existing results) or **50 rounds** (for deeper convergence analysis)?

> [!IMPORTANT]
> **Q3:** Should we also add **runtime benchmarks** (measuring `aggregate_fit` wall-clock time per round) to prove the O(K²·c) efficiency claim? This requires adding timing instrumentation to each strategy.

---

## Proposed Changes

### Component 1: Bug Fix — `server_evaluate_fn` Missing Metrics

Before any new experiments, fix the `weighted_f1` and `fpr` always-zero bug. Currently [server.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/server/server.py#L79-L86) manually computes only `macro_f1` and `accuracy`, ignoring the full `compute_metrics()` function in [evaluator.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/evaluation/evaluator.py#L10-L34).

#### [MODIFY] [server.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/server/server.py)

- Replace the manual `f1_score()` + accuracy computation (lines 79–86) with a call to `compute_metrics(y_test, preds)`
- Return all metrics: `macro_f1`, `weighted_f1`, `accuracy`, `fpr`
- Remove the unused `from sklearn.metrics import f1_score` import (it'll come through `compute_metrics`)

```diff
- from sklearn.metrics import f1_score
+ from src.components.evaluation.evaluator import compute_metrics

  # In server_evaluate_fn, replace:
- macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
- accuracy = float((preds == y_test).mean())
- return loss, {"macro_f1": macro_f1, "accuracy": accuracy, "round": server_round}
+ metrics = compute_metrics(y_test, preds)
+ return loss, {
+     "macro_f1": metrics["macro_f1"],
+     "weighted_f1": metrics["weighted_f1"],
+     "accuracy": metrics["accuracy"],
+     "fpr": metrics["fpr"],
+     "round": server_round,
+ }
```

---

### Component 2: Novel Strategy — Multi-Stage Final-Layer Triage (`TriageAggregator`)

This is the **core novel contribution**. The architecture from the research document:

```
Client MLP Weights → Slice W_final
    ↓
[Stage 1 — Fast Filter]  Pairwise Cosine Similarity on W_final
    → MAD Z-Score Computation
    → If Z-Score > threshold: Mark 'Benign'
    → If Z-Score < threshold: Mark 'Suspicious'
    ↓
[Stage 2 — Deep Verification] (ONLY for Suspicious updates)
    → Stack suspicious final-layer vectors into matrix
    → Truncated SVD → Project onto principal component
    → Assign refined anomaly score
    ↓
[Stage 3 — Momentum & Projection]
    → Merge Stage 1 benign scores + Stage 2 refined scores
    → EMA Trust Update: r_t = ρ · r_{t-1} + (1-ρ) · Score
    → Capped Simplex Projection → Exact aggregation weights [0.0, cap_t]
    ↓
[Stage 4 — Global Aggregation]
    → Apply weights to FULL PyTorch model → Return secured global model
```

**Why this is novel vs. existing literature:**
- vs. HRA: scores only final layer → removes Non-IID feature-space noise that shatters geometric medians
- vs. FREPD: requires zero warm-up rounds (cosine + SVD are dynamic, no AE training needed)
- vs. FedLAW: single-round communication (simplex computed server-side from weight analytics)
- vs. Layerwise Cosine: processes one layer instead of all layers → fraction of computational cost
- vs. SpectralKrum: restricting SVD to final layer makes label-flip the dominant spectral anomaly (SpectralKrum fails on label-flip because the signal is buried in full-model noise)

#### [NEW] [triage_aggregator.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/server/triage_aggregator.py)

New file implementing `TriageAggregator(fl.server.strategy.Strategy)`:

**Key design decisions:**
- Reuses `extract_final_layer()`, `compute_layer_wise_cosine_similarity()`, `compute_mad_scores()`, `temperature_scaled_softmax()`, `project_capped_simplex()` from existing [aggregator.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/server/aggregator.py)
- Reuses `_spectral_filter()` from existing [ssfg_aggregator.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/server/ssfg_aggregator.py)
- **Stage 1**: Cosine+MAD on ALL clients (identical to Variant A)
- **Stage 2**: Only clients with `mad_score < soft_threshold` get secondary SVD verification. The `soft_threshold` is a new config param (default: `-2.0`, less aggressive than the hard `-3.0` to cast a wider triage net)
- **Score Merge**: Benign clients keep their Stage 1 MAD score. Suspicious clients get their score replaced by the SVD-based projection score (which is more discriminative in the reduced suspicious-only subspace)
- **Stage 3-4**: Same EMA + simplex as Variant A
- Logs additional metrics: `n_suspicious` (how many went to Stage 2), `n_benign_fast` (how many passed Stage 1 directly)
- Optional: timing instrumentation for runtime benchmarks

```python
class TriageAggregator(fl.server.strategy.Strategy):
    """
    Multi-Stage Final-Layer Triage (MSFT) — Novel defense pipeline.
    
    Stage 1: Fast cosine+MAD filter on final classification layer
    Stage 2: SVD spectral verification ONLY for suspicious clients
    Stage 3: EMA trust + capped simplex projection
    Stage 4: Weighted aggregation across full model
    """
```

---

### Component 3: Ablation Study — Full-Model Cosine (`FullModelCosineAggregator`)

The research document states: *"Ablation 1 will implement Full-Model Cosine paired with Simplex, designed to empirically prove that full-model scoring fails catastrophically on Non-IID data by falsely rejecting benign clients."*

This ablation is essential to defend the final-layer hypothesis against reviewer challenges.

#### [NEW] [ablation_aggregators.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/server/ablation_aggregators.py)

Contains two ablation strategies:

**1. `FullModelCosineAggregator`** — Cosine similarity on the ENTIRE flattened model (all ~58k params), then MAD → EMA → Simplex. This should *fail badly* on Non-IID data (falsely flagging good clients with rare traffic).

```python
def _extract_full_model(ndarrays: List[np.ndarray]) -> np.ndarray:
    """Flatten ALL model parameters into a single 1D vector."""
    return np.concatenate([nd.flatten() for nd in ndarrays])
```

**2. `FinalLayerNoSimplexAggregator`** — Final-layer cosine+MAD → EMA → simple softmax normalization (NO capped simplex). This proves that the simplex is mathematically necessary to bound extreme magnitude attacks.

```python
# Instead of project_capped_simplex(), just normalize:
weights = trust_weights / trust_weights.sum()
```

---

### Component 4: Dual-Stream Profiling Strategy (Optional — based on Q1 answer)

Pipeline 5 from the research document: evaluate final layer for **both** directional shift (cosine) AND magnitude explosion (L₂ norm MAD), then combine scores.

#### [NEW] [dual_stream_aggregator.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/server/dual_stream_aggregator.py) *(conditional on Q1)*

```python
class DualStreamAggregator(fl.server.strategy.Strategy):
    """
    Dual-Stream Profiling: Cosine direction + L2 magnitude on final layer.
    Combined score fed into EMA + Simplex.
    """
```

---

### Component 5: Wiring — Strategy Factory and Training Pipeline

#### [MODIFY] [training_pipeline.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/pipelines/training_pipeline.py)

Update `_build_strategy()` to support new strategy names:

```diff
  def _build_strategy(strategy_name: str, initial_parameters):
      if strategy_name == "robust":
          return RobustFLIDSStrategy(initial_parameters=initial_parameters)
      elif strategy_name == "ssfg":
          return SSFGAggregator(initial_parameters=initial_parameters)
+     elif strategy_name == "triage":
+         return TriageAggregator(initial_parameters=initial_parameters)
+     elif strategy_name == "full_model_cosine":
+         return FullModelCosineAggregator(initial_parameters=initial_parameters)
+     elif strategy_name == "final_no_simplex":
+         return FinalLayerNoSimplexAggregator(initial_parameters=initial_parameters)
+     elif strategy_name == "dual_stream":
+         return DualStreamAggregator(initial_parameters=initial_parameters)
      else:
          return get_baseline_strategy(strategy_name)
```

---

### Component 6: Enhanced Experiment Runner

#### [MODIFY] [run_all_experiments.py](file:///c:/Users/rauna/Videos/FL%20IDS/run_all_experiments.py)

Add the novel strategies and ablations to the experiment matrix:

```python
def main():
    # 1. Novel MSFT (Triage) — sweep across attacker ratios
    run_attack_sweep(strategy_name="triage")
    
    # 2. Existing defenses — sweep at 30% attack ratio
    CONFIG["attack"]["attacker_ratio"] = 0.30
    for strategy in ["robust", "ssfg", "fedavg", "trimmed_mean", "krum"]:
        run_experiment(results_suffix=f"_{strategy}_ratio_30pct", strategy_name=strategy)
    
    # 3. Ablation studies — at 30% attack ratio
    for ablation in ["full_model_cosine", "final_no_simplex"]:
        run_experiment(results_suffix=f"_{ablation}_ratio_30pct", strategy_name=ablation)
    
    # 4. Optional: Dual-stream — at 30%
    run_experiment(results_suffix="_dual_stream_ratio_30pct", strategy_name="dual_stream")
```

---

### Component 7: Config Updates

#### [MODIFY] [config.yaml](file:///c:/Users/rauna/Videos/FL%20IDS/src/configs/config.yaml)

Add triage-specific parameters under `defense:`:

```yaml
defense:
  # ... existing params ...
  
  # Multi-Stage Triage (MSFT) params
  triage_soft_threshold: -2.0   # Stage 1 → Stage 2 transition threshold
                                 # More lenient than hard mad_threshold (-3.0)
                                 # Casts wider net for SVD verification
  svd_keep_ratio: 0.9           # Fraction of singular values retained in Stage 2
```

---

### Component 8: Tests for Novel Strategies

#### [NEW] [test_triage.py](file:///c:/Users/rauna/Videos/FL%20IDS/tests/test_triage.py)

Unit tests for `TriageAggregator`:
1. **test_all_benign_passthrough** — When all MAD scores are above threshold, no clients go to Stage 2
2. **test_suspicious_triggers_svd** — When some clients are flagged, Stage 2 SVD is invoked on the subset
3. **test_attacker_weight_zeroed** — With clear outlier, final simplex weight is exactly 0.0
4. **test_metrics_contain_triage_info** — Returned metrics include `n_suspicious`, `n_benign_fast`
5. **test_stage2_only_runs_on_suspicious** — SVD is NOT computed for benign clients (efficiency test)

#### [NEW] [test_ablation_aggregators.py](file:///c:/Users/rauna/Videos/FL%20IDS/tests/test_ablation_aggregators.py)

Unit tests for ablation strategies:
1. **test_full_model_cosine_aggregation** — Basic aggregation works
2. **test_final_no_simplex_no_capping** — Weights are not bounded by cap_t
3. **test_full_model_flags_benign_noniid** — With diverse (Non-IID-like) synthetic weights, full-model incorrectly flags benign clients as malicious

---

### Component 9: Runtime Benchmarking (conditional on Q3)

#### [MODIFY] [training_pipeline.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/pipelines/training_pipeline.py)

Add timing instrumentation around `strategy.aggregate_fit()`:

```python
import time

t0 = time.perf_counter()
aggregated, agg_metrics = strategy.aggregate_fit(server_round, flower_results, [])
agg_time = time.perf_counter() - t0
agg_metrics["agg_time_ms"] = round(agg_time * 1000, 2)
```

#### [MODIFY] [evaluator.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/evaluation/evaluator.py)

Add `agg_time_ms` column to `log_round_results` if present in metrics.

---

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| [server.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/server/server.py) | MODIFY | Fix `weighted_f1`/`fpr` = 0 bug |
| [triage_aggregator.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/server/triage_aggregator.py) | NEW | **Core novelty**: Multi-Stage Final-Layer Triage |
| [ablation_aggregators.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/server/ablation_aggregators.py) | NEW | Full-Model Cosine + Final-No-Simplex ablations |
| [dual_stream_aggregator.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/components/server/dual_stream_aggregator.py) | NEW | *(Optional)* Dual-Stream Profiling |
| [training_pipeline.py](file:///c:/Users/rauna/Videos/FL%20IDS/src/pipelines/training_pipeline.py) | MODIFY | Wire new strategies + timing |
| [run_all_experiments.py](file:///c:/Users/rauna/Videos/FL%20IDS/run_all_experiments.py) | MODIFY | Add novel + ablation experiments |
| [config.yaml](file:///c:/Users/rauna/Videos/FL%20IDS/src/configs/config.yaml) | MODIFY | Add triage thresholds |
| [test_triage.py](file:///c:/Users/rauna/Videos/FL%20IDS/tests/test_triage.py) | NEW | Unit tests for MSFT |
| [test_ablation_aggregators.py](file:///c:/Users/rauna/Videos/FL%20IDS/tests/test_ablation_aggregators.py) | NEW | Unit tests for ablations |

---

## Verification Plan

### Automated Tests

```bash
# Run all tests including new ones
python -m pytest tests/ -v

# Run only new strategy tests
python -m pytest tests/test_triage.py tests/test_ablation_aggregators.py -v
```

### Manual Verification

1. **Smoke Test** — Run triage strategy for 3 rounds at 0% attack to verify it produces valid F1 scores:
   ```bash
   python -c "from src.pipelines.training_pipeline import run_experiment; run_experiment(results_suffix='_triage_smoke', strategy_name='triage')"
   ```
   (After setting `num_rounds: 3` temporarily in config)

2. **Ablation Validation** — Run `full_model_cosine` at 30% attack for 10 rounds, verify F1 drops significantly more than `triage` or `robust` (proving the hypothesis)

3. **Statistical Significance** — After full experiment matrix:
   - Run 5 independent seeds for each strategy at 30% attack ratio
   - Perform paired t-test between MSFT and each baseline
   - Verify p < 0.05 for F1 improvement over FedAvg, Krum
   - Document in the capstone report

4. **Runtime Benchmarking** — Compare `agg_time_ms` across strategies to prove O(K²·c) << O(K²·d) claim

### Expected Outcomes

| Strategy | Expected F1 (30% attack) | Rationale |
|----------|--------------------------|-----------|
| **MSFT (Triage)** | **≥ 0.54** | SVD secondary check catches edge cases MAD misses |
| Robust (Var A) | ~0.537 | Existing result — no Stage 2 |
| SSFG (Var C) | ~0.539 | SVD on all clients — less discriminative |
| FedAvg | ~0.498 | No defense |
| Krum | ~0.468 | Catastrophic Non-IID failure |
| **Full-Model Cosine (Ablation)** | **< 0.50** | Falsely rejects benign Non-IID clients |
| **Final-No-Simplex (Ablation)** | **< 0.53** | Cannot zero-out attacker weights |

---

## Execution Order

1. **Fix the bug** — Component 1 (server.py) — 5 minutes
2. **Implement MSFT** — Component 2 (triage_aggregator.py) — 30 minutes
3. **Implement ablations** — Component 3 (ablation_aggregators.py) — 20 minutes
4. **Wire everything** — Components 5-7 (pipeline, runner, config) — 15 minutes
5. **Write tests** — Component 8 — 20 minutes
6. **Run tests** — Verify all pass — 5 minutes
7. **Smoke test** — 3-round quick run of triage — 10 minutes
8. **Full experiment matrix** — Overnight run — ~15-20 hours
9. **Runtime benchmarks** — Component 9 (conditional) — 10 minutes

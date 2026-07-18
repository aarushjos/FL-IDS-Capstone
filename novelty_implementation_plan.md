# Novelty Implementation Plan — MSFT + Ablations

## Analysis of Existing Code

From reading the codebase, here is exactly what already exists and is reusable:

| Function | File | Reused by |
|----------|------|-----------|
| `extract_final_layer()` | `aggregator.py` | Triage Stage 1 + ablations |
| `compute_layer_wise_cosine_similarity()` | `aggregator.py` | Triage Stage 1 |
| `compute_mad_scores()` | `aggregator.py` | Triage Stage 1 |
| `temperature_scaled_softmax()` | `aggregator.py` | Triage Stage 3 |
| `project_capped_simplex()` | `aggregator.py` | Triage Stage 3 |
| `_spectral_filter()` | `ssfg_aggregator.py` | Triage Stage 2 |
| `_update_ema_reputation()` | pattern in both aggregators | Triage Stage 3 |

**Key insight from reading the code:** `SSFGAggregator` applies SVD to ALL clients before cosine scoring. Triage's novelty is applying SVD **only to the suspicious subset** — which is a meaningful architectural difference, not just a cosmetic one.

**`server.py` bug check:** Already fixed — `compute_metrics()` is imported and used correctly on line 79. Bug from the novelty plan is already resolved. ✅

**`_build_strategy()` in `training_pipeline.py`:** Simple if/elif at line 85-91. Extend by adding 3 new branches.

---

## Files to Create/Modify

| File | Action | Lines est. |
|------|--------|-----------|
| `triage_aggregator.py` | NEW | ~80 |
| `ablation_aggregators.py` | NEW | ~80 |
| `training_pipeline.py` | MODIFY | +3 lines |
| `run_all_experiments.py` | MODIFY | replace body |
| `config.yaml` | MODIFY | +2 keys |

---

## Proposed Changes

### 1. `triage_aggregator.py` — Core Novelty

**Logic:**
```
All clients → extract final layer
    ↓
Stage 1: Cosine + MAD on ALL clients
    → mad_score >= soft_threshold  →  "Benign" (keep Stage 1 score)
    → mad_score <  soft_threshold  →  "Suspicious" (go to Stage 2)
    ↓
Stage 2: SVD on suspicious-only submatrix
    → get refined score from SVD projection
    ↓
Stage 3: Merge scores → EMA → Simplex
Stage 4: Weighted aggregation on full model
```

**Config keys needed:** `triage_soft_threshold: -2.0`, `svd_keep_ratio: 0.9`  
(softer than hard `mad_threshold: -3.0` to cast a wider net for Stage 2)

---

### 2. `ablation_aggregators.py` — Two Ablations

**`FullModelCosineAggregator`** — Same pipeline as Variant A but cosine on the full flattened model (~58k params). Should fail on Non-IID data (flags good clients with rare traffic as outliers).

**`FinalLayerNoSimplexAggregator`** — Same as Variant A but replaces `project_capped_simplex()` with plain softmax normalization. Proves simplex is necessary to zero out extreme attackers.

---

### 3. `training_pipeline.py` — 3 new strategy names

```python
elif strategy_name == "triage":
    return TriageAggregator(initial_parameters=initial_parameters)
elif strategy_name == "full_model_cosine":
    return FullModelCosineAggregator(initial_parameters=initial_parameters)
elif strategy_name == "final_no_simplex":
    return FinalLayerNoSimplexAggregator(initial_parameters=initial_parameters)
```

---

### 4. `run_all_experiments.py` — Full matrix

Runs: triage, robust, ssfg, fedavg, trimmed_mean, krum, full_model_cosine, final_no_simplex — each at 3 attacker ratios.

---

### 5. `config.yaml` — 2 new keys under `defense:`

```yaml
triage_soft_threshold: -2.0
svd_keep_ratio: 0.9
```

---

## Code Style Rules (per your request)
- No unnecessary comments, no print statements, no verbose variable names
- Match the style of `ssfg_aggregator.py` exactly (short, clean)
- Reuse all existing helper functions — no duplication

---

## Execution Order

1. `config.yaml` — add 2 keys (2 min)
2. `triage_aggregator.py` — implement (15 min)
3. `ablation_aggregators.py` — implement (10 min)
4. `training_pipeline.py` — add 3 branches (2 min)
5. `run_all_experiments.py` — update matrix (5 min)
6. Smoke test: `num_rounds: 3`, run triage strategy

> [!IMPORTANT]
> Proceed with implementation?

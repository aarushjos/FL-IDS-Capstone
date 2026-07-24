import sys
import copy
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.configs.config import CONFIG
from src.logging.logger import logging
from src.pipelines.training_pipeline import run_experiment


# ── All valid strategy names ──────────────────────────────────────────────────
# "fedavg" | "trimmed_mean" | "krum"               <- classical baselines
# "geomed" | "layerwise_cosine_krum" | "hra"        <- SOTA baselines (NEW)
# "robust" | "ssfg" | "triage"                      <- our novel strategies
# "full_model_cosine" | "final_no_simplex"           <- ablations
# ─────────────────────────────────────────────────────────────────────────────


def _run(strategy: str, attack_type: str, ratio: float):
    """Run one experiment cell. Saves CONFIG state, restores it after."""
    saved_type = CONFIG["attack"]["attack_type"]
    saved_ratio = CONFIG["attack"]["attacker_ratio"]
    try:
        CONFIG["attack"]["attack_type"] = attack_type
        CONFIG["attack"]["attacker_ratio"] = ratio
        suffix = f"_{strategy}_{attack_type}_{int(ratio * 100):02d}pct"
        logging.info(f"[Experiment] strategy={strategy} attack={attack_type} ratio={ratio}")
        run_experiment(results_suffix=suffix, strategy_name=strategy)
    finally:
        CONFIG["attack"]["attack_type"] = saved_type
        CONFIG["attack"]["attacker_ratio"] = saved_ratio


# ── Experiment 1: Clean baseline ──────────────────────────────────────────────
def exp_clean():
    """0% attackers across all strategies. Verifies no-defense overhead."""
    strategies = [
        "fedavg", "trimmed_mean", "krum",
        "geomed", "layerwise_cosine_krum", "hra",
        "robust", "ssfg", "triage",
    ]
    for s in strategies:
        _run(s, "label_flip", 0.0)   # attack_type irrelevant at ratio 0


# ── Experiment 2: Label-Flip Sweep ───────────────────────────────────────────
def exp_label_flip():
    """Standard Byzantine: targeted label-flip at 10/30/50% attacker ratios."""
    strategies = [
        "fedavg", "krum", "geomed", "layerwise_cosine_krum", "hra",
        "robust", "triage",
    ]
    for ratio in [0.10, 0.30, 0.50]:
        for s in strategies:
            _run(s, "label_flip", ratio)


# ── Experiment 3: Backdoor ────────────────────────────────────────────────────
def exp_backdoor():
    """Constrain-and-Scale backdoor at 10/30% attacker ratios."""
    strategies = [
        "fedavg", "geomed", "hra", "robust", "triage",
    ]
    for ratio in [0.10, 0.30]:
        for s in strategies:
            _run(s, "backdoor", ratio)


# ── Experiment 4: Min-Max Attack (NDSS 2021) ─────────────────────────────────
def exp_min_max():
    """AGR-agnostic Min-Max at 30%: targets clustering defenses."""
    strategies = ["fedavg", "geomed", "hra", "robust", "triage"]
    for s in strategies:
        _run(s, "min_max", 0.30)


# ── Experiment 5: LIE Attack (FedLAW ICLR 2026) ─────────────────────────────
def exp_lie():
    """LIE attack at 30%: mean+z*std evades variance-based MAD scoring."""
    strategies = ["fedavg", "geomed", "hra", "robust", "triage"]
    for s in strategies:
        _run(s, "lie", 0.30)


# ── Experiment 6: Adaptive Mimicry (Trust-then-Strike) ───────────────────────
def exp_mimicry():
    """Act benign for N rounds, then strike. Tests EMA temporal memory."""
    strategies = ["fedavg", "geomed", "hra", "triage"]
    for s in strategies:
        _run(s, "trust_then_strike", 0.30)


# ── Experiment 7: Ablation Study ─────────────────────────────────────────────
def exp_ablation():
    """30% LIE attack across MSFT + ablation variants.
    Proves each component of MSFT is necessary:
      full_model_cosine  -> fails on Non-IID diversity
      final_no_simplex   -> attacker monopolises softmax
      robust (no SVD)    -> LIE evades pure MAD
      triage (full MSFT) -> catches LIE via SVD hybrid
    """
    strategies = ["triage", "robust", "full_model_cosine", "final_no_simplex"]
    for s in strategies:
        _run(s, "lie", 0.30)


# ── CLI entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FL-IDS Experiment Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=["clean", "label_flip", "backdoor", "min_max", "lie", "mimicry", "ablation", "all"],
        help=(
            "clean        - 0%% attackers across all strategies (sanity check)\n"
            "label_flip   - Targeted label-flip at 10/30/50%% attacker ratios\n"
            "backdoor     - Constrain-and-Scale backdoor at 10/30%%\n"
            "min_max      - NDSS 2021 Min-Max attack at 30%%\n"
            "lie          - FedLAW 2026 LIE attack at 30%%\n"
            "mimicry      - Trust-then-Strike adaptive mimicry at 30%%\n"
            "ablation     - MSFT ablation study (LIE 30%% on all variants)\n"
            "all          - Run every experiment above in sequence\n"
        ),
    )
    args = parser.parse_args()

    suite = {
        "clean":      exp_clean,
        "label_flip": exp_label_flip,
        "backdoor":   exp_backdoor,
        "min_max":    exp_min_max,
        "lie":        exp_lie,
        "mimicry":    exp_mimicry,
        "ablation":   exp_ablation,
    }

    if args.exp == "all":
        for name, fn in suite.items():
            logging.info(f"\n{'='*60}\n[Experiment Suite] Starting: {name}\n{'='*60}")
            fn()
    else:
        suite[args.exp]()

    logging.info(f"[Experiment] Suite '{args.exp}' completed.")


if __name__ == "__main__":
    main()

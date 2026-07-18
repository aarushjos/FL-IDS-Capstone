import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.configs.config import CONFIG
from src.logging.logger import logging
from src.pipelines.training_pipeline import run_experiment


RATIOS = CONFIG["experiment"]["attacker_ratios"]  # [0.10, 0.30, 0.50]

# Strategies to sweep across all attacker ratios
SWEEP_STRATEGIES = ["triage", "robust", "ssfg"]

# Strategies to run only at 30% (for baseline comparison table)
FIXED_STRATEGIES = ["fedavg", "trimmed_mean", "krum", "full_model_cosine", "final_no_simplex"]
FIXED_RATIO = 0.30


def run_all():
    # Phase 1: sweep triage + robust + ssfg at all 3 attack ratios
    for strategy in SWEEP_STRATEGIES:
        for ratio in RATIOS:
            logging.info(f"[Experiment] strategy={strategy} ratio={ratio}")
            CONFIG["attack"]["attacker_ratio"] = ratio
            suffix = f"_{strategy}_ratio_{int(ratio * 100):02d}pct"
            run_experiment(results_suffix=suffix, strategy_name=strategy)

    # Phase 2: fixed ratio comparison for baselines + ablations
    CONFIG["attack"]["attacker_ratio"] = FIXED_RATIO
    for strategy in FIXED_STRATEGIES:
        logging.info(f"[Experiment] strategy={strategy} ratio={FIXED_RATIO}")
        suffix = f"_{strategy}_ratio_{int(FIXED_RATIO * 100):02d}pct"
        run_experiment(results_suffix=suffix, strategy_name=strategy)

    logging.info("[Experiment] All done.")


if __name__ == "__main__":
    run_all()

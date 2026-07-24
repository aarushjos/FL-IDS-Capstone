import sys
import math
import random

import numpy as np
import torch
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from src.configs.config import CONFIG
from src.configs.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR, ensure_dirs
from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.components.model.model import MLPClassifier, get_model_parameters, set_model_parameters
from src.components.data.data_partitioner import load_partition
from src.components.data.torch_dataset import make_dataloader
from src.components.client.client import FLIDSClient
from src.components.server.aggregator import RobustFLIDSStrategy
from src.components.server.baselines import get_baseline_strategy
from src.components.server.ssfg_aggregator import SSFGAggregator
from src.components.server.triage_aggregator import TriageAggregator
from src.components.server.ablation_aggregators import FullModelCosineAggregator, FinalLayerNoSimplexAggregator
from src.components.server.server import get_initial_parameters, server_evaluate_fn
from src.components.evaluation.evaluator import log_round_results, log_trust_scores


class _SimpleProxy(ClientProxy):
    def __init__(self, cid):
        super().__init__(cid)
    def reconnect(self, *a, **kw): pass
    def get_properties(self, *a, **kw): pass
    def get_parameters(self, *a, **kw): pass
    def fit(self, *a, **kw): pass
    def evaluate(self, *a, **kw): pass


def _select_malicious_ids(num_clients: int, attacker_ratio: float, seed: int = 42) -> set:
    num_attackers = int(math.floor(num_clients * attacker_ratio))
    rng = random.Random(seed)
    return set(rng.sample(range(num_clients), num_attackers))


def _make_client(cid: int, is_poisoned: bool, model_cfg: dict, fed_cfg: dict, attack_cfg: dict) -> FLIDSClient:
    X_train, y_train, X_val, y_val = load_partition(cid)

    batch_size = fed_cfg["batch_size"]
    train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = MLPClassifier(
        input_dim=model_cfg["input_dim"],
        hidden_dims=model_cfg["hidden_dims"],
        num_classes=model_cfg["num_classes"],
        dropout_rate=model_cfg["dropout_rate"],
    )

    client_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "local_epochs": fed_cfg["local_epochs"],
        "lr": fed_cfg["learning_rate"],
        "weight_decay": 1e-5,
        "num_classes": model_cfg["num_classes"],
        "weight_cap": 10.0,
        "is_poisoned": is_poisoned,
        "attack_type": attack_cfg["attack_type"],
        "attack_start_round": attack_cfg["attack_start_round"],
        "source_class": attack_cfg["source_class"],
        "target_class": attack_cfg["target_class"],
        "trigger_feature_idx": attack_cfg["trigger_feature_idx"],
        "trigger_values": attack_cfg["trigger_values"],
        "inject_ratio": attack_cfg["inject_ratio"],
        "scale_to_benign_norm": attack_cfg["scale_to_benign_norm"],
        "benign_norm_target": 1.0,
        "min_max_epsilon": float(attack_cfg.get("min_max_epsilon", 0.5)),
        "lie_z_clip": float(attack_cfg.get("lie_z_clip", 2.0)),
        "trust_rounds": int(attack_cfg.get("trust_rounds", 5)),
        "strike_attack_type": str(attack_cfg.get("strike_attack_type", "min_max")),
    }

    return FLIDSClient(
        cid=str(cid),
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        config=client_config,
        X_train_raw=X_train,
        y_train_raw=y_train,
    )


def _build_strategy(strategy_name: str, initial_parameters):
    if strategy_name == "robust":
        return RobustFLIDSStrategy(initial_parameters=initial_parameters)
    elif strategy_name == "ssfg":
        return SSFGAggregator(initial_parameters=initial_parameters)
    elif strategy_name == "triage":
        return TriageAggregator(initial_parameters=initial_parameters)
    elif strategy_name == "full_model_cosine":
        return FullModelCosineAggregator(initial_parameters=initial_parameters)
    elif strategy_name == "final_no_simplex":
        return FinalLayerNoSimplexAggregator(initial_parameters=initial_parameters)
    elif strategy_name == "hra":
        from src.components.server.hra_aggregator import HRABaseline
        return HRABaseline(initial_parameters=initial_parameters)
    else:
        return get_baseline_strategy(strategy_name)


def run_experiment(results_suffix: str = "", strategy_name: str = "robust") -> None:
    try:
        ensure_dirs()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        model_cfg = CONFIG["model"]
        fed_cfg = CONFIG["federated"]
        attack_cfg = CONFIG["attack"]

        num_clients = fed_cfg["num_clients"]
        clients_per_round = fed_cfg["clients_per_round"]
        num_rounds = fed_cfg["num_rounds"]
        attacker_ratio = float(attack_cfg["attacker_ratio"])

        malicious_ids = _select_malicious_ids(num_clients, attacker_ratio)
        if malicious_ids:
            logging.info(f"[Pipeline] Malicious client IDs: {sorted(malicious_ids)}")
        else:
            logging.info("[Pipeline] Phase 1 — no malicious clients (attacker_ratio=0.0).")

        initial_parameters = get_initial_parameters()
        strategy = _build_strategy(strategy_name, initial_parameters)
        logging.info(f"[Pipeline] Strategy: {strategy_name}")
        global_params = parameters_to_ndarrays(initial_parameters)

        results_file = f"round_results{results_suffix}.csv"
        trust_file = f"trust_scores{results_suffix}.csv"

        logging.info(
            f"[Pipeline] Starting FL loop — {num_clients} clients, "
            f"{clients_per_round} per round, {num_rounds} rounds."
        )

        rng = random.Random(CONFIG["data"]["random_seed"])

        for server_round in range(1, num_rounds + 1):
            logging.info(f"\n[Pipeline] ===== Round {server_round}/{num_rounds} =====")

            sampled_ids = rng.sample(range(num_clients), clients_per_round)

            fit_config = {
                "server_round": server_round,
                "local_epochs": fed_cfg["local_epochs"],
                "lr": fed_cfg["learning_rate"],
                "batch_size": fed_cfg["batch_size"],
            }

            # Collect fit results from sampled clients
            raw_results = []
            for cid in sampled_ids:
                client = _make_client(
                    cid=cid,
                    is_poisoned=(cid in malicious_ids),
                    model_cfg=model_cfg,
                    fed_cfg=fed_cfg,
                    attack_cfg=attack_cfg,
                )
                client_params, num_examples, metrics = client.fit(global_params, fit_config)
                raw_results.append((cid, client_params, num_examples, metrics))

                if metrics.get("attack_active"):
                    logging.info(f"  [Client {cid}] ATTACK ACTIVE | loss={metrics['train_loss']:.4f}")
                else:
                    logging.info(f"  [Client {cid}] loss={metrics['train_loss']:.4f} acc={metrics['train_accuracy']:.4f}")

            # Build Flower FitRes objects for aggregator
            flower_results = []
            for cid, client_params, num_examples, metrics in raw_results:
                proxy = _SimpleProxy(str(cid))
                fit_res = FitRes(
                    status=Status(code=Code.OK, message=""),
                    parameters=ndarrays_to_parameters(client_params),
                    num_examples=num_examples,
                    metrics=metrics,
                )
                flower_results.append((proxy, fit_res))

            # Aggregate
            aggregated, agg_metrics = strategy.aggregate_fit(server_round, flower_results, [])
            if aggregated is not None:
                global_params = parameters_to_ndarrays(aggregated)
                logging.info(
                    f"[Pipeline] Round {server_round} aggregated — "
                    f"flagged={agg_metrics.get('flagged', 0)}, "
                    f"min_trust={agg_metrics.get('min_trust', 0):.4f}, "
                    f"max_trust={agg_metrics.get('max_trust', 0):.4f}"
                )

                # Log trust scores for heatmap (only for strategies that track reputation)
                if hasattr(strategy, "reputation_scores"):
                    log_trust_scores(server_round, strategy.reputation_scores, filename=trust_file)

            # Server-side evaluation + CSV logging
            eval_result = server_evaluate_fn(server_round, ndarrays_to_parameters(global_params), {})
            if eval_result:
                loss, eval_metrics = eval_result
                logging.info(
                    f"[Pipeline] Round {server_round} eval — "
                    f"macro_f1={eval_metrics.get('macro_f1', 0):.4f}, "
                    f"acc={eval_metrics.get('accuracy', 0):.4f}, "
                    f"loss={loss:.4f}"
                )
                log_round_results(server_round, eval_metrics, filename=results_file)

        # Save final global model
        final_model = MLPClassifier(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            num_classes=model_cfg["num_classes"],
            dropout_rate=model_cfg["dropout_rate"],
        )
        set_model_parameters(final_model, global_params)
        save_path = MODELS_DIR / f"fl_global_model{results_suffix}.pth"
        torch.save(final_model.state_dict(), save_path)
        logging.info(f"[Pipeline] Final global model saved to {save_path}")

    except Exception as e:
        raise FLIDSException(e, sys)


if __name__ == "__main__":
    run_experiment()

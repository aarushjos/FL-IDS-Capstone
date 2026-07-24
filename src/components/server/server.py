import sys
from typing import Dict, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from src.components.evaluation.evaluator import compute_metrics

from src.configs.config import CONFIG
from src.configs.paths import PREPROCESSED_DIR, MODELS_DIR
from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.components.model.model import MLPClassifier, get_model_parameters, set_model_parameters
from src.components.server.aggregator import RobustFLIDSStrategy


def get_initial_parameters() -> Parameters:
    try:
        model_cfg = CONFIG["model"]
        model = MLPClassifier(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            num_classes=model_cfg["num_classes"],
            dropout_rate=model_cfg["dropout_rate"],
        )

        checkpoint = MODELS_DIR / "baseline_mlp.pth"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu")
            state_dict = saved.get("model_state_dict", saved)
            model.load_state_dict(state_dict, strict=False)
            logging.info(f"[Server] Loaded baseline checkpoint from {checkpoint}")
        else:
            logging.info("[Server] No checkpoint found — using random initial parameters.")

        return ndarrays_to_parameters(get_model_parameters(model))

    except Exception as e:
        raise FLIDSException(e, sys)


def server_evaluate_fn(
    server_round: int,
    parameters: Parameters,
    config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    try:
        test_path = PREPROCESSED_DIR / "test_set.npz"
        if not test_path.exists():
            return None

        data = np.load(test_path)
        keys = list(data.keys())
        X_test = data["X_test"].astype(np.float32) if "X_test" in keys else data["X"].astype(np.float32)
        y_test = data["y_test"].astype(np.int64) if "y_test" in keys else data["y"].astype(np.int64)

        model_cfg = CONFIG["model"]
        model = MLPClassifier(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            num_classes=model_cfg["num_classes"],
            dropout_rate=model_cfg["dropout_rate"],
        )
        set_model_parameters(model, parameters_to_ndarrays(parameters))
        model.eval()

        with torch.no_grad():
            logits = model(torch.tensor(X_test))
            preds = torch.argmax(logits, dim=1).numpy()

        loss = float(
            torch.nn.CrossEntropyLoss()(
                logits,
                torch.tensor(y_test),
            ).item()
        )

        metrics = compute_metrics(y_test, preds)

        logging.info(
            f"[Server] Round {server_round} — loss={loss:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"acc={metrics['accuracy']:.4f}, "
            f"fpr={metrics['fpr']:.4f}"
        )

        return loss, {
            "macro_f1":    metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "accuracy":    metrics["accuracy"],
            "fpr":         metrics["fpr"],
            "round":       server_round,
        }

    except Exception as e:
        raise FLIDSException(e, sys)


def build_strategy() -> RobustFLIDSStrategy:
    initial_parameters = get_initial_parameters()
    return RobustFLIDSStrategy(initial_parameters=initial_parameters)


if __name__ == "__main__":
    try:
        strategy = build_strategy()

        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=CONFIG["federated"]["num_rounds"]),
            strategy=strategy,
        )

    except Exception as e:
        raise FLIDSException(e, sys)

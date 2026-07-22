import argparse

import flwr as fl
from flwr.common import ndarrays_to_parameters

from src.components.client.client import FLIDSClient
from src.components.data.data_partitioner import (
    load_partition_dataloaders,
)
from src.components.model.model import (
    MLPClassifier,
    get_model_parameters,
)
from src.components.server.aggregator import RobustFLIDSStrategy
from src.configs.config import CONFIG
from src.configs.paths import DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an FL-IDS experiment.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=None,
        help="Gradient clipping max norm. Use 1.0 to enable.",
    )

    parser.add_argument(
        "--experiment-name",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--strategy",
        required=True,
        choices=["fedavg", "robust"],
    )

    parser.add_argument(
        "--attack",
        default="none",
        choices=[
            "none",
            "label_flip",
            "sign_flip",
        ],
    )

    parser.add_argument(
        "--poison-fraction",
        default=0.0,
        type=float,
    )

    return parser.parse_args()


ARGS = parse_args()

EXPERIMENT_NAME = ARGS.experiment_name
STRATEGY_NAME = ARGS.strategy
ATTACK_TYPE = ARGS.attack
POISON_FRACTION = ARGS.poison_fraction
GRADIENT_CLIP_NORM = ARGS.gradient_clip


NUM_CLIENTS = CONFIG["federated"].get(
    "num_clients",
    20,
)

CLIENTS_PER_ROUND = CONFIG["federated"].get(
    "clients_per_round",
    20,
)

NUM_ROUNDS = CONFIG["federated"].get(
    "num_rounds",
    5,
)

LOCAL_EPOCHS = CONFIG["federated"].get(
    "local_epochs",
    3,
)

BATCH_SIZE = CONFIG["federated"].get(
    "batch_size",
    256,
)

LEARNING_RATE = CONFIG["federated"].get(
    "learning_rate",
    1e-3,
)

INPUT_DIM = CONFIG["model"]["input_dim"]
NUM_CLASSES = CONFIG["model"]["num_classes"]
HIDDEN_DIMS = CONFIG["model"]["hidden_dims"]

SOURCE_CLASS = CONFIG["attack"].get(
    "source_class",
    4,
)

TARGET_CLASS = CONFIG["attack"].get(
    "target_class",
    0,
)

NUM_MALICIOUS_CLIENTS = int(
    NUM_CLIENTS * POISON_FRACTION
)


def check_experiment_config():
    if NUM_CLIENTS <= 0:
        raise ValueError(
            "NUM_CLIENTS must be greater than zero."
        )

    if not 1 <= CLIENTS_PER_ROUND <= NUM_CLIENTS:
        raise ValueError(
            "clients_per_round must be between 1 "
            "and num_clients."
        )

    if not 0.0 <= POISON_FRACTION <= 1.0:
        raise ValueError(
            "poison_fraction must be between 0 and 1."
        )

    if (
        ATTACK_TYPE == "none"
        and POISON_FRACTION != 0.0
    ):
        raise ValueError(
            "A clean experiment must use "
            "--poison-fraction 0.0."
        )

    if (
        ATTACK_TYPE != "none"
        and NUM_MALICIOUS_CLIENTS == 0
    ):
        raise ValueError(
            "Attack enabled, but poison fraction produced "
            "zero malicious clients."
        )


def check_partitions_exist():
    missing = []

    for cid in range(NUM_CLIENTS):
        path = DATA_DIR / f"client_{cid:04d}.npz"

        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} client partition files. "
            "Run `python -m src.pipelines.data_pipeline` first.\n"
            f"First missing file: {missing[0]}"
        )


def make_model():
    return MLPClassifier(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        num_classes=NUM_CLASSES,
    )


def client_fn(cid: str):
    cid_int = int(cid)

    is_poisoned = (
        ATTACK_TYPE != "none"
        and cid_int < NUM_MALICIOUS_CLIENTS
    )

    train_loader, val_loader = (
        load_partition_dataloaders(
            client_id=cid_int,
            batch_size=BATCH_SIZE,
        )
    )

    model = make_model()

    client = FLIDSClient(
        cid=cid,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        config={
            "local_epochs": LOCAL_EPOCHS,
            "lr": LEARNING_RATE,
            "device": "cpu",
            "num_classes": NUM_CLASSES,

            # Attack configuration
            "is_poisoned": is_poisoned,
            "attack_type": ATTACK_TYPE,

            # Zero ensures the attack is active from the
            # first round for both FedAvg and the custom
            # robust strategy.
            "attack_start_round": 0,

            "source_class": SOURCE_CLASS,
            "target_class": TARGET_CLASS,
            "sign_flip_scale": 1.0,

            # LayerNorm baseline: no gradient clipping
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
        },
    )

    return client.to_client()


def weighted_average(metrics):
    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    val_accuracy = sum(
        num_examples
        * metric.get(
            "val_accuracy",
            0.0,
        )
        for num_examples, metric in metrics
    ) / total_examples

    macro_f1 = sum(
        num_examples
        * metric.get(
            "macro_f1",
            0.0,
        )
        for num_examples, metric in metrics
    ) / total_examples

    weighted_f1 = sum(
        num_examples
        * metric.get(
            "weighted_f1",
            0.0,
        )
        for num_examples, metric in metrics
    ) / total_examples

    total_target_samples = sum(
        metric.get(
            "target_class_samples",
            0.0,
        )
        for _, metric in metrics
    )

    if total_target_samples > 0:
        target_class_accuracy = sum(
            metric.get(
                "target_class_samples",
                0.0,
            )
            * metric.get(
                "target_class_accuracy",
                0.0,
            )
            for _, metric in metrics
        ) / total_target_samples
    else:
        target_class_accuracy = 0.0

    return {
        "val_accuracy": float(val_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "target_class_accuracy": float(
            target_class_accuracy
        ),
    }


def fit_config(server_round: int):
    return {
        "server_round": server_round,
    }


def build_strategy():
    initial_model = make_model()

    initial_parameters = ndarrays_to_parameters(
        get_model_parameters(
            initial_model
        )
    )

    if STRATEGY_NAME == "robust":
        return RobustFLIDSStrategy(
            initial_parameters=initial_parameters,
            evaluate_metrics_aggregation_fn=(
                weighted_average
            ),
        )

    return fl.server.strategy.FedAvg(
        fraction_fit=(
            CLIENTS_PER_ROUND / NUM_CLIENTS
        ),
        fraction_evaluate=(
            CLIENTS_PER_ROUND / NUM_CLIENTS
        ),
        min_fit_clients=CLIENTS_PER_ROUND,
        min_evaluate_clients=(
            CLIENTS_PER_ROUND
        ),
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=(
            weighted_average
        ),
    )


if __name__ == "__main__":
    check_experiment_config()
    check_partitions_exist()

    print("=" * 65)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("Normalization: BatchNorm")
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Attack: {ATTACK_TYPE}")
    print(f"Poison fraction: {POISON_FRACTION}")
    print(
        "Malicious clients: "
        f"{NUM_MALICIOUS_CLIENTS}/{NUM_CLIENTS}"
    )
    print(f"Total clients: {NUM_CLIENTS}")
    print(f"Clients per round: {CLIENTS_PER_ROUND}")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Input dimension: {INPUT_DIM}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Source class: {SOURCE_CLASS}")
    print(f"Target class: {TARGET_CLASS}")
    print(
        "Gradient clipping:",
        (
            f"enabled (max_norm={GRADIENT_CLIP_NORM})"
            if GRADIENT_CLIP_NORM is not None
            else "disabled"
        ),
    )
    print("=" * 65)

    strategy = build_strategy()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS,
        ),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 0.0,
        },
    )
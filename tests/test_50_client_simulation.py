import flwr as fl
from flwr.common import ndarrays_to_parameters

from src.components.model.model import MLPClassifier, get_model_parameters
from src.components.client.client import FLIDSClient
from src.components.data.data_partitioner import load_partition_dataloaders
from src.components.server.aggregator import RobustFLIDSStrategy
from src.configs.config import CONFIG
from src.configs.paths import DATA_DIR


NUM_CLIENTS = 50
NUM_ROUNDS = 3
BATCH_SIZE = CONFIG["federated"].get("batch_size", 32)

INPUT_DIM = CONFIG["model"]["input_dim"]
NUM_CLASSES = CONFIG["model"]["num_classes"]
HIDDEN_DIMS = CONFIG["model"]["hidden_dims"]

# Options: "fedavg" or "robust"
STRATEGY_NAME = "robust"


def check_partitions_exist():
    missing = []
    for cid in range(NUM_CLIENTS):
        path = DATA_DIR / f"client_{cid:04d}.npz"
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} client partition files. "
            f"Run `python -m src.pipelines.data_pipeline` first.\n"
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
    is_poisoned = cid_int < 5

    train_loader, val_loader = load_partition_dataloaders(
        client_id=cid_int,
        batch_size=BATCH_SIZE,
    )

    model = make_model()

    client = FLIDSClient(
        cid=cid,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        config={
            "local_epochs": CONFIG["federated"].get(
                "local_epochs",
                1,
            ),
            "lr": CONFIG["federated"].get(
                "learning_rate",
                1e-3,
            ),
            "device": "cpu",
            "num_classes": NUM_CLASSES,

            # Sign-flipping attack
            "is_poisoned": is_poisoned,
            "attack_type": "sign_flip",
            "attack_start_round": 1,
            "sign_flip_scale": 1.0,

            # Disable gradient clipping for the first baseline.
            "gradient_clip_norm": 1.0,
        },
    )

    return client.to_client()


def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    val_accuracy = sum(
        num_examples * m.get("val_accuracy", 0.0)
        for num_examples, m in metrics
    ) / total_examples

    return {"val_accuracy": float(val_accuracy)}

def fit_config(server_round: int):
    return {
        "server_round": server_round,
    }


def build_strategy():
    initial_model = make_model()
    initial_parameters = ndarrays_to_parameters(
        get_model_parameters(initial_model)
    )

    if STRATEGY_NAME == "robust":
        return RobustFLIDSStrategy(
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
    )

    return fl.server.strategy.FedAvg(
        fraction_fit=0.4,
        fraction_evaluate=0.4,
        min_fit_clients=20,
        min_evaluate_clients=20,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )


if __name__ == "__main__":
    check_partitions_exist()

    strategy = build_strategy()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 0.0,
        },
    )
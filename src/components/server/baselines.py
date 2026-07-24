import numpy as np
import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Optional, Union

from src.configs.config import CONFIG
from src.components.server.server import get_initial_parameters


class FedAvgBaseline(fl.server.strategy.Strategy):
    def __init__(self, initial_parameters: Parameters):
        super().__init__()
        self.initial_parameters = initial_parameters

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_fit(self, server_round, results: List[Tuple], failures):
        if not results:
            return None, {}

        all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_examples = [fit_res.num_examples for _, fit_res in results]
        total = sum(num_examples)

        aggregated = [
            sum(p[i] * n for p, n in zip(all_params, num_examples)) / total
            for i in range(len(all_params[0]))
        ]
        return ndarrays_to_parameters(aggregated), {}

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


class FedTrimmedMeanBaseline(fl.server.strategy.Strategy):
    def __init__(self, initial_parameters: Parameters, beta: float = 0.2):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.beta = beta

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_fit(self, server_round, results: List[Tuple], failures):
        if not results:
            return None, {}

        all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        K = len(all_params)
        trim = max(1, int(self.beta * K))

        aggregated = []
        for i in range(len(all_params[0])):
            stacked = np.stack([p[i].flatten() for p in all_params], axis=0)
            stacked = np.sort(stacked, axis=0)
            trimmed = stacked[trim: K - trim]
            mean = trimmed.mean(axis=0)
            aggregated.append(mean.reshape(all_params[0][i].shape))

        return ndarrays_to_parameters(aggregated), {}

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


class KrumBaseline(fl.server.strategy.Strategy):
    def __init__(self, initial_parameters: Parameters, num_byzantine: int, multi_k: int = 1):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.num_byzantine = num_byzantine
        self.multi_k = multi_k

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_fit(self, server_round, results: List[Tuple], failures):
        if not results:
            return None, {}

        all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        K = len(all_params)
        n_select = K - self.num_byzantine - 2

        flat = [np.concatenate([p.flatten() for p in params]) for params in all_params]

        scores = np.zeros(K)
        for i in range(K):
            dists = sorted(np.sum((flat[i] - flat[j]) ** 2) for j in range(K) if j != i)
            scores[i] = sum(dists[:n_select])

        top_k = np.argsort(scores)[: self.multi_k]
        selected = [all_params[i] for i in top_k]

        aggregated = [
            np.mean([p[i] for p in selected], axis=0)
            for i in range(len(selected[0]))
        ]
        return ndarrays_to_parameters(aggregated), {}

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


def get_baseline_strategy(name: str) -> fl.server.strategy.Strategy:
    initial_parameters = get_initial_parameters()
    fed_cfg = CONFIG["federated"]
    defense_cfg = CONFIG["defense"]

    if name == "fedavg":
        return FedAvgBaseline(initial_parameters)
    elif name == "trimmed_mean":
        return FedTrimmedMeanBaseline(initial_parameters, beta=0.2)
    elif name == "krum":
        num_byzantine = int(defense_cfg["max_byzantine_fraction"] * fed_cfg["clients_per_round"])
        return KrumBaseline(initial_parameters, num_byzantine=num_byzantine, multi_k=1)
    elif name == "geomed":
        return GeoMedianBaseline(initial_parameters)
    elif name == "hra":
        from src.components.server.hra_aggregator import HRABaseline
        return HRABaseline(initial_parameters)
    elif name == "layerwise_cosine_krum":
        num_byzantine = int(defense_cfg["max_byzantine_fraction"] * fed_cfg["clients_per_round"])
        return LayerwiseCosineKrumBaseline(initial_parameters, num_byzantine=num_byzantine)
    else:
        raise ValueError(f"Unknown baseline: {name}")


def _geometric_median(points: np.ndarray, max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("_geometric_median requires at least 1 point")
    if len(points) == 1:
        return points[0].copy()
    median = points.mean(axis=0)
    for _ in range(max_iter):
        dists = np.linalg.norm(points - median, axis=1, keepdims=True)
        dists = np.maximum(dists, 1e-9)
        weights = 1.0 / dists
        new_median = (points * weights).sum(axis=0) / weights.sum()
        if np.linalg.norm(new_median - median) < tol:
            break
        median = new_median
    return median


class GeoMedianBaseline(fl.server.strategy.Strategy):
    """
    Geometric Median aggregation (Weiszfeld algorithm).

    Foundation of HRA (2026) and KBS (2025). All client updates are scored
    by their distance to the geometric median; the median itself serves as
    the aggregated model. This is the strongest classical robust aggregator —
    a necessary baseline to show MSFT improvement over.
    """

    def __init__(self, initial_parameters: Parameters):
        super().__init__()
        self.initial_parameters = initial_parameters

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        flat = np.stack([np.concatenate([p.flatten() for p in params]) for params in all_params])
        gm_flat = _geometric_median(flat)

        shapes = [p.shape for p in all_params[0]]
        agg = []
        idx = 0
        for shape in shapes:
            size = int(np.prod(shape))
            agg.append(gm_flat[idx: idx + size].reshape(shape))
            idx += size

        return ndarrays_to_parameters(agg), {}

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


class LayerwiseCosineKrumBaseline(fl.server.strategy.Strategy):
    """
    Layerwise Cosine Krum (KBS 2025).

    Applies Krum selection per-layer using Cosine distance instead of
    Euclidean. Consistently achieves +6-13% accuracy over standard Krum
    under label-flip attacks (e.g., 94.6% vs 82.8% on CIFAR-10).
    """

    def __init__(self, initial_parameters: Parameters, num_byzantine: int):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.num_byzantine = num_byzantine

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        from scipy.spatial.distance import cdist
        all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        K = len(all_params)
        n_select = max(1, K - self.num_byzantine - 2)
        agg = []

        for layer_idx in range(len(all_params[0])):
            layer_vecs = np.stack([p[layer_idx].flatten() for p in all_params])
            cos_dists = cdist(layer_vecs, layer_vecs, metric="cosine")
            scores = np.zeros(K)
            for i in range(K):
                sorted_dists = np.sort(cos_dists[i])
                scores[i] = sorted_dists[1: n_select + 1].sum()
            best = int(np.argmin(scores))
            agg.append(all_params[best][layer_idx])

        return ndarrays_to_parameters(agg), {}

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


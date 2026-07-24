import sys
import numpy as np
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays

from src.configs.config import CONFIG
from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.components.server.aggregator import clip_to_median_norm


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


class HRABaseline(fl.server.strategy.Strategy):
    """
    Hybrid Reputation Aggregation (HRA, 2026).

    The most competitive modern NIDS-specific defense. Implements:
      1. Distance to Geometric Median as instantaneous anomaly score.
      2. Static threshold piecewise weight phi(delta_j).
      3. EMA reputation update: r_j = rho*r_j + (1-rho)*phi(delta_j).
      4. Final weighted aggregation: w_j = r_j * phi(delta_j) (normalised).

    Achieves 98.66% accuracy on 5G traffic NIDS under 30% malicious clients.
    This is the key competitor MSFT must outperform to claim SOTA.

    Key weakness vs MSFT: static T_low/T_high thresholds are dataset-specific;
    a mis-tuned T_high drops accuracy by 43% (shown in HRA paper ablation).
    Our MAD Z-Score is self-normalising and requires no manual threshold tuning.
    """

    def __init__(self, initial_parameters: Parameters):
        super().__init__()
        self.initial_parameters = initial_parameters

        d = CONFIG["defense"]
        f = CONFIG["federated"]

        self.ema_momentum = float(d["ema_momentum"])
        self.initial_rep = float(d["initial_reputation"])

        # HRA static thresholds — set conservatively; see paper Section IV-B.
        # T_low: below this, client is fully penalised (weight=0).
        # T_high: above this, client is fully trusted (weight=1).
        self.t_low = float(d.get("hra_t_low", 0.3))
        self.t_high = float(d.get("hra_t_high", 0.7))

        self.clients_per_round = int(f["clients_per_round"])
        self.reputation: dict = {}

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def _phi(self, delta: float) -> float:
        if delta >= self.t_high:
            return 1.0
        if delta <= self.t_low:
            return 0.0
        return (delta - self.t_low) / (self.t_high - self.t_low)

    def aggregate_fit(self, server_round, results, failures):
        try:
            if not results:
                return None, {}

            ids = [str(p.cid) for p, _ in results]
            all_params = [parameters_to_ndarrays(r.parameters) for _, r in results]

            all_params = clip_to_median_norm(all_params)

            flat = np.stack([np.concatenate([p.flatten() for p in params]) for params in all_params])
            gm = _geometric_median(flat)

            max_dist = float(np.max(np.linalg.norm(flat - gm, axis=1))) + 1e-9
            deltas = 1.0 - np.linalg.norm(flat - gm, axis=1) / max_dist

            phis = np.array([self._phi(float(d)) for d in deltas])

            for cid, phi in zip(ids, phis):
                prev = self.reputation.get(cid, self.initial_rep)
                self.reputation[cid] = self.ema_momentum * prev + (1 - self.ema_momentum) * phi

            rep = np.array([self.reputation[cid] for cid in ids])
            raw_weights = rep * phis
            total = raw_weights.sum()
            if total < 1e-9:
                final_weights = np.ones(len(ids)) / len(ids)
            else:
                final_weights = raw_weights / total

            flagged = int((phis == 0.0).sum())
            logging.info(f"[HRA] Round {server_round}: {flagged}/{len(ids)} fully penalised.")

            agg = [
                np.average(np.stack([p[i] for p in all_params]), axis=0, weights=final_weights)
                for i in range(len(all_params[0]))
            ]

            return ndarrays_to_parameters(agg), {
                "round": server_round,
                "flagged": flagged,
                "min_trust": float(final_weights.min()),
                "max_trust": float(final_weights.max()),
            }

        except Exception as e:
            raise FLIDSException(e, sys)

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None

import sys
import numpy as np
import flwr as fl
from flwr.common import Parameters, FitRes, ndarrays_to_parameters, parameters_to_ndarrays

from src.configs.config import CONFIG
from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.components.server.aggregator import (
    extract_final_layer,
    clip_to_median_norm,
    compute_layer_wise_cosine_similarity,
    compute_mad_scores,
    temperature_scaled_softmax,
    project_capped_simplex,
)


def _spectral_filter(vectors: np.ndarray, keep_ratio: float = 0.9) -> np.ndarray:
    U, S, Vt = np.linalg.svd(vectors, full_matrices=False)
    k = max(1, int(keep_ratio * len(S)))
    S_filtered = S.copy()
    S_filtered[k:] = 0.0
    return (U * S_filtered) @ Vt


class SSFGAggregator(fl.server.strategy.Strategy):
    """
    Variant C — Sparse Spectral Filter + Gradient aggregation.

    Extends Variant A by applying SVD-based spectral filtering to the
    stacked final-layer weight matrix before computing cosine similarity.
    This suppresses low-rank adversarial perturbations that bypass MAD.
    """

    def __init__(self, initial_parameters: Parameters):
        super().__init__()
        self.initial_parameters = initial_parameters

        defense = CONFIG["defense"]
        federated = CONFIG["federated"]

        self.ema_momentum = defense["ema_momentum"]
        self.temperature = defense["temperature"]
        self.mad_threshold = defense["mad_threshold"]
        self.initial_reputation = defense["initial_reputation"]

        effective_K = federated["clients_per_round"]
        b_f = int(defense["max_byzantine_fraction"] * effective_K)
        self.cap_t = 1.0 / max(effective_K - b_f, 1)

        self.reputation_scores: dict = {}

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def _update_ema_reputation(self, client_ids: list, scores: np.ndarray) -> np.ndarray:
        for cid, score in zip(client_ids, scores):
            prev = self.reputation_scores.get(cid, self.initial_reputation)
            self.reputation_scores[cid] = self.ema_momentum * prev + (1 - self.ema_momentum) * float(score)
        return np.array([self.reputation_scores[cid] for cid in client_ids])

    def aggregate_fit(self, server_round, results, failures):
        try:
            if not results:
                return None, {}

            client_ids = [str(proxy.cid) for proxy, _ in results]
            all_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

            all_ndarrays = clip_to_median_norm(all_ndarrays)

            # Extract final layers and apply spectral filter
            final_layers = np.stack([extract_final_layer(nd) for nd in all_ndarrays])
            filtered_layers = _spectral_filter(final_layers)

            # Cosine similarity + MAD on filtered representation
            sim_matrix = compute_layer_wise_cosine_similarity(filtered_layers)
            mad_scores = compute_mad_scores(sim_matrix)

            n_flagged = int((mad_scores < self.mad_threshold).sum())
            logging.info(f"[SSFG] Round {server_round}: {n_flagged}/{len(client_ids)} flagged.")

            reputation = self._update_ema_reputation(client_ids, mad_scores)
            trust_weights = temperature_scaled_softmax(reputation, self.temperature)
            final_weights = project_capped_simplex(trust_weights, self.cap_t)

            global_params = [
                np.average(
                    np.stack([nd[i] for nd in all_ndarrays]),
                    axis=0,
                    weights=final_weights,
                )
                for i in range(len(all_ndarrays[0]))
            ]

            metrics = {
                "round": server_round,
                "flagged": n_flagged,
                "min_trust": float(final_weights.min()),
                "max_trust": float(final_weights.max()),
            }

            return ndarrays_to_parameters(global_params), metrics

        except Exception as e:
            raise FLIDSException(e, sys)

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None

import sys
import numpy as np
import flwr as fl
from flwr.common import Parameters, FitRes, ndarrays_to_parameters, parameters_to_ndarrays

from src.configs.config import CONFIG
from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.components.server.aggregator import (
    extract_final_layer,
    compute_layer_wise_cosine_similarity,
    compute_mad_scores,
    temperature_scaled_softmax,
    project_capped_simplex,
)
from src.components.server.ssfg_aggregator import _spectral_filter


class TriageAggregator(fl.server.strategy.Strategy):
    """
    MSFT — Multi-Stage Final-Layer Triage.

    Stage 1: Cosine + MAD on all clients.
    Stage 2: SVD only on suspicious subset (MAD < soft_threshold).
    Stage 3: Merge scores → EMA reputation → capped simplex weights.
    Stage 4: Weighted aggregation across all model layers.
    """

    def __init__(self, initial_parameters: Parameters):
        super().__init__()
        self.initial_parameters = initial_parameters

        d = CONFIG["defense"]
        f = CONFIG["federated"]

        self.ema_momentum = d["ema_momentum"]
        self.temperature = d["temperature"]
        self.mad_threshold = d["mad_threshold"]
        self.soft_threshold = d["triage_soft_threshold"]
        self.svd_keep_ratio = d["svd_keep_ratio"]
        self.initial_rep = d["initial_reputation"]

        K = f["clients_per_round"]
        b = int(d["max_byzantine_fraction"] * K)
        self.cap_t = 1.0 / max(K - b, 1)

        self.reputation: dict = {}

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def _ema(self, ids, scores):
        for cid, s in zip(ids, scores):
            prev = self.reputation.get(cid, self.initial_rep)
            self.reputation[cid] = self.ema_momentum * prev + (1 - self.ema_momentum) * float(s)
        return np.array([self.reputation[cid] for cid in ids])

    def aggregate_fit(self, server_round, results, failures):
        try:
            if not results:
                return None, {}

            ids = [str(p.cid) for p, _ in results]
            weights = [parameters_to_ndarrays(r.parameters) for _, r in results]

            # Stage 1: cosine + MAD on all clients
            layers = np.stack([extract_final_layer(w) for w in weights])
            sim = compute_layer_wise_cosine_similarity(layers)
            mad = compute_mad_scores(sim)

            # Stage 2: re-score suspicious clients with SVD on their submatrix only
            scores = mad.copy()
            suspicious = np.where(mad < self.soft_threshold)[0]
            if len(suspicious) >= 2:
                sub = _spectral_filter(layers[suspicious], self.svd_keep_ratio)
                sub_sim = compute_layer_wise_cosine_similarity(sub)
                sub_mad = compute_mad_scores(sub_sim)
                scores[suspicious] = sub_mad
                logging.info(f"[Triage] Round {server_round}: {len(suspicious)} clients sent to SVD stage.")

            flagged = int((scores < self.mad_threshold).sum())
            logging.info(f"[Triage] Round {server_round}: {flagged}/{len(ids)} flagged after triage.")

            # Stage 3: EMA → softmax → capped simplex
            rep = self._ema(ids, scores)
            trust = temperature_scaled_softmax(rep, self.temperature)
            final = project_capped_simplex(trust, self.cap_t)

            # Stage 4: weighted aggregation across all layers
            agg = [
                np.average(np.stack([w[i] for w in weights]), axis=0, weights=final)
                for i in range(len(weights[0]))
            ]

            return ndarrays_to_parameters(agg), {
                "round": server_round,
                "flagged": flagged,
                "suspicious": len(suspicious),
                "min_trust": float(final.min()),
                "max_trust": float(final.max()),
            }

        except Exception as e:
            raise FLIDSException(e, sys)

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None

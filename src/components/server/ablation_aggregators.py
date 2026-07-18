import sys
import numpy as np
import flwr as fl
from flwr.common import Parameters, FitRes, ndarrays_to_parameters, parameters_to_ndarrays

from src.configs.config import CONFIG
from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.components.server.aggregator import (
    compute_layer_wise_cosine_similarity,
    compute_mad_scores,
    temperature_scaled_softmax,
    project_capped_simplex,
    extract_final_layer,
)


def _flat_model(ndarrays):
    return np.concatenate([a.flatten() for a in ndarrays])


class _BaseAblation(fl.server.strategy.Strategy):
    def __init__(self, initial_parameters: Parameters):
        super().__init__()
        self.initial_parameters = initial_parameters

        d = CONFIG["defense"]
        f = CONFIG["federated"]

        self.ema_momentum = d["ema_momentum"]
        self.temperature = d["temperature"]
        self.mad_threshold = d["mad_threshold"]
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

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


class FullModelCosineAggregator(_BaseAblation):
    """
    Ablation 1: Cosine similarity on the FULL flattened model instead of final layer.
    Expected to fail on Non-IID data — benign clients with rare traffic look like outliers.
    Proves that final-layer-only scoring is the correct design choice.
    """

    def aggregate_fit(self, server_round, results, failures):
        try:
            if not results:
                return None, {}

            ids = [str(p.cid) for p, _ in results]
            weights = [parameters_to_ndarrays(r.parameters) for _, r in results]

            vecs = np.stack([_flat_model(w) for w in weights])
            sim = compute_layer_wise_cosine_similarity(vecs)
            mad = compute_mad_scores(sim)

            rep = self._ema(ids, mad)
            trust = temperature_scaled_softmax(rep, self.temperature)
            final = project_capped_simplex(trust, self.cap_t)

            agg = [
                np.average(np.stack([w[i] for w in weights]), axis=0, weights=final)
                for i in range(len(weights[0]))
            ]

            flagged = int((mad < self.mad_threshold).sum())
            logging.info(f"[FullModelCosine] Round {server_round}: {flagged}/{len(ids)} flagged.")

            return ndarrays_to_parameters(agg), {"round": server_round, "flagged": flagged}

        except Exception as e:
            raise FLIDSException(e, sys)


class FinalLayerNoSimplexAggregator(_BaseAblation):
    """
    Ablation 2: Final-layer cosine + MAD but using plain softmax instead of capped simplex.
    Expected to underperform — without the simplex cap, extreme attackers can still dominate.
    Proves the capped simplex projection is a necessary component.
    """

    def aggregate_fit(self, server_round, results, failures):
        try:
            if not results:
                return None, {}

            ids = [str(p.cid) for p, _ in results]
            weights = [parameters_to_ndarrays(r.parameters) for _, r in results]

            layers = np.stack([extract_final_layer(w) for w in weights])
            sim = compute_layer_wise_cosine_similarity(layers)
            mad = compute_mad_scores(sim)

            rep = self._ema(ids, mad)
            # Plain softmax — no capped simplex (this is the ablated component)
            final = temperature_scaled_softmax(rep, self.temperature)

            agg = [
                np.average(np.stack([w[i] for w in weights]), axis=0, weights=final)
                for i in range(len(weights[0]))
            ]

            flagged = int((mad < self.mad_threshold).sum())
            logging.info(f"[FinalLayerNoSimplex] Round {server_round}: {flagged}/{len(ids)} flagged.")

            return ndarrays_to_parameters(agg), {"round": server_round, "flagged": flagged}

        except Exception as e:
            raise FLIDSException(e, sys)

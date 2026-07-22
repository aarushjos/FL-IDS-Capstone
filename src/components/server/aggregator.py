import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from scipy.spatial.distance import pdist, squareform

from src.configs.config import CONFIG
from src.exception.exception import FLIDSException
from src.logging.logger import logging


EvaluateMetricsAggregationFn = Callable[
    [List[Tuple[int, Dict[str, Scalar]]]],
    Dict[str, Scalar],
]


def extract_final_layer(
    ndarrays: List[np.ndarray],
) -> np.ndarray:
    """
    Extract and flatten the final classification-layer weights.

    The final two arrays are expected to be:
        output.weight
        output.bias
    """
    weight = ndarrays[-2]
    return weight.flatten()


def compute_layer_wise_cosine_similarity(
    final_layers: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between client final layers.
    """
    distances = squareform(
        pdist(
            final_layers,
            metric="cosine",
        )
    )

    similarities = 1.0 - distances

    # Protect against NaN values from zero vectors.
    return np.nan_to_num(
        similarities,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )


def compute_mad_scores(
    sim_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate robust MAD-based consensus scores.
    """
    consensus = np.median(
        sim_matrix,
        axis=1,
    )

    median_consensus = np.median(consensus)

    mad = np.median(
        np.abs(
            consensus - median_consensus
        )
    )

    return (
        0.6745
        * (consensus - median_consensus)
        / (mad + 1e-9)
    )


def temperature_scaled_softmax(
    scores: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Convert reputation scores into normalized trust weights.
    """
    scaled = scores * temperature
    shifted = scaled - np.max(scaled)

    exp_values = np.exp(shifted)

    return exp_values / (
        exp_values.sum() + 1e-9
    )


def project_capped_simplex(
    values: np.ndarray,
    cap_t: float,
) -> np.ndarray:
    """
    Project values onto a capped unit simplex:

        sum(weights) = 1
        0 <= weight_i <= cap_t
    """

    def feasibility(gamma: float) -> float:
        return (
            np.clip(
                values - gamma,
                0.0,
                cap_t,
            ).sum()
            - 1.0
        )

    lower = values.min() - 1.0
    upper = values.max()

    for _ in range(64):
        midpoint = (
            lower + upper
        ) / 2.0

        if feasibility(midpoint) > 0:
            lower = midpoint
        else:
            upper = midpoint

    weights = np.clip(
        values - upper,
        0.0,
        cap_t,
    )

    total = weights.sum()

    if total > 1e-9:
        weights = weights / total

    return weights


class RobustFLIDSStrategy(
    fl.server.strategy.Strategy
):
    """
    AL-CMT: Adaptive Layer-Wise Cosine-MAD Trust aggregation.

    Per-round process:

    1. Deserialize client models.
    2. Extract each client's final classification layer.
    3. Compute pairwise cosine similarities.
    4. Calculate robust MAD scores.
    5. Update persistent EMA reputation scores.
    6. Apply temperature-scaled softmax.
    7. Project weights onto a capped simplex.
    8. Aggregate every model layer using the trust weights.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        evaluate_metrics_aggregation_fn: Optional[
            EvaluateMetricsAggregationFn
        ] = None,
    ):
        super().__init__()

        self.initial_parameters = (
            initial_parameters
        )

        self.evaluate_metrics_aggregation_fn = (
            evaluate_metrics_aggregation_fn
        )

        defense = CONFIG["defense"]
        federated = CONFIG["federated"]

        self.ema_momentum = float(
            defense["ema_momentum"]
        )

        self.temperature = float(
            defense["temperature"]
        )

        self.mad_threshold = float(
            defense["mad_threshold"]
        )

        self.initial_reputation = float(
            defense["initial_reputation"]
        )

        self.max_byzantine_fraction = float(
            defense["max_byzantine_fraction"]
        )

        self.clients_per_round = int(
            federated["clients_per_round"]
        )

        self.num_rounds = int(
            federated["num_rounds"]
        )

        self.local_epochs = int(
            federated["local_epochs"]
        )

        self.lr = float(
            federated["learning_rate"]
        )

        self.batch_size = int(
            federated["batch_size"]
        )

        effective_clients = (
            self.clients_per_round
        )

        expected_byzantine_clients = int(
            self.max_byzantine_fraction
            * effective_clients
        )

        self.cap_t = 1.0 / max(
            effective_clients
            - expected_byzantine_clients,
            1,
        )

        self.reputation_scores: Dict[
            str,
            float,
        ] = {}

        logging.info(
            "[Aggregator] AL-CMT initialized — "
            f"cap_t={self.cap_t:.4f}, "
            f"temperature={self.temperature}, "
            f"ema_momentum={self.ema_momentum}"
        )

    def _update_ema_reputation(
        self,
        client_ids: List[str],
        mad_scores: np.ndarray,
    ) -> np.ndarray:
        momentum = self.ema_momentum
        threshold = self.mad_threshold

        for index, client_id in enumerate(
            client_ids
        ):
            if (
                client_id
                not in self.reputation_scores
            ):
                self.reputation_scores[
                    client_id
                ] = self.initial_reputation

            reward = (
                mad_scores[index]
                if mad_scores[index] >= threshold
                else threshold
            )

            old_reputation = (
                self.reputation_scores[
                    client_id
                ]
            )

            self.reputation_scores[
                client_id
            ] = (
                momentum * old_reputation
                + (1.0 - momentum) * reward
            )

        return np.array(
            [
                self.reputation_scores[
                    client_id
                ]
                for client_id in client_ids
            ],
            dtype=np.float64,
        )

    def initialize_parameters(
        self,
        client_manager: (
            fl.server.client_manager.ClientManager
        ),
    ) -> Optional[Parameters]:
        parameters = self.initial_parameters

        # Return initial parameters only once.
        self.initial_parameters = None

        return parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: (
            fl.server.client_manager.ClientManager
        ),
    ) -> List[
        Tuple[
            ClientProxy,
            fl.common.FitIns,
        ]
    ]:
        config = {
            "server_round": server_round,
            "local_epochs": self.local_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
        }

        fit_instructions = fl.common.FitIns(
            parameters,
            config,
        )

        clients = client_manager.sample(
            num_clients=self.clients_per_round,
            min_num_clients=(
                self.clients_per_round
            ),
        )

        return [
            (
                client,
                fit_instructions,
            )
            for client in clients
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[
            Tuple[
                ClientProxy,
                FitRes,
            ]
        ],
        failures: List[
            Union[
                Tuple[
                    ClientProxy,
                    FitRes,
                ],
                BaseException,
            ]
        ],
    ) -> Tuple[
        Optional[Parameters],
        Dict[str, Scalar],
    ]:
        try:
            if not results:
                logging.warning(
                    "[Aggregator] "
                    f"Round {server_round}: "
                    "no client results received."
                )

                return None, {}

            if failures:
                logging.warning(
                    "[Aggregator] "
                    f"Round {server_round}: "
                    f"{len(failures)} client(s) failed."
                )

            client_ids = [
                str(client_proxy.cid)
                for client_proxy, _ in results
            ]

            all_ndarrays = [
                parameters_to_ndarrays(
                    fit_result.parameters
                )
                for _, fit_result in results
            ]

            final_layers = np.stack(
                [
                    extract_final_layer(
                        client_parameters
                    )
                    for client_parameters
                    in all_ndarrays
                ]
            )

            similarity_matrix = (
                compute_layer_wise_cosine_similarity(
                    final_layers
                )
            )

            mad_scores = compute_mad_scores(
                similarity_matrix
            )

            number_flagged = int(
                (
                    mad_scores
                    < self.mad_threshold
                ).sum()
            )

            logging.info(
                "[Aggregator] "
                f"Round {server_round}: "
                f"{number_flagged}/"
                f"{len(client_ids)} clients "
                "flagged by MAD."
            )

            reputation_scores = (
                self._update_ema_reputation(
                    client_ids,
                    mad_scores,
                )
            )

            trust_weights = (
                temperature_scaled_softmax(
                    reputation_scores,
                    self.temperature,
                )
            )

            final_weights = (
                project_capped_simplex(
                    trust_weights,
                    self.cap_t,
                )
            )

            if (
                not np.isfinite(
                    final_weights
                ).all()
                or final_weights.sum() <= 0
            ):
                logging.warning(
                    "[Aggregator] Invalid trust "
                    "weights detected. Falling back "
                    "to uniform aggregation."
                )

                final_weights = np.ones(
                    len(results),
                    dtype=np.float64,
                ) / len(results)

            logging.info(
                "[Aggregator] "
                f"Round {server_round}: "
                "trust weights "
                f"min={final_weights.min():.4f}, "
                f"max={final_weights.max():.4f}, "
                "zero_count="
                f"{int((final_weights == 0).sum())}"
            )

            global_parameters = [
                np.average(
                    np.stack(
                        [
                            client_parameters[
                                layer_index
                            ]
                            for client_parameters
                            in all_ndarrays
                        ]
                    ),
                    axis=0,
                    weights=final_weights,
                )
                for layer_index in range(
                    len(all_ndarrays[0])
                )
            ]

            aggregated_parameters = (
                ndarrays_to_parameters(
                    global_parameters
                )
            )

            aggregation_metrics: Dict[
                str,
                Scalar,
            ] = {
                "round": server_round,
                "clients": len(client_ids),
                "flagged": number_flagged,
                "min_trust": float(
                    final_weights.min()
                ),
                "max_trust": float(
                    final_weights.max()
                ),
            }

            return (
                aggregated_parameters,
                aggregation_metrics,
            )

        except Exception as error:
            raise FLIDSException(
                error,
                sys,
            )

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: (
            fl.server.client_manager.ClientManager
        ),
    ) -> List[
        Tuple[
            ClientProxy,
            fl.common.EvaluateIns,
        ]
    ]:
        evaluate_instructions = (
            fl.common.EvaluateIns(
                parameters,
                {
                    "server_round": server_round,
                },
            )
        )

        clients = client_manager.sample(
            num_clients=self.clients_per_round,
            min_num_clients=(
                self.clients_per_round
            ),
        )

        return [
            (
                client,
                evaluate_instructions,
            )
            for client in clients
        ]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[
            Tuple[
                ClientProxy,
                EvaluateRes,
            ]
        ],
        failures: List[
            Union[
                Tuple[
                    ClientProxy,
                    EvaluateRes,
                ],
                BaseException,
            ]
        ],
    ) -> Tuple[
        Optional[float],
        Dict[str, Scalar],
    ]:
        if not results:
            return None, {}

        if failures:
            logging.warning(
                "[Aggregator] "
                f"Evaluation round {server_round}: "
                f"{len(failures)} client(s) failed."
            )

        total_examples = sum(
            evaluate_result.num_examples
            for _, evaluate_result in results
        )

        if total_examples <= 0:
            return None, {}

        weighted_loss = sum(
            evaluate_result.loss
            * evaluate_result.num_examples
            for _, evaluate_result in results
        ) / total_examples

        aggregated_metrics: Dict[
            str,
            Scalar,
        ] = {
            "round": server_round,
        }

        if (
            self.evaluate_metrics_aggregation_fn
            is not None
        ):
            client_metrics = [
                (
                    evaluate_result.num_examples,
                    evaluate_result.metrics,
                )
                for _, evaluate_result in results
            ]

            calculated_metrics = (
                self.evaluate_metrics_aggregation_fn(
                    client_metrics
                )
            )

            aggregated_metrics.update(
                calculated_metrics
            )
        else:
            logging.warning(
                "[Aggregator] No evaluation metric "
                "aggregation function was supplied."
            )

        return (
            float(weighted_loss),
            aggregated_metrics,
        )

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[
        Tuple[
            float,
            Dict[str, Scalar],
        ]
    ]:
        return None
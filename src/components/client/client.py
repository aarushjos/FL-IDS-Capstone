import sys
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.components.client.attacker import (
    flip_labels,
    inject_backdoor_trigger,
    scale_gradient_to_norm,
)
from src.components.model.model import (
    MLPClassifier,
    get_model_parameters,
    set_model_parameters,
)
from src.exception.exception import FLIDSException
from src.logging.logger import logging


class FLIDSClient(fl.client.NumPyClient):
    """
    Flower client representing one IoT edge gateway.

    Supported attacks:
    - Label flipping: modifies labels during local training.
    - Sign flipping: reverses and scales the client's model update.
    - Backdoor: injects a trigger into raw local training samples.

    Raw client data is never sent to the server.
    """

    def __init__(
        self,
        cid: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: MLPClassifier,
        config: Optional[Dict[str, Any]] = None,
        X_train_raw: Optional[np.ndarray] = None,
        y_train_raw: Optional[np.ndarray] = None,
    ):
        self.cid = str(cid)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.config = config or {}

        # Required for backdoor injection and optional class-weight calculation.
        self.X_train_raw = X_train_raw
        self.y_train_raw = y_train_raw

        self.device = torch.device(
            self.config.get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )

        self.model.to(self.device)

        self.local_epochs = int(
            self.config.get(
                "local_epochs",
                1,
            )
        )

        self.lr = float(
            self.config.get(
                "lr",
                1e-3,
            )
        )

        self.weight_decay = float(
            self.config.get(
                "weight_decay",
                0.0,
            )
        )

        self.is_poisoned = bool(
            self.config.get(
                "is_poisoned",
                False,
            )
        )

        self.weight_cap = float(
            self.config.get(
                "weight_cap",
                10.0,
            )
        )

        self.num_classes = int(
            self.config.get(
                "num_classes",
                15,
            )
        )

        # Set to None or 0 to disable clipping.
        self.gradient_clip_norm = self.config.get(
            "gradient_clip_norm",
            None,
        )

    def get_parameters(
        self,
        config,
    ) -> List[np.ndarray]:
        return get_model_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config,
    ) -> Tuple[
        List[np.ndarray],
        int,
        Dict[str, float],
    ]:
        try:
            # Load the latest global model received from the server.
            set_model_parameters(
                self.model,
                parameters,
            )

            # Save a copy of the global model.
            # Sign flipping operates on:
            # local_update = local_parameters - global_parameters
            global_parameters = [
                np.array(
                    layer,
                    copy=True,
                )
                for layer in parameters
            ]

            # ----------------------------------------------------------
            # Attack configuration
            # ----------------------------------------------------------
            server_round = int(
                config.get(
                    "server_round",
                    0,
                )
            )

            attack_start = int(
                self.config.get(
                    "attack_start_round",
                    1,
                )
            )

            attack_type = str(
                self.config.get(
                    "attack_type",
                    "none",
                )
            ).lower()

            attack_active = (
                self.is_poisoned
                and server_round >= attack_start
                and attack_type != "none"
            )

            source_class = int(
                self.config.get(
                    "source_class",
                    4,
                )
            )

            target_class = int(
                self.config.get(
                    "target_class",
                    0,
                )
            )

            # ----------------------------------------------------------
            # Backdoor DataLoader
            # ----------------------------------------------------------
            active_loader = self.train_loader

            if (
                attack_active
                and attack_type in ("backdoor", "both")
            ):
                if (
                    self.X_train_raw is not None
                    and self.y_train_raw is not None
                ):
                    trigger_indices = self.config.get(
                        "trigger_feature_idx",
                        [0, 5],
                    )

                    trigger_values = self.config.get(
                        "trigger_values",
                        [999999, 1],
                    )

                    injection_ratio = float(
                        self.config.get(
                            "inject_ratio",
                            0.1,
                        )
                    )

                    X_poisoned, y_poisoned = (
                        inject_backdoor_trigger(
                            self.X_train_raw,
                            self.y_train_raw,
                            trigger_indices,
                            trigger_values,
                            injection_ratio,
                        )
                    )

                    poisoned_dataset = TensorDataset(
                        torch.tensor(
                            X_poisoned,
                            dtype=torch.float32,
                        ),
                        torch.tensor(
                            y_poisoned,
                            dtype=torch.long,
                        ),
                    )

                    active_loader = DataLoader(
                        poisoned_dataset,
                        batch_size=(
                            self.train_loader.batch_size
                            or 64
                        ),
                        shuffle=True,
                        drop_last=True,
                    )

                    logging.info(
                        f"[Client {self.cid}] "
                        "Backdoor DataLoader rebuilt."
                    )
                else:
                    logging.warning(
                        f"[Client {self.cid}] "
                        "Backdoor requested, but raw "
                        "training arrays were not provided."
                    )

            # ----------------------------------------------------------
            # Optimizer
            # ----------------------------------------------------------
            self.model.train()

            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

            # ----------------------------------------------------------
            # Class-weighted loss
            # ----------------------------------------------------------
            if self.y_train_raw is not None:
                y_for_weights = np.asarray(
                    self.y_train_raw,
                    dtype=np.int64,
                )
            else:
                label_batches = [
                    y_batch
                    .detach()
                    .cpu()
                    .numpy()
                    for _, y_batch in active_loader
                ]

                if label_batches:
                    y_for_weights = np.concatenate(
                        label_batches
                    )
                else:
                    y_for_weights = np.array(
                        [],
                        dtype=np.int64,
                    )

            classes = np.arange(
                self.num_classes
            )

            try:
                if len(y_for_weights) == 0:
                    raise ValueError(
                        "No labels available for "
                        "class-weight calculation."
                    )

                raw_weights = compute_class_weight(
                    class_weight="balanced",
                    classes=classes,
                    y=y_for_weights,
                )
            except Exception:
                # Some Non-IID clients may not contain every class.
                # Fall back to an unweighted loss for those clients.
                raw_weights = np.ones(
                    self.num_classes,
                    dtype=np.float32,
                )

            capped_weights = np.clip(
                raw_weights,
                a_min=None,
                a_max=self.weight_cap,
            )

            weight_tensor = torch.tensor(
                capped_weights,
                dtype=torch.float32,
                device=self.device,
            )

            loss_fn = nn.CrossEntropyLoss(
                weight=weight_tensor,
            )

            total_loss = 0.0
            total_examples = 0
            correct = 0

            # Counts label-flip operations across all local epochs.
            total_labels_flipped = 0
            total_source_labels_seen = 0

            # ----------------------------------------------------------
            # Local training
            # ----------------------------------------------------------
            for _ in range(self.local_epochs):
                for x, y in active_loader:
                    x = x.to(
                        self.device
                    ).float()

                    y = y.to(
                        self.device
                    ).long()

                    # Label flipping is applied only to malicious
                    # clients after attack_start_round.
                    if (
                        attack_active
                        and attack_type
                        in ("label_flip", "both")
                    ):
                        original_y = y.clone()

                        source_count = int(
                            (
                                original_y
                                == source_class
                            )
                            .sum()
                            .item()
                        )

                        total_source_labels_seen += (
                            source_count
                        )

                        flipped_y = flip_labels(
                            original_y
                            .detach()
                            .cpu()
                            .numpy(),
                            source_class,
                            target_class,
                        )

                        y = torch.tensor(
                            flipped_y,
                            dtype=torch.long,
                            device=self.device,
                        )

                        batch_flipped = int(
                            (
                                original_y != y
                            )
                            .sum()
                            .item()
                        )

                        total_labels_flipped += (
                            batch_flipped
                        )

                    optimizer.zero_grad()

                    logits = self.model(x)

                    loss = loss_fn(
                        logits,
                        y,
                    )

                    loss.backward()

                    # Optional gradient clipping.
                    if (
                        self.gradient_clip_norm
                        is not None
                    ):
                        clip_norm = float(
                            self.gradient_clip_norm
                        )

                        if clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=clip_norm,
                            )

                    optimizer.step()

                    batch_size = y.size(0)

                    total_loss += (
                        loss.item()
                        * batch_size
                    )

                    total_examples += (
                        batch_size
                    )

                    predictions = torch.argmax(
                        logits,
                        dim=1,
                    )

                    correct += int(
                        (
                            predictions == y
                        )
                        .sum()
                        .item()
                    )

            if (
                attack_active
                and attack_type
                in ("label_flip", "both")
            ):
                print(
                    f"Client {self.cid} | "
                    f"Round {server_round} | "
                    f"Label flip "
                    f"{source_class}->{target_class} | "
                    f"Flipped "
                    f"{total_labels_flipped}/"
                    f"{total_source_labels_seen} "
                    f"source-label occurrences "
                    f"across {self.local_epochs} "
                    "local epoch(s)",
                    flush=True,
                )

                logging.info(
                    f"[Client {self.cid}] "
                    f"Round {server_round}: "
                    f"label flip "
                    f"{source_class}->{target_class}, "
                    f"flipped={total_labels_flipped}, "
                    f"source_seen="
                    f"{total_source_labels_seen}."
                )

            average_loss = (
                total_loss
                / max(
                    total_examples,
                    1,
                )
            )

            train_accuracy = (
                correct
                / max(
                    total_examples,
                    1,
                )
            )

            # Parameters after honest or label-poisoned
            # local training.
            local_parameters = (
                get_model_parameters(
                    self.model
                )
            )

            # ----------------------------------------------------------
            # Sign-flipping attack
            # ----------------------------------------------------------
            if (
                attack_active
                and attack_type == "sign_flip"
            ):
                sign_flip_scale = float(
                    self.config.get(
                        "sign_flip_scale",
                        1.0,
                    )
                )

                print(
                    f"Client {self.cid} | "
                    f"Round {server_round} | "
                    "Applying SIGN FLIP",
                    flush=True,
                )

                malicious_parameters = []

                for (
                    local_layer,
                    global_layer,
                ) in zip(
                    local_parameters,
                    global_parameters,
                ):
                    local_update = (
                        local_layer
                        - global_layer
                    )

                    malicious_update = (
                        -sign_flip_scale
                        * local_update
                    )

                    malicious_layer = (
                        global_layer
                        + malicious_update
                    )

                    malicious_parameters.append(
                        malicious_layer.astype(
                            local_layer.dtype,
                            copy=False,
                        )
                    )

                local_parameters = (
                    malicious_parameters
                )

                logging.info(
                    f"[Client {self.cid}] "
                    "Sign-flipping attack applied "
                    f"with scale={sign_flip_scale}."
                )

            # ----------------------------------------------------------
            # Backdoor norm-scaling option
            # ----------------------------------------------------------
            if (
                attack_active
                and attack_type
                in ("backdoor", "both")
            ):
                scale_to_benign_norm = bool(
                    self.config.get(
                        "scale_to_benign_norm",
                        False,
                    )
                )

                target_norm = float(
                    self.config.get(
                        "benign_norm_target",
                        1.0,
                    )
                )

                if scale_to_benign_norm:
                    local_parameters = (
                        scale_gradient_to_norm(
                            local_parameters,
                            target_norm,
                        )
                    )

            metrics = {
                "train_loss": float(
                    average_loss
                ),
                "train_accuracy": float(
                    train_accuracy
                ),
                "cid": (
                    float(self.cid)
                    if self.cid.isdigit()
                    else -1.0
                ),
                "is_poisoned": float(
                    self.is_poisoned
                ),
                "attack_active": float(
                    attack_active
                ),
                "labels_flipped": float(
                    total_labels_flipped
                ),
                "source_labels_seen": float(
                    total_source_labels_seen
                ),
            }

            return (
                local_parameters,
                total_examples,
                metrics,
            )

        except Exception as error:
            raise FLIDSException(
                error,
                sys,
            )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config,
    ) -> Tuple[
        float,
        int,
        Dict[str, float],
    ]:
        try:
            set_model_parameters(
                self.model,
                parameters,
            )

            self.model.eval()

            loss_fn = nn.CrossEntropyLoss()

            total_loss = 0.0
            total_examples = 0
            correct = 0

            all_predictions: List[int] = []
            all_targets: List[int] = []

            with torch.no_grad():
                for x, y in self.val_loader:
                    x = x.to(
                        self.device
                    ).float()

                    y = y.to(
                        self.device
                    ).long()

                    logits = self.model(x)

                    loss = loss_fn(
                        logits,
                        y,
                    )

                    batch_size = y.size(0)

                    total_loss += (
                        loss.item()
                        * batch_size
                    )

                    total_examples += (
                        batch_size
                    )

                    predictions = torch.argmax(
                        logits,
                        dim=1,
                    )

                    correct += int(
                        (
                            predictions == y
                        )
                        .sum()
                        .item()
                    )

                    all_predictions.extend(
                        predictions
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    all_targets.extend(
                        y
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )

            average_loss = (
                total_loss
                / max(
                    total_examples,
                    1,
                )
            )

            accuracy = (
                correct
                / max(
                    total_examples,
                    1,
                )
            )

            macro_f1 = f1_score(
                all_targets,
                all_predictions,
                labels=list(
                    range(
                        self.num_classes
                    )
                ),
                average="macro",
                zero_division=0,
            )

            weighted_f1 = f1_score(
                all_targets,
                all_predictions,
                labels=list(
                    range(
                        self.num_classes
                    )
                ),
                average="weighted",
                zero_division=0,
            )

            # For a source_class -> target_class attack,
            # measure recall of the attacked SOURCE class.
            attacked_class = int(
                self.config.get(
                    "source_class",
                    4,
                )
            )

            attacked_class_total = sum(
                target == attacked_class
                for target in all_targets
            )

            attacked_class_correct = sum(
                target == attacked_class
                and prediction
                == attacked_class
                for target, prediction
                in zip(
                    all_targets,
                    all_predictions,
                )
            )

            attacked_class_recall = (
                attacked_class_correct
                / attacked_class_total
                if attacked_class_total > 0
                else 0.0
            )

            metrics = {
                "val_accuracy": float(
                    accuracy
                ),
                "macro_f1": float(
                    macro_f1
                ),
                "weighted_f1": float(
                    weighted_f1
                ),

                # Kept under the existing metric names so your
                # weighted_average() function continues to work.
                # This now represents attacked source-class recall.
                "target_class_accuracy": float(
                    attacked_class_recall
                ),
                "target_class_samples": float(
                    attacked_class_total
                ),

                "cid": (
                    float(self.cid)
                    if self.cid.isdigit()
                    else -1.0
                ),
            }

            return (
                float(average_loss),
                total_examples,
                metrics,
            )

        except Exception as error:
            raise FLIDSException(
                error,
                sys,
            )
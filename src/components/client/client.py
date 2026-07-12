import sys
from typing import Dict, List, Tuple, Optional, Any

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.components.model.model import (
    MLPClassifier,
    get_model_parameters,
    set_model_parameters,
)
from src.components.client.attacker import (
    flip_labels,
    inject_backdoor_trigger,
    scale_gradient_to_norm,
)


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

        # Required only for backdoor injection and class-weight calculation.
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
            self.config.get("local_epochs", 1)
        )
        self.lr = float(
            self.config.get("lr", 1e-3)
        )
        self.weight_decay = float(
            self.config.get("weight_decay", 0.0)
        )

        self.is_poisoned = bool(
            self.config.get("is_poisoned", False)
        )
        self.weight_cap = float(
            self.config.get("weight_cap", 10.0)
        )
        self.num_classes = int(
            self.config.get("num_classes", 27)
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
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:

        # Load the latest global model received from the server.
        set_model_parameters(self.model, parameters)

        # Save an untouched copy of the global model.
        # Sign flipping operates on:
        # local_update = local_parameters - global_parameters
        global_parameters = [
            np.array(layer, copy=True)
            for layer in parameters
        ]

        # ── Attack configuration ──────────────────────────────────────
        server_round = int(
            config.get("server_round", 0)
        )
        attack_start = int(
            self.config.get("attack_start_round", 1)
        )
        attack_type = str(
            self.config.get("attack_type", "none")
        ).lower()

        attack_active = (
            self.is_poisoned
            and server_round >= attack_start
            and attack_type != "none"
        )

        source_class = int(
            self.config.get("source_class", 1)
        )
        target_class = int(
            self.config.get("target_class", 0)
        )

        # ── Backdoor DataLoader ───────────────────────────────────────
        active_loader = self.train_loader

        if attack_active and attack_type in ("backdoor", "both"):
            if (
                self.X_train_raw is not None
                and self.y_train_raw is not None
            ):
                trigger_idx = self.config.get(
                    "trigger_feature_idx",
                    [0, 5],
                )
                trigger_vals = self.config.get(
                    "trigger_values",
                    [999999, 1],
                )
                inject_ratio = float(
                    self.config.get("inject_ratio", 0.1)
                )

                X_poisoned, y_poisoned = inject_backdoor_trigger(
                    self.X_train_raw,
                    self.y_train_raw,
                    trigger_idx,
                    trigger_vals,
                    inject_ratio,
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
                    batch_size=self.train_loader.batch_size or 64,
                    shuffle=True,
                    drop_last=True,
                )

                logging.info(
                    f"[Client {self.cid}] "
                    f"Backdoor DataLoader rebuilt."
                )
            else:
                logging.warning(
                    f"[Client {self.cid}] Backdoor requested, "
                    f"but raw arrays were not provided."
                )

        # ── Optimizer ─────────────────────────────────────────────────
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # ── Class-weighted loss ───────────────────────────────────────
        if self.y_train_raw is not None:
            y_for_weights = self.y_train_raw
        else:
            label_batches = [
                y_batch.detach().cpu().numpy()
                for _, y_batch in active_loader
            ]

            if label_batches:
                y_for_weights = np.concatenate(label_batches)
            else:
                y_for_weights = np.array([], dtype=np.int64)

        classes = np.arange(self.num_classes)

        try:
            if len(y_for_weights) == 0:
                raise ValueError("No labels available.")

            raw_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_for_weights,
            )
        except Exception:
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
            weight=weight_tensor
        )

        total_loss = 0.0
        total_examples = 0
        correct = 0

        # ── Local training ────────────────────────────────────────────
        for _ in range(self.local_epochs):
            for x, y in active_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()

                # Label-flipping is a data-poisoning attack.
                if (
                    attack_active
                    and attack_type in ("label_flip", "both")
                ):
                    flipped_y = flip_labels(
                        y.detach().cpu().numpy(),
                        source_class,
                        target_class,
                    )

                    y = torch.tensor(
                        flipped_y,
                        dtype=torch.long,
                        device=self.device,
                    )

                optimizer.zero_grad()

                logits = self.model(x)
                loss = loss_fn(logits, y)

                loss.backward()

                # Optional gradient clipping.
                # Disabled when gradient_clip_norm is None or <= 0.
                if self.gradient_clip_norm is not None:
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
                    loss.item() * batch_size
                )
                total_examples += batch_size

                predictions = torch.argmax(
                    logits,
                    dim=1,
                )
                correct += (
                    predictions == y
                ).sum().item()

        avg_loss = (
            total_loss / max(total_examples, 1)
        )
        train_accuracy = (
            correct / max(total_examples, 1)
        )

        # Parameters after honest local training.
        params = get_model_parameters(self.model)

        # ── Sign-flipping attack ──────────────────────────────────────
        if attack_active and attack_type == "sign_flip":
            sign_flip_scale = float(
                self.config.get(
                    "sign_flip_scale",
                    10.0,
                )
            )
            print(
                f"Client {self.cid} | "
                f"Round {server_round} | "
                f"Applying SIGN FLIP"
            )

            malicious_parameters = []

            for local_layer, global_layer in zip(
                params,
                global_parameters,
            ):
                # Honest local update:
                # delta = local - global
                local_update = (
                    local_layer - global_layer
                )

                # Reverse and amplify the update.
                malicious_update = (
                    -sign_flip_scale * local_update
                )

                # Convert the malicious update back into parameters.
                malicious_layer = (
                    global_layer + malicious_update
                )

                malicious_parameters.append(
                    malicious_layer.astype(
                        local_layer.dtype,
                        copy=False,
                    )
                )

            params = malicious_parameters

            logging.info(
                f"[Client {self.cid}] Sign-flipping attack "
                f"applied with scale={sign_flip_scale}."
            )

        # ── Backdoor norm-scaling option ──────────────────────────────
        if attack_active and attack_type in ("backdoor", "both"):
            scale_to_norm = bool(
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

            if scale_to_norm:
                params = scale_gradient_to_norm(
                    params,
                    target_norm,
                )

        metrics = {
            "train_loss": float(avg_loss),
            "train_accuracy": float(train_accuracy),
            "cid": (
                float(self.cid)
                if self.cid.isdigit()
                else -1.0
            ),
            "is_poisoned": float(self.is_poisoned),
            "attack_active": float(attack_active),
        }

        return (
            params,
            total_examples,
            metrics,
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config,
    ) -> Tuple[float, int, Dict[str, float]]:

        set_model_parameters(
            self.model,
            parameters,
        )

        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_examples = 0
        correct = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()

                logits = self.model(x)
                loss = loss_fn(logits, y)

                batch_size = y.size(0)

                total_loss += (
                    loss.item() * batch_size
                )
                total_examples += batch_size

                predictions = torch.argmax(
                    logits,
                    dim=1,
                )
                correct += (
                    predictions == y
                ).sum().item()

        avg_loss = (
            total_loss / max(total_examples, 1)
        )
        accuracy = (
            correct / max(total_examples, 1)
        )

        metrics = {
            "val_accuracy": float(accuracy),
            "cid": (
                float(self.cid)
                if self.cid.isdigit()
                else -1.0
            ),
        }

        return (
            float(avg_loss),
            total_examples,
            metrics,
        )
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import List

from src.logging.logger import logging
from src.exception.exception import FLIDSException


class MLPClassifier(nn.Module):
    """
    Lightweight funnel MLP for CIC-IDS2017 intrusion detection.
    Architecture: input → 64 → 32 → 16 → num_classes
    - BatchNorm before activation: stabilizes covariate shift across attack types
    - LeakyReLU(0.01): preserves variance of Z-scored (negative) features
    - Dropout(0.2): regularization without underfitting on reduced features
    - Raw logits output: CrossEntropyLoss applies log-softmax internally
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=dropout_rate),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Serialize PyTorch state_dict → list of NumPy arrays (for Flower)."""
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Deserialize list of NumPy arrays → PyTorch state_dict (for Flower)."""
    try:
        state_dict = dict(zip(model.state_dict().keys(),
                              [torch.tensor(p) for p in parameters]))
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        raise FLIDSException(e, sys)

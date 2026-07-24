import sys
import numpy as np
from typing import List, Tuple

from src.logging.logger import logging
from src.exception.exception import FLIDSException


# ═══════════════════════════════════════════════════════
# ATTACK 1: Targeted Semantic Label-Flipping
# ═══════════════════════════════════════════════════════
#   Selectively flips ONE source class → target class.
#   Default: DDoS (3) → Benign (0).
#   NOT random noise — FedAvg naturally averages random noise out.
#   This is a focused, semantically meaningful attack that degrades
#   the model's ability to detect one specific critical attack type.

def flip_labels(
    y: np.ndarray,
    source_class: int,
    target_class: int,
) -> np.ndarray:
    """
    Flip all instances of source_class to target_class in label array.

    Args:
        y:            Label array (int64 NumPy array).
        source_class: Class index to flip FROM (e.g., DDoS = 3).
        target_class: Class index to flip TO   (e.g., Benign = 0).

    Returns:
        y_flipped: Modified label array (copy — original is unchanged).
    """
    try:
        y_flipped = y.copy()
        mask = y_flipped == source_class
        n_flipped = int(mask.sum())
        y_flipped[mask] = target_class
        logging.info(
            f"[Attacker] Label flip: {n_flipped} samples "
            f"class {source_class} → class {target_class}"
        )
        return y_flipped
    except Exception as e:
        raise FLIDSException(e, sys)


# ═══════════════════════════════════════════════════════
# ATTACK 2: Stealthy Backdoor Trigger Injection
# ═══════════════════════════════════════════════════════
#   Injects synthetic traffic rows with a specific anomalous feature
#   signature (e.g., Flow_Duration=999999 AND ACK_Flag_Count=1).
#   Mislabels these injected rows as Benign.
#   At inference time, any traffic matching this signature is
#   misclassified as Benign regardless of true attack type.

def inject_backdoor_trigger(
    X: np.ndarray,
    y: np.ndarray,
    trigger_feature_idx: List[int],
    trigger_values: List[float],
    inject_ratio: float,
    benign_class: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Append trigger-stamped rows mislabeled as benign to the local dataset.

    Args:
        X:                  Feature matrix (float32 NumPy array).
        y:                  Label array (int64 NumPy array).
        trigger_feature_idx: Column indices to stamp with trigger values.
        trigger_values:     Corresponding values to write at those indices.
        inject_ratio:       Fraction of dataset rows to poison (e.g., 0.1 = 10%).
        benign_class:       Integer label to assign to poisoned rows (default 0).

    Returns:
        (X_poisoned, y_poisoned): Augmented feature matrix and label array.
    """
    try:
        n_inject = max(1, int(len(X) * inject_ratio))
        src_idx = np.random.choice(len(X), size=n_inject, replace=False)

        X_trigger = X[src_idx].copy()
        for feat_idx, val in zip(trigger_feature_idx, trigger_values):
            X_trigger[:, feat_idx] = val

        y_trigger = np.full(n_inject, benign_class, dtype=y.dtype)

        X_poisoned = np.vstack([X, X_trigger])
        y_poisoned = np.concatenate([y, y_trigger])

        logging.info(
            f"[Attacker] Backdoor: injected {n_inject} trigger rows "
            f"(ratio={inject_ratio}, benign_class={benign_class})"
        )
        return X_poisoned, y_poisoned
    except Exception as e:
        raise FLIDSException(e, sys)


# ═══════════════════════════════════════════════════════
# ATTACK 3: Gradient Norm Scaling (Stealth Bypass)
# ═══════════════════════════════════════════════════════
#   Scales outgoing weight update L2-norm to match the benign client
#   average, bypassing naive norm-clipping defenses on the server.
#   Used after backdoor injection so the poisoned gradient is not
#   rejected by magnitude-based filters.

def scale_gradient_to_norm(
    local_weights: List[np.ndarray],
    target_norm: float,
) -> List[np.ndarray]:
    """
    Scale whole gradient to match target L2-norm — stealth norm-capping bypass.

    Args:
        local_weights: List of NumPy arrays (full model state_dict values).
        target_norm:   Desired L2-norm for the flattened parameter vector.

    Returns:
        Scaled list of NumPy arrays with the same structure as local_weights.
    """
    try:
        flat = np.concatenate([w.flatten() for w in local_weights])
        current_norm = np.linalg.norm(flat)
        if current_norm < 1e-9:
            logging.warning("[Attacker] Gradient norm near zero — skipping scaling.")
            return local_weights
        scale = target_norm / current_norm
        scaled = [w * scale for w in local_weights]
        logging.info(
            f"[Attacker] Gradient norm scaled: {current_norm:.4f} → "
            f"{target_norm:.4f} (factor={scale:.4f})"
        )
        return scaled
    except Exception as e:
        raise FLIDSException(e, sys)


# ═══════════════════════════════════════════════════════
# ATTACK 4: Min-Max Attack (NDSS 2021)
# ═══════════════════════════════════════════════════════
#   AGR-agnostic attack: maximises deviation of the poisoned update
#   from the cohort mean while staying within epsilon of the nearest
#   benign client. Specifically designed to evade Cosine-similarity and
#   clustering-based defenses. Proved to drop Krum to 30.1% accuracy.

def min_max_attack(
    local_update: List[np.ndarray],
    all_updates: List[List[np.ndarray]],
    epsilon: float = 0.5,
) -> List[np.ndarray]:
    """
    Min-Max attack: craft a poisoned update that maximises damage while
    staying within epsilon of the nearest benign client's update.
    """
    try:
        local_flat = np.concatenate([u.flatten() for u in local_update])
        all_flat = np.stack([np.concatenate([u.flatten() for u in up]) for up in all_updates])
        mean_flat = all_flat.mean(axis=0)

        dists = np.linalg.norm(all_flat - local_flat, axis=1)
        nearest_dist = float(np.min(dists[dists > 0])) if (dists > 0).any() else 1.0

        direction = local_flat - mean_flat
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-9:
            return local_update

        scale = min(epsilon, nearest_dist) / dir_norm
        poisoned_flat = mean_flat + direction * scale

        shapes = [u.shape for u in local_update]
        result, idx = [], 0
        for shape in shapes:
            size = int(np.prod(shape))
            result.append(poisoned_flat[idx: idx + size].reshape(shape))
            idx += size

        logging.info("[Attacker] Min-Max attack applied.")
        return result
    except Exception as e:
        raise FLIDSException(e, sys)


# ═══════════════════════════════════════════════════════
# ATTACK 5: Min-Sum Attack (NDSS 2021)
# ═══════════════════════════════════════════════════════
#   Minimises sum of cosine similarities to all other clients,
#   directly attacking our pairwise Cosine+MAD scoring by ensuring
#   the poisoned update sits as far as possible from all benign updates.

def min_sum_attack(
    local_update: List[np.ndarray],
    all_updates: List[List[np.ndarray]],
) -> List[np.ndarray]:
    """
    Min-Sum attack: craft a poisoned update that minimises its cosine
    similarity to all other client updates, directly targeting our MAD scoring.
    """
    try:
        all_flat = np.stack([np.concatenate([u.flatten() for u in up]) for up in all_updates])
        mean_flat = all_flat.mean(axis=0)

        direction = -mean_flat
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-9:
            return local_update

        local_flat = np.concatenate([u.flatten() for u in local_update])
        local_norm = np.linalg.norm(local_flat)
        poisoned_flat = (direction / dir_norm) * local_norm

        shapes = [u.shape for u in local_update]
        result, idx = [], 0
        for shape in shapes:
            size = int(np.prod(shape))
            result.append(poisoned_flat[idx: idx + size].reshape(shape))
            idx += size

        logging.info("[Attacker] Min-Sum attack applied.")
        return result
    except Exception as e:
        raise FLIDSException(e, sys)


# ═══════════════════════════════════════════════════════
# ATTACK 6: LIE "Little Is Enough" Attack (FedLAW 2026)
# ═══════════════════════════════════════════════════════
#   Injects b_{k,j} = mu_k + z * sigma_k per parameter, where z puts
#   the update just outside the benign variance bound. Specifically designed
#   to evade variance-based scoring (like our MAD Z-score). Tests whether
#   our EMA reputation correctly identifies these stealthy attackers over time.

def lie_attack(
    all_updates: List[List[np.ndarray]],
    z_clip: float = 2.0,
) -> List[np.ndarray]:
    """
    LIE attack: craft an update equal to mean + z * std per parameter.
    Stays within plausible benign variance to evade per-round MAD scoring,
    but diverges from honest client direction over multiple rounds.
    """
    try:
        all_flat = np.stack([np.concatenate([u.flatten() for u in up]) for up in all_updates])
        mu = all_flat.mean(axis=0)
        sigma = all_flat.std(axis=0)

        poisoned_flat = mu + z_clip * sigma

        first = all_updates[0]
        shapes = [u.shape for u in first]
        result, idx = [], 0
        for shape in shapes:
            size = int(np.prod(shape))
            result.append(poisoned_flat[idx: idx + size].reshape(shape))
            idx += size

        logging.info(f"[Attacker] LIE attack applied (z={z_clip}).")
        return result
    except Exception as e:
        raise FLIDSException(e, sys)


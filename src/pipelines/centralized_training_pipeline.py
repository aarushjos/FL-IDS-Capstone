import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.configs.config import CONFIG
from src.configs.paths import PREPROCESSED_DIR, MODELS_DIR, DATA_DIR, ensure_dirs
from src.components.model.model import MLPClassifier
from src.components.data.torch_dataset import make_dataloader
from src.components.data.data_partitioner import load_partition


# ─── Loss Function ────────────────────────────────────────────────────────────

def get_weighted_loss_fn(y_train: np.ndarray, device: torch.device) -> nn.CrossEntropyLoss:
    """Capped inverse-frequency class weights to handle CIC-IDS2017 imbalance."""
    cap = CONFIG["centralized"]["weight_cap"]
    classes = np.unique(y_train)
    raw_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    capped = np.clip(raw_weights, a_min=None, a_max=cap)
    weight_tensor = torch.tensor(capped, dtype=torch.float32).to(device)
    logging.info(f"Class weights (capped at {cap}): {dict(zip(classes.tolist(), capped.tolist()))}")
    return nn.CrossEntropyLoss(weight=weight_tensor, reduction="mean")


# ─── Train / Eval ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_loss += criterion(out, y).item() * X.size(0)
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_preds, all_targets


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_centralized_training():
    try:
        print("\n" + "="*60)
        print("  FL-IDS Centralized Baseline Training")
        print("="*60)
        logging.info("Centralized training pipeline started")
        ensure_dirs()

        cfg = CONFIG["centralized"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device]  {device}" + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "  (no GPU found)"))
        logging.info(f"Device: {device}")

        # Load test set
        test_data = np.load(PREPROCESSED_DIR / "test_set.npz")
        X_test, y_test = test_data["X_test"], test_data["y_test"]

        # Build train set from all 50 client shards (already scaled + partitioned)
        num_clients = CONFIG["federated"]["num_clients"]
        X_parts, y_parts = [], []
        for cid in range(num_clients):
            shard = np.load(DATA_DIR / f"client_{cid:04d}.npz")
            X_parts.append(shard["X_train"])
            y_parts.append(shard["y_train"])
        X_train = np.concatenate(X_parts, axis=0)
        y_train = np.concatenate(y_parts, axis=0)

        print(f"[Data]    Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
        print(f"[Data]    Features: {X_train.shape[1]}  |  Classes: {len(np.unique(y_train))}")
        logging.info(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
        logging.info(f"Features: {X_train.shape[1]} | Classes: {len(np.unique(y_train))}")

        # DataLoaders
        batch_size = cfg["batch_size"]
        train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
        test_loader  = make_dataloader(X_test,  y_test,  batch_size=batch_size, shuffle=False)

        # Model
        input_dim  = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=cfg["hidden_dims"],
            num_classes=num_classes,
            dropout_rate=cfg["dropout_rate"],
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Model]   Hidden dims: {cfg['hidden_dims']}  |  Parameters: {total_params:,}")
        print(f"[Train]   Epochs: {cfg['epochs']}  |  Batch size: {cfg['batch_size']}  |  LR: {cfg['learning_rate']}")
        print("-"*60)
        logging.info(f"Model parameters: {total_params:,}")

        # Loss, optimizer, scheduler
        criterion = get_weighted_loss_fn(y_train, device)
        optimizer = Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min",
            factor=cfg["scheduler_factor"],
            patience=cfg["scheduler_patience"],
        )

        # Training loop
        best_macro_f1 = 0.0
        save_path = MODELS_DIR / "baseline_mlp.pth"

        for epoch in range(cfg["epochs"]):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_f1, _, _ = evaluate(model, test_loader, criterion, device)
            scheduler.step(val_loss)

            is_best = val_f1 > best_macro_f1
            flag = "  * NEW BEST" if is_best else ""
            print(
                f"  Epoch {epoch+1:03d}/{cfg['epochs']} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val Loss: {val_loss:.4f} "
                f"| Macro F1: {val_f1:.4f}"
                f"{flag}"
            )
            logging.info(
                f"Epoch {epoch+1:02d}/{cfg['epochs']} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Macro F1: {val_f1:.4f}"
            )

            # Save best model
            if is_best:
                best_macro_f1 = val_f1
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dims": cfg["hidden_dims"],
                    "num_classes": num_classes,
                    "dropout_rate": cfg["dropout_rate"],
                    "val_macro_f1": best_macro_f1,
                }, save_path)
                logging.info(f"  ✅ New best saved → {save_path}")

        # Final evaluation with classification report
        print("-"*60)
        print(f"[Done]    Best Macro F1: {best_macro_f1:.4f}")
        print(f"[Saved]   {save_path}")
        print("="*60)
        logging.info(f"\nTraining complete. Best Macro F1: {best_macro_f1:.4f}")
        logging.info(f"Model saved at: {save_path}")

        _, final_f1, preds, targets = evaluate(model, test_loader, criterion, device)
        print("\n--- Final Classification Report ---")
        print(classification_report(targets, preds, zero_division=0))

    except Exception as e:
        raise FLIDSException(e, sys)

if __name__ == "__main__":
    run_centralized_training()
    
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.configs.paths import RAW_DIR, PREPROCESSED_DIR, ensure_dirs
from src.components.data.data_loader import load_cicids2018
from src.components.data.data_preprocessor import preprocess
from src.components.data.data_partitioner import run_partitioning


def run_data_pipeline():
    try:
        logging.info("Data pipeline started (CSE-CIC-IDS2018)")
        ensure_dirs()

        df_raw = load_cicids2018()

        raw_path = RAW_DIR / "cicids2018_raw.parquet"
        df_raw.to_parquet(raw_path, index=False)
        logging.info(f"Raw parquet saved → {raw_path}")

        X, y, feature_cols, le, scaler = preprocess(df_raw)

        with open(PREPROCESSED_DIR / "label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

        with open(PREPROCESSED_DIR / "feature_cols.pkl", "wb") as f:
            pickle.dump(feature_cols, f)

        with open(PREPROCESSED_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        np.savez_compressed(PREPROCESSED_DIR / "test_set.npz", X_test=X_test, y_test=y_test)

        run_partitioning(X_train, y_train)

        logging.info("Data pipeline completed")
        return feature_cols, le, scaler, X_test, y_test

    except Exception as e:
        raise FLIDSException(e, sys)


if __name__ == "__main__":
    run_data_pipeline()
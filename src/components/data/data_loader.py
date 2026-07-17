import sys
from pathlib import Path

import pandas as pd

from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.configs.paths import RAW_DIR

# Kaggle dataset identifier for CSE-CIC-IDS2018
_KAGGLE_DATASET = "solarmainframe/ids-intrusion-csv"


def _find_csv_dir() -> Path:
    """
    Locate the directory containing IDS2018 CSV files.

    Search order:
      1. artifacts/raw/ids2018/ — if CSVs already placed manually or cached here
      2. kagglehub download cache — auto-downloaded on first call
    """
    local_dir = RAW_DIR / "ids2018"
    if local_dir.exists() and any(local_dir.glob("*.csv")):
        logging.info(f"Using locally cached IDS2018 CSVs from {local_dir}")
        return local_dir

    logging.info(
        "IDS2018 CSVs not found locally — downloading via kagglehub "
        f"(dataset: {_KAGGLE_DATASET}) …"
    )
    logging.info(
        "NOTE: First download is ~1.6 GB. Subsequent runs use the kagglehub cache."
    )

    try:
        import kagglehub  # noqa: PLC0415

        download_path = Path(kagglehub.dataset_download(_KAGGLE_DATASET))
        logging.info(f"kagglehub download complete. Files at: {download_path}")
        return download_path

    except ImportError:
        raise ImportError(
            "kagglehub is not installed. Run:  pip install kagglehub\n"
            "Or manually place the IDS2018 CSVs in artifacts/raw/ids2018/ "
            "and re-run the pipeline."
        )
    except Exception as e:
        raise FLIDSException(e, sys)


def load_cicids2018() -> pd.DataFrame:
    """
    Load CSE-CIC-IDS2018 by concatenating all day-based CSV files.

    On the first call, automatically downloads the dataset from Kaggle via
    kagglehub (~1.6 GB). Subsequent calls reuse the kagglehub cache —
    no repeated downloads.

    If you prefer to manage files manually, place the CSV files in:
        artifacts/raw/ids2018/
    and the auto-download is skipped entirely.

    IDS2018 CSV column names have leading/trailing whitespace — stripped automatically.
    The 'Label' column is present in all files and matches the preprocessor convention.

    Returns:
        pd.DataFrame: Concatenated dataset (~16 M rows, ~80 columns).
    """
    try:
        csv_dir = _find_csv_dir()
        csv_files = sorted(csv_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {csv_dir} after download attempt.\n"
                f"Manually place IDS2018 CSVs in artifacts/raw/ids2018/ and retry."
            )

        logging.info(
            f"Loading {len(csv_files)} CSV file(s) from {csv_dir} …"
        )

        dfs = []
        for csv_path in csv_files:
            logging.info(f"  Reading {csv_path.name} …")
            df = pd.read_csv(
                csv_path,
                low_memory=False,
                # IDS2018 has literal "Infinity"/"-Infinity" strings in numeric cols
                # — treat them as NaN so they survive StandardScaler imputation.
                na_values=["Infinity", "infinity", "-Infinity", "-infinity", ""],
                # Some IDS2018 files have the header row repeated mid-file.
                # on_bad_lines='skip' discards those rows silently.
                on_bad_lines="skip",
            )

            # Strip leading/trailing whitespace from column names (IDS2018 quirk).
            df.columns = df.columns.str.strip()

            # Force all feature columns to numeric dtype.
            # Keeps 'Label' as-is (string class names).
            # Any remaining non-numeric values become NaN — handled by impute().
            label_col = "Label"
            for col in df.columns:
                if col != label_col:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            dfs.append(df)


        combined = pd.concat(dfs, ignore_index=True)
        logging.info(
            f"CSE-CIC-IDS2018 loaded: {combined.shape[0]:,} rows, "
            f"{combined.shape[1]} columns"
        )
        return combined

    except Exception as e:
        raise FLIDSException(e, sys)


# ---------------------------------------------------------------------------
# Legacy loader (CIC-IDS2017 via HuggingFace) — kept for reference only.
# NOT used by the active pipeline.
# ---------------------------------------------------------------------------
def load_cicids2017() -> pd.DataFrame:
    """DEPRECATED — pipeline now uses load_cicids2018()."""
    try:
        from datasets import load_dataset  # optional dependency

        logging.info("Loading CIC-IDS2017 from HuggingFace (deprecated)")
        ds = load_dataset("bvk/CICIDS-2017")
        df = ds["train"].to_pandas()
        logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise FLIDSException(e, sys)

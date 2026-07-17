import sys
import shutil
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
# Run from project root: python scripts/reset_artifacts.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PREPROCESSED_DIR = ARTIFACTS_DIR / "preprocessed"
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"

def confirm(prompt: str) -> bool:
    ans = input(prompt + " [y/N]: ").strip().lower()
    return ans == "y"

def main():
    print("=" * 60)
    print("FL-IDS Artifact Reset — IDS2017 → IDS2018 Migration")
    print("=" * 60)
    print()
    print("This script will DELETE:")
    print(f"  • All client partitions in  {DATA_DIR}")
    print(f"  • All preprocessed .pkl/.npz in {PREPROCESSED_DIR}")
    print(f"  • All model checkpoints in  {MODELS_DIR}")
    print()
    print("This script will KEEP:")
    print(f"  • Raw parquet files in {ARTIFACTS_DIR / 'raw'}")
    print(f"  • Experiment result CSVs in {ARTIFACTS_DIR / 'results'}")
    print()

    if not confirm("Are you sure you want to proceed?"):
        print("Aborted — no files deleted.")
        sys.exit(0)

    deleted = 0

    # Delete client partition shards
    for f in DATA_DIR.glob("client_*.npz"):
        f.unlink()
        deleted += 1
        print(f"  Deleted: {f.name}")

    # Delete preprocessed artifacts
    for pattern in ("*.pkl", "*.npz"):
        for f in PREPROCESSED_DIR.glob(pattern):
            f.unlink()
            deleted += 1
            print(f"  Deleted: {f.name}")

    # Delete model checkpoints
    for f in MODELS_DIR.glob("*.pth"):
        f.unlink()
        deleted += 1
        print(f"  Deleted: {f.name}")

    print()
    print(f"Done — {deleted} file(s) deleted.")
    print()
    print("Next steps:")
    print("  1. Ensure CSVs are in artifacts/raw/ids2018/*.csv")
    print("  2. Run: python -m src.pipelines.data_pipeline")
    print("  3. Run EDA notebook: notebooks/08_ids2018_eda.ipynb")

if __name__ == "__main__":
    main()

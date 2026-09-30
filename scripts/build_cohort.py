"""Step 1. Build the analysis cohort from NHANES 1999–2008 and the Linked Mortality File (files downloaded on first
use, or beforehand by scripts/download_nhanes.py) and write data/cohort.parquet, data/splits.json and
results/cohort_flow.md."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity.nhanes import build_cohort

if __name__ == "__main__":
    # --n_eval sets the size of the evaluation split; the first 200 records of the seeded permutation stay the held-out
    # 'pilot' split
    n_eval = int(sys.argv[sys.argv.index("--n_eval") + 1]) if "--n_eval" in sys.argv else 1000
    df = build_cohort(n_eval=n_eval)
    print(df["split"].value_counts())
    print("deaths by split:\n", df.groupby("split")["death_10y"].agg(["sum", "mean"]))

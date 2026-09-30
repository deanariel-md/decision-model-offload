"""Download NHANES III adult/exam/lab + LMF, build data/cohort_nhanes3.parquet (split='prior') and
results/cohort_flow_nhanes3.md. Run after build_cohort.py; needs internet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity.nhanes3 import build_nhanes3_cohort

if __name__ == "__main__":
    df = build_nhanes3_cohort()
    print(f"NHANES III prior cohort: {len(df):,} adults, {int(df.death_10y.sum()):,} deaths within 10 years")
    print(df.race_ethnicity.value_counts(dropna=False))

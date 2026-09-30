"""eICU replication, step 1: the records. No API calls.
  python scripts/build_eicu.py demo [--src <folder of eICU-CRD Demo v2.0.1>]   # open demo -> data/eicu/demo_cohort.parquet
  python scripts/build_eicu.py full --src <folder of eICU-CRD v2.0> [--demo-src <demo folder>]
                                                    # credentialed, local only -> data/eicu/full_fit.parquet
For the demo, --src defaults to config/eicu.yaml sources.demo; the full build needs --src (or sources.full), and
--demo-src defaults to sources.demo. The full build removes every demo stay and patient, and every demo hospital only if
demo_hospitals is true (config/eicu.yaml exclude_from_fit), and reports how many demo ids it found: zero found stays
would mean the ids differ."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import eicu as E

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["demo", "full"])
    ap.add_argument("--src", default=None)
    ap.add_argument("--demo-src", default=None)
    a = ap.parse_args()
    if a.source == "demo":
        df = E.build_demo(a.src)
        print(df.groupby("split")[E.OUTCOME].agg(["size", "sum"]).rename(columns={"size": "records", "sum": "deaths"}))
        print(df["ethnicity"].value_counts().to_string())
        print("flow: data/eicu/demo_flow.md")
    else:
        if a.src is None and not E.CFG["sources"].get("full"):
            ap.error("full: pass the folder of the credentialed eICU-CRD v2.0 as --src")
        df = E.build_full(a.src, a.demo_src)
        print(f"fitting set: {len(df):,} records, {int(df[E.OUTCOME].sum()):,} deaths; flow: data/eicu/full_flow.md")

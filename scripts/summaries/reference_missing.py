"""Missing values handled by the outcome reference (Supplementary Note 3, PROBAST+AI appraisal).

Counts, per split of data/cohort.parquet, the participants with a numeric model field set to the fitting-set median
(SimpleImputer with an indicator) and with a categorical field kept as its own 'missing' level, using the reference's
own field lists (jevity.reference.NUMERIC, CATEGORICAL). No model is fitted and no model is called.

  python scripts/summaries/reference_missing.py -> results/summaries/reference_missing.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevity.reference import CATEGORICAL, NUMERIC  # noqa: E402

BIOMARKERS = ["albumin_g_dl", "creatinine_mg_dl", "glucose_mg_dl", "crp_mg_dl", "lymphocyte_pct", "mcv_fl",
              "rdw_pct", "alk_phos_u_l", "wbc_10e3_ul"]

c = pd.read_parquet(ROOT / "data" / "cohort.parquet")
out = {"note": __doc__.strip().splitlines()[0]}
for split, d in c.groupby("split"):
    num = d[NUMERIC].isna()
    share = (num.mean() * 100).sort_values(ascending=False)
    out[split] = {
        "n": int(len(d)),
        "deaths": int(d["death_10y"].sum()),
        "numeric_imputed": int(num.any(axis=1).sum()),
        "numeric_imputed_pct": float(num.any(axis=1).mean() * 100),
        "categorical_missing": int(d[CATEGORICAL].isna().any(axis=1).sum()),
        "biomarkers_incomplete": int(d[BIOMARKERS].isna().any(axis=1).sum()),
        "numeric_missing_pct_by_field": {k: round(float(v), 2) for k, v in share[share > 0].items()},
    }
assert out["fit"]["n"] == 11971 and out["fit"]["deaths"] == 1899, "fitting set differs from Supplementary Table 9"
assert out["eval"]["biomarkers_incomplete"] == 0 and out["pilot"]["biomarkers_incomplete"] == 0
(ROOT / "results" / "summaries" / "reference_missing.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "numeric_missing_pct_by_field"} if isinstance(v, dict) else v
                  for k, v in out.items()}, indent=1))
print("fit top fields:", list(out["fit"]["numeric_missing_pct_by_field"].items())[:5])

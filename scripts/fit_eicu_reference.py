"""eICU replication, step 2 (local machine; the full fitting set never leaves it). Fits the primary spline reference
and the LightGBM and XGBoost references on data/eicu/full_fit.parquet (--only: the named references alone, keeping the
others' entries in reference_check.json) and checks how well each transfers to the demo patients (discrimination,
calibration in the large, Brier), beside APACHE IVa's own predicted hospital mortality. A diagnostic, not a tuning
step: the penalty is set in config/eicu.yaml. Writes data/eicu/reference_<name>.joblib and
results/eicu/reference_check.json.
  python scripts/fit_eicu_reference.py [--only xgboost]"""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import eicu as E, eicu_reference as ER

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / E.CFG["out_dir"]


def check(p: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    return {"n": int(ok.sum()), "deaths": int(y.sum()), "auroc": round(float(roc_auc_score(y, p)), 3),
            "observed_to_expected": round(float(y.sum() / p.sum()), 3), "brier": round(float(brier_score_loss(y, p)), 4)}


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    fit = pd.read_parquet(D / "full_fit.parquet")
    demo = pd.read_parquet(D / "demo_cohort.parquet")
    assert not set(fit["SEQN"]) & set(demo["SEQN"]), "a demo stay is in the fitting set"
    y = demo[E.OUTCOME].to_numpy()
    res = ROOT / "results" / "eicu" / "reference_check.json"
    out = {"fit_n": len(fit), "fit_deaths": int(fit[E.OUTCOME].sum()), "demo": {}}
    if only and res.exists():                    # --only xgboost: fit the named references, keep the others' files
        out["demo"] = json.loads(res.read_text())["demo"]
    for name, fn in (("spline_logit", ER.fit_spline_logit), ("lightgbm", ER.fit_lgbm), ("xgboost", ER.fit_xgb)):
        if only and name not in only:
            continue
        ref = fn(fit)
        joblib.dump(ref, D / f"reference_{name}.joblib")
        out["demo"][name] = check(ref.predict(demo), y)
        print(name, out["demo"][name])
    out["demo"]["apache_iva"] = check(demo["apache_iva_pred"].to_numpy(dtype=float), y)
    print("apache_iva", out["demo"]["apache_iva"])
    (ROOT / "results" / "eicu").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "eicu" / "reference_check.json").write_text(json.dumps(out, indent=2))

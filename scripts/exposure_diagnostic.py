"""Exposure diagnostic (limited specificity; never a clearance test). For a seeded subset of evaluation records, one
field is replaced by "MISSING" and each LLM family is asked for its value. Exact-match rate at the recorded precision
and absolute error are compared with an imputation baseline (gradient boosting on the other fields, trained on the
fitting split). A model that recovers exact values far more often than the baseline has likely seen the records.
  python scripts/exposure_diagnostic.py [--simulate]      -> results/eval/exposure.json, results/eval/exposure_calls.parquet
  python scripts/exposure_diagnostic.py --from-calls      # recompute exposure.json from exposure_calls.parquet (no calls)"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity.clients import FillClient, RawStore, MODELS, PROMPTS, active_families
from jevity.serialize import to_state, to_json, NUMERIC_FIELDS

ROOT = Path(__file__).resolve().parents[1]
COL_OF = {k: c for k, c, _, _ in NUMERIC_FIELDS}


def masked_text(rec: dict, field: str) -> str:
    st = to_state(rec)
    if field in st:
        st[field] = {"value": "MISSING", "unit": st[field]["unit"]} if isinstance(st[field], dict) else "MISSING"
    return to_json(st)


def baseline_predictions(fit: pd.DataFrame, ev: pd.DataFrame, field: str) -> np.ndarray:
    import lightgbm as lgb
    target = COL_OF[field]
    feats = [c for _, c, _, _ in NUMERIC_FIELDS if c != target]
    Xf = fit[feats].apply(pd.to_numeric, errors="coerce"); yf = pd.to_numeric(fit[target], errors="coerce")
    ok = yf.notna()
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1).fit(Xf[ok], yf[ok])
    return m.predict(ev[feats].apply(pd.to_numeric, errors="coerce"))


def summarise(df: pd.DataFrame, n_profiles: int, fields: list[str]) -> tuple[pd.DataFrame, dict]:
    """Exact-match rate at the recorded precision and absolute error per system, against the imputation baseline."""
    df["exact"] = np.isclose(df.value.astype(float), df.truth.astype(float), atol=1e-9)
    df["abs_err"] = (df.value.astype(float) - df.truth.astype(float)).abs()
    summ = df.groupby("model").agg(n=("exact", "size"), valid=("value", lambda s: float(s.notna().mean())),
                                   exact_match=("exact", "mean"), mae=("abs_err", "mean")).reset_index()
    b = summ.set_index("model").loc["imputation_baseline"]
    summ["exact_match_ratio_to_baseline"] = summ.exact_match / max(b.exact_match, 1e-9)
    summ["flag_possible_exposure"] = (summ.exact_match >= 0.05) & (summ.exact_match > 2 * b.exact_match) & (summ.model != "imputation_baseline")
    out = {"n_profiles": int(n_profiles), "fields": fields, "by_model": summ.round(4).to_dict("records"),
           "by_model_field": df.groupby(["model", "field"])["exact"].mean().round(4).reset_index().to_dict("records"),
           "note": "Limited specificity: a negative result does not exclude exposure; a positive result is a flag, not proof."}
    return summ, out


def write(out: dict) -> None:
    (ROOT / "results" / "eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "eval" / "exposure.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--simulate", action="store_true"); ap.add_argument("--yes", action="store_true")
    ap.add_argument("--from-calls", action="store_true", help="recompute the summary from results/eval/exposure_calls.parquet")
    a = ap.parse_args()
    ex = PROMPTS["exposure"]; fields = list(ex["fields"])
    calls_f = ROOT / "results" / "eval" / "exposure_calls.parquet"
    if a.from_calls:
        df = pd.read_parquet(calls_f)
        summ, out = summarise(df, int((df.model == "imputation_baseline").sum()), fields)
        write(out)
        print(summ.round(3).to_string())
        sys.exit(0)
    co = pd.read_parquet(ROOT / "data" / "cohort.parquet")
    fit = co[co.split == "fit"]
    ev = co[co.split == "eval"].sample(n=min(ex["n_profiles"], (co.split == "eval").sum()), random_state=20260922).reset_index(drop=True)
    ev["field"] = [fields[i % len(fields)] for i in range(len(ev))]
    fams = active_families(("primary",))   # memorisation probe: primary LLMs
    if not a.simulate and not a.yes and input(f"{len(ev) * len(fams)} fill calls. Proceed? [y/N] ").strip().lower() != "y":
        sys.exit(0)
    rows = []
    rng = np.random.default_rng(1)
    for f in fields:
        sub = ev[ev.field == f]
        base = baseline_predictions(fit, sub, f)
        dec = ex["fields"][f]["decimals"]
        truth = pd.to_numeric(sub[COL_OF[f]], errors="coerce").round(dec).to_numpy()
        for j, (_, r) in enumerate(sub.iterrows()):
            rows.append({"model": "imputation_baseline", "field": f, "truth": truth[j], "value": round(float(base[j]), dec)})
            for fam in fams:
                if a.simulate:   # simulated: every family imputes like the baseline plus noise
                    v = round(float(base[j] + rng.normal(0, np.nanstd(truth) * 0.3)), dec)
                    res = {"value": v, "valid": True}
                else:
                    res = FillClient(fam, RawStore(ROOT / "runs" / "exposure")).fill(masked_text(r.to_dict(), f), f)
                rows.append({"model": fam, "field": f, "truth": truth[j],
                             "value": None if res["value"] is None else round(float(res["value"]), dec)})
    df = pd.DataFrame(rows)
    (ROOT / "results" / "eval").mkdir(parents=True, exist_ok=True)
    df.to_parquet(calls_f, index=False)
    summ, out = summarise(df, len(ev), fields)
    write(out)
    print(summ.round(3).to_string())

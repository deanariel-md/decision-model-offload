"""Masked-field test on the open eICU demo (as scripts/exposure_diagnostic.py does for NHANES; limited specificity,
never a clearance test). For 200 evaluation stays (40 per field, each with the field
recorded; config/eicu.yaml `exposure`), one field's value is replaced by "MISSING" and each primary chatbot is asked
for it on the standard route. Exact-match rate at the recorded precision and absolute error are compared with an
imputation baseline: LightGBM on the other numeric fields, trained on the demo stays outside these 200 (open data; the
credentialed database is never read). A chatbot that recovers exact values far more often than the baseline has likely
seen the records. Answers in runs/eicu_demo; writes results/eicu/exposure.json.
  python scripts/eicu_exposure.py [--yes] [--workers 8]"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import eicu as E                                              # noqa: E402
from jevity.clients import FillClient, RawStore, active_families         # noqa: E402

EX = E.CFG["exposure"]
SEED = 20260922


def masked_text(rec: dict, field: str) -> str:
    st = E.to_state(rec)
    if field not in st:
        raise ValueError(f"{field} is not in the record")
    st[field] = {"value": "MISSING", "unit": st[field]["unit"]} if isinstance(st[field], dict) else "MISSING"
    return E.to_json(st)


def pick(demo: pd.DataFrame, n: int = EX["n_profiles"], seed: int = SEED) -> pd.DataFrame:
    """n evaluation stays, n / fields per field, each with that field recorded; no stay used twice."""
    ev = demo[demo.split == "eval"]
    per = n // len(EX["fields"])
    rng = np.random.default_rng(seed)
    used, rows = set(), []
    for f, spec in EX["fields"].items():
        pool = ev[ev[spec["column"]].notna() & ~ev.SEQN.isin(used)]
        take = pool.iloc[rng.permutation(len(pool))[:per]]
        used |= set(take.SEQN)
        rows.append(take.assign(field=f))
    return pd.concat(rows, ignore_index=True)


def imputation(train: pd.DataFrame, test: pd.DataFrame, field: str) -> np.ndarray:
    import lightgbm as lgb
    target = EX["fields"][field]["column"]
    feats = [c for _, c, _, _ in E.CFG["numeric"] if c != target]
    X, y = train[feats].apply(pd.to_numeric, errors="coerce"), pd.to_numeric(train[target], errors="coerce")
    ok = y.notna()
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0).fit(X[ok], y[ok])
    return m.predict(test[feats].apply(pd.to_numeric, errors="coerce"))


def summarise(df: pd.DataFrame, n: int, note: str) -> dict:
    df = df.copy()
    df["exact"] = np.isclose(df.value.astype(float), df.truth.astype(float), atol=1e-9)
    df["abs_err"] = (df.value.astype(float) - df.truth.astype(float)).abs()
    summ = df.groupby("model").agg(n=("exact", "size"), valid=("value", lambda s: float(s.notna().mean())),
                                   exact_match=("exact", "mean"), mae=("abs_err", "mean")).reset_index()
    b = summ.set_index("model").loc["imputation_baseline"]
    summ["exact_match_ratio_to_baseline"] = summ.exact_match / max(b.exact_match, 1e-9)
    summ["flag_possible_exposure"] = (summ.exact_match >= 0.05) & (summ.exact_match > 2 * b.exact_match) \
        & (summ.model != "imputation_baseline")
    return {"n_profiles": n, "fields": list(EX["fields"]), "by_model": summ.round(4).to_dict("records"),
            "by_model_field": df.groupby(["model", "field"])["exact"].mean().round(4).reset_index().to_dict("records"),
            "note": note}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    demo = pd.read_parquet(ROOT / "data" / "eicu" / "demo_cohort.parquet")
    ev = pick(demo)
    train = demo[~demo.SEQN.isin(ev.SEQN)]
    fams = active_families(("primary",))
    if not a.yes and input(f"{len(ev) * len(fams)} fill calls. Proceed? [y/N] ").strip().lower() != "y":
        sys.exit(0)
    store = RawStore(ROOT / "runs" / "eicu_demo")
    rows = []
    for f, spec in EX["fields"].items():
        sub = ev[ev.field == f].reset_index(drop=True)
        dec = spec["decimals"]
        truth = pd.to_numeric(sub[spec["column"]], errors="coerce").round(dec).to_numpy()
        base = imputation(train, sub, f)
        texts = [masked_text(r.to_dict(), f) for _, r in sub.iterrows()]
        rows += [{"model": "imputation_baseline", "field": f, "truth": truth[j], "value": round(float(base[j]), dec)}
                 for j in range(len(sub))]
        for fam in fams:
            cl = FillClient(fam, store, "eicu_demo")
            with ThreadPoolExecutor(a.workers) as pool:
                res = list(pool.map(lambda t: cl.fill(t, f, spec["unit"]), texts))
            rows += [{"model": fam, "field": f, "truth": truth[j],
                      "value": None if r["value"] is None else round(float(r["value"]), dec)} for j, r in enumerate(res)]
        print(f"{f}: {len(sub)} stays asked of {', '.join(fams)}", flush=True)
    out = summarise(pd.DataFrame(rows), len(ev), "Limited specificity: a negative result does not exclude exposure; a "
                    "positive result is a flag, not proof. Imputation trained on the other demo stays (open data).")
    (ROOT / "results" / "eicu").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "eicu" / "exposure.json").write_text(json.dumps(out, indent=1, default=float))
    print(pd.DataFrame(out["by_model"]).to_string())

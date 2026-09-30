"""AUROC 95% intervals for the five free-plan and open-weight models (Fig. 1a; stored answers only, no model call).

results/summaries/risk_classic.json holds AUROC intervals for Jev, the five main LLMs and the two references but not for
the free-plan and open-weight models. This computes them with risk_classic.py's exact procedure: the 12,697 baseline
records every main system answered, probabilities clipped to [0.005, 0.995], 2,000 percentile-bootstrap resamples of
records drawn with numpy default_rng(20260927) in the same order, so every system is resampled on the same records.
Checks before writing: (1) the main LLMs' intervals recomputed here equal risk_classic.json; (2) each free-plan and
open-weight model's point AUROC equals results/summaries/wide_cheaper_structured.json.

    python scripts/summaries/auroc_free_open_ci.py [--cheaper <folder with calls.parquet>]
Reads results/wide/calls.parquet, data/cohort.parquet (scripts/build_cohort.py), the cheaper systems' calls.parquet
(--cheaper, default results/wide_cheaper) and results/summaries/{risk_classic,wide_cheaper_structured}.json (run
risk_classic.py and wide_cheaper_structured.py first). Writes results/summaries/auroc_free_open_ci.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "summaries"))
from risk_classic import auroc  # noqa: E402  (the same AUROC function)
from cost_frontier import MAIN  # noqa: E402

CHEAP = ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
J = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
OUT = ROOT / "results" / "summaries" / "auroc_free_open_ci.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheaper", default=str(ROOT / "results" / "wide_cheaper"))
    a = ap.parse_args()
    w = pd.read_parquet(ROOT / "results" / "wide" / "calls.parquet")
    b = w[(w["edit"] == "baseline") & (w["variant"] == "raw") & (~w["annotated"]) & (w["repeat"] == 0)]
    piv = b.pivot_table(index="profile", columns="model", values="p", aggfunc="first")
    val = b.pivot_table(index="profile", columns="model", values="valid", aggfunc="first").fillna(False).astype(bool)
    common = val.all(axis=1)
    P = piv[common].clip(0.005, 0.995)
    y = pd.read_parquet(ROOT / "data" / "cohort.parquet").set_index("SEQN")["death_10y"].reindex(P.index).astype(int)
    c = pd.read_parquet(Path(a.cheaper) / "calls.parquet")
    c = c[(c["edit"] == "baseline") & (c["variant"] == "raw") & (~c["annotated"]) & (c["repeat"] == 0)]
    cp = c.pivot_table(index="profile", columns="model", values="p", aggfunc="first").reindex(P.index)
    cv = c.pivot_table(index="profile", columns="model", values="valid", aggfunc="first").reindex(P.index)
    assert cv[CHEAP].fillna(False).astype(bool).all().all(), "a free-plan or open-weight answer is missing on the records"
    for s in CHEAP:
        P[s] = cp[s].astype(float).clip(0.005, 0.995)
    yy = y.to_numpy() == 1
    n = len(yy)
    assert n == 12697
    rng = np.random.default_rng(20260927)
    idx = rng.integers(0, n, size=(2000, n))
    rc = J(ROOT / "results" / "summaries" / "risk_classic.json")["systems"]
    wc = J(ROOT / "results" / "summaries" / "wide_cheaper_structured.json")["systems"]
    out = {"note": __doc__.split("\n\n")[0], "n": n, "deaths": int(yy.sum()), "B": 2000, "seed": 20260927,
           "checks": {}, "systems": {}}
    for s in MAIN + CHEAP:
        q = P[s].to_numpy()
        est = auroc(q, yy)
        boot = np.array([auroc(q[ix], yy[ix]) for ix in idx])
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        if s in MAIN:
            assert abs(est - rc[s]["auroc"]) < 1e-9 and max(abs(u - v) for u, v in zip(ci, rc[s]["auroc_ci"])) < 1e-12, s
            out["checks"][s] = "auroc and auroc_ci equal results/summaries/risk_classic.json"
        else:
            assert abs(est - wc[s]["auroc"]) < 1e-9, s
            out["systems"][s] = {"auroc": est, "auroc_ci": ci}
            out["checks"][s] = "auroc equals results/summaries/wide_cheaper_structured.json"
        print(f"{s:12s} AUROC {est:.4f} ({ci[0]:.4f}-{ci[1]:.4f})")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()

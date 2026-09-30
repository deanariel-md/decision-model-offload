"""Classic prediction metrics for ten-year risk of death (stored answers only).

On the 12,697 baseline records every system answered (as sens_spec.py): AUROC with a 95% percentile bootstrap interval
(2,000 resamples of records, seed 20260927) and the paired difference Jev minus each system; calibration in the large
(mean predicted risk against the observed ten-year death share, and their ratio); calibration intercept and slope from
results/wide/analysis.json; and, for the calibration plot, mean predicted risk and observed death share by decile of
each system's own predictions. Systems: Jev asked as the chatbots were, Jev with structured input
(results/wide_docuse, --docuse), the five main chatbots and the two references (out-of-fold predictions). Log-loss
recomputed here must equal results/wide/analysis.json and docuse_summary.json (asserted).

  python scripts/summaries/risk_classic.py --docuse . -> results/summaries/risk_classic.json
Also reads results/wide/calls.parquet, data/cohort.parquet (scripts/build_cohort.py), data/reference_oof.parquet
(scripts/crossfit_reference.py) and results/summaries/docuse_summary.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cost_frontier import MAIN  # noqa: E402

J = lambda p: json.loads(Path(p).read_text())


def auroc(q, y):
    r = pd.Series(q).rank().to_numpy()
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docuse", required=True); a = ap.parse_args()
    w = pd.read_parquet("results/wide/calls.parquet")
    b = w[(w["edit"] == "baseline") & (w["variant"] == "raw") & (~w["annotated"]) & (w["repeat"] == 0)]
    piv = b.pivot_table(index="profile", columns="model", values="p", aggfunc="first")
    val = b.pivot_table(index="profile", columns="model", values="valid", aggfunc="first").fillna(False).astype(bool)
    common = val.all(axis=1)
    P = piv[common].clip(0.005, 0.995)
    y = pd.read_parquet("data/cohort.parquet").set_index("SEQN")["death_10y"].reindex(P.index).astype(int)
    oof = pd.read_parquet("data/reference_oof.parquet").set_index("SEQN").reindex(P.index)
    P["reference_spline_logit"] = oof["p_spline_logit"].clip(0.005, 0.995)
    P["reference_age_sex"] = oof["p_age_sex"].clip(0.005, 0.995)
    d = pd.read_parquet(f"{a.docuse}/results/wide_docuse/calls.parquet")
    d = d[(d["variant"] == "documented") & (d["repeat"] == 0) & (d["edit"] == "baseline")].set_index("profile")
    assert d.loc[P.index, "valid"].all()
    P["jev_documented"] = d.loc[P.index, "p"].astype(float).clip(0.005, 0.995)
    ll = lambda q: float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
    wa = J("results/wide/analysis.json")["prediction"]["systems"]
    for s in ["jev"] + MAIN + ["reference_spline_logit", "reference_age_sex"]:
        assert abs(ll(P[s]) - wa[s]["log_loss"]) < 1e-6, s
        assert abs(auroc(P[s].to_numpy(), y.to_numpy() == 1) - wa[s]["auroc"]) < 1e-9, s
    ds = J("results/summaries/docuse_summary.json")["risk"]["jev_documented"]
    assert abs(ll(P["jev_documented"]) - ds["log_loss"]) < 1e-6
    yy = y.to_numpy() == 1
    n = len(yy)
    rng = np.random.default_rng(20260927)
    idx = rng.integers(0, n, size=(2000, n))
    S = ["jev", "jev_documented"] + MAIN + ["reference_spline_logit", "reference_age_sex"]
    boot = {s: np.array([auroc(P[s].to_numpy()[ix], yy[ix]) for ix in idx]) for s in S}
    obs = float(yy.mean())
    out = {"n": n, "deaths": int(yy.sum()), "observed_death_share": obs, "systems": {}}
    for s in S:
        q = P[s].to_numpy()
        dec = pd.qcut(pd.Series(q).rank(method="first"), 10, labels=False).to_numpy()
        out["systems"][s] = {
            "auroc": auroc(q, yy), "auroc_ci": [float(np.quantile(boot[s], 0.025)), float(np.quantile(boot[s], 0.975))],
            "mean_p": float(q.mean()), "observed_to_expected": obs / float(q.mean()), "log_loss": ll(P[s]),
            "calibration_slope": ds["calibration_slope"] if s == "jev_documented" else wa.get(s, {}).get("calibration_slope"), "calibration_intercept": wa.get(s, {}).get("calibration_intercept"),
            "deciles": [{"mean_p": float(q[dec == k].mean()), "observed": float(yy[dec == k].mean()), "n": int((dec == k).sum())}
                        for k in range(10)]}
        if s not in ("jev",):
            dd = boot["jev"] - boot[s]
            out["systems"][s]["jev_minus_auroc"] = {"est": out["systems"]["jev"]["auroc"] - out["systems"][s]["auroc"] if "jev" in out["systems"] else None,
                                                    "ci": [float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))]}
    for s in S[1:]:
        out["systems"][s]["jev_minus_auroc"]["est"] = out["systems"]["jev"]["auroc"] - out["systems"][s]["auroc"]
    ch = [out["systems"][s] for s in MAIN]
    out["summary"] = {"chatbots_auroc": {"min": min(c["auroc"] for c in ch), "max": max(c["auroc"] for c in ch)},
                      "chatbots_mean_p": {"min": min(c["mean_p"] for c in ch), "max": max(c["mean_p"] for c in ch)},
                      "jev_minus_chatbots_auroc": {"min": min(c["jev_minus_auroc"]["est"] for c in ch),
                                                   "max": max(c["jev_minus_auroc"]["est"] for c in ch)},
                      "jev_minus_chatbots_auroc_upper_ci_max": max(c["jev_minus_auroc"]["ci"][1] for c in ch)}
    Path("results/summaries/risk_classic.json").write_text(json.dumps(out, indent=1))
    for s in S:
        x = out["systems"][s]
        print(f"{s:24s} AUROC {x['auroc']:.3f} ({x['auroc_ci'][0]:.3f}-{x['auroc_ci'][1]:.3f}) mean_p {x['mean_p']:.3f} O/E {x['observed_to_expected']:.2f}",
              "" if s == "jev" else f"jev-: {x['jev_minus_auroc']['est']:+.3f} ({x['jev_minus_auroc']['ci'][0]:+.3f},{x['jev_minus_auroc']['ci'][1]:+.3f})")
    print("observed", round(obs, 4))


if __name__ == "__main__":
    main()

"""Arm 1 hybrid: Jev answers, with its death probability, the half of the wide-set records on which its top risk-band
probability is highest (ties by record id); each primary chatbot answers the rest (rule and test: jevity.hybrid).
  python scripts/hybrid_arm1.py run [--yes] [--workers 16]   # Jev only: the risk-bands question on every wide-set
                                                             # baseline record; raw replies in runs/eval (as the wide run)
  python scripts/hybrid_arm1.py analyze                      # results/wide/hybrid.json (jevity.hybrid.hybrid_block)
`run` calls no model but Jev; it prints its cost estimate and asks before sending (--yes skips the question), and
writes results/wide/calls_jev_bands.parquet. `analyze` reads that file, results/wide/calls.parquet,
data/cohort.parquet, data/reference_oof.parquet and results/eval/cost_speed.json."""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity.clients import MODELS, RawStore, active_families, estimate_tokens     # noqa: E402
from jevity.runner import Call, execute, make_client_factory                       # noqa: E402
from jevity import hybrid as H                                                     # noqa: E402

OUT = ROOT / "results" / "wide"
BANDS = OUT / "calls_jev_bands.parquet"


def plan() -> list[Call]:
    st = pd.read_parquet(ROOT / "data" / "states_wide.parquet")
    st = st[(st.edit == "baseline") & (~st.annotated)]
    return [Call("jev_bands", int(r.profile), "baseline", False, 0, r.text) for r in st.itertuples(index=False)]


def analyze() -> dict:
    calls = pd.read_parquet(OUT / "calls.parquet")
    bands = pd.read_parquet(BANDS)
    co = pd.read_parquet(ROOT / "data" / "cohort.parquet")
    y = co.set_index(co.SEQN.astype(int))["death_10y"].astype(float)
    o = pd.read_parquet(ROOT / "data" / "reference_oof.parquet").set_index("SEQN")
    chat = active_families(("primary",))
    preds = {}
    for m in ["jev"] + chat:
        b = calls[(calls.model == m) & (calls.edit == "baseline") & calls.valid & (calls.variant == "raw")]
        preds[m] = b.drop_duplicates("profile").set_index("profile")["p"].astype(float)
    for col, name in (("p_spline_logit", "reference_spline_logit"), ("p_lightgbm", "reference_lightgbm"),
                      ("p_age_sex", "reference_age_sex"), ("p_phenoage_10y", "phenoage_10y")):
        if col in o:
            preds[name] = o[col].astype(float)
    common = sorted(set.intersection(*[set(s.dropna().index) for s in preds.values()]) & set(y.index))
    bb = bands[bands.valid].drop_duplicates("profile").set_index("profile")
    conf = bb["band_probabilities"].map(H.top_probability).reindex(common).to_numpy(float)
    own = bb["confidence"].astype(float).reindex(common).to_numpy(float)
    cs = json.loads((ROOT / "results" / "eval" / "cost_speed.json").read_text())["systems"]
    price = float(MODELS["jev"]["price_per_mtok_input"])
    cost = {m: float(cs[m]["usd_list_per_usable"]) for m in ["jev"] + chat}
    cost["jev_bands"] = float(bands.tokens_in.mean()) * price / 1e6
    P = {k: preds[k].reindex(common).to_numpy(float) for k in ["jev"] + chat + ["reference_spline_logit", "reference_age_sex"]}
    res = H.hybrid_block(P, y.reindex(common).to_numpy(float), conf, np.array(common), chat, conf_own=own, cost=cost)
    bp = bb["p"].astype(float).reindex(common)
    jp = preds["jev"].reindex(common)
    res["bands_vs_primary"] = {"n": int(bp.notna().sum()), "mean_band_implied": float(bp.mean()),
                               "mean_primary": float(jp[bp.notna()].mean()),
                               "mean_abs_difference": float((bp - jp).abs().mean())}
    res["meta"] = {"rule": "docs/arm1_hybrid_rule.md", "added_after_outcomes_seen": True, "band_calls": int(len(bands)),
                   "band_usable": int(bands.valid.sum()), "cost_list_usd_per_answer": cost}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "analyze"])
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--workers", type=int, default=16, help="parallel Jev calls (run)")
    a = ap.parse_args()
    if a.what == "analyze":
        res = analyze()
        (OUT / "hybrid.json").write_text(json.dumps(res, indent=1, default=float))
        print(f"records {res['n_records']:,}, deaths {res['deaths']:,}; Jev answers {res['n_jev']:,} "
              f"({res['deaths_jev_half']} deaths); margin {res['margin']:.4f}; critical value {res['critical_value']:.2f}")
        for s, v in res["chatbots"].items():
            lo, hi = v["ci_simultaneous"]
            print(f"  hybrid with {s:7s} {v['hybrid_log_loss']:.4f} vs {v['chatbot_log_loss']:.4f}: diff {v['diff']:+.4f} "
                  f"({lo:+.4f} to {hi:+.4f}) {v['outcome']}; confidence minus random {v['confidence_minus_random']:+.4f}")
        sys.exit(0)
    calls = plan()
    ev = pd.read_parquet(ROOT / "results" / "eval" / "calls.parquet", columns=["model", "tokens_in"])
    ev = ev[(ev.model == "jev_bands") & ev.tokens_in.notna()]          # measured band-question tokens (evaluation run)
    tok = float(ev.tokens_in.mean()) if len(ev) else float(np.mean([estimate_tokens(c.text) for c in calls])) + 250
    usd = len(calls) * tok * float(MODELS["jev"]["price_per_mtok_input"]) / 1e6
    print(f"Jev risk bands on the wide set: {len(calls):,} calls (Jev only), about ${usd:.2f} at list price")
    if not a.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    df = execute(calls, make_client_factory(RawStore(ROOT / "runs" / "eval"), "nhanes"), "nhanes", BANDS,
                 workers={"default": a.workers})
    print(df.groupby("model")["valid"].agg(["size", "sum", "mean"]).to_string())
    print(df.groupby("provider").size().to_string())
    print(f"wrote {BANDS}")

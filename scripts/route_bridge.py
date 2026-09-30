"""Route bridge (secondary, NHANES only): the first 100 evaluation records in the seeded block order are sent, as
baselines, through the standard (synchronous) route for each active batch-route family (GPT-5.6 Sol and Luna, Claude
Opus 5.5, Gemini 3.8 Flash and Gemini 3.5 Flash-Lite; disabled families are skipped). Reports agreement with the batch
answers for the same records and synchronous latency. D never uses these calls; they share the raw-response store
under their own request identity. Prints its cost estimate and asks before sending.
  python scripts/route_bridge.py [--n 100] [--yes]      (after the batch answers are collected)
  python scripts/route_bridge.py --from-calls           # recompute route_bridge.json from route_bridge_calls.parquet"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity.clients import LLMClient, RawStore, MODELS, PROMPTS, estimate_tokens, active_families
from jevity.runner import Call, block_order

ROOT = Path(__file__).resolve().parents[1]
CLIP = (0.005, 0.995)


def logit(p):
    p = np.clip(np.asarray(p, float), *CLIP)
    return np.log(p / (1 - p))


def summarise(df: pd.DataFrame) -> dict:
    """Per family: agreement of the standard-route answers with the batch answers for the same records, and latency."""
    out = {}
    for f, g in df.groupby("family"):
        ok = g.dropna(subset=["p_batch", "p_standard"])
        d = np.abs(logit(ok.p_standard) - logit(ok.p_batch)) if len(ok) else np.array([])
        out[f] = {"n_both_usable": int(len(ok)), "n_records": int(len(g)),
                  "share_identical_0.005": float((np.abs(ok.p_standard - ok.p_batch) <= 0.005).mean()) if len(ok) else None,
                  "mean_abs_diff_logit": float(d.mean()) if len(d) else None,
                  "mean_diff_logit_standard_minus_batch": float((logit(ok.p_standard) - logit(ok.p_batch)).mean()) if len(ok) else None,
                  "median_latency_s": float(g.latency_s.median()) if g.latency_s.notna().any() else None,
                  "models_reported": sorted({str(x) for x in pd.concat([g.model_batch, g.model_standard]).dropna()})}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--yes", action="store_true")
    ap.add_argument("--from-calls", action="store_true", help="recompute the summary from results/eval/route_bridge_calls.parquet")
    a = ap.parse_args()
    res_dir = ROOT / "results" / "eval"
    if a.from_calls:
        out = summarise(pd.read_parquet(res_dir / "route_bridge_calls.parquet"))
        (res_dir / "route_bridge.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        sys.exit(0)
    st = pd.read_parquet(ROOT / "data" / "states_eval.parquet")
    base = st[(st.edit == "baseline") & ~st.annotated]
    order = [c.profile for c in block_order([Call("x", int(p), "baseline", False, 0, "") for p in base.profile])]
    first = list(dict.fromkeys(order))[: a.n]
    base = base.set_index("profile").loc[first].reset_index()
    fams = [f for f in (MODELS.get("batch") or {}).get("families", {}) if f in active_families()]   # never a disabled family
    tok = base.text.map(estimate_tokens).mean() + 140
    cost = sum(len(base) * (tok * MODELS["families"][f]["price_in"] + 400 * MODELS["families"][f]["price_out"]) / 1e6
               for f in fams)
    print(f"route bridge: {len(base)} records x {fams}, about ${cost:.2f} on the standard route")
    if not a.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit(0)
    store = RawStore(ROOT / "runs" / "eval")
    stmt = PROMPTS["statements"]["nhanes"]
    rows = []
    for f in fams:
        for r in base.itertuples(index=False):
            b = LLMClient(f, store).probability(r.text, stmt)                 # cached batch answer (never sent here)
            s = LLMClient(f, store, standard=True).probability(r.text, stmt)  # standard route
            rows.append({"family": f, "profile": int(r.profile), "p_batch": b.get("p"), "p_standard": s.get("p"),
                         "latency_s": s.get("latency_s"), "provider_standard": s.get("provider"),
                         "model_batch": b.get("model_reported"), "model_standard": s.get("model_reported")})
    df = pd.DataFrame(rows)
    out = summarise(df)
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(res_dir / "route_bridge_calls.parquet", index=False)
    (res_dir / "route_bridge.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

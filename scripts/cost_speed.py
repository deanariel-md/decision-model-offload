"""Cost, speed and value for arm 1: per system, cost per usable estimate as billed (the batch price where the batch
route was used) and at the standard list price, with mean tokens in and out (every call counts in the cost; a failed
call adds its billed tokens and no usable estimate); runtime from the timing sample (results/eval/timing_sample.json:
median and 90th percentile seconds, one question at a time); baseline accuracy against observed deaths (log-loss and
Brier, from the prediction block of results/eval/analysis.json) and the mean per-call error on the edits (|z - d|,
log-odds); which systems are dominated (another is no more expensive at list price and at least as accurate, strictly
better on one of the two); and, per primary chatbot, the non-inferiority test of Jev's baseline prediction
(results/eval/analysis.json) with the cost ratio. Every number comes from results/eval/*.json and calls.parquet.
The columns arms 2 and 3 also report, on the main pass (first raw answers): input, output and reasoning tokens per
answer (reasoning as the raw reply reports it, jevity.supplement.reasoning_tokens; output includes it), billed and list
cost per 1,000 answers and list cost of one million answers. Reasoning tokens are read once from the raw replies and
kept in results/<split>/reasoning_tokens.parquet (keyed by calls.parquet's raw_path); a reply already in that file is
not read again, so the file alone reproduces the reasoning columns.
  python scripts/cost_speed.py [eval]
Writes results/<split>/cost_speed.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODELS = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
METRICS = {"log_loss": "Baseline log-loss (lower is better)", "brier": "Baseline Brier score (lower is better)",
           "per_call_error": "Mean per-call error |z − d| (log-odds)"}


def main_pass(calls: pd.DataFrame) -> pd.Series:
    """First raw answers (repeat 0, unannotated, raw variant): the answers the per-answer columns count."""
    ok = pd.Series(True, index=calls.index)
    if "repeat" in calls:
        ok &= calls["repeat"] == 0
    if "annotated" in calls:
        ok &= ~calls["annotated"].astype(bool)
    if "variant" in calls:
        ok &= calls["variant"] == "raw"
    return ok


def reasoning_by_call(calls: pd.DataFrame, cache: Path | None = None, workers: int = 8) -> pd.Series:
    """Reasoning tokens per main-pass chatbot call, from its raw reply (NaN where none is reported), indexed like calls.
    `cache` (parquet: raw_path, reasoning_tokens) is read first and extended with the replies not yet in it."""
    import json as _json
    from concurrent.futures import ThreadPoolExecutor
    from jevity.supplement import reasoning_tokens
    rows = calls[main_pass(calls) & ~calls.model.str.startswith("jev")]
    have = pd.read_parquet(cache) if cache is not None and cache.exists() else         pd.DataFrame({"raw_path": pd.Series(dtype=str), "reasoning_tokens": pd.Series(dtype=float)})
    pending = sorted(set(rows.raw_path.dropna().astype(str)) - set(have.raw_path))

    def one(p: str) -> float:
        f = Path(p) if Path(p).is_absolute() else ROOT / p
        try:
            v = reasoning_tokens(_json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            v = None
        return np.nan if v is None else float(v)

    if pending:
        with ThreadPoolExecutor(workers) as ex:
            vals = list(ex.map(one, pending, chunksize=256))
        have = pd.concat([have, pd.DataFrame({"raw_path": pending, "reasoning_tokens": vals})], ignore_index=True)
        if cache is not None:
            have.to_parquet(cache, index=False)
    m = have.drop_duplicates("raw_path").set_index("raw_path")["reasoning_tokens"]
    return rows.raw_path.astype(str).map(m).reindex(calls.index)


def call_costs(calls: pd.DataFrame, models_cfg: dict = MODELS, reasoning: pd.Series | None = None) -> dict:
    """Per calls.model value (Jev's probability question as 'jev'; its secondary questions left out): calls, usable,
    mean tokens, billed and list-price totals and per usable estimate; and, on the main pass, tokens per answer and cost
    per 1,000 and per million answers (the columns arms 2 and 3 use). `reasoning`: reasoning_by_call."""
    out = {}
    first = main_pass(calls)
    for m, g in calls.groupby("model"):
        if m.startswith("jev") and m != "jev":
            continue
        fam = "jev" if m == "jev" else m.split("@")[0]
        tin = g.tokens_in.fillna(0).to_numpy(float) if "tokens_in" in g else np.zeros(len(g))
        tout = g.tokens_out.fillna(0).to_numpy(float) if "tokens_out" in g else np.zeros(len(g))
        if fam == "jev":
            billed = listp = tin * models_cfg["jev"]["price_per_mtok_input"] / 1e6
        elif fam in models_cfg["families"]:
            f = models_cfg["families"][fam]
            b = ((models_cfg.get("batch") or {}).get("families") or {}).get(fam, f)
            batch = (g["route"] == "batch").to_numpy() if "route" in g else np.zeros(len(g), bool)
            listp = (tin * f["price_in"] + tout * f["price_out"]) / 1e6
            billed = np.where(batch, (tin * b["price_in"] + tout * b["price_out"]) / 1e6, listp)
        else:
            continue
        n_ok = int(g.valid.sum())
        routes = g["route"].fillna("none").value_counts().to_dict() if "route" in g else {}
        out[m] = {"calls": int(len(g)), "usable": n_ok, "failed": int(len(g) - n_ok),
                  "mean_tokens_in": float(tin.mean()) if len(g) else None,
                  "mean_tokens_out": float(tout.mean()) if len(g) else None,
                  "usd_billed_total": float(billed.sum()), "usd_list_total": float(listp.sum()),
                  "usd_billed_per_usable": float(billed.sum() / n_ok) if n_ok else None,
                  "usd_list_per_usable": float(listp.sum() / n_ok) if n_ok else None,
                  "routes": {str(k): int(v) for k, v in routes.items()},
                  "price_assumed": (models_cfg["families"].get(fam) or {}).get("transport") == "hf_router"}
        f = first.loc[g.index].to_numpy()
        n = int(f.sum())
        rt = (reasoning.loc[g.index][f] if reasoning is not None else pd.Series(dtype=float)).dropna()
        out[m] |= {"answers": n,
                   "input_tokens_per_answer": float(tin[f].mean()) if n else None,
                   "output_tokens_per_answer": float(tout[f].mean()) if n and fam != "jev" else None,
                   "reasoning_tokens_per_answer": float(rt.mean()) if len(rt) else None,
                   "answers_reporting_reasoning": int(len(rt)),
                   "usd_billed_per_1000_answers": float(billed[f].mean() * 1000) if n else None,
                   "usd_list_per_1000_answers": float(listp[f].mean() * 1000) if n else None,
                   "usd_list_per_million_answers": float(listp[f].mean() * 1e6) if n else None}
    return out


def dominated(points: dict[str, tuple[float, float]]) -> dict[str, list[str]]:
    """points: system -> (list-price cost, error; lower is better for both). A system is dominated by every other one
    that costs no more and is at least as accurate, and is strictly better on at least one."""
    out = {}
    for a, (ca, ea) in points.items():
        out[a] = sorted(b for b, (cb, eb) in points.items()
                        if b != a and cb <= ca and eb <= ea and (cb < ca or eb < ea))
    return out


def value(res: dict, costs: dict, timing: dict | None) -> dict:
    tiers = res["meta"].get("tiers", {})
    pred = (res.get("prediction") or {}).get("systems", {})
    err = res.get("per_call_error", {})
    full = [m for m in res["meta"]["models"] if "@" not in m and tiers.get(m, "primary") in ("jev", "primary", "free", "pair")]
    rows = {}
    for m in res["meta"]["models"]:
        c = costs.get(m) or {}
        t = ((timing or {}).get("systems") or {}).get(m) or {}
        rows[m] = {"tier": tiers.get(m), **c,
                   "timing_median_s": t.get("median_s"), "timing_p90_s": t.get("p90_s"), "timing_n": t.get("n"),
                   "log_loss": (pred.get(m) or {}).get("log_loss"), "log_loss_ci": (pred.get(m) or {}).get("log_loss_ci"),
                   "brier": (pred.get(m) or {}).get("brier"), "brier_ci": (pred.get(m) or {}).get("brier_ci"),
                   "per_call_error": (err.get(m) or {}).get("mae_logit"),
                   "per_call_error_ci": (err.get(m) or {}).get("mae_logit_ci")}
    dom = {}
    for k in METRICS:
        pts = {m: (rows[m]["usd_list_per_usable"], rows[m][k]) for m in full
               if rows[m].get("usd_list_per_usable") is not None and rows[m].get(k) is not None}
        dom[k] = dominated(pts)
    for m in rows:
        rows[m]["dominated_by"] = {k: dom[k].get(m) for k in METRICS if m in dom[k]}
    jev = rows.get("jev") or {}
    ni, ni_b = (res.get("prediction") or {}).get("focal_vs_llms", {}), (res.get("prediction") or {}).get("focal_vs_llms_brier", {})
    cheaper = {}
    for m in [x for x in full if tiers.get(x) == "primary"]:
        p = (ni.get("pairs") or {}).get(f"jev - {m}")
        if not p:
            continue
        lr = (rows[m].get("usd_list_per_usable") / jev["usd_list_per_usable"]
              if rows[m].get("usd_list_per_usable") and jev.get("usd_list_per_usable") else None)
        br = (rows[m].get("usd_billed_per_usable") / jev["usd_billed_per_usable"]
              if rows[m].get("usd_billed_per_usable") and jev.get("usd_billed_per_usable") else None)
        cheaper[m] = {"log_loss_diff_jev_minus": p["diff"], "ci_simultaneous": p["ci_simultaneous"],
                      "margin": p.get("noninferiority_margin"), "noninferior": p.get("noninferior"),
                      "brier_diff_jev_minus": ((ni_b.get("pairs") or {}).get(f"jev - {m}") or {}).get("diff"),
                      "cost_ratio_list": lr, "cost_ratio_billed": br,
                      "acceptable_cheaper_choice": bool(p.get("noninferior")) and lr is not None and lr > 1}
    return {"systems": rows, "dominated": dom, "jev_vs_primary": cheaper,
            "n_prediction_records": (res.get("prediction") or {}).get("n_records"),
            "notes": {"billed": "tokens x the route's configured price (batch price for batch answers); the Hugging Face "
                                "pair at an assumed price until billed", "list": "tokens x the standard price",
                      "jev": "Jev's probability question only; its secondary questions are left out",
                      "dominance": "among systems answered on the full evaluation set, at list price per usable estimate",
                      "per_answer": "main pass (first raw answers, unusable ones included); reasoning tokens as the raw "
                                    "reply reports them, part of the output tokens; None where no reply reports them"}}


if __name__ == "__main__":
    split = next((x for x in sys.argv[1:] if not x.startswith("--")), "eval")
    out = ROOT / "results" / split
    res = json.loads((out / "analysis.json").read_text())
    calls = pd.read_parquet(out / "calls.parquet")
    tpath = out / "timing_sample.json"
    timing = json.loads(tpath.read_text()) if tpath.exists() else None
    rt = None if "--no_reasoning" in sys.argv else reasoning_by_call(calls, out / "reasoning_tokens.parquet")
    v = value(res, call_costs(calls, reasoning=rt), timing)
    v["timing_sample"] = None if timing is None else {k: timing.get(k) for k in ("n_records", "window_min", "started",
                                                                                "finished", "store", "note", "simulated")}
    (out / "cost_speed.json").write_text(json.dumps(v, indent=1, default=float))
    for m, r in v["systems"].items():
        print(f"{m:14s} usable {r.get('usable')}/{r.get('calls')}  billed/usable ${r.get('usd_billed_per_usable') or 0:.5f}  "
              f"list/usable ${r.get('usd_list_per_usable') or 0:.5f}  median {r.get('timing_median_s')} s  "
              f"dominated {r.get('dominated_by')}")
    for m, c in v["jev_vs_primary"].items():
        print(f"Jev vs {m}: non-inferior {c['noninferior']}, list-price cost ratio {c['cost_ratio_list']}, "
              f"acceptable cheaper choice {c['acceptable_cheaper_choice']}")
    print(f"wrote {out / 'cost_speed.json'}")

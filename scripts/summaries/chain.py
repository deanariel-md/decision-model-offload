"""Three-step chain: Jev answers its most confident half (the tested rule, hybrid_share.jev_items); of the
items Jev passes on, a cheap chatbot answers the half on which its own top probability is highest (ties by item id);
an expensive chatbot answers the rest. Beside it: cheap -> expensive with the same half rule, Jev -> expensive (the
tested hybrid) and each system alone. Cost at list price: every step pays for every item it sees. Stored answers only.

  python scripts/summaries/chain.py -> results/summaries/chain.json
Reads the stored calls and keys listed in hybrid_all.py and results/summaries/timing_summary.json (run
timing_summary.py first; the expected median time of each chain comes from those timing samples).
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from hybrid_all import SETS, score  # noqa: E402
from cost_frontier import PRICE, NAME, MAIN  # noqa: E402

CHEAP = ["gpt_free", "gemini_free", "medgemma", "gemma"]
J = lambda p: json.loads(Path(p).read_text())
TIMING = {"next_step": "next_step", "board_exam": "board_exam", "staging_names": "staging", "staging_definitions": "staging"}


def half_rule(top, eligible, ids):
    """Most confident half of the eligible items by top probability, ties by item id."""
    el = [i for i in range(len(ids)) if eligible[i]]
    order = sorted(el, key=lambda i: (-(top[i] if np.isfinite(top[i]) else -1.0), ids[i]))
    keep = np.zeros(len(ids), bool); keep[order[: len(el) // 2]] = True
    return keep


def main(B=2000, seed=20260927):
    rng = np.random.default_rng(seed)
    tm = J("results/summaries/timing_summary.json")
    res = {"note": __doc__.split("\n\n")[0], "sets": {}}
    for key, label, calls, variant, afile, truth_fn, metric, _ in SETS:
        an = J(afile)
        c = pd.read_parquet(calls)
        c = c[(c["variant"] == variant) & (c["repeat"] == 0) & c["system"].isin(PRICE)].copy()
        ids = sorted(c.item_id.unique())
        truth = pd.Series(truth_fn(ids)).reindex(ids)
        c["valid"] = c["valid"].fillna(False).astype(bool)
        c["ok"] = c["valid"] & (c["answer"].values == truth.reindex(c["item_id"]).values)
        c["usd"] = [(ti * PRICE[s][0] + to * PRICE[s][1]) / 1e6 for s, ti, to in zip(c.system, c.tokens_in.fillna(0), c.tokens_out.fillna(0))]
        OK = c.pivot_table(index="item_id", columns="system", values="ok", aggfunc="first").reindex(ids).astype(bool)
        U = c.pivot_table(index="item_id", columns="system", values="usd", aggfunc="first").reindex(ids).fillna(0.0)
        TOP = c.pivot_table(index="item_id", columns="system", values="top_prob", aggfunc="first").reindex(ids)
        TOP = TOP.where(c.pivot_table(index="item_id", columns="system", values="valid", aggfunc="first").reindex(ids).fillna(False).astype(bool), -1.0)
        strat = (truth == "A").to_numpy() if metric == "balanced_accuracy" else None
        n = len(ids); idx = rng.integers(0, n, size=(B, n))
        use_j = np.array([i in set(an["hybrid_share"]["jev_items"]) for i in ids])
        per = lambda u: float(1000 * u.sum() / n)
        t = tm.get(TIMING.get(key, ""), {})
        med = lambda s: t.get(s, {}).get("median_s") if t else None
        out = {"label": label, "metric": metric, "n_items": n, "jev_alone": float(score(OK["jev"].to_numpy(), strat)), "chains": {}}
        for e in MAIN:
            for ch in CHEAP:
                use_c = half_rule(TOP[ch].to_numpy(float), ~use_j, ids)
                use_e = ~use_j & ~use_c
                chain = np.where(use_j, OK["jev"], np.where(use_c, OK[ch], OK[e]))
                cheap_exp = np.where(half_rule(TOP[ch].to_numpy(float), np.ones(n, bool), ids), OK[ch], OK[e])
                use_c2 = half_rule(TOP[ch].to_numpy(float), np.ones(n, bool), ids)
                e_alone = OK[e].to_numpy()
                d = score(chain[idx], None if strat is None else strat[idx]) - score(e_alone[idx], None if strat is None else strat[idx])
                d2 = score(chain[idx], None if strat is None else strat[idx]) - score(cheap_exp[idx], None if strat is None else strat[idx])
                cost_chain = per(U["jev"]) + per(U[ch].where(~use_j, 0)) + per(U[e].where(use_e, 0))
                cost_ce = per(U[ch]) + per(U[e].where(~use_c2, 0))
                lat = None
                if all(med(s) is not None for s in ("jev", ch, e)):
                    lat = {"chain": med("jev") + (~use_j).mean() * med(ch) + use_e.mean() * med(e),
                           "cheap_then_expensive": med(ch) + (~use_c2).mean() * med(e), "expensive_alone": med(e)}
                out["chains"][f"jev>{ch}>{e}"] = {
                    "cheap": ch, "expensive": e, "share": {"jev": float(use_j.mean()), "cheap": float(use_c.mean()), "expensive": float(use_e.mean())},
                    "accuracy": {"chain": float(score(chain, strat)), "cheap_then_expensive": float(score(cheap_exp, strat)),
                                 "jev_then_expensive": float(score(np.where(use_j, OK["jev"], OK[e]), strat)),
                                 "expensive_alone": float(score(e_alone, strat)), "cheap_alone": float(score(OK[ch].to_numpy(), strat))},
                    "chain_minus_expensive_pts": {"est": float(100 * (score(chain, strat) - score(e_alone, strat))),
                                                  "ci": [float(100 * np.nanquantile(d, 0.025)), float(100 * np.nanquantile(d, 0.975))]},
                    "chain_minus_cheap_then_expensive_pts": {"est": float(100 * (score(chain, strat) - score(cheap_exp, strat))),
                                                             "ci": [float(100 * np.nanquantile(d2, 0.025)), float(100 * np.nanquantile(d2, 0.975))]},
                    "usd_per_1000": {"chain": cost_chain, "cheap_then_expensive": cost_ce,
                                     "jev_then_expensive": per(U["jev"]) + per(U[e].where(~use_j, 0)),
                                     "expensive_alone": per(U[e]), "cheap_alone": per(U[ch])},
                    "median_seconds_expected": lat}
        res["sets"][key] = out
    Path("results/summaries/chain.json").write_text(json.dumps(res, indent=1))
    for k, v in res["sets"].items():
        print("\n" + v["label"], "Jev alone", round(100 * v["jev_alone"], 1))
        for nm in ("jev>gpt_free>gpt", "jev>gpt_free>claude", "jev>gemini_free>gpt", "jev>gemma>gpt"):
            x = v["chains"][nm]; a = x["accuracy"]; u = x["usd_per_1000"]
            print(f"  {nm:22s} chain {100*a['chain']:5.1f} ${u['chain']:.3f} | cheap>exp {100*a['cheap_then_expensive']:5.1f} ${u['cheap_then_expensive']:.3f} | "
                  f"jev>exp {100*a['jev_then_expensive']:5.1f} ${u['jev_then_expensive']:.3f} | exp {100*a['expensive_alone']:5.1f} ${u['expensive_alone']:.3f} | "
                  f"d_exp {x['chain_minus_expensive_pts']['est']:+.1f} {[round(q,1) for q in x['chain_minus_expensive_pts']['ci']]} lat {x['median_seconds_expected']}")


if __name__ == "__main__":
    main()

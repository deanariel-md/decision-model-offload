"""Jev-first hybrid with every system as the partner, on every task with a categorical answer (stored answers only,
no model calls).

Rule as in each analysis file: Jev answers the items in hybrid_share.jev_items (its most confident half, ties by item
id), the partner answers the rest. Accuracy as in the analysis (balanced accuracy for patient messages, accuracy for
board-examination questions, exact accuracy for staging and kidney injury; an unusable answer is wrong). Cost at
standard list price from each answer's tokens: Jev on every item plus the partner on the items Jev passes on. For the
five main chatbots the hybrid accuracy and cost must equal the analysis file's; the prices must equal those in
config/models.yaml (cost_frontier._check_prices).

  python scripts/summaries/hybrid_all.py -> results/summaries/hybrid_all.json
Keys: patient messages from the item identifier; board examination from data/arm2_ext/items.csv; staging from
data/arm3/items.csv; kidney injury from the two answer sets that scored 1.000 in results/eicu_aki (as sens_spec.py).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent))
from cost_frontier import PRICE, NAME, MAIN, CHEAP, _check_prices  # noqa: E402

PARTNERS = MAIN + CHEAP
B = 2000
SEED = 20260927
J = lambda p: json.loads(Path(p).read_text())


def arm2_truth(ids):
    return {i: ("B" if "_lv_" in i else "A") for i in ids}


def board_truth(ids):
    it = pd.read_csv("data/arm2_ext/items.csv", dtype=str)
    return dict(zip(it["item_id"], it["truth"]))


def staging_truth(ids):
    it = pd.read_csv("data/arm3/items.csv", dtype=str)
    return dict(zip(it["report_id"], it["level"]))


def aki_truth(ids):  # reproduced from two answer sets that scored 1.000 against the key (as sens_spec.py)
    ck = pd.read_parquet("results/eicu_aki/calls.parquet")
    dk = J("results/eicu_aki/analysis_definitions.json")
    assert dk["systems"]["gpt"]["exact_accuracy"] == 1.0 and dk["systems"]["gemini"]["exact_accuracy"] == 1.0
    d0 = ck[(ck["variant"] == "definitions") & (ck["repeat"] == 0)]
    g1 = d0[d0["system"] == "gpt"].set_index("item_id")["answer"]
    g2 = d0[d0["system"] == "gemini"].set_index("item_id")["answer"].reindex(g1.index)
    assert (g1 == g2).all()
    return g1.to_dict()


SETS = [  # key, label, calls, variant, analysis file, truth, metric, cluster
    ("next_step", "Patient messages", "results/arm2/calls.parquet", "main", "results/arm2/analysis.json", arm2_truth,
     "balanced_accuracy", lambda i: i.split("_lv_")[0].split("_ac_")[0]),
    ("board_exam", "Board-examination questions", "results/arm2_ext/calls.parquet", "medhelm",
     "results/arm2_ext/analysis.json", board_truth, "accuracy", None),
    ("staging_names", "Colon cancer staging, names", "results/arm3/calls.parquet", "names", "results/arm3/analysis.json",
     staging_truth, "exact_accuracy", None),
    ("staging_definitions", "Colon cancer staging, definitions", "results/arm3/calls.parquet", "definitions",
     "results/arm3/analysis_definitions.json", staging_truth, "exact_accuracy", None),
    ("aki_names", "Kidney injury, names", "results/eicu_aki/calls.parquet", "names", "results/eicu_aki/analysis.json",
     aki_truth, "exact_accuracy", None),
    ("aki_definitions", "Kidney injury, definitions", "results/eicu_aki/calls.parquet", "definitions",
     "results/eicu_aki/analysis_definitions.json", aki_truth, "exact_accuracy", None),
]


def score(ok, strat):
    """ok: bool array over items; strat: None or bool array (balanced accuracy over two strata)."""
    if strat is None:
        return ok.mean(axis=-1)
    return (np.where(strat, ok, 0).sum(-1) / strat.sum(-1) + np.where(~strat, ok, 0).sum(-1) / (~strat).sum(-1)) / 2


def boot_diff(a, b, strat, idx):
    """Mean of a minus b under resampled item index rows idx (shape B x n)."""
    s = None if strat is None else strat[idx]
    return score(a[idx], s) - score(b[idx], s)


def run_set(key, label, calls, variant, afile, truth_fn, metric, cluster, rng):
    an = J(afile)
    assert an.get("variant", variant) == variant, (afile, an.get("variant"))
    c = pd.read_parquet(calls)
    c = c[(c["variant"] == variant) & (c["repeat"] == 0) & c["system"].isin(PRICE)].copy()
    ids = sorted(c["item_id"].unique())
    truth = pd.Series(truth_fn(ids)).reindex(ids)
    assert truth.notna().all(), key
    c["valid"] = c["valid"].fillna(False).astype(bool)
    c["ok"] = c["valid"] & (c["answer"].values == truth.reindex(c["item_id"]).values)
    c["usd"] = [(ti * PRICE[s][0] + to * PRICE[s][1]) / 1e6
                for s, ti, to in zip(c["system"], c["tokens_in"].fillna(0), c["tokens_out"].fillna(0))]
    OK = c.pivot_table(index="item_id", columns="system", values="ok", aggfunc="first").reindex(ids).astype(bool)
    U = c.pivot_table(index="item_id", columns="system", values="usd", aggfunc="first").reindex(ids).fillna(0.0)
    strat = (truth == "A").to_numpy() if metric == "balanced_accuracy" else None
    for s in ["jev"] + PARTNERS:
        acc = score(OK[s].to_numpy(), strat)
        assert abs(acc - an["systems"][s][metric]) < 1e-9, (key, s, acc, an["systems"][s][metric])
        assert abs(U[s].sum() - an["systems"][s]["cost"]["usd_list_total"]) < 1e-6, (key, s)
    use_j = pd.Series(OK.index.isin(set(an["hybrid_share"]["jev_items"])), index=ids).to_numpy()
    n = len(ids)
    idx = rng.integers(0, n, size=(B, n))
    groups = None
    if cluster:
        g = pd.Series([cluster(i) for i in ids])
        gl = sorted(g.unique()); gi = [np.flatnonzero(g.to_numpy() == x) for x in gl]
        gidx = []
        for _ in range(B):
            pick = rng.integers(0, len(gl), size=len(gl))
            gidx.append(np.concatenate([gi[k] for k in pick]))
        groups = {"n_groups": len(gl), "idx": gidx}
    jev = OK["jev"].to_numpy()
    usd_jev = float(U["jev"].sum())
    out = {"label": label, "metric": metric, "n_items": n, "n_jev": int(use_j.sum()),
           "jev_alone": {"accuracy": float(score(jev, strat)), "usd_per_1000": 1000 * usd_jev / n},
           "partners": {}}
    for s in PARTNERS:
        p = OK[s].to_numpy()
        h = np.where(use_j, jev, p)
        acc_h, acc_p = float(score(h, strat)), float(score(p, strat))
        usd_h = usd_jev + float(U[s].where(~use_j, 0.0).sum())
        usd_p = float(U[s].sum())
        if s in MAIN:
            hs = an["hybrid_share"]["comparisons"][s]
            assert abs(acc_h - hs[metric]) < 1e-9, (key, s, acc_h, hs[metric])
            assert abs(usd_h - an["hybrid_cost"][s]["hybrid_usd_total"]) < 1e-6, (key, s, usd_h)
        d = boot_diff(h, p, strat, idx)
        rand = float(score(0.5 * jev + 0.5 * p, strat))  # expected accuracy of a random half to Jev
        r = {"name": NAME[s], "group": "main" if s in MAIN else "other",
             "partner_alone": acc_p, "hybrid": acc_h, "diff_pts": 100 * (acc_h - acc_p),
             "diff_ci_pts": [float(100 * np.quantile(d, 0.025)), float(100 * np.quantile(d, 0.975))],
             "confident_minus_random_pts": 100 * (acc_h - rand),
             "usd_per_1000_partner": 1000 * usd_p / n, "usd_per_1000_hybrid": 1000 * usd_h / n,
             "cost_share": usd_h / usd_p if usd_p else None}
        if groups:
            dg = np.array([score(h[ix], strat[ix]) - score(p[ix], strat[ix]) for ix in groups["idx"]])
            r["diff_ci_pts_by_recommendation"] = [float(100 * np.nanquantile(dg, 0.025)), float(100 * np.nanquantile(dg, 0.975))]
        out["partners"][s] = r
    if groups:
        out["n_recommendations"] = groups["n_groups"]
    # cost-accuracy frontier over Jev alone, every partner alone and every hybrid
    pts = [("jev", out["jev_alone"]["accuracy"], out["jev_alone"]["usd_per_1000"])]
    for s, r in out["partners"].items():
        pts += [(s, r["partner_alone"], r["usd_per_1000_partner"]), ("jev+" + s, r["hybrid"], r["usd_per_1000_hybrid"])]
    out["frontier"] = sorted([a for a, acc, u in pts if not any((b2 >= acc and v <= u) and (b2 > acc or v < u)
                                                                  for _, b2, v in pts)],
                             key=lambda a: next(u for x, _, u in pts if x == a))
    return out


def main():
    _check_prices()
    rng = np.random.default_rng(SEED)
    res = {"note": "Jev-first hybrid (Jev's most confident half, ties by item id) with every system as partner; stored "
                   "answers; cost at standard list price; 95% bootstrap intervals over items (and over recommendations "
                   "for patient messages)", "B": B, "seed": SEED, "sets": {}}
    for args in SETS:
        res["sets"][args[0]] = run_set(*args, rng)
    S = res["sets"]
    rngd = lambda xs: {"min": min(xs), "max": max(xs)}
    res["summary"] = {k: {"other_diff_pts": rngd([S[k]["partners"][s]["diff_pts"] for s in CHEAP]),
                          "other_cost_share": rngd([S[k]["partners"][s]["cost_share"] for s in CHEAP]),
                          "luna": {x: S[k]["partners"]["gpt_free"][x] for x in
                                   ("partner_alone", "hybrid", "diff_pts", "diff_ci_pts", "usd_per_1000_partner",
                                    "usd_per_1000_hybrid", "cost_share")},
                          "frontier": S[k]["frontier"]} for k in S}
    Path("results/summaries/hybrid_all.json").write_text(json.dumps(res, indent=1))
    for k, v in S.items():
        print(f"\n{v['label']} (n={v['n_items']}; Jev alone {100*v['jev_alone']['accuracy']:.1f}% ${v['jev_alone']['usd_per_1000']:.3f})")
        for s, r in v["partners"].items():
            ci = r["diff_ci_pts"]
            extra = f" rec-CI {r['diff_ci_pts_by_recommendation'][0]:+.1f} to {r['diff_ci_pts_by_recommendation'][1]:+.1f}" if "diff_ci_pts_by_recommendation" in r else ""
            print(f"  {r['name']:24s} alone {100*r['partner_alone']:5.1f} ${r['usd_per_1000_partner']:6.3f} | hybrid {100*r['hybrid']:5.1f} "
                  f"${r['usd_per_1000_hybrid']:6.3f} ({100*r['cost_share']:3.0f}%) diff {r['diff_pts']:+5.1f} ({ci[0]:+.1f} to {ci[1]:+.1f}){extra} "
                  f"vs random {r['confident_minus_random_pts']:+.1f}")
        print("  frontier:", v["frontier"])


if __name__ == "__main__":
    main()

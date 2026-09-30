"""Further hybrids of Jev with chatbots (stored answers only, no model calls, no network).

Every configuration computed is written out, not only the best:

1. Agreement cascade. Jev and a cheap chatbot (GPT-5.6 Luna, Gemini 3.5 Flash-Lite, Claude Sonnet 5) answer every item.
   If both answers are usable and identical, that answer stands; otherwise (any disagreement, or either answer
   unusable) the item goes to the top chatbot (GPT-5.6 Sol, Claude Opus 5.5, Gemini 3.8 Flash). Metric as in
   hybrid_all.py. Difference from the top chatbot alone with a paired 95% percentile bootstrap (2,000 resamples over
   items; for patient messages also over recommendations, as hybrid_all.py). Cost at list price: Jev and the cheap
   chatbot on every item, the top chatbot on the escalated items. Compared with the confident-half hybrid Jev + top
   (hybrid_all.py) and the chain Jev -> cheap -> top (chain.py's rule; the Claude Sonnet 5 chains are not in
   chain.json and are computed here with chain.py's own half_rule).
2. Cost-accuracy frontier per set over every system alone, every Jev-first hybrid (hybrid_all), every chain and
   cheap -> expensive pair (chain.py's rule, chain.json plus Claude Sonnet 5), and the agreement cascades. A point is on
   the frontier when no other point has accuracy at least as high and cost at most as high, with one strictly better.
3. Structured input as a safety net (colon cancer). Jev as its developer documents (results/arm3_docuse, versions
   documented and documented_notes, repeat 0). Joint confidence = product of the top probabilities of the four
   questions (parts.parquet), computed the same way for the main 1,000 reports, the 100 misleading-feature reports and
   the 20 non-regional-site reports (it equals the calls' top_prob; the calls' jev_confidence is the product of Jev's
   own per-question `confidence` fields, a different number, used here only as a sensitivity analysis). Cut per
   version = the median (and, separately, the 25th percentile) of the joint confidence over the MAIN 1,000 reports
   (numpy default quantile). A report is sent to the chatbot when its joint confidence is below the cut (by more than
   1e-9); otherwise Jev's answer stands. The same cut is applied unchanged to the two other sets. Chatbot = each
   of the five main chatbots with the names prompt (main: results/arm3/calls.parquet; misleading-feature:
   results/arm3_confuser; non-regional: results/arm3_nonregional_sites). Key: main = data/arm3/items.csv; misleading-
   feature = Gemini 3.8 Flash's answers, which scored 100/100 in arm3_confuser/analysis.json (reproduced: every
   system's count there must match); non-regional = IVA-IVB for all 20 (summary.json). Cost at list price: Jev
   structured on every report, the chatbot on the reports sent. Intervals: Clopper-Pearson for accuracies; paired
   bootstrap over reports (2,000) for differences.

  python scripts/summaries/hybrid_explore.py -> results/summaries/hybrid_explore.json
Run from the folder that holds results/arm3_docuse and results/arm3_nonregional_sites, after hybrid_all.py,
chain.py, timing_summary.py and a first docuse_summary.py (it reads their outputs in results/summaries/).
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_all import SETS, score, B, SEED  # noqa: E402
from cost_frontier import PRICE, NAME, MAIN, CHEAP as CHEAP_ALL, _check_prices  # noqa: E402
from chain import half_rule, CHEAP as CHAIN_CHEAP, TIMING  # noqa: E402

J = lambda p: json.loads(Path(p).read_text())
CASC_CHEAP = ["gpt_free", "gemini_free", "claude_free"]
CASC_TOP = ["gpt", "claude", "gemini"]
CHAINS_CHEAP = CHAIN_CHEAP + ["claude_free"]
QS = ["t_category", "regional_nodes", "tumor_deposits", "distant_metastasis"]
TOL = 1e-9


# ---------------------------------------------------------------- loading (as hybrid_all.run_set / chain.main)
def load_set(key, label, calls, variant, afile, truth_fn, metric, cluster):
    an = J(afile)
    c = pd.read_parquet(calls)
    c = c[(c["variant"] == variant) & (c["repeat"] == 0) & c["system"].isin(PRICE)].copy()
    ids = sorted(c["item_id"].unique())
    truth = pd.Series(truth_fn(ids)).reindex(ids)
    assert truth.notna().all(), key
    c["valid"] = c["valid"].fillna(False).astype(bool)
    c["ok"] = c["valid"] & (c["answer"].values == truth.reindex(c["item_id"]).values)
    c["usd"] = [(ti * PRICE[s][0] + to * PRICE[s][1]) / 1e6
                for s, ti, to in zip(c["system"], c["tokens_in"].fillna(0), c["tokens_out"].fillna(0))]
    piv = lambda v: c.pivot_table(index="item_id", columns="system", values=v, aggfunc="first").reindex(ids)
    OK = piv("ok").astype(bool)
    U = piv("usd").fillna(0.0)
    V = piv("valid").fillna(False).astype(bool)
    ANS = c.pivot_table(index="item_id", columns="system", values="answer", aggfunc="first").reindex(ids).where(V)
    TOP = piv("top_prob").where(V, -1.0)
    strat = (truth == "A").to_numpy() if metric == "balanced_accuracy" else None
    use_j = np.array([i in set(an["hybrid_share"]["jev_items"]) for i in ids])
    return dict(key=key, label=label, metric=metric, cluster=cluster, ids=ids, n=len(ids), OK=OK, U=U, V=V, ANS=ANS,
                TOP=TOP, strat=strat, use_j=use_j)


def boot_idx(d, rng):
    n = d["n"]
    idx = rng.integers(0, n, size=(B, n))
    gidx = None
    if d["cluster"]:
        g = pd.Series([d["cluster"](i) for i in d["ids"]])
        gl = sorted(g.unique()); gi = [np.flatnonzero(g.to_numpy() == x) for x in gl]
        gidx = [np.concatenate([gi[k] for k in rng.integers(0, len(gl), size=len(gl))]) for _ in range(B)]
    return idx, gidx


def diff_ci(a, b, strat, idx, gidx):
    """Estimate and percentile 95% CI (points) of score(a) - score(b); by recommendation when gidx is given."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = {"est": float(100 * (score(a, strat) - score(b, strat)))}
    s = None if strat is None else strat[idx]
    d = score(a[idx], s) - score(b[idx], s)
    out["ci"] = [float(100 * np.nanquantile(d, 0.025)), float(100 * np.nanquantile(d, 0.975))]
    if gidx is not None:
        dg = np.array([score(a[ix], strat[ix]) - score(b[ix], strat[ix]) for ix in gidx])
        out["ci_by_recommendation"] = [float(100 * np.nanquantile(dg, 0.025)), float(100 * np.nanquantile(dg, 0.975))]
    return out


def verify_against_hybrid_all(d, ha):
    """Every partner alone and every confident-half hybrid (accuracy and cost) must equal hybrid_all.json."""
    OK, U, n, strat, use_j = d["OK"], d["U"], d["n"], d["strat"], d["use_j"]
    h = ha["sets"][d["key"]]
    assert h["n_items"] == n and h["n_jev"] == int(use_j.sum())
    assert abs(score(OK["jev"].to_numpy(), strat) - h["jev_alone"]["accuracy"]) < 1e-12
    n_checked = 0
    for s, r in h["partners"].items():
        p = OK[s].to_numpy()
        hy = np.where(use_j, OK["jev"].to_numpy(), p)
        assert abs(score(p, strat) - r["partner_alone"]) < 1e-12, (d["key"], s)
        assert abs(score(hy, strat) - r["hybrid"]) < 1e-12, (d["key"], s)
        assert abs(1000 * U[s].sum() / n - r["usd_per_1000_partner"]) < 1e-9, (d["key"], s)
        assert abs(1000 * (U["jev"].sum() + U[s].where(~use_j, 0).sum()) / n - r["usd_per_1000_hybrid"]) < 1e-9
        n_checked += 1
    return n_checked


def chain_point(d, ch, e):
    """chain.py's rule: Jev keeps its confident half; of the rest the cheap chatbot keeps the half with its highest top
    probability; the expensive chatbot answers the remainder. Also cheap -> expensive with the same half rule."""
    OK, U, n, ids, use_j, strat = d["OK"], d["U"], d["n"], d["ids"], d["use_j"], d["strat"]
    top = d["TOP"][ch].to_numpy(float)
    use_c = half_rule(top, ~use_j, ids)
    use_e = ~use_j & ~use_c
    chain = np.where(use_j, OK["jev"], np.where(use_c, OK[ch], OK[e]))
    use_c2 = half_rule(top, np.ones(n, bool), ids)
    ce = np.where(use_c2, OK[ch], OK[e])
    per = lambda u: float(1000 * u.sum() / n)
    return {"chain": {"ok": chain, "accuracy": float(score(chain, strat)),
                      "usd_per_1000": per(U["jev"]) + per(U[ch].where(~use_j, 0)) + per(U[e].where(use_e, 0))},
            "cheap_then_expensive": {"ok": ce, "accuracy": float(score(ce, strat)),
                                     "usd_per_1000": per(U[ch]) + per(U[e].where(~use_c2, 0))}}


def frontier(points):
    on = []
    for p in points:
        dom = any((q["accuracy"] >= p["accuracy"] - 1e-12 and q["usd_per_1000"] <= p["usd_per_1000"] + 1e-12) and
                  (q["accuracy"] > p["accuracy"] + 1e-12 or q["usd_per_1000"] < p["usd_per_1000"] - 1e-12) for q in points)
        if not dom:
            on.append(p)
    return sorted(on, key=lambda p: p["usd_per_1000"])


# ---------------------------------------------------------------- analyses 1 and 2
def analyses_1_2(rng):
    ha = J("results/summaries/hybrid_all.json")
    ch_json = J("results/summaries/chain.json")
    tm = J("results/summaries/timing_summary.json")
    out1, out2, checks = {}, {}, {}
    for args in SETS:
        d = load_set(*args)
        key, OK, U, n, strat, use_j, ANS, V = d["key"], d["OK"], d["U"], d["n"], d["strat"], d["use_j"], d["ANS"], d["V"]
        checks[key] = {"hybrid_all_partners_reproduced": verify_against_hybrid_all(d, ha)}
        idx, gidx = boot_idx(d, rng)
        per = lambda u: float(1000 * u.sum() / n)
        acc = lambda x: float(score(np.asarray(x, float), strat))
        jev = OK["jev"].to_numpy()
        t = tm.get(TIMING.get(key, ""), {})
        med = lambda s: t.get(s, {}).get("median_s") if t else None
        # chains: recompute every chain.json entry and check it, add Claude Sonnet 5 with the same rule
        chains = {}
        nchk = 0
        for e in MAIN:
            for c_ in CHAINS_CHEAP:
                cp = chain_point(d, c_, e)
                nm = f"jev>{c_}>{e}"
                if nm in ch_json["sets"][key]["chains"]:
                    ref = ch_json["sets"][key]["chains"][nm]
                    assert abs(cp["chain"]["accuracy"] - ref["accuracy"]["chain"]) < 1e-12, (key, nm)
                    assert abs(cp["chain"]["usd_per_1000"] - ref["usd_per_1000"]["chain"]) < 1e-9, (key, nm)
                    assert abs(cp["cheap_then_expensive"]["accuracy"] - ref["accuracy"]["cheap_then_expensive"]) < 1e-12
                    assert abs(cp["cheap_then_expensive"]["usd_per_1000"] - ref["usd_per_1000"]["cheap_then_expensive"]) < 1e-9
                    nchk += 1
                chains[(c_, e)] = cp
        checks[key]["chain_json_entries_reproduced"] = nchk
        # agreement cascades
        casc = {}
        for c_ in CASC_CHEAP:
            agree = (V["jev"] & V[c_] & (ANS["jev"] == ANS[c_])).to_numpy()
            ok_agree = OK[c_].to_numpy()
            assert (ok_agree[agree] == jev[agree]).all()
            for e in CASC_TOP:
                top = OK[e].to_numpy()
                h = np.where(agree, ok_agree, top)
                usd = per(U["jev"]) + per(U[c_]) + per(U[e].where(~agree, 0))
                usd_top = per(U[e])
                hyb = np.where(use_j, jev, top)
                usd_hyb = per(U["jev"]) + per(U[e].where(~use_j, 0))
                cp = chains[(c_, e)]["chain"]
                lat = None
                if all(med(s) is not None for s in ("jev", c_, e)):
                    lat = {"cascade_parallel_first_step": max(med("jev"), med(c_)) + (~agree).mean() * med(e),
                           "top_alone": med(e)}
                casc[f"agree(jev,{c_})>{e}"] = {
                    "cheap": c_, "top": e, "cheap_name": NAME[c_], "top_name": NAME[e],
                    "accuracy": acc(h), "top_alone": acc(top), "cheap_alone": acc(OK[c_]), "jev_alone": acc(jev),
                    "share_escalated": float((~agree).mean()), "n_escalated": int((~agree).sum()),
                    "agreed": {"n": int(agree.sum()), "correct": int(ok_agree[agree].sum()),
                               "wrong_kept": int((~ok_agree[agree]).sum())},
                    "minus_top_alone_pts": diff_ci(h, top, strat, idx, gidx),
                    "usd_per_1000": usd, "usd_per_1000_top_alone": usd_top, "cost_share_vs_top": usd / usd_top,
                    "confident_half_jev_top": {"accuracy": acc(hyb), "usd_per_1000": usd_hyb,
                                               "cost_share_vs_top": usd_hyb / usd_top},
                    "chain_jev_cheap_top": {"accuracy": cp["accuracy"], "usd_per_1000": cp["usd_per_1000"],
                                            "cost_share_vs_top": cp["usd_per_1000"] / usd_top,
                                            "source": "chain.json" if c_ in CHAIN_CHEAP else "computed here, chain.py rule"},
                    "minus_confident_half_pts": diff_ci(h, hyb, strat, idx, gidx),
                    "minus_chain_pts": diff_ci(h, cp["ok"], strat, idx, gidx),
                    "median_seconds_expected": lat}
        out1[key] = {"label": d["label"], "metric": d["metric"], "n_items": n, "cascades": casc}
        if d["cluster"]:
            out1[key]["n_recommendations"] = len(set(d["cluster"](i) for i in d["ids"]))
        # frontier
        pts = [{"id": "jev", "kind": "alone", "accuracy": acc(jev), "usd_per_1000": per(U["jev"])}]
        for s in MAIN + CHEAP_ALL:
            pts.append({"id": s, "kind": "alone", "accuracy": acc(OK[s]), "usd_per_1000": per(U[s])})
            pts.append({"id": f"jev+{s}", "kind": "confident_half_hybrid",
                        "accuracy": acc(np.where(use_j, jev, OK[s])),
                        "usd_per_1000": per(U["jev"]) + per(U[s].where(~use_j, 0))})
        for (c_, e), cp in chains.items():
            pts.append({"id": f"jev>{c_}>{e}", "kind": "chain", "accuracy": cp["chain"]["accuracy"],
                        "usd_per_1000": cp["chain"]["usd_per_1000"]})
            pts.append({"id": f"{c_}>{e}", "kind": "cheap_then_expensive", "accuracy": cp["cheap_then_expensive"]["accuracy"],
                        "usd_per_1000": cp["cheap_then_expensive"]["usd_per_1000"]})
        for nm, r in casc.items():
            pts.append({"id": nm, "kind": "agreement_cascade", "accuracy": r["accuracy"], "usd_per_1000": r["usd_per_1000"]})
        fr = frontier(pts)
        out2[key] = {"label": d["label"], "n_points": len(pts),
                     "n_points_by_kind": pd.Series([p["kind"] for p in pts]).value_counts().to_dict(),
                     "frontier": [{k: p[k] for k in ("id", "kind", "accuracy", "usd_per_1000")} for p in fr],
                     "points": pts}
    return out1, out2, checks


# ---------------------------------------------------------------- analysis 3
def cp_ci(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def load_structured():
    """Jev structured: answers, correctness, joint confidence, reported confidence and list cost per report."""
    it = pd.read_csv("data/arm3/items.csv", dtype=str)
    main_key = dict(zip(it["report_id"], it["level"]))
    cf = pd.read_parquet("results/arm3_confuser/calls.parquet")
    cf = cf[(cf["variant"] == "names") & (cf["repeat"] == 0)].copy()
    g = cf[cf["system"] == "gemini"].set_index("item_id")
    assert g["valid"].all()
    conf_key = g["answer"].to_dict()
    an = J("results/arm3_confuser/analysis.json")
    for s, x in cf.groupby("system"):
        x = x.set_index("item_id")
        k = int((x["valid"].fillna(False) & (x["answer"] == pd.Series(conf_key).reindex(x.index))).sum())
        assert k == an["systems"][s]["correct"], (s, k)
    nr = J("results/arm3_nonregional_sites/summary.json")
    sets = {}
    dc = pd.read_parquet("results/arm3_docuse/calls.parquet"); dc = dc[dc["repeat"] == 0]
    dp = pd.read_parquet("results/arm3_docuse/parts.parquet"); dp = dp[dp["repeat"] == 0]
    nc = pd.read_parquet("results/arm3_nonregional_sites/calls_jev.parquet"); nc = nc[nc["repeat"] == 0]
    npp = pd.read_parquet("results/arm3_nonregional_sites/parts.parquet"); npp = npp[npp["repeat"] == 0]
    pr = pd.read_csv("results/arm3_nonregional_sites/per_report.csv")
    main_ids = sorted(main_key); conf_ids = sorted(conf_key)
    nr_ids = sorted(nc["item_id"].unique())
    assert len(main_ids) == 1000 and len(conf_ids) == 100 and len(nr_ids) == 20
    spec = {"main": (dc, dp, main_ids, main_key),
            "misleading_feature": (dc, dp, conf_ids, conf_key),
            "nonregional_sites": (nc, npp, nr_ids, {i: nr["truth"]["level"] for i in nr_ids})}
    for name, (calls, parts, ids, key) in spec.items():
        for v in ("documented", "documented_notes"):
            c = calls[calls["variant"] == v].set_index("item_id").reindex(ids)
            p = parts[parts["variant"] == v].set_index("item_id").reindex(ids)
            assert c["answer"].notna().all() and p[QS[0] + "_top"].notna().all(), (name, v)
            joint = np.prod([p[q + "_top"].to_numpy(float) for q in QS], axis=0)
            assert np.allclose(joint, c["top_prob"].to_numpy(float), atol=1e-9), (name, v)
            ok = (c["valid"].fillna(False) & (c["answer"] == pd.Series(key).reindex(ids))).to_numpy()
            usd = (c["tokens_in"].fillna(0) * PRICE["jev"][0] + c["tokens_out"].fillna(0) * PRICE["jev"][1]).to_numpy() / 1e6
            sets[(name, v)] = {"ids": ids, "ok": ok, "joint": joint, "reported": c["jev_confidence"].to_numpy(float), "usd": usd,
                               "key": key}
    # checks against the recorded counts
    ds = J("results/summaries/docuse_summary.json")["staging"]
    for v in ("documented", "documented_notes"):
        assert sets[("main", v)]["ok"].sum() == 1000
        assert abs(sets[("misleading_feature", v)]["ok"].mean() - ds[v]["misleading_feature"]["exact_accuracy"]) < 1e-12
        assert sets[("nonregional_sites", v)]["ok"].sum() == nr["systems"][f"jev/{v}"]["correct"]
        r = pr[(pr["system"] == "jev") & (pr["variant"] == v)].set_index("report_id").reindex(nr_ids)
        assert np.allclose(r["joint_confidence"].to_numpy(float), sets[("nonregional_sites", v)]["joint"], atol=1e-9)
    # chatbots (names prompt) on each set
    bots = {}
    a3 = pd.read_parquet("results/arm3/calls.parquet")
    a3 = a3[(a3["variant"] == "names") & (a3["repeat"] == 0)]
    nb = pd.read_parquet("results/arm3_nonregional_sites/calls_chatbots.parquet")
    nb = nb[(nb["variant"] == "names") & (nb["repeat"] == 0)]
    for name, df, ids, key in (("main", a3, main_ids, main_key), ("misleading_feature", cf, conf_ids, conf_key),
                               ("nonregional_sites", nb, nr_ids, {i: nr["truth"]["level"] for i in nr_ids})):
        for s in MAIN:
            x = df[df["system"] == s].set_index("item_id").reindex(ids)
            assert x["system"].notna().all(), (name, s)
            ok = (x["valid"].fillna(False) & (x["answer"] == pd.Series(key).reindex(ids))).to_numpy()
            usd = (x["tokens_in"].fillna(0) * PRICE[s][0] + x["tokens_out"].fillna(0) * PRICE[s][1]).to_numpy() / 1e6
            bots[(name, s)] = {"ok": ok, "usd": usd}
    ha = J("results/summaries/hybrid_all.json")["sets"]["staging_names"]["partners"]
    for s in MAIN:
        assert abs(bots[("main", s)]["ok"].mean() - ha[s]["partner_alone"]) < 1e-12, s
        assert abs(1000 * bots[("main", s)]["usd"].mean() - ha[s]["usd_per_1000_partner"]) < 1e-9, s
    return sets, bots


def analysis_3(rng):
    sets, bots = load_structured()
    out = {"confidence_used": "joint = product of the four questions' top probabilities (arm3_docuse/parts.parquet and "
                              "arm3_nonregional_sites/parts.parquet); equals the calls' top_prob and per_report.csv's "
                              "joint_confidence. The calls' jev_confidence (product of Jev's own per-question confidence "
                              "fields) is a different number and is used only in the sensitivity block.",
           "rule": "cut fixed on the main 1,000 reports per version; send a report to the chatbot when its confidence is "
                   "below the cut by more than 1e-9; Jev's answer stands otherwise",
           "primary": {}, "sensitivity_reported_confidence": {}}
    idx_cache = {}
    for measure, block in (("joint", "primary"), ("reported", "sensitivity_reported_confidence")):
        for v in ("documented", "documented_notes"):
            m = sets[("main", v)][measure]
            cuts = {"median": float(np.quantile(m, 0.5)), "q25": float(np.quantile(m, 0.25))}
            for cname, cut in cuts.items():
                cfg = {"version": v, "cut_name": cname, "cut": cut, "sets": {}}
                for name in ("main", "misleading_feature", "nonregional_sites"):
                    S = sets[(name, v)]
                    n = len(S["ids"])
                    if (name, n) not in idx_cache:
                        idx_cache[(name, n)] = rng.integers(0, n, size=(B, n))
                    idx = idx_cache[(name, n)]
                    send = S[measure] < cut - TOL
                    ok = S["ok"]
                    r = {"n": n, "jev_correct": int(ok.sum()), "jev_wrong": int((~ok).sum()),
                         "jev_accuracy": float(ok.mean()), "jev_accuracy_ci": cp_ci(int(ok.sum()), n),
                         "sent": int(send.sum()), "share_sent": float(send.mean()),
                         "wrong_sent": int((send & ~ok).sum()), "wrong_kept": int((~send & ~ok).sum()),
                         "right_sent": int((send & ok).sum()), "right_kept": int((~send & ok).sum()),
                         "confidence_of_wrong": {"min": float(S[measure][~ok].min()), "max": float(S[measure][~ok].max())} if (~ok).any() else None,
                         "jev_usd_per_1000": float(1000 * S["usd"].mean()), "chatbots": {}}
                    for s in MAIN:
                        bo = bots[(name, s)]
                        h = np.where(send, bo["ok"], ok)
                        usd = float(1000 * (S["usd"].sum() + bo["usd"][send].sum()) / n)
                        usd_bot = float(1000 * bo["usd"].mean())
                        dj = h[idx].mean(1) - ok[idx].mean(1)
                        db = h[idx].mean(1) - bo["ok"][idx].mean(1)
                        r["chatbots"][s] = {
                            "name": NAME[s], "chatbot_alone": float(bo["ok"].mean()), "chatbot_correct": int(bo["ok"].sum()),
                            "hybrid_correct": int(h.sum()), "hybrid_accuracy": float(h.mean()),
                            "hybrid_accuracy_ci": cp_ci(int(h.sum()), n),
                            "chatbot_wrong_on_sent": int((send & ~bo["ok"]).sum()),
                            "hybrid_minus_jev_pts": {"est": float(100 * (h.mean() - ok.mean())),
                                                     "ci": [float(100 * np.quantile(dj, 0.025)), float(100 * np.quantile(dj, 0.975))]},
                            "hybrid_minus_chatbot_pts": {"est": float(100 * (h.mean() - bo["ok"].mean())),
                                                         "ci": [float(100 * np.quantile(db, 0.025)), float(100 * np.quantile(db, 0.975))]},
                            "usd_per_1000_hybrid": usd, "usd_per_1000_chatbot": usd_bot, "cost_share_vs_chatbot": usd / usd_bot}
                    cfg["sets"][name] = r
                out[block][f"{v}|{cname}"] = cfg
    return out


def main():
    _check_prices()
    rng = np.random.default_rng(SEED)
    a1, a2, checks = analyses_1_2(rng)
    a3 = analysis_3(rng)
    res = {"note": __doc__.split("\n\n  python")[0], "B": B, "seed": SEED, "cost_basis": "list price, standard tier",
           "checks": checks, "agreement_cascade": a1, "frontier": a2, "structured_safety_net": a3}
    Path("results/summaries/hybrid_explore.json").write_text(json.dumps(res, indent=1))
    # ------------------------------------------------ print
    print("checks:", checks)
    for k, v in a1.items():
        print(f"\n{v['label']} (n={v['n_items']})")
        for nm, r in v["cascades"].items():
            m = r["minus_top_alone_pts"]; ci = m.get("ci_by_recommendation", m["ci"])
            print(f"  {nm:32s} {100*r['accuracy']:5.1f} vs top {100*r['top_alone']:5.1f} d {m['est']:+5.1f} ({ci[0]:+.1f},{ci[1]:+.1f}) "
                  f"esc {100*r['share_escalated']:4.1f}% agreed-wrong {r['agreed']['wrong_kept']:3d} ${r['usd_per_1000']:.3f} "
                  f"({100*r['cost_share_vs_top']:.0f}%) | half {100*r['confident_half_jev_top']['accuracy']:5.1f} "
                  f"${r['confident_half_jev_top']['usd_per_1000']:.3f} | chain {100*r['chain_jev_cheap_top']['accuracy']:5.1f} "
                  f"${r['chain_jev_cheap_top']['usd_per_1000']:.3f} | -half {r['minus_confident_half_pts']['est']:+.1f} "
                  f"{[round(x,1) for x in r['minus_confident_half_pts']['ci']]}")
        print("  frontier:", [(p["id"], round(100 * p["accuracy"], 1), round(p["usd_per_1000"], 3)) for p in a2[k]["frontier"]])
    for blk in ("primary", "sensitivity_reported_confidence"):
        print("\n==", blk)
        for nm, cfg in a3[blk].items():
            print(f" {nm} cut={cfg['cut']:.4f}")
            for sn, r in cfg["sets"].items():
                hs = " ".join(f"{s}:{x['hybrid_correct']}" for s, x in r["chatbots"].items())
                us = " ".join("%.2f" % x["usd_per_1000_hybrid"] for x in r["chatbots"].values())
                print(f"   {sn:20s} n={r['n']:4d} jev {r['jev_correct']:4d} sent {r['sent']:4d} wrong sent {r['wrong_sent']}/{r['jev_wrong']} "
                      f"right sent {r['right_sent']}/{r['jev_correct']} | hybrid correct {hs} | $ {us}")


if __name__ == "__main__":
    main()

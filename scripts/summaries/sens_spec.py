"""Sensitivity and specificity for Jev, the chatbots and the Jev-first hybrids (from the stored answers).

Two kinds, both descriptive:
A. Confidence as a filter on a system's own errors (every choice and ordered-scale task version): each system keeps its
   most confident half (top probability, ties by item identifier, as the tested hybrid rule does for Jev); sensitivity =
   share of its correct answers kept, specificity = share of its wrong answers held back; with the confidence AUROC
   (top probability against a correct answer). Unusable answers count as wrong and have no confidence (ranked last).
B. Clinical decisions made binary (every system and every Jev-first hybrid, 50% rule, with each main chatbot):
   next step: ordering the test or treatment; positive = indicated scenario;
   colon cancer: stage III or IV (systemic treatment) and stage IV, against lower stages;
   kidney injury: any AKI (stage 1-3) and severe AKI (stage 2-3), against lower stages.
   An unusable answer counts as a miss among positives and as a false alarm among negatives.
C. Discrimination (AUROC) of each binary clinical endpoint, for every system and hybrid, scoring each item by the
   probability the system gave to the positive levels (lists summing to 0.9-1.1 rescaled; unusable answers and other
   lists left out and counted).
D. Risk estimation: sensitivity and specificity for death within ten years on the 12,697 baseline records every system
   answered, at a predicted risk of 20% and with each predictor flagging as many people as died (its highest-risk
   2,002); for Jev, the chatbots, the references and the risk-band hybrid (Jev's death probability on the half with its
   highest band probability, ties by record, the chatbot's elsewhere). Log-loss recomputed here must equal
   results/wide/analysis.json and the hybrid's results/wide/hybrid.json (asserted).
Keys: next step from the item identifier; board examination from data/arm2_ext/items.csv (built by
python scripts/build_arm2_ext_items.py --download); colon cancer from data/arm3/items.csv; kidney injury from
the answers of GPT-5.6 Sol and Gemini 3.8 Flash in the version with definitions, both scored exactly 1.000 there
and identical on every stay (asserted), so no eICU data file is read (only the stored answers in
results/eicu_aki/calls.parquet); death within ten years from data/cohort.parquet (scripts/build_cohort.py), with the
references' out-of-fold predictions from data/reference_oof.parquet (scripts/crossfit_reference.py) and Jev's risk-band
answers from results/wide/calls_jev_bands.parquet.
Every system's accuracy and every hybrid's accuracy recomputed here must equal the analysis files (asserted).

  python scripts/summaries/sens_spec.py -> results/summaries/sens_spec.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
ALL = ["jev"] + MAIN + ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
J = lambda p: json.loads(Path(p).read_text())


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def prop(k, n):
    return {"k": int(k), "n": int(n), "p": (k / n) if n else None, "ci": wilson(k, n)}


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    r = pd.Series(score).rank().to_numpy()
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load(calls, variant, truth):
    c = calls[(calls["variant"] == variant) & (calls["repeat"] == 0)].copy()
    c["truth"] = c["item_id"].map(truth)
    assert c["truth"].notna().all(), variant
    c["valid"] = c["valid"].fillna(False).astype(bool)
    c["ans"] = c["answer"].where(c["valid"], None)
    c["ok"] = c["valid"] & (c["ans"] == c["truth"])
    return c


def filter_stats(g):
    g = g.copy()
    g["conf"] = g["top_prob"].where(g["valid"], -1.0).fillna(-1.0)
    g = g.sort_values(["conf", "item_id"], ascending=[False, True])
    half = np.zeros(len(g), bool)
    half[: len(g) // 2] = True
    ok = g["ok"].to_numpy()
    n_ok, n_err = int(ok.sum()), int((~ok).sum())
    return {"n": int(len(g)), "n_kept": int(half.sum()), "n_correct": n_ok, "n_errors": n_err,
            "sensitivity": prop((ok & half).sum(), n_ok), "specificity": prop((~ok & ~half).sum(), n_err),
            "errors_let_through": int((~ok & half).sum()), "accuracy_kept": float(ok[half].mean()),
            "confidence_auroc": (lambda h: auroc(h["top_prob"], h["ok"]) if len(h) else None)(g[g["valid"] & g["top_prob"].notna()]),
            "n_valid_without_probability": int((g["valid"] & g["top_prob"].isna()).sum())}


def binary(ans, truth, pos):
    """pos: set of positive labels. ans None = unusable."""
    t = truth.isin(pos)
    called = ans.isin(pos) & ans.notna()
    neg_ok = (~ans.isin(pos)) & ans.notna()
    return {"sensitivity": prop((t & called).sum(), t.sum()), "specificity": prop((~t & neg_ok).sum(), (~t).sum())}


def p_pos(probs, status, pos):
    if status not in ("ok", "renormalised") or probs is None or (isinstance(probs, float) and np.isnan(probs)):
        return np.nan
    d = json.loads(probs) if isinstance(probs, str) else dict(probs)
    tot = sum(float(v) for v in d.values())
    if not (0.9 <= tot <= 1.1) or tot == 0:
        return np.nan
    return sum(float(v) for k, v in d.items() if k in pos) / tot


def auc_block(score, truth, pos):
    ok = score.notna()
    y = truth.isin(pos)
    return {"auroc": auroc(score[ok], y[ok]), "n_scored": int(ok.sum()), "n_left_out": int((~ok).sum())}


def hybrid_answers(c, jev_items, s):
    j = c[c["system"] == "jev"].set_index("item_id")
    b = c[c["system"] == s].set_index("item_id")
    idx = j.index
    use_j = idx.isin(set(jev_items))
    ans = pd.Series(np.where(use_j, j["ans"].reindex(idx), b["ans"].reindex(idx)), index=idx)
    truth = j["truth"]
    return ans, truth


def task(calls, variant, truth, an, metric, endpoints):
    c = load(calls, variant, truth)
    out = {"filter": {}, "systems": {}, "hybrids": {}}
    for s, g in c.groupby("system"):
        if s not in ALL:
            continue
        if metric == "balanced_accuracy":
            acc = g.groupby("truth")["ok"].mean().mean()
        else:
            acc = g["ok"].mean()
        assert abs(acc - an["systems"][s][metric]) < 1e-9, (variant, s, acc, an["systems"][s][metric])
        out["filter"][s] = filter_stats(g)
        gi = g.set_index("item_id")
        out["systems"][s] = {k: binary(gi["ans"], gi["truth"], pos) for k, pos in endpoints.items()}
        for k, pos in endpoints.items():
            sc = pd.Series([p_pos(a, b_, pos) for a, b_ in zip(gi["probs"], gi["prob_status"])], index=gi.index)
            out["systems"][s][k].update(auc_block(sc, gi["truth"], pos))
    jev_items = an["hybrid_share"]["jev_items"]
    for s in MAIN:
        ans, tr = hybrid_answers(c, jev_items, s)
        ok = ans.eq(tr) & ans.notna()
        acc = ok.groupby(tr).mean().mean() if metric == "balanced_accuracy" else ok.mean()
        assert abs(acc - an["hybrid_share"]["comparisons"][s][metric]) < 1e-9, (variant, s, acc)
        out["hybrids"][s] = {k: binary(ans, tr, pos) for k, pos in endpoints.items()}
        j = c[c["system"] == "jev"].set_index("item_id")
        b_ = c[c["system"] == s].set_index("item_id").reindex(j.index)
        use_j = j.index.isin(set(jev_items))
        for k, pos in endpoints.items():
            sj = pd.Series([p_pos(a, st, pos) for a, st in zip(j["probs"], j["prob_status"])], index=j.index)
            sb = pd.Series([p_pos(a, st, pos) for a, st in zip(b_["probs"], b_["prob_status"])], index=j.index)
            out["hybrids"][s][k].update(auc_block(sj.where(use_j, sb), j["truth"], pos))
    return out


def main():
    res = {"note": "descriptive; see module docstring for definitions", "tasks": {}}
    # next step
    a2 = J("results/arm2/analysis.json")
    c2 = pd.read_parquet("results/arm2/calls.parquet")
    items2 = c2["item_id"].unique()
    truth2 = {i: ("B" if "_lv_" in i else "A") for i in items2}
    res["tasks"]["next_step"] = task(c2, "main", truth2, a2, "balanced_accuracy", {"order_when_indicated": {"A"}})
    # board examination (filter only; no binary clinical endpoint)
    ax = J("results/arm2_ext/analysis.json")
    it = pd.read_csv("data/arm2_ext/items.csv", dtype=str)
    cx = load(pd.read_parquet("results/arm2_ext/calls.parquet"), "medhelm", dict(zip(it["item_id"], it["truth"])))
    res["tasks"]["board_exam"] = {"filter": {}}
    for s, g in cx.groupby("system"):
        assert abs(g["ok"].mean() - ax["systems"][s]["accuracy"]) < 1e-9, s
        res["tasks"]["board_exam"]["filter"][s] = filter_stats(g)
    # colon cancer
    it3 = pd.read_csv("data/arm3/items.csv", dtype=str)
    truth3 = dict(zip(it3["report_id"], it3["level"]))
    c3 = pd.read_parquet("results/arm3/calls.parquet")
    ep3 = {"stage_iii_or_iv": {"IIIA", "IIIB", "IIIC", "IVA-IVB", "IVC"}, "stage_iv": {"IVA-IVB", "IVC"}}
    for v, fn in (("names", "analysis.json"), ("definitions", "analysis_definitions.json")):
        res["tasks"][f"staging_{v}"] = task(c3, v, truth3, J(f"results/arm3/{fn}"), "exact_accuracy", ep3)
    # kidney injury: key from two perfect, identical answer sets
    ck = pd.read_parquet("results/eicu_aki/calls.parquet")
    dk = J("results/eicu_aki/analysis_definitions.json")
    assert dk["systems"]["gpt"]["exact_accuracy"] == 1.0 and dk["systems"]["gemini"]["exact_accuracy"] == 1.0
    d0 = ck[(ck["variant"] == "definitions") & (ck["repeat"] == 0)]
    g1 = d0[d0["system"] == "gpt"].set_index("item_id")["answer"]
    g2 = d0[d0["system"] == "gemini"].set_index("item_id")["answer"].reindex(g1.index)
    assert (g1 == g2).all() and g1.value_counts().to_dict() == dk["strata"], "AKI key"
    truthk = g1.to_dict()
    epk = {"any_aki": {"stage 1", "stage 2", "stage 3"}, "severe_aki": {"stage 2", "stage 3"}}
    for v, fn in (("names", "analysis.json"), ("definitions", "analysis_definitions.json")):
        res["tasks"][f"aki_{v}"] = task(ck, v, truthk, J(f"results/eicu_aki/{fn}"), "exact_accuracy", epk)
    # risk estimation (arm 1)
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
    wa = J("results/wide/analysis.json")["prediction"]
    ll = lambda q: float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
    for s in ["jev"] + MAIN + ["reference_spline_logit", "reference_age_sex"]:
        assert abs(ll(P[s]) - wa["systems"][s]["log_loss"]) < 1e-6, (s, ll(P[s]))
    bd = pd.read_parquet("results/wide/calls_jev_bands.parquet").set_index("profile").reindex(P.index)
    top = bd["band_probabilities"].map(lambda d: max(json.loads(d).values()) if isinstance(d, str) else -1.0)
    order = sorted(P.index, key=lambda i: (-top[i], i))
    hy = J("results/wide/hybrid.json")
    jev_half = set(order[: hy["n_jev"]])
    half = P.index.isin(jev_half)
    k_dead = int(y.sum())
    def rates(q):
        r = {}
        for name, flag in (("threshold_20", q >= 0.20), ("flag_as_many_as_died", q.rank(ascending=False, method="first") <= k_dead)):
            r[name] = {"sensitivity": prop((flag & (y == 1)).sum(), (y == 1).sum()), "specificity": prop((~flag & (y == 0)).sum(), (y == 0).sum()),
                       "n_flagged": int(flag.sum())}
        r["auroc"] = auroc(q, y == 1)
        return r
    t1 = {"n": int(len(y)), "deaths": k_dead, "systems": {s: rates(P[s]) for s in ["jev"] + MAIN + ["reference_spline_logit", "reference_age_sex"]},
          "hybrids": {}}
    curve = {c_["share"]: c_ for c_ in hy["share_curve"]}
    for s in MAIN:
        q = P["jev"].where(half, P[s])
        assert abs(ll(q) - curve[0.5][s]["hybrid"]) < 1e-6, (s, ll(q), curve[0.5][s]["hybrid"])
        t1["hybrids"][s] = rates(q)
    res["tasks"]["risk"] = t1

    # summary ranges
    T = res["tasks"]
    rng = lambda xs: {"min": min(xs), "max": max(xs)}
    bx = T["board_exam"]["filter"]
    sd = T["staging_definitions"]
    miss = lambda d: d["stage_iii_or_iv"]["sensitivity"]["n"] - d["stage_iii_or_iv"]["sensitivity"]["k"]
    res["summary"] = {
        "board_exam_through_share": {"jev": bx["jev"]["errors_let_through"] / bx["jev"]["n_errors"],
                                     "chatbots": rng([bx[s]["errors_let_through"] / bx[s]["n_errors"] for s in MAIN])},
        "board_exam_auroc": {"jev": bx["jev"]["confidence_auroc"], "chatbots": rng([bx[s]["confidence_auroc"] for s in MAIN])},
        "staging_definitions_hybrid_missed_iii_iv": rng([miss(sd["hybrids"][s]) for s in MAIN]),
        "staging_definitions_chatbot_missed_iii_iv": {s: miss(sd["systems"][s]) for s in MAIN},
        "staging_definitions_n_iii_iv": sd["systems"]["jev"]["stage_iii_or_iv"]["sensitivity"]["n"],
        "staging_definitions_jev_missed_iii_iv": miss(sd["systems"]["jev"]),
        "aki_names_hybrid_minus_chatbot_spec_any_pts": rng([100 * (T["aki_names"]["hybrids"][s]["any_aki"]["specificity"]["p"]
                                                              - T["aki_names"]["systems"][s]["any_aki"]["specificity"]["p"]) for s in MAIN]),
        "next_step": {"jev_spec": T["next_step"]["systems"]["jev"]["order_when_indicated"]["specificity"]["p"],
                      "jev_sens": T["next_step"]["systems"]["jev"]["order_when_indicated"]["sensitivity"]["p"],
                      "chatbots_spec": rng([T["next_step"]["systems"][s]["order_when_indicated"]["specificity"]["p"] for s in MAIN])},
        "staging_names_jev_sens_iii_iv": T["staging_names"]["systems"]["jev"]["stage_iii_or_iv"]["sensitivity"]["p"],
        "staging_names_chatbots_sens_iii_iv": rng([T["staging_names"]["systems"][s]["stage_iii_or_iv"]["sensitivity"]["p"] for s in MAIN]),
        "aki_names_jev_spec_any": T["aki_names"]["systems"]["jev"]["any_aki"]["specificity"]["p"],
        "aki_names_chatbots_spec_any": rng([T["aki_names"]["systems"][s]["any_aki"]["specificity"]["p"] for s in MAIN])}
    Path("results/summaries/sens_spec.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=0))
    # print a compact view
    for t, v in res["tasks"].items():
        if "filter" not in v:
            continue
        f = v["filter"]["jev"]
        print(f"{t:20s} Jev filter sens {f['sensitivity']['p']:.3f} spec {f['specificity']['p'] if f['specificity']['p'] is not None else float('nan'):.3f} "
              f"AUROC {f['confidence_auroc']}")
        for s in MAIN:
            x = v["filter"][s]
            print(f"   {s:8s} errors {x['n_errors']:4d} spec {x['specificity']['p'] if x['specificity']['p'] is not None else float('nan'):.3f} "
                  f"AUROC {x['confidence_auroc']}")
        if "systems" in v:
            for k in v["systems"]["jev"]:
                sj = v["systems"]["jev"][k]
                print(f"   {k:22s} Jev sens {sj['sensitivity']['p']:.3f} spec {sj['specificity']['p']:.3f} | chatbots sens "
                      f"{min(v['systems'][s][k]['sensitivity']['p'] for s in MAIN):.3f}-{max(v['systems'][s][k]['sensitivity']['p'] for s in MAIN):.3f} "
                      f"spec {min(v['systems'][s][k]['specificity']['p'] for s in MAIN):.3f}-{max(v['systems'][s][k]['specificity']['p'] for s in MAIN):.3f} | hybrids sens "
                      f"{min(v['hybrids'][s][k]['sensitivity']['p'] for s in MAIN):.3f}-{max(v['hybrids'][s][k]['sensitivity']['p'] for s in MAIN):.3f} "
                      f"spec {min(v['hybrids'][s][k]['specificity']['p'] for s in MAIN):.3f}-{max(v['hybrids'][s][k]['specificity']['p'] for s in MAIN):.3f}")


if __name__ == "__main__":
    main()

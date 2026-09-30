"""Further analyses of the Jev-first workflows, from the stored answers (no model calls, no network); every
configuration computed is written out.

A. Cheap-model control, GPT-5.6 Luna (cheap) and GPT-5.6 Sol (top), on the six choice and ordered-scale sets of
   hybrid_all.py: Luna alone; Sol alone; Jev + Luna (Jev keeps its confident half, Luna the rest; hybrid_all.py);
   Luna -> Sol (Luna keeps its confident half, Sol the rest; chain.py); Jev -> Luna -> Sol (chain.py); agreement cascade
   (Jev and Luna answer every item, Sol answers where they differ or either answer is unusable; hybrid_explore.py);
   Luna-confidence escalation matched to the cascade (Luna answers every item and Sol the same number of items the
   cascade escalated, those with Luna's lowest top probability, unusable first, ties by item identifier). Accuracy as
   hybrid_all.py, list cost per 1,000. Differences: paired 95% percentile bootstrap, 2,000 resamples over items, seed
   20260927 (patient messages also by recommendation). Diagnostics per escalation rule: Luna's errors escalated,
   corrected by Sol, Luna's right answers made wrong by Sol; Luna's error rate where Jev agrees and where it does not.
B. Probability aggregation. For each grouped endpoint two rules: collapse (the chosen level's group; sens_spec.py) and
   sum (probability summed over the group's levels, lists rescaled as sens_spec.py; positive at >= 0.5; for main stage
   the group with the largest sum, ties to the lower group). Where probabilities are unusable the sum rule takes the
   collapse rule's call (counted). Colon cancer stage (names, definitions): stage IV, stage III-IV, main stage
   (0, I, II, III, IV). Kidney injury (names, definitions): stage 2-3. Every system.
C. Complete structured workflow, colon cancer. Jev with structured input answers every report; its answer stands when
   all four top probabilities are 1.00 (joint >= 1 - 1e-9, hybrid_explore.py's cut), otherwise the chatbot's answer
   with stage names. Principal version documented; documented_notes reported beside it. Ordinary sets: main 1,000 and
   the 200 held-out reports (key "heldout"); difficult sets: 100 with a misleading feature and 20 at non-regional
   sites. Per set and chatbot: share kept by Jev, exact accuracy, errors, consequential errors (final answer in a
   different group of 0-II, III, IV from the key; unusable counts as one), list cost per 1,000 and as a share of the
   chatbot alone. Full confidence as a test for Jev's errors per version (no pooling of versions). Counts asserted
   against hybrid_explore.json.
D. Threshold transfer, board-examination questions, by dataset (MedQA, Medbullets). Jev's answer stands at top
   probability 1.00, otherwise each main chatbot's (the "fixed_*" summary keys). A threshold chosen on one dataset
   (the lowest observed top probability t at which Jev's answers with top >= t are at least as accurate as GPT-5.6 Sol
   there) is applied to the other (the "tuned_*" summary keys).
E. Probability-vector overhead. Tokens of each chatbot's probability vector are estimated as the characters of the
   parsed vector's JSON divided by 3 (an upper estimate), removed from its output tokens except where the workflow
   uses that model's probabilities (Luna in Luna -> Sol, the chain and Luna-confidence escalation), and cost shares
   recomputed.

  python scripts/summaries/workflow_checks.py --docuse . --heldout .
      -> results/summaries/workflow_checks.json
--docuse: the folder holding results/arm3_docuse and results/arm3_nonregional_sites; --heldout: the folder holding
data/arm3_heldout/items.csv and results/arm3_heldout. Also reads the stored calls and keys listed in hybrid_all.py,
results/arm3_confuser/calls.parquet and the outputs of hybrid_all.py, chain.py, sens_spec.py and hybrid_explore.py
in results/summaries/ (run those first).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_all import SETS, score, B, SEED, board_truth, staging_truth, aki_truth  # noqa: E402
from cost_frontier import PRICE, NAME, MAIN, CHEAP  # noqa: E402
from chain import half_rule  # noqa: E402
from hybrid_explore import load_set, boot_idx, diff_ci, cp_ci  # noqa: E402
from sens_spec import prop, p_pos  # noqa: E402

J = lambda p: json.loads(Path(p).read_text())
TOL = 1e-9
LUNA, SOL = "gpt_free", "gpt"
ALL = ["jev"] + MAIN + CHEAP + ["claude_free"]
MAJOR = {"0": "0", "I": "I", "IIA": "II", "IIB": "II", "IIC": "II", "IIIA": "III", "IIIB": "III", "IIIC": "III",
         "IVA-IVB": "IV", "IVC": "IV"}
MAJ_ORDER = ["0", "I", "II", "III", "IV"]
GRP3 = {k: ("0-II" if v in ("0", "I", "II") else v) for k, v in MAJOR.items()}
EP3 = {"stage_iv": {"IVA-IVB", "IVC"}, "stage_iii_or_iv": {"IIIA", "IIIB", "IIIC", "IVA-IVB", "IVC"}}
EPK = {"severe_aki": {"stage 2", "stage 3"}}
per = lambda u, n: float(1000 * np.sum(u) / n)


def pts(x):
    return float(100 * x)


# ------------------------------------------------------------------------------------------------ A
def part_a(rng):
    ha = J("results/summaries/hybrid_all.json")["sets"]
    he = J("results/summaries/hybrid_explore.json")["agreement_cascade"]
    ch = J("results/summaries/chain.json")["sets"]
    out, masks = {}, {}
    for spec in SETS:
        d = load_set(*spec)
        key, n, OK, U, V, ANS, TOP, strat, ids = (d[k] for k in ("key", "n", "OK", "U", "V", "ANS", "TOP", "strat", "ids"))
        idx, gidx = boot_idx(d, rng)
        ok = {s: OK[s].to_numpy() for s in ("jev", LUNA, SOL)}
        u = {s: U[s].to_numpy() for s in ("jev", LUNA, SOL)}
        use_j = d["use_j"]
        # Jev + Luna
        hyb = np.where(use_j, ok["jev"], ok[LUNA])
        # Luna -> Sol, chain
        topL = TOP[LUNA].to_numpy(float)
        keepL = half_rule(topL, np.ones(n, bool), ids)
        l2s = np.where(keepL, ok[LUNA], ok[SOL])
        useC = half_rule(topL, ~use_j, ids)
        chain = np.where(use_j, ok["jev"], np.where(useC, ok[LUNA], ok[SOL]))
        # cascade
        agree = (V["jev"] & V[LUNA] & (ANS["jev"] == ANS[LUNA])).to_numpy()
        esc = ~agree
        casc = np.where(agree, ok[LUNA], ok[SOL])
        k = int(esc.sum())
        # Luna-confidence escalation, same number of items
        order = sorted(range(n), key=lambda i: (topL[i] if np.isfinite(topL[i]) else -1.0, ids[i]))
        escL = np.zeros(n, bool); escL[order[:k]] = True
        lconf = np.where(escL, ok[SOL], ok[LUNA])
        acc = {
            "luna_alone": score(ok[LUNA], strat), "sol_alone": score(ok[SOL], strat), "jev_alone": score(ok["jev"], strat),
            "jev_plus_luna": score(hyb, strat), "luna_then_sol": score(l2s, strat), "chain_jev_luna_sol": score(chain, strat),
            "agreement_cascade": score(casc, strat), "luna_confidence_matched": score(lconf, strat)}
        usd = {
            "luna_alone": per(u[LUNA], n), "sol_alone": per(u[SOL], n), "jev_alone": per(u["jev"], n),
            "jev_plus_luna": per(u["jev"], n) + per(u[LUNA][~use_j], n),
            "luna_then_sol": per(u[LUNA], n) + per(u[SOL][~keepL], n),
            "chain_jev_luna_sol": per(u["jev"], n) + per(u[LUNA][~use_j], n) + per(u[SOL][~use_j & ~useC], n),
            "agreement_cascade": per(u["jev"], n) + per(u[LUNA], n) + per(u[SOL][esc], n),
            "luna_confidence_matched": per(u[LUNA], n) + per(u[SOL][escL], n)}
        # checks against the stored analyses
        hp = ha[key]["partners"][LUNA]
        assert abs(acc["jev_plus_luna"] - hp["hybrid"]) < 1e-12 and abs(usd["jev_plus_luna"] - hp["usd_per_1000_hybrid"]) < 1e-9, key
        cc = ch[key]["chains"]["jev>gpt_free>gpt"]
        assert abs(acc["chain_jev_luna_sol"] - cc["accuracy"]["chain"]) < 1e-12, key
        assert abs(acc["luna_then_sol"] - cc["accuracy"]["cheap_then_expensive"]) < 1e-12, key
        assert abs(usd["luna_then_sol"] - cc["usd_per_1000"]["cheap_then_expensive"]) < 1e-9, key
        ec = he[key]["cascades"]["agree(jev,gpt_free)>gpt"]
        assert ec["n_escalated"] == k and abs(ec["accuracy"] - acc["agreement_cascade"]) < 1e-12, key
        assert abs(ec["usd_per_1000"] - usd["agreement_cascade"]) < 1e-9, key
        W = {"jev_plus_luna": hyb, "luna_then_sol": l2s, "chain_jev_luna_sol": chain, "agreement_cascade": casc,
             "luna_confidence_matched": lconf, "sol_alone": ok[SOL]}
        diffs = {
            "jev_plus_luna_minus_luna": diff_ci(hyb, ok[LUNA], strat, idx, gidx),
            "chain_minus_luna": diff_ci(chain, ok[LUNA], strat, idx, gidx),
            "chain_minus_luna_then_sol": diff_ci(chain, l2s, strat, idx, gidx),
            "cascade_minus_luna_confidence": diff_ci(casc, lconf, strat, idx, gidx),
            "cascade_minus_sol": diff_ci(casc, ok[SOL], strat, idx, gidx),
            "luna_confidence_minus_sol": diff_ci(lconf, ok[SOL], strat, idx, gidx),
            "luna_minus_sol": diff_ci(ok[LUNA], ok[SOL], strat, idx, gidx)}
        lw = ~ok[LUNA]
        diag = {}
        for name, E in (("agreement_cascade", esc), ("luna_confidence_matched", escL)):
            diag[name] = {"escalated": int(E.sum()), "luna_errors": int(lw.sum()),
                          "luna_errors_escalated": int((E & lw).sum()),
                          "corrected_by_sol": int((E & lw & ok[SOL]).sum()),
                          "introduced_by_sol": int((E & ~lw & ~ok[SOL]).sum()),
                          "luna_errors_kept": int((~E & lw).sum())}
        diag["luna_error_rate_when_jev_agrees"] = prop(int((agree & lw).sum()), int(agree.sum()))
        diag["luna_error_rate_when_jev_differs"] = prop(int((esc & lw).sum()), int(esc.sum()))
        out[key] = {"label": d["label"], "metric": d["metric"], "n": n, "accuracy": acc, "usd_per_1000": usd,
                    "cost_share_vs_luna": {k2: v / usd["luna_alone"] for k2, v in usd.items()},
                    "cost_share_vs_sol": {k2: v / usd["sol_alone"] for k2, v in usd.items()},
                    "differences_pts": diffs, "escalation": diag, "share_escalated": k / n,
                    "dominated_by_luna_alone": {w: bool(usd[w] >= usd["luna_alone"] and acc[w] <= acc["luna_alone"])
                                                for w in usd if w != "luna_alone"}}
        masks[key] = {"ids": ids, "use_j": use_j, "keepL": keepL, "chain_sol": ~use_j & ~useC, "esc": esc, "escL": escL}
    return out, masks


# ------------------------------------------------------------------------------------------------ B
def sums(probs, status, levels):
    if status not in ("ok", "renormalised") or probs is None or (isinstance(probs, float) and np.isnan(probs)):
        return None
    dct = json.loads(probs) if isinstance(probs, str) else dict(probs)
    tot = sum(float(v) for v in dct.values())
    if not (0.9 <= tot <= 1.1) or tot == 0:
        return None
    return {k: float(dct.get(k, 0.0)) / tot for k in levels}


def part_b():
    ss = J("results/summaries/sens_spec.json")["tasks"]
    out = {}
    jobs = [("staging_names", "results/arm3/calls.parquet", "names", staging_truth, EP3, True),
            ("staging_definitions", "results/arm3/calls.parquet", "definitions", staging_truth, EP3, True),
            ("aki_names", "results/eicu_aki/calls.parquet", "names", aki_truth, EPK, False),
            ("aki_definitions", "results/eicu_aki/calls.parquet", "definitions", aki_truth, EPK, False)]
    for key, calls, variant, tfn, eps, major in jobs:
        c = pd.read_parquet(calls)
        c = c[(c["variant"] == variant) & (c["repeat"] == 0)].copy()
        ids = sorted(c["item_id"].unique())
        truth = pd.Series(tfn(ids)).reindex(ids)
        levels = sorted(truth.unique()) if not major else list(MAJOR)
        res = {}
        for s in ALL:
            g = c[c["system"] == s].set_index("item_id").reindex(ids)
            if g["system"].isna().all():
                continue
            valid = g["valid"].fillna(False).astype(bool)
            ans = g["answer"].where(valid)
            P = [sums(p, st, levels) for p, st in zip(g["probs"], g["prob_status"])]
            fb = np.array([p is None for p in P])
            r = {"n": len(ids), "probabilities_unusable": int(fb.sum()), "unusable_answers": int((~valid).sum())}
            for ep, pos in eps.items():
                t = truth.isin(pos).to_numpy()
                col = (ans.isin(pos) & ans.notna()).to_numpy()
                colneg = ((~ans.isin(pos)) & ans.notna()).to_numpy()
                ps = np.array([np.nan if p is None else sum(p[k] for k in pos) for p in P])
                sm = np.where(fb, col, ps >= 0.5)
                smneg = np.where(fb, colneg, ps < 0.5)
                ref = ss[key]["systems"][s][ep]
                assert int((t & col).sum()) == ref["sensitivity"]["k"] and int((~t & colneg).sum()) == ref["specificity"]["k"], (key, s, ep)
                r[ep] = {"collapse": {"sensitivity": prop((t & col).sum(), t.sum()), "specificity": prop((~t & colneg).sum(), (~t).sum())},
                         "sum": {"sensitivity": prop((t & sm).sum(), t.sum()), "specificity": prop((~t & smneg).sum(), (~t).sum())},
                         "calls_changed": int((col != sm).sum())}
            if major:
                tm = truth.map(MAJOR).to_numpy()
                cm = ans.map(MAJOR).to_numpy()
                smj = []
                for p, a in zip(P, cm):
                    if p is None:
                        smj.append(a)
                    else:
                        gs = [sum(v for k, v in p.items() if MAJOR[k] == m) for m in MAJ_ORDER]
                        smj.append(MAJ_ORDER[int(np.argmax(gs))])
                smj = np.array(smj, dtype=object)
                r["main_stage"] = {"collapse": prop(int(sum(a == b for a, b in zip(cm, tm))), len(ids)),
                                   "sum": prop(int(sum(a == b for a, b in zip(smj, tm))), len(ids))}
            res[s] = r
        out[key] = res
    return out


# ------------------------------------------------------------------------------------------------ C
def part_c(W, H):
    it = pd.read_csv("data/arm3/items.csv", dtype=str)
    main_key = dict(zip(it["report_id"], it["level"]))
    cf = pd.read_parquet("results/arm3_confuser/calls.parquet")
    cf = cf[(cf["variant"] == "names") & (cf["repeat"] == 0)]
    conf_key = cf[cf["system"] == "gemini"].set_index("item_id")["answer"].to_dict()
    nr = J(f"{W}/results/arm3_nonregional_sites/summary.json")
    hi = pd.read_csv(f"{H}/data/arm3_heldout/items.csv", dtype=str)
    held_key = dict(zip(hi["report_id"], hi["level"]))
    dc = pd.read_parquet(f"{W}/results/arm3_docuse/calls.parquet"); dc = dc[dc["repeat"] == 0]
    nc = pd.read_parquet(f"{W}/results/arm3_nonregional_sites/calls_jev.parquet"); nc = nc[nc["repeat"] == 0]
    hc = pd.read_parquet(f"{H}/results/arm3_heldout/documented_calls.parquet"); hc = hc[hc["repeat"] == 0]
    nr_ids = sorted(nc["item_id"].unique())
    nr_key = {i: nr["truth"]["level"] for i in nr_ids}
    a3 = pd.read_parquet("results/arm3/calls.parquet"); a3 = a3[(a3["variant"] == "names") & (a3["repeat"] == 0)]
    nb = pd.read_parquet(f"{W}/results/arm3_nonregional_sites/calls_chatbots.parquet"); nb = nb[(nb["variant"] == "names") & (nb["repeat"] == 0)]
    hb = pd.read_parquet(f"{H}/results/arm3_heldout/calls.parquet"); hb = hb[(hb["variant"] == "names") & (hb["repeat"] == 0)]
    spec = {"main": (dc, a3, main_key), "heldout": (hc, hb, held_key),
            "misleading_feature": (dc, cf, conf_key), "nonregional_sites": (nc, nb, nr_key)}
    ref = J("results/summaries/hybrid_explore.json")["structured_safety_net"]["primary"]
    out, IDX = {}, {}
    for v in ("documented", "documented_notes"):
        o = {"sets": {}}
        for name, (jc, bc, key) in spec.items():
            ids = sorted(key)
            j = jc[jc["variant"] == v].set_index("item_id").reindex(ids)
            assert j["answer"].notna().all(), (v, name)
            truth = pd.Series(key).reindex(ids)
            jans = j["answer"].where(j["valid"].fillna(False).astype(bool))
            jok = (jans == truth).to_numpy()
            keep = (j["top_prob"].to_numpy(float) >= 1 - TOL)
            ju = (j["tokens_in"].fillna(0) * PRICE["jev"][0] + j["tokens_out"].fillna(0) * PRICE["jev"][1]).to_numpy() / 1e6
            n = len(ids)
            r = {"n": n, "jev_correct": int(jok.sum()), "kept_by_jev": int(keep.sum()), "share_kept": float(keep.mean()),
                 "jev_wrong_kept": int((keep & ~jok).sum()), "jev_wrong_sent": int((~keep & ~jok).sum()),
                 "jev_right_full_confidence": int((keep & jok).sum()), "jev_usd_per_1000": per(ju, n), "chatbots": {}}
            if name in ("main", "misleading_feature", "nonregional_sites"):
                rr = ref[f"{v}|median"]["sets"][name]
                assert rr["sent"] == n - r["kept_by_jev"] and rr["wrong_kept"] == r["jev_wrong_kept"] and rr["jev_correct"] == r["jev_correct"], (v, name)
            for s in MAIN:
                b = bc[bc["system"] == s].set_index("item_id").reindex(ids)
                assert b["system"].notna().all(), (name, s)
                bans = b["answer"].where(b["valid"].fillna(False).astype(bool))
                fin = pd.Series(np.where(keep, jans, bans), index=ids)
                fok = (fin == truth).to_numpy()
                bok = (bans == truth).to_numpy()
                cons = lambda a: int(sum((x is None or (isinstance(x, float) and np.isnan(x)) or GRP3.get(x) != GRP3[t])
                                         for x, t in zip(a, truth)))
                bu = (b["tokens_in"].fillna(0) * PRICE[s][0] + b["tokens_out"].fillna(0) * PRICE[s][1]).to_numpy() / 1e6
                cost = per(ju, n) + per(bu[~keep], n)
                idx = IDX.setdefault(n, np.random.default_rng(SEED).integers(0, n, size=(B, n)))
                dd = fok[idx].mean(1) - bok[idx].mean(1)
                r["chatbots"][s] = {"diff_pts": {"est": pts(fok.mean() - bok.mean()),
                                                 "ci": [pts(np.quantile(dd, 0.025)), pts(np.quantile(dd, 0.975))],
                                                 "lower_bound_bonferroni_one_sided": pts(np.quantile(dd, 0.025 / len(MAIN)))},
                                    "accuracy": prop(int(fok.sum()), n), "errors": int((~fok).sum()),
                                    "consequential_errors": cons(fin.tolist()),
                                    "chatbot_alone_accuracy": prop(int(bok.sum()), n), "chatbot_alone_errors": int((~bok).sum()),
                                    "chatbot_alone_consequential": cons(bans.tolist()),
                                    "usd_per_1000": cost, "usd_per_1000_chatbot": per(bu, n), "cost_share": cost / per(bu, n)}
                if name in ("main", "misleading_feature", "nonregional_sites"):
                    assert ref[f"{v}|median"]["sets"][name]["chatbots"][s]["hybrid_correct"] == int(fok.sum()), (v, name, s)
            o["sets"][name] = r
        # full confidence as a test for Jev's errors, this version only
        for grp, names in (("ordinary", ("main", "heldout")), ("difficult", ("misleading_feature", "nonregional_sites")),
                           ("all", tuple(spec))):
            S = [o["sets"][x] for x in names]
            wrong = sum(x["n"] - x["jev_correct"] for x in S)
            wrong_sent = sum(x["jev_wrong_sent"] for x in S)
            right = sum(x["jev_correct"] for x in S)
            right_full = sum(x["jev_right_full_confidence"] for x in S)
            o[f"full_confidence_test_{grp}"] = {"sensitivity": prop(wrong_sent, wrong), "specificity": prop(right_full, right)}
        out[v] = o
    return out


# ------------------------------------------------------------------------------------------------ D
def part_d():
    spec = [s for s in SETS if s[0] == "board_exam"][0]
    d = load_set(*spec)
    ids, OK, U, TOP = d["ids"], d["OK"], d["U"], d["TOP"]
    it = pd.read_csv("data/arm2_ext/items.csv", dtype=str)
    src = pd.Series(dict(zip(it["item_id"], it["source"]))).reindex(ids).to_numpy()
    top = TOP["jev"].to_numpy(float)
    jok = OK["jev"].to_numpy()
    allkeep = top >= 1 - TOL
    out = {"overall_full_confidence": prop(int((allkeep & jok).sum()), int(allkeep.sum())), "datasets": {}, "transfer": {}}

    def workflow(mask, keep):
        r = {"n": int(mask.sum()), "kept": int((keep & mask).sum()),
             "kept_accuracy": prop(int((keep & mask & jok).sum()), int((keep & mask).sum())), "chatbots": {}}
        for s in MAIN:
            bo = OK[s].to_numpy()
            h = np.where(keep, jok, bo)
            cu = U["jev"].to_numpy()[mask].sum() + U[s].to_numpy()[mask & ~keep].sum()
            r["chatbots"][s] = {"workflow_accuracy": float(h[mask].mean()), "chatbot_alone": float(bo[mask].mean()),
                                "diff_pts": pts(h[mask].mean() - bo[mask].mean()),
                                "cost_share": float(cu / U[s].to_numpy()[mask].sum())}
        return r

    for ds in ("medqa", "medbullets"):
        out["datasets"][ds] = workflow(src == ds, allkeep)
    for dev, test in (("medqa", "medbullets"), ("medbullets", "medqa")):
        m = src == dev
        sol = OK[SOL].to_numpy()[m].mean()
        cand = sorted(set(top[m][top[m] >= 0]), reverse=True)
        t_sel = None
        for t in cand:
            k = m & (top >= t - TOL)
            if jok[k].mean() >= sol:
                t_sel = t
        t_sel = t_sel if t_sel is not None else 1.0
        out["transfer"][f"{dev}_to_{test}"] = {"threshold": float(t_sel), "sol_accuracy_dev": float(sol),
                                               "dev": workflow(m, top >= t_sel - TOL),
                                               "test": workflow(src == test, top >= t_sel - TOL)}
    return out


# ------------------------------------------------------------------------------------------------ E
def vec_tokens(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 0.0
    dct = json.loads(p) if isinstance(p, str) else dict(p)
    return len(json.dumps(dct)) / 3.0


def part_e(MASKS):
    out = {}
    for key, _, calls, variant, *_ in SETS:
        c = pd.read_parquet(calls)
        c = c[(c["variant"] == variant) & (c["repeat"] == 0) & c["system"].isin(PRICE)].copy()
        c["vec"] = [0.0 if s == "jev" else vec_tokens(p) for s, p in zip(c["system"], c["probs"])]
        c["vec"] = np.minimum(c["vec"], c["tokens_out"].fillna(0))
        c["usd"] = [(ti * PRICE[s][0] + to * PRICE[s][1]) / 1e6 for s, ti, to in zip(c["system"], c["tokens_in"].fillna(0), c["tokens_out"].fillna(0))]
        c["usd_nv"] = c["usd"] - c["vec"] * [PRICE[s][1] / 1e6 for s in c["system"]]
        M = MASKS[key]
        ids = M["ids"]; n = len(ids)
        F = c.pivot_table(index="item_id", columns="system", values="usd", aggfunc="first").reindex(ids).fillna(0.0)
        NV = c.pivot_table(index="item_id", columns="system", values="usd_nv", aggfunc="first").reindex(ids).fillna(0.0)
        r = {"vector_share_of_cost": {s: float(1 - NV[s].sum() / F[s].sum()) for s in F if s != "jev" and F[s].sum() > 0},
             "mean_vector_tokens_est": {s: float(c[c["system"] == s]["vec"].mean()) for s in F if s != "jev"},
             "mean_output_tokens": {s: float(c[c["system"] == s]["tokens_out"].fillna(0).mean()) for s in F if s != "jev"}}
        uj = M["use_j"]; jv = F["jev"].sum()
        hh = {}
        for s in MAIN + [LUNA]:
            if s not in F:
                continue
            full = (jv + F[s][~uj].sum()) / F[s].sum()
            nv = (jv + NV[s][~uj].sum()) / NV[s].sum()
            hh[s] = {"cost_share_as_reported": float(full), "cost_share_without_vectors": float(nv)}
        r["jev_plus_chatbot_half"] = hh
        L, S = LUNA, SOL
        solnv, lunanv = NV[S].sum(), NV[L].sum()
        wf = {"luna_then_sol": F[L].sum() + NV[S][~M["keepL"]].sum(),
              "chain_jev_luna_sol": jv + F[L][~uj].sum() + NV[S][M["chain_sol"]].sum(),
              "agreement_cascade": jv + NV[L].sum() + NV[S][M["esc"]].sum(),
              "luna_confidence_matched": F[L].sum() + NV[S][M["escL"]].sum(),
              "jev_plus_luna": jv + NV[L][~uj].sum()}
        r["workflows_without_unused_vectors"] = {w: {"share_of_sol": float(v / solnv), "share_of_luna": float(v / lunanv)} for w, v in wf.items()}
        out[key] = r
    return out


# ------------------------------------------------------------------------------------------------ summary
def summarise(res):
    A, Bg, C, D, E = (res[k] for k in ("A_cheap_control", "B_aggregation", "C_structured_workflow", "D_threshold_transfer",
                                        "E_vector_overhead"))
    DEF = ["next_step", "board_exam", "staging_definitions", "aki_definitions"]
    rng = lambda v: {"min": float(min(v)), "max": float(max(v))}
    P = C["documented"]["sets"]
    ORD, DIF = ("main", "heldout"), ("misleading_feature", "nonregional_sites")
    never_worse = all(P[x]["chatbots"][s]["accuracy"]["k"] >= P[x]["chatbots"][s]["chatbot_alone_accuracy"]["k"]
                      for x in P for s in MAIN)
    no_cons_added = all(P[x]["chatbots"][s]["consequential_errors"] <= P[x]["chatbots"][s]["chatbot_alone_consequential"]
                        for x in P for s in MAIN)
    assert never_worse and no_cons_added
    worst_lb = min(P[x]["chatbots"][s]["diff_pts"]["lower_bound_bonferroni_one_sided"] for x in P for s in MAIN)
    q = J("results/summaries/hybrid_explore.json")["structured_safety_net"]["primary"]["documented|q25"]
    t = D["transfer"]
    S = {
        "workflow": {
            "ordinary_n": sum(P[x]["n"] for x in ORD),
            "ordinary_share_kept": rng([P[x]["share_kept"] for x in ORD]),
            "ordinary_wrong_kept": sum(P[x]["jev_wrong_kept"] for x in ORD),
            "ordinary_cost_share": rng([P[x]["chatbots"][s]["cost_share"] for x in ORD for s in MAIN]),
            "ordinary_sol_correct": sum(P[x]["chatbots"][SOL]["accuracy"]["k"] for x in ORD),
            "ordinary_sol_cost_share": rng([P[x]["chatbots"][SOL]["cost_share"] for x in ORD]),
            "never_less_accurate_than_chatbot": never_worse, "worst_lower_bound_pts": worst_lb,
            "non_inferior_all": bool(worst_lb > -5), "no_consequential_errors_added": no_cons_added,
            "claude_main_errors": {"alone": P["main"]["chatbots"]["claude"]["chatbot_alone_errors"], "workflow": P["main"]["chatbots"]["claude"]["errors"]},
            "glm_main_errors": {"alone": P["main"]["chatbots"]["glm"]["chatbot_alone_errors"], "workflow": P["main"]["chatbots"]["glm"]["errors"]},
            "misleading_kept": P["misleading_feature"]["kept_by_jev"], "nonregional_kept": P["nonregional_sites"]["kept_by_jev"],
            "difficult_n": sum(P[x]["n"] for x in DIF),
            "difficult_wrong_kept": sum(P[x]["jev_wrong_kept"] for x in DIF),
            "misleading_cost_share": rng([P["misleading_feature"]["chatbots"][s]["cost_share"] for s in MAIN]),
            "nonregional_cost_share": rng([P["nonregional_sites"]["chatbots"][s]["cost_share"] for s in MAIN]),
            "full_confidence": C["documented"]["full_confidence_test_all"],
            "full_confidence_difficult": C["documented"]["full_confidence_test_difficult"],
            "notes_version_full_confidence": C["documented_notes"]["full_confidence_test_all"],
            "wrong_kept_at_q25": sum(v["wrong_kept"] for v in q["sets"].values()), "q25_cut": q["cut"],
            "diff_pts": rng([P[x]["chatbots"][m]["diff_pts"]["est"] for x in P for m in MAIN]),
        },
        "transfer": {
            "board_kept": D["overall_full_confidence"]["n"], "board_kept_right": D["overall_full_confidence"]["k"],
            "medqa_kept": D["datasets"]["medqa"]["kept_accuracy"], "medbullets_kept": D["datasets"]["medbullets"]["kept_accuracy"],
            "fixed_diff_pts": rng([c["diff_pts"] for ds in D["datasets"].values() for c in ds["chatbots"].values()]),
            "fixed_cost_share": rng([c["cost_share"] for ds in D["datasets"].values() for c in ds["chatbots"].values()]),
            "tuned_medqa_to_medbullets_sol_pts": t["medqa_to_medbullets"]["test"]["chatbots"][SOL]["diff_pts"],
            "tuned_medbullets_to_medqa_sol_pts": t["medbullets_to_medqa"]["test"]["chatbots"][SOL]["diff_pts"],
            "tuned_thresholds": [t["medqa_to_medbullets"]["threshold"], t["medbullets_to_medqa"]["threshold"]],
            "tuned_loss_max_pts": float(-min(c["diff_pts"] for k in ("medqa_to_medbullets", "medbullets_to_medqa")
                                             for c in t[k]["test"]["chatbots"].values())),
        },
        "cheap": {
            "luna_share_of_sol": rng([A[k]["cost_share_vs_sol"]["luna_alone"] for k in A]),
            "luna_minus_sol_pts": rng([A[k]["differences_pts"]["luna_minus_sol"]["est"] for k in A]),
            "luna_below_sol_pts": rng([-A[k]["differences_pts"]["luna_minus_sol"]["est"] for k in A]),
            "jev_plus_luna_share_of_luna_defined": rng([A[k]["cost_share_vs_luna"]["jev_plus_luna"] for k in DEF]),
            "jev_plus_luna_minus_luna_pts_defined": rng([A[k]["differences_pts"]["jev_plus_luna_minus_luna"]["est"] for k in DEF]),
            "jev_plus_luna_lower_ci_defined_min": float(min(A[k]["differences_pts"]["jev_plus_luna_minus_luna"]["ci"][0] for k in DEF)),
            "jev_plus_luna_saving_usd_per_1000_defined": rng([A[k]["usd_per_1000"]["luna_alone"] - A[k]["usd_per_1000"]["jev_plus_luna"] for k in DEF]),
            "chain_dominated_by_luna": [k for k in A if A[k]["dominated_by_luna_alone"]["chain_jev_luna_sol"]],
            "cascade_minus_luna_confidence_pts": {k: A[k]["differences_pts"]["cascade_minus_luna_confidence"] for k in A},
        },
        "aggregation": {k: {ep: {r: Bg[k]["jev"][ep][r]["sensitivity"]["p"] for r in ("collapse", "sum")}
                            for ep in ("stage_iv", "stage_iii_or_iv", "severe_aki") if ep in Bg[k]["jev"]} for k in Bg},
        "vector_overhead": {
            "max_share_of_chatbot_cost": float(max(v for k in E for s, v in E[k]["vector_share_of_cost"].items() if s in MAIN + [LUNA])),
            "max_change_half_hybrid_share": float(max(abs(v["cost_share_without_vectors"] - v["cost_share_as_reported"])
                                                      for k in E for v in E[k]["jev_plus_chatbot_half"].values())),
            "max_change_half_hybrid_share_pts": float(100 * max(abs(v["cost_share_without_vectors"] - v["cost_share_as_reported"])
                                                                for k in E for v in E[k]["jev_plus_chatbot_half"].values())),
        },
    }
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docuse", required=True)
    ap.add_argument("--heldout", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    res = {"note": __doc__.split("\n\n")[0], "B": B, "seed": SEED}
    res["A_cheap_control"], MASKS = part_a(rng)
    res["B_aggregation"] = part_b()
    res["C_structured_workflow"] = part_c(a.docuse, a.heldout)
    res["D_threshold_transfer"] = part_d()
    res["E_vector_overhead"] = part_e(MASKS)
    res["summary"] = summarise(res)
    Path("results/summaries/workflow_checks.json").write_text(json.dumps(res, indent=1))
    print("written")


if __name__ == "__main__":
    main()

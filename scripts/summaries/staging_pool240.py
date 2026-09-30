"""Cancer staging on the pool of 240 reports: every system in every form of the question, from the stored answers.

Reads the answers of Jev and the ten LLMs in every form (results/arm3_llm_forms/calls.parquet), the four report sets'
item files (the key: stage level, substage, T, N, M, nodes involved, tumour deposits) and the pool (data/arm3_pool/
pool240.csv). Writes results/summaries/staging_pool240.json:
  sets, levels, substages     the pool's composition
  cells                       answers stored per system and form
  acc                         accuracy by report group (Wilson 95% CI), fully confident answers and those wrong,
                              unusable answers, stage-group crossings, mean input tokens
  t6                          answers by confidence level (below 0.50, 0.50-0.74, 0.75-0.99, 1.00) and the AUC of
                              confidence for a correct answer
  cost                        list-price cost per 1,000 answers
  jev_vs                      Jev minus each LLM (paired, stratified bootstrap; non-inferiority margin -5 points,
                              one Holm family of ten LLMs per comparison)
  form_minus_names            each system's change from its own stage-name form
  pairing                     Jev kept at full confidence, the LLM otherwise: accuracy, cost share, Bonferroni bound
  t13, t13e                   clinical decisions (stage III-IV, stage IV): sensitivity, specificity, AUROC, and the
                              same decisions from summed probabilities
  parts                       Jev's four questions in the four-question forms: share right per question
  extra                       weighted kappa, ranked probability score, LLM consensus on a wrong stage, Jev's top
                              probabilities and wrong answers kept at 0.99
  combo, t6_combo             Jev and each LLM in the same form (Jev's answer at full confidence, the LLM's otherwise)

    python scripts/summaries/staging_pool240.py [--root .] [--out results/summaries/staging_pool240.json]

Float sums of probabilities are added left to right, in the level order below, so that ties between reports are the
same on every Python version."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260929
NB = 2000
TOL = 1e-9
MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
FREE = ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
FORMS = ["names", "named_field", "definitions", "structure_only", "documented_no_examples", "documented", "documented_notes"]
SETS = ["main", "heldout", "misleading_feature", "nonregional_sites"]
ITEM_FILES = {"main": "data/arm3/items.csv", "heldout": "data/arm3_heldout/items.csv",
              "misleading_feature": "data/arm3/confuser_items.csv",
              "nonregional_sites": "data/arm3_nonregional_sites/items.csv"}
CALLS = "results/arm3_llm_forms/calls.parquet"
POOL = "data/arm3_pool/pool240.csv"
LEV = ["0", "I", "IIA", "IIB", "IIC", "IIIA", "IIIB", "IIIC", "IVA-IVB", "IVC"]
# the levels of each clinical decision, in the order their probabilities are added
POS = {"III-IV": ("IIIB", "IIIC", "IVC", "IIIA", "IVA-IVB"), "IV": ("IVA-IVB", "IVC")}
G3 = [("0", ("0",)), ("I", ("I",)), ("II", ("IIA", "IIB", "IIC")), ("III", ("IIIA", "IIIB", "IIIC")),
      ("IV", ("IVA-IVB", "IVC"))]


def add(values) -> float:
    """Plain left-to-right addition."""
    t = 0
    for x in values:
        t = t + x
    return t


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, c - h), min(1, c + h)]


def grp(level):
    if not isinstance(level, str):
        return None
    if level in ("0", "I", "IIA", "IIB", "IIC"):
        return "0-II"
    return "III" if level.startswith("III") else ("IV" if level.startswith("IV") else None)


def ok(d):
    return (d.valid.fillna(False) & d.correct.fillna(False)).to_numpy(bool)


BAND = [("u", lambda p: p < 0.5), ("s", lambda p: (p >= 0.5) & (p < 0.75)), ("v", lambda p: (p >= 0.75) & (p < 1 - TOL)),
        ("c", lambda p: p >= 1 - TOL)]


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    m = ~np.isnan(score)
    score, y = score[m], y[m]
    if y.all() or (~y).all():
        return None
    r = pd.Series(score).rank().to_numpy()
    n1 = y.sum()
    n0 = (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def holm(ps, alpha=0.025):
    order = np.argsort(ps)
    m = len(ps)
    rej = [False] * m
    for i, j in enumerate(order):
        if ps[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej


def probs(d):
    P = []
    for x in d.probs:
        try:
            p = json.loads(x) if isinstance(x, str) else None
        except Exception:
            p = None
        P.append(p if isinstance(p, dict) else None)
    return P


def band(n):
    n = int(n)
    return "none" if n == 0 else ("1" if n == 1 else ("2 to 3" if n <= 3 else ("4 to 6" if n <= 6 else "7 or more")))


def main(root: Path, out_path: Path) -> dict:
    # ---- stored answers
    C = pd.read_parquet(root / CALLS)
    C = C.drop_duplicates(["system", "form", "item_id"], keep="last")
    for k in ["valid", "correct", "full_confidence", "excluded", "missing"]:
        C[k] = C[k].astype(str).str.lower().eq("true")
    C["top_prob"] = pd.to_numeric(C.top_prob, errors="coerce")
    C["usd_list"] = pd.to_numeric(C.usd_list, errors="coerce")
    C["tokens_in"] = pd.to_numeric(C.tokens_in, errors="coerce")
    SYS = [s for s in ["jev"] + MAIN + FREE if s in set(C.system)]
    # ---- key
    T = pd.concat([pd.read_csv(root / p, dtype=str)[["report_id", "level", "substage", "t", "n", "m", "nodes_involved",
                                                      "tumour_deposits"]].assign(set=s)
                   for s, p in ITEM_FILES.items()]).set_index("report_id")
    pool = pd.read_csv(root / POOL, dtype=str)
    PID = pool.report_id.tolist()
    T = T.loc[PID]
    assert len(T) == 240
    G = {"pool240": PID, "ordinary": T.index[T.set.isin(["main", "heldout"])].tolist(),
         "built_to_mislead": T.index[T.set.isin(["misleading_feature", "nonregional_sites"])].tolist(),
         "misleading_feature": T.index[T.set == "misleading_feature"].tolist(),
         "nonregional_sites": T.index[T.set == "nonregional_sites"].tolist()}
    STRAT = T.set.to_numpy()
    T["g"] = T.level.map(grp)

    def get(s, fm):
        d = C[(C.system == s) & (C.form == fm)].set_index("item_id")
        return d.reindex(PID) if len(d) else None

    out = {"sources": [CALLS], "systems": SYS, "n": 240, "sets": T.set.value_counts().to_dict(),
           "levels": T.level.value_counts().to_dict(), "substages": T.substage.value_counts().to_dict(), "cells": {},
           "acc": {}, "t6": [], "cost": {}}
    D = {}
    for s in SYS:
        for fm in FORMS:
            d = get(s, fm)
            if d is None or d.answer.isna().all() and d.valid.isna().all():
                continue
            n_have = int(d.system.notna().sum())
            out["cells"][f"{s}|{fm}"] = n_have
            if n_have < 240:
                continue
            D[(s, fm)] = d
            o = ok(d)
            tp = np.where(d.valid.fillna(False).to_numpy(bool), d.top_prob.fillna(0).to_numpy(float), np.nan)
            prob_ok = ~np.isnan(tp)
            r = {}
            for g, ids in G.items():
                m = np.isin(PID, ids)
                k = int(o[m].sum())
                n = int(m.sum())
                r[g] = {"k": k, "n": n, "p": k / n, "ci": wilson(k, n), "full": int((tp[m] >= 1 - TOL).sum()),
                        "full_err": int(((tp[m] >= 1 - TOL) & ~o[m]).sum())}
            ga = d.answer.map(grp).to_numpy()
            gt = T.g.to_numpy()
            r["unusable"] = int((~d.valid.fillna(False)).sum())
            r["III_to_0II"] = int(((gt == "III") & (ga == "0-II")).sum())
            r["cross"] = int(((ga != gt)).sum())
            r["tokens_in_mean"] = float(d.tokens_in.mean()) if d.tokens_in.notna().any() else None
            out["acc"][f"{s}|{fm}"] = r
            # answers by confidence level
            vv = d.valid.fillna(False).to_numpy(bool)
            row = {"system": s, "form": fm, "n": 240, "unusable": int((~prob_ok).sum()), "right": int((o & prob_ok).sum()),
                   "noprob": int((vv & ~prob_ok).sum())}
            for b, fn in BAND:
                m = prob_ok & fn(np.nan_to_num(tp, nan=-1))
                row[b + "_right"] = int((m & o).sum())
                row[b + "_wrong"] = int((m & ~o).sum())
            row["auc"] = auroc(np.where(prob_ok, tp, np.nan), o)
            out["t6"].append(row)
            # cost per 1,000 answers at list price
            out["cost"][f"{s}|{fm}"] = float(d.usd_list.sum() / 240 * 1000) if d.usd_list.notna().all() else None
    # ---- bootstrap (stratified by report set)
    rng = np.random.default_rng(SEED)
    IDX = [np.where(STRAT == st)[0] for st in SETS]
    BOOT = np.stack([np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in IDX]) for _ in range(NB)])

    def bdiff(a, b, mask=None):
        a = a.astype(float)
        b = b.astype(float)
        if mask is not None:
            ix = np.where(mask)[0]
            strat = STRAT[ix]
            rr = np.random.default_rng(SEED)
            parts = [ix[strat == st] for st in SETS if (strat == st).any()]
            bi = np.stack([np.concatenate([rr.choice(p, len(p), replace=True) for p in parts]) for _ in range(NB)])
            est = (a[ix] - b[ix]).mean()
            bd = (a[bi] - b[bi]).mean(1)
        else:
            est = (a - b).mean()
            bd = (a[BOOT] - b[BOOT]).mean(1)
        return 100 * est, 100 * bd

    def ni_block(jform, sform, systems):
        res = {}
        pn = []
        pw = []
        for s in systems:
            if (s, sform) not in D or ("jev", jform) not in D:
                continue
            est, bd = bdiff(ok(D[("jev", jform)]), ok(D[(s, sform)]))
            res[s] = {"est": est, "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
            pn.append(((bd <= -5).sum() + 1) / (NB + 1))
            pw.append(((bd >= -5).sum() + 1) / (NB + 1))
        ks = list(res)
        if ks:
            rn = holm(np.array(pn))
            rw = holm(np.array(pw))
            for i, s in enumerate(ks):
                res[s]["verdict"] = "non-inferior" if rn[i] else ("worse" if rw[i] else "inconclusive")
        return res

    out["jev_vs"] = {}
    for jf, sf in [("names", "names"), ("definitions", "definitions"), ("documented", "names"),
                   ("documented", "definitions"), ("documented_notes", "names"), ("documented_notes", "definitions"),
                   ("documented", "documented"), ("structure_only", "structure_only"),
                   ("documented_no_examples", "documented_no_examples"), ("documented_notes", "documented_notes")]:
        blk = ni_block(jf, sf, MAIN + FREE)   # one Holm family of ten LLMs
        out["jev_vs"][f"{jf}|{sf}|all"] = blk
        out["jev_vs"][f"{jf}|{sf}|main"] = {k: v for k, v in blk.items() if k in MAIN}
        out["jev_vs"][f"{jf}|{sf}|free"] = {k: v for k, v in blk.items() if k in FREE}
    # each system's change from its own stage-name form
    out["form_minus_names"] = {}
    for s in SYS:
        for fm in FORMS[1:]:
            if (s, fm) in D and (s, "names") in D:
                est, bd = bdiff(ok(D[(s, fm)]), ok(D[(s, "names")]))
                out["form_minus_names"][f"{s}|{fm}"] = {"est": est, "ci": [float(np.percentile(bd, 2.5)),
                                                                           float(np.percentile(bd, 97.5))]}

    # ---- pairing: Jev (jform) kept at full confidence, the LLM (lform) otherwise
    def usd(d):
        return d.usd_list.to_numpy(float)

    def pairing(jform, s, lform, gname):
        if ("jev", jform) not in D or (s, lform) not in D:
            return None
        J = D[("jev", jform)]
        L = D[(s, lform)]
        keep = (J.valid.fillna(False) & (J.top_prob >= 1 - TOL)).to_numpy(bool)
        jo, lo = ok(J), ok(L)
        po = np.where(keep, jo, lo)
        ja = J.answer.map(grp).to_numpy()
        la = np.where(L.valid.fillna(False), L.answer.map(grp), None)
        pa = np.where(keep, ja, la)
        gt = T.g.to_numpy()
        m = np.isin(PID, G[gname])
        est, bd = bdiff(po, lo, mask=m if gname != "pool240" else None)
        cj, cl = usd(J), usd(L)
        cost = (cj[m].sum() + cl[m & ~keep].sum()) / cl[m].sum()
        return {"n": int(m.sum()), "kept": int((keep & m).sum()), "kept_wrong": int((keep & m & ~jo).sum()),
                "pair": int(po[m].sum()), "alone": int(lo[m].sum()),
                "est": est, "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                "lb_bonf": float(np.percentile(bd, 100 * 0.025 / 10)),
                "cross_pair": int((m & (pa != gt)).sum()), "cross_alone": int((m & (la != gt)).sum()),
                "cost_share": float(cost),
                "usd_pair_per_1000": float((cj[m].sum() + cl[m & ~keep].sum()) / m.sum() * 1000),
                "usd_alone_per_1000": float(cl[m].sum() / m.sum() * 1000)}

    out["pairing"] = {}
    for jf in ["documented", "documented_notes"]:
        for lf in ["names", "documented"]:
            for s in MAIN + ["gpt_free"] + FREE[1:]:
                for g in G:
                    r = pairing(jf, s, lf, g)
                    if r:
                        out["pairing"][f"{jf}|{s}|{lf}|{g}"] = r

    # ---- clinical decisions (stage III-IV, stage IV) and the same decisions from summed probabilities
    out["t13"] = {}
    out["t13e"] = {}
    for (s, fm), d in D.items():
        if fm not in ("names", "definitions", "documented", "documented_notes"):
            continue
        v = d.valid.fillna(False).to_numpy(bool)
        P = probs(d)
        ans = d.answer.to_numpy()
        for ep, pos in POS.items():
            y = T.level.isin(pos).to_numpy()
            call = np.array([(a in pos) if vv else None for a, vv in zip(ans, v)], dtype=object)
            tp = int(sum(1 for c, yy in zip(call, y) if yy and c is True))
            tn = int(sum(1 for c, yy in zip(call, y) if (not yy) and c is False))
            sc = np.array([add(float(p.get(lv, 0) or 0) for lv in pos) if (p and vv) else np.nan for p, vv in zip(P, v)])
            out["t13"][f"{s}|{fm}|{ep}"] = {"pos": int(y.sum()), "neg": int((~y).sum()), "sens": tp / y.sum(),
                                            "sens_ci": wilson(tp, int(y.sum())), "spec": tn / (~y).sum(),
                                            "spec_ci": wilson(tn, int((~y).sum())), "auroc": auroc(sc, y)}
            # summed probabilities: the decision is positive when the probability summed over its levels is 0.5 or
            # more; main stage: the group with the largest sum, ties to the lower; an unreadable list keeps the level
            if fm in ("names", "definitions"):
                def MS(lv):
                    return None if lv is None else next((g for g, ls in G3 if lv in ls), None)
                c_sum = []
                gs = []
                for p, vv, cl, a in zip(P, v, call, ans):
                    if p and vv:
                        tot = add(float(x or 0) for x in p.values()) or 1.0
                        c_sum.append(add(float(p.get(lv, 0) or 0) for lv in pos) / tot >= 0.5)
                        sums = [add(float(p.get(lv, 0) or 0) for lv in ls) for _, ls in G3]
                        gs.append(G3[int(np.argmax(sums))][0])
                    else:
                        c_sum.append(cl)
                        gs.append(MS(a) if vv else None)
                tp2 = int(sum(1 for c, yy in zip(c_sum, y) if yy and c is True))
                tn2 = int(sum(1 for c, yy in zip(c_sum, y) if (not yy) and c is False))
                chg = int(sum(1 for a_, b_ in zip(call, c_sum) if a_ is not None and b_ is not None and a_ != b_))
                tg = np.array([MS(lv) for lv in T.level], dtype=object)
                lg = np.array([MS(a) if vv else None for a, vv in zip(ans, v)], dtype=object)
                out["t13e"][f"{s}|{fm}|{ep}"] = {"sens_level": tp / y.sum(), "sens_sum": tp2 / y.sum(),
                                                 "spec_level": tn / (~y).sum(), "spec_sum": tn2 / (~y).sum(),
                                                 "changed": chg, "main_acc_level": float(np.mean(lg == tg)),
                                                 "main_acc_sum": float(np.mean(np.array(gs, dtype=object) == tg))}

    # ---- Jev's four questions in the four-question forms
    TRUTH = {"t_category": T.t.tolist(), "regional_nodes": [band(x) for x in T.nodes_involved],
             "tumor_deposits": ["present" if int(x) > 0 else "absent" for x in T.tumour_deposits],
             "distant_metastasis": ["none" if m == "M0" else m for m in T.m]}
    out["parts"] = {}
    for fm in ["structure_only", "documented_no_examples", "documented", "documented_notes"]:
        if ("jev", fm) not in D:
            continue
        d = D[("jev", fm)]
        acc = {q: 0 for q in TRUTH}
        allc = 0
        for i, x in enumerate(d.parts):
            try:
                p = json.loads(x)
            except Exception:
                continue
            a = [p.get(q, {}).get("choice") == TRUTH[q][i] for q in TRUTH]
            for q, aa in zip(TRUTH, a):
                acc[q] += aa
            allc += all(a)
        out["parts"][fm] = {q: v / 240 for q, v in acc.items()}
        out["parts"][fm]["all_four"] = allc / 240

    # ---- weighted kappa, ranked probability score, LLM consensus on a wrong stage, Jev's top probabilities
    LI = {lv: i for i, lv in enumerate(LEV)}

    def qwk(a, b):
        m = np.array([x is not None for x in a])
        a = np.array([LI[x] for x in np.array(a)[m]])
        b = np.array([LI[x] for x in np.array(b)[m]])
        K = len(LEV)
        O = np.zeros((K, K))
        for x, y in zip(a, b):
            O[x, y] += 1
        Wt = np.array([[(i - j) ** 2 / (K - 1) ** 2 for j in range(K)] for i in range(K)])
        E = np.outer(O.sum(1), O.sum(0)) / O.sum()
        return float(1 - (Wt * O).sum() / (Wt * E).sum())

    def rps(P, t):
        s = []
        for p, tt in zip(P, t):
            if not p:
                continue
            v = np.array([float(p.get(lv, 0) or 0) for lv in LEV])
            if v.sum() <= 0:
                continue
            v = v / v.sum()
            o = np.zeros(len(LEV))
            o[LI[tt]] = 1
            s.append(((np.cumsum(v) - np.cumsum(o)) ** 2).sum() / (len(LEV) - 1))
        return float(np.mean(s)) if s else None

    out["extra"] = {}
    for (s, fm), d in D.items():
        if fm not in ("names", "definitions"):
            continue
        v = d.valid.fillna(False).to_numpy(bool)
        ans = [a if (vv and a in LI) else None for a, vv in zip(d.answer, v)]
        out["extra"][f"{s}|{fm}"] = {"kappa": qwk(ans, T.level.tolist()),
                                     "rps": rps([p if vv else None for p, vv in zip(probs(d), v)], T.level.tolist())}
    for fm in ("names", "definitions"):
        A = [D[(s, fm)] for s in MAIN if (s, fm) in D]
        if len(A) == len(MAIN):
            cnt = 0
            for i, tl in enumerate(T.level):
                vals = [a.answer.iloc[i] for a in A if bool(a.valid.iloc[i]) and a.answer.iloc[i] != tl]
                if vals and pd.Series(vals).value_counts().iloc[0] >= 3:
                    cnt += 1
            out["extra"][f"consensus_wrong|{fm}"] = cnt
    out["extra"]["jev_names_max_top"] = float(D[("jev", "names")].top_prob.max())
    for fm in ("documented", "documented_notes"):
        J = D[("jev", fm)]
        o = ok(J)
        out["extra"][f"wrong_kept_099|{fm}"] = int(((J.top_prob >= 0.99) & ~o).sum())
        out["extra"][f"kept_099|{fm}"] = int((J.top_prob >= 0.99).sum())

    # ---- Jev and each LLM in the same form: Jev's answer at full confidence, the LLM's otherwise
    out["combo"] = {}
    out["t6_combo"] = []
    for fm in FORMS:
        if ("jev", fm) not in D:
            continue
        J = D[("jev", fm)]
        jv = J.valid.fillna(False).to_numpy(bool)
        jt = J.top_prob.to_numpy(float)
        jo = ok(J)
        keep = jv & (np.nan_to_num(jt) >= 1 - TOL)
        for m in MAIN + FREE:
            if (m, fm) not in D:
                continue
            L = D[(m, fm)]
            lv = L.valid.fillna(False).to_numpy(bool)
            lt = L.top_prob.to_numpy(float)
            lo = ok(L)
            co = np.where(keep, jo, lo)
            cv = np.where(keep, True, lv)
            ct = np.where(keep, jt, np.where(lv, lt, np.nan))
            r = {}
            for g, ids in G.items():
                mm = np.isin(PID, ids)
                k = int(co[mm].sum())
                nn = int(mm.sum())
                r[g] = {"k": k, "n": nn, "p": k / nn, "ci": wilson(k, nn)}
            cj, cl = usd(J), usd(L)
            r["cost_per_1000"] = float((cj.sum() + cl[~keep].sum()) / 240 * 1000)
            r["kept"] = int(keep.sum())
            out["combo"][f"{m}|{fm}"] = r
            prob_ok = cv & ~np.isnan(ct)
            row = {"system": m, "form": fm, "n": 240, "unusable": int((~prob_ok).sum()), "right": int((co & prob_ok).sum()),
                   "kept": int(keep.sum())}
            for b, fn in BAND:
                mk = prob_ok & fn(np.nan_to_num(ct, nan=-1))
                row[b + "_right"] = int((mk & co).sum())
                row[b + "_wrong"] = int((mk & ~co).sum())
            row["noprob"] = 0
            row["auc"] = auroc(np.where(prob_ok, ct, np.nan), co)
            out["t6_combo"].append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("systems", SYS, "; complete cells", sum(1 for v in out["cells"].values() if v >= 240), "of", len(out["cells"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/summaries/staging_pool240.json")
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out)
    main(root, out if out.is_absolute() else root / out)

"""Clinical metrics from the stored answers (no model calls).

A. Accuracy by share: at each share of items given to Jev (10-90%, its most confident items by top probability, ties
   by item identifier, unusable answers last, as the hybrid rule), Jev's accuracy on the items it keeps, beside the
   hybrid (Jev on those items, each main chatbot on the rest) and the same share handed to Jev at random. Hybrid and
   random values are read from each analysis file's coverage_curve; the hybrid is recomputed here and must match it.
B. Full confidence as a test for Jev's errors with structured input (colon cancer stage): an answer short of joint
   confidence 1.00 (product of the four questions' top probabilities) counts as flagged. Sensitivity = wrong answers
   flagged; specificity = right answers with full confidence. From results/summaries/hybrid_explore.json
   (structured_safety_net, primary, median cut = 1.00), per version and report set, and pooled.
C. Clinical decisions for Jev given structured input (stage III-IV and stage IV on the 1,000 main reports, both
   versions), in the form of results/summaries/sens_spec.json, from the structured-input calls (--docuse folder).
   Also a count stated in the manuscript: the chatbots that missed no stage III-IV report with definitions shown.

  python scripts/summaries/clinical_metrics.py [--docuse <folder holding results/arm3_docuse>]
    -> results/summaries/clinical_metrics.json
Run after hybrid_all.py's inputs are in place and after sens_spec.py and hybrid_explore.py.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_all import SETS, score  # noqa: E402
from sens_spec import wilson, prop, binary, p_pos, auc_block  # noqa: E402

MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
J = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
rng = lambda v: (lambda l: {"min": float(min(l)), "max": float(max(l))})(list(v))


def share_sweep():
    out = {}
    for key, label, calls, variant, afile, truth_fn, metric, _ in SETS:
        an = J(afile)
        c = pd.read_parquet(calls)
        c = c[(c["variant"] == variant) & (c["repeat"] == 0) & c["system"].isin(["jev"] + MAIN)].copy()
        ids = sorted(c["item_id"].unique())
        truth = pd.Series(truth_fn(ids)).reindex(ids)
        c["valid"] = c["valid"].fillna(False).astype(bool)
        c["ok"] = c["valid"] & (c["answer"].values == truth.reindex(c["item_id"]).values)
        OK = c.pivot_table(index="item_id", columns="system", values="ok", aggfunc="first").reindex(ids).astype(bool)
        j = c[c["system"] == "jev"].set_index("item_id").reindex(ids)
        conf = np.where(j["valid"] & j["top_prob"].notna(), j["top_prob"].astype(float), -np.inf)
        order = sorted(range(len(ids)), key=lambda i: (-conf[i], ids[i]))
        strat = (truth == "A").to_numpy() if metric == "balanced_accuracy" else None
        n = len(ids)
        jev = OK["jev"].to_numpy()
        cc = an["coverage_curve"]
        rows = []
        for i, s in enumerate(cc["shares"]):
            k = int(math.floor(s * n + 1e-9))
            keep = np.zeros(n, bool)
            keep[order[:k]] = True
            if abs(s - 0.5) < 1e-9:
                assert set(np.array(ids)[keep]) == set(an["hybrid_share"]["jev_items"]), key
            nk = int(jev[keep].sum())
            r = {"share": float(s), "n_jev": k,
                 "jev_kept": {"n": k, "correct": nk, "wrong": k - nk, "accuracy": nk / k, "ci": wilson(nk, k)},
                 "hybrid": {}, "random": {}, "cost_share": {}}
            for m in MAIN:
                p = cc["systems"][m][i]
                assert p["n_jev"] == k, (key, m, s)
                h = float(score(np.where(keep, jev, OK[m].to_numpy()), strat))
                assert abs(h - p[f"{metric}_confidence_routed"]) < 1e-9, (key, m, s, h)
                r["hybrid"][m] = h
                r["random"][m] = p[f"{metric}_random_routed"]
                # plain accuracy (share of answers right) for every task, as Jev's kept accuracy
                am = "accuracy" if metric == "balanced_accuracy" else metric
                ha = float(np.where(keep, jev, OK[m].to_numpy()).mean())
                assert abs(ha - p[f"{am}_confidence_routed"]) < 1e-9, (key, m, s, ha)
                r.setdefault("hybrid_acc", {})[m] = ha
                r.setdefault("random_acc", {})[m] = p[f"{am}_random_routed"]
                usd_c = an["systems"][m]["cost"]["usd_list_total"] / n * 1000
                r["cost_share"][m] = p["usd_list_per_1000_confidence_routed"] / usd_c
            r["hybrid_range"] = rng(r["hybrid"].values())
            r["hybrid_acc_range"] = rng(r["hybrid_acc"].values())
            r["random_acc_range"] = rng(r["random_acc"].values())
            r["random_range"] = rng(r["random"].values())
            r["cost_share_range"] = rng(r["cost_share"].values())
            rows.append(r)
        out[key] = {"label": label, "metric": metric, "n_items": n,
                    "jev_all": float(score(jev, strat)), "jev_all_acc": float(jev.mean()),
                    "chatbot_alone": {m: an["systems"][m][metric] for m in MAIN},
                    "chatbot_alone_range": rng(an["systems"][m][metric] for m in MAIN),
                    "shares": rows}
    return out


def full_confidence():
    sn = J("results/summaries/hybrid_explore.json")["structured_safety_net"]["primary"]
    out, tot = {}, {"wrong": 0, "wrong_flagged": 0, "right": 0, "right_full": 0}
    for v in ["documented", "documented_notes"]:
        b = sn[f"{v}|median"]
        assert abs(b["cut"] - 1.0) < 1e-12, v
        out[v] = {}
        for s in ["main", "misleading_feature", "nonregional_sites"]:
            x = b["sets"][s]
            assert x["jev_wrong"] == x["wrong_sent"] + x["wrong_kept"] and x["jev_correct"] == x["right_sent"] + x["right_kept"]
            out[v][s] = {"n": x["n"], "wrong": x["jev_wrong"], "wrong_flagged": x["wrong_sent"],
                         "right": x["jev_correct"], "right_full": x["right_kept"],
                         "sensitivity": prop(x["wrong_sent"], x["jev_wrong"]),
                         "specificity": prop(x["right_kept"], x["jev_correct"])}
            tot["wrong"] += x["jev_wrong"]; tot["wrong_flagged"] += x["wrong_sent"]
            tot["right"] += x["jev_correct"]; tot["right_full"] += x["right_kept"]
        w = sum(out[v][s]["wrong"] for s in out[v]); wf = sum(out[v][s]["wrong_flagged"] for s in out[v])
        r = sum(out[v][s]["right"] for s in out[v]); rf = sum(out[v][s]["right_full"] for s in out[v])
        out[v]["all"] = {"sensitivity": prop(wf, w), "specificity": prop(rf, r)}
    out["pooled"] = {**tot, "sensitivity": prop(tot["wrong_flagged"], tot["wrong"]),
                     "specificity": prop(tot["right_full"], tot["right"]),
                     "n_reports_per_version": sum(sn["documented|median"]["sets"][s]["n"] for s in
                                                  ["main", "misleading_feature", "nonregional_sites"])}
    return out


def structured_decisions(docuse: Path):
    it = pd.read_csv("data/arm3/items.csv", dtype=str)
    truth = dict(zip(it["report_id"], it["level"]))
    c = pd.read_parquet(docuse / "results" / "arm3_docuse" / "calls.parquet")
    c = c[(c["system"] == "jev") & (c["repeat"] == 0) & c["item_id"].isin(truth)].copy()
    ends = {"stage_iii_or_iv": {"IIIA", "IIIB", "IIIC", "IVA-IVB", "IVC"}, "stage_iv": {"IVA-IVB", "IVC"}}
    out = {}
    for v in ["documented", "documented_notes"]:
        g = c[c["variant"] == v].set_index("item_id")
        assert len(g) == 1000, (v, len(g))
        t = pd.Series(truth).reindex(g.index)
        ans = g["answer"].where(g["valid"].fillna(False).astype(bool), None)
        assert (ans == t).all(), v   # every main report staged correctly (docuse_summary: exact accuracy 1.000)
        out[v] = {}
        for e, pos in ends.items():
            sc = pd.Series([p_pos(p, st, pos) for p, st in zip(g["probs"], g["prob_status"])], index=g.index)
            out[v][e] = {**binary(ans, t, pos), **auc_block(sc, t, pos)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docuse", default=".")
    a = ap.parse_args()
    ss = J("results/summaries/sens_spec.json")
    missed = ss["summary"]["staging_definitions_chatbot_missed_iii_iv"]
    res = {"note": "descriptive; stored answers only; see module docstring",
           "share_sweep": share_sweep(),
           "full_confidence": full_confidence(),
           "structured_decisions": structured_decisions(Path(a.docuse)),
           "text": {"staging_definitions_chatbots_missing_none": sum(v == 0 for k, v in missed.items() if k in MAIN)}}
    fc = res["full_confidence"]["pooled"]
    assert fc["wrong"] == fc["wrong_flagged"] == 40, "manuscript: 'flagged all 40 wrong answers'"
    assert res["text"]["staging_definitions_chatbots_missing_none"] == 4, "manuscript: 'four of the chatbots alone missed none'"
    Path("results/summaries/clinical_metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res["share_sweep"].items():
        print(f"{v['label']:36s}", " ".join(f"{int(100*r['share'])}%:{100*r['jev_kept']['accuracy']:.1f}/"
              f"{100*r['hybrid_acc_range']['min']:.1f}-{100*r['hybrid_acc_range']['max']:.1f}" for r in v["shares"]))
    print("full confidence pooled: sens", fc["sensitivity"], "spec", fc["specificity"])
    print(json.dumps(res["structured_decisions"], indent=0)[:1500])


if __name__ == "__main__":
    main()

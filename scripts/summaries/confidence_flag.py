"""How well Jev's confidence separated its right answers from its wrong ones (from the stored outputs).

For each choice-type task version, Jev's "confident" flag is the tested hybrid rule (its most confident 50%, ties by
item id, the items listed in analysis.json hybrid_share.jev_items); for next-step choice also a top probability of 1.00.
Reports, for Jev's own answers: correct answers kept by the flag (sensitivity), errors let through, errors held back
(specificity), accuracy inside the flag, and the confidence AUROC (top probability against correct, all thresholds).

  python scripts/summaries/confidence_flag.py --arm2 results/arm2 --arm3 results/arm3 --items data/arm3/items.csv
    -> results/summaries/confidence_flag.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def flag_stats(correct: pd.Series, flag: pd.Series, top: pd.Series) -> dict:
    c, f = correct.astype(bool), flag.astype(bool)
    n_ok, n_err = int(c.sum()), int((~c).sum())
    kept, through = int((c & f).sum()), int((~c & f).sum())
    return {"n": int(len(c)), "n_correct": n_ok, "n_errors": n_err, "n_flagged": int(f.sum()),
            "share_flagged": float(f.mean()),
            "correct_kept": kept, "sensitivity": kept / n_ok if n_ok else None,
            "errors_let_through": through, "errors_held_back": n_err - through,
            "specificity": (n_err - through) / n_err if n_err else None,
            "accuracy_in_flag": kept / int(f.sum()) if f.sum() else None,
            "accuracy_outside_flag": float(c[~f].mean()) if (~f).any() else None,
            "confidence_auroc": auroc(top, c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm2", default="results/arm2")
    ap.add_argument("--arm3", default="results/arm3")
    ap.add_argument("--items", default="data/arm3/items.csv")
    ap.add_argument("--out", default="results/summaries/confidence_flag.json")
    a = ap.parse_args()
    out = {"note": "descriptive; Jev's own answers; flag = the tested hybrid rule unless stated", "tasks": {}}

    d2 = Path(a.arm2)
    an2 = json.loads((d2 / "analysis.json").read_text())
    c2 = pd.read_parquet(d2 / "calls.parquet")
    j2 = c2[(c2.variant == "main") & (c2.repeat == 0) & (c2.system == "jev")].set_index("item_id")
    truth2 = np.where(j2.index.str.contains("_lv_"), "B", "A")
    ok2 = pd.Series(j2["answer"].values == truth2, index=j2.index)
    assert abs(ok2.mean() - an2["systems"]["jev"]["accuracy"]) < 1e-9
    half2 = pd.Series(j2.index.isin(an2["hybrid_share"]["jev_items"]), index=j2.index)
    out["tasks"]["next_step"] = {
        "probability_1": flag_stats(ok2, j2["top_prob"] >= 0.9999, j2["top_prob"]),
        "half": flag_stats(ok2, half2, j2["top_prob"])}

    d3 = Path(a.arm3)
    items = pd.read_csv(a.items, dtype=str)
    truth3 = dict(zip(items["report_id"], items["level"]))
    c3 = pd.read_parquet(d3 / "calls.parquet")
    for version, fn in (("staging_names", "analysis.json"), ("staging_definitions", "analysis_definitions.json")):
        an = json.loads((d3 / fn).read_text())
        v = "names" if version.endswith("names") else "definitions"
        j = c3[(c3.variant == v) & (c3.repeat == 0) & (c3.system == "jev")].set_index("item_id")
        ok = pd.Series(j["valid"].values & (j["answer"].values == j.index.map(truth3).values), index=j.index)
        assert abs(ok.mean() - an["systems"]["jev"]["exact_accuracy"]) < 1e-9, (version, ok.mean())
        half = pd.Series(j.index.isin(an["hybrid_share"]["jev_items"]), index=j.index)
        out["tasks"][version] = {"half": flag_stats(ok, half, j["top_prob"]),
                                 "top_prob_max": float(j["top_prob"].max()), "top_prob_median": float(j["top_prob"].median())}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for t, v in out["tasks"].items():
        for k, s in v.items():
            if isinstance(s, dict):
                print(f"{t:22s} {k:14s} sens {s['sensitivity']:.3f} spec {s['specificity']:.3f} errors through "
                      f"{s['errors_let_through']}/{s['n_errors']} acc_in {s['accuracy_in_flag']:.3f} auroc {s['confidence_auroc']:.3f}")


if __name__ == "__main__":
    main()

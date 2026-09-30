"""Arm 3 cancer staging: where the wrong answers went (descriptive, from the stored outputs).

For each system and version (level names only, with stage definitions): the confusion matrix of true level by answered
level (main pass, unusable answers in their own column), accuracy on the main stage (0, I, II, III, IV), and three
crossings: stage III reports answered 0-II, IVC reports answered IVA-IVB, and wrong answers that stay within 0-II.
Exact accuracy recomputed here must equal analysis.json, and Jev's crossing counts must equal
results/summaries/arm3_summary.json (run arm3_summary.py first).

  python scripts/summaries/arm3_confusion.py [--dir results/arm3] [--items data/arm3/items.csv]
    -> results/summaries/arm3_confusion.json
"""
import argparse, json
from pathlib import Path
import pandas as pd

LEVELS = ["0", "I", "IIA", "IIB", "IIC", "IIIA", "IIIB", "IIIC", "IVA-IVB", "IVC"]
MAIN_STAGE = {"0": "0", "I": "I", "IIA": "II", "IIB": "II", "IIC": "II", "IIIA": "III", "IIIB": "III", "IIIC": "III",
              "IVA-IVB": "IV", "IVC": "IV"}
LOW = {"0", "I", "IIA", "IIB", "IIC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/arm3")
    ap.add_argument("--items", default="data/arm3/items.csv")
    ap.add_argument("--out", default="results/summaries/arm3_confusion.json")
    a = ap.parse_args()
    d = Path(a.dir)
    items = pd.read_csv(a.items, dtype=str)
    truth = dict(zip(items["report_id"], items["level"]))
    calls = pd.read_parquet(d / "calls.parquet")
    res = {"note": "descriptive; main pass; rows true level, columns answered level", "levels": LEVELS, "versions": {}}
    for version, fn in (("names", "analysis.json"), ("definitions", "analysis_definitions.json")):
        an = json.loads((d / fn).read_text())
        c = calls[(calls["variant"] == version) & (calls["repeat"] == 0)].copy()
        c["truth"] = c["item_id"].map(truth)
        assert c["truth"].notna().all()
        c["valid"] = c["valid"].fillna(False).astype(bool)
        c["ans"] = c["answer"].where(c["valid"], "unusable")
        out = {}
        for s, g in c.groupby("system"):
            ok = g["ans"] == g["truth"]
            assert abs(ok.mean() - an["systems"][s]["exact_accuracy"]) < 1e-9, (version, s)
            assert set(g.loc[g["valid"], "ans"]) <= set(LEVELS), (version, s, set(g["ans"]) - set(LEVELS))
            cm = pd.crosstab(g["truth"], g["ans"]).reindex(index=LEVELS, columns=LEVELS + ["unusable"], fill_value=0)
            v = g[g["valid"]]
            main_ok = g["valid"] & (g["ans"].map(MAIN_STAGE) == g["truth"].map(MAIN_STAGE))
            out[s] = {"n": int(len(g)), "confusion": cm.astype(int).values.tolist(),
                      "exact_accuracy": float(ok.mean()), "main_stage_accuracy": float(main_ok.mean()),
                      "stage_iii_n": int(g["truth"].map(MAIN_STAGE).eq("III").sum()),
                      "stage_iii_to_0_ii": int((v["truth"].map(MAIN_STAGE).eq("III") & v["ans"].isin(LOW)).sum()),
                      "ivc_n": int(g["truth"].eq("IVC").sum()),
                      "ivc_to_iva_ivb": int((v["truth"].eq("IVC") & v["ans"].eq("IVA-IVB")).sum()),
                      "errors": int((~ok).sum()),
                      "errors_within_0_ii": int((v["truth"].isin(LOW) & v["ans"].isin(LOW) & (v["ans"] != v["truth"])).sum())}
        res["versions"][version] = out
    j = res["versions"]["names"]["jev"]
    summ = json.loads(Path("results/summaries/arm3_summary.json").read_text())["jev"]["crossings"]
    for k in ("stage_iii_to_0_ii", "ivc_to_iva_ivb", "errors_within_0_ii", "errors"):
        assert j[k] == summ[k], (k, j[k], summ[k])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for v, out in res["versions"].items():
        for s, x in out.items():
            print(f"{v:12s} {s:12s} exact {x['exact_accuracy']:.3f} main {x['main_stage_accuracy']:.3f} "
                  f"III->0-II {x['stage_iii_to_0_ii']:3d} IVC->IVA-IVB {x['ivc_to_iva_ivb']:3d}")


if __name__ == "__main__":
    main()

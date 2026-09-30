"""eICU AKI supplement tables in the arm 3 style (scripts/arm3_supplement.py's run): repeatability, cost, each chatbot
against Jev, the hybrid at analysis.hybrid_share, the shared coverage curve, and for every version Jev's confident half
(the share of its correct answers kept and of its wrong answers let through), the confusion by stage and the baseline
sensitivity; then the side set's tables. Descriptive: the confirmatory outcomes are the shared analysis's. Unusable
answers count as wrong.

Baseline sensitivity: the key compares each value with earlier values only; the records whose stage changes when the
baseline is read either side (items column baseline_sensitive) are set apart. Per system: exact accuracy without them,
Jev minus the system without them (paired), and on them the share matching the key and the share matching the
either-side stage.

Side set (config/eicu_aki_side.yaml): per subset (patients on maintenance dialysis; stays with no creatinine value in
the first 72 h) and system, exact accuracy against the key and the answers given. Written only once its calls exist.

    python scripts/run_categorical.py eicu_aki analyze
    python scripts/run_categorical.py eicu_aki_side analyze
    python scripts/eicu_aki_supplement.py

Writes results/eicu_aki/supplement_<version>.json and supplement_tables.md, and results/eicu_aki_side/side.json and
side_tables.md. No network, no model call."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import arm3_supplement as A3
import run_categorical as RC
from jevity import categorical as K
from jevity import categorical_analysis as KA
from jevity.clients import MODELS

UNPARSED = "unusable"
MAIN_CHATBOTS_TIER = "primary"


def _share(num: np.ndarray, den: np.ndarray) -> float | None:
    return float(num.sum() / den.sum()) if den.sum() else None


def _boot_ci(stat, idx: np.ndarray, ci: float) -> list[float] | None:
    x = np.array([np.nan if (v := stat(b)) is None else v for b in idx], dtype=float)
    x = x[~np.isnan(x)]
    return KA.percentile_ci(x, ci) if len(x) else None


def confident_half(t: KA.Table, share: float, idx: np.ndarray, ci: float) -> dict:
    """Jev's confident half (KA.hybrid_split, Jev's output only): of its correct answers the share inside (kept), of its
    wrong answers the share inside (let through), and its accuracy inside and outside; intervals over the resampled
    items (the half stays as chosen on the full set)."""
    use = KA.hybrid_split(t, share)
    c = t.correct[:, t.col("jev")].astype(bool)
    w = ~c
    fns = [lambda ix: _share(c[ix] & use[ix], c[ix]), lambda ix: _share(w[ix] & use[ix], w[ix]),
           lambda ix: _share(c[ix] & use[ix], use[ix]), lambda ix: _share(c[ix] & ~use[ix], ~use[ix])]
    all_ = np.arange(len(c))
    point = [f(all_) for f in fns]
    cis = [_boot_ci(f, idx, ci) for f in fns]
    return {"n_confident": int(use.sum()), "correct": int(c.sum()), "wrong": int(w.sum()),
            "kept": point[0], "kept_ci": cis[0], "let_through": point[1], "let_through_ci": cis[1],
            "accuracy_confident": point[2], "accuracy_confident_ci": cis[2],
            "accuracy_other": point[3], "accuracy_other_ci": cis[3]}


def confusion(t: KA.Table, s: str) -> dict:
    """Counts: rows the key's stage, columns the answer (and unusable)."""
    j = t.col(s)
    cols = list(t.labels) + [UNPARSED]
    counts = {k: {a: 0 for a in cols} for k in t.labels}
    for tr, an in zip(t.truth, t.answer[:, j]):
        counts[t.labels[tr]][t.labels[an] if an >= 0 else UNPARSED] += 1
    return {"answers": cols, "counts": counts}


def baseline_sensitivity(t: KA.Table, sensitive: np.ndarray, either: np.ndarray, idx: np.ndarray, ci: float) -> dict:
    """`sensitive`: records whose stage depends on the baseline reading; `either`: the either-side stage index."""
    keep = ~sensitive
    jj = t.col("jev")
    out = {"n_records": int(len(keep)), "n_sensitive": int(sensitive.sum()), "systems": {}}
    for s in t.systems:
        j = t.col(s)
        cor = t.correct[:, j].astype(bool)
        acc = lambda ix, cor=cor: _share(cor[ix] & keep[ix], keep[ix])
        row = {"accuracy_without": acc(np.arange(len(cor))), "accuracy_without_ci": _boot_ci(acc, idx, ci),
               "on_sensitive_key": _share(cor & sensitive, sensitive),
               "on_sensitive_either_side": _share((t.answer[:, j] == either) & sensitive, sensitive)}
        if s != "jev":
            cj = t.correct[:, jj].astype(bool)
            diff = lambda ix, cor=cor: (None if not keep[ix].any() else
                                        float(cj[ix][keep[ix]].mean() - cor[ix][keep[ix]].mean()))
            row["jev_minus_system_without"] = diff(np.arange(len(cor)))
            row["jev_minus_system_without_ci"] = _boot_ci(diff, idx, ci)
        out["systems"][s] = row
    return out


def _item_flags(arm: K.ArmSpec) -> pd.DataFrame:
    p = ROOT / arm.config["items"]["main"]
    return pd.read_csv(p, dtype=str, keep_default_na=False).set_index(arm.config["items"]["id_column"])


def aki_extras(arm: K.ArmSpec, t: KA.Table, n_boot: int, seed: int, ci: float) -> dict:
    share = float(arm.config["analysis"]["hybrid_share"]["jev_share"])
    idx = KA.strat_index(t.strata, n_boot, seed)
    f = _item_flags(arm).loc[t.items.item_id]
    sensitive = (f["baseline_sensitive"] == "true").to_numpy()
    either = np.array([t.labels.index(x) for x in f["stage_either_side"]])
    return {"settings": {"confident_half": ("Jev's confident half at the hybrid's share (its highest top probability, "
                                            "ties by item id): of its correct answers the share inside (kept), of its "
                                            "wrong answers, unusable ones included, the share inside (let through); "
                                            "records resampled within stage, the half fixed"),
                         "confusion": "rows the key's stage, columns the answer; unusable answers in their own column",
                         "baseline_sensitivity": ("the records whose stage "
                                                  "changes with a baseline read either side set apart; exact accuracy "
                                                  "without them, Jev minus each system without them (paired), and on "
                                                  "them the share matching the key and the either-side stage")},
            "confident_half": confident_half(t, share, idx, ci),
            "confusion": {s: confusion(t, s) for s in t.systems},
            "baseline_sensitivity": baseline_sensitivity(t, sensitive, either, idx, ci)}


def _pct(x) -> str:
    return "–" if x is None else f"{x:.1%}"


def _cis(v) -> str:
    return "" if not v else f" ({v[0]:.1%} to {v[1]:.1%})"


def sensitivity_markdown(results: dict) -> str:
    L = ["", "## Baseline sensitivity", "",
         "The key compares each value with earlier values only. Set apart: the records whose stage changes when the "
         "baseline is read either side of a value (for example a creatinine that falls after admission).", ""]
    for v, r in results.items():
        b = r["baseline_sensitivity"]
        L += [f"### Version: {v} ({b['n_sensitive']} of {b['n_records']} records set apart)", "",
              "| System | Exact accuracy without them | Jev minus the system without them | On them: matches the key | "
              "On them: matches the either-side stage |", "|---|---|---|---|---|"]
        for s, x in b["systems"].items():
            d = x.get("jev_minus_system_without")
            dc = x.get("jev_minus_system_without_ci")
            dtxt = "–" if d is None else f"{d:+.3f}" + ("" if not dc else f" ({dc[0]:+.3f} to {dc[1]:+.3f})")
            L.append(f"| {s} | {_pct(x['accuracy_without'])}{_cis(x['accuracy_without_ci'])} | {dtxt} | "
                     f"{_pct(x['on_sensitive_key'])} | {_pct(x['on_sensitive_either_side'])} |")
        L.append("")
    return "\n".join(L)


def side(arm: K.ArmSpec, paths: RC.Paths) -> dict | None:
    """The side set per subset and system: exact accuracy against the key (unusable wrong) and the answers given."""
    f = paths.calls()
    if not f.exists():
        return None
    calls = pd.read_parquet(f)
    items, sys_, *_ = RC.design(arm)
    a = arm.config["analysis"]
    seed, n_boot, ci = int(arm.config["seed"]), int(a["n_boot"]), float(a["ci"])
    out = {"arm": arm.name, "note": "descriptive; a test of its own, never pooled with the main set", "versions": {}}
    for v in sorted({i.variant for i in items}, key=lambda x: (x != arm.config.get("primary_variant"), x)):
        t = KA.Table(calls, RC.items_frame(items, v), sys_, arm.labels, MODELS, v)
        res = {}
        for sub in sorted(set(t.strata)):
            rows = np.where(t.strata == sub)[0]
            idx = np.random.default_rng(seed).choice(rows, size=(n_boot, len(rows)), replace=True)
            key = {lv: int((t.truth[rows] == k).sum()) for k, lv in enumerate(t.labels)}
            res[sub] = {"n": int(len(rows)), "key": key, "systems": {}}
            for s in sys_:
                j = t.col(s)
                cor = t.correct[:, j].astype(bool)
                ans = t.answer[rows, j]
                res[sub]["systems"][s] = {
                    "exact_accuracy": float(cor[rows].mean()),
                    "exact_accuracy_ci": KA.percentile_ci(cor[idx].mean(axis=1), ci),
                    "answers": {**{lv: int((ans == k).sum()) for k, lv in enumerate(t.labels)},
                                UNPARSED: int((ans < 0).sum())}}
        out["versions"][v] = res
    out = KA.clean(out)
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "side.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    L = ["# eICU AKI side set", "", "Generated by `scripts/eicu_aki_supplement.py` from `results/eicu_aki_side/side.json`; "
         "do not edit by hand. Descriptive; a test of its own, never pooled with the main set. Exact accuracy against "
         "the key on what each record shows (unusable answers wrong), 95% bootstrap percentile intervals.", ""]
    for v, res in out["versions"].items():
        for sub, x in res.items():
            key = ", ".join(f"{k} {n}" for k, n in x["key"].items() if n)
            cols = list(arm.labels) + [UNPARSED]
            L += [f"## Version: {v}; {sub.replace('_', ' ')} ({x['n']} records; the key: {key})", "",
                  "| System | Exact accuracy | " + " | ".join(f"Answered {c}" for c in cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
            for s, y in x["systems"].items():
                L.append(f"| {s} | {y['exact_accuracy']:.3f}{'' if not y['exact_accuracy_ci'] else ' ({:.3f} to {:.3f})'.format(*y['exact_accuracy_ci'])} | "
                         + " | ".join(str(y["answers"][c]) for c in cols) + " |")
            L.append("")
    (paths.results / "side_tables.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    arm = K.load_arm("eicu_aki")
    paths = RC.Paths(arm)
    out = A3.run(arm, paths, primary_extras=None, every_extras=aki_extras, title="eICU AKI supplement tables",
                 unit="records", script="scripts/eicu_aki_supplement.py")
    md = paths.results / "supplement_tables.md"
    md.write_text(md.read_text(encoding="utf-8").rstrip() + "\n" + sensitivity_markdown(out), encoding="utf-8")
    print(f"wrote {', '.join(f'supplement_{v}.json' for v in out)} and supplement_tables.md under {paths.results}")
    sa = K.load_arm("eicu_aki_side")
    s = side(sa, RC.Paths(sa))
    print("side set: " + ("wrote side.json and side_tables.md" if s else "no calls yet"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

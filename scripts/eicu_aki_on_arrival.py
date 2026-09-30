"""eICU AKI: AKI already under way on unit arrival, a sensitivity analysis of the key.

The key assesses creatinine values from unit admission only (values before admission are reference values). This script
re-keys the main set with the values before admission assessed too (same KDIGO rule, same windows, from 7 days before
admission), lists the records whose stage changes, and gives each system's exact accuracy and Jev minus each main
chatbot under both keys, main pass, both versions. Descriptive; no call is made. Reads data/eicu_aki/items.csv (built by
scripts/make_eicu_aki.py from the eICU-CRD demo) and results/eicu_aki/calls.parquet.

    python scripts/eicu_aki_on_arrival.py      # writes results/eicu_aki/on_arrival.json and on_arrival.md
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import kdigo as KD  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "eicu_aki.yaml").read_text(encoding="utf-8"))
RULE = KD.Rule.from_config(CFG["kdigo"])
MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
LINE = re.compile(r"([+-])(\d+) h (\d{2}) min: (\d+\.\d{2})$")
OUT = ROOT / "results" / "eicu_aki"


def parse(record: str) -> tuple[list[tuple[int, Decimal]], int | None]:
    """(values, dialysis minute) from the record text alone."""
    values, dialysis = [], None
    for line in record.split("\n"):
        if m := LINE.match(line):
            values.append(((int(m[2]) * 60 + int(m[3])) * (-1 if m[1] == "-" else 1), Decimal(m[4])))
        elif line.startswith("Dialysis started: ") and (d := line.removeprefix("Dialysis started: ")) != "no":
            m = re.fullmatch(r"([+-])(\d+) h (\d{2}) min", d)
            dialysis = (int(m[2]) * 60 + int(m[3])) * (-1 if m[1] == "-" else 1)
    return values, dialysis


def rekey(items: pd.DataFrame, rule: KD.Rule) -> pd.DataFrame:
    """The key and the key with values before admission assessed, from each record's text."""
    wide = replace(rule, assess_from=-rule.ratio_window)
    rows = []
    for r in items.itertuples():
        values, dialysis = parse(r.record)
        key = KD.LEVELS[KD.highest_stage(values, dialysis, rule).stage]
        assert key == r.stage, r.item_id                        # the stored key, reproduced from the text
        rows.append({"item_id": r.item_id, "key": key,
                     "on_arrival": KD.LEVELS[KD.highest_stage(values, dialysis, wide).stage],
                     "baseline_sensitive": str(r.baseline_sensitive).lower() == "true"})
    return pd.DataFrame(rows)


def accuracy(calls: pd.DataFrame, keys: pd.DataFrame, variant: str) -> dict:
    """Exact accuracy per system under each key (unusable answers wrong), main pass."""
    c = calls[(calls["variant"] == variant) & (calls["repeat"] == 0)].merge(keys, on="item_id", how="inner")
    assert c.groupby("system").size().eq(len(keys)).all()
    out = {}
    for s, g in c.groupby("system"):
        out[s] = {k: float((g["valid"] & (g["answer"] == g[k])).mean()) for k in ("key", "on_arrival")}
    return out


def main() -> int:
    items = pd.read_csv(ROOT / "data" / "eicu_aki" / "items.csv")
    keys = rekey(items, RULE)
    changed = keys[keys["key"] != keys["on_arrival"]]
    calls = pd.read_parquet(OUT / "calls.parquet")
    res = {"rule": "values before unit admission assessed too, by the same KDIGO rule and windows",
           "n_records": len(keys), "n_changed": len(changed),
           "changed": changed[["item_id", "key", "on_arrival"]].to_dict("records"),
           "changed_baseline_sensitive": int(changed["baseline_sensitive"].sum()),
           "max_shift_any_accuracy_bound": len(changed) / len(keys), "versions": {}}
    for v in ("names", "definitions"):
        acc = accuracy(calls, keys, v)
        diffs = {m: {k: acc["jev"][k] - acc[m][k] for k in ("key", "on_arrival")} for m in MAIN}
        for d in diffs.values():
            d["shift"] = d["on_arrival"] - d["key"]
        res["versions"][v] = {"accuracy": acc, "jev_minus": diffs,
                              "max_abs_shift_jev_minus": max(abs(d["shift"]) for d in diffs.values())}
    (OUT / "on_arrival.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["# eICU AKI: AKI under way on arrival", "",
         f"The key assesses values from unit admission only. With the values before admission assessed too (same rule "
         f"and windows), {len(changed)} of {len(keys)} main records change stage "
         f"({res['changed_baseline_sensitive']} of them among the baseline-sensitive records):", "",
         "| Record | Key | With values before admission |", "|---|---|---|"]
    L += [f"| {r['item_id']} | {r['key']} | {r['on_arrival']} |" for r in res["changed"]]
    for v, r in res["versions"].items():
        L += ["", f"## {v.capitalize()} version: Jev minus each main chatbot, exact accuracy (points)", "",
              "| Chatbot | Key | With values before admission | Shift |", "|---|---|---|---|"]
        L += [f"| {m} | {100 * d['key']:+.1f} | {100 * d['on_arrival']:+.1f} | {100 * d['shift']:+.1f} |"
              for m, d in r["jev_minus"].items()]
    L += ["", "Main pass; unusable answers counted wrong. Descriptive.", ""]
    (OUT / "on_arrival.md").write_text("\n".join(L), encoding="utf-8")
    for v, r in res["versions"].items():
        print(v, "largest shift in Jev minus a main chatbot:", f"{100 * r['max_abs_shift_jev_minus']:.1f} points")
    print(f"{len(changed)} of {len(keys)} records change; wrote on_arrival.json and on_arrival.md under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Robustness check on reworded reports: Jev used as documented on 20 arm 3 reports rewritten by hand in other words
and layouts, with every staging fact kept (depth of invasion, involved and examined regional nodes,
tumour deposits, distant metastasis sites). The template, questions, criteria and config are those of the documented
run (config/docuse_staging.yaml, unchanged); only the report text differs. Only Jev is called.
  python scripts/docuse/run_reworded.py --build        # data/arm3_reworded/items.csv from reworded_reports.txt
  python scripts/docuse/run_reworded.py --dry-run      # builds and validates the 40 requests; sends nothing
  python scripts/docuse/run_reworded.py --run --yes    # 20 reports x 2 versions = 40 calls
  python scripts/docuse/run_reworded.py --analyze      # results/arm3_docuse_reworded/summary.json
Sample: two main reports per level, drawn with numpy default_rng(20260927) (sample_ids). Truth columns are copied from
data/arm3/items.csv by source report id. Raw responses: runs/arm3_docuse_reworded/ (first write wins)."""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical as K
from jevity import docuse_staging as D
from jevity.clients import RawStore

SEED = 20260927
TEXT = ROOT / "data" / "arm3_reworded" / "reworded_reports.txt"
ITEMS = ROOT / "data" / "arm3_reworded" / "items.csv"
STORE = ROOT / "runs" / "arm3_docuse_reworded"
RESULTS = ROOT / "results" / "arm3_docuse_reworded"
TRUTH_COLS = ["level", "substage", "t", "n", "m", "nodes_involved", "nodes_examined", "tumour_deposits", "layout",
              "with_definitions"]


def _run_staging():
    spec = importlib.util.spec_from_file_location("run_staging", ROOT / "scripts" / "docuse" / "run_staging.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample_ids() -> list[str]:
    """Two main reports per level, drawn with a fixed seed."""
    main = pd.read_csv(ROOT / D.CFG["items"]["main"], dtype={"report_id": str}, keep_default_na=False)
    rng, out = np.random.default_rng(SEED), []
    for lv in D.LEVELS:
        out += sorted(rng.choice(sorted(main.loc[main.level == lv, "report_id"]), 2, replace=False).tolist())
    return out


def parse_text(path: Path = TEXT) -> list[dict]:
    blocks = re.split(r"^===== ", path.read_text(encoding="utf-8").replace("\r\n", "\n"), flags=re.M)
    out = []
    for b in blocks:
        if not b.strip():
            continue
        head, _, body = b.partition("\n")
        rid, src = (s.strip() for s in head.split("|"))
        out.append({"report_id": rid, "source_report_id": src, "report": body.strip()})
    return out


def build() -> pd.DataFrame:
    rows = parse_text()
    main = pd.read_csv(ROOT / D.CFG["items"]["main"], dtype={"report_id": str}, keep_default_na=False).set_index("report_id")
    assert [r["source_report_id"] for r in rows] == sample_ids(), "reworded reports do not follow the drawn sample"
    df = pd.DataFrame([{**r, **main.loc[r["source_report_id"], TRUTH_COLS].to_dict()} for r in rows])
    assert not df.report.isin(main.report.str.strip()).any(), "a reworded report equals an original report"
    ITEMS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ITEMS, index=False, lineterminator="\n")
    print(f"wrote {ITEMS.relative_to(ROOT)}: {len(df)} reports; sha256 {K.file_sha256(ITEMS)}")
    return df


def load_items() -> list[K.Item]:
    df = pd.read_csv(ITEMS, dtype={"report_id": str}, keep_default_na=False)
    return [K.Item(r["report_id"], r["report"].strip(), str(r["level"]), D.LEVELS, D.LEVELS, (None,) * len(D.LEVELS),
                   str(r["level"]), str(r["substage"]), v)
            for v in D.VARIANTS for r in df.to_dict("records")]


def plan(items: list[K.Item]) -> list[K.CatCall]:
    return [K.CatCall("jev", i.item_id, 0, i.variant) for i in items]


def dry_run() -> dict:
    arm, items = D.arm_spec(), load_items()
    cl = D.DocuseJevClient(RawStore(STORE), arm)
    bad = {f"{i.variant}/{i.item_id}": p for i in items if (p := D.validate_request(cl.request(i), i))}
    rs = _run_staging()
    proj = rs.projection(arm, items, plan(items))
    out = {"requests": len(items), "reports": len({i.item_id for i in items}), "invalid": len(bad), "problems": bad,
           "tokens_in": int(proj.tokens_in.sum()), "usd_list": float(proj.usd.sum()),
           "config_sha256": D.config_sha256(), "items_sha256": K.file_sha256(ITEMS),
           "examples_found_in_reports": D.example_overlaps(sorted({i.state for i in items}))}
    print(json.dumps(out, indent=1))
    return out


def run(yes: bool) -> None:
    arm, items = D.arm_spec(), load_items()
    calls = plan(items)
    rs = _run_staging()
    print(f"{arm.name} reworded: {len(calls)} Jev calls ({len({c.item_id for c in calls})} reports; "
          f"variants {', '.join(D.VARIANTS)})")
    K.load_keys(arm)
    rs.spending_stop(arm, rs.projection(arm, items, calls))
    if not yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return
    done = K.execute(calls, items, D.make_factory(arm, RawStore(STORE)), None, workers=4, arm_name=arm.name)
    print(done.groupby("parse", dropna=False).size().to_string())
    err = done[done.error.notna()]
    if len(err):
        print(f"WARNING {len(err)} calls failed (rerun --run to retry):", err.error.str[:120].value_counts().to_dict())


def analyze() -> dict:
    arm, items = D.arm_spec(), load_items()
    df, parts = D.calls_table(RawStore(STORE), arm, items, plan(items))
    truth = pd.read_csv(ITEMS, dtype={"report_id": str}, keep_default_na=False).set_index("report_id")
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "calls.parquet", index=False)
    parts.to_parquet(RESULTS / "parts.parquet", index=False)
    orig = pd.read_parquet(ROOT / D.CFG["results"] / "calls.parquet")
    orig = orig[orig.repeat == 0].set_index(["item_id", "variant"]).answer
    per_variant, rows = {}, []
    for v in D.VARIANTS:
        d = df[df.variant == v].set_index("item_id")
        p = parts[parts.variant == v].set_index("item_id")
        ok, part_ok = [], {q: [] for q in D.QUESTION_IDS}
        for rid, t in truth.iterrows():
            want = D.parts_of_truth(t.t, int(t.nodes_involved), int(t.tumour_deposits), t.m)
            got = {q: (p.loc[rid, f"{q}_choice"] if rid in p.index else None) for q in D.QUESTION_IDS}
            ans = d.answer.get(rid)
            ok.append(ans == t.level)
            for q in D.QUESTION_IDS:
                part_ok[q].append(got[q] == want[q])
            rows.append({"variant": v, "report_id": rid, "source_report_id": t.source_report_id, "truth": t.level,
                         "answer": ans, "correct": ans == t.level,
                         "answer_on_original_report": orig.get((t.source_report_id, v)),
                         **{f"{q}_choice": got[q] for q in D.QUESTION_IDS},
                         **{f"{q}_truth": want[q] for q in D.QUESTION_IDS}})
        per_variant[v] = {"n_reports": len(truth), "answered": int(d.answer.notna().sum()),
                          "usable": int(d.valid.sum()) if len(d) else 0,
                          "exact_correct": int(sum(ok)), "exact_accuracy": float(np.mean(ok)),
                          "part_accuracy": {q: float(np.mean(x)) for q, x in part_ok.items()},
                          "wrong": [r for r in rows if r["variant"] == v and not r["correct"]]}
    summ = {"check": "reworded reports", "written": datetime.datetime.now().isoformat(timespec="seconds"),
            "config": "config/docuse_staging.yaml", "config_sha256": D.config_sha256(),
            "items": str(ITEMS.relative_to(ROOT)).replace("\\", "/"), "items_sha256": K.file_sha256(ITEMS),
            "sample": {"seed": SEED, "per_level": 2, "source_report_ids": sample_ids()},
            "calls": len(df), "providers": df.provider.value_counts().to_dict(),
            "models_reported": df.model_reported.value_counts().to_dict(),
            "usd_reported": float(df.usd_reported.sum()), "per_variant": per_variant}
    pd.DataFrame(rows).to_csv(RESULTS / "per_report.csv", index=False, lineterminator="\n")
    (RESULTS / "summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    print(json.dumps({v: {k: s[k] for k in ("answered", "usable", "exact_correct", "exact_accuracy", "part_accuracy")}
                      for v, s in per_variant.items()}, indent=1))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    if a.build:
        build()
    elif a.dry_run:
        dry_run()
    elif a.run:
        run(a.yes)
    else:
        analyze()


if __name__ == "__main__":
    main()

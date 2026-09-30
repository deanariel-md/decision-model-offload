"""Every LLM in every staging form: the coverage audit. Read only; no request is sent and no key is read.

1. Indexes every stored arm 3 LLM answer (repeat 0, not Jev) on the 240 pool reports ->
   results/arm3_llm_forms/stored_llm_index.parquet (jevity.llm_forms_audit.SOURCES).
2. Every (system, form, report) of 10 x 7 x 240 against the index with llm_forms_audit.match ->
   results/arm3_llm_forms/coverage.parquet (reused / identical_to_names / gap / build_error), and the system x form x
   set table to stdout. scripts/llm_forms/run.py sends the gaps; scripts/llm_forms/collect.py reads the reused answers.

  python scripts/llm_forms/audit.py                  # everything
  python scripts/llm_forms/audit.py --reuse-index    # read the index instead of rebuilding it
  python scripts/llm_forms/audit.py --no-coverage    # the index only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jevity import llm_forms_audit as A  # noqa: E402


def index_tables(df: pd.DataFrame) -> dict:
    d = A.describe_index(df, A.pool_reports())
    counts = (df.groupby(["system", "set", "stored_variant"]).agg(
        n=("item_id", "size"), usable=("valid", "sum"),
        route=("route", lambda x: ", ".join(f"{k} {v}" for k, v in x.value_counts().items())),
        source=("source", lambda x: ", ".join(sorted(set(x))))).reset_index())
    order = {s: i for i, s in enumerate(A.SYSTEMS)}
    counts = counts.sort_values(["system", "set", "stored_variant"],
                                key=lambda c: c.map(order) if c.name == "system" else c).reset_index(drop=True)
    grid = df.pivot_table(index="system", columns=["stored_variant", "set"], values="item_id", aggfunc="count",
                          fill_value=0)
    grid = grid.reindex(index=list(A.SYSTEMS), fill_value=0)
    cols = [(v, s) for v in ("names", "definitions") for s in A.SETS]
    grid = grid.reindex(columns=pd.MultiIndex.from_tuples(cols), fill_value=0)
    grid.columns = [f"{v} {s}" for v, s in grid.columns]
    d["counts"], d["grid"] = counts, grid.reset_index()
    return d


def coverage_tables(cov: pd.DataFrame) -> dict:
    t = A.summary(cov)
    cell = lambda r: (f"{r['reused']}/{r['identical_to_names']}/{r['gap']}"  # noqa: E731
                      + (f"/E{r['build_error']}" if r["build_error"] else ""))
    t["cell"] = t.apply(cell, axis=1)
    grid = t.pivot_table(index=["system", "form"], columns="set", values="cell", aggfunc="first").reindex(
        columns=list(A.SETS)).reset_index()
    so, fo = {s: i for i, s in enumerate(A.SYSTEMS)}, {f: i for i, f in enumerate(A.FORMS)}
    grid = grid.sort_values(["system", "form"], key=lambda c: c.map(so if c.name == "system" else fo)).reset_index(
        drop=True)
    by_form = cov.pivot_table(index="form", columns="status", values="item_id", aggfunc="count", fill_value=0)
    by_form = by_form.reindex(index=list(A.FORMS), columns=list(A.STATUSES), fill_value=0).reset_index()
    by_system = cov.pivot_table(index="system", columns="status", values="item_id", aggfunc="count", fill_value=0)
    by_system = by_system.reindex(index=list(A.SYSTEMS), columns=list(A.STATUSES), fill_value=0).reset_index()
    gaps = cov[(cov.status == "gap") & (cov.nearest_diffs != "[]")]
    why = {}
    for r in gaps.itertuples():
        for dline in json.loads(r.nearest_diffs):
            key = dline.split(" (")[0].split(":")[0]
            why[key] = why.get(key, 0) + 1
    notes = {}
    for r in cov[cov.status == "reused"].itertuples():
        for x in json.loads(r.notes):
            k = x.split(" (")[0].split(" vs ")[0]
            notes[k] = notes.get(k, 0) + 1
    itn = cov[cov.status == "identical_to_names"]
    names_status = {s: int(itn.notes.str.contains(f"names cell: {s}\"", regex=False).sum()) for s in STATUS_NAMES}
    return {"summary": t.drop(columns=["cell"]), "grid": grid, "by_form": by_form, "by_system": by_system,
            "gap_reasons": why, "reuse_notes": notes, "itn_names_status": names_status}


STATUS_NAMES = ("reused", "gap")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-index", action="store_true", help="read the existing index instead of rebuilding it")
    ap.add_argument("--no-coverage", action="store_true", help="skip the coverage against the builder")
    args = ap.parse_args()
    t0 = time.time()
    A.OUT.mkdir(parents=True, exist_ok=True)
    if args.reuse_index and A.INDEX.exists():
        df = A.load_index()
    else:
        df = A.build_index()
        A.write_index(df)
    print(f"index: {len(df)} stored LLM answers -> {A.INDEX}")
    it = index_tables(df)
    print(it["grid"].to_string(index=False))
    for c in it["checks"]:
        print(" ", c)

    if not args.no_coverage:
        from jevity import llm_forms as F
        multi: list[str] = []
        cov = A.coverage(F.build, df, log=multi.append)
        cov.to_parquet(A.COVERAGE, index=False, compression="zstd")
        ct = coverage_tables(cov)
        print(f"coverage: {len(cov)} cells -> {A.COVERAGE}; {cov.status.value_counts().to_dict()}")
        print("reused / identical_to_names / gap (/E build_error):")
        print(ct["grid"].to_string(index=False))
        print(ct["by_form"].to_string(index=False))
        print(f"notes on reused cells: {ct['reuse_notes'] or 'none'}; identical_to_names cells by their names cell: "
              f"{ct['itn_names_status']}; why the nearest stored answer was not identical (gaps): "
              f"{ct['gap_reasons'] or 'none'}")
        if multi:
            print(f"{len(multi)} cells with several identical stored answers (earliest taken)")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

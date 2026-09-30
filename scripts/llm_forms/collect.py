"""Every LLM in every staging form: results/arm3_llm_forms/calls.parquet from the stored answers only (no call).

One row per system (Jev and the 10 LLMs) x form x report of the 240 reports:
- jev: Jev's stored answer for that form and report (jevity.llm_forms.jev_source), re-scored with
  jevity.llm_forms_parse.jev_row and checked against Jev's stored calls table (answer, valid, top_prob);
- reused: the stored LLM answer whose request is character-identical (coverage.parquet, scripts/llm_forms/audit.py),
  re-scored with llm_row;
- identical_to_names: named_field cells whose built request equals the names request: the names row, relabelled;
- new: the answer in runs/arm3_llm_forms for the built request; a cell with no stored answer is kept with
  missing = True (never an answer, never scored as one).
A reply naming a model other than the main runs' (scripts/llm_forms/run.py EXPECTED_MODEL) is excluded and logged.

  python scripts/llm_forms/collect.py [--systems gpt claude ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from jevity import llm_forms as LF  # noqa: E402
from jevity import llm_forms_parse as LP  # noqa: E402
from jevity.clients import MODELS, RawStore  # noqa: E402
from run import COVERAGE, EXPECTED_MODEL, STORE, answered, retry_req, transport_failed  # noqa: E402

OUT = ROOT / "results" / "arm3_llm_forms"
JEV_MODEL = "typesafe/jev-1.13-20260917"


def usd_list(system: str, tin, tout) -> float | None:
    if system == "jev":
        pi, po = 0.042, 0.0
    elif system == "gpt":       # GPT-5.6 Sol at its list price, US$5 in and US$30 out per million tokens
        pi, po = 5.0, 30.0
    else:
        f = MODELS["families"][system]
        pi, po = float(f["price_in"]), float(f["price_out"])
    if tin is None or pd.isna(tin):
        return None
    return (float(tin) * pi + float(0 if tout is None or pd.isna(tout) else tout) * po) / 1e6


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect(systems: list[str] | None = None) -> pd.DataFrame:
    pool = LF.pool().set_index("report_id")
    cov = pd.read_parquet(COVERAGE)
    if systems:
        cov = cov[cov.system.isin(systems)]
    store = RawStore(STORE)
    rows = []
    if not systems or "jev" in systems:
        for form in LF.FORMS:
            for rid in pool.index:
                path, _ = LF.jev_source(form, rid)
                r = LP.jev_row(form, load(path))
                r.update(system="jev", form=form, item_id=rid, raw_path=str(path), source="jev_stored")
                rows.append(r)
    names_rows = {}
    for c in cov.sort_values("form", key=lambda s: s.map({f: i for i, f in enumerate(LF.FORMS)})).itertuples():
        truth = pool.level[c.item_id]
        if c.status == "reused":
            got = load(c.raw_path)
            r = LP.llm_row(c.system, c.form, c.item_id, truth, got)
            r.update(source="reused", raw_path=str(c.raw_path),
                     transport_failed=transport_failed(got.get("response")))
        elif c.status == "identical_to_names":
            r = dict(names_rows[(c.system, c.item_id)])
            r.update(form=c.form, source="identical_to_names")
        else:
            req = LF.build(c.system, c.form, c.item_id)
            cached, k, failed = answered(store, req)    # the first complete reply
            if cached is None and failed is None:
                r = {"system": c.system, "form": c.form, "item_id": c.item_id, "truth": truth, "valid": False,
                     "answer": None, "missing": True, "source": "new", "raw_path": str(store.path(req))}
            elif cached is None:                        # every stored try failed in transport: unusable, flagged
                r = LP.llm_row(c.system, c.form, c.item_id, truth, failed)
                r.update(source="new", raw_path=str(store.path(retry_req(req, k - 1))), transport_retry=k - 1,
                         transport_failed=True)
            else:
                r = LP.llm_row(c.system, c.form, c.item_id, truth, cached)
                r.update(source="new", raw_path=str(store.path(retry_req(req, k))), transport_retry=k,
                         transport_failed=False)
            if k > 0:
                r["first_raw_path"] = str(store.path(req))
        r.update(system=c.system, form=c.form, item_id=c.item_id)
        if c.form == "names":
            names_rows[(c.system, c.item_id)] = r
        rows.append(r)
    df = pd.DataFrame(rows)
    df["set"] = df.item_id.map(pool.set)
    df["truth"] = df.item_id.map(pool.level)
    df["missing"] = df.get("missing", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["valid"] = df.valid.fillna(False).astype(bool)
    df["correct"] = df.valid & (df.answer == df.truth)
    df["usd_list"] = [usd_list(s, a, b) for s, a, b in zip(df.system, df.get("tokens_in"), df.get("tokens_out"))]
    exp = {**EXPECTED_MODEL, "jev": JEV_MODEL}
    bad = (~df.missing) & df.model_reported.notna() & (df.model_reported != df.system.map(exp))
    df["excluded"] = bad
    df["exclude_reason"] = [f"model {m!r}, main runs {exp[s]!r}" if b else None
                            for s, m, b in zip(df.system, df.model_reported, bad)]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--out", default=str(OUT / "calls.parquet"))
    a = ap.parse_args()
    df = collect(a.systems)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    t = df.groupby(["system", "form", "source"]).size().unstack(fill_value=0)
    print(t.to_string())
    print("missing cells:", int(df.missing.sum()), "| excluded:", int(df.excluded.sum()))
    if "transport_retry" in df:
        tr = df[df.transport_retry.fillna(0) > 0]
        print("answered on a transport retry:", len(tr), tr.groupby("system").size().to_dict(),
              "| every try failed in transport:", int(df.get("transport_failed", pd.Series(dtype=bool)).fillna(False).sum()))
    print(df[~df.missing].groupby("system").valid.mean().round(3).to_string())


if __name__ == "__main__":
    main()

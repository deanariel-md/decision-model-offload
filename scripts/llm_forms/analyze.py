"""Every LLM in every form of the colon cancer staging question that Jev answered, on the 240 reports: the analysis
(stored answers only; no model call). Descriptive and paired only.

Input: results/arm3_llm_forms/calls.parquet (scripts/llm_forms/collect.py: one row per system incl. jev x form x
report of the 240) and config/models.yaml's standard list prices. The checks also read
results/summaries/pool_summary.json and workflow_groups.json.

  python scripts/llm_forms/analyze.py                   -> results/arm3_llm_forms/analysis.json,
                                                           results/summaries/symmetric_forms_summary.md,
                                                           results/summaries/symmetric_forms_table6_rows.json
  python scripts/llm_forms/analyze.py --systems gpt glm -> results/arm3_llm_forms/partial/<system>.json (Jev as comparator)
  python scripts/llm_forms/analyze.py --stub            -> a synthetic calls table and its outputs in a temp folder

Code paths shared with the staging summaries: Wilson interval and AUROC from scripts/summaries/sens_spec.py (prop,
auroc); names, price table and LLM groups from scripts/summaries/cost_frontier.py; score and boot_diff as hybrid_all.py
and the stratified resample index as pool_summary.py (in scripts/llm_forms/boot.py); the full-confidence cut and the
bands as src/jevity/llm_forms_parse.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "summaries"))
sys.path.insert(0, str(Path(__file__).parent))
from boot import B, SEED, SETS, diff_ci, stratified_index, workflow_pool240_index  # noqa: E402
from sens_spec import prop, auroc  # noqa: E402
from cost_frontier import NAME, MAIN, CHEAP, PRICE as FRONTIER_PRICE  # noqa: E402

OUT = ROOT / "results" / "arm3_llm_forms"
SUMMARIES = ROOT / "results" / "summaries"
POOL = ROOT / "data" / "arm3_pool" / "pool240.csv"
CONFIG = ROOT / "config" / "models.yaml"

LLMS = MAIN + CHEAP
SYSTEMS = ["jev"] + LLMS
FORMS = ["names", "named_field", "definitions", "structure_only", "documented_no_examples", "documented",
         "documented_notes"]                      # Supplementary Table 5a's row order
SINGLE = ("names", "named_field", "definitions")
TOL = 1e-9                                        # full confidence: top probability >= 1 - TOL
BANDS = ["<0.50", "0.50-0.74", "0.75-0.99", "1.00"]
# GPT-5.6 Sol: US$5 in and US$30 out per million tokens, its standard list price (as config/models.yaml and
# scripts/summaries/cost_frontier.py); 2 / 10 is the sensitivity.
GPT_PRIMARY = (5.0, 30.0)
GPT_SENS = (2.0, 10.0)
PART_FAIL = {"sum_out_of_range", "values_invalid", "labels_unknown", "labels_duplicate", "missing"}  # a question's list failed
M1 = {"M1a", "M1b", "M1c"}                        # distant_metastasis options with a metastasis
GROUPS = {"pool240": SETS, "ordinary": ["main", "heldout"],
          "built_to_mislead": ["misleading_feature", "nonregional_sites"], **{s: [s] for s in SETS}}
FORM_LABEL = {"names": "One question, level names", "named_field": "The same, report sent as a named field",
              "definitions": "Medical context: each level's T, N and M categories",
              "structure_only": "Task decomposition: four questions, bare options",
              "documented_no_examples": "Both, options defined without examples",
              "documented": "Both: recommended form", "documented_notes": "Recommended form with AJCC notes"}
GROUP_LABEL = {"pool240": "Pool", "ordinary": "Ordinary", "built_to_mislead": "Built to mislead", "main": "First 1,000",
               "heldout": "Written by another model", "misleading_feature": "Misleading feature",
               "nonregional_sites": "Metastatic node at an unusual site"}
LEVELS = ["0", "I", "IIA", "IIB", "IIC", "IIIA", "IIIB", "IIIC", "IVA-IVB", "IVC"]
COLUMNS = ["system", "form", "set", "item_id", "truth", "answer", "valid", "probs", "prob_status", "top_prob", "parse",
           "parts", "model_reported", "provider", "route", "source", "tokens_in", "tokens_out", "tokens_reasoning",
           "usd_reported", "usd_list", "latency_s", "finish_reason", "raw_path", "error", "excluded", "exclude_reason",
           "missing", "correct", "full_confidence", "confidence_band"]


# ------------------------------------------------------------------------------------------------ inputs
def prices(config: Path = CONFIG) -> dict:
    """Standard list prices, US$ per million tokens: families.<s>.price_in / price_out; Jev input only."""
    y = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    out = {"jev": (float(y["jev"]["price_per_mtok_input"]), 0.0)}
    for s in LLMS:
        f = y["families"][s]
        out[s] = (float(f["price_in"]), float(f["price_out"]))
    out["gpt"] = GPT_PRIMARY
    return out


def band(p: float) -> str:
    """As llm_forms_parse.band: <0.50; 0.50 up to 0.75; 0.75 up to full confidence; full confidence."""
    if p >= 1 - TOL:
        return "1.00"
    if p >= 0.75:
        return "0.75-0.99"
    if p >= 0.50:
        return "0.50-0.74"
    return "<0.50"


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), False).astype(bool)


def _parts(x) -> dict:
    """The parts JSON of a four-question row ({question id: {choice, prob_status, ...}}); {} when absent."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    try:
        v = json.loads(x) if isinstance(x, str) else x
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def part_failed(parts: dict) -> bool:
    """Any question's probability list failed the sum rule (or was unreadable)."""
    return any(isinstance(p, dict) and p.get("prob_status") in PART_FAIL for p in parts.values())


def tx_nx_m1(parts: dict) -> bool:
    """T cannot be assessed or regional nodes not stated, with an M1 option (the table gives no stage)."""
    ch = {q: (p.get("choice") if isinstance(p, dict) else None) for q, p in parts.items()}
    return (ch.get("t_category") == "cannot be assessed" or ch.get("regional_nodes") == "not stated") \
        and ch.get("distant_metastasis") in M1


def load_calls(path: Path) -> pd.DataFrame:
    c = pd.read_parquet(path)
    need = {"system", "form", "set", "item_id", "truth", "answer", "valid", "top_prob", "tokens_in", "tokens_out"}
    assert need <= set(c.columns), sorted(need - set(c.columns))
    c = c.copy()
    for col in ("missing", "excluded"):
        c[col] = _bool(c[col]) if col in c else False
    for col in ("exclude_reason", "source"):
        if col not in c:
            c[col] = None
    for col in ("usd_reported", "usd_list", "tokens_reasoning"):
        c[col] = pd.to_numeric(c[col], errors="coerce") if col in c else np.nan
    c["valid"] = _bool(c["valid"])
    for col in ("top_prob", "tokens_in", "tokens_out"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c["item_id"] = c["item_id"].astype(str)
    assert set(c.system) <= set(SYSTEMS), sorted(set(c.system) - set(SYSTEMS))
    assert set(c.form) <= set(FORMS), sorted(set(c.form) - set(FORMS))
    assert set(c.set) <= set(SETS), sorted(set(c.set) - set(SETS))
    dup = c.duplicated(["system", "form", "item_id"])
    assert not dup.any(), f"{int(dup.sum())} doubled (system, form, report) rows; one answer per request"
    return c


def universe(c: pd.DataFrame, pool: Path | None) -> pd.DataFrame:
    """The reports (item_id, set, truth), sorted by report id. The pool file when given; else the calls' own reports."""
    if pool is not None:
        u = pd.read_csv(pool, dtype=str).rename(columns={"report_id": "item_id"})[["item_id", "set"]]
        known = u.set_index("item_id").set
        assert c.item_id.isin(known.index).all(), "a report outside the pool"
        assert (c.item_id.map(known) == c.set).all(), "a report's set differs from the pool file"
    else:
        u = c[["item_id", "set"]].drop_duplicates()
    assert not u.item_id.duplicated().any(), "a report in two sets"
    t = c[c.truth.notna()].groupby("item_id").truth
    assert (t.nunique() == 1).all(), "rows disagree on a report's truth"
    u = u.assign(truth=u.item_id.map(t.first())).sort_values("item_id").reset_index(drop=True)
    return u


class Cell:
    """One system x form over the universe's reports (arrays in report-id order). g: that cell's rows."""

    def __init__(self, g: pd.DataFrame, s: str, f: str, u: pd.DataFrame, price: tuple):
        d = g.set_index("item_id").reindex(u.item_id)
        row = d["system"].notna().to_numpy()
        miss = ~row | _bool(d["missing"]).to_numpy()
        exc = row & ~miss & _bool(d["excluded"]).to_numpy()
        self.system, self.form, self.n_expected = s, f, len(u)
        self.A = row & ~miss & ~exc                                     # analysed answers
        self.valid = self.A & _bool(d["valid"]).to_numpy()
        self.ok = self.valid & (d["answer"].to_numpy(object) == u.truth.to_numpy(object))
        self.top = d["top_prob"].to_numpy(float)
        parts = [_parts(p) for p in (d["parts"] if "parts" in d else pd.Series([None] * len(d)))]
        self.part_failed = np.array([part_failed(p) for p in parts], bool)
        self.parse = (d["parse"] if "parse" in d else pd.Series([None] * len(d))).to_numpy(object)
        self.unstaged = self.A & (self.parse == "unstaged")
        self.unstaged_tx_nx_m1 = self.unstaged & np.array([tx_nx_m1(p) for p in parts], bool)
        # usable answer with a usable probability list (four-question forms: every question's list passed the rule)
        self.conf = self.valid & np.isfinite(self.top) & ~self.part_failed
        self.full = self.conf & (np.nan_to_num(self.top, nan=-1.0) >= 1 - TOL)
        self.tin = d["tokens_in"].fillna(0).to_numpy(float)
        self.tout = d["tokens_out"].fillna(0).to_numpy(float)
        self.treason = d["tokens_reasoning"].fillna(0).to_numpy(float)
        self.usd_rep = d["usd_reported"].to_numpy(float)
        self.usd_list_stored = d["usd_list"].to_numpy(float)
        self.price = price
        self.usd = self.usd_at(price)
        self.sets = u.set.to_numpy()
        self.ids = u.item_id.to_numpy()
        self.source = d["source"].to_numpy(object)
        self.n_rows, self.n_missing, self.n_excluded = int(row.sum()), int(miss.sum()), int(exc.sum())
        self.excluded_items = [{"item_id": i, "reason": r} for i, r, e in
                               zip(u.item_id, d["exclude_reason"].to_numpy(object), exc) if e]
        self.missing_items = [i for i, m in zip(u.item_id, miss) if m]
        self.correct_stored = d["correct"].to_numpy(object) if "correct" in d else None

    @property
    def complete(self) -> bool:
        return bool(self.A.all())

    def usd_at(self, price: tuple) -> np.ndarray:
        return np.where(self.A, (self.tin * price[0] + self.tout * price[1]) / 1e6, 0.0)


def cells(c: pd.DataFrame, u: pd.DataFrame, systems: list, P: dict) -> dict:
    by = {k: g for k, g in c.groupby(["system", "form"], sort=False)}
    empty = c.iloc[0:0]
    return {(s, f): Cell(by.get((s, f), empty), s, f, u, P[s]) for s in systems for f in FORMS}


# ------------------------------------------------------------------------------------------------ numbers
def cell_counts(x: Cell) -> dict:
    src = pd.Series(x.source[x.A]).fillna("unknown").value_counts().to_dict()
    return {"n_expected": x.n_expected, "rows": x.n_rows, "analysed": int(x.A.sum()), "complete": x.complete,
            "missing": x.n_missing, "excluded": x.n_excluded,
            "sources": {k: int(v) for k, v in sorted(src.items())},
            "unusable": int((x.A & ~x.valid).sum()), "without_probability": int((x.valid & ~x.conf).sum()),
            "part_list_failed": int((x.valid & x.part_failed).sum()),
            "unstaged": int(x.unstaged.sum()), "unstaged_tx_or_nx_with_m1": int(x.unstaged_tx_nx_m1.sum()),
            "excluded_items": x.excluded_items,
            "missing_items": x.missing_items if len(x.missing_items) < x.n_expected else "all"}


def accuracy(x: Cell) -> dict:
    out = {}
    for g, sets in GROUPS.items():
        m = np.isin(x.sets, sets)
        r = prop(int((x.ok & m).sum()), int((x.A & m).sum()))
        r.update(n_expected=int(m.sum()), complete=bool(x.A[m].all()))
        out[g] = r
    return out


def full_by_group(x: Cell) -> dict:
    """Share of analysed answers at full confidence in each report group (Table A's brackets), with its counts."""
    out = {}
    for g, sets in GROUPS.items():
        m = np.isin(x.sets, sets)
        k, n = int((x.full & m).sum()), int((x.A & m).sum())
        out[g] = {"k": k, "n": n, "share": k / n if n else None}
    return out


def _shares(r: dict) -> dict:
    """Accuracy, full-confidence share and each band's share of answers and share correct, from the counts in r."""
    n = r["n"]
    r["accuracy"] = r["correct"] / n if n else None
    r["full_confidence_share"] = r["full_confidence_n"] / n if n else None
    for b in r["bands"].values():
        b["share"] = b["answers"] / n if n else None
        b["correct_share"] = b["correct"] / b["answers"] if b["answers"] else None
    return r


def confidence(x: Cell) -> dict:
    n = int(x.A.sum())
    bands = {b: {"answers": 0, "correct": 0, "incorrect": 0} for b in BANDS}
    for p, ok in zip(x.top[x.conf], x.ok[x.conf]):
        k = bands[band(float(p))]
        k["answers"] += 1
        k["correct" if ok else "incorrect"] += 1
    return _shares({"n": n, "correct": int(x.ok.sum()), "unusable": int((x.A & ~x.valid).sum()),
                    "without_probability": int((x.valid & ~x.conf).sum()), "with_confidence": int(x.conf.sum()),
                    "full_confidence_n": int(x.full.sum()), "errors_at_full_confidence": int((x.full & ~x.ok).sum()),
                    "bands": bands, "auroc": auroc(x.top[x.conf], x.ok[x.conf]) if x.conf.any() else None})


def cost(x: Cell, sens: tuple | None = None) -> dict:
    n = int(x.A.sum())
    tot = float(x.usd.sum())
    rep = x.A & np.isfinite(x.usd_rep)
    r = {"n": n, "tokens_in": float(x.tin[x.A].sum()), "tokens_out": float(x.tout[x.A].sum()),
         "tokens_reasoning": float(x.treason[x.A].sum()), "price_in": x.price[0], "price_out": x.price[1],
         "usd_list_total": tot, "usd_per_1000": (1000 * tot / n) if n else None,
         "reported": {"n_with": int(rep.sum()), "usd_total": float(x.usd_rep[rep].sum()),
                      "usd_per_1000": (1000 * float(x.usd_rep[rep].sum()) / int(rep.sum())) if rep.any() else None}}
    if sens is not None:
        t2 = float(x.usd_at(sens).sum())
        r["sensitivity"] = {"price_in": sens[0], "price_out": sens[1], "usd_list_total": t2,
                            "usd_per_1000": (1000 * t2 / n) if n else None}
    return r


def paired(a: Cell, b: Cell, idx: np.ndarray) -> dict:
    """a minus b, both complete; else the reason it was skipped."""
    if not (a.complete and b.complete):
        return {"skipped": "incomplete cell: " + ", ".join(f"{x.system} {x.form} ({int(x.A.sum())} of {x.n_expected})"
                                                           for x in (a, b) if not x.complete)}
    r = diff_ci(a.ok, b.ok, idx)
    r.update(n=int(a.A.sum()), a={"system": a.system, "form": a.form, "correct": int(a.ok.sum())},
             b={"system": b.system, "form": b.form, "correct": int(b.ok.sum())})
    return r


def pairing(j: Cell, x: Cell, idx: np.ndarray, sens: tuple | None = None) -> dict:
    """Jev (documented) keeps its full-confidence answers, x answers the rest; against x alone."""
    if not (j.complete and x.complete):
        return {"skipped": "incomplete cell: " + ", ".join(f"{y.system} {y.form} ({int(y.A.sum())} of {y.n_expected})"
                                                           for y in (j, x) if not y.complete)}
    keep = j.full
    pair_ok = np.where(keep, j.ok, x.ok)
    n = int(j.A.sum())
    ju, xu = float(j.usd.sum()), float(x.usd.sum())
    xp = float(x.usd[~keep].sum())
    r = {"n": n, "kept": int(keep.sum()), "jev_correct_kept": int((keep & j.ok).sum()),
         "pair_correct": int(pair_ok.sum()), "alone_correct": int(x.ok.sum()), **diff_ci(pair_ok, x.ok, idx),
         "usd_jev_total": ju, "usd_llm_total": xu, "usd_llm_passed_on": xp,
         "usd_per_1000_pair": 1000 * (ju + xp) / n, "usd_per_1000_alone": 1000 * xu / n,
         "cost_share": (ju + xp) / xu if xu else None}
    if sens is not None:
        su = x.usd_at(sens)
        r["sensitivity"] = {"price_in": sens[0], "price_out": sens[1], "usd_llm_total": float(su.sum()),
                            "usd_llm_passed_on": float(su[~keep].sum()),
                            "cost_share": (ju + float(su[~keep].sum())) / float(su.sum()) if su.sum() else None}
    return r


def pooled(conf: dict, members: list, form: str) -> dict | None:
    xs = [conf[s][form] for s in members if s in conf and form in conf[s]]
    if not xs:
        return None
    r = {"members": [s for s in members if s in conf and form in conf[s]], "n": sum(x["n"] for x in xs),
         "correct": sum(x["correct"] for x in xs), "unusable": sum(x["unusable"] for x in xs),
         "without_probability": sum(x["without_probability"] for x in xs),
         "full_confidence_n": sum(x["full_confidence_n"] for x in xs),
         "errors_at_full_confidence": sum(x["errors_at_full_confidence"] for x in xs),
         "bands": {b: {k: sum(x["bands"][b][k] for x in xs) for k in ("answers", "correct", "incorrect")} for b in BANDS}}
    a = [x["auroc"] for x in xs if x["auroc"] is not None]
    r["auroc_range"] = [min(a), max(a)] if a else None
    r["full_confidence_by_member"] = {s: conf[s][form]["full_confidence_n"] for s in r["members"]}
    return _shares(r)


# ------------------------------------------------------------------------------------------------ checks
def check_staircase(C: dict, path: Path) -> dict:
    """Jev's right answers and full-confidence counts on pool 240 against pool_summary.json (6 forms)."""
    if not path.exists():
        return {"ok": None, "detail": f"{path.name} not found"}
    st = json.loads(path.read_text(encoding="utf-8"))["staircase"]["pool240"]["versions"]
    rows, ok = {}, True
    for f, v in st.items():
        x = C.get(("jev", f))
        if x is None or not x.complete:
            rows[f] = "skipped: Jev cell incomplete"
            continue
        mine = {"k": int(x.ok.sum()), "full_confidence_n": int(x.full.sum()),
                "errors_at_full_confidence": int((x.full & ~x.ok).sum()), "unusable": int((x.A & ~x.valid).sum())}
        stored = {"k": v["accuracy"]["k"], "full_confidence_n": v["full_confidence_n"],
                  "errors_at_full_confidence": v["errors_at_full_confidence"], "unusable": v["unusable"]}
        rows[f] = {"this_run": mine, "stored": stored, "match": mine == stored}
        ok &= mine == stored
    return {"ok": ok, "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "forms": rows}


def check_primary(C: dict, path: Path) -> dict:
    """Reproduce workflow_groups.json group pool240 (Jev documented at full confidence, each main LLM with names
    otherwise) from calls.parquet, with that file's rule, seed and resampling (workflow_groups.pool240)."""
    if not path.exists():
        return {"ok": None, "detail": f"{path.name} not found"}
    wg = json.loads(path.read_text(encoding="utf-8"))
    g = wg["groups"].get("pool240")
    if g is None:
        return {"ok": None, "detail": "no group pool240"}
    j = C.get(("jev", "documented"))
    if j is None or not j.complete:
        return {"ok": None, "detail": "Jev documented incomplete"}
    keep = np.nan_to_num(j.top, nan=-1.0) >= 1 - TOL                  # workflow_groups: no validity test on keep
    order = [k for k in SETS if (j.sets == k).any()]
    sizes = {k: int((j.sets == k).sum()) for k in order}
    idx = workflow_pool240_index(sizes, seed=wg["seed"], n_boot=wg["B"], order=order)
    N = len(j.ids)
    out, ok = {"kept": {"this_run": int(keep.sum()), "stored": g["kept"]}}, int(keep.sum()) == g["kept"]
    for s in MAIN:
        x = C.get((s, "names"))
        if x is None or not x.complete:
            out[s] = "skipped: names cell incomplete"
            continue
        fok = np.where(keep, j.ok, x.ok)
        bok = x.ok
        hk, ak = int(fok.sum()), int(bok.sum())
        bs = sum(fok[j.sets == k][idx[k]].sum(1) - bok[j.sets == k][idx[k]].sum(1) for k in order) / N
        mine = {"pair_correct": hk, "alone_correct": ak, "diff_pts": float(100 * (hk - ak) / N),
                "ci": [float(100 * np.quantile(bs, 0.025)), float(100 * np.quantile(bs, 0.975))]}
        shares = {"config": (float(j.usd.sum()) + float(x.usd[~keep].sum())) / float(x.usd.sum())}
        if s == "gpt":
            su = x.usd_at(GPT_SENS)
            shares["gpt_5_30"] = (float(j.usd.sum()) + float(su[~keep].sum())) / float(su.sum())
        st = g["llms"][s]
        m = (hk == st["pair_correct"] and ak == st["alone_correct"] and abs(mine["diff_pts"] - st["diff_pts"]) < 1e-9
             and all(abs(a - b) < 1e-9 for a, b in zip(mine["ci"], st["ci"])))
        basis = next((k for k, v in shares.items() if abs(v - st["cost_share"]) < 1e-9), None)
        out[s] = {"this_run": mine, "cost_share_this_run": shares, "stored": {k: st[k] for k in
                  ("pair_correct", "alone_correct", "diff_pts", "ci", "cost_share")},
                  "match_accuracy": m, "cost_share_price_basis": basis}
        ok &= m and basis is not None
    return {"ok": ok, "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "seed": wg["seed"], "n_boot": wg["B"], **out}


def check_columns(C: dict) -> dict:
    """Stored `correct` and `usd_list` against this run's values, on analysed answers."""
    bad_c, bad_u = [], []
    for (s, f), x in C.items():
        if x.correct_stored is not None:
            st = x.correct_stored
            have = x.A & pd.notna(st)
            if have.any() and not (st[have].astype(bool) == x.ok[have]).all():
                bad_c.append(f"{s} {f}")
        have = x.A & np.isfinite(x.usd_list_stored)
        if have.any() and np.abs(x.usd_list_stored[have] - x.usd[have]).max() > 1e-12:
            bad_u.append(f"{s} {f}")
    return {"ok": not bad_c, "correct_disagrees": bad_c, "usd_list_disagrees": bad_u}


# ------------------------------------------------------------------------------------------------ analysis
def analyse(c: pd.DataFrame, u: pd.DataFrame, P: dict, systems: list, meta: dict, checks: dict | None = None) -> dict:
    C = cells(c, u, systems, P)
    idx = stratified_index(u.set.to_numpy(), SEED, B)
    llms = [s for s in systems if s != "jev"]
    sens = lambda s: GPT_SENS if s == "gpt" else None
    res = {"note": "Every LLM in every form of the colon cancer staging question, 240 reports. Descriptive and paired; "
                   "stored answers only.",
           **meta, "seed": SEED, "n_boot": B, "tol_full_confidence": TOL, "bands": BANDS,
           "prices": {s: list(P[s]) for s in systems}, "price_sensitivity": {"gpt": list(GPT_SENS)},
           "systems": systems, "forms": FORMS, "sets": SETS,
           "groups": {g: int(np.isin(u.set, v).sum()) for g, v in GROUPS.items()},
           "resample_index_sha256": hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest(),
           "cells": {s: {f: cell_counts(C[(s, f)]) for f in FORMS} for s in systems}}
    res["incomplete_cells"] = [{"system": s, "form": f, "analysed": res["cells"][s][f]["analysed"],
                                "missing": res["cells"][s][f]["missing"], "excluded": res["cells"][s][f]["excluded"]}
                               for s in systems for f in FORMS if not res["cells"][s][f]["complete"]]
    res["systems_absent"] = [s for s in SYSTEMS if s not in systems or all(C[(s, f)].n_rows == 0 for f in FORMS)]
    res["accuracy"] = {s: {f: accuracy(C[(s, f)]) for f in FORMS} for s in systems}
    res["full_confidence_by_group"] = {s: {f: full_by_group(C[(s, f)]) for f in FORMS} for s in systems}
    res["confidence"] = {s: {f: confidence(C[(s, f)]) for f in FORMS} for s in systems}
    res["confidence_pooled"] = {grp: {f: pooled(res["confidence"], mem, f) for f in FORMS}
                                for grp, mem in (("main_five", MAIN), ("free_open_five", CHEAP))
                                if any(s in systems for s in mem)}
    res["cost"] = {s: {f: cost(C[(s, f)], sens(s)) for f in FORMS} for s in systems}
    res["paired"] = {
        "form_minus_names": {s: {f: paired(C[(s, f)], C[(s, "names")], idx) for f in FORMS if f != "names"} for s in llms},
        "jev_minus_llm": {f: {s: paired(C[("jev", f)], C[(s, f)], idx) for s in llms} for f in FORMS} if "jev" in systems else {}}
    for s in llms:                                   # named_field answers reused from names: flag the zero difference
        x = C[(s, "named_field")]
        ident = bool(x.A.any() and (x.source[x.A] == "identical_to_names").all())
        res["paired"]["form_minus_names"][s]["named_field"]["identical_to_names"] = ident
    res["pairing_sensitivity"] = ({s: pairing(C[("jev", "documented")], C[(s, "documented")], idx, sens(s)) for s in llms}
                                  if "jev" in systems else {})
    res["checks"] = checks(C) if checks else {"columns": check_columns(C)}
    return res


def check_prices(C) -> dict:
    """Prices against scripts/summaries/cost_frontier.py; a difference is named."""
    diff = {s: {"analysis": list(C[(s, "names")].price), "cost_frontier": list(FRONTIER_PRICE[s])}
            for s in sorted({k[0] for k in C}) if tuple(FRONTIER_PRICE[s]) != tuple(C[(s, "names")].price)}
    detail = "; ".join(f"{NAME[s]} {v['analysis'][0]:g} / {v['analysis'][1]:g} here, {v['cost_frontier'][0]:g} / "
                       f"{v['cost_frontier'][1]:g} in cost_frontier.py" for s, v in diff.items())
    if set(diff) == {"gpt"}:
        detail += " (GPT-5.6 Sol here at its list price, 5 / 30; 2 / 10 is the sensitivity)"
    return {"ok": not diff, "differs": diff, **({"detail": detail} if diff else {})}


def run_checks(pool_summary: Path, workflow: Path):
    return lambda C: {"columns": check_columns(C), "jev_staircase": check_staircase(C, pool_summary),
                      "primary_pairing": check_primary(C, workflow),
                      "prices_equal_cost_frontier": check_prices(C)}


# ------------------------------------------------------------------------------------------------ text outputs
def m(x: str) -> str:                               # minus sign; a signed zero prints as 0 (build_supplement.m)
    import re
    x = re.sub(r"^[+-](0(?:\.0+)?)$", r"\1", x)
    return x.replace("-", "−")


def pc(x, d=1):
    return "—" if x is None else m(f"{100 * x:.{d}f}")


def acc_ci(a: dict) -> str:
    if a["p"] is None:
        return "—"
    return f"{a['k']:,}/{a['n']:,}, {pc(a['p'])} ({pc(a['ci'][0])}–{pc(a['ci'][1])})"


def diff_s(r: dict) -> str:
    if "skipped" in r:
        return "—"
    e, (lo, hi) = r["est_pts"], r["ci_pts"]
    if abs(lo - hi) < 1e-12 and abs(lo - e) < 1e-12:
        return m(f"{e:+.1f}")
    return f"{m(f'{e:+.1f}')} ({m(f'{lo:+.1f}')} to {m(f'{hi:+.1f}')})"


def usd_s(x):
    if x is None:
        return "—"
    return f"{x:,.2f}" if x >= 1 else (f"{x:.3f}" if x >= 0.01 else f"{x:.4f}")


def table(head, rows, left=1):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * left + ["---:"] * (len(head) - left)) + "|"]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows])


def md(res: dict) -> str:
    S, A, K = res["systems"], res["accuracy"], res["confidence"]
    L = ["# Symmetric forms: every system in every form, colon cancer stage, pool 240", "",
         f"Stored answers only; descriptive and paired. Source: `{res['analysis_file']}`. Calls: `{res['calls']}` "
         f"(sha256 {res['calls_sha256'][:16]}...). Bootstrap: {res['n_boot']:,} paired resamples stratified by report set, "
         f"seed {res['seed']}; 95% percentile intervals.", ""]
    if res.get("stub"):
        L += ["**STUB: synthetic answers for testing; no number here is a result.**", ""]
    inc = res["incomplete_cells"]
    if res["systems_absent"] or inc:
        L += [f"**Incomplete.** Systems without answers: {', '.join(res['systems_absent']) or 'none'}. "
              f"Cells without all {res['groups']['pool240']} answers ({len(inc)}): "
              + "; ".join(f"{x['system']} {x['form']} {x['analysed']} (missing {x['missing']}, excluded {x['excluded']})"
                          for x in inc) + ". Tables show what the stored answers allow; an asterisk marks an incomplete "
              "cell (accuracy over its answers); paired numbers need complete cells.", ""]
    else:
        L += [f"All {len(S)} systems x {len(FORMS)} forms complete.", ""]
    g = res["groups"]
    cols = list(GROUPS)
    # (a) Table 5a layout
    L += ["## Table A (Supplementary Table 5a layout): every form of question, every system, on the pool and by report set", ""]
    rows = []
    for f in FORMS:
        for i, s in enumerate(S):
            cells_ = []
            for grp in cols:
                a, fc = A[s][f][grp], res["full_confidence_by_group"][s][f][grp]["share"]
                cells_.append("—" if a["p"] is None else f"{pc(a['p'])} ({pc(fc, 0)}){'*' if not a['complete'] else ''}")
            rows.append([FORM_LABEL[f] if i == 0 else "", NAME[s]] + cells_)
    L += [table(["Form of question", "System"] + [f"{GROUP_LABEL[c]} ({g[c]})" for c in cols], rows, left=2), "",
          "Exact accuracy, %; in brackets, % of answers at full confidence (a top probability of 1.00; for the four-question "
          "forms, 1.00 on every question). An unusable answer counts as wrong. Pool: the 120 reports built to mislead "
          f"(misleading feature {g['misleading_feature']}, metastatic node at an unusual site {g['nonregional_sites']}) and "
          f"120 ordinary reports drawn at random ({g['main']} of the first 1,000 and {g['heldout']} written by another "
          "model). Every system received the same content in each form; the LLMs answered in the reply format of the main "
          "runs. Table 5a's columns for all 1,320 reports are left out: the LLMs answered the pool only.", ""]
    # (b) pairing sensitivity
    ps = res["pairing_sensitivity"]
    L += ["## Table B: Jev in the recommended form at full confidence, each LLM in the recommended form otherwise", ""]
    rows = []
    for s in [x for x in S if x != "jev"]:
        r = ps.get(s, {"skipped": "no Jev"})
        if "skipped" in r:
            rows.append([NAME[s], "—", "—", "—", "—", "—"])
            continue
        rows.append([NAME[s], f"{r['alone_correct']} of {r['n']}", f"{r['pair_correct']} of {r['n']}", diff_s(r),
                     f"{usd_s(r['usd_per_1000_pair'])} / {usd_s(r['usd_per_1000_alone'])}", pc(r["cost_share"], 1)])
    kept = next((r for r in ps.values() if "skipped" not in r), None)
    L += [table(["LLM", "Correct, LLM alone", "Correct, Jev then LLM", "Difference, points (95% CI)",
                 "US$ per 1,000 reports, Jev then LLM / LLM alone", "Cost, % of the LLM alone"], rows), ""]
    note = ("Jev in the recommended form answered every report; its answer stood where it was at full confidence"
            + (f" ({kept['kept']} of {kept['n']} reports, {kept['jev_correct_kept']} of them correct)" if kept else "")
            + "; the LLM, also in the recommended form, answered the rest. Difference: the pair minus the LLM alone. Cost at "
              "the standard list price from each answer's tokens: Jev on every report and the LLM on the reports passed on.")
    g5 = ps.get("gpt", {}).get("sensitivity")
    if g5:
        note += (f" With GPT-5.6 Sol at US${g5['price_in']:g} and US${g5['price_out']:g} per million tokens, the pair cost "
                 f"{pc(g5['cost_share'])}% of GPT-5.6 Sol alone.")
    note += " The primary pairing (the LLM with level names) is unchanged; its reproduction is under Checks."
    L += [note, ""]
    # supporting tables
    L += ["## Supporting tables", "", "### Accuracy on the pool: correct/n, % (95% Wilson interval)", ""]
    L += [table(["System"] + [FORM_LABEL[f] for f in FORMS],
                [[NAME[s]] + [acc_ci(A[s][f]["pool240"]) + ("*" if not A[s][f]["pool240"]["complete"] else "")
                              for f in FORMS] for s in S]), ""]
    fm = res["paired"]["form_minus_names"]
    if fm:
        L += ["### Each LLM's change from its own level-names answers, points (95% CI)", ""]
        L += [table(["LLM"] + [FORM_LABEL[f] for f in FORMS if f != "names"],
                    [[NAME[s]] + [diff_s(fm[s][f]) + (" (same answers)" if fm[s][f].get("identical_to_names") else "")
                                  for f in FORMS if f != "names"] for s in fm]), ""]
    jm = res["paired"]["jev_minus_llm"]
    if jm:
        ll = [s for s in S if s != "jev"]
        L += ["### Jev minus each LLM in the same form, points (95% CI)", ""]
        L += [table(["LLM"] + [FORM_LABEL[f] for f in FORMS], [[NAME[s]] + [diff_s(jm[f][s]) for f in FORMS] for s in ll]), ""]
    L += ["### Cost per 1,000 answers, US$ (standard list price, each answer's tokens)", ""]
    rows = [[NAME[s]] + [usd_s(res["cost"][s][f]["usd_per_1000"]) for f in FORMS] for s in S]
    if "gpt" in S:
        rows.append([f"{NAME['gpt']} at US${GPT_SENS[0]:g} / US${GPT_SENS[1]:g}"]
                    + [usd_s(res["cost"]["gpt"][f]["sensitivity"]["usd_per_1000"]) for f in FORMS])
    L += [table(["System"] + [FORM_LABEL[f] for f in FORMS], rows), "",
          "Reported (OpenRouter) cost per 1,000 answers, where present, is in analysis.json (cost.<system>.<form>.reported).", ""]
    L += ["### Confidence: % at full confidence (errors at full confidence); unusable; without a probability list", ""]
    L += [table(["System"] + [FORM_LABEL[f] for f in FORMS],
                [[NAME[s]] + [f"{pc(K[s][f]['full_confidence_share'])} ({K[s][f]['errors_at_full_confidence']}); "
                              f"{K[s][f]['unusable']}; {K[s][f]['without_probability']}" for f in FORMS] for s in S]), "",
          "Confidence bands per system and form: symmetric_forms_table6_rows.json.", ""]
    L += ["### Answers per cell: analysed (new / reused / same as level names / Jev stored); missing; excluded", ""]
    ab = {"new": "new", "reused": "reused", "identical_to_names": "same", "jev_stored": "Jev"}
    L += [table(["System"] + [FORM_LABEL[f] for f in FORMS],
                [[NAME[s]] + [f"{c['analysed']} ({', '.join(f'{ab.get(k, k)} {v}' for k, v in c['sources'].items())}); "
                              f"{c['missing']}; {c['excluded']}" for f in FORMS for c in [res["cells"][s][f]]] for s in S]), ""]
    four = [f for f in FORMS if f not in SINGLE]
    L += ["### Four-question forms: answers the staging table does not stage (all); of them, T not assessable or nodes "
          "not stated with M1", ""]
    L += [table(["System"] + [FORM_LABEL[f] for f in four],
                [[NAME[s]] + [f"{res['cells'][s][f]['unstaged']}; {res['cells'][s][f]['unstaged_tx_or_nx_with_m1']}"
                              for f in four] for s in S]), "",
          "Such an answer is unusable and counts as wrong (the table Jev's answers were scored with is unchanged). "
          "Descriptive only.", ""]
    ex = [(s, f, e) for s in S for f in FORMS for e in res["cells"][s][f]["excluded_items"]]
    if ex:
        L += ["Excluded answers: " + "; ".join(f"{s} {f} {e['item_id']} ({e['reason']})" for s, f, e in ex) + ".", ""]
    L += ["## Checks", ""]
    for k, v in res["checks"].items():
        L.append(f"- {k}: {'passed' if v.get('ok') else ('not run' if v.get('ok') is None else 'FAILED')}"
                 + (f" ({v['detail']})" if v.get("detail") else "")
                 + (f"; stored usd_list differs in {', '.join(v['usd_list_disagrees'])}" if v.get("usd_list_disagrees") else "")
                 + (f"; stored correct differs in {', '.join(v['correct_disagrees'])}" if v.get("correct_disagrees") else ""))
    pp = res["checks"].get("primary_pairing", {})
    for s in MAIN:
        r = pp.get(s)
        if isinstance(r, dict):
            L.append(f"  - {NAME[s]}: pair {r['this_run']['pair_correct']} / alone {r['this_run']['alone_correct']}, "
                     f"stored {r['stored']['pair_correct']} / {r['stored']['alone_correct']}; accuracy and interval "
                     f"{'match' if r['match_accuracy'] else 'DIFFER'}; cost share price basis: {r['cost_share_price_basis']}")
    return "\n".join(L) + "\n"


TABLE6_TASK = "Cancer staging, 240 pathology reports"


def table6_rows(res: dict) -> dict:
    """Rows for the new Supplementary Table 6, read from analysis.json's confidence and confidence_pooled blocks."""
    def row(form, system, label, k, auc_key):
        r = {"task": TABLE6_TASK, "form": form, "form_label": FORM_LABEL[form], "system": system, "system_label": label,
             **{x: k[x] for x in ("n", "correct", "accuracy", "unusable", "without_probability", "full_confidence_n",
                                  "full_confidence_share", "errors_at_full_confidence")},
             "bands": {b: dict(k["bands"][b]) for b in BANDS}}
        r[auc_key] = k.get(auc_key)
        return r
    rows = []
    for f in FORMS:
        for s in res["systems"]:
            rows.append({**row(f, s, NAME[s], res["confidence"][s][f], "auroc"), "trace": f"confidence.{s}.{f}"})
        for grp, lab in (("main_five", "Five main LLMs, pooled"), ("free_open_five", "Five free and open-weight LLMs, pooled")):
            k = res.get("confidence_pooled", {}).get(grp, {}).get(f)
            if k:
                rows.append({**row(f, grp, lab, k, "auroc_range"), "members": k["members"],
                             "full_confidence_by_member": k["full_confidence_by_member"],
                             "trace": f"confidence_pooled.{grp}.{f}"})
    return {"note": "Supplementary Table 6 rows (cancer staging, pool 240): answers per confidence band with correct and "
                    "incorrect counts; bands <0.50, 0.50-0.74, 0.75-0.99, 1.00 (full confidence, top probability >= "
                    "1 - 1e-9; four-question forms: product of the four top probabilities). Unusable answers and answers "
                    "without a usable probability list are counted and left out of the bands. Shares are fractions "
                    "(share: of all n answers; correct_share: of the band's answers). Pooled AUROC: the range over "
                    "members with a wrong answer. Every number is read from analysis.json at `trace`.",
            "stub": bool(res.get("stub")), "source": res["analysis_file"],
            "bands": BANDS, "rows": rows}


# ------------------------------------------------------------------------------------------------ stub
def make_stub(seed: int = 1, jev_full: bool = True, drop: list | None = None,
              sizes: dict | None = None) -> pd.DataFrame:
    """Synthetic calls table with the calls table's columns (never a result). drop: [(system, form)] cells left
    out."""
    rng = np.random.default_rng(seed)
    sizes = sizes or {"main": 102, "heldout": 18, "misleading_feature": 100, "nonregional_sites": 20}
    items = [(f"stub{i:03d}", s) for i, s in enumerate(k for k, v in sizes.items() for _ in range(v))]
    truth = {i: LEVELS[rng.integers(len(LEVELS))] for i, _ in items}
    base = {"jev": 0.45, **{s: 0.95 for s in MAIN}, **{s: 0.85 for s in CHEAP}}
    lift = {"names": 0.0, "named_field": 0.01, "definitions": 0.3, "structure_only": 0.35,
            "documented_no_examples": 0.4, "documented": 0.4, "documented_notes": 0.5}
    drop = set(drop or [])
    rows, names_row = [], {}
    for s in SYSTEMS:
        for f in FORMS:
            if (s, f) in drop:
                continue
            p_ok = min(0.99, base[s] + (lift[f] if s == "jev" else lift[f] / 10))
            for iid, st in items:
                if f == "named_field" and s in MAIN[:2] and (s, iid) in names_row:   # request identical to names
                    rows.append({**names_row[(s, iid)], "form": f, "source": "identical_to_names"})
                    continue
                valid = bool(rng.random() > (0.0 if s == "jev" else 0.02))
                ok = valid and bool(rng.random() < p_ok)
                t = truth[iid]
                ans = t if ok else (LEVELS[(LEVELS.index(t) + 1 + rng.integers(len(LEVELS) - 1)) % len(LEVELS)] if valid else None)
                u = rng.random()
                top = None if not valid else (1.0 if (u < 0.3 and (ok or s != "jev") and (jev_full or s != "jev"))
                                              else float(np.round(rng.uniform(0.3, 0.99), 2)))
                if valid and s != "jev" and rng.random() < 0.01:
                    top = None                                   # a list failing the sum rule
                if not jev_full and s == "jev" and top is not None:
                    top = min(top, 0.99)
                parse, parts = ("ok" if valid else "unparsed"), None
                if f not in SINGLE:
                    parts = {q: {"choice": c_, "prob_status": "ok", "top": 1.0} for q, c_ in
                             (("t_category", "T3"), ("regional_nodes", "none"), ("tumor_deposits", "absent"),
                              ("distant_metastasis", "none"))}
                    parse = "parts" if valid else ("unstaged" if rng.random() < 0.5 else "unparsed")
                    if parse == "unstaged" and rng.random() < 0.5:
                        parts["t_category"]["choice"], parts["distant_metastasis"]["choice"] = "cannot be assessed", "M1a"
                    elif valid and s != "jev" and top is not None and rng.random() < 0.02:
                        parts["distant_metastasis"]["prob_status"] = "sum_out_of_range"   # a question's list failed
                tin = int(rng.integers(400, 2500))
                tout = 1 if s == "jev" else int(rng.integers(40, 900))
                rows.append({"system": s, "form": f, "set": st, "item_id": iid, "truth": t, "answer": ans, "valid": valid,
                             "probs": json.dumps({ans: top}) if valid and top is not None else None,
                             "prob_status": "ok" if top is not None else ("sum_off" if valid else None),
                             "top_prob": top, "parse": parse, "parts": json.dumps(parts) if parts else None,
                             "model_reported": "stub", "provider": "stub", "route": "standard",
                             "source": "jev_stored" if s == "jev" else ("reused" if f == "names" and s in MAIN else "new"),
                             "tokens_in": float(tin), "tokens_out": float(tout),
                             "tokens_reasoning": float(0 if s == "jev" else tout // 2),
                             "usd_reported": float(np.round(tin * 1e-6, 8)) if rng.random() < 0.5 else None,
                             "usd_list": None, "latency_s": float(rng.uniform(0.2, 9.0)), "finish_reason": "stop",
                             "raw_path": f"runs/stub/{s}_{f}_{iid}.json", "error": None if valid else "stub unparsed",
                             "excluded": False, "exclude_reason": None, "missing": False})
                if f == "names":
                    names_row[(s, iid)] = rows[-1]
    df = pd.DataFrame(rows)
    df["correct"] = df.valid & (df.answer == df.truth)
    df["full_confidence"] = df.valid & (df.top_prob.fillna(-1) >= 1 - TOL)
    df["confidence_band"] = [band(p) if v and pd.notna(p) else None for v, p in zip(df.valid, df.top_prob)]
    return df[COLUMNS]


# ------------------------------------------------------------------------------------------------ main
def rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)).replace("\\", "/") if p.is_relative_to(ROOT) else str(p)


def dump(obj, path: Path) -> None:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=conv), encoding="utf-8")


def run(calls: Path, out_dir: Path, summaries_dir: Path, systems: list | None = None, pool: Path | None = POOL,
        config: Path = CONFIG, stub: bool = False, pool_summary: Path | None = None, workflow: Path | None = None) -> dict:
    c = load_calls(calls)
    u = universe(c, None if stub else pool)
    P = prices(config)
    partial = systems is not None
    present = [s for s in SYSTEMS if s in set(c.system)]
    sys_ = ["jev"] + [s for s in SYSTEMS if s != "jev" and s in (systems or SYSTEMS)] if partial else SYSTEMS
    meta = {"stub": stub, "calls": rel(calls), "calls_sha256": hashlib.sha256(Path(calls).read_bytes()).hexdigest(),
            "config": rel(config), "pool": None if (stub or pool is None) else rel(pool), "systems_in_calls": present,
            "analysis_file": rel(out_dir / "analysis.json")}
    chk = None if stub else run_checks(pool_summary or summaries_dir / "pool_summary.json",
                                       workflow or summaries_dir / "workflow_groups.json")
    res = analyse(c, u, P, sys_, meta, chk)
    if partial:
        for s in [x for x in sys_ if x in (systems or [])]:
            keys = ("cells", "accuracy", "confidence", "cost", "full_confidence_by_group")
            part = {k: res[k] for k in ("note", "stub", "calls", "calls_sha256", "seed", "n_boot", "tol_full_confidence",
                                        "bands", "price_sensitivity", "sets", "groups", "resample_index_sha256")}
            part.update(system=s, prices={x: res["prices"][x] for x in {"jev", s}},
                        **{k: {x: res[k][x] for x in {"jev", s}} for k in keys})
            if s != "jev":
                part["paired"] = {"form_minus_names": res["paired"]["form_minus_names"][s],
                                  "jev_minus_llm": {f: res["paired"]["jev_minus_llm"][f][s] for f in FORMS}}
                part["pairing_sensitivity"] = res["pairing_sensitivity"][s]
            part["complete"] = all(res["cells"][s][f]["complete"] for f in FORMS)
            dump(part, out_dir / "partial" / f"{s}.json")
        return res
    dump(res, out_dir / "analysis.json")
    (summaries_dir / "symmetric_forms_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (summaries_dir / "symmetric_forms_summary.md").write_text(md(res), encoding="utf-8")
    dump(table6_rows(res), summaries_dir / "symmetric_forms_table6_rows.json")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calls", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--summaries-dir", default=None, help="summary md and Table 6 rows (default results/summaries)")
    ap.add_argument("--systems", nargs="+", default=None, help="score these systems only -> <out-dir>/partial/<s>.json")
    ap.add_argument("--pool", default=str(POOL), help="pool file; 'none' takes the reports from the calls table")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stub", action="store_true", help="write a synthetic calls table and analyse it (tests only)")
    a = ap.parse_args(argv)
    if a.systems:
        assert set(a.systems) <= set(SYSTEMS), sorted(set(a.systems) - set(SYSTEMS))
    if a.stub:                            # synthetic numbers stay out of results/ unless a folder is named
        import tempfile
        out = Path(a.out_dir or Path(tempfile.gettempdir()) / "jevity_llm_forms_stub")
        calls = Path(a.calls or out / "calls_STUB.parquet")
        assert calls.resolve() != (OUT / "calls.parquet").resolve(), "the stub never replaces the real calls table"
        calls.parent.mkdir(parents=True, exist_ok=True)
        make_stub().to_parquet(calls, index=False)
        wd = Path(a.summaries_dir or out)
    else:
        out, calls, wd = Path(a.out_dir or OUT), Path(a.calls or OUT / "calls.parquet"), Path(a.summaries_dir or SUMMARIES)
    pool = None if a.pool.lower() in ("", "none") else Path(a.pool)
    res = run(calls, out, wd, a.systems, pool, Path(a.config), a.stub)
    print(f"systems {len(res['systems'])}, incomplete cells {len(res['incomplete_cells'])}, "
          f"absent {res['systems_absent'] or 'none'}")
    for k, v in res["checks"].items():
        print(f"check {k}: {'passed' if v.get('ok') else ('not run' if v.get('ok') is None else 'FAILED')}")
    print("wrote", rel(out / ("partial" if a.systems else "analysis.json")))


if __name__ == "__main__":
    main()

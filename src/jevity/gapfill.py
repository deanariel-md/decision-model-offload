"""Jev on the eICU demo records in the recommended form, and the outcome Choice questions: Jev only, three items.
  1. eICU death before hospital discharge in the recommended form: the eICU record (eicu.to_state, the open demo only)
     with every numeric field replaced by the name of its published category, as docuse_risk does for NHANES
     (config/gapfill_eicu.yaml; fields shared with NHANES reuse config/docuse_risk.yaml's source and cut points).
  2. An eICU outcome Choice (died in the unit; died in hospital after the unit; discharged alive), the NHANES risk-band
     question's template with its time reference adapted, on the records as given (drop-in) and the categorised
     records.
  3. The NHANES risk-band question, unchanged, on the recommended-form records (docuse_risk.build_states).
Every request goes through clients.JevClient (typesafe/jev-1.13 on OpenRouter, standard route, allow_fallbacks false);
live sends pass a SpendGuard (with an optional spending stop). eICU records are checked against the demo's own
patient table before any call."""
from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import docuse_risk as DR
from . import eicu as E
from .clients import MODELS, PROMPTS, ROOT, JevClient, RawStore, _real01

CONFIG_PATH = ROOT / "config" / "gapfill_eicu.yaml"
CFG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else None
EICU_POP = "eicu_demo"
TOKENS_PER_CHAR = 0.46          # Jev bills about 0.44 tokens per character of the posted body (measured); projection


# ------------------------------------------------------------------------------------------------ demo only
def demo_stay_ids(root: Path = ROOT) -> set[int]:
    """Every stay id in the open demo's own patient table."""
    pt = E.read_table(root / E.CFG["sources"]["demo"], "patient", [E.ID])
    return set(pt[E.ID].astype(int))


def eval_cohort(root: Path = ROOT) -> pd.DataFrame:
    """The 1,297 evaluation stays of data/eicu/demo_cohort.parquet, refused unless every row is a demo stay."""
    co = pd.read_parquet(root / "data" / "eicu" / "demo_cohort.parquet")
    if set(co["source"]) != {"demo"}:
        raise SystemExit(f"demo_cohort.parquet holds a source other than the demo: {sorted(set(co['source']))}")
    ev = co[co["split"] == "eval"].copy()
    bad = set(ev[E.ID].astype(int)) - demo_stay_ids(root)
    if bad:
        raise SystemExit(f"{len(bad)} stays are not in the demo's patient table; nothing is sent")
    return ev.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ item 1: categories
def _value(rec: dict, col: str, nd: int) -> float | None:
    """A value exactly as the original record shows it (eicu._num: the same rounding); None when missing."""
    return E._num(rec.get(col), nd)


def _decimals(col: str) -> int:
    return next(nd for _, c, _, nd in E.CFG["numeric"] if c == col)


def category(name: str, rec: dict, cfg: dict | None = None) -> str | None:
    """The category label of one configured field for one cohort row, or None when an input is missing."""
    cfg = cfg or CFG
    f = cfg["fields"][name]
    kind = f.get("kind")
    cols = f["from_columns"]
    vals = [_value(rec, c, _decimals(c)) for c in cols]
    if any(v is None for v in vals):
        return None
    if kind == "egfr":                                      # docuse_risk's equation and KDIGO bands, unchanged
        nh = DR.CFG["fields"]["kidney_function"]
        sex = rec.get("sex")
        if sex not in nh["equation"]:
            return None
        age = float(rec["age"])                             # eICU top code "> 89" is 90 in the cohort (eicu.parse_age)
        return DR.band(DR.egfr_ckd_epi_2021(vals[0], age, sex, nh["equation"]), nh["bands"])
    if kind == "pf_ratio":                                  # PaO2 / FiO2, FiO2 recorded in per cent
        pao2, fio2 = vals
        if fio2 <= 0:
            return None
        v = pao2 / (fio2 / 100.0)
        support = rec.get(f["support_column"]) == "yes"
        for b in f["bands"]:
            if b.get("needs_support") and not support:
                continue
            if "below" in b and v < float(b["below"]):
                return b["label"]
            if "below" not in b and "at_most" not in b:
                return b["label"]
        raise ValueError("pf_ratio bands need a last band without a cut point")
    v = vals[0]
    if f.get("from_nhanes"):                                # the NHANES field's bands, unchanged
        nh = DR.CFG["fields"][f["from_nhanes"]]
        assert not nh.get("scale") and not nh.get("by_sex"), f["from_nhanes"]
        bands = nh["bands"]
    else:
        bands = f["bands"]
    if f.get("by_sex"):
        if rec.get("sex") not in bands:
            return None
        bands = bands[rec["sex"]]
    return DR.band(v, bands, f.get("scale"), f.get("decimals"))


def categorised_state(rec: dict, cfg: dict | None = None) -> "OrderedDict[str, Any]":
    """The recommended-form eICU state: eicu.to_state's record with every numeric field replaced, at its place, by its
    category (fields with two inputs at the place of the first); kept fields unchanged; a missing value stays missing."""
    cfg = cfg or CFG
    orig = E.to_state(rec)
    kept = set(cfg["state"]["kept_fields"])
    source_of = {k: name for name, f in cfg["fields"].items() for k in f["from"]}
    out: OrderedDict[str, Any] = OrderedDict()
    done: set[str] = set()
    for k, v in orig.items():
        if k in kept:
            out[k] = v
        elif k in source_of:
            name = source_of[k]
            if name in done:
                continue
            done.add(name)
            lab = category(name, rec, cfg)
            if lab is not None:
                out[cfg["fields"][name]["key"]] = lab
        else:
            raise KeyError(f"original field {k!r} is neither kept nor categorised (config/gapfill_eicu.yaml)")
    return out


# ------------------------------------------------------------------------------------------------ item 2: eICU outcome
def outcome_of(row: pd.Series) -> str:
    """The option that happened, from the demo's unit and hospital discharge status."""
    unit, hosp = str(row["unitdischargestatus"]).strip(), str(row["hospitaldischargestatus"]).strip()
    o = CFG["band_question"]["outcomes"]
    if unit == "Expired":
        return o["unit_death"]
    if hosp == "Expired":
        return o["hospital_death_after_unit"]
    if hosp == "Alive":
        return o["alive"]
    raise ValueError((unit, hosp))


class EicuBandClient(JevClient):
    """Jev's eICU outcome Choice: the NHANES risk-band request with the eICU question in place of the NHANES one."""

    def __init__(self, store: RawStore, repeat: int = 0):
        super().__init__(store, EICU_POP, repeat=repeat, question="risk_bands")

    def _questions(self, statement: str) -> dict:
        q = CFG["band_question"]
        return {q["question_id"]: {"type": "choice", "instructions": q["instructions"], "criteria": dict(q["criteria"])}}

    def _result(self, req: dict, cached: dict) -> dict:
        resp = cached["response"]
        q = CFG["band_question"]
        p, parse, extra = None, "missing_answer", {}
        try:
            a = resp["answers"][q["question_id"]]
            probs = a.get("probabilities") or {}
            extra = {"band_probabilities": probs, "confidence": a.get("confidence")}
            if not set(probs) <= set(q["criteria"]):
                parse = "bands_labels"
            else:
                vals = {k: _real01(probs.get(k, 0.0)) for k in q["criteria"]}
                if any(v is None for v in vals.values()) or not 0.99 <= sum(vals.values()) <= 1.01:
                    parse = "bands_mass"
                else:
                    p, parse = min(1.0, sum(vals[k] for k in q["death_options"])), "bands"
        except Exception:
            pass
        return {"p": p, "valid": p is not None, "provider": resp.get("provider"), "model_reported": resp.get("model"),
                "usage": resp.get("usage"), "raw_path": str(self.store.path(req)), "parse": parse,
                "route": (cached.get("meta") or {}).get("route", "standard"), "response_id": resp.get("id"),
                "latency_s": (cached.get("meta") or {}).get("latency_s"), **extra}


# ------------------------------------------------------------------------------------------------ spending
class SpendGuard:
    """Refuses a live request once spent + held + this request's projection would pass the cap (None: no cap).
    spent starts from every stored reply of this work (usage.cost as reported)."""

    def __init__(self, cap: float | None = None, spent: float = 0.0):
        self.cap = None if cap is None else float(cap)
        self.spent, self.held, self.lock = float(spent), 0.0, threading.Lock()

    def reserve(self, est: float) -> None:
        with self.lock:
            if self.cap is not None and self.spent + self.held + est > self.cap:
                raise RuntimeError(f"spend cap: spent ${self.spent:.4f} + held ${self.held:.4f} + ${est:.6f} > ${self.cap:.2f}")
            self.held += est

    def settle(self, est: float, actual: float) -> None:
        with self.lock:
            self.held -= est
            self.spent += actual


def body(req: dict) -> dict:
    return {k: v for k, v in req.items() if not k.startswith("_")}


def projected_usd(req: dict) -> float:
    n = len(json.dumps(body(req), ensure_ascii=False))
    return n * TOKENS_PER_CHAR * float(MODELS["jev"]["price_per_mtok_input"]) / 1e6


def stored_spend(store: RawStore) -> float:
    tot = 0.0
    if not store.run_dir.exists():
        return 0.0
    for f in store.run_dir.glob("*.json"):
        try:
            u = (json.loads(f.read_text(encoding="utf-8")).get("response") or {}).get("usage") or {}
            tot += float(u.get("cost") or 0.0)
        except Exception:
            continue
    return tot


def guarded(cls):
    """A client class whose live sends pass the guard (stored replies are read without it)."""

    class Guarded(cls):
        guard: SpendGuard | None = None

        def _send(self, req: dict) -> dict:
            if self.guard is None:
                raise RuntimeError("live calls need a spend guard (scripts/gapfill/run.py)")
            est = projected_usd(req)
            self.guard.reserve(est)
            actual = est
            try:
                out = super()._send(req)
                u = (out.get("response") or {}).get("usage") or {}
                actual = float(u.get("cost") if u.get("cost") is not None else est)
                return out
            finally:
                self.guard.settle(est, actual)

    Guarded.__name__ = f"Guarded{cls.__name__}"
    return Guarded


def statement(population: str) -> str:
    return PROMPTS["statements"][population]


def nan_to_none(x):
    return None if isinstance(x, float) and math.isnan(x) else x

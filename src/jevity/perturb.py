"""Apply one-thing-at-a-time edits to a cohort record and log the diff. The support rule decides whether an
edited record stays inside the fitting sample's covariate support."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors

from . import serialize as S

ROOT = Path(__file__).resolve().parents[2]
PERT = yaml.safe_load((ROOT / "config" / "perturbations.yaml").read_text(encoding="utf-8"))


@dataclass
class Edit:
    name: str
    klass: str                       # irrelevant | clinical | label
    record: dict[str, Any]           # edited record (irrelevant edits: identical to baseline)
    diff: dict[str, Any] = field(default_factory=dict)
    applicable: bool = True
    reason: str = ""


def _get(rec: dict, col: str):
    v = rec.get(col)
    return None if (isinstance(v, float) and math.isnan(v)) else v


def apply_record_edit(rec: dict[str, Any], name: str) -> Edit:
    klass = "clinical" if name in PERT["clinical"] else "label"
    spec = PERT[klass][name]
    col = spec["column"]
    base = _get(rec, col)
    if base is None:
        return Edit(name, klass, dict(rec), applicable=False, reason=f"{col} missing")
    for k, v in spec.get("require", {}).items():
        if _get(rec, k) != v:
            return Edit(name, klass, dict(rec), applicable=False, reason=f"requires {k}={v}")
    for k, v in spec.get("require_max", {}).items():
        if _get(rec, k) is None or _get(rec, k) > v:
            return Edit(name, klass, dict(rec), applicable=False, reason=f"requires {k}<={v}")
    op, val = spec["op"], spec["value"]
    new = {"add": lambda: base + val, "multiply": lambda: base * val, "set": lambda: val}[op]()
    out = dict(rec)
    out[col] = new
    return Edit(name, klass, out, diff={"column": col, "from": base, "to": new, "op": op})


def render(rec: dict[str, Any], irrelevant: str | None = None, seed: int = 0, annotated: bool = False,
           drop: list[str] | None = None) -> str:
    """Serialise a record; `irrelevant` names an I-class rendering; annotated adds category labels; `drop` removes
    fields (M2 mitigation: race_ethnicity)."""
    state = S.to_state(rec)
    if drop:
        state = S.drop_fields(state, drop)
    if annotated:
        state = S.annotate(state)
    if irrelevant is None:
        return S.to_json(state)
    op = PERT["irrelevant"][irrelevant]["op"]
    if op == "shuffle_fields":
        return S.to_json(S.shuffle_fields(state, seed))
    if op == "convert_units":
        return S.to_json(S.convert_units(state))
    if op == "to_prose":
        return S.to_prose(state)
    if op == "padding":
        return S.to_json(state) + "\n\n" + PADDING_TEXT
    raise ValueError(op)


PADDING_TEXT = (
    "Interview administration notes: the household interview was conducted in English in the respondent's "
    "home; the examination took place in a mobile examination centre; consent forms were completed; the "
    "respondent was reminded to bring medication containers; the dietary recall used the automated "
    "multiple-pass method; the session lasted approximately three hours; transport was provided."
)


class SupportChecker:
    """Nearest-neighbour support rule on standardised numeric covariates (log-transformed where skewed)."""

    def __init__(self, fit_df: pd.DataFrame, quantile: float | None = None, exclude_strata: tuple = (),
                 cfg: dict | None = None):
        cfg = cfg or PERT["support"]           # eICU passes config/eicu.yaml `support`; NHANES uses perturbations.yaml
        self.cols = [c for c in cfg["numeric_columns"] if c in fit_df.columns]
        self.logs = set(cfg["log_columns"])
        X = self._matrix(fit_df)
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0) + 1e-9
        Z = self._standardise(X)
        self.nn = NearestNeighbors(n_neighbors=2).fit(Z)
        d, _ = self.nn.kneighbors(Z)          # column 0 is self
        q = quantile or cfg["quantile"]
        self.threshold = float(np.quantile(d[:, 1], q))
        # positivity for categorical edits: a multinomial model of the category on the numeric covariates; the edited
        # record is supported if its probability of the new category is at least the 2.5th percentile of that
        # probability among fitting records that hold it (it resembles some real members of the group). Nearest-
        # neighbour distance in ~20 dimensions is insensitive to a single strongly shifted covariate.
        from sklearn.linear_model import LogisticRegression
        self.strata: dict[str, tuple[Any, dict[Any, float]]] = {}
        self.eligible: dict[str, set] = {}       # levels with enough fitting records, kept even when no model is fitted
        self.exclude_strata = tuple(exclude_strata)
        pq = float(cfg.get("propensity_quantile", 0.025))
        for c in [c for c in cfg.get("strata_columns", []) if c in fit_df.columns]:
            y = fit_df[c].astype(object)
            ok = y.notna().to_numpy()
            counts = y[ok].value_counts()
            levels = counts[counts >= int(cfg.get("min_stratum", 50))].index
            self.eligible[c] = set(levels)
            if len(levels) < 2:
                continue
            m = LogisticRegression(max_iter=2000).fit(Z[ok], y[ok].to_numpy())
            P_ = m.predict_proba(Z[ok])
            thr = {lev: float(np.quantile(P_[y[ok].to_numpy() == lev, list(m.classes_).index(lev)], pq)) for lev in levels}
            self.strata[c] = (m, thr)
        lo, hi = cfg.get("range_quantiles", [0.0, 1.0])
        self.ranges = {c: (float(fit_df[c].quantile(lo)), float(fit_df[c].quantile(hi)))
                       for c in self.cols if pd.to_numeric(fit_df[c], errors="coerce").notna().any()}

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.cols].astype(float).to_numpy()
        for j, c in enumerate(self.cols):
            if c in self.logs:
                X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
        return X

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0)

    def distance(self, rec: dict[str, Any]) -> float:
        row = pd.DataFrame([rec])
        for c in self.cols:
            if c not in row.columns:
                row[c] = np.nan
        Z = self._standardise(self._matrix(row))
        d, _ = self.nn.kneighbors(Z, n_neighbors=1)
        return float(d[0, 0])

    def category_rule(self, rec: dict[str, Any], changed: str | None = None) -> bool:
        """Rule (b), category positivity, alone: the new category has enough fitting records and, where a propensity
        model exists, the edited record's probability of it reaches the configured quantile. True for other edits."""
        if changed not in self.eligible:
            return True
        lev = rec.get(changed)
        if lev not in self.eligible[changed]:
            return False                         # a rare or unseen target is never supported, model or not
        if changed not in self.strata:
            return True
        m, thr = self.strata[changed]
        row = pd.DataFrame([rec])
        for c in self.cols:
            if c not in row.columns:
                row[c] = np.nan
        p = m.predict_proba(self._standardise(self._matrix(row)))[0, list(m.classes_).index(lev)]
        return bool(p >= thr[lev])

    def supported(self, rec: dict[str, Any], changed: str | None = None) -> bool:
        """changed: the edited column. Categorical edits need neighbours in the new category (rule b, unless
        exclude_strata names the column); numeric edits must stay inside the fitting range; every edited record must
        also pass the joint rule."""
        if changed in self.ranges:
            v = pd.to_numeric(pd.Series([rec.get(changed)]), errors="coerce").iloc[0]
            lo, hi = self.ranges[changed]
            if not (pd.notna(v) and lo <= v <= hi):
                return False
        if changed in self.eligible and changed not in self.exclude_strata and not self.category_rule(rec, changed):
            return False
        return self.distance(rec) <= self.threshold

"""Arm 3 metrics on the 10 ordered levels (0 to IIIC, IVA-IVB and IVC): exact accuracy, quadratic weighted kappa,
off-by-one rate and, for Jev, the mean absolute error of its expected level; and the secondary measure main-stage
accuracy (answers collapsed to 0, I, II, III, IV).

Pure functions of truth labels, answered labels and probability maps. An unusable answer (None or a label outside the
scale) counts as wrong in accuracy and is left out of kappa, which needs a level. The paired bootstrap, Holm correction,
proper scores and confidence analyses are shared with arm 2 (jevity.categorical_analysis)."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from jevity.crc import LEVELS, MAIN_STAGES, main_stage

RENORMALISE = (0.9, 1.1)      # a probability list summing inside this range is rescaled; outside it, left out


def level_index(label, levels: Sequence[str] = LEVELS) -> int | None:
    """Position of a label on the ordered scale; None for a missing or unknown label."""
    try:
        return list(levels).index(str(label)) if label is not None else None
    except ValueError:
        return None


def _indices(truth, pred, levels) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([level_index(x, levels) for x in truth], dtype=float)
    if np.isnan(t).any():
        raise ValueError("every truth label must be on the scale")
    p = np.array([np.nan if (i := level_index(x, levels)) is None else i for x in pred], dtype=float)
    if len(t) != len(p):
        raise ValueError("truth and answers differ in length")
    return t, p


def exact_accuracy(truth, pred, levels: Sequence[str] = LEVELS) -> float:
    t, p = _indices(truth, pred, levels)
    return float(np.mean(p == t))


def off_by_one_rate(truth, pred, levels: Sequence[str] = LEVELS) -> float:
    """Share of all items answered exactly one level from the truth (e.g. IIIB for IIIA)."""
    t, p = _indices(truth, pred, levels)
    return float(np.mean(np.abs(p - t) == 1))


def within_one_rate(truth, pred, levels: Sequence[str] = LEVELS) -> float:
    """Share of all items answered at most one level from the truth."""
    t, p = _indices(truth, pred, levels)
    return float(np.mean(np.abs(p - t) <= 1))


def quadratic_weighted_kappa(truth, pred, levels: Sequence[str] = LEVELS) -> float:
    """Cohen's kappa with quadratic weights over the whole scale, on items with a usable answer. NaN if fewer than two
    such items or if chance disagreement is zero."""
    t, p = _indices(truth, pred, levels)
    ok = ~np.isnan(p)
    t, p = t[ok].astype(int), p[ok].astype(int)
    k = len(levels)
    if len(t) < 2:
        return float("nan")
    obs = np.zeros((k, k))
    np.add.at(obs, (t, p), 1)
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    i, j = np.indices((k, k))
    w = (i - j) ** 2 / (k - 1) ** 2
    denom = (w * exp).sum()
    return float("nan") if denom == 0 else float(1 - (w * obs).sum() / denom)


def accuracy_by_level(truth, pred, levels: Sequence[str] = LEVELS) -> dict[str, float]:
    t, p = _indices(truth, pred, levels)
    return {g: float(np.mean(p[t == i] == i)) for i, g in enumerate(levels) if (t == i).any()}


def main_stage_accuracy(truth, pred) -> float:
    """Secondary measure: accuracy after collapsing the answers and truth to the main stage (0, I, II, III, IV), so IIB
    for IIA and IVC for IVA-IVB count as right. From the same answers; no extra calls. An unusable answer counts as
    wrong."""
    t, p = _indices(truth, pred, LEVELS)
    main = np.array([MAIN_STAGES.index(main_stage(lv)) for lv in LEVELS])
    ok = ~np.isnan(p)
    return float(np.mean(ok & (main[np.where(ok, p, 0).astype(int)] == main[t.astype(int)])))


def renormalised(probs: Mapping | None, levels: Sequence[str] = LEVELS) -> dict[str, float] | None:
    """A probability map on the scale, rescaled to sum to one when its total is 0.9-1.1; None when a label is off the
    scale, a value is not a finite number in [0, 1], or the total is outside that range. An omitted level has zero
    mass."""
    if not isinstance(probs, Mapping) or not probs or not set(map(str, probs)) <= set(levels):
        return None
    vals = {}
    for g in levels:
        v = probs.get(g, 0.0)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            return None
        vals[g] = float(v)
    total = sum(vals.values())
    if not RENORMALISE[0] <= total <= RENORMALISE[1]:
        return None
    return {g: v / total for g, v in vals.items()}


def expected_level(probs: Mapping | None, levels: Sequence[str] = LEVELS) -> float | None:
    """Mean level under a probability map, on the level-index scale (0 = stage 0, 9 = IVC)."""
    q = renormalised(probs, levels)
    return None if q is None else float(sum(i * q[g] for i, g in enumerate(levels)))


def expected_level_mae(truth, probs: Sequence | None = None, levels: Sequence[str] = LEVELS,
                       expected: Sequence[float | None] | None = None) -> dict:
    """Mean absolute error, in levels, of the expected level (Jev's Score answer). Pass `probs` (one map per item) or
    `expected` (Jev's own continuous score, already on the level-index scale). Items without a usable value are left
    out and counted."""
    if (probs is None) == (expected is None):
        raise ValueError("pass exactly one of probs and expected")
    ev = [expected_level(p, levels) for p in probs] if probs is not None else list(expected)
    t = [level_index(x, levels) for x in truth]
    if len(t) != len(ev) or any(i is None for i in t):
        raise ValueError("truth must be on the scale and match the answers in length")
    err = [abs(e - i) for e, i in zip(ev, t) if e is not None and math.isfinite(e)]
    return {"mae": float(np.mean(err)) if err else float("nan"), "n": len(err), "n_excluded": len(t) - len(err)}


def summarise(truth, pred, probs: Sequence | None = None, levels: Sequence[str] = LEVELS) -> dict:
    """All arm 3 point estimates for one system and question version."""
    _, p = _indices(truth, pred, levels)
    out = {"n": len(p), "n_unusable": int(np.isnan(p).sum()),
           "exact_accuracy": exact_accuracy(truth, pred, levels),
           "off_by_one_rate": off_by_one_rate(truth, pred, levels),
           "within_one_rate": within_one_rate(truth, pred, levels),
           "quadratic_weighted_kappa": quadratic_weighted_kappa(truth, pred, levels)}
    if list(levels) == LEVELS:
        out["main_stage_accuracy"] = main_stage_accuracy(truth, pred)
    if probs is not None:
        mae = expected_level_mae(truth, probs, levels)
        out.update(expected_level_mae=mae["mae"], expected_level_n=mae["n"])
    return out


def alias_log(rows, by: Sequence[str] = ("system", "variant")) -> list[dict]:
    """Reply rule, the same for every system: an answer of IVA or IVB counts as IVA-IVB. How often it applied, from
    the result rows the shared code stores: answers taken through an alias (answer_how "alias") and probability lists
    with an alias key (prob_alias), per system and question version."""
    df = pd.DataFrame(rows)
    ans = df["answer_how"].eq("alias") if "answer_how" in df else pd.Series(False, index=df.index)
    prob = df["prob_alias"].map(lambda v: v is True) if "prob_alias" in df else pd.Series(False, index=df.index)
    g = df.assign(_a=ans, _p=prob).groupby(list(by), sort=True)
    return [{**dict(zip(by, k if isinstance(k, tuple) else (k,))), "n": int(len(sub)), "alias_answers": int(sub._a.sum()),
             "alias_probability_lists": int(sub._p.sum())} for k, sub in g]

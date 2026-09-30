"""Analysis, arms 2 and 3.

Per system, main pass (repeat 0), every item: the arm's accuracy (arm 2 balanced accuracy, arm 3 exact accuracy) with
arm-specific rates; an unusable answer (no parseable choice, or no collected response) counts as not correct and adds
the cost of any billed tokens; the unusable rate is reported per system. Comparisons: Jev minus each chatbot, paired
bootstrap over items stratified by the truth stratum (scenario type in arm 2; 2,000 resamples), 95% percentile CI and
two-sided bootstrap p (Holm across the primary chatbots). Non-inferiority (margin 5 points): the Holm-adjusted
one-sided lower confidence bound at alpha 0.025 across the five primary chatbots (holm_lower_bounds); free and pair
systems descriptive, with the unadjusted one-sided bound. Outcome per comparison: non-inferior, worse (the mirror Holm
procedure: upper bound below -margin) or inconclusive. Hybrid at a set share (arm 2): Jev answers the half of the items
with its highest top probability, the chatbot the rest; hybrid minus chatbot, with the same Holm procedures, outcomes
and list-price cost (hybrid_share).

Confidence, from each system's own probability list, defined identically for every system: the top-choice
(maximum) probability. Proper score (multiclass Brier, arm 2; ranked probability score, arm 3); routing accuracy on the
most confident 50% and 80% of items and the area under the risk-coverage curve (ties in confidence: expected value under
random order); expected calibration error in 10 equal-count bins with the reliability table; AUROC of confidence for
right against wrong answers, only with at least 20 errors. Only answers with a usable choice and a probability list that
passed the sum rule enter; Jev's own confidence field is reported alongside.

Cost and value: cost per answer billed (batch price on the batch route) and at list price, from reported tokens (Jev:
input only, output free); per answer the input, output and reasoning tokens, billed and list cost per 1,000 answers and
the list cost of one million answers; list cost per correct answer with a bootstrap CI; for each chatbot the extra cost
per extra correct answer against Jev, only where the chatbot has more correct answers (otherwise Jev cheaper and at
least as accurate); the Jev-first hybrid at Jev's confidence deciles (no extra calls) and, for the cost table, one
hybrid per chatbot with its list cost and saving against the chatbot alone; and the acceptable-cheaper verdict: the
Holm-adjusted lower bound of (Jev minus chatbot) above -5 points and Jev's list-price cost per correct answer lower;
billed costs reported alongside. Repeatability on the repeat set: share answered identically in all repeats, and
Fleiss' kappa (arm 2) or ICC(2,1) of the level index (arm 3), bootstrap CIs. Random-half control (the definitions of
arm 3's supplement): the hybrid's share (else 50%) handed to Jev at random, the exact expectation over every set of
that size; confidence-routed minus random-routed per chatbot, paired over the resampled items (routing); the random
line and list cost per 1,000 answers at every point of the decile curve (hybrid) and of the share curve, 10-90%
(coverage_curve). supplement_tables turns one result into the supplement tables.

Style check (arm 2; descriptive): word and negation counts of each message, a style-only logistic baseline (leave one
recommendation out) as a reference row, and each system's balanced accuracy on the items that baseline gets right and
wrong. All functions are pure."""
from __future__ import annotations

import datetime
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

BATCH_MISSING = "batch result not collected"


# ------------------------------------------------------------------------------------------------ small metrics
def accuracy_metric(correct: np.ndarray, strata: np.ndarray, name: str) -> float:
    """balanced_accuracy: unweighted mean of the per-stratum accuracies; exact_accuracy / accuracy: overall share."""
    correct = np.asarray(correct, float)
    if name == "balanced_accuracy":
        return float(np.mean([correct[strata == s].mean() for s in np.unique(strata)]))
    return float(correct.mean())


def metric_boot(C: np.ndarray, strata: np.ndarray, name: str) -> np.ndarray:
    """accuracy_metric for every row of a (B, n) matrix of resampled correctness (columns keep their stratum)."""
    C = np.asarray(C, float)
    if name == "balanced_accuracy":
        return np.mean([C[:, strata == s].mean(axis=1) for s in np.unique(strata)], axis=0)
    return C.mean(axis=1)


def strat_index(strata: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(B, n) resampling indices: column j is drawn with replacement from the items of stratum strata[j]."""
    rng = np.random.default_rng(seed)
    idx = np.empty((n_boot, len(strata)), dtype=int)
    for s in np.unique(strata):
        mem = np.flatnonzero(strata == s)
        idx[:, mem] = mem[rng.integers(0, len(mem), size=(n_boot, len(mem)))]
    return idx


def percentile_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def boot_p_two_sided(d: np.ndarray) -> float:
    B = len(d)
    return float(min(1.0, 2 * min(((d <= 0).sum() + 1) / (B + 1), ((d >= 0).sum() + 1) / (B + 1))))


def holm_lower_bounds(diffs: dict[str, np.ndarray], margin: float, alpha: float = 0.025) -> dict[str, dict]:
    """Holm's step-down procedure for the one-sided non-inferiority hypotheses H0: difference <= -margin, reported as
    lower confidence bounds. The comparisons are taken in order of their bootstrap count k = #(d* <= -margin) (ties: the
    larger estimate first); step i (1-based) of m uses level alpha / (m - i + 1), and the bound is the lower order
    statistic of the bootstrap distribution at that level (quantile method 'lower': the bound exceeds -margin exactly
    when that level is at least k / (B - 1), so bound and decision cannot disagree). At the first bound not above -margin
    the procedure stops: that comparison and every later one keep the level of that step and are not declared
    non-inferior."""
    m = len(diffs)
    k = {s: int((d <= -margin).sum()) for s, d in diffs.items()}
    order = sorted(diffs, key=lambda s: (k[s], -float(np.median(diffs[s])), s))
    out, stopped, level = {}, False, alpha / max(m, 1)
    for i, s in enumerate(order):
        if not stopped:
            level = alpha / (m - i)
        lb = float(np.quantile(diffs[s], level, method="lower"))
        ok = (not stopped) and lb > -margin
        stopped = stopped or not ok
        out[s] = {"holm_rank": i + 1, "holm_level": level, "holm_lower_bound": lb, "noninferior": ok}
    return out


def holm_upper_bounds(diffs: dict[str, np.ndarray], margin: float, alpha: float = 0.025) -> dict[str, dict]:
    """The mirror of holm_lower_bounds for H0: difference >= -margin against "worse" (difference < -margin), reported as
    upper confidence bounds: order by k = #(d* >= -margin) (ties: the smaller estimate first), step i at alpha / (m - i
    + 1), the bound the upper order statistic at that level (quantile method 'higher'); after the first bound not below
    -margin every later comparison keeps that level and is not declared worse."""
    m = len(diffs)
    k = {s: int((d >= -margin).sum()) for s, d in diffs.items()}
    order = sorted(diffs, key=lambda s: (k[s], float(np.median(diffs[s])), s))
    out, stopped, level = {}, False, alpha / max(m, 1)
    for i, s in enumerate(order):
        if not stopped:
            level = alpha / (m - i)
        ub = float(np.quantile(diffs[s], 1 - level, method="higher"))
        bad = (not stopped) and ub < -margin
        stopped = stopped or not bad
        out[s] = {"holm_rank_worse": i + 1, "holm_level_worse": level, "holm_upper_bound": ub, "worse": bad}
    return out


def outcome(noninferior: bool, worse: bool) -> str:
    """Non-inferior (lower bound above -margin), worse (upper bound below -margin), otherwise inconclusive."""
    assert not (noninferior and worse)
    return "non-inferior" if noninferior else "worse" if worse else "inconclusive"


def holm_outcomes(diffs: dict[str, np.ndarray], margin: float, alpha: float) -> dict[str, dict]:
    """Both Holm procedures across one family and each comparison's outcome."""
    lo, hi = holm_lower_bounds(diffs, margin, alpha), holm_upper_bounds(diffs, margin, alpha)
    return {s: {**lo[s], **hi[s], "outcome": outcome(lo[s]["noninferior"], hi[s]["worse"])} for s in diffs}


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    keys = sorted(p, key=lambda k: p[k])
    m, run, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def brier(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multiclass Brier score per item: sum over options of (p - outcome)^2 (0 best, 2 worst)."""
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1
    return ((P - Y) ** 2).sum(axis=1)


def rps(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score per item: mean over the K-1 thresholds of (cumulative p - cumulative outcome)^2."""
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1
    K = P.shape[1]
    return ((np.cumsum(P, axis=1) - np.cumsum(Y, axis=1))[:, :-1] ** 2).sum(axis=1) / (K - 1)


def quadratic_weighted_kappa(y: np.ndarray, yhat: np.ndarray, K: int) -> float | None:
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    if len(y) == 0:
        return None
    O = np.zeros((K, K))
    np.add.at(O, (y, yhat), 1)
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return None if den == 0 else float(1 - (W * O).sum() / den)


def _expected_cum_correct(conf: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Expected number correct among the k most confident items, k = 1..n, ties in random order."""
    order = np.argsort(-conf, kind="stable")
    c, y = conf[order], np.asarray(correct, float)[order]
    cum, run, i, n = np.empty(len(c)), 0.0, 0, len(c)
    while i < n:
        j = i
        while j < n and c[j] == c[i]:
            j += 1
        rate = y[i:j].mean()
        cum[i:j] = run + rate * np.arange(1, j - i + 1)
        run += y[i:j].sum()
        i = j
    return cum


def selective_accuracy(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float | None:
    if len(conf) == 0:
        return None
    k = max(1, int(round(coverage * len(conf))))
    return float(_expected_cum_correct(np.asarray(conf, float), correct)[k - 1] / k)


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), 1 - _expected_cum_correct(np.asarray(conf, float), correct) / k


def aurc(conf: np.ndarray, correct: np.ndarray) -> float | None:
    """Area under the risk-coverage curve: mean selective risk over coverages k/n (lower is better)."""
    return None if len(conf) == 0 else float(risk_coverage(conf, correct)[1].mean())


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float | None, list[dict]]:
    """Expected calibration error in equal-count bins (items sorted by confidence, split into `bins` near-equal groups)."""
    if len(conf) == 0:
        return None, []
    order = np.argsort(np.asarray(conf, float), kind="stable")
    table, tot = [], 0.0
    for part in np.array_split(order, min(bins, len(order))):
        c, a = float(np.mean(conf[part])), float(np.mean(np.asarray(correct, float)[part]))
        table.append({"n": int(len(part)), "mean_confidence": c, "accuracy": a})
        tot += len(part) * abs(a - c)
    return float(tot / len(order)), table


def auroc(score: np.ndarray, positive: np.ndarray) -> float | None:
    """Probability that a random positive outranks a random negative (ties count one half)."""
    from scipy.stats import rankdata
    positive = np.asarray(positive, bool)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------------------------ costs
def call_costs(system: str, tokens_in: np.ndarray, tokens_out: np.ndarray, batch: np.ndarray,
               models_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(billed, list) USD per call from reported tokens (missing tokens: nothing billed). Jev: input tokens at its
    input price, output free. Chatbots: config list prices; billed at the batch price where the batch route was used."""
    tin, tout = np.nan_to_num(np.asarray(tokens_in, float)), np.nan_to_num(np.asarray(tokens_out, float))
    if system == "jev":
        c = tin * models_cfg["jev"]["price_per_mtok_input"] / 1e6
        return c, c
    fam = system.split("@")[0]
    f = models_cfg["families"][fam]
    b = ((models_cfg.get("batch") or {}).get("families") or {}).get(fam, f)
    listp = (tin * f["price_in"] + tout * f["price_out"]) / 1e6
    return np.where(np.asarray(batch, bool), (tin * b["price_in"] + tout * b["price_out"]) / 1e6, listp), listp


# ------------------------------------------------------------------------------------------------ main-pass table
class Table:
    """Main pass (repeat 0) of one variant, items x systems, aligned on the items' order."""

    def __init__(self, calls: pd.DataFrame, items: pd.DataFrame, systems: list[str], labels: tuple[str, ...],
                 models_cfg: dict, variant: str = "main"):
        self.items, self.systems, self.labels = items.reset_index(drop=True), list(systems), tuple(labels)
        n, S, K = len(self.items), len(systems), len(labels)
        c = calls[(calls["repeat"] == 0) & (calls["variant"] == variant) & calls.system.isin(systems)]
        dup = c.duplicated(["system", "item_id"])
        assert not dup.any(), f"duplicate main-pass rows: {c[dup][['system', 'item_id']].head().values.tolist()}"
        pos = {i: k for k, i in enumerate(self.items.item_id)}
        lab = {l: k for k, l in enumerate(labels)}
        self.truth = np.array([lab[t] for t in self.items.truth])
        self.strata = self.items.stratum.astype(str).to_numpy()
        self.answer = np.full((n, S), -1)
        self.top = np.full((n, S), np.nan)
        self.P = np.full((n, S, K), np.nan)
        self.expected = np.full((n, S), np.nan)
        self.jev_conf = np.full(n, np.nan)
        self.cost_billed, self.cost_list = np.zeros((n, S)), np.zeros((n, S))
        self.tokens = {k: np.full((n, S), np.nan) for k in ("in", "out", "reasoning")}   # as reported; NaN if not
        self.present = np.zeros((n, S), bool)
        self.not_top = np.zeros((n, S), bool)
        self.prob_status: dict[str, dict] = {}
        self.parse: dict[str, dict] = {}
        self.answer_how: dict[str, dict] = {}
        self.prob_alias: dict[str, int] = {}
        self.refused: dict[str, int] = {}
        for j, s in enumerate(systems):
            g = c[c.system == s]
            g = g[g.item_id.isin(pos)]
            rows = g.item_id.map(pos).to_numpy(int)
            got = (g.error.fillna("") != BATCH_MISSING).to_numpy() if "error" in g else np.ones(len(g), bool)
            self.present[rows[got], j] = True
            ok = g.valid.fillna(False).astype(bool).to_numpy() & got
            ans = g.answer.to_numpy()
            self.answer[rows[ok], j] = [lab[a] for a in ans[ok]]
            for r, pj, t, e in zip(rows, g.probs, g.top_prob, g.expected_level):
                if isinstance(pj, str) and pj:
                    d = json.loads(pj)
                    self.P[r, j] = [d.get(l, 0.0) for l in labels]     # a label the item does not offer: zero
                    self.top[r, j] = t if t is not None and not pd.isna(t) else max(d.values())
                if e is not None and not pd.isna(e):
                    self.expected[r, j] = e
            if "choice_not_top" in g:
                self.not_top[rows, j] = g.choice_not_top.fillna(False).astype(bool).to_numpy()
            batch = (g.route.fillna("") == "batch").to_numpy() if "route" in g else np.zeros(len(g), bool)
            bb, ll = call_costs(s, g.tokens_in.to_numpy(float) if "tokens_in" in g else np.zeros(len(g)),
                                g.tokens_out.to_numpy(float) if "tokens_out" in g else np.zeros(len(g)), batch, models_cfg)
            self.cost_billed[rows, j], self.cost_list[rows, j] = bb, ll
            for k, col in (("in", "tokens_in"), ("out", "tokens_out"), ("reasoning", "tokens_reasoning")):
                if col in g:
                    self.tokens[k][rows, j] = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            if s == "jev" and "jev_confidence" in g:
                self.jev_conf[rows] = g.jev_confidence.to_numpy(float)
            self.prob_status[s] = g.prob_status.fillna("none").value_counts().to_dict() if "prob_status" in g else {}
            self.parse[s] = g.parse.fillna("none").value_counts().to_dict() if "parse" in g else {}
            self.answer_how[s] = g.answer_how.fillna("none").value_counts().to_dict() if "answer_how" in g else {}
            self.prob_alias[s] = int(g.prob_alias.fillna(False).astype(bool).sum()) if "prob_alias" in g else 0
            self.refused[s] = int((g.finish_reason == "content_filter").sum()) if "finish_reason" in g else 0
        self.correct = self.answer == self.truth[:, None]
        self.usable = self.answer >= 0

    def col(self, s: str) -> int:
        return self.systems.index(s)


# ------------------------------------------------------------------------------------------------ blocks
def rates(t: Table, j: int, spec: dict) -> dict:
    out = {}
    for name, r in (spec or {}).items():
        among = r.get("among")
        mask = np.ones(len(t.items), bool) if among is None else (t.strata == str(among))
        hit = np.isin(t.answer[:, j], [t.labels.index(a) for a in r["answer_in"]])
        out[name] = float(hit[mask].mean()) if mask.any() else None
    return out


def confidence_block(conf: np.ndarray, correct: np.ndarray, P: np.ndarray | None, y: np.ndarray, kind: str,
                     coverages: list[float], bins: int, min_errors: int) -> dict:
    """Every measure on the answers that have a confidence (and, for the proper score, a probability list)."""
    m = ~np.isnan(conf)
    conf, correct = conf[m], correct[m]
    out: dict[str, Any] = {"n": int(m.sum()), "n_errors": int((~correct).sum())}
    if P is not None:
        Pm, ym = P[m], y[m]
        ps = brier(Pm, ym) if kind == "choice" else rps(Pm, ym)
        out["brier" if kind == "choice" else "rps"] = float(ps.mean()) if len(ps) else None
    out["accuracy_answered"] = float(correct.mean()) if len(correct) else None
    out["selective_accuracy"] = {str(c): selective_accuracy(conf, correct, c) for c in coverages}
    out["aurc"] = aurc(conf, correct)
    e, table = ece(conf, correct, bins)
    out["ece"], out["reliability"] = e, table
    if len(conf):
        cov, risk = risk_coverage(conf, correct)
        keep = np.unique(np.linspace(0, len(cov) - 1, min(len(cov), 50)).astype(int))
        out["risk_coverage"] = {"coverage": cov[keep].tolist(), "risk": risk[keep].tolist()}
    out["auroc"] = auroc(conf, correct) if out["n_errors"] >= min_errors else None
    out["auroc_note"] = None if out["n_errors"] >= min_errors else f"fewer than {min_errors} errors"
    return out


def system_block(t: Table, s: str, kind: str, a: dict, idx: np.ndarray) -> dict:
    j = t.col(s)
    corr, n = t.correct[:, j], len(t.items)
    pm = a["primary_metric"]
    boot = metric_boot(corr[idx], t.strata, pm)
    out: dict[str, Any] = {
        "n_items": n, "present": int(t.present[:, j].sum()), "usable": int(t.usable[:, j].sum()),
        "usable_share": float(t.usable[:, j].mean()), "unusable": int((~t.usable[:, j]).sum()),
        "unusable_rate": float((~t.usable[:, j]).mean()), "correct": int(corr.sum()),
        "accuracy": float(corr.mean()), pm: accuracy_metric(corr, t.strata, pm),
        f"{pm}_ci": percentile_ci(boot, a["ci"]),
        "accuracy_by_stratum": {str(v): float(corr[t.strata == v].mean()) for v in np.unique(t.strata)},
        "accuracy_by_stratum_ci": {str(v): percentile_ci(corr[idx][:, t.strata == v].mean(axis=1), a["ci"])
                                   for v in np.unique(t.strata)},
        "rates": rates(t, j, a.get("rates")), "choice_not_top": int(t.not_top[:, j].sum()),
        "prob_status": t.prob_status.get(s, {}), "parse": t.parse.get(s, {}),
        "answer_how": t.answer_how.get(s, {}), "prob_alias": t.prob_alias.get(s, 0),
        "refused_by_provider": t.refused.get(s, 0)}
    if kind == "score":
        ok = t.usable[:, j]
        K = len(t.labels)
        out["qwk"] = quadratic_weighted_kappa(t.truth[ok], t.answer[ok, j], K)
        out["qwk_n"] = int(ok.sum())
        out["off_by_one"] = float((ok & (np.abs(t.answer[:, j] - t.truth) == 1)).mean())
        e = ~np.isnan(t.expected[:, j])
        out["expected_level_mae"] = float(np.abs(t.expected[e, j] - t.truth[e]).mean()) if e.any() else None
        out["expected_level_n"] = int(e.sum())
    has_p = ~np.isnan(t.P[:, j, 0]) & t.usable[:, j]
    conf = np.where(has_p, t.top[:, j], np.nan)
    out["confidence"] = confidence_block(conf, corr, np.nan_to_num(t.P[:, j]), t.truth, kind, a["coverages"],
                                         a["ece_bins"], a["min_errors_auroc"])
    if s == "jev":
        jc = np.where(t.usable[:, j] & ~np.isnan(t.jev_conf), t.jev_conf, np.nan)
        out["jev_own_confidence"] = confidence_block(jc, corr, None, t.truth, kind, a["coverages"], a["ece_bins"],
                                                     a["min_errors_auroc"])
    if a.get("certain_at") is not None:
        out["certain"] = certain_block(corr, np.where(has_p, t.top[:, j], np.nan), float(a["certain_at"]), idx, a["ci"])
    out["cost"] = cost_block(t, j, idx, a["ci"])
    return out


def certain_block(correct: np.ndarray, top: np.ndarray, at: float, idx: np.ndarray, level: float) -> dict:
    """The share of items a system answers with probability 1.00 (top probability at or above `at`,
    0.995: 1.00 at two decimals, with a usable answer and a probability list), its accuracy there and on every other item
    (unusable answers counted wrong), and the difference; percentile intervals over the resampled items (a resample
    with no item on one side leaves that side out and is counted)."""
    sure = ~np.isnan(top) & (top >= at - 1e-12)
    C, S = np.asarray(correct, float)[idx], sure[idx]
    ns, nr = S.sum(axis=1), (~S).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_s, acc_r = (C * S).sum(axis=1) / ns, (C * ~S).sum(axis=1) / nr
    both = (ns > 0) & (nr > 0)
    ci = lambda x, m: percentile_ci(x[m], level) if m.any() else None
    return {"at": at, "n": int(sure.sum()), "n_items": int(len(sure)), "share": float(sure.mean()),
            "share_ci": percentile_ci(S.mean(axis=1), level),
            "n_exactly_1": int((~np.isnan(top) & (top >= 1 - 1e-12)).sum()),
            "accuracy": float(correct[sure].mean()) if sure.any() else None, "accuracy_ci": ci(acc_s, ns > 0),
            "accuracy_below": float(correct[~sure].mean()) if (~sure).any() else None,
            "accuracy_below_ci": ci(acc_r, nr > 0),
            "diff": float(correct[sure].mean() - correct[~sure].mean()) if sure.any() and (~sure).any() else None,
            "diff_ci": ci(acc_s - acc_r, both), "resamples_one_side_empty": int((~both).sum())}


def option_count_contrast(tables: dict[str, "Table"], counts: dict[str, np.ndarray], a: dict, idx: np.ndarray) -> dict:
    """Does accuracy depend on the number of options? The same questions asked with two option
    lists (two variants; per item the shorter list holds the longer one's options less one wrong option; MedQA keeps
    their order, Medbullets' 4-option file reorders them on most items). Per system, accuracy with the
    longer list minus the shorter, paired over the same questions (resampled within the stratum); by stratum; and Jev's
    change minus each chatbot's (does an extra option cost Jev more?). Descriptive: no multiplicity adjustment. Also the
    mean top probability with each list and the share of usable answers."""
    (va, ta), (vb, tb) = list(tables.items())
    na, nb = counts[va], counts[vb]
    assert (ta.items.item_id.to_numpy() == tb.items.item_id.to_numpy()).all() and (na != nb).all()
    more = na > nb
    lvl, strata = a["ci"], ta.strata
    out: dict[str, Any] = {"variants": [va, vb], "n_items": int(len(na)),
                           "options_by_stratum": {str(s): {va: sorted(set(na[strata == s].tolist())),
                                                           vb: sorted(set(nb[strata == s].tolist()))}
                                                  for s in np.unique(strata)},
                           "contrast": "accuracy with the longer option list minus the shorter, same questions",
                           "systems": {}, "jev_minus_chatbot": {}}
    delta = {}
    for s in ta.systems:
        ja, jb = ta.col(s), tb.col(s)
        c_long = np.where(more, ta.correct[:, ja], tb.correct[:, jb]).astype(float)
        c_short = np.where(more, tb.correct[:, jb], ta.correct[:, ja]).astype(float)
        top_l = np.where(more, np.where(ta.usable[:, ja], ta.top[:, ja], np.nan), np.where(tb.usable[:, jb], tb.top[:, jb], np.nan))
        top_s = np.where(more, np.where(tb.usable[:, jb], tb.top[:, jb], np.nan), np.where(ta.usable[:, ja], ta.top[:, ja], np.nan))
        d = c_long - c_short
        delta[s] = d[idx].mean(axis=1)
        out["systems"][s] = {
            "accuracy_longer": float(c_long.mean()), "accuracy_shorter": float(c_short.mean()),
            "diff": float(d.mean()), "diff_ci": percentile_ci(delta[s], lvl),
            "diff_by_stratum": {str(v): {"diff": float(d[strata == v].mean()),
                                         "ci": percentile_ci(d[idx][:, strata == v].mean(axis=1), lvl)}
                                for v in np.unique(strata)},
            "right_only_longer": int(((c_long == 1) & (c_short == 0)).sum()),
            "right_only_shorter": int(((c_long == 0) & (c_short == 1)).sum()),
            "mean_top_longer": float(np.nanmean(top_l)) if (~np.isnan(top_l)).any() else None,
            "mean_top_shorter": float(np.nanmean(top_s)) if (~np.isnan(top_s)).any() else None,
            "usable_longer": float(np.where(more, ta.usable[:, ja], tb.usable[:, jb]).mean()),
            "usable_shorter": float(np.where(more, tb.usable[:, jb], ta.usable[:, ja]).mean())}
    if "jev" in delta:
        for s in [x for x in ta.systems if x != "jev"]:
            est = out["systems"]["jev"]["diff"] - out["systems"][s]["diff"]
            out["jev_minus_chatbot"][s] = {"diff": est, "ci": percentile_ci(delta["jev"] - delta[s], lvl)}
    return out


def cost_block(t: Table, j: int, idx: np.ndarray, level: float) -> dict:
    """Per answer (one main-pass call per item, unusable ones included) the mean reported input,
    output and reasoning tokens (reasoning is part of output; None where the provider reports none); billed and list
    cost per 1,000 answers; list cost of one million answers; cost per correct answer, the list one with a bootstrap
    percentile CI (the ratio of sums over the same resampled items; a resample without a correct answer is left out
    and counted)."""
    n, corr = len(t.items), t.correct[:, j]
    cb, cl, nc = float(t.cost_billed[:, j].sum()), float(t.cost_list[:, j].sum()), int(corr.sum())
    tok = {}
    for k, x in t.tokens.items():
        v = x[:, j][~np.isnan(x[:, j])]
        tok[f"tokens_{k}_per_answer"] = float(v.mean()) if len(v) else None
        tok[f"tokens_{k}_n"] = int(len(v))
    bc = corr[idx].sum(axis=1)
    ok = bc > 0
    ratio = t.cost_list[:, j][idx].sum(axis=1)[ok] / bc[ok]
    return {**tok, "usd_billed_total": cb, "usd_list_total": cl,
            "usd_billed_per_answer": cb / n, "usd_list_per_answer": cl / n,
            "usd_billed_per_1000_answers": cb / n * 1000, "usd_list_per_1000_answers": cl / n * 1000,
            "usd_list_per_million_answers": cl / n * 1e6,
            "usd_billed_per_correct": cb / nc if nc else None, "usd_list_per_correct": cl / nc if nc else None,
            "usd_list_per_correct_ci": percentile_ci(ratio, level) if ok.any() else None,
            "resamples_without_a_correct_answer": int((~ok).sum())}


def boot_diff(t: Table, s: str, a: dict, idx: np.ndarray) -> np.ndarray:
    """Bootstrap distribution of Jev minus chatbot s on the primary metric (paired: the same resampled items)."""
    pm, cj, cc = a["primary_metric"], t.correct[:, t.col("jev")], t.correct[:, t.col(s)]
    return metric_boot(cj[idx], t.strata, pm) - metric_boot(cc[idx], t.strata, pm)


def compare(t: Table, s: str, a: dict, idx: np.ndarray) -> dict:
    """Jev minus chatbot s on the arm's primary metric: estimate, 95% percentile CI, two-sided bootstrap p and the
    unadjusted one-sided lower bound at alpha (the non-inferiority bound of a system outside the primary family)."""
    pm, jj, jc = a["primary_metric"], t.col("jev"), t.col(s)
    cj, cc = t.correct[:, jj], t.correct[:, jc]
    d = boot_diff(t, s, a, idx)
    lo, hi = percentile_ci(d, a["ci"])
    return {"diff": accuracy_metric(cj, t.strata, pm) - accuracy_metric(cc, t.strata, pm), "ci": [lo, hi],
            "p_two_sided": boot_p_two_sided(d),
            "lower_bound_unadjusted": float(np.quantile(d, a["alpha_one_sided"], method="lower")),
            "upper_bound_unadjusted": float(np.quantile(d, 1 - a["alpha_one_sided"], method="higher")),
            "both_correct": int((cj & cc).sum()), "jev_only": int((cj & ~cc).sum()), "chatbot_only": int((~cj & cc).sum())}


def value_vs_jev(t: Table, s: str, idx: np.ndarray, level: float) -> dict:
    """Extra cost per extra correct answer when chatbot s replaces Jev: difference in total cost over difference in
    correct answers, with the per-1,000-item differences; bootstrap percentile CIs (the ratio over resamples where the
    chatbot has more correct answers; the share of the others reported). The ratio is reported only where the chatbot
    is more accurate (more correct answers); otherwise the verdict reads Jev cheaper and at least as
    accurate (list price)."""
    jj, jc, n = t.col("jev"), t.col(s), len(t.items)
    dc = t.correct[:, jc].astype(float) - t.correct[:, jj]
    more = bool(dc.sum() > 0)
    cheaper = bool(t.cost_list[:, jj].sum() <= t.cost_list[:, jc].sum())
    out: dict[str, Any] = {"chatbot_more_accurate": more, "jev_cheaper_list": cheaper,
                           "verdict_list": ("chatbot more accurate" if more else "Jev cheaper and at least as accurate"
                                            if cheaper else "Jev at least as accurate, not cheaper")}
    for basis, cost in (("list", t.cost_list), ("billed", t.cost_billed)):
        dk = cost[:, jc] - cost[:, jj]
        est_k, est_c = float(dk.sum()), float(dc.sum())
        bk, bc = dk[idx].sum(axis=1), dc[idx].sum(axis=1)
        pos = bc > 0
        ratio = bk[pos] / bc[pos]
        out[basis] = {"extra_usd_per_1000_items": est_k / n * 1000, "extra_correct_per_1000_items": est_c / n * 1000,
                      "extra_usd_per_1000_items_ci": percentile_ci(bk / n * 1000, level),
                      "extra_correct_per_1000_items_ci": percentile_ci(bc / n * 1000, level),
                      "usd_per_extra_correct": est_k / est_c if est_c > 0 else None,
                      "usd_per_extra_correct_ci": percentile_ci(ratio, level) if pos.sum() >= 20 else None,
                      "share_resamples_chatbot_not_more_correct": float(1 - pos.mean()),
                      "note": None if est_c > 0 else "chatbot not more often correct than Jev: no extra correct answers to price"}
    return out


def hybrid(t: Table, s: str, a: dict) -> dict:
    """Jev answers when its top probability is at or above the threshold (and it gave a usable answer with a
    probability list), chatbot s otherwise; Jev is always asked, the chatbot only on the passed items. Thresholds at
    the deciles (0-90%) of Jev's top probability, plus one above its maximum (chatbot alone). Each point also carries
    the list and billed cost per 1,000 answers and, beside it, the random line: the same share handed to Jev at random
    (random_routed), with its list cost per 1,000 answers."""
    pm, jj, jc = a["primary_metric"], t.col("jev"), t.col(s)
    topj = np.where(t.usable[:, jj], t.top[:, jj], np.nan)
    valid = ~np.isnan(topj)
    chat_alone = accuracy_metric(t.correct[:, jc], t.strata, pm)
    ths = sorted(set(np.quantile(topj[valid], np.arange(0, 1.0, 0.1)).tolist())) if valid.any() else []
    cj, cc, kj, kc = t.correct[:, jj], t.correct[:, jc], t.cost_list[:, jj], t.cost_list[:, jc]
    grid = []
    for q in ths + [math.inf]:
        use = valid & (topj >= q)
        corr = np.where(use, cj, cc)
        m, sh = accuracy_metric(corr, t.strata, pm), float(use.mean())
        lst = float((kj + np.where(use, 0, kc)).mean())
        bil = float((t.cost_billed[:, jj] + np.where(use, 0, t.cost_billed[:, jc])).mean())
        grid.append({"threshold": None if q == math.inf else float(q), "share_jev": sh, pm: m,
                     "accuracy": float(corr.mean()),
                     "usd_list_per_answer": lst, "usd_billed_per_answer": bil,
                     "usd_list_per_1000_answers": lst * 1000, "usd_billed_per_1000_answers": bil * 1000,
                     f"{pm}_random_routed": random_routed(cj, cc, sh, t.strata, pm),
                     "accuracy_random_routed": random_routed(cj, cc, sh, t.strata, "accuracy"),
                     "usd_list_per_1000_random_routed": float(kj.mean() + (1 - sh) * kc.mean()) * 1000,
                     "within_margin": bool(m >= chat_alone - a["margin"] - 1e-12)})
    first = next((g for g in grid if g["within_margin"]), None)
    return {"chatbot_alone": chat_alone, "grid": grid, "first_within_margin": first}


def hybrid_split(t: Table, share: float) -> np.ndarray:
    """The items Jev answers in the hybrid at a set share, from Jev's output only (never the truth): the
    floor(share x n) items with its highest top probability, ties by item id; an item without a usable Jev answer and
    probability list ranks last. (The floor allows for rounding in the product: 0.7 x 180 is 126, not 125.)"""
    jj = t.col("jev")
    conf = np.where(t.usable[:, jj] & ~np.isnan(t.top[:, jj]), t.top[:, jj], -np.inf)
    order = sorted(range(len(t.items)), key=lambda i: (-conf[i], str(t.items.item_id.iloc[i])))
    use = np.zeros(len(t.items), bool)
    use[order[:int(math.floor(share * len(t.items) + 1e-9))]] = True
    return use


def hybrid_share(t: Table, s: str, a: dict, idx: np.ndarray, use: np.ndarray) -> tuple[dict, np.ndarray]:
    """Hybrid (Jev on the items in `use`, chatbot s on the rest) minus chatbot s alone on the primary metric, paired
    bootstrap on the same resampled items; list-price cost of the hybrid (Jev on every item plus the chatbot on its
    share) against the chatbot alone. Returns the block and the bootstrap differences for the Holm procedures."""
    pm, jj, jc = a["primary_metric"], t.col("jev"), t.col(s)
    hc = np.where(use, t.correct[:, jj], t.correct[:, jc])
    cc = t.correct[:, jc]
    hb, cb = metric_boot(hc[idx], t.strata, pm), metric_boot(cc[idx], t.strata, pm)
    d = hb - cb
    return {"share_jev": float(use.mean()), "n_jev": int(use.sum()),
            "share_jev_by_stratum": {str(v): float(use[t.strata == v].mean()) for v in np.unique(t.strata)},
            pm: accuracy_metric(hc, t.strata, pm), f"{pm}_ci": percentile_ci(hb, a["ci"]),
            "chatbot_alone": accuracy_metric(cc, t.strata, pm),
            "diff": accuracy_metric(hc, t.strata, pm) - accuracy_metric(cc, t.strata, pm),
            "ci": list(percentile_ci(d, a["ci"])), "cost_list": hybrid_cost(t, s, use)}, d


def hybrid_cost(t: Table, s: str, use: np.ndarray) -> dict:
    """List-price cost of the hybrid (Jev asked on every item, chatbot s only on the items outside `use`) against
    chatbot s alone, and the saving."""
    jj, jc, n = t.col("jev"), t.col(s), len(t.items)
    hc = np.where(use, t.correct[:, jj], t.correct[:, jc])
    cc = t.correct[:, jc]
    k_h = float(t.cost_list[:, jj].sum() + t.cost_list[~use, jc].sum())
    k_c = float(t.cost_list[:, jc].sum())
    return {"hybrid_usd_total": k_h, "chatbot_usd_total": k_c,
            "hybrid_usd_per_1000_items": k_h / n * 1000, "chatbot_usd_per_1000_items": k_c / n * 1000,
            "saving_usd_per_1000_items": (k_c - k_h) / n * 1000,
            "saving_share": 1 - k_h / k_c if k_c > 0 else None,
            "hybrid_over_chatbot": k_h / k_c if k_c > 0 else None,
            "hybrid_usd_per_correct": k_h / hc.sum() if hc.sum() else None,
            "chatbot_usd_per_correct": k_c / cc.sum() if cc.sum() else None}


def hybrid_point(t: Table, s: str, a: dict, use: np.ndarray, rule: str) -> dict:
    """The hybrid for the cost table: its share answered by Jev, primary metric beside the chatbot alone (point
    estimates), and hybrid_cost."""
    pm, jj, jc = a["primary_metric"], t.col("jev"), t.col(s)
    hc = np.where(use, t.correct[:, jj], t.correct[:, jc])
    return {"rule": rule, "share_jev": float(use.mean()), "n_jev": int(use.sum()),
            pm: accuracy_metric(hc, t.strata, pm), "chatbot_alone": accuracy_metric(t.correct[:, jc], t.strata, pm),
            **hybrid_cost(t, s, use)}


def random_routed(cj: np.ndarray, cc: np.ndarray, share: float, strata: np.ndarray, name: str) -> float | np.ndarray:
    """The metric expected when Jev answers a random set of
    the items of the given share and the chatbot the rest, the mean over every such set (exact, no seed). Each item is
    Jev's with probability `share` and both metrics are linear in the items' correctness, so it is share x Jev's metric
    + (1 - share) x the chatbot's. On (B, n) resampled correctness (columns keep their stratum), one value per
    resample."""
    if np.ndim(cj) == 2:
        return share * metric_boot(cj, strata, name) + (1 - share) * metric_boot(cc, strata, name)
    return share * accuracy_metric(cj, strata, name) + (1 - share) * accuracy_metric(cc, strata, name)


def _routing_metrics(pm: str) -> list[str]:
    return [pm] + (["accuracy"] if pm == "balanced_accuracy" else [])


def routing_control(t: Table, s: str, a: dict, use: np.ndarray, idx: np.ndarray) -> dict:
    """Random-half control for the hybrid (arm 3's supplement computes the same): confidence-routed
    (Jev on the items in `use`, its highest top probabilities) minus random-routed (the same share handed to Jev at
    random), paired over the resampled items (the main analysis's resamples, within the truth stratum), on the primary
    metric (and plain accuracy where the primary is balanced accuracy)."""
    jj, jc = t.col("jev"), t.col(s)
    cj, cc = t.correct[:, jj], t.correct[:, jc]
    share = float(use.mean())
    hc = np.where(use, cj, cc)
    out: dict[str, Any] = {"share_jev": share, "n_jev": int(use.sum())}
    for name in _routing_metrics(a["primary_metric"]):
        conf, rnd = accuracy_metric(hc, t.strata, name), float(random_routed(cj, cc, share, t.strata, name))
        d = metric_boot(hc[idx], t.strata, name) - random_routed(cj[idx], cc[idx], share, t.strata, name)
        out[name] = {"confidence_routed": conf, "random_routed": rnd, "diff": conf - rnd,
                     "diff_ci": list(percentile_ci(d, a["ci"]))}
    return out


def coverage_curve(t: Table, s: str, a: dict, shares: list[float]) -> list[dict]:
    """Arm 3's coverage curve: at each share of Jev (hybrid_split: its most confident items), the
    metric and the list cost per 1,000 answers (Jev asked on every item, chatbot s on the rest), routed by confidence
    and at random (expected over every set of the same size)."""
    pm, jj, jc = a["primary_metric"], t.col("jev"), t.col(s)
    cj, cc, kj, kc = t.correct[:, jj], t.correct[:, jc], t.cost_list[:, jj], t.cost_list[:, jc]
    pts = []
    for c in shares:
        use = hybrid_split(t, c)
        sh = float(use.mean())
        p = {"share": float(c), "share_jev": sh, "n_jev": int(use.sum())}
        for name in _routing_metrics(pm):
            p[f"{name}_confidence_routed"] = accuracy_metric(np.where(use, cj, cc), t.strata, name)
            p[f"{name}_random_routed"] = float(random_routed(cj, cc, sh, t.strata, name))
        p["usd_list_per_1000_confidence_routed"] = float((kj + np.where(use, 0.0, kc)).mean() * 1000)
        p["usd_list_per_1000_random_routed"] = float((kj.mean() + (1 - sh) * kc.mean()) * 1000)
        pts.append(p)
    return pts


def acceptable_cheaper(cmp: dict, jev_sys: dict, chat_sys: dict, a: dict) -> dict:
    """Jev is an acceptable cheaper alternative to a chatbot if the lower bound of (Jev minus chatbot) is above -margin
    (primary chatbots: the Holm-adjusted bound; others: the unadjusted one-sided bound, descriptive) and Jev's
    list-price cost per correct answer is lower. The same rule on billed costs is reported alongside."""
    lb = cmp["holm_lower_bound"] if cmp.get("confirmatory") else cmp["lower_bound_unadjusted"]
    out = {"lower_bound": lb, "bound": "holm" if cmp.get("confirmatory") else "unadjusted (descriptive)",
           "margin": -a["margin"], "basis": a.get("cost_basis", "list"), "within_margin": bool(lb > -a["margin"])}
    for basis in ("list", "billed"):
        jc, cc = jev_sys["cost"][f"usd_{basis}_per_correct"], chat_sys["cost"][f"usd_{basis}_per_correct"]
        cheaper = jc is not None and (cc is None or jc < cc)
        out[f"jev_usd_{basis}_per_correct"], out[f"chatbot_usd_{basis}_per_correct"] = jc, cc
        out[f"jev_cheaper_{basis}"] = bool(cheaper)
        out[f"verdict_{basis}"] = bool(out["within_margin"] and cheaper)
    out["verdict"] = out[f"verdict_{out['basis']}"]
    return out


def fleiss_kappa(counts: np.ndarray) -> np.ndarray:
    """Fleiss' kappa from category counts, shape (..., items, categories), every item rated the same number of times
    m >= 2; NaN where chance agreement is 1 (every answer in one category)."""
    N = np.asarray(counts, float)
    m = N.sum(axis=-1)
    p_item = ((N ** 2).sum(axis=-1) - m) / (m * (m - 1))
    p_cat = N.sum(axis=-2) / N.sum(axis=(-2, -1))[..., None]
    pe = (p_cat ** 2).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pe < 1 - 1e-12, (p_item.mean(axis=-1) - pe) / (1 - pe), np.nan)


def icc_2_1(Y: np.ndarray) -> np.ndarray:
    """ICC(2,1) (Shrout and Fleiss 1979; McGraw and Wong's ICC(A,1)): two-way random effects, absolute agreement, one
    rating, from Y of shape (..., items, raters); NaN where the denominator is 0 (no variation)."""
    Y = np.asarray(Y, float)
    n, k = Y.shape[-2], Y.shape[-1]
    g = Y.mean(axis=(-2, -1), keepdims=True)
    ssr = k * ((Y.mean(axis=-1, keepdims=True) - g) ** 2).sum(axis=(-2, -1))
    ssc = n * ((Y.mean(axis=-2, keepdims=True) - g) ** 2).sum(axis=(-2, -1))
    sse = ((Y - g) ** 2).sum(axis=(-2, -1)) - ssr - ssc
    with np.errstate(divide="ignore", invalid="ignore"):
        msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
        den = msr + (k - 1) * mse + k * (msc - mse) / n
        return np.where(np.isfinite(den) & (np.abs(den) > 1e-12), (msr - mse) / den, np.nan)


def repeatability(calls: pd.DataFrame, items: pd.DataFrame, rep_ids: list[str], systems: list[str],
                  labels: tuple[str, ...], kind: str, n_repeats: int, n_boot: int, seed: int, level: float,
                  variant: str = "main") -> dict:
    """Repeatability on the repeat set per system (repeat 0 is the main pass). The items with a usable
    answer in every repeat enter; the counts of the others are reported. Share of items answered identically in all
    repeats; arm 2 (choice) Fleiss' kappa across the repeats' answers, arm 3 (score) ICC(2,1) of the level index (the
    answer's position on the ordered scale). 95% percentile CIs from a bootstrap over those items, stratified by the truth
    stratum; resamples where the coefficient is undefined are left out and counted."""
    lab = {l: k for k, l in enumerate(labels)}
    strata = items.set_index("item_id").stratum.astype(str)
    c = calls[(calls.variant == variant) & calls.item_id.isin(rep_ids) & (calls["repeat"] < n_repeats)]
    if "error" in c:
        c = c[c.error.fillna("") != BATCH_MISSING]
    name = "fleiss_kappa" if kind == "choice" else "icc_2_1"
    out = {}
    for s in systems:
        g = c[c.system == s]
        code = [lab.get(x, -1) if v else -1 for x, v in zip(g.answer, g.valid.fillna(False).astype(bool))]
        wide = pd.DataFrame({"item_id": g.item_id.to_numpy(), "repeat": g["repeat"].to_numpy(), "code": code}).pivot_table(
            index="item_id", columns="repeat", values="code", aggfunc="first") if len(g) else pd.DataFrame()
        complete = wide.dropna() if wide.shape[1] == n_repeats else wide.iloc[:0]
        usable = complete[(complete >= 0).all(axis=1)].sort_index()
        R = usable.to_numpy(int)
        r = {"items": len(rep_ids), "repeats": n_repeats, "items_complete": int(len(complete)),
             "items_all_usable": int(len(R)),
             "measure": name, "identical_share": None, "identical_share_ci": None, name: None, f"{name}_ci": None,
             "undefined_resamples": None}
        if len(R):
            idx = strat_index(strata.loc[usable.index].to_numpy(), n_boot, seed)
            same = (R == R[:, :1]).all(axis=1)
            if kind == "choice":
                N = np.stack([(R == k).sum(axis=1) for k in range(len(labels))], axis=1)
                est, boot = fleiss_kappa(N), fleiss_kappa(N[idx])
            else:
                est, boot = icc_2_1(R), icc_2_1(R[idx])
            ok = ~np.isnan(boot)
            r.update({"identical_share": float(same.mean()), "identical_share_ci": percentile_ci(same[idx].mean(axis=1), level),
                      name: None if np.isnan(est) else float(est),
                      f"{name}_ci": percentile_ci(boot[ok], level) if ok.any() else None,
                      "undefined_resamples": int((~ok).sum())})
        out[s] = r
    return out


def timing_summary(df: pd.DataFrame | None) -> dict:
    if df is None or not len(df):
        return {}
    out = {}
    for s, g in df.groupby("system", sort=False):
        lat = g.latency_s.dropna().astype(float)
        out[s] = {"n": int(len(g)), "usable": int(g.valid.fillna(False).sum()), "errors": int(g.error.notna().sum()),
                  "median_s": float(lat.median()) if len(lat) else None,
                  "p90_s": float(lat.quantile(0.9)) if len(lat) else None,
                  "mean_s": float(lat.mean()) if len(lat) else None}
        w = g.window_start.dropna() if "window_start" in g else ()
        if len(w):                                                         # local time, as in timing_sample.json
            out[s]["window_start"] = datetime.datetime.fromtimestamp(float(w.iloc[0])).isoformat(timespec="seconds")
    return out


# ------------------------------------------------------------------------------------------------ style check
def style_features(texts, spec: dict) -> np.ndarray:
    """(n, k): the number of matches of each regex in spec['features'] (config order) in each text."""
    flags = re.IGNORECASE if spec.get("ignore_case", False) else 0
    pats = [re.compile(p, flags) for p in spec["features"].values()]
    return np.array([[len(p.findall(str(s))) for p in pats] for s in texts], float)


def logit_fit(X: np.ndarray, y: np.ndarray, class_weight: str | None) -> tuple[float, np.ndarray]:
    """Unpenalised logistic regression of y on X: (intercept, slopes) on the scale of X. Fitted on standardised
    columns (the same fit, better conditioned); a constant column gets slope 0."""
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    m = LogisticRegression(penalty=None, class_weight=class_weight, max_iter=10_000).fit((X - mu) / sd, y)
    beta = m.coef_[0] / sd
    return float(m.intercept_[0] - (beta * mu).sum()), beta


def style_baseline(X: np.ndarray, y: np.ndarray, groups: np.ndarray, class_weight: str | None) -> np.ndarray:
    """Leave-one-group-out log-odds of y = 1 for every row, from a fit on the other groups only."""
    z = np.full(len(y), np.nan)
    for g in np.unique(groups):
        out = groups == g
        if len(np.unique(y[~out])) < 2:
            raise ValueError(f"style baseline: the fold without {g} holds one class only")
        b0, b = logit_fit(X[~out], y[~out], class_weight)
        z[out] = b0 + X[out] @ b
    return z


def _subset(correct: np.ndarray, strata: np.ndarray, all_strata: np.ndarray, pm: str, n_boot: int, seed: int,
            level: float) -> dict:
    """The primary metric on a subset of items, with its own stratified bootstrap; None when a stratum is absent."""
    out: dict[str, Any] = {"n": int(len(correct)),
                           "n_by_stratum": {str(s): int((strata == s).sum()) for s in all_strata},
                           "accuracy_by_stratum": {str(s): float(correct[strata == s].mean()) if (strata == s).any()
                                                   else None for s in all_strata}}
    whole = all(out["n_by_stratum"][str(s)] for s in all_strata)
    out[pm] = accuracy_metric(correct, strata, pm) if whole else None
    out[f"{pm}_ci"] = (percentile_ci(metric_boot(correct[strat_index(strata, n_boot, seed)], strata, pm), level)
                       if whole else None)
    return out


def style_check(t: Table, a: dict, idx: np.ndarray, seed: int) -> dict:
    """Descriptive. Two features of each item's message (config
    analysis.style_check), a style-only baseline (logistic regression of the positive stratum on them, leave one group
    out, the out-of-fold prediction mapped to the answer its stratum implies) reported beside the systems with the
    systems' own bootstrap resamples, and every system's primary metric on the items the baseline gets right
    (style-matching) and wrong (style-misleading)."""
    spec, pm = a["style_check"], a["primary_metric"]
    it = t.items
    miss = {"state", "group"} - set(it.columns)
    if miss:
        raise ValueError(f"style check needs item columns {sorted(miss)}")
    strata = np.unique(t.strata)
    pos = str(spec["positive_stratum"])
    if len(strata) != 2 or pos not in strata:
        raise ValueError(f"style check needs two strata including {pos}; got {strata.tolist()}")
    neg = str(strata[strata != pos][0])
    implied = {}
    for s in strata:
        tr = set(it.truth[t.strata == s])
        assert len(tr) == 1, f"stratum {s} has several answers {tr}"
        implied[str(s)] = t.labels.index(tr.pop())
    X = style_features(it.state, spec)
    y = (t.strata == pos).astype(int)
    z = style_baseline(X, y, it.group.astype(str).to_numpy(), spec.get("class_weight"))
    pred = np.where(z > 0, pos, neg)
    correct = np.where(z > 0, implied[pos], implied[neg]) == t.truth
    b0, beta = logit_fit(X, y, spec.get("class_weight"))
    names = list(spec["features"])
    base = {"n_items": len(it), pm: accuracy_metric(correct, t.strata, pm),
            f"{pm}_ci": percentile_ci(metric_boot(correct[idx], t.strata, pm), a["ci"]),
            "accuracy": float(correct.mean()), "correct": int(correct.sum()),
            "accuracy_by_stratum": {str(s): float(correct[t.strata == s].mean()) for s in strata},
            "predicted": {str(s): {str(p): int(((t.strata == s) & (pred == p)).sum()) for p in strata} for s in strata},
            "answers": {pos: t.labels[implied[pos]], neg: t.labels[implied[neg]]},
            "folds": int(it.group.nunique()),
            "log_odds_per_unit_all_items": dict(zip(names, beta.tolist())),
            "note": "reference row, not a system: item resampling only (the regression is not refitted per resample)"}
    feats = {str(s): {n: {"mean": float(X[t.strata == s, k].mean()), "median": float(np.median(X[t.strata == s, k]))}
                      for k, n in enumerate(names)} for s in strata}
    n_boot = int(a["n_boot"])
    sysout = {}
    for j, s in enumerate(t.systems):
        c = t.correct[:, j]
        m = _subset(c[correct], t.strata[correct], strata, pm, n_boot, seed, a["ci"])
        w = _subset(c[~correct], t.strata[~correct], strata, pm, n_boot, seed, a["ci"])
        sysout[s] = {"style_matching": m, "style_misleading": w,
                     "matching_minus_misleading": (m[pm] - w[pm]) if m[pm] is not None and w[pm] is not None else None}
    return {"spec": spec, "features_by_stratum": feats, "baseline": base,
            "n_style_matching": int(correct.sum()), "n_style_misleading": int((~correct).sum()),
            "style_misleading_items": it.item_id[~correct].tolist(), "systems": sysout}


# ------------------------------------------------------------------------------------------------ whole arm
# ------------------------------------------------------------------------------------------------ post hoc
POST_HOC_NOTE = "Post hoc: added after the results were seen; descriptive."


def rates_by_group(t: Table, a: dict, spec: dict, use: np.ndarray | None, idx: np.ndarray) -> dict:
    """Each rate of analysis.rates per group (items.group: arm 2's recommendation) for every system and for the hybrid
    (Jev on the items in `use`, spec['chatbot'] on the rest), as counts; the mean probability each system puts on
    spec['label'] among the spec['among'] items; totals with percentile intervals over the main resamples."""
    rs = a.get("rates") or {}
    names = spec.get("rates") or list(rs)
    grp = t.items["group"].astype(str).to_numpy()
    lab_col = spec.get("label_column")
    labels = dict(zip(grp, t.items[lab_col].astype(str))) if lab_col and lab_col in t.items else {}
    answers = {s: t.answer[:, t.col(s)] for s in t.systems}
    ch = spec.get("chatbot")
    if use is not None and ch in t.systems:
        answers["hybrid"] = np.where(use, answers["jev"], answers[ch])
    li = t.labels.index(spec["label"]) if spec.get("label") else None
    among_p = t.strata == str(spec["among"]) if spec.get("among") else np.ones(len(grp), bool)
    hit = {(s, n): np.isin(ans, [t.labels.index(x) for x in rs[n]["answer_in"]]) for s, ans in answers.items() for n in names}
    within = {n: np.ones(len(grp), bool) if rs[n].get("among") is None else t.strata == str(rs[n]["among"]) for n in names}

    def mean_p(m, s):
        v = t.P[m & among_p, t.col(s), li]
        return float(np.nanmean(v)) if li is not None and (~np.isnan(v)).any() else None

    groups = {}
    for g in sorted(set(grp)):
        m = grp == g
        groups[g] = {"label": labels.get(g),
                     "n_by_stratum": {str(v): int((m & (t.strata == v)).sum()) for v in np.unique(t.strata)},
                     "rates": {s: {n: {"n": int((hit[s, n] & m & within[n]).sum()), "of": int((m & within[n]).sum())}
                                   for n in names} for s in answers},
                     "mean_prob": {s: mean_p(m, s) for s in t.systems} if li is not None else None}
    allm = np.ones(len(grp), bool)
    total = {s: {n: {"n": int((hit[s, n] & within[n]).sum()), "of": int(within[n].sum()),
                     "share": float(hit[s, n][within[n]].mean()) if within[n].any() else None,
                     "ci": percentile_ci(hit[s, n][idx][:, within[n]].mean(axis=1), a["ci"]) if within[n].any() else None}
                 for n in names} for s in answers}
    return {"rates": names, "rate_definitions": {n: rs[n] for n in names}, "chatbot": ch, "label": spec.get("label"),
            "among": spec.get("among"), "hybrid": "Jev on its most confident share (the hybrid rule), the chatbot the rest"
            if "hybrid" in answers else None, "groups": groups, "total": total,
            "mean_prob_total": {s: mean_p(allm, s) for s in t.systems} if li is not None else None}


def cluster_sensitivity(t: Table, a: dict, prim: list[str], seed: int, use: np.ndarray | None = None) -> dict:
    """Jev minus each primary chatbot on the primary metric with whole groups resampled (items.group: the items of one
    recommendation share an option set, so their answers are not independent), the same Holm procedures and margin as the
    main analysis; and the estimate with each group left out in turn. With `use` (the hybrid's split), on the same
    resamples: the hybrid minus each primary chatbot (the same Holm procedures and margin) and confidence-routed minus
    random-routed."""
    pm, B = a["primary_metric"], int(a["n_boot"])
    grp = t.items["group"].astype(str).to_numpy()
    gs = sorted(set(grp))
    members = [np.flatnonzero(grp == g) for g in gs]
    rng = np.random.default_rng([seed, 2])
    cj, n_strata = t.correct[:, t.col("jev")], len(np.unique(t.strata))
    diffs, keep = {s: np.full(B, np.nan) for s in prim}, np.ones(B, bool)
    hyb, rout = {s: np.full(B, np.nan) for s in prim}, {s: np.full(B, np.nan) for s in prim}
    share = float(use.mean()) if use is not None else None
    for b in range(B):
        ix = np.concatenate([members[k] for k in rng.integers(0, len(gs), len(gs))])
        st = t.strata[ix]
        if len(np.unique(st)) < n_strata:
            keep[b] = False
            continue
        mj = accuracy_metric(cj[ix], st, pm)
        for s in prim:
            cs = t.correct[ix, t.col(s)]
            mc = accuracy_metric(cs, st, pm)
            diffs[s][b] = mj - mc
            if use is not None:
                mh = accuracy_metric(np.where(use[ix], cj[ix], cs), st, pm)
                hyb[s][b], rout[s][b] = mh - mc, mh - float(random_routed(cj[ix], cs, share, st, pm))
    diffs = {s: d[keep] for s, d in diffs.items()}
    ho = holm_outcomes(diffs, a["margin"], a["alpha_one_sided"])
    est = {s: accuracy_metric(cj, t.strata, pm) - accuracy_metric(t.correct[:, t.col(s)], t.strata, pm) for s in prim}
    loo = {}
    for g in gs:
        m = grp != g
        loo[g] = {s: accuracy_metric(cj[m], t.strata[m], pm) - accuracy_metric(t.correct[m, t.col(s)], t.strata[m], pm)
                  for s in prim}
    out = {"rule": "whole groups resampled with replacement (as many as there are), their items kept; Holm across the "
                   "primary chatbots and the margin as the main analysis", "n_groups": len(gs),
           "resamples": int(keep.sum()), "resamples_missing_a_stratum": int((~keep).sum()),
           "comparisons": {s: {"diff": est[s], "ci": list(percentile_ci(diffs[s], a["ci"])), **ho[s]} for s in prim},
           "leave_one_group_out": loo}
    if use is not None:
        hyb, rout = {s: d[keep] for s, d in hyb.items()}, {s: d[keep] for s, d in rout.items()}
        hho = holm_outcomes(hyb, a["margin"], a["alpha_one_sided"])
        hyb_est, rout_est = {}, {}
        for s in prim:
            cs = t.correct[:, t.col(s)]
            mh = accuracy_metric(np.where(use, cj, cs), t.strata, pm)
            hyb_est[s] = mh - accuracy_metric(cs, t.strata, pm)
            rout_est[s] = mh - float(random_routed(cj, cs, share, t.strata, pm))
        out["hybrid"] = {"share_jev": share, "n_jev": int(use.sum()),
                         "comparisons": {s: {"diff": hyb_est[s], "ci": list(percentile_ci(hyb[s], a["ci"])), **hho[s]}
                                         for s in prim}}
        out["routing"] = {s: {"diff": rout_est[s], "ci": list(percentile_ci(rout[s], a["ci"]))} for s in prim}
    return out


def certain_errors(t: Table, at: float, prim: list[str]) -> dict:
    """Every item Jev answers with probability 1.00 (top probability at or above `at`) and gets wrong, with each primary
    chatbot's answer: where they give Jev's answer, the key may be disputed."""
    j = t.col("jev")
    top = np.where(t.usable[:, j], t.top[:, j], np.nan)
    lab = lambda k: t.labels[k] if k >= 0 else None
    rows = []
    for i in np.flatnonzero(~np.isnan(top) & (top >= at - 1e-12) & ~t.correct[:, j]):
        ans = {s: lab(t.answer[i, t.col(s)]) for s in prim}
        rows.append({"item_id": str(t.items.item_id.iloc[i]), "stratum": str(t.strata[i]),
                     "group": str(t.items["group"].iloc[i]) if "group" in t.items else None,
                     "jev_answer": lab(t.answer[i, j]), "truth": lab(t.truth[i]), "jev_top": float(top[i]),
                     "chatbots": ans, "chatbots_with_jev": int(sum(v == lab(t.answer[i, j]) for v in ans.values())),
                     "chatbots_with_truth": int(sum(v == lab(t.truth[i]) for v in ans.values()))})
    n_sure = int((~np.isnan(top) & (top >= at - 1e-12)).sum())
    return {"at": at, "n_certain": n_sure, "n_wrong": len(rows), "items": rows}


def by_category(t: Table, a: dict, cats: dict, prim: list[str], seed: int) -> dict:
    """The primary metric per category of an item attribute in the items file (arm 2: clinical domain, evidence grade;
    the second item set: the exam), every system, with percentile intervals from resamples within the category
    (stratified by the truth stratum), and Jev minus each primary chatbot, paired. Descriptive: no test, no adjustment.
    A category without every truth stratum falls back to plain accuracy."""
    pm, out = a["primary_metric"], {}
    for name, cols in cats.items():
        val = t.items[list(cols)].astype(str).apply(lambda r: " ".join(x for x in r if x and x != "nan"), axis=1).to_numpy()
        blk = {}
        for v in sorted(set(val)):
            m = val == v
            st = t.strata[m]
            metric = pm if len(np.unique(st)) == len(np.unique(t.strata)) else "accuracy"
            idx = strat_index(st, int(a["n_boot"]), seed)
            boots, sysb = {}, {}
            for s in t.systems:
                c = t.correct[m, t.col(s)]
                boots[s] = metric_boot(c[idx], st, metric)
                sysb[s] = {"value": accuracy_metric(c, st, metric), "ci": list(percentile_ci(boots[s], a["ci"]))}
            blk[v] = {"n": int(m.sum()), "n_by_stratum": {str(x): int((st == x).sum()) for x in np.unique(st)},
                      "metric": metric, "systems": sysb,
                      "jev_minus": {s: {"diff": sysb["jev"]["value"] - sysb[s]["value"],
                                        "ci": list(percentile_ci(boots["jev"] - boots[s], a["ci"]))} for s in prim}}
        out[name] = {"columns": list(cols), "categories": blk}
    return out


def post_hoc(t: Table, a: dict, prim: list[str], seed: int) -> dict:
    """The post hoc block (config analysis.post_hoc): rates by group, the group-resampled sensitivity, Jev's wrong answers
    at probability 1.00, the metric by category."""
    ph, out = a.get("post_hoc") or {}, {"note": POST_HOC_NOTE}
    hf = a.get("hybrid_share")
    if ph.get("rates_by_group") and "group" in t.items:
        use = hybrid_split(t, float(hf["jev_share"])) if hf else None
        idx = strat_index(t.strata, int(a["n_boot"]), seed)
        out["rates_by_group"] = rates_by_group(t, a, ph["rates_by_group"], use, idx)
    if ph.get("cluster_sensitivity") and "group" in t.items and prim:
        out["cluster_sensitivity"] = cluster_sensitivity(t, a, prim, seed,
                                                         hybrid_split(t, float(hf["jev_share"])) if hf else None)
    if ph.get("certain_errors") and a.get("certain_at") is not None:
        out["certain_errors"] = certain_errors(t, float(a["certain_at"]), prim)
    cats = {k: v for k, v in (ph.get("by_category") or {}).items() if all(c in t.items for c in v)}
    if cats:
        out["by_category"] = by_category(t, a, cats, prim, seed)
    out["confidence_table"] = bool(ph.get("confidence_table"))
    return out


def settings(arm_cfg: dict) -> dict:
    return {"ci": 0.95, "margin": 0.05, "alpha_one_sided": 0.025, "coverages": [0.5, 0.8], "ece_bins": 10,
            "min_errors_auroc": 20, "n_boot": 2000, "cost_basis": "list", **arm_cfg["analysis"]}


def option_count_analysis(calls: pd.DataFrame, frames: dict[str, pd.DataFrame], counts: dict[str, np.ndarray],
                          arm_cfg: dict, labels: tuple[str, ...], systems: list[str], models_cfg: dict) -> dict:
    """option_count_contrast on the main pass of two variants (the same items in the same order); resampling within the
    stratum with the arm seed."""
    a = settings(arm_cfg)
    tables = {v: Table(calls, f, systems, labels, models_cfg, v) for v, f in frames.items()}
    t0 = next(iter(tables.values()))
    idx = strat_index(t0.strata, int(a["n_boot"]), int(arm_cfg["seed"]))
    return clean({"arm": arm_cfg["arm"], "settings": {"n_boot": int(a["n_boot"]), "ci": a["ci"], "seed": int(arm_cfg["seed"]),
                                                      "bootstrap": "paired over questions, stratified by the stratum"},
                  **option_count_contrast(tables, counts, a, idx)})


def analyze(calls: pd.DataFrame, items: pd.DataFrame, arm_cfg: dict, labels: tuple[str, ...], systems: list[str],
            tiers: dict[str, str], models_cfg: dict, rep_ids: list[str] | None = None,
            timing: pd.DataFrame | None = None, variant: str = "main") -> dict:
    """items: item_id, truth (canonical), stratum. tiers: system -> 'jev' | 'primary' | 'free' | 'pair'."""
    a = settings(arm_cfg)
    kind, seed = arm_cfg["kind"], int(arm_cfg["seed"])
    t = Table(calls, items, systems, labels, models_cfg, variant)
    idx = strat_index(t.strata, int(a["n_boot"]), seed)
    out: dict[str, Any] = {
        "arm": arm_cfg["arm"], "kind": kind, "variant": variant, "n_items": len(t.items),
        "primary_metric": a["primary_metric"], "labels": list(labels),
        "strata": {str(k): int(v) for k, v in zip(*np.unique(t.strata, return_counts=True))},
        "settings": {k: a[k] for k in ("n_boot", "ci", "margin", "alpha_one_sided", "coverages", "ece_bins",
                                       "min_errors_auroc", "cost_basis")}
                    | {"seed": seed, "bootstrap": "paired over items, stratified by the truth stratum",
                       "unusable": "not correct; cost of billed tokens counted",
                       "noninferiority": "Holm-adjusted one-sided lower bound across the primary chatbots",
                       "outcome": "non-inferior: Holm-adjusted lower bound above -margin; worse: Holm-adjusted upper "
                                  "bound below -margin; otherwise inconclusive",
                       "acceptable_cheaper": "Holm-adjusted lower bound above -margin and lower list-price cost per "
                                             "correct answer"}
                    | ({"certain_at": float(a["certain_at"])} if a.get("certain_at") is not None else {}),
        "completeness": {s: {"present": int(t.present[:, t.col(s)].sum()),
                             "missing": int((~t.present[:, t.col(s)]).sum())} for s in systems},
        "systems": {}, "comparisons": {}, "value": {}, "hybrid": {}, "acceptable_cheaper": {}}
    speed = timing_summary(timing)
    for s in systems:
        out["systems"][s] = {"tier": tiers.get(s), **system_block(t, s, kind, a, idx), "speed": speed.get(s)}
    chat = [s for s in systems if s != "jev"]
    prim = [s for s in chat if tiers.get(s) == "primary"]
    for s in chat:
        out["comparisons"][s] = {"tier": tiers.get(s), "confirmatory": s in prim, **compare(t, s, a, idx)}
    ph = holm({s: out["comparisons"][s]["p_two_sided"] for s in prim})
    hb = holm_outcomes({s: boot_diff(t, s, a, idx) for s in prim}, a["margin"], a["alpha_one_sided"])
    for s in chat:
        c = out["comparisons"][s]
        c["p_two_sided_holm"] = ph.get(s)
        if s in hb:
            c.update(hb[s], outcome_basis="holm")
        else:                               # outside the primary family: unadjusted bounds, descriptive
            ni, bad = c["lower_bound_unadjusted"] > -a["margin"], c["upper_bound_unadjusted"] < -a["margin"]
            c.update({"holm_rank": None, "holm_level": None, "holm_lower_bound": None, "noninferior": bool(ni),
                      "holm_rank_worse": None, "holm_level_worse": None, "holm_upper_bound": None, "worse": bool(bad),
                      "outcome": outcome(ni, bad), "outcome_basis": "unadjusted (descriptive)"})
        out["value"][s] = value_vs_jev(t, s, idx, a["ci"])
        out["hybrid"][s] = hybrid(t, s, a)
        out["acceptable_cheaper"][s] = acceptable_cheaper(c, out["systems"]["jev"], out["systems"][s], a)
    hf = a.get("hybrid_share")
    if hf and prim:
        use = hybrid_split(t, float(hf["jev_share"]))
        blocks, diffs = {}, {}
        for s in prim:
            blocks[s], diffs[s] = hybrid_share(t, s, a, idx, use)
        ho = holm_outcomes(diffs, a["margin"], a["alpha_one_sided"])
        out["hybrid_share"] = {"rule": {"jev_share": float(hf["jev_share"]), "confidence": "Jev's top probability",
                                        "ties": "item id", "uses_truth": False},
                               "jev_items": t.items.item_id[use].tolist(),
                               "comparisons": {s: {**blocks[s], **ho[s]} for s in prim}}
    # the cost table's hybrid: the set share where the arm has one (arm 2); else the first threshold on the decile grid
    # within the margin of the chatbot alone
    jj = t.col("jev")
    topj = np.where(t.usable[:, jj], t.top[:, jj], np.nan)
    out["hybrid_cost"] = {}
    for s in chat:
        if hf:
            use, rule = hybrid_split(t, float(hf["jev_share"])), f"Jev on its most confident {float(hf['jev_share']):.0%}"
        else:
            thr = out["hybrid"][s]["first_within_margin"]["threshold"]
            use = ~np.isnan(topj) & (topj >= (math.inf if thr is None else thr))
            rule = "first decile of Jev's top probability within the margin of the chatbot alone"
        out["hybrid_cost"][s] = hybrid_point(t, s, a, use, rule)
    # random-half control and the share curve (as arm 3's supplement): the hybrid's share, else 50%
    share = float(hf["jev_share"]) if hf else 0.5
    use = hybrid_split(t, share)
    shares = [float(c) for c in a.get("coverage_curve", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]
    out["routing"] = {"rule": {"jev_share": share, "confidence_routed": "Jev on its highest top probabilities, ties by "
                               "item id", "random_routed": "the same share handed to Jev at random: the expectation over "
                               "every set of that size (exact, no seed)", "interval": "paired over the resampled items"},
                      "systems": {s: routing_control(t, s, a, use, idx) for s in chat}}
    out["coverage_curve"] = {"shares": shares, "systems": {s: coverage_curve(t, s, a, shares) for s in chat}}
    if rep_ids:
        out["repeatability"] = repeatability(calls, t.items, rep_ids, systems, labels, kind,
                                             int(arm_cfg["repeats"]["n_repeats"]), int(a["n_boot"]), seed, a["ci"],
                                             variant)
    if a.get("style_check"):
        out["style_check"] = style_check(t, a, idx, seed)
    if a.get("post_hoc"):
        out["post_hoc"] = post_hoc(t, a, prim, seed)
    return clean(out)


def clean(o: Any) -> Any:
    """JSON-safe: numpy scalars to Python, NaN and inf to None, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------------------------------------ supplement tables
TIER_ORDER = {"jev": 0, "primary": 1, "free": 2, "pair": 3}
METRIC_NAME = {"balanced_accuracy": "balanced accuracy", "exact_accuracy": "exact accuracy", "accuracy": "accuracy"}
WORD = {2: "two", 3: "three", 4: "four", 5: "five"}


def _ci(v, ci) -> tuple:
    return (v, *(ci if ci else (None, None)))


def supplement_tables(res: dict) -> dict[str, dict]:
    """The supplement tables of one arm and version, from its analysis result only (every number
    traces to the results JSON). Repeatability; cost per answer and speed; cost per correct answer, against Jev and in the
    hybrid. Each table: title, columns (header, format), one row per system (tiers in order: Jev, primary, free, pair)
    with values or (estimate, low, high), and notes."""
    given = list(res["systems"])
    ss = sorted(given, key=lambda s: (TIER_ORDER.get(res["systems"][s]["tier"], 9), given.index(s)))
    pm, n_boot, lvl = res["primary_metric"], res["settings"]["n_boot"], res["settings"]["ci"]
    arm = _arm_title(res)
    out = {}

    def rows(cols, skip_jev=()):
        return [{"system": s, "tier": res["systems"][s]["tier"],
                 **{h: (None if s == "jev" and h in skip_jev else f(s)) for h, _, f in cols}} for s in ss]

    rep = res.get("repeatability")
    if rep:
        name = next(iter(rep.values()))["measure"]
        head = "Fleiss' kappa" if name == "fleiss_kappa" else "ICC(2,1) of the level index"
        cols = [("Items, every answer usable", "int", lambda s: rep[s]["items_all_usable"]),
                ("Answered identically", "share", lambda s: _ci(rep[s]["identical_share"], rep[s]["identical_share_ci"])),
                (head, "coef", lambda s: _ci(rep[s][name], rep[s][f"{name}_ci"]))]
        r0 = next(iter(rep.values()))
        out["repeatability"] = {
            "title": f"Repeatability on the repeat set, {arm}", "columns": [(h, k) for h, k, _ in cols], "rows": rows(cols),
            "notes": [f"{r0['items']} items asked {WORD.get(r0['repeats'], r0['repeats'])} times (the first answer is "
                      "the main pass); items with a usable answer every time enter.",
                      "Answered identically: share of those items with the same answer every time. "
                      + ("Fleiss' kappa across the answers." if name == "fleiss_kappa" else
                         "ICC(2,1): two-way random effects, absolute agreement, one rating, on the answer's position "
                         "on the ordered scale."),
                      f"{lvl:.0%} percentile intervals from {n_boot:,} bootstrap resamples of items within the truth "
                      "stratum."]}
    c = lambda s: res["systems"][s]["cost"]
    sp = lambda s: res["systems"][s].get("speed") or {}
    cols = [("Input tokens per answer", "tok", lambda s: c(s)["tokens_in_per_answer"]),
            ("Output tokens per answer", "tok", lambda s: c(s)["tokens_out_per_answer"]),
            ("of which reasoning", "tok", lambda s: c(s)["tokens_reasoning_per_answer"]),
            ("Billed US$ per 1,000 answers", "usd", lambda s: c(s)["usd_billed_per_1000_answers"]),
            ("List US$ per 1,000 answers", "usd", lambda s: c(s)["usd_list_per_1000_answers"]),
            ("List US$ per million answers", "usd", lambda s: c(s)["usd_list_per_million_answers"]),
            ("Median seconds per answer", "sec", lambda s: sp(s).get("median_s")),
            ("90th percentile seconds", "sec", lambda s: sp(s).get("p90_s"))]
    timed = any(sp(s).get("median_s") is not None for s in ss)
    wins: dict[str, list[str]] = {}
    for s in ss:
        if sp(s).get("window_start"):
            wins.setdefault(sp(s)["window_start"], []).append(s)
    if not timed:
        sec = "Seconds: the timing sample has not run (dashes)."
    elif len(wins) <= 1:
        sec = "Seconds: the timing sample, every system answering the same items one at a time within one window."
    else:
        sec = ("Seconds: the timing sample, every system answering the same items one at a time, in "
               f"{WORD.get(len(wins), len(wins))} windows (start, local time): "
               + "; ".join(f"{w.replace('T', ' ')[:16]}, {', '.join(v)}" for w, v in sorted(wins.items())) + ".")
    out["cost"] = {
        "title": f"Tokens, cost and speed per answer, {arm}", "columns": [(h, k) for h, k, _ in cols], "rows": rows(cols),
        "notes": [f"Means over the {res['n_items']} main-pass answers, unusable ones included, from the tokens each "
                  "provider reported; reasoning tokens are part of the output tokens (a dash: none reported). Jev's "
                  "output is free.",
                  "List: the developer's list price; billed: half the list price on the batch route, the list price on the "
                  "standard route (from the reported tokens; a provider's own charge can be slightly lower with prompt "
                  "caching).",
                  sec]}
    v, hc = res.get("value") or {}, res.get("hybrid_cost") or {}

    def extra(s):
        x = v[s]
        if not x["chatbot_more_accurate"]:
            return x["verdict_list"]
        return _ci(x["list"]["usd_per_extra_correct"], x["list"]["usd_per_extra_correct_ci"])

    cols = [("List US$ per correct answer", "usd", lambda s: _ci(c(s)["usd_list_per_correct"], c(s)["usd_list_per_correct_ci"])),
            ("Extra list US$ per extra correct answer against Jev", "usd", extra),
            ("Hybrid: share answered by Jev", "pct", lambda s: hc[s]["share_jev"]),
            (f"Hybrid {METRIC_NAME.get(pm, pm)}", "share", lambda s: hc[s][pm]),
            (f"Chatbot alone {METRIC_NAME.get(pm, pm)}", "share", lambda s: hc[s]["chatbot_alone"]),
            ("Hybrid list US$ per 1,000 answers", "usd", lambda s: hc[s]["hybrid_usd_per_1000_items"]),
            ("Saving against the chatbot alone, US$ per 1,000 answers", "usd", lambda s: hc[s]["saving_usd_per_1000_items"]),
            ("Saving, share of the chatbot's cost", "pct", lambda s: hc[s]["saving_share"])]
    rule = next((h["rule"] for h in hc.values()), None)
    out["value"] = {
        "title": f"Cost per correct answer, against Jev and in the hybrid, {arm}",
        "columns": [(h, k) for h, k, _ in cols], "rows": rows(cols, skip_jev=[h for h, _, _ in cols[1:]]),
        "notes": [f"List prices. Cost per correct answer: total cost over correct answers ({lvl:.0%} percentile "
                  f"interval, {n_boot:,} bootstrap resamples of items within the truth stratum).",
                  "Against Jev: the difference in total cost over the difference in correct answers, reported only "
                  "where the chatbot has more correct answers than Jev (interval over the resamples where it does).",
                  f"Hybrid: {rule}; Jev asked on every item, the chatbot only on the rest. No extra calls."
                  if rule else "Hybrid: none."]}
    ro = (res.get("routing") or {}).get("systems") or {}
    if ro:
        names = _routing_metrics(pm)
        cols = [("Share answered by Jev", "pct", lambda s: ro[s]["share_jev"])]
        for name in names:
            lab = METRIC_NAME.get(name, name)
            cols += [(f"{lab[0].upper() + lab[1:]}, confidence-routed", "share", lambda s, k=name: ro[s][k]["confidence_routed"]),
                     (f"{lab[0].upper() + lab[1:]}, random-routed", "share", lambda s, k=name: ro[s][k]["random_routed"]),
                     (f"Confidence minus random, {lab}", "share",
                      lambda s, k=name: _ci(ro[s][k]["diff"], ro[s][k]["diff_ci"]))]
        out["routing"] = {
            "title": f"Routing by Jev's confidence against routing at random, {arm}",
            "columns": [(h, k) for h, k, _ in cols], "rows": rows(cols, skip_jev=[h for h, _, _ in cols]),
            "notes": [f"Jev answers the {res['routing']['rule']['jev_share']:.0%} of items with its highest top probability "
                      "(ties by item id), the chatbot the rest; random-routed: the same share handed to Jev at random, the "
                      "expectation over every set of that size (exact, no seed).",
                      f"Difference: confidence-routed minus random-routed; {lvl:.0%} percentile interval, {n_boot:,} "
                      "bootstrap resamples of items within the truth stratum, paired."]}
    cc, hsh = res.get("coverage_curve"), res.get("hybrid_share")
    if cc and hsh:                                  # arm 2: the share curve beside the set share
        lab = METRIC_NAME.get(pm, pm)
        heads = [("Items answered by Jev", "int"), (f"{lab[0].upper() + lab[1:]}, Jev's most confident share first", "share"),
                 (f"{lab[0].upper() + lab[1:]}, the same share at random", "share"), ("Chatbot alone", "share"),
                 ("Confidence-routed minus chatbot alone", "dec3"), ("List US$ per 1,000 answers", "usd"),
                 ("Saving against the chatbot alone", "pct")]
        rows = []
        for s in [x for x in ss if res["systems"][x]["tier"] == "primary" and x in cc["systems"]]:
            alone, k_alone = res["systems"][s][pm], c(s)["usd_list_per_1000_answers"]
            for p in cc["systems"][s]:
                m, k = p[f"{pm}_confidence_routed"], p["usd_list_per_1000_confidence_routed"]
                vals = [p["n_jev"], m, p[f"{pm}_random_routed"], alone, m - alone, k, 1 - k / k_alone if k_alone else None]
                rows.append({"system": s, "tier": f"{p['share']:.0%}", **{h: v for (h, _), v in zip(heads, vals)}})
        out["coverage_curve"] = {
            "title": f"Jev answers a growing share, its most confident first; the chatbot the rest, {arm}",
            "row_heads": ["Chatbot", "Share answered by Jev"], "columns": heads, "rows": rows,
            "notes": [f"Jev answers the share of items with its highest top probability (ties by item id), the chatbot the "
                      "rest; random: the same share handed to Jev at random (the exact expectation). Point estimates; "
                      f"list prices, Jev asked on every item. Only the {hsh['rule']['jev_share']:.0%} share is tested (the hybrid "
                      "table); the other shares are descriptive."]}
    return out


def _arm_title(res: dict) -> str:
    """"arm 2", "arm 3, definitions version"; the second item set in words ("arm 2, second item set, MedHELM option
    counts")."""
    arm = re.sub(r"^arm(\d)", r"arm \1", str(res["arm"]))
    variant = str(res.get("variant", "main"))
    if "_ext" not in arm:
        return f"{arm}, {variant} version" if variant != "main" else arm
    arm = arm.replace("_ext", ", second item set")
    for a, b in (("medhelm", "MedHELM option counts"), ("other_count", "other option counts"),
                 ("decision subset", "decision subgroup")):
        variant = variant.replace(a, b)
    return f"{arm}, {variant}" if variant != "main" else arm


def result_tables(res: dict) -> dict[str, dict]:
    """Arm 2's second item set: the results as tables, from the analysis result only. Accuracy per
    system, overall and per stratum, Jev minus the system with its outcome, and the items answered with probability 1.00;
    the hybrid at the set share against each primary chatbot with the Holm bounds and the list cost."""
    given = list(res["systems"])
    ss = sorted(given, key=lambda s: (TIER_ORDER.get(res["systems"][s]["tier"], 9), given.index(s)))
    pm, n_boot, lvl = res["primary_metric"], res["settings"]["n_boot"], res["settings"]["ci"]
    arm, name = _arm_title(res), METRIC_NAME.get(pm, pm)
    sy, cmp = res["systems"], res["comparisons"]
    strata = list(res["strata"])
    out = {}
    cols = [(name[0].upper() + name[1:], "share", lambda s: _ci(sy[s][pm], sy[s][f"{pm}_ci"]))]
    for v in strata:
        cols.append((f"{v} ({res['strata'][v]:,})", "share",
                     lambda s, v=v: _ci(sy[s]["accuracy_by_stratum"][v], (sy[s].get("accuracy_by_stratum_ci") or {}).get(v))))
    cols += [("Unusable answers", "int", lambda s: sy[s]["unusable"]),
             ("Jev minus system", "share", lambda s: None if s == "jev" else _ci(cmp[s]["diff"], cmp[s]["ci"])),
             ("Outcome", "text", lambda s: None if s == "jev" else cmp[s]["outcome"]
              + ("" if cmp[s]["confirmatory"] else " (descriptive)"))]
    if all("certain" in sy[s] for s in ss):
        ce = lambda s: sy[s]["certain"]
        cols += [("Share answered with probability 1.00", "pct", lambda s: ce(s)["share"]),
                 ("Accuracy at 1.00", "share", lambda s: _ci(ce(s)["accuracy"], ce(s)["accuracy_ci"])),
                 ("Accuracy below 1.00", "share", lambda s: _ci(ce(s)["accuracy_below"], ce(s)["accuracy_below_ci"]))]
    notes = [f"{name[0].upper() + name[1:]} on all {res['n_items']:,} items (an unusable answer counts as wrong); by stratum "
             "in the next columns. " f"{lvl:.0%} percentile intervals, {n_boot:,} bootstrap resamples of items within the stratum.",
             f"Jev minus system: paired over the same items; outcome against a {res['settings']['margin']:.0%} margin, Holm-adjusted "
             "across the primary chatbots (non-inferior, worse or inconclusive); other tiers descriptive, unadjusted."]
    if "certain_at" in res["settings"]:
        notes.append(f"Probability 1.00: a top probability of at least {res['settings']['certain_at']} (1.00 at two decimals), "
                     "with a usable answer; accuracy at 1.00 and on every other item.")
    refused = {s: sy[s].get("refused_by_provider", 0) for s in ss if sy[s].get("refused_by_provider")}
    if refused:
        notes.append("Unusable answers include replies the provider's own safety filter withheld (finish reason "
                     "content_filter): " + ", ".join(f"{s} {k}" for s, k in refused.items()) + ".")
    out["accuracy"] = {"title": f"Accuracy, {arm}", "columns": [(h, k) for h, k, _ in cols],
                       "rows": [{"system": s, "tier": sy[s]["tier"], **{h: f(s) for h, _, f in cols}} for s in ss],
                       "notes": notes}
    hs = (res.get("hybrid_share") or {}).get("comparisons") or {}
    if hs:
        cols = [("Hybrid", "share", lambda s: _ci(hs[s][pm], hs[s][f"{pm}_ci"])),
                ("Chatbot alone", "share", lambda s: hs[s]["chatbot_alone"]),
                ("Hybrid minus chatbot", "share", lambda s: _ci(hs[s]["diff"], hs[s]["ci"])),
                ("Holm lower bound", "share", lambda s: hs[s]["holm_lower_bound"]),
                ("Holm upper bound", "share", lambda s: hs[s]["holm_upper_bound"]),
                ("Outcome", "text", lambda s: hs[s]["outcome"]),
                ("Hybrid list US$ per 1,000", "usd", lambda s: hs[s]["cost_list"]["hybrid_usd_per_1000_items"]),
                ("Chatbot list US$ per 1,000", "usd", lambda s: hs[s]["cost_list"]["chatbot_usd_per_1000_items"]),
                ("Saving", "pct", lambda s: hs[s]["cost_list"]["saving_share"])]
        r = res["hybrid_share"]["rule"]
        n_jev = next(iter(hs.values()))["n_jev"]
        out["hybrid"] = {"title": f"Hybrid against each primary chatbot, {arm}", "columns": [(h, k) for h, k, _ in cols],
                         "rows": [{"system": s, "tier": sy[s]["tier"], **{h: f(s) for h, _, f in cols}} for s in hs],
                         "notes": [f"Hybrid: Jev answers its most confident {r['jev_share']:.0%} ({n_jev:,} of {res['n_items']:,} "
                                   "items; its top probability, ties by item id; the truth never used), the chatbot the rest.",
                                   f"Hybrid minus chatbot alone, paired; {lvl:.0%} percentile interval; Holm bounds at one-sided "
                                   f"{res['settings']['alpha_one_sided']} across the primary chatbots against a "
                                   f"{res['settings']['margin']:.0%} margin. List prices; Jev asked on every item."]}
    return out


def option_count_table(res: dict) -> dict:
    """The option-count contrast (option_count_analysis) as a table."""
    sy, jm = res["systems"], res.get("jev_minus_chatbot") or {}
    given = list(sy)
    strata = list(next(iter(sy.values()))["diff_by_stratum"])
    obs = res["options_by_stratum"]
    va, vb = res["variants"]
    cols = [("Accuracy, longer list", "share", lambda s: sy[s]["accuracy_longer"]),
            ("Accuracy, shorter list", "share", lambda s: sy[s]["accuracy_shorter"]),
            ("Longer minus shorter", "share", lambda s: _ci(sy[s]["diff"], sy[s]["diff_ci"]))]
    for v in strata:
        k = sorted({*obs[v][va], *obs[v][vb]})
        cols.append((f"{v} ({k[-1]} minus {k[0]} options)", "share",
                     lambda s, v=v: _ci(sy[s]["diff_by_stratum"][v]["diff"], sy[s]["diff_by_stratum"][v]["ci"])))
    cols += [("Mean top probability, longer / shorter", "text",
              lambda s: None if sy[s]["mean_top_longer"] is None else f"{sy[s]['mean_top_longer']:.2f} / {sy[s]['mean_top_shorter']:.2f}"),
             ("Jev's change minus the system's", "share", lambda s: _ci(jm[s]["diff"], jm[s]["ci"]) if s in jm else None)]
    return {"title": f"Accuracy by number of options, same questions, {_arm_title({'arm': res['arm']})}",
            "columns": [(h, k) for h, k, _ in cols],
            "rows": [{"system": s, "tier": "", **{h: f(s) for h, _, f in cols}} for s in given],
            "notes": [f"The same {res['n_items']:,} questions asked with two option lists (the shorter holds the longer one's "
                      "options less one wrong option; MedQA keeps their order, Medbullets' 4-option source file reorders "
                      "them on most questions); longer minus shorter, paired over the questions, "
                      f"{res['settings']['ci']:.0%} percentile interval, {res['settings']['n_boot']:,} bootstrap resamples within "
                      "the stratum. Descriptive; an unusable answer counts as wrong.",
                      "Jev's change minus the system's: does one more option cost Jev more than the system (negative) or less?"]}


def _fmt(x, kind: str) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if isinstance(x, str) or kind == "text":
        return str(x)
    if isinstance(x, tuple):
        est, lo, hi = x
        return _fmt(est, kind) + ("" if lo is None else f" ({_fmt(lo, kind)} to {_fmt(hi, kind)})")
    if kind == "int":
        return f"{int(x):,}"
    if kind == "tok":
        return f"{x:,.0f}" if abs(x) >= 10 or x == 0 else f"{x:.1f}"
    if kind == "usd":                       # two significant figures below 1, never an exponent
        a = abs(x)
        if a >= 1 or a == 0:
            return f"{x:,.0f}" if a >= 100 else f"{x:.1f}" if a >= 10 else f"{x:.2f}"
        return f"{x:.{max(2, 1 - math.floor(math.log10(a)))}f}"
    if kind == "pct":
        return f"{x:.0%}"
    if kind == "sec":
        return f"{x:.1f}"
    if kind == "dec3":
        return f"{x:.3f}"
    return f"{x:.2f}"


def _count(r: dict | None) -> str | None:
    return None if r is None else f"{r['n']} of {r['of']}"


def _count_ci(r: dict) -> str:
    s = f"{r['n']} of {r['of']}"
    if r.get("share") is None:
        return s
    ci = r.get("ci")
    return s + f" ({r['share']:.0%}" + (f"; {ci[0]:.0%} to {ci[1]:.0%})" if ci else ")")


def post_hoc_tables(res: dict) -> dict[str, dict]:
    """Tables of the post hoc block, from the analysis result only; each note says post hoc."""
    ph = res.get("post_hoc")
    if not ph:
        return {}
    arm, sy, out = _arm_title(res), res["systems"], {}
    given = list(sy)
    ss = sorted(given, key=lambda s: (TIER_ORDER.get(sy[s]["tier"], 9), given.index(s)))
    prim = [s for s in ss if sy[s]["tier"] == "primary"]
    pm, lvl, n_boot = res["primary_metric"], res["settings"]["ci"], res["settings"]["n_boot"]
    mname = METRIC_NAME.get(pm, pm)
    word = lambda n: n.replace("_", " ")
    rg = ph.get("rates_by_group")
    if rg:
        ch, others = rg["chatbot"], [s for s in prim if s != rg["chatbot"]]
        strata_of = {n: rg["rate_definitions"][n].get("among") for n in rg["rates"]}
        cols = []
        for n in rg["rates"]:
            among = strata_of[n]
            cols.append((f"{word(among) if among else 'All'} items", "int", ("n", among)))
            cols.append((f"Jev {word(n)}", "text", ("rate", "jev", n)))
            if n == rg["rates"][0] and rg.get("mean_prob_total"):
                cols.append((f"Jev mean probability on {rg['label']} ({word(rg['among'])})", "share", ("prob", "jev")))
            if rg.get("hybrid"):
                cols.append((f"Hybrid {word(n)} (Jev, then {ch})", "text", ("rate", "hybrid", n)))
            cols.append((f"{ch} {word(n)}", "text", ("rate", ch, n)))
            cols.append((f"Other primary chatbots, {word(n)}", "text", ("others", n)))

        def cell(g, spec, total=False):
            kind = spec[0]
            if kind == "n":
                return (sum(rg["groups"][x]["n_by_stratum"].get(spec[1], 0) for x in rg["groups"]) if total
                        else g["n_by_stratum"].get(spec[1], 0)) if spec[1] else None
            if kind == "rate":
                return _count_ci(rg["total"][spec[1]][spec[2]]) if total else _count(g["rates"][spec[1]][spec[2]])
            if kind == "prob":
                return rg["mean_prob_total"][spec[1]] if total else (g["mean_prob"] or {}).get(spec[1])
            src = (lambda s: rg["total"][s][spec[1]]) if total else (lambda s: g["rates"][s][spec[1]])
            return ", ".join(f"{s} {src(s)['n']}" for s in others)

        rows = [{"system": k, "tier": g["label"] or "", **{h: cell(g, sp) for h, _, sp in cols}} for k, g in rg["groups"].items()]
        rows.append({"system": "All", "tier": "", **{h: cell(None, sp, True) for h, _, sp in cols}})
        defs = "; ".join(f"{word(n)}: answer {', '.join(d['answer_in'])}" + (f" on {'an' if word(d['among'])[:1] in 'aeiou' else 'a'} {word(d['among'])} item" if d.get("among") else "")
                         for n, d in rg["rate_definitions"].items())
        out["rates_by_group"] = {
            "title": f"Overuse and false restraint by recommendation, {arm}", "row_heads": ["Recommendation", "Intervention"],
            "columns": [(h, k) for h, k, _ in cols], "rows": rows,
            "notes": [f"Counts: {defs}. An unusable answer counts in the denominator only. All row: share with a {lvl:.0%} "
                      f"percentile interval ({n_boot:,} resamples of items within the truth stratum).",
                      "Hybrid: Jev on its most confident half (the hybrid rule), the chatbot on the rest.",
                      POST_HOC_NOTE]}
    cs = ph.get("cluster_sensitivity")
    if cs:
        cmp = res["comparisons"]
        cols = [("Jev minus system", "share", lambda s: _ci(cmp[s]["diff"], cmp[s]["ci"])),
                ("Outcome, items resampled (main analysis)", "text", lambda s: cmp[s]["outcome"]),
                ("Interval, recommendations resampled", "text",
                 lambda s: "{:.2f} to {:.2f}".format(*cs["comparisons"][s]["ci"])),
                ("Holm lower bound", "share", lambda s: cs["comparisons"][s]["holm_lower_bound"]),
                ("Holm upper bound", "share", lambda s: cs["comparisons"][s]["holm_upper_bound"]),
                ("Outcome, recommendations resampled", "text", lambda s: cs["comparisons"][s]["outcome"])]
        out["cluster_sensitivity"] = {
            "title": f"Jev minus each primary chatbot with whole recommendations resampled, {arm}",
            "columns": [(h, k) for h, k, _ in cols],
            "rows": [{"system": s, "tier": sy[s]["tier"], **{h: f(s) for h, _, f in cols}} for s in prim],
            "notes": [f"{mname[0].upper() + mname[1:]}. The {cs['n_groups']} recommendations resampled with replacement, their "
                      f"items kept ({cs['resamples']:,} resamples): the messages of one recommendation share an option set, "
                      "so their answers are not independent. Holm procedures and the "
                      f"{res['settings']['margin']:.0%} margin as the main analysis (non-inferior, worse or inconclusive).",
                      POST_HOC_NOTE]}
        hyc = cs.get("hybrid")
        if hyc:
            hs, ro = res["hybrid_share"]["comparisons"], res["routing"]["systems"]
            cols = [("Hybrid minus system", "share", lambda s: _ci(hs[s]["diff"], hs[s]["ci"])),
                    ("Outcome, items resampled (main analysis)", "text", lambda s: hs[s]["outcome"]),
                    ("Hybrid interval, recommendations resampled", "text",
                     lambda s: "{:.3f} to {:.3f}".format(*hyc["comparisons"][s]["ci"])),
                    ("Holm lower bound", "dec3", lambda s: hyc["comparisons"][s]["holm_lower_bound"]),
                    ("Outcome, recommendations resampled", "text", lambda s: hyc["comparisons"][s]["outcome"]),
                    ("Confidence minus random routing", "share", lambda s: _ci(ro[s][pm]["diff"], ro[s][pm]["diff_ci"])),
                    ("Routing interval, recommendations resampled", "text",
                     lambda s: "{:.3f} to {:.3f}".format(*cs["routing"][s]["ci"]))]
            out["hybrid_cluster_sensitivity"] = {
                "title": f"The hybrid and confidence routing with whole recommendations resampled, {arm}",
                "columns": [(h, k) for h, k, _ in cols],
                "rows": [{"system": s, "tier": sy[s]["tier"], **{h: f(s) for h, _, f in cols}} for s in prim],
                "notes": [f"{mname[0].upper() + mname[1:]}. Hybrid: Jev on its most confident {hyc['share_jev']:.0%} "
                          f"({hyc['n_jev']} items; the hybrid rule), the chatbot the rest, minus the chatbot alone. The "
                          f"{cs['n_groups']} recommendations resampled with replacement, their items kept "
                          f"({cs['resamples']:,} resamples, the same as the table above); Holm procedures and the "
                          f"{res['settings']['margin']:.0%} margin as the main analysis. Routing: the hybrid minus the same "
                          "share handed to Jev at random (the exact expectation).",
                          POST_HOC_NOTE]}
        loo = cs["leave_one_group_out"]
        labels = {k: g["label"] for k, g in (rg or {}).get("groups", {}).items()}
        cols = [(s, "share", s) for s in prim]
        rows = [{"system": g, "tier": labels.get(g) or "", **{s: v[s] for s in prim}} for g, v in loo.items()]
        rows.append({"system": "None left out", "tier": "", **{s: cmp[s]["diff"] for s in prim}})
        out["leave_one_out"] = {
            "title": f"Jev minus each primary chatbot with one recommendation left out, {arm}",
            "row_heads": ["Recommendation left out", "Intervention"], "columns": [(h, k) for h, k, _ in cols], "rows": rows,
            "notes": [f"{mname[0].upper() + mname[1:]}, point estimates.", POST_HOC_NOTE]}
    ce = ph.get("certain_errors")
    if ce:
        cols = [("Group", "text", lambda r: r["group"]), ("Jev's answer", "text", lambda r: r["jev_answer"]),
                ("Key", "text", lambda r: r["truth"]), ("Jev's top probability", "text", lambda r: f"{r['jev_top']:.3f}"),
                ("Primary chatbots with Jev's answer", "text", lambda r: f"{r['chatbots_with_jev']} of {len(r['chatbots'])}"),
                ("Primary chatbots with the key", "text", lambda r: f"{r['chatbots_with_truth']} of {len(r['chatbots'])}")]
        out["certain_errors"] = {
            "title": f"Jev's wrong answers at probability 1.00, {arm}", "row_heads": ["Item", "Stratum"],
            "columns": [(h, k) for h, k, _ in cols],
            "rows": [{"system": r["item_id"], "tier": r["stratum"], **{h: f(r) for h, _, f in cols}} for r in ce["items"]],
            "notes": [f"{ce['n_wrong']} wrong of the {ce['n_certain']:,} items Jev answered with a top probability of at least "
                      f"{ce['at']} (1.00 at two decimals). Where the primary chatbots give Jev's answer, the key may be "
                      "disputed; read against the key by a physician.", POST_HOC_NOTE]}
    if ph.get("confidence_table"):
        cf = lambda s: sy[s]["confidence"]
        ps = "brier" if any("brier" in cf(s) for s in ss) else "rps"
        cols = [("Brier score (multiclass)" if ps == "brier" else "Ranked probability score", "dec3", lambda s: cf(s).get(ps)),
                ("Calibration error (ECE)", "dec3", lambda s: cf(s)["ece"]),
                ("AUROC, confidence for correctness", "dec3", lambda s: cf(s)["auroc"]),
                ("Area under the risk-coverage curve", "dec3", lambda s: cf(s)["aurc"]),
                ("Accuracy, most confident 50%", "share", lambda s: cf(s)["selective_accuracy"].get("0.5")),
                ("Accuracy, most confident 80%", "share", lambda s: cf(s)["selective_accuracy"].get("0.8"))]
        if all("certain" in sy[s] for s in ss):
            cols += [("Share at probability 1.00", "pct", lambda s: sy[s]["certain"]["share"]),
                     ("Accuracy at 1.00", "share", lambda s: _ci(sy[s]["certain"]["accuracy"], sy[s]["certain"]["accuracy_ci"]))]
        out["confidence"] = {
            "title": f"Confidence, {arm}", "columns": [(h, k) for h, k, _ in cols],
            "rows": [{"system": s, "tier": sy[s]["tier"], **{h: f(s) for h, _, f in cols}} for s in ss],
            "notes": ["Brier score: squared error of the probability list against the answer, summed over the options and "
                      "averaged over answers (0 best, 2 worst; it penalises wrong answers as well as miscalibration). "
                      "Calibration error: mean gap between the top probability and accuracy over "
                      f"{res['settings']['ece_bins']} equal-count bins (tied probabilities fill the bins in item-id order, "
                      "so for a system with many tied probabilities it depends on that order). AUROC: how well the "
                      f"top probability separates right from wrong answers (a dash: fewer than "
                      f"{res['settings']['min_errors_auroc']} errors). Risk-coverage area: lower is better.",
                      "Accuracy on each system's own most confident items; ties broken at random in expectation.",
                      "Unusable answers carry no probability list and are left out of the columns above; the share at "
                      "probability 1.00 is of every item.",
                      POST_HOC_NOTE]}
    for name, bc in (ph.get("by_category") or {}).items():
        cats = bc["categories"]
        cols = []
        for v, blk in cats.items():
            cols.append((f"{v} ({blk['n']:,})", "share", lambda s, b=blk: _ci(b["systems"][s]["value"], b["systems"][s]["ci"])))
            cols.append((f"{v}: Jev minus system", "share",
                         lambda s, b=blk: _ci(b["jev_minus"][s]["diff"], b["jev_minus"][s]["ci"]) if s in b["jev_minus"] else None))
        metrics = sorted({b["metric"] for b in cats.values()})
        out[f"by_{name}"] = {
            "title": f"{', '.join(METRIC_NAME.get(m, m) for m in metrics).capitalize()} by {word(name)}, {arm}",
            "columns": [(h, k) for h, k, _ in cols],
            "rows": [{"system": s, "tier": sy[s]["tier"], **{h: f(s) for h, _, f in cols}} for s in ss],
            "notes": [f"Categories from the items file ({', '.join(bc['columns'])}). {lvl:.0%} percentile "
                      f"intervals, {n_boot:,} resamples of items within the category and truth stratum; Jev minus system "
                      "paired, for the primary chatbots. Descriptive: no test, no adjustment.", POST_HOC_NOTE]}
    return out


def table_frame(tab: dict) -> pd.DataFrame:
    """Numbers as stored (an interval becomes three columns: estimate, low, high)."""
    h0, h1 = [h.lower() for h in tab.get("row_heads", ["system", "tier"])]
    recs = []
    for r in tab["rows"]:
        d = {h0: r["system"], h1: r["tier"]}
        for h, _ in tab["columns"]:
            x = r[h]
            if isinstance(x, tuple):
                d[h], d[f"{h} (low)"], d[f"{h} (high)"] = x
            else:
                d[h] = x
        recs.append(d)
    return pd.DataFrame(recs)


def table_markdown(tab: dict) -> str:
    heads = list(tab.get("row_heads", ["System", "Tier"])) + [h for h, _ in tab["columns"]]
    lines = [f"**{tab['title']}**", "", "| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for r in tab["rows"]:
        cells = [r["system"], r["tier"]] + [_fmt(r[h], k) for h, k in tab["columns"]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines + [""] + tab["notes"]) + "\n"

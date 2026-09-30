"""Statistics for scripts/gapfill/analyze.py. The paired DeLong test for two correlated AUROCs on the same records (fast
DeLong: Sun and Xu, IEEE Signal Process Lett 2014;21:1389-93; midranks, so a tie counts one half), and a descriptive
version of the arm 1 hybrid: jevity.hybrid.hybrid_block's routing, clipped log-loss and record resamples, with no
non-inferiority margin, plus AUROC. Pure functions."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

from jevity.analysis import CLIP, _ci, _ll
from jevity.eicu_analysis import metrics
from jevity.hybrid import SHARES, jev_split

Z95 = float(norm.ppf(0.975))


def delong_components(y, P) -> tuple[np.ndarray, np.ndarray]:
    """AUROC of each row of P (k predictors x n records) and their k x k DeLong covariance matrix, by Sun and Xu's
    midrank algorithm: placement values V10 (per death) and V01 (per survivor), S = cov(V10)/m + cov(V01)/n with
    ddof 1. Raises ValueError unless y is 0/1 with at least two records of each class and P is finite."""
    y = np.asarray(y, float)
    P = np.atleast_2d(np.asarray(P, float))
    if P.shape[1] != len(y) or not np.isfinite(P).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("y must be 0/1 and P finite, one column per record")
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())
    if m == 0 or n == 0:
        raise ValueError("y has one class: AUROC is undefined")
    if m < 2 or n < 2:
        raise ValueError("DeLong variance needs at least two records of each class")
    X, Y = P[:, pos], P[:, ~pos]
    tx, ty, tz = rankdata(X, axis=1), rankdata(Y, axis=1), rankdata(np.hstack([X, Y]), axis=1)
    auc = (tz[:, :m].sum(1) - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:, :m] - tx) / n                    # share of survivors each death outranks (ties one half)
    v01 = 1 - (tz[:, m:] - ty) / m                # share of deaths that outrank each survivor
    S = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
    return auc, S


def delong_ci(y, p) -> list[float]:
    """95% interval for one AUROC: normal approximation with the DeLong variance, truncated to [0, 1]."""
    auc, S = delong_components(y, p)
    se = math.sqrt(max(float(S[0, 0]), 0.0))
    return [max(0.0, float(auc[0] - Z95 * se)), min(1.0, float(auc[0] + Z95 * se))]


def delong_paired(y, p1, p2) -> dict:
    """Paired DeLong test of AUROC(p1) - AUROC(p2) on the same records: two-sided normal test and 95% normal interval.
    When the variance of the difference is zero: an exact zero difference (identical or monotone-equivalent predictors)
    gives z 0, p 1 and interval [0, 0]; any other difference gives z, p and interval None (the test is undefined).
    Pass probabilities clipped to CLIP to match the AUROCs of eicu_analysis.metrics."""
    y = np.asarray(y, float)
    auc, S = delong_components(y, np.vstack([np.asarray(p1, float), np.asarray(p2, float)]))
    diff = float(auc[0] - auc[1])
    var = max(float(S[0, 0] + S[1, 1] - 2 * S[0, 1]), 0.0)
    se = math.sqrt(var)
    if se > 1e-12:
        z = diff / se
        p, ci = float(2 * norm.sf(abs(z))), [diff - Z95 * se, diff + Z95 * se]
    elif abs(diff) < 1e-12:
        z, p, ci = 0.0, 1.0, [0.0, 0.0]
    else:
        z, p, ci = None, None, None
    return {"auroc_1": float(auc[0]), "auroc_2": float(auc[1]), "diff": diff, "se_diff": se, "ci_diff": ci, "z": z,
            "p_two_sided": p, "n": len(y), "positives": int(y.sum()), "negatives": int(len(y) - y.sum()),
            "var_1": float(S[0, 0]), "var_2": float(S[1, 1]), "cov": float(S[0, 1])}


def record_resamples(n: int, B: int = 1000, seed: int = 29) -> tuple[np.ndarray, np.ndarray]:
    """hybrid_block's record resamples: B rows of n draws from np.random.default_rng(seed), drawn row by row as
    hybrid_block draws them. Returns the index matrix (B x n) and its count matrix C (B x n, float), which equals
    hybrid_block's C for the same n, B and seed."""
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.integers(0, n, n) for _ in range(B)])
    C = np.stack([np.bincount(r, minlength=n) for r in idx]).astype(float)
    return idx, C


def _auroc_draws(y: np.ndarray, p: np.ndarray, idx: np.ndarray, chunk: int = 100) -> np.ndarray:
    """AUROC (midranks) of p on each resample row of idx; NaN for a resample with one class."""
    out = np.empty(len(idx))
    for a in range(0, len(idx), chunk):
        Y, R = y[idx[a:a + chunk]], rankdata(p[idx[a:a + chunk]], axis=1)
        pos = Y.sum(1)
        neg = Y.shape[1] - pos
        with np.errstate(divide="ignore", invalid="ignore"):
            out[a:a + chunk] = np.where((pos > 0) & (neg > 0), ((R * Y).sum(1) - pos * (pos + 1) / 2) / (pos * neg), np.nan)
    return out


def hybrid_descriptive(P: dict[str, np.ndarray], y: np.ndarray, conf: np.ndarray, ids: np.ndarray, chatbots: list[str],
                       share: float = 0.5, shares=SHARES, B: int = 1000, seed: int = 29, cost: dict | None = None) -> dict:
    """jevity.hybrid.hybrid_block without the non-inferiority margin, so P needs no references: P holds per-record
    probabilities on the common records for 'jev' and each chatbot (other keys are ignored). Same routing (jev_split),
    log-loss (clipped to CLIP) and record resamples, so every log-loss number the two share is equal. Adds AUROC of the
    hybrid's and the chatbot's probabilities (clipped to CLIP, as eicu_analysis.metrics) and their difference, with 95%
    percentile intervals from the same resamples (resamples with one class dropped). cost (optional): list US$ per
    answer for 'jev', 'jev_bands' and each chatbot."""
    y = np.asarray(y, float)
    n = len(y)
    Pc = {k: np.clip(np.asarray(P[k], float), *CLIP) for k in ["jev", *chatbots]}
    LL = {k: _ll(v, y) for k, v in Pc.items()}
    idx, C = record_resamples(n, B, seed)
    use = jev_split(conf, ids, share)
    rows = {}
    for s in chatbots:
        h = np.where(use, LL["jev"], LL[s])
        d = h - LL[s]
        rand = share * LL["jev"] + (1 - share) * LL[s]              # exact expectation of random routing, per record
        ph = np.where(use, Pc["jev"], Pc[s])
        a_h, a_s = float(roc_auc_score(y, ph)), float(roc_auc_score(y, Pc[s]))
        dh, ds = _auroc_draws(y, ph, idx), _auroc_draws(y, Pc[s], idx)
        rows[s] = {"hybrid_log_loss": float(h.mean()), "hybrid_log_loss_ci": _ci(C @ h / n),
                   "chatbot_log_loss": float(LL[s].mean()), "diff": float(d.mean()), "ci": _ci(C @ d / n),
                   "random_routed_log_loss": float(rand.mean()), "confidence_minus_random": float((h - rand).mean()),
                   "confidence_minus_random_ci": _ci(C @ (h - rand) / n),
                   "hybrid_auroc": a_h, "hybrid_auroc_ci": _ci(dh), "chatbot_auroc": a_s, "chatbot_auroc_ci": _ci(ds),
                   "auroc_diff": a_h - a_s, "auroc_diff_ci": _ci(dh - ds)}
        if cost:
            k_h = cost["jev"] + cost["jev_bands"] + (1 - use.mean()) * cost[s]
            rows[s]["usd_list_per_1000_records"] = {"hybrid": k_h * 1000, "chatbot_alone": cost[s] * 1000,
                                                    "hybrid_over_chatbot": k_h / cost[s] if cost[s] else None}
    curve = []
    for sh in shares:
        u = jev_split(conf, ids, sh)
        pt = {"share": sh, "n_jev": int(u.sum())}
        for s in chatbots:
            pt[s] = {"hybrid": float(np.where(u, LL["jev"], LL[s]).mean()),
                     "random": float((sh * LL["jev"] + (1 - sh) * LL[s]).mean()), "chatbot_alone": float(LL[s].mean())}
        curve.append(pt)
    return {"label": "descriptive (no non-inferiority margin)", "n_records": n, "deaths": int(y.sum()), "share": share,
            "n_jev": int(use.sum()), "deaths_jev_half": int(y[use].sum()), "deaths_chatbot_half": int(y[~use].sum()),
            "jev_log_loss": float(LL["jev"].mean()), "jev_auroc": float(roc_auc_score(y, Pc["jev"])), "B": B,
            "seed": seed, "usable_band_answers": int(np.isfinite(conf).sum()), "chatbots": rows, "share_curve": curve}


def calibration_summary(p, y) -> dict:
    """eicu_analysis.metrics (mean predicted, observed share, observed-to-expected ratio, clipped log-loss, Brier, AUROC,
    calibration slope and intercept) with the record and death counts."""
    y = np.asarray(y, float)
    return {"n": len(y), "deaths": int(y.sum()), **metrics(p, y)}

"""Arm 1 hybrid, the rule arms 2 and 3 use. Jev answers the share of records where its top risk-band probability is
highest (ties by profile id; no usable band answer ranks last) with its primary death probability; the chatbot answers
the rest. Tested as arm 1 tests Jev at baseline (analysis.prediction_block): log-loss, one set of record resamples,
max-t simultaneous intervals across the five chatbots, margin NI_FRACTION x the age + sex to reference gap. Pure
functions."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from jevity.analysis import CLIP, NI_FRACTION, _ci, _ll

SHARES = tuple(round(0.1 * k, 1) for k in range(1, 10))


def top_probability(band_json) -> float:
    """Largest of Jev's band probabilities; NaN when there is no usable list."""
    if band_json is None or (isinstance(band_json, float) and math.isnan(band_json)):
        return float("nan")
    d = json.loads(band_json) if isinstance(band_json, str) else dict(band_json)
    v = [float(x) for x in d.values() if x is not None]
    return max(v) if v else float("nan")


def jev_split(conf: np.ndarray, ids: np.ndarray, share: float) -> np.ndarray:
    """The floor(share x n) records with Jev's highest confidence, ties by id; NaN confidence ranks last."""
    c = np.where(np.isfinite(conf), conf, -np.inf)
    order = sorted(range(len(ids)), key=lambda i: (-c[i], ids[i]))
    use = np.zeros(len(ids), bool)
    use[order[:int(math.floor(share * len(ids) + 1e-9))]] = True
    return use


def hybrid_block(P: dict[str, np.ndarray], y: np.ndarray, conf: np.ndarray, ids: np.ndarray, chatbots: list[str],
                 conf_own: np.ndarray | None = None, cost: dict | None = None, share: float = 0.5,
                 shares=SHARES, B: int = 1000, seed: int = 29) -> dict:
    """P holds per-record probabilities on the common records for 'jev', each chatbot, 'reference_spline_logit' and
    'reference_age_sex'. cost (optional): list US$ per answer for 'jev', 'jev_bands' and each chatbot."""
    n = len(y)
    LL = {k: _ll(np.clip(np.asarray(v, float), *CLIP), y) for k, v in P.items()}
    gap = float(LL["reference_age_sex"].mean() - LL["reference_spline_logit"].mean())
    margin = NI_FRACTION * gap
    rng = np.random.default_rng(seed)
    C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)
    use = jev_split(conf, ids, share)
    D, pts, rows = [], [], {}
    for s in chatbots:
        h = np.where(use, LL["jev"], LL[s])
        d = h - LL[s]
        D.append(C @ d / n); pts.append(float(d.mean()))
        rand = share * LL["jev"] + (1 - share) * LL[s]              # exact expectation of random routing, per record
        rows[s] = {"hybrid_log_loss": float(h.mean()), "hybrid_log_loss_ci": _ci(C @ h / n),
                   "chatbot_log_loss": float(LL[s].mean()), "random_routed_log_loss": float(rand.mean()),
                   "confidence_minus_random": float((h - rand).mean()), "confidence_minus_random_ci": _ci(C @ (h - rand) / n)}
    D, pts = np.column_stack(D), np.array(pts)
    se = D.std(0, ddof=1) + 1e-12
    crit = float(np.quantile(np.max(np.abs(D - D.mean(0)) / se, axis=1), 0.95))
    for k, s in enumerate(chatbots):
        lo, hi = float(pts[k] - crit * se[k]), float(pts[k] + crit * se[k])
        rows[s].update({"diff": float(pts[k]), "ci": _ci(D[:, k]), "ci_simultaneous": [lo, hi],
                        "outcome": "non-inferior" if hi < margin else "worse" if lo > margin else "inconclusive"})
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
    out = {"n_records": n, "deaths": int(y.sum()), "share": share, "n_jev": int(use.sum()),
           "deaths_jev_half": int(y[use].sum()), "deaths_chatbot_half": int(y[~use].sum()),
           "jev_log_loss": float(LL["jev"].mean()), "age_sex_to_reference_gap": gap, "margin": margin,
           "critical_value": crit, "B": B, "seed": seed, "usable_band_answers": int(np.isfinite(conf).sum()),
           "chatbots": rows, "share_curve": curve}
    if conf_own is not None:                                       # descriptive: routed by Jev's own confidence field
        u2 = jev_split(conf_own, ids, share)
        out["routed_by_own_confidence_field"] = {s: float(np.where(u2, LL["jev"], LL[s]).mean() - LL[s].mean())
                                                 for s in chatbots}
        ok = np.isfinite(conf) & np.isfinite(conf_own)
        out["top_probability_vs_own_field_spearman"] = float(pd.Series(conf[ok]).corr(pd.Series(conf_own[ok]), method="spearman")) if ok.sum() > 2 else None
    return out

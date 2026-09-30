"""eICU replication: baseline prediction of death before hospital discharge on the open eICU-CRD demo, descriptive, no
non-inferiority test. Per predictor (Jev, the five primary chatbots, APACHE IVa's own predicted hospital mortality, and
the references fitted on the full eICU when they have been fitted), on the records
every predictor scored: log-loss (probabilities clipped to [0.005, 0.995], as for NHANES), Brier score, AUROC,
calibration slope and intercept (logistic regression of the outcome on the predicted log-odds, as the NHANES prediction
block), mean predicted against observed mortality and their ratio. 95% percentile intervals from resampling records
(one set of resamples shared by every predictor)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jevity.analysis import CLIP, logit, wlogit_fit


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    """Point values for one predictor on one set of records."""
    p = np.clip(np.asarray(p, float), *CLIP)
    y = np.asarray(y, float)
    a, b = wlogit_fit(logit(p), y)
    both = 0 < y.sum() < len(y)
    return {"log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "brier": float(((p - y) ** 2).mean()),
            "auroc": float(roc_auc_score(y, p)) if both else None,
            "calibration_slope": b, "calibration_intercept": a,
            "mean_predicted": float(p.mean()), "observed": float(y.mean()),
            "observed_to_expected": float(y.mean() / p.mean())}


def baseline_block(preds: dict[str, pd.Series], y: pd.Series, B: int = 2000, seed: int = 20260922,
                   ci: float = 0.95) -> dict:
    """Every predictor on the records all of them scored; the counts each scored alone are reported beside."""
    names = [n for n, s in preds.items() if s is not None]
    common = sorted(set.intersection(*[set(preds[n].dropna().index) for n in names]) & set(y.dropna().index))
    yy = y.loc[common].to_numpy(float)
    P = {n: preds[n].loc[common].to_numpy(float) for n in names}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(B, len(common)))
    q = [(1 - ci) / 2, 1 - (1 - ci) / 2]
    out = {"n_records": len(common), "deaths": int(yy.sum()), "B": B, "seed": seed,
           "scored_alone": {n: int(preds[n].dropna().index.isin(y.dropna().index).sum()) for n in names},
           "predictors": {}}
    for n in names:
        point = metrics(P[n], yy)
        draws = [metrics(P[n][i], yy[i]) for i in idx]
        res = {}
        for k, v in point.items():
            vals = np.array([d[k] for d in draws if d[k] is not None and np.isfinite(d[k])], float)
            res[k] = v
            res[f"{k}_ci"] = [float(x) for x in np.quantile(vals, q)] if len(vals) else None
        out["predictors"][n] = res
    return out

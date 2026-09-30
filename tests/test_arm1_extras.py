"""scripts/arm1_extras.py: the subgroup list, the smoking gradient and the slope formula. Synthetic only."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("arm1_extras", ROOT / "scripts" / "arm1_extras.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)


def _people(n=600, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"age": rng.integers(40, 80, n), "sex": rng.choice(["male", "female"], n),
                         "race_ethnicity": rng.choice(["Non-Hispanic White", "Non-Hispanic Black"], n),
                         "income_poverty_ratio": rng.uniform(0, 5, n), "diabetes": rng.choice(["yes", "no"], n),
                         "smoking": rng.choice(["never", "former", "current"], n)})


def test_subgroups_list_and_partitions():
    d = _people()
    g = X.subgroups(d)
    assert sum(g[f"age {a}-{a + 9}"].sum() for a in (40, 50, 60, 70)) == len(d)
    assert (g["sex male"] | g["sex female"]).all() and (g["diabetes"] ^ g["no diabetes"]).all()
    assert sum(g[f"smoking {s}"].sum() for s in ("never", "former", "current")) == len(d)


def test_smoking_gradient_and_slope():
    d = _people()
    y = np.zeros(len(d))
    cur = d.smoking.to_numpy() == "current"
    P = pd.DataFrame({"flat": np.full(len(d), 0.2), "steep": np.where(cur, 0.4, 0.1)})
    s = X.smoking_gradient(P, d, y, B=50)
    assert abs(s["systems"]["flat"]["log_odds_gap"]) < 1e-12
    assert abs(s["systems"]["steep"]["log_odds_gap"] - (np.log(0.4 / 0.6) - np.log(0.1 / 0.9))) < 1e-9
    t = pd.DataFrame({"edit": ["a", "a", "b", "b"], "feature": ["f1", "f1", "smoking", "smoking"],
                      "z_logit": [0.5, 0.5, 0.2, 0.2], "d_logit": [0.5, 0.5, 1.0, 1.0]})
    assert abs(X.slope(t, ("smoking",)) - 1.0) < 1e-12 and X.slope(t) < 1.0


def test_class_weights_and_complement_hundredths():
    t = pd.DataFrame({"edit": ["a", "b", "c"], "feature": ["f1", "f1", "f2"]})
    c = {"edit_means": [{"edit": "a", "d_logit": 1.0}, {"edit": "b", "d_logit": 1.0}, {"edit": "c", "d_logit": 1.0}]}
    w = X.class_weights(c, t)
    assert abs(w["f1"] - 0.5) < 1e-12 and abs(w["f2"] - 0.5) < 1e-12
    p, q = [0.30, 0.10, 0.60, 0.07], [0.65, 0.95, 0.40, 0.90]          # sums 0.95, 1.05, 1.00, 0.97 (float: some > 0.05 off)
    n = len(p)
    base = dict(edit="baseline", repeat=0, annotated=False, valid=True, variant="raw")
    calls = pd.DataFrame([dict(base, model="jev", profile=i, p=p[i]) for i in range(n)]
                         + [dict(base, model="jev_complement", profile=i, p=q[i]) for i in range(n)])
    r = X.complement_hundredths(calls)
    assert r["within_0.05_inclusive"] == 4 and r["strictly_within_0.05"] == 2 and r["at_exactly_0.95_or_1.05"] == 2

"""Arm 1 hybrid (jevity.hybrid): routing, the random-half expectation and the margin rule. Synthetic only."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import hybrid as H          # noqa: E402


def test_top_probability_and_split():
    assert H.top_probability(json.dumps({"a": 0.1, "b": 0.7, "c": 0.2})) == 0.7
    assert np.isnan(H.top_probability(None))
    conf = np.array([0.9, np.nan, 0.9, 0.5, 0.95])
    ids = np.array([5, 1, 3, 2, 4])
    use = H.jev_split(conf, ids, 0.6)                      # floor(3.0) = 3: 0.95 (id 4), then 0.9 by id (3, 5)
    assert use.tolist() == [True, False, True, False, True]
    assert H.jev_split(conf, ids, 0.2).tolist() == [False, False, False, False, True]
    assert not H.jev_split(conf, ids, 0.9)[1] or H.jev_split(conf, ids, 1.0)[1]   # NaN ranks last


def test_hybrid_block_rules():
    rng = np.random.default_rng(1)
    n = 3000
    risk = rng.uniform(0.02, 0.6, n)
    y = (rng.random(n) < risk).astype(float)
    conf = 1 - np.abs(risk - 0.3)                          # arbitrary confidence, unrelated to the truth draws
    P = {"jev": risk, "good": risk, "bad": np.full(n, 0.5),
         "reference_spline_logit": risk, "reference_age_sex": np.clip(risk * 0.5 + 0.1, 0.01, 0.99)}
    out = H.hybrid_block(P, y, conf, np.arange(n), ["good", "bad"], B=200, shares=(0.5,))
    g, b = out["chatbots"]["good"], out["chatbots"]["bad"]
    assert abs(g["diff"]) < 1e-12 and g["outcome"] == "non-inferior"      # Jev equals the chatbot: no difference
    assert b["diff"] < 0 and b["outcome"] == "non-inferior"                # Jev better than a flat chatbot
    assert out["n_jev"] == n // 2 and out["margin"] > 0
    ll_j = np.mean(-(y * np.log(np.clip(risk, .005, .995)) + (1 - y) * np.log(1 - np.clip(risk, .005, .995))))
    exp_rand = 0.5 * ll_j + 0.5 * np.mean(-(y * np.log(0.5) + (1 - y) * np.log(0.5)))
    assert abs(b["random_routed_log_loss"] - exp_rand) < 1e-12
    worse = H.hybrid_block({**P, "jev": np.full(n, 0.99)}, y, conf, np.arange(n), ["good"], B=200, shares=(0.5,))
    assert worse["chatbots"]["good"]["outcome"] == "worse"

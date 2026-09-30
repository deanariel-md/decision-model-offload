"""Gap-fill statistics: paired DeLong against the slow placement-value computation, and the descriptive hybrid against
jevity.hybrid.hybrid_block. Synthetic only."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import gapfill_stats as G   # noqa: E402
from jevity import hybrid as H          # noqa: E402
from jevity.analysis import CLIP, _ci   # noqa: E402


def _tied_data(n=300, seed=11):
    rng = np.random.default_rng(seed)
    risk = rng.uniform(0.02, 0.7, n)
    y = (rng.random(n) < risk).astype(float)
    p1 = np.round(np.clip(risk + rng.normal(0, 0.15, n), 0.01, 0.99), 2)      # ties within and across classes
    p2 = np.round(np.clip(0.6 * risk + rng.normal(0, 0.1, n), 0.01, 0.99), 1)   # heavy ties, correlated with p1
    return y, p1, p2


def _slow(y, p1, p2):
    """DeLong (1988) the slow way: psi over every death-survivor pair, placement values, ddof 1."""
    X1, Y1, X2, Y2 = p1[y == 1], p1[y == 0], p2[y == 1], p2[y == 0]
    psi1 = (X1[:, None] > Y1[None, :]) + 0.5 * (X1[:, None] == Y1[None, :])
    psi2 = (X2[:, None] > Y2[None, :]) + 0.5 * (X2[:, None] == Y2[None, :])
    m, n = psi1.shape
    v10, v01 = np.vstack([psi1.mean(1), psi2.mean(1)]), np.vstack([psi1.mean(0), psi2.mean(0)])
    S = np.cov(v10) / m + np.cov(v01) / n
    return psi1.mean(), psi2.mean(), S


def test_delong_matches_slow_and_sklearn():
    y, p1, p2 = _tied_data()
    a1, a2, S = _slow(y, p1, p2)
    r = G.delong_paired(y, p1, p2)
    for got, p, slow in ((r["auroc_1"], p1, a1), (r["auroc_2"], p2, a2)):
        assert got == pytest.approx(roc_auc_score(y, p), abs=1e-12) and got == pytest.approx(slow, abs=1e-12)
    assert r["var_1"] == pytest.approx(S[0, 0], rel=1e-10)
    assert r["var_2"] == pytest.approx(S[1, 1], rel=1e-10)
    assert r["cov"] == pytest.approx(S[0, 1], rel=1e-10)
    assert r["se_diff"] ** 2 == pytest.approx(S[0, 0] + S[1, 1] - 2 * S[0, 1], rel=1e-10)
    assert r["diff"] == pytest.approx(a1 - a2, abs=1e-12)
    assert r["z"] == pytest.approx(r["diff"] / r["se_diff"])
    assert r["ci_diff"] == pytest.approx([r["diff"] - 1.959963984540054 * r["se_diff"],
                                          r["diff"] + 1.959963984540054 * r["se_diff"]])
    from scipy.stats import norm
    assert r["p_two_sided"] == pytest.approx(2 * norm.sf(abs(r["z"])))
    assert (r["n"], r["positives"], r["negatives"]) == (len(y), int(y.sum()), int(len(y) - y.sum()))
    lo, hi = G.delong_ci(y, p1)
    se1 = np.sqrt(S[0, 0])
    assert [lo, hi] == pytest.approx([a1 - 1.959963984540054 * se1, a1 + 1.959963984540054 * se1])


def test_delong_identical_monotone_and_degenerate():
    y, p1, _ = _tied_data()
    r = G.delong_paired(y, p1, p1)
    assert r["diff"] == 0 and r["se_diff"] == 0 and r["p_two_sided"] == 1.0 and r["ci_diff"] == [0.0, 0.0]
    m = G.delong_paired(y, p1, np.log(p1) * 3 + 1)                          # monotone transform: same AUROC
    assert m["auroc_1"] == m["auroc_2"] and m["diff"] == 0 and m["p_two_sided"] == 1.0
    assert G.delong_ci(y, p1) == G.delong_ci(y, p1 ** 3)
    perfect, flat = y.copy(), np.full(len(y), 0.3)                          # both variances zero, difference 0.5
    d = G.delong_paired(y, perfect, flat)
    assert d["diff"] == 0.5 and d["se_diff"] == 0 and d["p_two_sided"] is None and d["ci_diff"] is None
    with pytest.raises(ValueError):
        G.delong_paired(np.zeros(10), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    with pytest.raises(ValueError):
        G.delong_ci(np.ones(10), np.linspace(0, 1, 10))
    with pytest.raises(ValueError):
        G.delong_ci(np.r_[1, np.zeros(9)], np.linspace(0, 1, 10))          # one death: variance undefined


def test_resamples_equal_hybrid_block():
    for n in (7, 300, 1001):
        rng = np.random.default_rng(29)                                    # hybrid_block's recipe, verbatim
        C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(50)]).astype(float)
        idx, C2 = G.record_resamples(n, 50, 29)
        assert np.array_equal(C, C2) and idx.shape == (50, n)
        assert np.array_equal(C2, np.stack([np.bincount(r, minlength=n) for r in idx]))


def _hybrid_inputs(n=600, seed=3):
    rng = np.random.default_rng(seed)
    risk = rng.uniform(0.02, 0.6, n)
    y = (rng.random(n) < risk).astype(float)
    conf = np.round(rng.uniform(0.2, 0.9, n), 1)                             # heavy ties in the routing score
    conf[rng.choice(n, 40, replace=False)] = np.nan                         # no usable band answer
    ids = rng.permutation(n) + 1000
    P = {"jev": np.clip(risk + rng.normal(0, 0.05, n), 0.001, 0.999), "good": risk, "flat": np.full(n, 0.3),
         "noisy": np.clip(risk + rng.normal(0, 0.2, n), 0.0, 1.0)}
    return P, y, conf, ids


def test_hybrid_descriptive_equals_hybrid_block():
    P, y, conf, ids = _hybrid_inputs()
    bots = ["good", "flat", "noisy"]
    cost = {"jev": 0.0001, "jev_bands": 0.0001, "good": 0.01, "flat": 0.002, "noisy": 0.0}
    ref = {**P, "reference_spline_logit": P["good"], "reference_age_sex": np.clip(P["good"] * 0.5 + 0.1, 0.01, 0.99)}
    hb = H.hybrid_block(ref, y, conf, ids, bots, cost=cost, B=300)
    for PP in (P, ref):                                                     # extra keys in P are ignored
        hd = G.hybrid_descriptive(PP, y, conf, ids, bots, B=300, cost=cost)
        assert hd["label"] == "descriptive (no non-inferiority margin)"
        for k in ("n_records", "deaths", "share", "n_jev", "deaths_jev_half", "deaths_chatbot_half", "jev_log_loss",
                  "usable_band_answers", "share_curve", "B", "seed"):
            assert hd[k] == hb[k], k
        for s in bots:
            for k in ("diff", "ci", "hybrid_log_loss", "hybrid_log_loss_ci", "chatbot_log_loss",
                      "random_routed_log_loss", "confidence_minus_random", "confidence_minus_random_ci",
                      "usd_list_per_1000_records"):
                assert hd["chatbots"][s][k] == hb["chatbots"][s][k], (s, k)
            assert "margin" not in hd and "outcome" not in hd["chatbots"][s]


def test_hybrid_descriptive_auroc():
    P, y, conf, ids = _hybrid_inputs(n=240, seed=5)
    B = 60
    hd = G.hybrid_descriptive(P, y, conf, ids, ["noisy", "flat"], B=B)
    use = H.jev_split(conf, ids, 0.5)
    idx, _ = G.record_resamples(len(y), B, 29)
    for s in ("noisy", "flat"):
        ph, ps = np.clip(np.where(use, P["jev"], P[s]), *CLIP), np.clip(P[s], *CLIP)
        r = hd["chatbots"][s]
        assert r["hybrid_auroc"] == roc_auc_score(y, ph) and r["chatbot_auroc"] == roc_auc_score(y, ps)
        assert r["auroc_diff"] == pytest.approx(r["hybrid_auroc"] - r["chatbot_auroc"], abs=1e-15)
        dh = np.array([roc_auc_score(y[i], ph[i]) for i in idx])            # the same records as the log-loss draws
        ds = np.array([roc_auc_score(y[i], ps[i]) for i in idx])
        assert r["hybrid_auroc_ci"] == pytest.approx(_ci(dh), abs=1e-12)
        assert r["chatbot_auroc_ci"] == pytest.approx(_ci(ds), abs=1e-12)
        assert r["auroc_diff_ci"] == pytest.approx(_ci(dh - ds), abs=1e-12)
    assert hd["chatbots"]["flat"]["chatbot_auroc"] == 0.5
    assert hd["jev_auroc"] == roc_auc_score(y, np.clip(P["jev"], *CLIP))


def test_hybrid_descriptive_routing_ties_by_id():
    conf = np.array([0.9, 0.9, 0.9, 0.5, np.nan, 0.5, 0.9, 0.2])
    ids = np.array([7, 3, 5, 1, 0, 2, 4, 6])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0], float)
    jev = np.array([0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    bot = np.full(8, 0.5)
    hd = G.hybrid_descriptive({"jev": jev, "bot": bot}, y, conf, ids, ["bot"], share=0.25, shares=(0.25,), B=20)
    use = np.array([False, True, False, False, False, False, True, False])     # the 0.9s with the lowest ids: 3 and 4
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert hd["n_jev"] == 2 and hd["deaths_jev_half"] == 1
    assert hd["chatbots"]["bot"]["hybrid_log_loss"] == pytest.approx(np.where(use, ll(jev), ll(bot)).mean(), abs=1e-15)
    assert hd["chatbots"]["bot"]["hybrid_auroc"] == roc_auc_score(y, np.where(use, jev, bot))
    assert hd["usable_band_answers"] == 7


def test_calibration_summary():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1], float)
    p = np.array([0.8, 0.2, 0.1, 0.6, 0.3, 0.0, 0.2, 0.9])
    c = G.calibration_summary(p, y)
    pc = np.clip(p, *CLIP)
    assert (c["n"], c["deaths"]) == (8, 3)
    assert c["mean_predicted"] == pytest.approx(pc.mean()) and c["observed"] == pytest.approx(3 / 8)
    assert c["observed_to_expected"] == pytest.approx((3 / 8) / pc.mean())
    assert c["log_loss"] == pytest.approx(np.mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc))))

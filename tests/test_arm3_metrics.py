"""Arm 3 metrics on the 10 scored levels. No network."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import arm3_metrics as A3
from jevity import crc

L = crc.LEVELS


def test_scale_is_the_ten_levels():
    assert L == ["0", "I", "IIA", "IIB", "IIC", "IIIA", "IIIB", "IIIC", "IVA-IVB", "IVC"]
    assert A3.level_index("IVA-IVB") == 8 and A3.level_index("IVC") == 9
    assert A3.level_index("IVA") is None and A3.level_index("IV") is None and A3.level_index(None) is None


def test_accuracy_off_by_one_and_unusable_answers():
    truth = ["IIIA", "IIIA", "IIC", "IVC", "0"]
    pred = ["IIIA", "IIIB", "IIIA", None, "IVB"]                                  # a bare substage name is off the scale
    assert A3.exact_accuracy(truth, pred) == pytest.approx(1 / 5)               # unusable answers count as wrong
    assert A3.off_by_one_rate(truth, pred) == pytest.approx(2 / 5)              # IIIB for IIIA; IIIA for IIC
    assert A3.within_one_rate(truth, pred) == pytest.approx(3 / 5)
    assert A3.accuracy_by_level(truth, pred) == {"0": 0.0, "IIC": 0.0, "IIIA": 0.5, "IVC": 0.0}
    assert A3.off_by_one_rate(["IIIC", "IVA-IVB"], ["IVA-IVB", "IVC"]) == 1.0   # adjacent levels
    with pytest.raises(ValueError):
        A3.exact_accuracy(["IIIA", "IV"], ["IIIA", "IIIA"])                     # truth must be one of the 10 levels


def test_main_stage_accuracy():
    truth = ["IIA", "IIA", "IIIC", "IIIC", "I", "IVA-IVB", "0", "IVC"]
    pred = ["IIC", "IIIA", "IIIA", "IVA-IVB", "I", None, "I", "IVA-IVB"]
    assert A3.exact_accuracy(truth, pred) == pytest.approx(1 / 8)
    assert A3.main_stage_accuracy(truth, pred) == pytest.approx(4 / 8)        # IIC for IIA, IIIA for IIIC, IVA-IVB for IVC
    assert A3.main_stage_accuracy(truth, truth) == 1.0
    assert A3.main_stage_accuracy(["IIB"], ["IVB"]) == 0.0                     # off the scale: wrong
    assert "main_stage_accuracy" in A3.summarise(truth, pred)


def test_quadratic_weighted_kappa_matches_sklearn():
    rng = np.random.default_rng(20260922)
    t = rng.integers(0, 10, 400)
    p = np.clip(t + rng.integers(-2, 3, 400), 0, 9)
    truth, pred = [L[i] for i in t], [L[i] for i in p]
    ref = cohen_kappa_score(t, p, labels=list(range(10)), weights="quadratic")
    assert A3.quadratic_weighted_kappa(truth, pred) == pytest.approx(ref)
    assert A3.quadratic_weighted_kappa(truth, truth) == pytest.approx(1.0)
    pred_missing = list(pred)
    pred_missing[0] = None                                                     # left out of kappa
    ref2 = cohen_kappa_score(t[1:], p[1:], labels=list(range(10)), weights="quadratic")
    assert A3.quadratic_weighted_kappa(truth, pred_missing) == pytest.approx(ref2)
    assert math.isnan(A3.quadratic_weighted_kappa(["I", "I"], ["I", "I"]))      # no chance disagreement


def test_expected_level_and_renormalisation():
    assert A3.expected_level({"IIIA": 0.5, "IIIB": 0.5}) == pytest.approx(5.5)
    assert A3.expected_level({"IVC": 0.95}) == pytest.approx(9)                # 0.95 renormalised
    assert A3.expected_level({"I": 0.4, "IIA": 0.4}) is None                   # total 0.8: left out
    assert A3.expected_level({"I": 0.5, "IVA": 0.5}) is None                   # label off the scale
    assert A3.expected_level({"I": 1.2}) is None and A3.expected_level({"I": True}) is None
    assert A3.expected_level(None) is None
    r = A3.expected_level_mae(["IIIA", "IVC", "0"], [{"IIIA": 0.5, "IIIB": 0.5}, {"IVA-IVB": 1.0}, None])
    assert r == {"mae": pytest.approx(0.75), "n": 2, "n_excluded": 1}
    r2 = A3.expected_level_mae(["IIIA", "0"], expected=[5.25, None])
    assert r2["mae"] == pytest.approx(0.25) and r2["n"] == 1
    with pytest.raises(ValueError):
        A3.expected_level_mae(["I"])


def test_summary_on_synthetic_truth():
    truth = [L[i] for i in np.random.default_rng(20260922).permutation(np.repeat(np.arange(10), 6))]   # 6 per level
    pred = [L[min(L.index(g) + 1, 9)] if i % 4 == 0 else g for i, g in enumerate(truth)]   # every fourth one level up
    probs = [{p: 0.7, g: 0.3} if p != g else {g: 1.0} for p, g in zip(pred, truth)]
    s = A3.summarise(truth, pred, probs)
    wrong = sum(p != g for p, g in zip(pred, truth))
    assert s["n"] == 60 and s["n_unusable"] == 0
    assert s["exact_accuracy"] == pytest.approx(1 - wrong / 60)
    assert s["off_by_one_rate"] == pytest.approx(wrong / 60)
    assert s["expected_level_mae"] == pytest.approx(0.7 * wrong / 60)
    same_main = sum(crc.main_stage(p) == crc.main_stage(g) for p, g in zip(pred, truth))
    assert s["main_stage_accuracy"] == pytest.approx(same_main / 60)
    assert 0.9 < s["quadratic_weighted_kappa"] <= 1


def test_alias_log_counts_the_reply_rule():
    rows = [{"system": "gpt", "variant": "names", "answer_how": "alias", "prob_alias": True},
            {"system": "gpt", "variant": "names", "answer_how": "exact", "prob_alias": False},
            {"system": "gpt", "variant": "definitions", "answer_how": "case", "prob_alias": True},
            {"system": "jev", "variant": "names", "answer_how": "argmax", "prob_alias": None}]
    assert A3.alias_log(rows) == [
        {"system": "gpt", "variant": "definitions", "n": 1, "alias_answers": 0, "alias_probability_lists": 1},
        {"system": "gpt", "variant": "names", "n": 2, "alias_answers": 1, "alias_probability_lists": 1},
        {"system": "jev", "variant": "names", "n": 1, "alias_answers": 0, "alias_probability_lists": 0}]
    assert A3.alias_log([{"system": "gpt", "variant": "names", "answer_how": "exact"}])[0]["alias_probability_lists"] == 0

"""Arms 2 and 3 analysis on hand-made answers with known values: accuracy and rates, proper scores, routing with ties,
calibration, AUROC, Holm, the stratified paired bootstrap, costs and tokens, extra cost per extra correct answer, the
Jev-first hybrid and its saving, the acceptable-cheaper rule, repeatability (Fleiss' kappa, ICC(2,1)) and the
supplement tables."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical_analysis as KA

MODELS = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
ARM = yaml.safe_load((ROOT / "config" / "arm2.yaml").read_text(encoding="utf-8"))
L = ("A", "B", "C", "D")


# ------------------------------------------------------------------------------------------------ metrics
def test_accuracy_metrics():
    c = np.array([1, 1, 1, 0, 1, 0], bool)
    s = np.array(["lv", "lv", "lv", "lv", "ac", "ac"])
    assert KA.accuracy_metric(c, s, "balanced_accuracy") == pytest.approx((0.75 + 0.5) / 2)
    assert KA.accuracy_metric(c, s, "exact_accuracy") == pytest.approx(4 / 6)
    C2 = np.vstack([c, ~c])
    assert KA.metric_boot(C2, s, "balanced_accuracy") == pytest.approx([0.625, 0.375])


def test_proper_scores():
    P = np.array([[1.0, 0, 0, 0], [0.25, 0.25, 0.25, 0.25], [0, 1.0, 0, 0]])
    y = np.array([0, 0, 0])
    assert KA.brier(P, y) == pytest.approx([0.0, 0.75, 2.0])
    Q = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
    yq = np.array([1, 0, 1])
    assert KA.rps(Q, yq) == pytest.approx([0.0, (1 + 1) / 2, (0.25 + 0) / 2])


def test_qwk_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 6, 200)
    yh = np.clip(y + rng.integers(-1, 2, 200), 0, 5)
    assert KA.quadratic_weighted_kappa(y, yh, 6) == pytest.approx(cohen_kappa_score(y, yh, weights="quadratic"))
    assert KA.quadratic_weighted_kappa(y, y, 6) == pytest.approx(1.0)


def test_selective_accuracy_and_aurc_with_ties():
    conf = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.1])
    corr = np.array([1, 0, 1, 1, 0, 0], bool)
    assert KA.selective_accuracy(conf, corr, 2 / 6) == pytest.approx(0.5)            # the top tie: 1 of 2
    assert KA.selective_accuracy(conf, corr, 3 / 6) == pytest.approx((1 + 2 / 3) / 3)  # a third of the next tie
    assert KA.selective_accuracy(conf, corr, 1.0) == pytest.approx(0.5)
    flat = np.full(6, 0.7)
    assert KA.selective_accuracy(flat, corr, 0.5) == pytest.approx(0.5)              # all tied: overall accuracy
    risk = 1 - np.array([0.5, 1, 1 + 2 / 3, 2 + 1 / 3, 3, 3]) / np.arange(1, 7)
    assert KA.aurc(conf, corr) == pytest.approx(risk.mean())
    perfect = KA.aurc(np.array([0.9, 0.8, 0.2]), np.array([1, 1, 0], bool))
    worst = KA.aurc(np.array([0.2, 0.8, 0.9]), np.array([1, 1, 0], bool))
    assert perfect < worst


def test_ece_and_auroc():
    conf = np.array([0.95] * 10 + [0.55] * 10)
    corr = np.array([1] * 9 + [0] + [1] * 5 + [0] * 5, bool)
    e, tab = KA.ece(conf, corr, bins=2)
    assert e == pytest.approx((abs(0.5 - 0.55) + abs(0.9 - 0.95)) / 2)
    assert [b["n"] for b in tab] == [10, 10]
    rng = np.random.default_rng(3)
    s, pos = rng.random(100), rng.random(100) < 0.6
    s[:10] = s[10:20]                                                                   # ties
    assert KA.auroc(s, pos) == pytest.approx(roc_auc_score(pos, s))
    assert KA.auroc(s, np.ones(100, bool)) is None


def test_holm():
    adj = KA.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005, "e": 0.5})
    assert adj == pytest.approx({"d": 0.025, "a": 0.04, "c": 0.09, "b": 0.09, "e": 0.5})


def test_stratified_index_keeps_strata_and_is_seeded():
    s = np.array(["a"] * 7 + ["b"] * 3)
    idx = KA.strat_index(s, 500, 20260922)
    assert idx.shape == (500, 10) and (s[idx] == s[None, :]).all()
    assert (idx == KA.strat_index(s, 500, 20260922)).all()


def test_bootstrap_p_values():
    d = np.random.default_rng(0).normal(0.10, 0.02, 2000)
    assert KA.boot_p_two_sided(d) < 0.01
    d0 = np.random.default_rng(0).normal(0.0, 0.02, 2000)
    assert KA.boot_p_two_sided(d0) > 0.5


def test_holm_lower_bounds_step_down():
    rng = np.random.default_rng(7)
    diffs = {"a": rng.normal(0.08, 0.02, 2000), "b": rng.normal(0.02, 0.02, 2000), "c": rng.normal(-0.02, 0.02, 2000),
             "d": rng.normal(-0.06, 0.02, 2000), "e": rng.normal(0.00, 0.02, 2000)}
    h = KA.holm_lower_bounds(diffs, 0.05, 0.025)
    assert [s for s, _ in sorted(h.items(), key=lambda x: x[1]["holm_rank"])][:2] == ["a", "b"]
    assert h["a"]["holm_level"] == pytest.approx(0.025 / 5) and h["b"]["holm_level"] == pytest.approx(0.025 / 4)
    for s, r in h.items():                                          # bound and decision never disagree before a stop
        if r["noninferior"]:
            assert r["holm_lower_bound"] > -0.05
    assert h["a"]["noninferior"] and h["b"]["noninferior"] and not h["d"]["noninferior"]
    stop = min(r["holm_rank"] for r in h.values() if not r["noninferior"])
    assert all(not r["noninferior"] for r in h.values() if r["holm_rank"] > stop)          # step-down: none after
    lv = {r["holm_level"] for r in h.values() if r["holm_rank"] >= stop}
    assert len(lv) == 1                                                                  # later ones keep that level
    lb = np.quantile(diffs["a"], 0.025 / 5, method="lower")
    assert h["a"]["holm_lower_bound"] == lb
    assert KA.holm_lower_bounds(diffs, 0.05, 0.025)["a"]["holm_lower_bound"] < np.quantile(diffs["a"], 0.025)  # stricter


def test_holm_bound_is_exactly_dual_to_the_count():
    d = np.sort(np.random.default_rng(1).normal(0, 0.03, 2000))
    for k in (0, 1, 2, 5, 12, 13, 40):
        x = d.copy()
        x[:k] = -0.2
        x[k:] = np.maximum(x[k:], -0.049)                     # exactly k draws at or below -0.05
        for level in (0.005, 0.00625, 0.0083, 0.0125, 0.025):
            r = KA.holm_lower_bounds({"s": x}, 0.05, level)["s"]
            assert r["noninferior"] == (r["holm_lower_bound"] > -0.05) == (level >= k / (len(x) - 1))


def test_call_costs():
    b, l = KA.call_costs("jev", np.array([1e6, np.nan]), np.array([5e5, 1]), np.array([False, False]), MODELS)
    assert b == pytest.approx([0.042, 0.0]) and l == pytest.approx([0.042, 0.0])      # Jev: input only
    b, l = KA.call_costs("gpt", np.array([1e6, 1e6]), np.array([1e6, 1e6]), np.array([True, False]), MODELS)
    f, bt = MODELS["families"]["gpt"], MODELS["batch"]["families"]["gpt"]
    lst, bch = f["price_in"] + f["price_out"], bt["price_in"] + bt["price_out"]         # a million tokens each way
    assert l == pytest.approx([lst, lst]) and b == pytest.approx([bch, lst])            # batch rows at the batch price
    b, l = KA.call_costs("glm@low", np.array([1e6]), np.array([0]), np.array([False]), MODELS)
    assert l == pytest.approx([1.4])


# ------------------------------------------------------------------------------------------------ one hand-made arm
def _row(system, item, answer, probs=None, conf=None, tin=500, tout=100, route="standard", repeat=0, error=None):
    top = max(probs.values()) if probs else None
    return {"system": system, "item_id": item, "variant": "main", "repeat": repeat, "answer": answer,
            "valid": answer is not None, "probs": json.dumps(probs) if probs else None, "top_prob": top,
            "jev_confidence": conf, "expected_level": None, "tokens_in": tin, "tokens_out": tout, "route": route,
            "error": error, "parse": "json", "prob_status": "ok" if probs else "missing", "choice_not_top": False}


def _p(label, top):
    rest = (1 - top) / 3
    return {l: (top if l == label else rest) for l in L}


NEG = [2, 0, 3, 0, 0, 1, 0, 2]                                   # negations per message, eight words each


def hand_arm():
    """8 items (4 low-value, truth B; 4 appropriate-care, truth A). Jev: 6 right; gpt: 7 right, one unusable."""
    items = pd.DataFrame({"item_id": [f"i{k}" for k in range(8)], "truth": ["B"] * 4 + ["A"] * 4,
                          "stratum": ["low_value"] * 4 + ["appropriate_care"] * 4,
                          "group": ["r1", "r1", "r2", "r2", "r1", "r1", "r2", "r2"],
                          "state": [" ".join(["no"] * k + ["pain"] * (8 - k)) for k in NEG]})
    jev = ["B", "B", "A", "B", "A", "A", "B", "A"]              # wrong on i2 (overuse) and i6 (false restraint)
    jtop = [0.9, 0.8, 0.6, 0.95, 0.9, 0.7, 0.5, 0.85]
    gpt = ["B", "B", "B", "C", "A", "A", "A", None]             # wrong on i3 (distractor), unusable on i7
    rows = [_row("jev", f"i{k}", a, _p(a, t), conf=t, tin=1000, tout=80) for k, (a, t) in enumerate(zip(jev, jtop))]
    rows += [_row("gpt", f"i{k}", a, _p(a, 0.8) if a else None, tin=1000, tout=1000, route="batch")
             for k, a in enumerate(gpt)]
    return pd.DataFrame(rows), items


def test_table_rates_costs_and_verdicts():
    calls, items = hand_arm()
    t = KA.Table(calls, items, ["jev", "gpt"], L, MODELS)
    assert t.correct.sum(axis=0).tolist() == [6, 6] and t.usable.sum(axis=0).tolist() == [8, 7]
    a = {**ARM["analysis"], "n_boot": 200}
    idx = KA.strat_index(t.strata, 200, 1)
    j = KA.system_block(t, "jev", "choice", a, idx)
    g = KA.system_block(t, "gpt", "choice", a, idx)
    assert j["balanced_accuracy"] == pytest.approx(0.75) and g["balanced_accuracy"] == pytest.approx(0.75)
    assert g["unusable"] == 1 and g["unusable_rate"] == pytest.approx(1 / 8) and j["unusable_rate"] == 0
    assert j["rates"] == pytest.approx({"overuse": 0.25, "false_restraint": 0.25, "distractor": 0.0})
    assert g["rates"] == pytest.approx({"overuse": 0.0, "false_restraint": 0.0, "distractor": 0.125})
    assert j["cost"]["usd_list_total"] == pytest.approx(8 * 1000 * 0.042 / 1e6)
    f, bt = MODELS["families"]["gpt"], MODELS["batch"]["families"]["gpt"]
    assert g["cost"]["usd_list_total"] == pytest.approx(8 * (1000 * f["price_in"] + 1000 * f["price_out"]) / 1e6)  # unusable call still billed
    assert g["cost"]["usd_billed_total"] == pytest.approx(8 * (1000 * bt["price_in"] + 1000 * bt["price_out"]) / 1e6)
    assert g["cost"]["usd_list_per_correct"] == pytest.approx(g["cost"]["usd_list_total"] / 6)
    assert g["confidence"]["n"] == 7 and j["confidence"]["n"] == 8
    assert j["confidence"]["brier"] == pytest.approx(np.mean(KA.brier(np.nan_to_num(t.P[:, 0]), t.truth)))
    assert j["jev_own_confidence"]["n"] == 8 and j["confidence"]["auroc"] is None      # fewer than 20 errors
    c = KA.compare(t, "gpt", a, idx)
    assert c["diff"] == pytest.approx(0.0) and c["jev_only"] == 2 and c["chatbot_only"] == 2
    v = KA.value_vs_jev(t, "gpt", idx, 0.95)
    assert v["list"]["usd_per_extra_correct"] is None and v["list"]["note"]           # no extra correct answers
    assert not v["chatbot_more_accurate"] and v["verdict_list"] == "Jev cheaper and at least as accurate"
    h = KA.hybrid(t, "gpt", a)
    assert h["grid"][-1]["share_jev"] == 0 and h["grid"][-1]["balanced_accuracy"] == pytest.approx(0.75)
    assert h["grid"][0]["share_jev"] == 1 and h["grid"][0]["balanced_accuracy"] == pytest.approx(0.75)
    assert all(x["threshold"] is None or 0.5 <= x["threshold"] <= 0.95 for x in h["grid"])
    assert h["first_within_margin"] == h["grid"][0]
    top = next(x for x in h["grid"] if x["threshold"] == pytest.approx(0.85, abs=0.051) and x["share_jev"] == 0.375)
    assert top["accuracy"] == pytest.approx(7 / 8)                                    # Jev on i0, i3, i4; gpt the rest
    ac = KA.acceptable_cheaper(c, j, g, a)                                  # outside the primary family: unadjusted
    assert ac["bound"].startswith("unadjusted") and ac["lower_bound"] == c["lower_bound_unadjusted"]
    assert ac["verdict"] == (c["lower_bound_unadjusted"] > -0.05) and ac["jev_cheaper_list"]
    conf = {**c, "confirmatory": True, "holm_lower_bound": -0.04}
    assert KA.acceptable_cheaper(conf, j, g, a)["verdict"] is True                # bound above -5 and cheaper
    assert KA.acceptable_cheaper({**conf, "holm_lower_bound": -0.06}, j, g, a)["verdict"] is False
    dear = {**j, "cost": {**j["cost"], "usd_list_per_correct": 1.0, "usd_billed_per_correct": 1e-9}}
    v = KA.acceptable_cheaper(conf, dear, g, a)
    assert v["verdict"] is False and v["verdict_billed"] is True and v["basis"] == "list"   # list price decides


def test_extra_cost_per_extra_correct():
    calls, items = hand_arm()
    calls.loc[(calls.system == "gpt") & (calls.item_id == "i3"), ["answer", "probs"]] = ["B", json.dumps(_p("B", 0.8))]
    t = KA.Table(calls, items, ["jev", "gpt"], L, MODELS)
    idx = KA.strat_index(t.strata, 500, 1)
    vv = KA.value_vs_jev(t, "gpt", idx, 0.95)
    assert vv["chatbot_more_accurate"] and vv["verdict_list"] == "chatbot more accurate"      # 7 correct against 6
    v = vv["billed"]
    dk = t.cost_billed[:, 1].sum() - t.cost_billed[:, 0].sum()
    assert v["usd_per_extra_correct"] == pytest.approx(dk / 1)
    assert v["extra_correct_per_1000_items"] == pytest.approx(1000 / 8)
    assert v["extra_usd_per_1000_items"] == pytest.approx(dk / 8 * 1000)


def test_missing_answers_and_duplicates():
    calls, items = hand_arm()
    calls.loc[(calls.system == "gpt") & (calls.item_id == "i0"), "error"] = KA.BATCH_MISSING
    calls.loc[(calls.system == "gpt") & (calls.item_id == "i0"), ["answer", "valid"]] = [None, False]
    t = KA.Table(calls, items, ["jev", "gpt"], L, MODELS)
    assert t.present[:, 1].sum() == 7 and not t.correct[0, 1]
    with pytest.raises(AssertionError, match="duplicate"):
        KA.Table(pd.concat([calls, calls.iloc[:1]]), items, ["jev", "gpt"], L, MODELS)


def test_analyze_whole_arm_and_holm_family():
    calls, items = hand_arm()
    extra = []
    for s in ("claude", "gemini", "gpt_free"):
        extra += [_row(s, f"i{k}", a, _p(a, 0.7)) for k, a in enumerate(["B", "A", "B", "B", "A", "A", "A", "A"])]
    rep = [_row(s, "i0", a, _p(a, 0.7), repeat=r) for s in ("jev", "gpt") for r, a in ((1, "B"), (2, "A"))]
    calls = pd.concat([calls, pd.DataFrame(extra + rep)], ignore_index=True)
    tiers = {"jev": "jev", "gpt": "primary", "claude": "primary", "gemini": "primary", "gpt_free": "free"}
    cfg = {**ARM, "analysis": {**ARM["analysis"], "n_boot": 300}}
    out = KA.analyze(calls, items, cfg, L, list(tiers), tiers, MODELS, rep_ids=["i0"])
    json.dumps(out)                                                                   # JSON-safe
    assert set(out["comparisons"]) == {"gpt", "claude", "gemini", "gpt_free"}
    assert out["comparisons"]["gpt_free"]["p_two_sided_holm"] is None and not out["comparisons"]["gpt_free"]["confirmatory"]
    assert out["comparisons"]["gpt_free"]["holm_lower_bound"] is None
    for s in ("gpt", "claude", "gemini"):
        c = out["comparisons"][s]
        assert c["confirmatory"] and c["holm_level"] in (0.025 / 3, 0.025 / 2, 0.025)
        assert c["noninferior"] == (c["holm_lower_bound"] > -0.05) or not c["noninferior"]
        assert out["acceptable_cheaper"][s]["lower_bound"] == c["holm_lower_bound"]
    assert sorted(out["comparisons"][s]["holm_rank"] for s in ("gpt", "claude", "gemini")) == [1, 2, 3]
    ps = [out["comparisons"][s]["p_two_sided"] for s in ("gpt", "claude", "gemini")]
    assert min(out["comparisons"][s]["p_two_sided_holm"] for s in ("gpt", "claude", "gemini")) == pytest.approx(min(1, 3 * min(ps)))
    r = out["repeatability"]["jev"]                                   # i0 answered B, B, A
    assert r["items_complete"] == 1 and r["identical_share"] == 0.0 and r["measure"] == "fleiss_kappa"
    assert r["fleiss_kappa"] == pytest.approx((1 / 3 - 5 / 9) / (1 - 5 / 9))
    assert out["repeatability"]["claude"]["items_complete"] == 0 and out["repeatability"]["claude"]["fleiss_kappa"] is None
    assert out["strata"] == {"appropriate_care": 4, "low_value": 4}
    assert out["systems"]["jev"]["tier"] == "jev" and "speed" in out["systems"]["jev"]
    assert set(out["style_check"]["systems"]) == set(tiers) and out["style_check"]["baseline"]["folds"] == 2
    for s, c in out["comparisons"].items():                   # three outcomes; others descriptive
        assert c["outcome"] in ("non-inferior", "worse", "inconclusive")
        assert c["outcome_basis"] == ("holm" if c["confirmatory"] else "unadjusted (descriptive)")
    hf = out["hybrid_share"]
    assert set(hf["comparisons"]) == {"gpt", "claude", "gemini"} and hf["rule"]["uses_truth"] is False
    assert hf["jev_items"] == ["i0", "i3", "i4", "i7"] and all(c["n_jev"] == 4 for c in hf["comparisons"].values())
    assert sorted(c["holm_rank"] for c in hf["comparisons"].values()) == [1, 2, 3]
    no_hf = {**cfg, "analysis": {k: v for k, v in cfg["analysis"].items() if k != "hybrid_share"}}
    assert "hybrid_share" not in KA.analyze(calls, items, no_hf, L, list(tiers), tiers, MODELS)


# ------------------------------------------------------------------------------------------------ style check
def test_style_features_follow_the_config_definition():
    sc = ARM["analysis"]["style_check"]
    assert sc["features"]["negation_count"] == (r"\b(never|no|not|none|nothing|without|cannot|nobody|nowhere|neither|"
                                                r"nor|dont|cant|wont|didnt|doesnt|isnt|wasnt|\w+n['" "’" r"]t)\b")
    assert list(sc["features"]) == ["word_count", "negation_count"] and sc["ignore_case"] is True
    assert sc["positive_stratum"] == "low_value" and sc["class_weight"] == "balanced"
    X = KA.style_features(["I don't know. No, I never did.",
                           "Nothing is wrong; it's NOT notable, cannot say.",     # 'notable' not counted
                           "Without pain, none.",
                           "I DONT care, haven't hasn't didn't doesn't isn't",
                           "I can’t and couldn’t, won't, shouldn't, ain't",     # curly and straight n't
                           "nobody nowhere, neither nor. cant wont didnt doesnt isnt wasnt",
                           "North of the knot, Norway, nonetheless, Cannon, didn"],    # none of these
                          sc)
    assert X.tolist() == [[7, 3], [8, 3], [3, 2], [8, 6], [7, 5], [10, 10], [8, 0]]


def test_style_baseline_is_leave_one_group_out():
    neg = np.array([0, 0, 1, 1, 3, 3, 4, 4] * 6, float)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0] * 5 + [1, 1, 1, 1, 0, 0, 0, 0])             # r5 reverses the pattern
    X = np.column_stack([np.full(48, 90.0), neg])
    g = np.repeat([f"r{k}" for k in range(6)], 8)
    z = KA.style_baseline(X, y, g, "balanced")
    right = (z > 0) == (y == 1)
    assert not right[g == "r5"].any() and right[g != "r5"].mean() == 0.75             # r5 predicted without r5
    b0, b = KA.logit_fit(X[g != "r5"], y[g != "r5"], "balanced")
    assert z[g == "r5"] == pytest.approx(b0 + X[g == "r5"] @ b) and b[1] > 0 and b[0] == 0   # constant column: slope 0
    with pytest.raises(ValueError, match="one class"):
        KA.style_baseline(X[:8], np.r_[np.zeros(4), np.ones(4)], np.array(["a"] * 4 + ["b"] * 4), None)


def test_style_check_reference_row_and_split():
    calls, items = hand_arm()
    t = KA.Table(calls, items, ["jev", "gpt"], L, MODELS)
    a = {**ARM["analysis"], "n_boot": 300}
    idx = KA.strat_index(t.strata, 300, 1)
    st = KA.style_check(t, a, idx, 7)
    X = KA.style_features(items.state, a["style_check"])
    assert X[:, 1].tolist() == NEG and set(X[:, 0]) == {8}
    z = KA.style_baseline(X, (items.stratum == "low_value").to_numpy(int), items.group.to_numpy(), "balanced")
    right = np.where(z > 0, "B", "A") == items.truth.to_numpy()                      # low_value -> B, else A
    b = st["baseline"]
    assert st["style_misleading_items"] == items.item_id[~right].tolist() and st["n_style_matching"] == right.sum()
    assert b["answers"] == {"low_value": "B", "appropriate_care": "A"} and b["folds"] == 2
    assert b["balanced_accuracy"] == pytest.approx(KA.accuracy_metric(right, t.strata, "balanced_accuracy"))
    assert b["balanced_accuracy_ci"] == pytest.approx(                                 # the systems' own resamples
        KA.percentile_ci(KA.metric_boot(right[idx], t.strata, "balanced_accuracy"), 0.95))
    assert b["predicted"]["low_value"]["low_value"] == pytest.approx(4 * b["accuracy_by_stratum"]["low_value"])
    for j, s in enumerate(["jev", "gpt"]):
        for name, m in (("style_matching", right), ("style_misleading", ~right)):
            r = st["systems"][s][name]
            assert r["n"] == m.sum()
            if all(r["n_by_stratum"].values()):
                assert r["balanced_accuracy"] == pytest.approx(
                    KA.accuracy_metric(t.correct[m, j], t.strata[m], "balanced_accuracy"))
            else:
                assert r["balanced_accuracy"] is None
    r = KA._subset(np.array([True, False]), np.array(["lv", "lv"]), np.array(["ac", "lv"]), "balanced_accuracy",
                   50, 1, 0.95)
    assert r["balanced_accuracy"] is None and r["accuracy_by_stratum"] == {"ac": None, "lv": 0.5}
    with pytest.raises(ValueError, match="state"):
        KA.style_check(KA.Table(calls, items.drop(columns="state"), ["jev"], L, MODELS), a, idx, 7)


def test_score_arm_metrics():
    labels = ("0", "I", "II", "III", "IV")
    items = pd.DataFrame({"item_id": [f"r{k}" for k in range(10)], "truth": [labels[k % 5] for k in range(10)]})
    items["stratum"] = items.truth
    rows = []
    for k in range(10):
        t = k % 5
        ans = labels[min(t + (k >= 5), 4)]                     # second half one level too high (except the top)
        p = {l: (0.6 if l == ans else 0.1) for l in labels}
        rows.append({**_row("jev", f"r{k}", ans, p, conf=0.4), "expected_level": float(labels.index(ans)) + 0.2})
        rows.append(_row("gpt", f"r{k}", labels[t], {l: (1.0 if l == labels[t] else 0.0) for l in labels}))
    cfg = {**ARM, "kind": "score", "analysis": {"primary_metric": "exact_accuracy", "strata_column": "stage",
                                                "n_boot": 200, "margin": 0.05, "coverages": [0.5, 0.8], "ece_bins": 10,
                                                "min_errors_auroc": 20, "ci": 0.95}}
    tiers = {"jev": "jev", "gpt": "primary"}
    out = KA.analyze(pd.DataFrame(rows), items, cfg, labels, ["jev", "gpt"], tiers, MODELS)
    j, g = out["systems"]["jev"], out["systems"]["gpt"]
    assert j["exact_accuracy"] == pytest.approx(0.6) and g["exact_accuracy"] == 1.0
    assert j["off_by_one"] == pytest.approx(0.4) and g["qwk"] == pytest.approx(1.0)
    assert j["expected_level_mae"] == pytest.approx(np.mean([0.2] * 5 + [1.2] * 4 + [0.2]))
    assert g["confidence"]["rps"] == pytest.approx(0.0) and "rps" in j["confidence"] and "brier" not in j["confidence"]
    assert out["comparisons"]["gpt"]["diff"] == pytest.approx(-0.4)


def test_timing_summary():
    df = pd.DataFrame({"system": ["jev"] * 3 + ["gpt"] * 2, "latency_s": [3, 4, 5, 10, None],
                       "valid": [True, True, False, True, False], "error": [None, None, None, None, "x"]})
    s = KA.timing_summary(df)
    assert s["jev"]["median_s"] == 4 and s["gpt"]["errors"] == 1 and s["gpt"]["n"] == 2
    assert "window_start" not in s["jev"]
    w = KA.timing_summary(df.assign(window_start=[1.79e9] * 3 + [1.79e9 + 3600] * 2))
    assert len(w["jev"]["window_start"]) == 19 and w["gpt"]["window_start"] > w["jev"]["window_start"]


# ------------------------------------------------------------------------------------------------ outcomes and hybrid share
def test_holm_worse_and_three_outcomes():
    """Non-inferior (Holm-adjusted lower bound above -5), worse (Holm-adjusted upper bound below -5), otherwise
    inconclusive. The worse procedure is the exact mirror of the non-inferiority one."""
    rng = np.random.default_rng(3)
    diffs = {"ni": rng.normal(0.05, 0.01, 2000), "worse": rng.normal(-0.20, 0.01, 2000),
             "worse2": rng.normal(-0.15, 0.01, 2000), "unclear": rng.normal(-0.05, 0.05, 2000)}
    h = KA.holm_outcomes(diffs, 0.05, 0.025)
    assert {s: r["outcome"] for s, r in h.items()} == {"ni": "non-inferior", "worse": "worse", "worse2": "worse",
                                                         "unclear": "inconclusive"}
    assert h["worse"]["holm_rank_worse"] == 1 and h["worse"]["holm_level_worse"] == pytest.approx(0.025 / 4)
    mirror = KA.holm_lower_bounds({s: -d - 0.10 for s, d in diffs.items()}, 0.05, 0.025)
    for s in diffs:
        assert h[s]["holm_upper_bound"] == pytest.approx(-mirror[s]["holm_lower_bound"] - 0.10)
        assert h[s]["worse"] == mirror[s]["noninferior"] and h[s]["holm_rank_worse"] == mirror[s]["holm_rank"]
    with pytest.raises(AssertionError):
        KA.outcome(True, True)


def test_hybrid_share_uses_jev_confidence_only():
    calls, items = hand_arm()                     # Jev's top probabilities 0.9, 0.8, 0.6, 0.95, 0.9, 0.7, 0.5, 0.85
    t = KA.Table(calls, items, ["jev", "gpt"], L, MODELS)
    use = KA.hybrid_split(t, 0.5)
    assert items.item_id[use].tolist() == ["i0", "i3", "i4", "i7"]
    assert items.item_id[KA.hybrid_split(t, 0.25)].tolist() == ["i0", "i3"]           # 0.9 tie: i0 before i4
    flipped = items.assign(truth=["A"] * 4 + ["B"] * 4)                                # the truth never enters
    assert (KA.hybrid_split(KA.Table(calls, flipped, ["jev", "gpt"], L, MODELS), 0.5) == use).all()
    a = {**ARM["analysis"], "n_boot": 300}
    idx = KA.strat_index(t.strata, 300, 1)
    b, d = KA.hybrid_share(t, "gpt", a, idx, use)
    assert b["n_jev"] == 4 and b["share_jev"] == 0.5 and b["share_jev_by_stratum"] == {"appropriate_care": 0.5,
                                                                                         "low_value": 0.5}
    assert b["balanced_accuracy"] == pytest.approx(1.0) and b["chatbot_alone"] == pytest.approx(0.75)   # gpt wrong
    assert b["diff"] == pytest.approx(0.25) and len(d) == 300                   # on i3 and i7, where Jev answers
    k = b["cost_list"]
    assert k["hybrid_usd_total"] == pytest.approx(t.cost_list[:, 0].sum() + t.cost_list[~use, 1].sum())
    assert k["chatbot_usd_total"] == pytest.approx(t.cost_list[:, 1].sum()) and k["hybrid_over_chatbot"] < 1


# ------------------------------------------------------------------------------------------------ repeatability, cost, tables
def test_fleiss_kappa_and_icc_known_values():
    """Fleiss' kappa (arm 2) and ICC(2,1) (arm 3) across the repeats."""
    from statsmodels.stats.inter_rater import fleiss_kappa
    fleiss71 = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
                         [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    assert KA.fleiss_kappa(fleiss71) == pytest.approx(0.20993, abs=1e-5)
    rng = np.random.default_rng(5)
    B = np.stack([rng.multinomial(3, p, size=40) for p in ([0.5, 0.3, 0.1, 0.1], [0.25] * 4, [0.9, 0.1, 0, 0])])
    assert KA.fleiss_kappa(B) == pytest.approx([fleiss_kappa(b) for b in B])           # batched as one at a time
    assert np.isnan(KA.fleiss_kappa(np.array([[3, 0], [3, 0]])))                       # one category: undefined
    sf = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]])
    assert KA.icc_2_1(sf) == pytest.approx(0.29, abs=0.005)                            # Shrout and Fleiss 1979, Table 4
    assert KA.icc_2_1(np.stack([sf, sf[::-1]])) == pytest.approx([float(KA.icc_2_1(sf))] * 2)
    assert KA.icc_2_1(np.tile(np.arange(5.0)[:, None], (1, 3))) == pytest.approx(1.0)
    assert np.isnan(KA.icc_2_1(np.ones((5, 3)))) and np.isnan(KA.icc_2_1(sf[:1]))


def test_repeatability_on_the_repeat_set():
    items = pd.DataFrame({"item_id": [f"i{k}" for k in range(6)], "truth": ["B"] * 3 + ["A"] * 3,
                          "stratum": ["low_value"] * 3 + ["appropriate_care"] * 3})
    ans = {"jev": [["B", "B", "B"], ["B", "B", "B"], ["A", "B", "A"], ["A", "A", "A"], ["A", "A", "A"], ["C", "C", "C"]],
           "gpt": [["B", "B", "B"], ["B", None, "B"], ["A", "A", "A"], ["A", "A", "A"], ["A", "A", "A"], ["B", "B", "B"]]}
    calls = pd.DataFrame([_row(s, f"i{k}", a, repeat=r) for s, per in ans.items() for k, rs in enumerate(per)
                          for r, a in enumerate(rs)])
    calls = calls[~((calls.system == "gpt") & (calls.item_id == "i5") & (calls["repeat"] == 2))]   # one repeat missing
    ids = items.item_id.tolist()
    out = KA.repeatability(calls, items, ids, ["jev", "gpt"], L, "choice", 3, 500, 1, 0.95)
    j, g = out["jev"], out["gpt"]
    R = np.array([[L.index(a) for a in rs] for rs in ans["jev"]])
    N = np.stack([(R == k).sum(axis=1) for k in range(4)], axis=1)
    assert j["items_all_usable"] == 6 and j["identical_share"] == pytest.approx(5 / 6) and j["repeats"] == 3
    assert j["fleiss_kappa"] == pytest.approx(KA.fleiss_kappa(N))
    assert j["fleiss_kappa_ci"][0] <= j["fleiss_kappa"] <= j["fleiss_kappa_ci"][1]
    assert g["items_complete"] == 5 and g["items_all_usable"] == 4                       # i5 incomplete, i1 unusable
    assert g["identical_share"] == 1.0 and list(g["identical_share_ci"]) == [1.0, 1.0] and g["fleiss_kappa"] == 1.0
    assert g["undefined_resamples"] > 0                        # resamples holding only A answers: kappa undefined
    s = KA.repeatability(calls, items, ids, ["jev"], L, "score", 3, 500, 1, 0.95)["jev"]
    assert s["measure"] == "icc_2_1" and "fleiss_kappa" not in s and s["icc_2_1"] == pytest.approx(KA.icc_2_1(R))


def test_cost_block_tokens_and_cost_per_correct():
    calls, items = hand_arm()
    calls["tokens_reasoning"] = np.where(calls.system == "gpt", 400.0, np.nan)          # Jev reports none
    t = KA.Table(calls, items, ["jev", "gpt"], L, MODELS)
    idx = KA.strat_index(t.strata, 400, 1)
    j, g = KA.cost_block(t, 0, idx, 0.95), KA.cost_block(t, 1, idx, 0.95)
    assert (g["tokens_in_per_answer"], g["tokens_out_per_answer"], g["tokens_reasoning_per_answer"]) == (1000, 1000, 400)
    assert j["tokens_out_per_answer"] == 80 and j["tokens_reasoning_per_answer"] is None and j["tokens_reasoning_n"] == 0
    assert g["usd_list_per_1000_answers"] == pytest.approx(g["usd_list_total"] / 8 * 1000)
    f, bt = MODELS["families"]["gpt"], MODELS["batch"]["families"]["gpt"]
    share = (bt["price_in"] + bt["price_out"]) / (f["price_in"] + f["price_out"])        # 1,000 tokens each way
    assert g["usd_billed_per_1000_answers"] == pytest.approx(g["usd_list_per_1000_answers"] * share)  # batch route
    assert g["usd_list_per_million_answers"] == pytest.approx(g["usd_list_per_answer"] * 1e6)
    ratio = t.cost_list[:, 1][idx].sum(axis=1) / t.correct[:, 1][idx].sum(axis=1)
    assert g["usd_list_per_correct_ci"] == pytest.approx(KA.percentile_ci(ratio, 0.95))
    assert g["resamples_without_a_correct_answer"] == 0


def test_hybrid_cost_and_saving():
    calls, items = hand_arm()
    t = KA.Table(calls, items, ["jev", "gpt"], L, MODELS)
    use = KA.hybrid_split(t, 0.5)                                                        # i0, i3, i4, i7
    k = KA.hybrid_cost(t, "gpt", use)
    assert k["saving_usd_per_1000_items"] == pytest.approx((t.cost_list[use, 1].sum() - t.cost_list[:, 0].sum()) / 8 * 1000)
    assert k["saving_share"] == pytest.approx(1 - k["hybrid_usd_total"] / k["chatbot_usd_total"])
    p = KA.hybrid_point(t, "gpt", {"primary_metric": "balanced_accuracy"}, use, "half")
    assert p["balanced_accuracy"] == pytest.approx(1.0) and p["chatbot_alone"] == pytest.approx(0.75) and p["n_jev"] == 4


def test_random_half_control_and_coverage_curve():
    """The same share handed to Jev at random. The random-routed metric is the mean over every set of
    that size (checked by enumerating all 70 sets of 4 of 8 items); confidence minus random, paired; the curve carries
    list cost per 1,000 answers at every point, routed both ways."""
    import itertools
    calls, items = hand_arm()
    t = KA.Table(calls, items, ["jev", "gpt"], L, MODELS)
    a = {**ARM["analysis"], "n_boot": 200}
    cj, cc, kj, kc = t.correct[:, 0], t.correct[:, 1], t.cost_list[:, 0], t.cost_list[:, 1]
    sets = [np.isin(np.arange(8), s) for s in itertools.combinations(range(8), 4)]
    for name in ("balanced_accuracy", "accuracy"):
        every = np.mean([KA.accuracy_metric(np.where(u, cj, cc), t.strata, name) for u in sets])
        assert KA.random_routed(cj, cc, 0.5, t.strata, name) == pytest.approx(every)
    use = KA.hybrid_split(t, 0.5)                                                        # i0, i3, i4, i7: all right
    same = np.tile(np.arange(8), (5, 1))                                                 # resamples = the items
    r = KA.routing_control(t, "gpt", a, use, same)
    assert r["share_jev"] == 0.5 and r["n_jev"] == 4 and set(r) == {"share_jev", "n_jev", "balanced_accuracy", "accuracy"}
    b = r["balanced_accuracy"]
    assert b["confidence_routed"] == pytest.approx(1.0) and b["random_routed"] == pytest.approx(0.75)
    assert b["diff"] == pytest.approx(0.25) and b["diff_ci"] == pytest.approx([0.25, 0.25])
    idx = KA.strat_index(t.strata, 200, 1)
    d = (KA.metric_boot(np.where(use, cj, cc)[idx], t.strata, "balanced_accuracy")
         - 0.5 * KA.metric_boot(cj[idx], t.strata, "balanced_accuracy") - 0.5 * KA.metric_boot(cc[idx], t.strata, "balanced_accuracy"))
    assert KA.routing_control(t, "gpt", a, use, idx)["balanced_accuracy"]["diff_ci"] == pytest.approx(KA.percentile_ci(d))
    ex = KA.routing_control(t, "gpt", {**a, "primary_metric": "exact_accuracy"}, use, same)
    assert set(ex) == {"share_jev", "n_jev", "exact_accuracy"}                           # arm 3: one metric
    assert ex["exact_accuracy"]["random_routed"] == pytest.approx(0.5 * cj.mean() + 0.5 * cc.mean())   # arm 3's formula
    cur = KA.coverage_curve(t, "gpt", a, [0.25, 0.5])
    assert [p["n_jev"] for p in cur] == [2, 4] and cur[1]["balanced_accuracy_confidence_routed"] == pytest.approx(1.0)
    assert cur[1]["balanced_accuracy_random_routed"] == pytest.approx(0.75)
    assert cur[1]["usd_list_per_1000_confidence_routed"] == pytest.approx((kj.sum() + kc[~use].sum()) / 8 * 1000)
    cost_every = np.mean([(kj + np.where(u, 0, kc)).mean() for u in sets]) * 1000
    assert cur[1]["usd_list_per_1000_random_routed"] == pytest.approx(cost_every)
    g = KA.hybrid(t, "gpt", a)["grid"]
    for p in g:                                                                          # the 0-90% decile curve
        assert p["usd_list_per_1000_answers"] == pytest.approx(p["usd_list_per_answer"] * 1000)
        assert p["usd_billed_per_1000_answers"] == pytest.approx(p["usd_billed_per_answer"] * 1000)
        assert p["balanced_accuracy_random_routed"] == pytest.approx(
            KA.random_routed(cj, cc, p["share_jev"], t.strata, "balanced_accuracy"))
    assert g[-1]["usd_list_per_1000_random_routed"] == pytest.approx(g[-1]["usd_list_per_1000_answers"])  # chatbot alone


def test_supplement_tables_from_the_result():
    calls, items = hand_arm()
    calls["tokens_reasoning"] = np.where(calls.system == "gpt", 400.0, np.nan)
    rep = [_row(s, f"i{k}", a, _p(a, 0.7), repeat=r) for s in ("jev", "gpt") for k, a in ((0, "B"), (4, "A"))
           for r in (1, 2)]
    calls = pd.concat([calls, pd.DataFrame(rep)], ignore_index=True)
    tiers = {"jev": "jev", "gpt": "primary"}
    cfg = {**ARM, "analysis": {**ARM["analysis"], "n_boot": 200}}
    timing = pd.DataFrame({"system": ["jev", "gpt"] * 3, "latency_s": [3, 9, 4, 10, 5, 30], "valid": True, "error": None})
    res = json.loads(json.dumps(KA.analyze(calls, items, cfg, L, ["jev", "gpt"], tiers, MODELS, rep_ids=["i0", "i4"],
                                           timing=timing)))
    tabs = KA.supplement_tables(res)
    assert list(tabs) == ["repeatability", "cost", "value", "routing", "coverage_curve"]
    cu = tabs.pop("coverage_curve")                                 # the share curve, primary chatbots only
    assert [r["tier"] for r in cu["rows"]] == [f"{x}%" for x in range(10, 100, 10)] and {r["system"] for r in cu["rows"]} == {"gpt"}
    p5 = cu["rows"][4]
    assert p5["Chatbot alone"] == res["systems"]["gpt"]["balanced_accuracy"] and p5["Items answered by Jev"] == res["coverage_curve"]["systems"]["gpt"][4]["n_jev"]
    assert "descriptive" in cu["notes"][0] and KA.table_markdown(cu).startswith("**Jev answers a growing share")
    ro = {r["system"]: r for r in tabs["routing"]["rows"]}
    assert ro["gpt"]["Confidence minus random, balanced accuracy"] == pytest.approx(
        (res["routing"]["systems"]["gpt"]["balanced_accuracy"]["diff"],
         *res["routing"]["systems"]["gpt"]["balanced_accuracy"]["diff_ci"]))
    assert ro["gpt"]["Balanced accuracy, confidence-routed"] == pytest.approx(1.0) and ro["jev"]["Share answered by Jev"] is None
    assert res["routing"]["rule"]["jev_share"] == 0.5 and len(res["coverage_curve"]["systems"]["gpt"]) == 9
    for tab in tabs.values():
        assert [r["system"] for r in tab["rows"]] == ["jev", "gpt"] and len(KA.table_frame(tab)) == 2
        assert sum(line.startswith("| ") for line in KA.table_markdown(tab).splitlines()) == 3
    rp = {r["system"]: r for r in tabs["repeatability"]["rows"]}
    assert rp["gpt"]["Fleiss' kappa"][0] == res["repeatability"]["gpt"]["fleiss_kappa"] == 1.0
    c = {r["system"]: r for r in tabs["cost"]["rows"]}
    assert c["gpt"]["of which reasoning"] == 400 and c["jev"]["of which reasoning"] is None
    assert c["gpt"]["Median seconds per answer"] == 10 and c["gpt"]["90th percentile seconds"] == pytest.approx(26)
    assert tabs["cost"]["notes"][-1].endswith("one at a time within one window.")
    res2 = json.loads(json.dumps(res))                                 # two systems timed in separate windows
    res2["systems"]["jev"]["speed"]["window_start"] = "2026-09-27T02:12:56"
    res2["systems"]["gpt"]["speed"]["window_start"] = "2026-09-27T08:05:09"
    assert KA.supplement_tables(res2)["cost"]["notes"][-1] == (
        "Seconds: the timing sample, every system answering the same items one at a time, in two windows (start, local "
        "time): 2026-09-27 02:12, jev; 2026-09-27 08:05, gpt.")
    v = {r["system"]: r for r in tabs["value"]["rows"]}
    assert v["gpt"]["Extra list US$ per extra correct answer against Jev"] == "Jev cheaper and at least as accurate"
    assert v["gpt"]["Hybrid: share answered by Jev"] == 0.5 and v["jev"]["Hybrid list US$ per 1,000 answers"] is None
    assert (v["gpt"]["Saving against the chatbot alone, US$ per 1,000 answers"]
            == res["hybrid_cost"]["gpt"]["saving_usd_per_1000_items"])
    f = KA.table_frame(tabs["value"])
    assert {"List US$ per correct answer (low)", "List US$ per correct answer (high)"} <= set(f.columns)
    md = KA.table_markdown(tabs["cost"])
    assert "| jev | jev | " in md and " | – | " in md                                 # Jev: no reasoning reported
    assert [KA._fmt(x, "usd") for x in (1.26e-05, 0.0126, 3.456, 126.4, 12600.2)] == ["0.000013", "0.013", "3.46", "126",
                                                                                       "12,600"]
    assert KA._fmt((0.5, None, None), "share") == "0.50" and KA._fmt((0.5, 0.25, 0.75), "share") == "0.50 (0.25 to 0.75)"

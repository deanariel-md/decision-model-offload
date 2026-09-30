"""Descriptive supplement tables (src/jevity/supplement.py): agreement with analysis.prediction_block at its clipping
bound, the clipping arithmetic, run-to-run spread, extreme answers and the reference change. No network."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import analysis as A
from jevity import supplement as S


def _calls(rows):
    return pd.DataFrame([{"model": m, "profile": p, "edit": e, "annotated": False, "repeat": r, "variant": "raw",
                          "p": x, "valid": x is not None} for m, p, e, r, x in rows])


def test_analysis_bound_reproduces_prediction_block():
    rng = np.random.default_rng(5)
    idx = np.arange(400)
    y = pd.Series(rng.integers(0, 2, 400), index=idx)
    preds = {"jev": pd.Series(np.r_[0.0, 1.0, rng.uniform(0, 1, 398)], index=idx),
             "gpt": pd.Series(rng.uniform(0.01, 0.9, 400), index=idx),
             "reference_age_sex": pd.Series(rng.uniform(0.05, 0.5, 400), index=idx)}
    block = A.prediction_block(preds, y)
    mine = S.log_loss_clipping(preds, y)["by_clip"][str(A.CLIP[0])]["systems"]
    for nm, v in block["systems"].items():
        assert abs(mine[nm]["log_loss"] - v["log_loss"]) < 1e-12
        assert np.allclose(mine[nm]["log_loss_ci"], v["log_loss_ci"], atol=1e-12)


def test_clipping_bound_sets_the_penalty_for_a_certain_wrong_answer():
    idx = [1, 2]
    y = pd.Series([1, 0], index=idx)
    res = S.log_loss_clipping({"jev": pd.Series([0.0, 0.0], index=idx), "gpt": pd.Series([0.5, 0.5], index=idx)}, y, B=10)
    for c in S.CLIPS:
        s = res["by_clip"][str(c)]["systems"]["jev"]
        assert abs(s["log_loss"] - (-np.log(c) - np.log(1 - c)) / 2) < 1e-12
        assert s["share_clipped"] == 1.0
    assert res["by_clip"]["0.001"]["systems"]["jev"]["log_loss"] > res["by_clip"]["0.01"]["systems"]["jev"]["log_loss"]
    assert abs(res["by_clip"]["0.01"]["focal_minus"]["gpt"]["diff"]
               - (res["by_clip"]["0.01"]["systems"]["jev"]["log_loss"] - np.log(2))) < 1e-12


def test_repeat_spread_pools_item_variances():
    rows = [("gpt", 1, "baseline", r, x) for r, x in enumerate((0.2, 0.2, 0.2))]            # identical: variance 0
    rows += [("gpt", 2, "baseline", r, x) for r, x in enumerate((0.1, 0.3, None))]          # two valid answers
    rows += [("gpt", 3, "baseline", 0, 0.4)]                                                 # no repeat: not an item
    rows += [("gpt", 1, "I1_field_order", 0, 0.5)]                                           # no repeat: not an item
    res = S.repeat_spread(_calls(rows), ["gpt"], B=20)["gpt"]
    assert res["n_items"] == 2 and res["n_items_three_answers"] == 1 and res["n_profiles"] == 2
    v_pts = np.var([10.0, 30.0], ddof=1) / 2                                                 # mean of (0, 200)
    assert abs(res["pooled_sd_pts"] - np.sqrt(v_pts)) < 1e-9
    v_logit = np.var(A.logit(np.array([0.1, 0.3])), ddof=1) / 2
    assert abs(res["pooled_sd_logit"] - np.sqrt(v_logit)) < 1e-9
    assert res["share_items_identical"] == 0.5
    assert set(res["by_edit"]) == {"baseline"}


def test_extreme_answers_counts_first_raw_answers_only():
    rows = [("jev", 1, "baseline", 0, 0.0), ("jev", 2, "baseline", 0, 1.0), ("jev", 3, "baseline", 0, 0.3),
            ("jev", 1, "C4_sbp_plus20", 0, 0.0), ("jev", 1, "baseline", 1, 1.0), ("jev", 4, "baseline", 0, None)]
    res = S.extreme_answers(_calls(rows), ["jev"])["jev"]
    assert res["baseline"] == {"n": 3, "share_0": 1 / 3, "share_1": 1 / 3, "share_0_or_1": 2 / 3}
    assert res["all_answers"]["n"] == 4 and res["all_answers"]["share_0_or_1"] == 0.75


def test_reference_change_matches_instances_and_carries_refits():
    rng = np.random.default_rng(2)
    prof = np.arange(100, 160)
    inst = pd.DataFrame({"profile": prof, "edit": "L2a_income_low", "klass": "label", "feature": "income",
                         "applicable": True, "supported": [True] * 50 + [False] * 10,
                         "d_logit_spline_logit": rng.normal(0.2, 0.1, 60), "d_prob_spline_logit": rng.normal(0.02, 0.01, 60)})
    sup = inst[inst.supported]
    R = 7
    noise = rng.normal(0, 0.05, (len(sup), R))
    ref = A.Refits(sup.d_logit_spline_logit.to_numpy()[:, None] + noise, sup.d_prob_spline_logit.to_numpy()[:, None] + noise / 10,
                   {(int(p), "L2a_income_low"): i for i, p in enumerate(sup.profile)})
    res = S.reference_change(inst, {"spline_logit": ref}, prof, B=300, edits=["L2a_income_low"])["spline_logit"]["L2a_income_low"]
    assert res["n_records"] == 50 and res["klass"] == "label"
    assert abs(res["mean_logit"] - sup.d_logit_spline_logit.mean()) < 1e-12
    assert abs(res["odds_ratio"] - np.exp(sup.d_logit_spline_logit.mean())) < 1e-12
    lo, hi = res["mean_logit_ci"]
    assert lo < res["mean_logit"] < hi and res["refits"] == R
    assert abs(res["odds_ratio_ci"][0] - np.exp(lo)) < 1e-12


def test_output_cap_counts_length_stops_by_variant_and_edit():
    rows = [("glm", 1, "baseline", 0, 0.2), ("glm", 1, "C4_sbp_plus20", 0, None), ("glm", 2, "baseline", 0, None),
            ("glm", 2, "C4_sbp_plus20", 0, 0.3), ("gpt", 1, "baseline", 0, 0.2), ("jev", 1, "baseline", 0, 0.2)]
    c = _calls(rows)
    c["finish_reason"] = ["stop", "length", "stop", "stop", "stop", None]
    c["tokens_out"] = [900.0, 16000.0, 40.0, 1200.0, 150.0, None]
    res = S.output_cap(c, ["jev", "gpt", "glm"])
    assert "jev" not in res and res["gpt"]["variants"]["raw"]["at_cap"] == 0
    g = res["glm"]["variants"]["raw"]
    assert (g["n"], g["at_cap"], g["unusable"], g["unusable_at_cap"]) == (4, 1, 2, 1)
    assert g["tokens_out_max"] == 16000.0 and res["glm"]["max_tokens"] == 16000
    assert res["glm"]["by_edit_first_answers"]["C4_sbp_plus20"] == {"n": 2, "at_cap": 1, "share": 0.5}


def test_fill_missing_pairs_adds_extreme_contrasts_for_unanswered_supported_records():
    inst = pd.DataFrame({"profile": [1, 2, 3, 1, 2], "edit": ["C4_sbp_plus20"] * 3 + ["L4a_uninsured"] * 2,
                         "klass": ["clinical"] * 3 + ["label"] * 2, "feature": ["sbp"] * 3 + ["insurance"] * 2,
                         "applicable": True, "supported": [True, True, True, True, False],
                         "d_logit_spline_logit": [0.3, 0.2, 0.1, 0.05, 0.0], "d_prob_spline_logit": [0.03, 0.02, 0.01, 0.0, 0.0]})
    rows = [("glm", 1, "baseline", 0, 0.2), ("glm", 1, "C4_sbp_plus20", 0, 0.3), ("glm", 1, "L4a_uninsured", 0, None),
            ("glm", 2, "baseline", 0, 0.4), ("glm", 2, "C4_sbp_plus20", 0, None),
            ("glm", 3, "baseline", 0, None), ("glm", 3, "C4_sbp_plus20", 0, 0.5)]
    calls = _calls(rows)
    tab = A.contrast_table(calls, inst, "glm", "spline_logit")
    assert len(tab) == 1                                            # only profile 1's SBP pair is complete
    z = {"C4_sbp_plus20": 2.0, "L4a_uninsured": -1.0}
    out, added = S.fill_missing_pairs(tab, inst, calls, "glm", "spline_logit", [1, 2, 3], ["C4_sbp_plus20", "L4a_uninsured"], z)
    assert added == {"C4_sbp_plus20": 2, "L4a_uninsured": 1}        # profile 2 L4a is unsupported: never filled
    f = out.set_index(["profile", "edit"])
    assert f.loc[(2, "C4_sbp_plus20"), "z_logit"] == 2.0 and f.loc[(2, "C4_sbp_plus20"), "p_base"] == 0.4
    assert abs(A.logit(f.loc[(2, "C4_sbp_plus20"), "p"]) - (A.logit(0.4) + 2.0)) < 1e-9
    assert abs(f.loc[(3, "C4_sbp_plus20"), "p_base"] - 0.3) < 1e-12              # baseline missing: median of valid baselines (0.2, 0.4)
    assert f.loc[(1, "L4a_uninsured"), "z_logit"] == -1.0 and f.loc[(1, "L4a_uninsured"), "d_logit"] == 0.05
    assert f.loc[(1, "C4_sbp_plus20"), "z_logit"] == tab.z_logit.iloc[0]   # answered pairs untouched
    same, none = S.fill_missing_pairs(tab, inst, calls, "glm", "spline_logit", [1], ["C4_sbp_plus20"], z)
    assert none == {} and len(same) == 1


def test_icc_2_1_matches_shrout_fleiss_and_repeatability_counts_items():
    # Shrout and Fleiss (1979) Table 2: ICC(2,1) = 0.29 for their 6 targets x 4 judges
    x = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    assert abs(S.icc_2_1(x) - 0.2898) < 1e-3
    assert S.icc_2_1(np.ones((5, 3))) is None and S.icc_2_1(x[:1]) is None
    rows = []
    for p in range(1, 21):
        for e in ("baseline", "C1"):
            base = 0.05 + 0.04 * p + (0.1 if e == "C1" else 0.0)
            for r in range(3):
                rows.append(("gpt", p, e, r, base + (0.01 * r if p % 2 else 0.0)))
                rows.append(("jev", p, e, r, base))
    rows[-1] = ("jev", 20, "C1", 2, None)                       # one unusable repeat: item left out and counted
    rows += [("gpt", p, "C2", 0, 0.3) for p in range(1, 21)]    # main-pass only: not in the repeat set
    res = S.repeatability(_calls(rows), ["gpt", "jev"], B=200)
    g, j = res["gpt"], res["jev"]
    assert g["repeats"] == 3 and g["items"] == 40 and g["items_all_usable"] == 40 and g["records"] == 20
    assert abs(g["identical"] - 0.5) < 1e-12 and 0 < g["icc_2_1"] < 1
    assert g["icc_2_1_ci"][0] <= g["icc_2_1"] <= g["icc_2_1_ci"][1]
    assert j["items_with_unusable"] == 1 and j["items_all_usable"] == 39
    assert abs(j["icc_2_1"] - 1) < 1e-12 and j["identical"] == 1.0
    assert j["baseline"]["items"] == 20 and j["baseline"]["items_with_unusable"] == 0


def test_reasoning_tokens_reads_each_provider_shape():
    assert S.reasoning_tokens({"response": {"usage": {"completion_tokens_details": {"reasoning_tokens": 486}}}}) == 486
    assert S.reasoning_tokens({"response": {"usage": {"output_tokens_details": {"reasoning_tokens": 12}}}}) == 12
    assert S.reasoning_tokens({"response": {"usageMetadata": {"thoughtsTokenCount": 7}}}) == 7
    assert S.reasoning_tokens({"response": {"usage": {"input_tokens": 976, "output_tokens": 20}}}) is None
    assert S.reasoning_tokens(None) is None

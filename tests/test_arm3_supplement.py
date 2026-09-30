"""Arm 3 supplement tables (src/jevity/arm3_supplement.py, scripts/arm3_supplement.py): ICC(2,1) and the share
answered identically on the repeat set, reasoning tokens from the raw replies, the value rule against Jev, and the
tables from a simulated run through the shared code. No network."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from jevity import arm3_supplement as S


def test_icc_2_1_matches_shrout_and_fleiss():
    """Shrout and Fleiss (1979), Table 2: six targets, four judges; ICC(2,1) = 0.29."""
    x = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]])
    assert S.icc_2_1(x) == pytest.approx(0.2898, abs=1e-4)
    assert S.icc_2_1(np.array([[0, 0, 0], [4, 4, 4], [9, 9, 9]])) == pytest.approx(1.0)
    assert S.icc_2_1(np.full((5, 3), 2.0)) is None                                     # no variance: undefined
    assert S.icc_2_1(np.array([[1, 2, 3]])) is None
    shifted = np.array([[0, 1, 0], [4, 5, 4], [9, 9, 9], [2, 3, 2]])                    # one repeat reads a level higher
    assert 0.9 < S.icc_2_1(shifted) < 1.0


def test_repeatability_counts_resamples_and_seed():
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 10, 40).astype(float)
    x = np.column_stack([truth, truth, truth])
    x[:6, 2] = np.clip(x[:6, 2] + 1, 0, 9)                                              # 6 reports differ once
    x[6:9, 1] = np.nan                                                                  # 3 with an unusable answer
    r = S.repeatability(x, 500, 7)
    assert (r["reports"], r["reports_all_usable"], r["reports_with_unusable"]) == (40, 37, 3)
    same = [(row == row[0]).all() for row in x if not np.isnan(row).any()]
    assert r["identical"] == pytest.approx(np.mean(same))
    assert r["identical_ci"][0] <= r["identical"] <= r["identical_ci"][1]
    assert r["icc_2_1_ci"][0] <= r["icc_2_1"] <= r["icc_2_1_ci"][1] <= 1.0
    assert r == S.repeatability(x, 500, 7) and r != S.repeatability(x, 500, 8)
    empty = S.repeatability(np.full((4, 3), np.nan), 100, 1)
    assert empty["reports_all_usable"] == 0 and empty["icc_2_1"] is None and empty["identical"] is None


def test_reasoning_tokens_where_reported():
    chat = {"response": {"usage": {"prompt_tokens": 300, "completion_tokens": 600,
                                   "completion_tokens_details": {"reasoning_tokens": 512, "audio_tokens": 0}}}}
    assert S.reasoning_tokens(chat) == 512
    assert S.reasoning_tokens({"response": {"usage": {"output_tokens_details": {"reasoning_tokens": 64}}}}) == 64
    assert S.reasoning_tokens({"response": {"usageMetadata": {"thoughtsTokenCount": 223}}}) == 223
    assert S.reasoning_tokens({"response": {"usage": {"prompt_tokens": 297, "completion_tokens": 43}}}) is None
    assert S.reasoning_tokens({"response": {"usage": {"input_tokens": 487, "output_tokens": 46}}}) is None   # Jev
    assert S.reasoning_tokens(None) is None


def test_value_verdict():
    assert S.value_verdict(800, 850, 0.02, 4.0).startswith("chatbot more accurate")
    assert S.value_verdict(850, 850, 0.02, 4.0) == "Jev cheaper and at least as accurate"
    assert S.value_verdict(900, 850, 0.02, 4.0) == "Jev cheaper and at least as accurate"
    assert S.value_verdict(900, 850, 0.5, 0.0) == "Jev at least as accurate, chatbot cheaper"


# The random-half control and the coverage curve come from the shared analysis; an independent reference
# implementation is kept here, and the shared numbers must agree with it.
def _ref_random_routed(correct_jev, correct_chat, share_jev):
    """Exact accuracy expected when Jev answers a random set of the reports of the given share and the chatbot the rest:
    share x Jev's accuracy + (1 - share) x the chatbot's; on a (B, n) array, one value per resample."""
    cj, cc = np.asarray(correct_jev, float), np.asarray(correct_chat, float)
    return share_jev * cj.mean(axis=-1) + (1 - share_jev) * cc.mean(axis=-1)


def _ref_routing(correct_jev, correct_chat, use, idx, ci=0.95):
    """Confidence-routed minus random-routed exact accuracy, paired over the resampled reports."""
    cj, cc, use = np.asarray(correct_jev, float), np.asarray(correct_chat, float), np.asarray(use, bool)
    share, hc = float(use.mean()), np.where(use, cj, cc)
    conf, rnd = float(hc.mean()), float(_ref_random_routed(cj, cc, share))
    d = hc[idx].mean(axis=1) - _ref_random_routed(cj[idx], cc[idx], share)
    return {"n_jev": int(use.sum()), "confidence_routed": conf, "random_routed": rnd, "diff": conf - rnd,
            "diff_ci": [float(v) for v in np.quantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])]}


def _ref_coverage_point(correct_jev, correct_chat, use, usd_jev, usd_chat):
    """One point of the coverage curve: exact accuracy and list cost per 1,000 answers (Jev asked on every report, the
    chatbot on the rest), routed by confidence and at random."""
    cj, cc, use = np.asarray(correct_jev, float), np.asarray(correct_chat, float), np.asarray(use, bool)
    kj, kc = np.asarray(usd_jev, float), np.asarray(usd_chat, float)
    share = float(use.mean())
    return {"n_jev": int(use.sum()), "exact_accuracy_confidence_routed": float(np.where(use, cj, cc).mean()),
            "exact_accuracy_random_routed": float(_ref_random_routed(cj, cc, share)),
            "usd_list_per_1000_confidence_routed": float((kj + np.where(use, 0.0, kc)).mean() * 1000),
            "usd_list_per_1000_random_routed": float((kj.mean() + (1 - share) * kc.mean()) * 1000)}


def test_random_routing_is_the_mean_over_every_set():
    """The random-routed accuracy is exact: the mean over every set of reports of the same size handed to Jev (the
    reference and the shared code's)."""
    import itertools
    KA = pytest.importorskip("jevity.categorical_analysis")
    cj, cc = np.array([1, 0, 1, 1, 0, 0]), np.array([0, 1, 1, 0, 1, 0])
    sets = list(itertools.combinations(range(6), 3))
    brute = np.mean([np.where(np.isin(np.arange(6), s), cj, cc).mean() for s in sets])
    assert _ref_random_routed(cj, cc, 0.5) == pytest.approx(brute)
    assert KA.random_routed(cj, cc, 0.5, np.zeros(6, int), "exact_accuracy") == pytest.approx(brute)


def test_reference_routing_and_coverage_point():
    cj = np.array([1, 1, 1, 0, 0, 1, 0, 1])
    cc = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    use = np.array([1, 1, 0, 0, 0, 0, 0, 1], bool)          # Jev takes the three where it is right and the chatbot wrong
    idx = np.random.default_rng(1).integers(0, 8, size=(500, 8))
    r = _ref_routing(cj, cc, use, idx)
    assert r["confidence_routed"] == pytest.approx(np.where(use, cj, cc).mean())
    assert r["random_routed"] == pytest.approx(3 / 8 * cj.mean() + 5 / 8 * cc.mean())
    assert r["diff"] > 0 and r["diff_ci"][0] <= r["diff"] <= r["diff_ci"][1]
    p = _ref_coverage_point(cj, cc, use, np.full(8, 0.001), np.full(8, 0.01))
    assert p["usd_list_per_1000_confidence_routed"] == pytest.approx(p["usd_list_per_1000_random_routed"])
    assert p["usd_list_per_1000_random_routed"] == pytest.approx((0.001 + 5 / 8 * 0.01) * 1000)


def test_accuracy_by_format():
    correct = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    fmt = np.array(["synoptic"] * 4 + ["narrative"] * 4)
    idx = np.stack([np.r_[np.random.default_rng(b).integers(0, 4, 4), 4 + np.random.default_rng(b + 1).integers(0, 4, 4)]
                    for b in range(300)])
    r = S.accuracy_by_format(correct, fmt, idx)
    assert (r["synoptic"]["accuracy"], r["narrative"]["accuracy"], r["diff"]) == (0.75, 0.25, 0.5)
    assert r["synoptic"]["n"] == r["narrative"]["n"] == 4 and r["diff_ci"][0] <= 0.5 <= r["diff_ci"][1]


@pytest.mark.parametrize("tnm, answered, want", [
    (("T3", "N0", "M0"), "IIB", ["T"]),                # IIA read as IIB: T4a
    (("T3", "N1a", "M0"), "IVA-IVB", ["M"]),           # IIIB read as stage IV: a distant site
    (("T3", "N1a", "M0"), "IIIC", ["T", "N"]),         # T4b N1a or T3 N2b: a tie
    (("T3", "N0", "M0"), "IIIA", ["T+N"]),             # T1-T2 with a positive node: both
    (("T3", "N0", "M1c"), "IIA", ["M"]),               # IVC read as IIA: the peritoneal disease missed
    (("Tis", "N0", "M0"), "I", ["T"]),
])
def test_components_to_reach(tnm, answered, want):
    assert S.components_to_reach(*tnm, answered) == want


def test_wrong_answer_components():
    tnm = [("T3", "N0", "M0"), ("T3", "N1a", "M0"), ("T3", "N1a", "M0"), ("T1", "N0", "M0")]
    out = S.wrong_answer_components(tnm, ["IIB", "IIIC", "IIIB", None], ["IIA", "IIIB", "IIIB", "I"])
    assert out == {"wrong_with_level": 2, "unusable": 1, "by_component": {"T": 1, "T or N": 1}}


def test_supplement_from_a_simulated_run(tmp_path, monkeypatch):
    """Five synthetic reports per level (tests/synthetic_items.py), primary tier, both versions and the timing sample,
    through the real clients with a simulated sender; then the shared analysis and the supplement. Nothing is written inside the repository."""
    from jevity import categorical as K
    import arm3_supplement as AS
    import run_categorical as RC
    from jevity import clients as C
    from jevity.categorical_sim import SimulatedSender

    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(C, "_post", boom)
    monkeypatch.setattr(C.httpx, "Client", boom)
    cfg = yaml.safe_load((ROOT / "config" / "arm3.yaml").read_text(encoding="utf-8"))
    import synthetic_items as SI
    full = pd.read_csv(SI.items_path("arm3"), dtype=str, keep_default_na=False)
    full.groupby("level", sort=False).head(5).to_csv(tmp_path / "items.csv", index=False, lineterminator="\n")
    cfg["items"]["main"] = str(tmp_path / "items.csv")
    cfg["items"].pop("repeat_column")
    cfg.update(repeats={"n_items": 5, "n_repeats": 3}, systems={"tiers": ["primary"]}, route={"batch": False},
               timing={"n": 5, "window_min": 0})
    cfg["analysis"]["n_boot"] = 200
    (tmp_path / "arm3.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    arm = K.load_arm(tmp_path / "arm3.yaml")
    paths = RC.Paths(arm, tmp_path / "out")
    sender = SimulatedSender(arm, K.load_items(arm))
    RC.cmd_run(arm, paths, sender=sender)
    RC.cmd_timing(arm, paths, n=5, window_min=0, date="dryrun", sender=sender)
    RC.cmd_analyze(arm, paths)
    out = AS.run(arm, paths)
    systems = K.arm_systems(arm)
    from jevity import categorical_analysis as KA
    from jevity.clients import MODELS
    calls = pd.read_parquet(paths.calls())
    items, sys_, *_ = RC.design(arm)
    assert list(out) == ["names", "definitions"]                                        # the primary first
    for v, r in out.items():
        assert r["simulated"] and set(r["repeatability"]) == set(r["cost"]) == set(systems) == set(r["speed"])
        assert all(x["reports"] == 5 for x in r["repeatability"].values())
        for s, x in r["cost"].items():
            assert x["answers"] == r["n_items"] == 50
            assert x["usd_list_per_million_answers"] == pytest.approx(x["usd_list_per_1000_answers"] * 1000)
            assert x["input_tokens_per_answer"] > 0
        assert r["cost"]["jev"]["reasoning_tokens_per_answer"] is None
        for s, x in r["value"].items():
            assert x["chatbot_more_accurate"] == (x["correct_chatbot"] > x["correct_jev"])
            assert (x["usd_per_extra_correct"] is not None) == (x["chatbot_more_accurate"] and x["extra_usd_per_1000_items"] is not None)
            if not x["chatbot_more_accurate"]:
                assert x["verdict"] == "Jev cheaper and at least as accurate"          # Jev's list price is far lower
        h = r["hybrid"]
        assert h["n_jev"] == 25
        for x in h["systems"].values():
            assert x["saving_usd_per_1000"] == pytest.approx(x["usd_chatbot_per_1000"] - x["usd_hybrid_per_1000"])
            assert x["saving_usd_per_1000"] > 0
        assert all(x["median_s"] is not None and x["p90_s"] >= x["median_s"] for x in r["speed"].values())
        chat = [s for s in systems if s != "jev"]
        assert "routing" not in r and "coverage_curve" not in r            # the shared analysis's only
        shared = json.loads((paths.results / ("analysis.json" if v == "names" else f"analysis_{v}.json"))
                            .read_text(encoding="utf-8"))
        t = KA.Table(calls, RC.items_frame(items, v), sys_, arm.labels, MODELS, v)
        idx = KA.strat_index(t.strata, int(cfg["analysis"]["n_boot"]), int(cfg["seed"]))
        jj, use = t.col("jev"), KA.hybrid_split(t, 0.5)
        assert shared["routing"]["rule"]["jev_share"] == 0.5
        assert shared["coverage_curve"]["shares"] == pytest.approx([k / 10 for k in range(1, 10)])
        for s in chat:                                  # the shared numbers agree with the reference, points and intervals
            jc = t.col(s)
            ref = _ref_routing(t.correct[:, jj], t.correct[:, jc], use, idx, cfg["analysis"]["ci"])
            got = shared["routing"]["systems"][s]
            assert got["n_jev"] == ref["n_jev"] == h["n_jev"]
            for k in ("confidence_routed", "random_routed", "diff"):
                assert got["exact_accuracy"][k] == pytest.approx(ref[k], abs=1e-12)
            assert got["exact_accuracy"]["diff_ci"] == pytest.approx(ref["diff_ci"], abs=1e-12)
            assert got["exact_accuracy"]["confidence_routed"] == pytest.approx(h["systems"][s]["accuracy_hybrid"])
            for c, p in zip(shared["coverage_curve"]["shares"], shared["coverage_curve"]["systems"][s]):
                q = _ref_coverage_point(t.correct[:, jj], t.correct[:, jc], KA.hybrid_split(t, c), t.cost_list[:, jj],
                                        t.cost_list[:, jc])
                assert {k: p[k] for k in q} == pytest.approx(q, abs=1e-12)
        if v == "names":                                                   # by format and components: names only
            assert set(r["by_format"]) == set(r["components"]) == set(systems)
            correct = {"jev": r["value"][chat[0]]["correct_jev"], **{s: r["value"][s]["correct_chatbot"] for s in chat}}
            for s in systems:
                f, c = r["by_format"][s], r["components"][s]
                assert f["synoptic"]["n"] + f["narrative"]["n"] == r["n_items"]
                assert sum(c["by_component"].values()) == c["wrong_with_level"]
                assert c["wrong_with_level"] + c["unusable"] == r["n_items"] - correct[s]
        else:
            assert "by_format" not in r and "components" not in r
        written = json.loads((paths.results / f"supplement_{v}.json").read_text(encoding="utf-8"))
        assert written["cost"]["jev"]["usd_list_per_correct"] == r["cost"]["jev"]["usd_list_per_correct"]
    md = (paths.results / "supplement_tables.md").read_text(encoding="utf-8")
    assert md.count("### Repeatability") == 2 and all(f"| {s} |" in md for s in systems)
    assert "simulated dry run" in md and "not run yet" not in md
    assert "### Routing by Jev's confidence" not in md and md.count("### Coverage curve") == 2    # one table each
    RC.cmd_tables(arm, paths)
    for suffix in ("", "_definitions"):
        assert "Routing by Jev's confidence" in (paths.results / f"table_routing{suffix}.md").read_text(encoding="utf-8")
    assert md.count("### Exact accuracy by report format") == 1 and md.count("### Wrong answers by the component") == 1


def test_tables_say_when_timing_has_not_run():
    r = {"simulated": False, "repeat_set": 40, "n_items": 1000,
         "repeatability": {"jev": {"reports": 40, "reports_all_usable": 40, "identical": 1.0, "identical_ci": [1.0, 1.0],
                                   "icc_2_1": None, "icc_2_1_ci": None}},
         "cost": {"jev": {"input_tokens_per_answer": 700.0, "output_tokens_per_answer": 46.0,
                          "reasoning_tokens_per_answer": None, "usd_billed_per_1000_answers": 0.03,
                          "usd_list_per_1000_answers": 0.03, "usd_list_per_million_answers": 29.4,
                          "usd_list_per_correct": 0.00004, "usd_list_per_correct_ci": [0.00003, 0.00005]}},
         "value": {}, "hybrid": {"n_jev": 500, "jev_share": 0.5, "systems": {}}, "speed": {}}
    md = S.tables_markdown({"names": r})
    assert "| jev | 40 of 40 | 100.0% (100.0% to 100.0%) | n/a |" in md
    assert "not reported" in md and "The timing sample has not run yet" in md

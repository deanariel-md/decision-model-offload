"""Synthetic end-to-end dry runs (no network): arm 2 on 180 synthetic items (tests/synthetic_items.py; config/arm2.yaml
with items.main naming them) with simulated systems (requests, raw store and parsers as live; batch families answered
synchronously by the simulator), the timing sample, the analysis and the tables; and a small synthetic Score arm with
two variants, the path arm 3 uses. Also the cost projection and the stop rule."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from jevity import categorical as K
from jevity import clients as C
from jevity.categorical_sim import SimulatedSender
import run_categorical as RC
import synthetic_items as SI

ARM = SI.arm("arm2")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(C, "_post", boom)
    monkeypatch.setattr(C.httpx, "Client", boom)


def _listing(p: Path) -> set:
    return {str(x) for x in p.rglob("*")} if p.exists() else set()


def test_arm2_dry_run(tmp_path):
    before = _listing(ROOT / "results" / "arm2") | _listing(ROOT / "runs" / "arm2")
    out = RC.cmd_dryrun(ARM, tmp_path, n_timing=10)
    assert _listing(ROOT / "results" / "arm2") | _listing(ROOT / "runs" / "arm2") == before    # nothing in the repo
    res = tmp_path / "results" / "arm2"
    for f in ("design.json", "calls.parquet", "timing_sample.json", "timing_sample.parquet", "analysis.json",
              "table_repeatability.csv", "table_repeatability.md", "table_cost.csv", "table_cost.md", "table_value.csv",
              "table_value.md", "table_routing.csv", "table_routing.md"):
        assert (res / f).exists(), f
    assert out["simulated"] is True and pd.read_parquet(res / "calls.parquet").simulated.all()
    calls = pd.read_parquet(res / "calls.parquet")
    assert len(calls) == 11 * 180 + 11 * 40 * 2 and calls.error.isna().all()
    assert len(list((tmp_path / "runs" / "arm2").glob("*.json"))) == len(calls)             # every reply stored raw
    d = json.loads((res / "design.json").read_text(encoding="utf-8"))
    assert d["n_items"] == 180 and d["items_file"] == str(SI.items_path("arm2"))
    assert d["systems"]["jev"] == "jev" and d["systems"]["medgemma"] == "pair" and len(d["repeat_items"]) == 40
    assert d["jev_question_example"]["type"] == "choice"
    assert out["n_items"] == 180 and out["strata"] == {"appropriate_care": 55, "low_value": 125}
    assert all(c["missing"] == 0 for c in out["completeness"].values())
    assert list(out["systems"]) == K.arm_systems(ARM)
    prim = [s for s, c in out["comparisons"].items() if c["confirmatory"]]
    assert prim == ["gpt", "claude", "gemini", "muse", "glm"]
    assert sorted(out["comparisons"][s]["holm_rank"] for s in prim) == [1, 2, 3, 4, 5]
    for s in prim:
        c = out["comparisons"][s]
        assert c["noninferior"] == (c["holm_lower_bound"] > -0.05) or (not c["noninferior"] and c["holm_rank"] > 1)
        assert c["holm_lower_bound"] <= c["ci"][0] + 1e-12                              # stricter than the 95% CI
        assert out["acceptable_cheaper"][s]["lower_bound"] == c["holm_lower_bound"]
    assert all(out["comparisons"][s]["p_two_sided_holm"] is None for s in out["comparisons"] if s not in prim)
    for s, r in out["systems"].items():
        assert 0 <= r["balanced_accuracy"] <= 1 and r["balanced_accuracy_ci"][0] <= r["balanced_accuracy"] <= r["balanced_accuracy_ci"][1]
        assert set(r["rates"]) == {"overuse", "false_restraint", "distractor"}
        assert r["confidence"]["brier"] is not None and r["speed"]["n"] == 10
        assert r["cost"]["usd_list_per_correct"] > 0
        assert r["unusable"] == r["n_items"] - r["usable"] and 0 <= r["unusable_rate"] <= 1
    j = out["systems"]["jev"]
    assert j["cost"]["usd_list_per_answer"] < min(r["cost"]["usd_list_per_answer"] for s, r in out["systems"].items() if s != "jev")
    assert "jev_own_confidence" in j and j["parse"] == {"choice": 180}
    assert sum(out["systems"]["gpt"]["parse"].values()) == 180 and out["systems"]["gpt"]["parse"].get("unparsed", 0) > 0
    assert out["systems"]["gpt"]["prob_status"].get("sum_out_of_range", 0) > 0               # simulator's bad sums
    for s in prim:
        h = out["hybrid"][s]
        assert h["grid"][0]["share_jev"] == 1.0 and h["grid"][-1]["share_jev"] == 0.0
        assert out["acceptable_cheaper"][s]["verdict"] in (True, False)
        assert out["value"][s]["list"]["extra_correct_per_1000_items_ci"][0] <= out["value"][s]["list"]["extra_correct_per_1000_items"]
    for s, r in out["repeatability"].items():                      # repeatability on the repeat set
        assert r["items"] == 40 and r["items_complete"] == 40 and r["measure"] == "fleiss_kappa"
        assert r["fleiss_kappa_ci"][0] <= r["fleiss_kappa"] <= r["fleiss_kappa_ci"][1] and 0 < r["identical_share"] <= 1
    cost = {s: r["cost"] for s, r in out["systems"].items()}         # tokens and cost per answer
    assert cost["glm"]["tokens_reasoning_per_answer"] > 0 and cost["gemma"]["tokens_reasoning_per_answer"] is None
    assert cost["jev"]["tokens_reasoning_per_answer"] is None and cost["gpt"]["tokens_reasoning_per_answer"] == 0
    for s, k in cost.items():
        assert k["usd_list_per_million_answers"] == pytest.approx(k["usd_list_per_answer"] * 1e6)
        assert k["usd_list_per_correct_ci"][0] <= k["usd_list_per_correct"] <= k["usd_list_per_correct_ci"][1]
    for s, v in out["value"].items():
        assert v["verdict_list"] in ("chatbot more accurate", "Jev cheaper and at least as accurate")
        assert v["chatbot_more_accurate"] == (v["list"]["usd_per_extra_correct"] is not None)
    assert set(out["hybrid_cost"]) == set(out["comparisons"])      # every chatbot, at the set share
    assert all(h["n_jev"] == 90 and h["rule"] == "Jev on its most confident 50%" for h in out["hybrid_cost"].values())
    tv = pd.read_csv(res / "table_value.csv").set_index("system")
    assert tv.index.tolist()[0] == "jev" and tv.loc["gpt", "Hybrid: share answered by Jev"] == 0.5
    for s in prim:                                                  # three outcomes
        assert out["comparisons"][s]["outcome"] in ("non-inferior", "worse", "inconclusive")
    hf = out["hybrid_share"]                                        # hybrid, Jev on the half it is most confident on
    assert list(hf["comparisons"]) == prim and len(hf["jev_items"]) == 90 and hf["rule"]["uses_truth"] is False
    jtop = calls[(calls.system == "jev") & (calls["repeat"] == 0)].set_index("item_id").top_prob
    assert jtop[hf["jev_items"]].min() >= jtop.drop(hf["jev_items"]).max()      # Jev's most confident half
    for s, c in hf["comparisons"].items():
        assert c["n_jev"] == 90 and c["outcome"] in ("non-inferior", "worse", "inconclusive")
        assert c["cost_list"]["hybrid_usd_total"] > out["systems"]["jev"]["cost"]["usd_list_total"]
        assert out["hybrid_cost"][s]["saving_usd_per_1000_items"] == pytest.approx(c["cost_list"]["saving_usd_per_1000_items"])
    ro = out["routing"]                                             # the same half handed to Jev at random
    assert ro["rule"]["jev_share"] == 0.5 and set(ro["systems"]) == set(out["comparisons"])
    for s, r in ro["systems"].items():
        assert r["n_jev"] == 90 and set(r) == {"share_jev", "n_jev", "balanced_accuracy", "accuracy"}
        b = r["balanced_accuracy"]
        assert b["confidence_routed"] == pytest.approx(out["hybrid_cost"][s]["balanced_accuracy"])
        assert b["diff_ci"][0] <= b["diff"] <= b["diff_ci"][1]
        cur = out["coverage_curve"]["systems"][s]
        assert [p["n_jev"] for p in cur] == [18, 36, 54, 72, 90, 108, 126, 144, 162]
        assert cur[4]["balanced_accuracy_confidence_routed"] == pytest.approx(b["confidence_routed"])
        assert cur[4]["balanced_accuracy_random_routed"] == pytest.approx(b["random_routed"])
        assert all("usd_list_per_1000_answers" in p and "balanced_accuracy_random_routed" in p
                   for p in out["hybrid"][s]["grid"])
    tr = pd.read_csv(res / "table_routing.csv").set_index("system")
    assert "Confidence minus random, balanced accuracy (low)" in tr.columns and tr.index.tolist()[0] == "jev"
    st = out["style_check"]                                        # style check (descriptive)
    assert st["n_style_matching"] + st["n_style_misleading"] == 180 and st["baseline"]["folds"] == 14
    assert list(st["systems"]) == K.arm_systems(ARM) and 0 <= st["baseline"]["balanced_accuracy"] <= 1
    assert st["baseline"]["answers"] == {"low_value": "B", "appropriate_care": "A"}
    for s, r in st["systems"].items():
        assert r["style_matching"]["n"] + r["style_misleading"]["n"] == 180
    ph = out["post_hoc"]                                           # the post hoc block, descriptive
    rg = ph["rates_by_group"]
    assert len(rg["groups"]) == 14 and rg["chatbot"] == "gpt" and rg["hybrid"]
    assert sum(g["rates"]["jev"]["overuse"]["n"] for g in rg["groups"].values()) == rg["total"]["jev"]["overuse"]["n"]
    assert rg["total"]["jev"]["overuse"]["of"] == 125 and rg["total"]["jev"]["false_restraint"]["of"] == 55
    assert rg["total"]["jev"]["overuse"]["share"] == pytest.approx(out["systems"]["jev"]["rates"]["overuse"])
    assert rg["groups"]["rec_04"]["label"].startswith("synthetic intervention 4")
    hyb = rg["total"]["hybrid"]["overuse"]["n"]                    # the hybrid takes each item's answer from one of the two
    assert hyb <= rg["total"]["jev"]["overuse"]["n"] + rg["total"]["gpt"]["overuse"]["n"]
    cs = ph["cluster_sensitivity"]
    assert cs["n_groups"] == 14 and list(cs["comparisons"]) == prim and len(cs["leave_one_group_out"]) == 14
    for s, c in cs["comparisons"].items():
        assert c["diff"] == pytest.approx(out["comparisons"][s]["diff"]) and c["ci"][0] <= c["ci"][1]
        assert c["outcome"] in ("non-inferior", "worse", "inconclusive")
    hc = cs["hybrid"]                                             # the hybrid and routing on the same group resamples
    assert hc["n_jev"] == out["hybrid_share"]["comparisons"]["gpt"]["n_jev"] and list(hc["comparisons"]) == prim
    for s, c in hc["comparisons"].items():
        assert c["diff"] == pytest.approx(out["hybrid_share"]["comparisons"][s]["diff"]) and c["ci"][0] <= c["ci"][1]
        assert cs["routing"][s]["diff"] == pytest.approx(out["routing"]["systems"][s]["balanced_accuracy"]["diff"])
    assert set(ph["by_category"]) == {"clinical_domain", "evidence_grade"}
    assert set(ph["by_category"]["clinical_domain"]["categories"]) == {"imaging", "prescribing", "screening"}
    assert sum(b["n"] for b in ph["by_category"]["evidence_grade"]["categories"].values()) == 180
    assert ph["certain_errors"]["n_certain"] == out["systems"]["jev"]["certain"]["n"]
    for f in ("table_rates_by_group.md", "table_cluster_sensitivity.md", "table_hybrid_cluster_sensitivity.md",
              "table_leave_one_out.md", "table_certain_errors.md", "table_confidence.md", "table_by_clinical_domain.md", "table_by_evidence_grade.md"):
        assert (res / f).exists(), f
    fr = RC.items_frame(K.load_items(ARM), "main").set_index("item_id")               # features from the sent message
    csv = pd.read_csv(SI.items_path("arm2")).set_index("item_id").prompt_sent
    assert (fr.state == csv.loc[fr.index]).all()
    rerun = RC.cmd_run(ARM, RC.Paths(ARM, tmp_path),                                          # a rerun reads the store
                       sender=lambda *a: (_ for _ in ()).throw(AssertionError("resent")))
    assert rerun.valid.sum() == calls.valid.sum()


def test_plan_projection_and_stop_rule(capsys):
    tot = RC.cmd_plan(ARM)
    assert 0 < tot and ARM.config["budget"]["arm_stop_usd"] is None                  # null: no stop
    items, sys_, rep, calls, *_ = RC.design(ARM)
    p = RC.project_cost(ARM, items, calls).set_index("system")
    assert p.loc["gpt", "route"] == "batch" and p.loc["glm", "route"] == "standard" and p.loc["jev", "tokens_out"] == 0
    assert p.loc["gpt", "calls"] == 180 + 80
    std = RC.project_cost(ARM, items, calls, standard=True).set_index("system")
    m = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    f, bt = m["families"]["gpt"], m["batch"]["families"]["gpt"]
    tin, tout = p.loc["gpt", "tokens_in"], p.loc["gpt", "tokens_out"]
    assert std.loc["gpt", "usd"] == pytest.approx((tin * f["price_in"] + tout * f["price_out"]) / 1e6)    # list price
    assert p.loc["gpt", "usd"] == pytest.approx((tin * bt["price_in"] + tout * bt["price_out"]) / 1e6)    # batch price
    low = dict(ARM.config, budget={**ARM.config["budget"], "arm_stop_usd": 0.01})
    import dataclasses
    with pytest.raises(SystemExit, match="not run"):
        RC.check_budget(dataclasses.replace(ARM, config=low), p.reset_index())


def score_arm(tmp_path: Path) -> K.ArmSpec:
    names = ("0", "I", "IIA", "IIB", "IIC", "IIIA", "IIIB", "IIIC", "IVA-IVB", "IVC")   # arm 3's ten levels
    levels = [{"name": n, "definition": f"criteria for {n}"} for n in names]
    rows = [{"report_id": f"r{k:03d}", "report": f"Synthetic report {k}. Findings consistent with stage {names[k % 10]}.",
             "stage": names[k % 10]} for k in range(100)]
    pd.DataFrame(rows).to_csv(tmp_path / "reports.csv", index=False)
    cfg = {"arm": "armS", "kind": "score", "population": "synthetic_reports", "seed": 20260922,
           "instructions": "Assign the stage group.", "answer_key": "stage", "levels": levels,
           "answer_aliases": {"IVA": "IVA-IVB", "IVB": "IVA-IVB"},
           "variants": {"names": {"definitions": False}, "definitions": {"definitions": True}},
           "primary_variant": "names",
           "items": {"main": str(tmp_path / "reports.csv"), "id_column": "report_id", "state_column": "report",
                     "truth_column": "stage", "stratum_column": "stage"},
           "repeats": {"n_items": 5, "n_repeats": 3},
           "systems": {"tiers": ["primary"]}, "route": {"batch": False}, "jev": {"question_id": "stage_group"},
           "prompts": {"system": "You will see a report.", "reply": 'Reply with JSON {"stage": ..., "probabilities": {...}}.',
                       "user_template": "Report:\n{state}\n\n{instructions}\n\n{options}"},
           "analysis": {"primary_metric": "exact_accuracy", "strata_column": "stage", "n_boot": 200, "ci": 0.95,
                        "margin": 0.05, "coverages": [0.5, 0.8], "ece_bins": 10, "min_errors_auroc": 20},
           "timing": {"n": 5, "window_min": 0}, "budget": {"arm_stop_usd": None, "tokens_out": {"default": 500}}}
    p = tmp_path / "armS.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return K.load_arm(p)


def test_timing_in_two_windows(tmp_path):
    """A later timing window for some systems joins the table."""
    paths, hf = RC.Paths(ARM, tmp_path), ["medgemma", "gemma"]
    sender = SimulatedSender(ARM, K.load_items(ARM))
    first = [s for s in K.arm_systems(ARM) if s not in hf]
    RC.cmd_timing(ARM, paths, n=4, window_min=0, date="t", sender=sender, systems=first)
    assert set(pd.read_parquet(paths.results / "timing_sample.parquet").system) == set(first)
    for _ in range(2):                                                # a repeated piece replaces its own rows
        RC.cmd_timing(ARM, paths, n=4, window_min=0, date="t", sender=sender, systems=hf)
    df = pd.read_parquet(paths.results / "timing_sample.parquet")
    assert len(df) == 11 * 4 and set(df.system) == set(K.arm_systems(ARM))
    assert df.groupby("system").item_id.apply(tuple).nunique() == 1  # the same items for every system
    w = df.groupby("system").window_start.agg(["min", "max"])
    assert (w["min"] == w["max"]).all() and w.loc[hf, "min"].min() >= w.loc[first, "min"].max()
    js = json.loads((paths.results / "timing_sample.json").read_text(encoding="utf-8"))
    assert set(js["windows"]) == set(js["systems"]) == set(K.arm_systems(ARM))
    assert all(v["n"] == 4 for v in js["systems"].values())


def test_score_arm_dry_run(tmp_path):
    arm = score_arm(tmp_path)
    assert len(arm.labels) == 10 and arm.labels[-2:] == ("IVA-IVB", "IVC")
    items = K.load_items(arm)
    assert len(items) == 200 and {i.variant for i in items} == {"names", "definitions"}
    paths = RC.Paths(arm, tmp_path / "out")
    sender = SimulatedSender(arm, items)
    df = RC.cmd_run(arm, paths, sender=sender)
    assert len(df) == 6 * 200 + 6 * 5 * 2 * 2 and df.valid.mean() > 0.9
    design = json.loads((paths.results / "design.json").read_text(encoding="utf-8"))
    jq = design["jev_question_example"]
    assert design["n_items_by_variant"] == {"definitions": 100, "names": 100}
    assert jq["type"] == "score" and len(jq["criteria"]) == 10
    jev = df[df.system == "jev"]
    assert set(jev.answer.dropna()) <= set(arm.labels) and jev.expected_level.between(0, 9).all()
    raw = json.loads(Path(jev.raw_path.iloc[0]).read_text(encoding="utf-8"))
    assert sorted(raw["response"]["answers"]["stage_group"]["probabilities"], key=int) == [str(i) for i in range(10)]
    out = RC.cmd_analyze(arm, paths)
    assert (paths.results / "analysis_names.json").exists() and (paths.results / "analysis_definitions.json").exists()
    assert out["variant"] == "names" and out["primary_metric"] == "exact_accuracy" and out["simulated"] is True
    j = out["systems"]["jev"]
    assert j["qwk"] is not None and j["expected_level_mae"] is not None and "rps" in j["confidence"]
    assert 0 <= j["off_by_one"] <= 1 and "style_check" not in out and "hybrid_share" not in out   # arm 2 only
    assert "answer_how" in out["systems"]["gpt"] and out["systems"]["gpt"]["prob_alias"] == 0
    r = out["repeatability"]["jev"]                                   # ICC(2,1) of the level index
    assert r["measure"] == "icc_2_1" and r["items_complete"] == 5 and "fleiss_kappa" not in r
    assert {h["rule"] for h in out["hybrid_cost"].values()} == {
        "first decile of Jev's top probability within the margin of the chatbot alone"}
    ro = out["routing"]                                               # no set share: the random-half control at 50%
    assert ro["rule"]["jev_share"] == 0.5 and all(set(r) == {"share_jev", "n_jev", "exact_accuracy"}
                                                   for r in ro["systems"].values())
    RC.cmd_timing(arm, paths, sender=sender, date="t")
    tabs = RC.cmd_tables(arm, paths)                                  # both versions: names unsuffixed
    assert {f.name for f in tabs} == {f"table_{n}{v}.{e}" for n in ("repeatability", "cost", "value", "routing")
                                      for v in ("", "_definitions") for e in ("csv", "md")}
    assert "ICC(2,1) of the level index" in (paths.results / "table_repeatability.md").read_text(encoding="utf-8")

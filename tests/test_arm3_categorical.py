"""Arm 3 through the shared categorical code (src/jevity/categorical.py, scripts/run_categorical.py): the arm spec loads
with 10 levels, Jev's Score question takes them without an override, the items are the item file's rows, the reply
rule (IVA or IVB counts as IVA-IVB) holds, and a simulated run and analysis complete on both question versions. On
synthetic reports (tests/synthetic_items.py). No network."""
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
from jevity import arm3_metrics as A3
from jevity import categorical as K
from jevity import categorical_analysis as KA
from jevity import clients as C
from jevity import crc
import synthetic_items as SI

INSTR = ("Assign the pathological stage group for this colorectal cancer according to the AJCC Cancer Staging Manual, "
         "8th edition. Stage IVA-IVB here includes IVA and IVB.")
SYSTEM = ("Choose the stage group that is correct, and give the probability that each stage group is correct, as numbers "
          "between 0 and 1 that add up to 1. Always choose one stage group. Reply with a JSON object with two keys, "
          "\"stage\" (the stage group you choose, written exactly as in the list) and \"probabilities\" (an object with "
          "one number for each stage group), and nothing else.")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(C, "_post", boom)
    monkeypatch.setattr(C.httpx, "Client", boom)


@pytest.fixture(scope="module")
def arm():
    return SI.arm("arm3")


@pytest.fixture(scope="module")
def items(arm):
    return K.load_items(arm)


def test_arm_loads_with_ten_levels(arm):
    assert arm.kind == "score" and arm.answer_key == "stage" and arm.instructions == INSTR
    assert arm.labels == tuple(crc.LEVELS) and len(arm.labels) == 10 == K.JEV_SCORE_MAX_LEVELS
    assert "jev_score_max_levels" not in arm.config                 # the shared 10-level cap applies unchanged


def test_items_are_the_item_file_rows(items):
    """Both versions on the whole pool of 1,000 reports, each item its row of the item file."""
    reports = pd.read_csv(SI.items_path("arm3"), dtype=str, keep_default_na=False).set_index("report_id")
    by_v = {v: [i for i in items if i.variant == v] for v in ("names", "definitions")}
    assert len(by_v["names"]) == 1000 and len(by_v["definitions"]) == 1000 and len(items) == 2000
    assert {i.item_id for i in by_v["definitions"]} == {i.item_id for i in by_v["names"]} == set(reports.index)
    assert pd.Series([i.truth for i in by_v["definitions"]]).value_counts().to_dict() == {lv: 100 for lv in crc.LEVELS}
    for it in items:
        r = reports.loc[it.item_id]
        assert it.state == r["report"].strip() and it.truth == r["level"] and it.stratum == r["level"]
        assert it.group == r["substage"]
    by_level = pd.Series([i.truth for i in items if i.variant == "names"]).value_counts().to_dict()
    assert by_level == {lv: 100 for lv in crc.LEVELS}


def test_the_main_run_plan(arm, items):
    """Names 1,000 + definitions 1,000, and the same 40 repeats twice more in both versions: 2,160 calls per
    system."""
    import run_categorical as RC
    its, sys_, rep, calls = RC.design(arm)
    assert len(rep) == 40
    per = pd.Series([c.system for c in calls]).value_counts()
    assert (per == 1000 + 1000 + 40 * 2 * 2).all() and set(per.index) == set(K.arm_systems(arm))


def test_jev_score_question_both_versions(arm, items):
    names = next(i for i in items if i.variant == "names")
    defs = next(i for i in items if i.variant == "definitions")
    qn, qd = K.jev_question(arm, names), K.jev_question(arm, defs)
    assert qn == {"type": "score", "instructions": INSTR, "criteria": crc.LEVELS}
    assert qd["type"] == "score" and len(qd["criteria"]) == 10
    assert qd["criteria"][6] == "IIIB: T3-T4a N1a-N1c M0; T2-T3 N2a M0; T1-T2 N2b M0"
    assert qd["criteria"][8:] == ["IVA-IVB: any T, any N, M1a-M1b", "IVC: any T, any N, M1c"]
    import dataclasses                                               # more than 10 levels is refused
    eleven = dataclasses.replace(arm, labels=tuple(crc.STAGE_GROUPS))
    with pytest.raises(ValueError, match="2-10 levels"):
        K.jev_question(eleven, dataclasses.replace(names, presented=eleven.labels, canonical=eleven.labels,
                                                   texts=(None,) * 11))


def test_chatbot_prompt_and_schema(arm, items):
    it = next(i for i in items if i.variant == "definitions")
    assert K.system_text(arm) == SYSTEM                               # exactly this wording
    u = K.user_text(arm, it)
    assert u.startswith("Pathology report:\n" + it.state) and f"Question: {INSTR}" in u
    assert u.rstrip().endswith("IVA-IVB: any T, any N, M1a-M1b\nIVC: any T, any N, M1c")
    sch = K.reply_schema(arm, it.presented)["schema"]
    assert sch["properties"]["stage"]["enum"] == crc.LEVELS
    assert sch["properties"]["probabilities"]["required"] == crc.LEVELS


def test_jev_score_answer_on_ten_levels(arm, items):
    it = next(i for i in items if i.truth == "IIIB")
    p = {str(i): 0.0 for i in range(10)}
    p.update({"5": 0.2, "6": 0.7, "7": 0.1})
    r = K.parse_jev_answer({"type": "score", "score": 5.9, "confidence": 0.6, "probabilities": p}, arm, it)
    assert r["parse"] == "score" and r["answer"] == "IIIB" and r["expected_level"] == pytest.approx(5.9)
    assert K.parse_jev_answer({"probabilities": {"9": 1.0}}, arm, it)["answer"] == "IVC"
    assert K.parse_jev_answer({"probabilities": {"10": 1.0}}, arm, it)["parse"] == "score_invalid"   # an 11th level
    c = K.parse_chatbot_answer(json.dumps({"stage": "Stage IVC", "probabilities": {lv: (1.0 if lv == "IVC" else 0.0)
                                                                                   for lv in crc.LEVELS}}),
                               arm.labels, "stage")
    assert c["answer"] == "IVC" and c["prob_status"] == "ok"


def test_reply_rule_iva_or_ivb_counts_as_iva_ivb(arm, items, tmp_path):
    """An answer of IVA or IVB counts as IVA-IVB for every system, and how often is logged."""
    it = next(i for i in items if i.truth == "IVA-IVB" and i.variant == "names")
    probs = {**{lv: 0.0 for lv in crc.LEVELS if lv != "IVA-IVB"}, "IVA": 0.6, "IVB": 0.4}
    reply = {"choices": [{"message": {"content": json.dumps({"stage": "Stage IVA", "probabilities": probs})},
                          "finish_reason": "stop"}], "usage": {}}
    bot = K.CategoricalLLMClient(K.arm_systems(arm)[1], C.RawStore(tmp_path), arm, standard=True,
                                 sender=lambda system, req: dict(reply))
    row = bot.answer(it)
    assert row["answer"] == "IVA-IVB" and row["answer_how"] == "alias" and row["valid"]
    assert row["probs"]["IVA-IVB"] == pytest.approx(1.0) and row["prob_alias"] is True
    log = A3.alias_log([{**row, "system": "gpt", "variant": it.variant}])
    assert log[0]["alias_answers"] == 1 and log[0]["alias_probability_lists"] == 1


def test_arm3_metrics_agree_with_the_shared_analysis():
    rng = np.random.default_rng(20260922)
    y = rng.integers(0, 10, 300)
    yhat = np.clip(y + rng.integers(-2, 3, 300), 0, 9)
    truth, pred = [crc.LEVELS[i] for i in y], [crc.LEVELS[i] for i in yhat]
    assert A3.quadratic_weighted_kappa(truth, pred) == pytest.approx(KA.quadratic_weighted_kappa(y, yhat, 10))
    assert A3.off_by_one_rate(truth, pred) == pytest.approx(float((np.abs(yhat - y) == 1).mean()))


def test_simulated_run_and_analysis_on_a_subset(tmp_path):
    """Five reports per level (50), primary tier, both versions: run through the real clients with a simulated sender, then
    the analysis. Nothing is written inside the repository."""
    import run_categorical as RC
    from jevity.categorical_sim import SimulatedSender
    cfg = yaml.safe_load((ROOT / "config" / "arm3.yaml").read_text(encoding="utf-8"))
    full = pd.read_csv(SI.items_path("arm3"), dtype=str, keep_default_na=False)
    sub = full.groupby("level", sort=False).head(5)
    sub.to_csv(tmp_path / "items.csv", index=False, lineterminator="\n")
    cfg["items"]["main"] = str(tmp_path / "items.csv")
    cfg["items"].pop("repeat_column")                                    # draw 5 repeats from the subset
    cfg.update(repeats={"n_items": 5, "n_repeats": 3}, systems={"tiers": ["primary"]}, route={"batch": False},
               timing={"n": 5, "window_min": 0})
    cfg["analysis"]["n_boot"] = 200
    (tmp_path / "arm3.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    arm = K.load_arm(tmp_path / "arm3.yaml")
    its = K.load_items(arm)
    paths = RC.Paths(arm, tmp_path / "out")
    df = RC.cmd_run(arm, paths, sender=SimulatedSender(arm, its))
    n_sys = len(K.arm_systems(arm))
    assert len(df) == n_sys * 50 * 2 + n_sys * 5 * 2 * 2 and df.valid.mean() > 0.9
    out = RC.cmd_analyze(arm, paths)
    assert out["variant"] == "names" and out["primary_metric"] == "exact_accuracy"
    assert (paths.results / "analysis_definitions.json").exists()
    j = out["systems"]["jev"]
    assert j["qwk"] is not None and j["expected_level_mae"] is not None and "rps" in j["confidence"]

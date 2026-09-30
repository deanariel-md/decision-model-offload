"""Arm 3, Jev used as documented, structure only (src/jevity/docuse_structure_only.py, scripts/docuse/
run_structure_only.py): the request is the documented request with every criterion null, the spending cap, the calls
table from a raw store and the metrics. On synthetic reports (tests/synthetic_items.py) with synthetic Jev responses;
no network."""
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from jevity import categorical as K
from jevity import docuse_staging as D
from jevity import docuse_structure_only as S
from jevity.clients import MODELS, RawStore
import synthetic_items as SI


def _script():
    path = ROOT / "scripts" / "docuse" / "run_structure_only.py"
    spec = importlib.util.spec_from_file_location("run_structure_only", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _script()


@pytest.fixture(scope="module", autouse=True)
def synthetic_reports():
    """config/docuse_staging.yaml's item files and the main and confuser sets on the synthetic reports."""
    with pytest.MonkeyPatch.context() as mp:
        SI.use_in_docuse(mp)
        mp.setitem(S.SETS, "main", dataclasses.replace(S.SETS["main"], items_csv=SI.items_path("arm3")))
        mp.setitem(S.SETS, "confuser",
                   dataclasses.replace(S.SETS["confuser"], items_csv=SI.items_path("arm3_confuser")))
        yield


@pytest.fixture(scope="module")
def arm():
    return M.arm_spec()


@pytest.fixture(scope="module")
def main_items():
    return S.load_items("main")


def test_label_only_nulls_every_criterion_and_keeps_the_rest():
    qs = {"q1": {"instructions": "Pick one.", "criteria": {"b": {"what": "x"}, "a": "y", "c": None}, "extra": 1},
          "q2": {"instructions": "Other.", "criteria": {"z": "w"}}}
    out = S.label_only(qs)
    assert list(out) == ["q1", "q2"]
    assert list(out["q1"]["criteria"]) == ["b", "a", "c"] and all(v is None for v in out["q1"]["criteria"].values())
    assert out["q1"]["instructions"] == "Pick one." and out["q1"]["extra"] == 1
    assert qs["q1"]["criteria"]["a"] == "y"                    # the input is not changed


def test_no_cap_by_default(arm):
    assert S.SPEND_CAP_USD is None
    assert arm.config["budget"]["arm_stop_usd"] is None


def test_request_is_the_documented_request_with_null_criteria(arm, main_items, tmp_path):
    cl = S.StructureOnlyJevClient(RawStore(tmp_path / "store"), arm)
    items = main_items[:25] + S.load_items("confuser")[:5]
    seen = set()
    for it in items:
        req = cl.request(it)
        assert S.validate_request(req, it, arm) == []
        doc = S.documented_request(it, arm)
        assert req["_variant"] == S.VARIANT and doc["_variant"] == S.BASE_VARIANT
        assert D.body(req) == {**D.body(doc), "questions": S.label_only(D.body(doc)["questions"])}
        for qid in D.QUESTION_IDS:
            q, dq = req["questions"][qid], doc["questions"][qid]
            assert tuple(q["criteria"]) == D.options(qid) == tuple(dq["criteria"])
            assert all(v is None for v in q["criteria"].values())
            assert q["instructions"] == dq["instructions"]
        paths = S.differing_paths(D.body(doc), D.body(req))
        assert paths and all(p[0] == "questions" and p[2] == "criteria" for p in paths)
        seen.add(K._sha(req))
    assert len(seen) == len(items)


def test_validate_request_catches_a_criterion_left_in(arm, main_items, tmp_path):
    it = main_items[0]
    req = S.StructureOnlyJevClient(RawStore(tmp_path / "store"), arm).request(it)
    qid = D.QUESTION_IDS[0]
    bad = json.loads(json.dumps(req))
    bad["questions"][qid]["criteria"][D.options(qid)[0]] = "a description"
    problems = S.validate_request(bad, it, arm)
    assert any("null descriptions" in p for p in problems)
    assert not any("outside the criteria" in p for p in problems)
    bad = json.loads(json.dumps(req))
    bad["state"] = {"pathology_report": "another text"}
    assert any("outside the criteria" in p for p in S.validate_request(bad, it, arm))


def test_differing_paths():
    a = {"x": {"y": [1, 2]}, "z": 1}
    b = {"x": {"y": [1, 3]}, "w": 2, "z": 1}
    assert S.differing_paths(a, b) == [("w",), ("x", "y", 1)]
    assert S.differing_paths(a, a) == []


def test_spend_guard_with_and_without_a_cap():
    g = S.SpendGuard(None, spent=5.0)
    g.reserve(100.0)
    g.settle(100.0, 90.0)
    assert g.held == 0.0 and g.spent == 95.0
    g = S.SpendGuard(1.0, spent=0.5)
    g.reserve(0.4)
    with pytest.raises(RuntimeError, match="spend cap"):
        g.reserve(0.2)
    g.settle(0.4, 0.3)
    g.reserve(0.2)
    assert g.spent == pytest.approx(0.8) and g.held == pytest.approx(0.2)


def test_plan_and_paths(main_items, tmp_path):
    calls = M.plan("main", "full", main_items)
    assert [c.item_id for c in calls] == S.ordered_ids("main", main_items)
    assert {(c.system, c.repeat, c.variant) for c in calls} == {("jev", 0, S.VARIANT)} and len(calls) == 1000
    with pytest.raises(ValueError):
        M.plan("main", "repeats", main_items)
    p = M.Paths(tmp_path, simulated=True)
    assert p.writing == tmp_path / "results" / "summaries" / "ablation_staging.json"
    assert p.results == tmp_path / "results" / "arm3_docuse_structure_only"
    assert p.store("heldout").run_dir == tmp_path / "runs" / "arm3_heldout_structure_only"
    assert p.store("main").run_dir == p.store("confuser").run_dir
    assert M.Paths(simulated=False).writing == ROOT / "results" / "summaries" / "ablation_staging.json"


def test_calls_table_from_a_synthetic_store(arm, main_items, tmp_path):
    store = RawStore(tmp_path / "runs" / "arm3_docuse_structure_only")
    tp = S.truth_parts("main")
    fac = S.make_factory(arm, store, sender=D.synthetic_sender(tp, 0))
    sent = main_items[:30]
    for it in sent:
        fac("jev", 0).answer(it)
    calls, parts = S.calls_table(store, arm, main_items[:40])
    assert list(calls.columns) == list(S.CALL_COLUMNS)
    assert list(calls.item_id) == [i.item_id for i in sent] and set(calls.variant) == {S.VARIANT}
    assert set(calls.system) == {"jev"} and set(calls["repeat"]) == {0} and set(calls.arm) == {arm.name}
    assert list(parts.item_id) == list(calls.item_id)
    for q in D.QUESTION_IDS:
        assert parts[f"{q}_choice"].isin(D.options(q)).all()
    assert S.stored_spend(store) == 0.0
    again, _ = S.calls_table(store, arm, main_items[:40])
    pd.testing.assert_frame_equal(calls, again)


def test_metrics_on_synthetic_answers(main_items):
    items = main_items[:6]
    tp = S.truth_parts("main")
    ids = sorted(i.item_id for i in items)
    by = {i.item_id: i for i in items}
    other = {i: D.LEVELS[(D.LEVELS.index(by[i].truth) + 1) % len(D.LEVELS)] for i in ids}
    # (answer right?, valid, top_prob, tokens_in, questions answered wrong); the sixth report has no answer
    plan = [(True, True, 1.0, 100, ()), (True, True, 0.9, 200, ()), (False, True, 1.0, 300, ("t_category",)),
            (False, True, 0.5, 400, ("distant_metastasis",)), (False, False, np.nan, 500, tuple(M.QKEYS))]
    rows, prows = [], []
    for iid, (right, valid, top, tin, wrong_q) in zip(ids, plan):
        rows.append({"item_id": iid, "repeat": 0, "system": "jev", "valid": valid,
                     "answer": (by[iid].truth if right else other[iid]) if valid else None, "top_prob": top,
                     "tokens_in": tin, "usd_reported": tin * 1e-8})
        pr = {"item_id": iid, "repeat": 0}
        for q in M.QKEYS:
            pr[f"{q}_choice"] = None if not valid else ("wrong option" if q in wrong_q else tp[iid][q])
        prows.append(pr)
    out = M.metrics(pd.DataFrame(rows), pd.DataFrame(prows), items, tp)
    w, x = out["writing"], out["extra"]
    assert w["n"] == 6 and w["wrong"] == 4 and w["exact_accuracy"] == pytest.approx(2 / 6)
    assert w["ci"] == pytest.approx(M.wilson(2, 6))
    assert w["full_conf_share"] == pytest.approx(2 / 6)
    assert (w["sensitivity"]["k"], w["sensitivity"]["n"]) == (3, 4)
    assert (w["specificity"]["k"], w["specificity"]["n"]) == (1, 2)
    assert w["per_question"] == pytest.approx({"t": 3 / 6, "nodes": 4 / 6, "deposits": 4 / 6, "m": 3 / 6})
    price = float(MODELS["jev"]["price_per_mtok_input"])
    assert w["usd_per_1000"] == pytest.approx(300 * price / 1e6 * 1000)
    assert x["answered"] == 5 and x["usable"] == 4 and x["unusable_or_missing"] == 2
    assert x["stage_wrong_by_number_of_wrong_parts"] == {"0": 0, "1": 2, "2": 0, "3": 0, "4": 2}
    assert x["wrong_by_question"] == {"t": 3, "nodes": 2, "deposits": 2, "m": 3}

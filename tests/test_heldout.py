"""Arm 3 held-out reports: the tested systems' requests (arm 3's and the documented builders, only the report
differing) and the analysis end to end on synthetic reports and answers (tests/synthetic_items.py). No network."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "heldout"))
sys.path.insert(0, str(ROOT / "scripts" / "docuse"))

from jevity import categorical as K
from jevity import docuse_staging as D
from jevity import heldout as H
from jevity.clients import MODELS, RawStore
import synthetic_items as SI


def _body(req):
    return {k: v for k, v in req.items() if not k.startswith("_")}


def _script(name):
    """scripts/heldout/<name>.py under a module name of its own (scripts/analyze.py is another module "analyze")."""
    spec = importlib.util.spec_from_file_location(f"heldout_{name}", ROOT / "scripts" / "heldout" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ tested systems
def _arm3_item_pair(variant):
    a3 = SI.arm("arm3")
    it = next(i for i in K.load_items(a3) if i.variant == variant)
    return a3, it


@pytest.mark.parametrize("variant", ["names", "definitions"])
@pytest.mark.parametrize("system", ["jev", "gpt", "claude", "gemini", "muse", "glm"])
def test_arm3_requests_differ_only_in_the_report_and_the_route(tmp_path, system, variant):
    """The held-out request is arm 3's request with another report; the chatbots on the standard route (the provider
    pinned, allow_fallbacks false) where arm 3's GPT, Claude and Gemini went on the batch route."""
    a3, it = _arm3_item_pair(variant)
    report = "HELD-OUT REPORT\nLine two"
    harm = H.heldout_arm()
    new_it = K.Item("hld999", report, it.truth, it.presented, it.canonical, it.texts, it.stratum, it.group, it.variant)
    old_it = K.Item("hld999", report, it.truth, it.presented, it.canonical, it.texts, it.stratum, it.group, it.variant)
    store = RawStore(tmp_path)
    new = _body(K.make_factory(harm, store, standard=True)(system, 0).request(new_it))
    old = _body(K.make_factory(a3, store, standard=True)(system, 0).request(old_it))
    assert new == old
    orig = _body(K.make_factory(a3, store)(system, 0).request(old_it))       # as stored in arm 3 (batch where it was)
    assert {k: v for k, v in new.items() if k != "provider"} == {k: v for k, v in orig.items() if k != "provider"}
    if system != "jev":
        assert new["provider"] == {"only": MODELS["families"][system]["provider_only"], "allow_fallbacks": False}
    else:
        assert new["provider"]["allow_fallbacks"] is False


@pytest.mark.parametrize("variant", D.VARIANTS)
def test_documented_request_is_the_template_filled(tmp_path, variant):
    frame = pd.DataFrame([{"report_id": "hld999", "level": "IIA", "substage": "IIA", "report": " A report.\n"}])
    it = next(i for i in H.documented_items(frame) if i.variant == variant)
    req = D.DocuseJevClient(RawStore(tmp_path), D.arm_spec()).request(it)
    assert D.body(req) == D.fill(D.template(variant), "A report.")
    assert D.validate_request(req, it) == []


# ------------------------------------------------------------------------------------------------ analysis end to end
def test_analysis_end_to_end_on_synthetic_answers(tmp_path):
    """Synthetic answers through the real clients, stores and tables, then scripts/heldout/analyze.py: every summary
    number traces to its file."""
    A = _script("analyze")
    R = _script("run")
    from jevity.categorical_sim import SimulatedSender
    src = pd.read_csv(SI.items_path("arm3"), dtype=str, keep_default_na=False)
    src = src.groupby("level", group_keys=False).head(6).reset_index(drop=True)
    frame = pd.DataFrame({"report_id": [f"hld{i + 1:03d}" for i in range(len(src))], "level": src.level,
                          "substage": src.substage, "t": src.t, "n": src.n, "m": src.m,
                          "nodes_involved": src.nodes_involved.astype(int), "nodes_examined": src.nodes_examined.astype(int),
                          "tumour_deposits": src.tumour_deposits.astype(int), "m_sites": "", "layout": "narrative",
                          "report": src.report})
    items_path = tmp_path / "data" / "arm3_heldout" / "items.csv"
    items_path.parent.mkdir(parents=True)
    frame.to_csv(items_path, index=False, lineterminator="\n")
    frame = H.load_items_frame(items_path)
    store = RawStore(tmp_path / "runs")
    ids = frame.report_id.tolist()
    doc, a3 = R.system_plan(ids, R.ALL_SYSTEMS)
    ditems = H.documented_items(frame)
    K.execute(doc, ditems, D.make_factory(D.arm_spec(), store, sender=D.synthetic_sender(H.truth_parts(frame))), None,
              workers=2)
    harm = H.heldout_arm(items_path)
    aitems = K.load_items(harm)
    fac = K.make_factory(harm, store, sender=SimulatedSender(harm, aitems), standard=True)
    K.execute(a3, aitems, fac, None, workers=2)
    res = tmp_path / "results" / "arm3_heldout"
    res.mkdir(parents=True)
    dcalls, parts = D.calls_table(store, D.arm_spec(), ditems, doc)
    dcalls.to_parquet(res / "documented_calls.parquet", index=False)
    parts.to_parquet(res / "parts.parquet", index=False)
    R.stored_rows(a3, aitems, fac, harm.name).to_parquet(res / "calls.parquet", index=False)
    s = A.run(tmp_path, n_boot=60)
    assert s["n_reports"] == 60 and s["checker"]["summary_values_match_files"] > 100
    for arm in A.JEV_ARMS:
        for v in A.VERSIONS:
            vs = s["jev_arms"][arm][v]["vs_chatbots"]
            assert set(vs) == set(H.MAIN_CHATBOTS)
            assert all(c["outcome"][0] in ("non-inferior", "inconclusive", "worse") for c in vs.values())
    assert A.check(json.loads((res / "summary.json").read_text(encoding="utf-8")), res) == \
        s["checker"]["summary_values_match_files"]

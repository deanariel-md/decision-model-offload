"""Non-regional node sites the notes do not name (scripts/nonregional/run_nonregional.py): the sites, and the requests
(the stored runs' requests with only the report text changed), on 20 synthetic reports (tests/synthetic_items.py) in a
scratch repository root. No network."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("run_nonregional", ROOT / "scripts" / "nonregional" / "run_nonregional.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

from jevity import categorical as K
from jevity import docuse_staging as D
from jevity.clients import RawStore
import synthetic_items as SI


@pytest.fixture
def scratch_root(tmp_path, monkeypatch):
    """A repository root holding the config and a synthetic items file where the script reads it
    (data/arm3_nonregional_sites/items.csv): the script and the shared loader read the items there."""
    shutil.copytree(ROOT / "config", tmp_path / "config")
    items = tmp_path / "data" / "arm3_nonregional_sites" / "items.csv"
    items.parent.mkdir(parents=True)
    SI.nonregional_frame(R.SITES).to_csv(items, index=False, lineterminator="\n")
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "ITEMS", items)
    monkeypatch.setattr(K, "ROOT", tmp_path)
    return tmp_path


def test_sites_are_on_no_regional_list():
    lists = [n.lower() for v in R.SPEC["regional_nodes"].values() for n in v]
    for s in R.SITES:
        assert not any(s in n or n in s for n in lists), s
    assert all(x["source"] for x in R.SPEC["sites"])


def test_sites_named_in_no_note_criterion_or_example():
    text = D.CONFIG_PATH.read_text(encoding="utf-8").lower()
    for v in D.VARIANTS:
        sent = json.dumps(D.questions(v), ensure_ascii=False).lower()
        for s in R.SITES:
            assert s not in sent and s not in text, (v, s)
    for n in D.notes():
        assert not any(s in n["text"].lower() for s in R.SITES)


def test_requests_differ_from_the_stored_runs_only_in_the_report(scratch_root):
    ji, jc, ci, cc = R.plan(["jev"] + R.CHATBOTS)
    arm = R.chatbot_arm()
    assert arm.name == "arm3_confuser" and arm.config["variants"] == {"names": {"definitions": False}}
    assert arm.config["items"]["main"] == "data/arm3_nonregional_sites/items.csv"
    assert {i.variant for i in ci} == {"names"} and len(ci) == 20
    assert len(jc) == 40 and len(cc) == 100
    base = K.load_arm("arm3_confuser")
    assert (K.system_text(arm), arm.instructions, arm.labels) == (K.system_text(base), base.instructions, base.labels)
    f = K.make_factory(arm, RawStore(scratch_root / "runs"), standard=True)
    req = f("gpt").request(ci[0])
    assert req["response_format"]["json_schema"]["name"] == "arm3_confuser_answer"
    assert req["messages"][1]["content"].startswith("Pathology report:\n" + ci[0].state)
    assert req["provider"]["allow_fallbacks"] is False
    cl = D.DocuseJevClient(RawStore(scratch_root / "runs"), D.arm_spec())
    for v in D.VARIANTS:
        it = next(i for i in ji if i.variant == v)
        assert not D.validate_request(cl.request(it), it)
        assert cl.request(it)["questions"] == D.questions(v)
    assert not (scratch_root / "runs").exists()                     # building requests stores nothing

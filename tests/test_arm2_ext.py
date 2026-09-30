"""Arm 2's second item set: MedQA and Medbullets as MedHELM loads them, every question, two option counts, the
next-step subgroup. Loader, prompts, a simulated end-to-end run on a slice of the real items, the extra analyses and
tables, and the batch route. Skipped until data/arm2_ext/items.csv is built (python scripts/build_arm2_ext_items.py
--download). No network: any HTTP attempt fails the test."""
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from jevity import categorical as K
from jevity import categorical_analysis as KA
from jevity import clients as C
from jevity.categorical_sim import SimulatedSender
import run_categorical as RC
import build_arm2_ext_items as B

if not (ROOT / "data" / "arm2_ext" / "items.csv").exists():     # the question text is not ours to ship: build it first
    pytest.skip("no item file: python scripts/build_arm2_ext_items.py --download", allow_module_level=True)
ARM = K.load_arm("arm2_ext")
ITEMS = pd.read_csv(ROOT / "data" / "arm2_ext" / "items.csv", dtype=str, keep_default_na=False)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(C, "_post", boom)
    monkeypatch.setattr(C.httpx, "Client", boom)


def small_arm(tmp_path, n_medqa=24, n_mb=16) -> K.ArmSpec:
    """The arm on a slice of the real items (both sources, both subgroup labels), 200 resamples."""
    df = pd.concat([ITEMS[ITEMS.source == "medqa"].head(n_medqa), ITEMS[ITEMS.source == "medbullets"].head(n_mb)])
    p = tmp_path / "items.csv"
    df.to_csv(p, index=False, lineterminator="\n")
    c = ARM.config
    return dataclasses.replace(ARM, config={**c, "items": {**c["items"], "main": str(p)},
                                            "analysis": {**c["analysis"], "n_boot": 200},
                                            "budget": {**c["budget"], "manifest_roots": []}})


def test_items_as_medhelm_loads_them():
    man = json.loads((ROOT / "config" / "arm2_ext" / "items_manifest.json").read_text(encoding="utf-8"))
    assert man["items_by_source"] == {"medbullets": 298, "medqa": 1273} and man["medbullets_repeats_left_out"] == 10
    assert man["options_by_source"] == {"medbullets": 5, "medqa": 4} and man["options_alt_by_source"] == {"medbullets": 4, "medqa": 5}
    # the committed file has LF line endings only; a checkout with core.autocrlf=true writes CRLF
    raw = (ROOT / "data" / "arm2_ext" / "items.csv").read_bytes()
    assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == man["items_sha256"]
    items = K.load_items(ARM)
    assert {i.variant for i in items} == {"medhelm", "other_count"} and len(items) == 2 * 1571
    by = {(i.item_id, i.variant): i for i in items}
    a, b = by[("medqa_0000", "medhelm")], by[("medqa_0000", "other_count")]
    assert a.presented == ("A", "B", "C", "D") and b.presented == ("A", "B", "C", "D", "E")
    assert a.texts[a.presented.index(a.truth)] == b.texts[b.presented.index(b.truth)]      # same key, other letter maybe
    assert set(a.texts) < set(b.texts) and a.state == b.state
    m = by[("medbullets_000", "medhelm")]
    assert len(m.presented) == 5 and len(by[("medbullets_000", "other_count")].presented) == 4
    assert int(ITEMS.decision_question.eq("True").sum()) == 359


def test_lead_in_rule():
    s = ARM.config["source"]
    keep, drop = s["subgroup_patterns"], s["subgroup_exclude"]
    assert B.lead_in_type("Which next step would you take now?", keep, drop) == "next_step"
    assert B.lead_in_type("Which drug is the best treatment for her now?", keep, drop) == "treatment"
    assert B.lead_in_type("What is the most appropriate management for her now?", keep, drop) == "management"
    assert B.lead_in_type("What is the likeliest diagnosis here?", keep, drop) == ""
    assert B.lead_in_type("Which enzyme acts at the next step of this pathway?", keep, drop) == ""
    stem = "A man has a cough.\nSerum: Na+: 138 mEq/L K+: 4.2 mEq/L What is the likeliest diagnosis here?"
    assert B.question_sentence(stem) == "What is the likeliest diagnosis here?"


def test_prompts_are_the_basic_skeleton():
    it = next(i for i in K.load_items(ARM) if i.item_id == "medqa_0744" and i.variant == "medhelm")
    assert K.system_text(ARM).startswith("Answer this medical question, and give the probability that each option")
    u = K.user_text(ARM, it)
    assert u == it.state + "\n\n" + "\n".join(f"{l}. {t}" for l, t in zip(it.presented, it.texts)) + "\n"
    q = K.jev_question(ARM, it)
    assert q == {"type": "choice", "instructions": "Answer this medical question.",
                 "criteria": dict(zip(it.presented, it.texts))}
    assert K.reply_schema(ARM, it.presented)["schema"]["properties"]["choice"]["enum"] == ["A", "B", "C", "D"]


def test_simulated_run_analysis_and_tables(tmp_path):
    arm = small_arm(tmp_path)
    paths = RC.Paths(arm, tmp_path / "out")
    items = K.load_items(arm)
    sender = SimulatedSender(arm, items)
    df = RC.cmd_run(arm, paths, sender=sender)
    assert len(df) == 11 * 40 * 2 and df.valid.mean() > 0.9
    RC.cmd_timing(arm, paths, n=5, window_min=0, date="t", sender=sender)
    res = RC.cmd_analyze(arm, paths)
    assert res["primary_metric"] == "accuracy" and res["n_items"] == 40 and set(res["strata"]) == {"medqa", "medbullets"}
    j = res["systems"]["jev"]
    assert set(j["accuracy_by_stratum_ci"]) == {"medqa", "medbullets"} and "certain" in j
    ce = j["certain"]
    assert ce["at"] == 0.995 and 0 <= ce["share"] <= 1 and ce["n"] + (40 - ce["n"]) == 40
    assert res["hybrid_share"]["comparisons"]["gpt"]["n_jev"] == 20
    assert len(res["coverage_curve"]["systems"]["gpt"]) == 9 and "repeatability" not in res
    sub = json.loads((paths.results / "analysis_medhelm_decision.json").read_text(encoding="utf-8"))
    n_dec = int(pd.read_csv(arm.config["items"]["main"], dtype=str).decision_question.eq("True").sum())
    assert sub["n_items"] == n_dec and sub["subset"]["name"] == "decision"
    assert (paths.results / "analysis_other_count.json").exists() and (paths.results / "analysis_other_count_decision.json").exists()
    o = json.loads((paths.results / "options.json").read_text(encoding="utf-8"))
    assert o["n_items"] == 40 and o["options_by_stratum"]["medqa"] == {"medhelm": [4], "other_count": [5]}
    assert set(o["systems"]) == set(K.arm_systems(arm)) and "gpt" in o["jev_minus_chatbot"]
    g = o["systems"]["gpt"]
    assert g["diff"] == pytest.approx(g["accuracy_longer"] - g["accuracy_shorter"])
    made = RC.cmd_tables(arm, paths)
    names = {p.name for p in made}
    for n in ("table_accuracy.md", "table_hybrid.md", "table_cost.md", "table_value.md", "table_routing.md",
              "table_options.md", "table_options_decision.md", "table_accuracy_medhelm_decision.md",
              "table_accuracy_other_count.md"):
        assert n in names, n
    md = (paths.results / "table_accuracy.md").read_text(encoding="utf-8")
    assert "probability 1.00" in md and "medqa (24)" in md
    assert "table_repeatability.md" not in names
    ph = res["post_hoc"]                                                    # the post hoc block, descriptive
    assert ph["note"] == KA.POST_HOC_NOTE
    assert set(ph["by_category"]["exam"]["categories"]) <= {"medqa step1", "medqa step2&3", "medbullets"}
    assert sum(b["n"] for b in ph["by_category"]["exam"]["categories"].values()) == 40
    assert ph["certain_errors"]["n_certain"] == ce["n"] and ph["certain_errors"]["n_wrong"] == len(ph["certain_errors"]["items"])
    for n in ("table_confidence.md", "table_by_exam.md", "table_certain_errors.md"):
        assert n in names, n
    assert "post hoc" in (paths.results / "table_confidence.md").read_text(encoding="utf-8").lower()


def test_table_fills_labels_an_item_does_not_offer(tmp_path):
    arm = small_arm(tmp_path, 4, 4)
    paths = RC.Paths(arm, tmp_path / "out")
    items = K.load_items(arm)
    df = RC.cmd_run(arm, paths, sender=SimulatedSender(arm, items), systems=None)
    f = RC.items_frame(items, "medhelm")
    t = KA.Table(df, f, K.arm_systems(arm), arm.labels, C.MODELS, "medhelm")
    mq = f.source.eq("medqa").to_numpy() if "source" in f else np.array([i.startswith("medqa") for i in f.item_id])
    j = t.col("gpt")
    ok = ~np.isnan(t.P[:, j, 0])
    assert (t.P[mq & ok, j, 4] == 0).all()                            # MedQA offers A-D: no mass on E


def test_partial_runs_write_their_own_table(tmp_path):
    arm = small_arm(tmp_path, 3, 2)
    paths = RC.Paths(arm, tmp_path / "out")
    sender = SimulatedSender(arm, K.load_items(arm))
    RC.cmd_run(arm, paths, systems=["medgemma", "gemma"], sender=sender)
    assert (paths.results / "calls_part_medgemma_gemma.parquet").exists() and not paths.calls().exists()
    RC.cmd_run(arm, paths, sender=sender)
    assert len(pd.read_parquet(paths.calls())) == 11 * 5 * 2


def test_batches_split_by_answer_schema_and_rejections_not_counted(tmp_path):
    """Google's batch endpoint rejects a batch mixing 4- and 5-option answer schemas as a whole ("cannot share an
    upstream input ... Split the requests into one batch per configuration"): one batch per family and schema, and such
    a rejection is not an attempt of its requests (none was processed)."""
    from jevity import batch as BT
    arm = small_arm(tmp_path, 3, 2)
    paths = RC.Paths(arm, tmp_path / "out")
    store = paths.store()
    items, sys_, rep, calls = RC.design(arm)
    todo = [t for t in K.pending_batch(calls, items, arm, store) if t[0] == "gemini"]
    sent = []

    def post(payload):
        sent.append(payload)
        return {"id": f"b{len(sent)}", "status": "validating"}
    acc = lambda: {"key_usage": 0.0, "key_limit": None, "key_limit_remaining": None, "balance": 1000.0}
    made = K.submit_batch(todo, store, arm, post=post, account=acc)
    assert len(made) == 2                                              # a 4-option and a 5-option batch
    for p in sent:
        enums = {json.dumps(r["body"]["response_format"]["json_schema"]["schema"]["properties"]["choice"]["enum"])
                 for r in p["requests"]}
        assert len(enums) == 1
    man = BT.Manifest(store)
    b = man.items[0]
    b.update(status="failed", ingested=True)
    man.save()
    (man.dir / f"{b['key']}.result.json").write_text(json.dumps({"status": "failed", "request_counts": {"completed": 0},
        "error": {"message": "Batch request 'x' cannot share an upstream input with the requests before it"}}), encoding="utf-8")
    man = BT.Manifest(store)
    assert K.config_rejections(man) == {b["key"]}
    cids = set(man.requests(b["key"]))
    assert all(man.attempts()[c] == 1 for c in cids) and not any(c in K.batch_attempts(man) for c in cids)

def test_embedded_json_reading():
    """A reply that reasons first and ends with the requested JSON is read from that object (config
    parse.embedded_json, every system of the second item set); off by default."""
    lab = ("A", "B", "C", "D")
    js = '{"choice": "B", "probabilities": {"A": 0.05, "B": 0.85, "C": 0.05, "D": 0.05}}'
    for reply in (f"Looking at this case {{briefly}}: the answer is B.\n\n```json\n{js}\n```",
                  f"**Analysis:** ... so B.\n\n{js}",
                  f'An aside {{"choice": "A", "probabilities": {{"A": 1}}}} then the final answer:\n{js}'):
        p = K.parse_chatbot_answer(reply, lab, "choice", embedded=True)
        assert p["parse"] == "json_embedded" and p["answer"] == "B" and p["probs"]["B"] == pytest.approx(0.85)
        assert K.parse_chatbot_answer(reply, lab, "choice")["parse"] == "unparsed"          # off: not read
    assert K.parse_chatbot_answer("The answer is B.", lab, "choice", embedded=True)["parse"] == "unparsed"
    assert K.parse_chatbot_answer(js, lab, "choice", embedded=True)["parse"] == "json"      # one object: plain JSON
    assert ARM.config["parse"]["embedded_json"] is True and "parse" not in K.load_arm("arm2").config
def test_two_timing_processes_share_one_table(tmp_path):
    """Systems timed by several processes in the same window: every process writes the one timing table under a
    lock file, none dropping another's rows."""
    import threading
    arm = small_arm(tmp_path)
    paths = RC.Paths(arm, tmp_path / "out")
    sender = SimulatedSender(arm, K.load_items(arm))
    groups = [["jev", "gpt", "glm"], ["muse"], ["gemma"]]
    ts = [threading.Thread(target=RC.cmd_timing, args=(arm, paths),
                           kwargs=dict(n=4, window_min=0, date="t", sender=sender, systems=g)) for g in groups]
    [t.start() for t in ts]
    [t.join() for t in ts]
    df = pd.read_parquet(paths.results / "timing_sample.parquet")
    assert sorted(df.system.unique()) == sorted(s for g in groups for s in g) and len(df) == 5 * 4
    assert not (paths.results / "timing_sample.lock").exists()
    summ = json.loads((paths.results / "timing_sample.json").read_text(encoding="utf-8"))
    assert set(summ["windows"]) == set(df.system)

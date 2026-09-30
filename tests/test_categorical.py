"""Arm 2 design and the shared categorical clients: the item loader and its option texts, the seeded shuffle, request
shapes for Jev and the chatbots, the reply parsers, the raw-store round trip, the batch route, the keys and the batch
budget, on synthetic items (tests/synthetic_items.py). No network: any HTTP attempt fails the test."""
import dataclasses
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
from jevity import categorical as K
from jevity import clients as C
from jevity.clients import RawStore, BATCH_MISSING
import synthetic_items as SI

ARM = SI.arm("arm2")
ITEMS_CSV = SI.items_path("arm2")
CFG = yaml.safe_load((ROOT / "config" / "arm2.yaml").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(C, "_post", boom)
    monkeypatch.setattr(C.httpx, "Client", boom)


def with_budget(roots=(), **budget) -> K.ArmSpec:
    """The arm with its batch holds read from the given run roots only (none by default) and other budget settings."""
    return dataclasses.replace(ARM, config={**ARM.config, "budget": {**ARM.config["budget"], **budget,
                                                                     "manifest_roots": [str(r) for r in roots]}})


RICH = lambda: {"key_usage": 100.0, "key_limit": 1000, "key_limit_remaining": 900.0, "balance": 900.0}


def items():
    return K.load_items(ARM)


# ------------------------------------------------------------------------------------------------ options and shuffle
def test_shuffle_is_per_item_and_deterministic():
    a = K.shuffle_order("rec_01_ac_01")
    assert a == K.shuffle_order("rec_01_ac_01") and sorted(a) == ["A", "B", "C", "D"]
    assert a == ("A", "B", "D", "C")                                           # the seeded order; never changes
    orders = [K.shuffle_order(f"x{i}") for i in range(2400)]
    assert len(set(orders)) == 24                                              # every order occurs
    firsts = pd.Series([o[0] for o in orders]).value_counts()
    assert firsts.min() > 500                                                  # A-D equally often first


def test_items_carry_options_in_presented_order(tmp_path):
    its = items()
    rows = pd.read_csv(ITEMS_CSV, dtype=str, keep_default_na=False).set_index("item_id")
    assert len(its) == 180
    it = next(i for i in its if i.item_id == "rec_01_ac_02")                   # stored order DCAB
    assert it.canonical == ("D", "C", "A", "B") and it.presented == ("A", "B", "C", "D")
    assert it.texts == tuple(rows.loc["rec_01_ac_02", f"option_{c}"] for c in "DCAB")
    assert it.truth == "A" and it.to_canonical("C") == "A" and it.stratum == "appropriate_care"
    assert it.group == "rec_01" and it.state == rows.loc["rec_01_ac_02", "prompt_sent"].strip()

    def load(frame):
        frame.to_csv(tmp_path / "bad.csv", index=False)
        return K.load_items(dataclasses.replace(ARM, config={**ARM.config, "items": {**ARM.config["items"],
                                                                                    "main": str(tmp_path / "bad.csv")}}))
    bad = pd.read_csv(ITEMS_CSV, dtype=str, keep_default_na=False)
    bad.loc[0, "presented_order"] = "BACD" if bad.loc[0, "presented_order"] != "BACD" else "ABCD"
    with pytest.raises(AssertionError, match="seeded shuffle"):               # a hand-edited order is refused
        load(bad)
    for b, c in (("same", "same"), ("", None)):                                # a repeated or an empty option text
        bad = pd.read_csv(ITEMS_CSV, dtype=str, keep_default_na=False)
        bad.loc[0, "option_B"] = b
        if c is not None:
            bad.loc[0, "option_C"] = c
        with pytest.raises(AssertionError, match="not distinct"):
            load(bad)
    assert K.repeat_ids(ARM) == sorted(pd.read_csv(ITEMS_CSV).query("repeat_set").item_id)


def test_systems_and_plan():
    s = K.arm_systems(ARM)
    assert s == ["jev", "gpt", "claude", "gemini", "muse", "glm", "gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
    calls = K.plan_calls(items(), s, K.repeat_ids(ARM), 3)
    assert len(calls) == 11 * 180 + 11 * 40 * 2
    assert len(set(calls)) == len(calls)
    first = {c.item_id for c in calls[:11 * 50]}
    rank = {}
    for k, c in enumerate(calls):                                              # blocks of 50 items, all systems each
        rank.setdefault(c.item_id, k)
    blocks = [c.item_id for c in calls]
    ids_in_order = list(dict.fromkeys(blocks))
    assert len(ids_in_order) == 180
    b0 = set(ids_in_order[:50])
    last_b0 = max(i for i, c in enumerate(calls) if c.item_id in b0)
    assert all(c.item_id in b0 for c in calls[:last_b0 + 1])
    assert first <= b0


# ------------------------------------------------------------------------------------------------ requests
def test_chatbot_request(tmp_path, monkeypatch):
    store = RawStore(tmp_path)
    it = next(i for i in items() if i.item_id == "rec_01_ac_02")
    req = K.CategoricalLLMClient("gpt", store, ARM, standard=True).request(it)
    assert req["model"] == C.MODELS["families"]["gpt"]["slug"]
    assert req["provider"] == {"only": ["openai"], "allow_fallbacks": False}
    assert "temperature" not in req and "reasoning" not in req and req["seed"] == 20260922 and req["max_tokens"] == 16000
    sysm, user = req["messages"][0]["content"], req["messages"][1]["content"]
    assert sysm == ("Choose the option that is the most appropriate next step, and give the probability that each "
                    "option is the most appropriate next step, as numbers between 0 and 1 that add up to 1. Always "
                    "choose one option. Reply with a JSON object with two keys, \"choice\" (the letter of the option "
                    "you choose) and \"probabilities\" (an object with one number for each letter), and nothing else.")
    assert user.startswith("Message:\n" + it.state) and "Question: Which is the most appropriate next step for this person?" in user
    lines = user.splitlines()[-4:]
    assert [l[:3] for l in lines] == ["A. ", "B. ", "C. ", "D. "] and lines[2][3:] == it.texts[2]
    sch = req["response_format"]["json_schema"]
    assert sch["strict"] and sch["schema"]["properties"]["choice"]["enum"] == ["A", "B", "C", "D"]
    assert sch["schema"]["properties"]["probabilities"]["required"] == ["A", "B", "C", "D"]
    assert req["_item"] == "rec_01_ac_02" and req["_arm"] == "arm2"
    assert "seed" not in K.CategoricalLLMClient("claude", store, ARM, standard=True).request(it)
    b = K.CategoricalLLMClient("gpt", store, ARM)
    assert b.batch is not None and b.request(it)["provider"] == {"only": ["openai"]} and b.request(it)["_route"] == "batch"
    assert K.CategoricalLLMClient("glm", store, ARM).batch is None                     # no batch endpoint
    assert K.CategoricalLLMClient("claude_free", store, ARM).batch is None             # Sonnet 5 synchronous (arm 1)
    assert K.CategoricalLLMClient("gpt", store, dataclasses.replace(ARM, batch=False)).batch is None
    hf = K.CategoricalLLMClient("medgemma", store, ARM).request(it)
    assert "provider" not in hf and hf["model"] == C.MODELS["families"]["medgemma"]["hf_model"]
    r1 = K.CategoricalLLMClient("gpt", store, ARM, repeat=1, standard=True).request(it)
    assert C._sha(r1) != C._sha(req)                                                  # repeats are distinct requests
    monkeypatch.setitem(C.MODELS["families"], "retired", {**C.MODELS["families"]["gpt"], "disabled": True})
    with pytest.raises(RuntimeError):
        K.CategoricalLLMClient("retired", store, ARM)                                 # disabled family


def test_jev_request(tmp_path):
    it = next(i for i in items() if i.item_id == "rec_01_ac_02")
    req = K.CategoricalJevClient(RawStore(tmp_path), ARM).request(it)
    assert req["model"] == C.MODELS["jev"]["slug"] and req["state"] == it.state
    q = req["questions"]
    assert list(q) == ["next_step"] and q["next_step"]["type"] == "choice"
    assert q["next_step"]["instructions"] == "Which is the most appropriate next step for this person?"
    assert list(q["next_step"]["criteria"]) == ["A", "B", "C", "D"]
    assert list(q["next_step"]["criteria"].values()) == list(it.texts)
    assert req["provider"] == {"allow_fallbacks": False}


# ------------------------------------------------------------------------------------------------ parsing
L = ("A", "B", "C", "D")


@pytest.mark.parametrize("content,answer,status,parse", [
    ('{"choice": "B", "probabilities": {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}}', "B", "ok", "json"),
    ('```json\n{"choice": "b", "probabilities": {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}}\n```', "B", "ok", "json"),
    ('{"choice": "Option C.", "probabilities": {"A": 0.1, "B": 0.1, "C": 0.75, "D": 0.1}}', "C", "renormalised", "json"),
    ('{"choice": "(A)", "probabilities": {"A": 0.5, "B": 0.2, "C": 0.1, "D": 0.1}}', "A", "renormalised", "json"),
    ('{"choice": "A", "probabilities": {"A": 0.7, "B": 0.25}}', "A", "renormalised", "json"),        # omitted: zero
    ('{"choice": "A", "probabilities": {"A": 0.5, "B": 0.1, "C": 0.1, "D": 0.1}}', "A", "sum_out_of_range", "json"),
    ('{"choice": "A", "probabilities": {"A": 1.2, "B": 0, "C": 0, "D": 0}}', "A", "values_invalid", "json"),
    ('{"choice": "A", "probabilities": {"A": true, "B": 0, "C": 0, "D": 0}}', "A", "values_invalid", "json"),
    ('{"choice": "A", "probabilities": {"E": 1}}', "A", "labels_unknown", "json"),
    ('{"choice": "A", "probabilities": {"A": 1, "a": 0}}', "A", "labels_duplicate", "json"),
    ('{"choice": "E", "probabilities": {"A": 1, "B": 0, "C": 0, "D": 0}}', None, "ok", "json_invalid_answer"),
    ('{"answer": "A", "probabilities": {"A": 1, "B": 0, "C": 0, "D": 0}}', None, "ok", "json_no_answer"),
    ('{"choice": 1}', None, "missing", "json_invalid_answer"),
    ("The best answer is B.", None, "missing", "unparsed"),
    ('["A"]', None, "missing", "json_not_object"),
    ("", None, "missing", "empty"),
    (None, None, "missing", "empty"),
])
def test_parse_chatbot(content, answer, status, parse):
    r = K.parse_chatbot_answer(content, L, "choice")
    assert (r["answer"], r["prob_status"], r["parse"]) == (answer, status, parse)
    if r["probs"]:
        assert abs(sum(r["probs"].values()) - 1) < 1e-12


def test_match_label_for_stage_names():
    lv = ("0", "I", "IIA", "IIB", "IIIA", "IVC")
    assert K.match_label("Stage IIIA", lv) == ("IIIA", "normalised")
    assert K.match_label("iia", lv) == ("IIA", "case")
    assert K.match_label("stage 0", lv) == ("0", "normalised")
    assert K.match_label("III", lv) == (None, "unknown_label")
    assert K.match_label(3, lv) == (None, "not_string")


def test_parse_jev_choice():
    it = next(i for i in items() if i.item_id == "rec_01_ac_02")
    ans = {"type": "choice", "choice": "C", "probabilities": {"A": 0.05, "B": 0, "C": 0.9, "D": 0.05}, "confidence": 0.93}
    r = K.parse_jev_answer(ans, ARM, it)
    assert r["answer"] == "C" and r["jev_confidence"] == 0.93 and r["parse"] == "choice"
    row = K._row(r, it, ARM, {"usage": {"input_tokens": 300, "output_tokens": 80, "cost": 1.3e-5}}, {"latency_s": 3.2}, "x")
    assert row["answer"] == "A" and row["answer_presented"] == "C" and row["probs"]["A"] == 0.9    # canonical labels
    assert row["top_prob"] == 0.9 and row["tokens_in"] == 300 and row["usd_reported"] == 1.3e-5 and not row["choice_not_top"]
    assert K.parse_jev_answer(None, ARM, it)["parse"] == "missing_answer"
    assert K.parse_jev_answer({"choice": "Z"}, ARM, it)["parse"] == "choice_invalid"


def score_arm(n_levels: int = 5, **cfg) -> K.ArmSpec:
    levels = [{"name": f"L{i}", "definition": f"definition {i}"} for i in range(n_levels)]
    c = {**ARM.config, "arm": "armS", "kind": "score", "answer_key": "stage", "levels": levels,
         "variants": {"names": {"definitions": False}, "definitions": {"definitions": True}}, **cfg}
    return dataclasses.replace(ARM, name="armS", kind="score", answer_key="stage", labels=tuple(l["name"] for l in levels),
                               config=c)


def test_score_question_and_parser():
    arm = score_arm(5)
    it = K.Item("r1", "report", "L2", arm.labels, arm.labels, (None,) * 5, "L2", "", "names")
    q = K.jev_question(arm, it)
    assert q == {"type": "score", "instructions": arm.instructions, "criteria": ["L0", "L1", "L2", "L3", "L4"]}
    itd = dataclasses.replace(it, texts=tuple(f"definition {i}" for i in range(5)))
    assert K.jev_question(arm, itd)["criteria"][1] == "L1: definition 1"
    assert K.options_block(arm, itd).splitlines()[4] == "L4: definition 4"
    ans = {"type": "score", "score": 2.3, "confidence": 0.5, "probabilities": {"0": 0, "1": 0.1, "2": 0.5, "3": 0.4, "4": 0}}
    r = K.parse_jev_answer(ans, arm, it)
    assert r["answer"] == "L2" and r["expected_level"] == 2.3 and r["parse"] == "score"
    tie = {"score": 2.6, "probabilities": {"0": 0, "1": 0, "2": 0.45, "3": 0.45, "4": 0.1}}
    assert K.parse_jev_answer(tie, arm, it)["answer"] == "L3"                      # tie: nearest the expected level
    off = {"probabilities": {"0": 0, "1": 0, "2": 0.4, "3": 0.2, "4": 0}}         # sum 0.6: answer counts, no confidence
    r = K.parse_jev_answer(off, arm, it)
    assert r["answer"] == "L2" and r["probs"] is None and r["prob_status"] == "sum_out_of_range"
    assert K.parse_jev_answer({"probabilities": {"0": 0, "1": 0}}, arm, it)["parse"] == "score_invalid"
    assert K.parse_jev_answer({"probabilities": {"7": 1}}, arm, it)["parse"] == "score_invalid"
    c = K.parse_chatbot_answer('{"stage": "Stage L3", "probabilities": {"L0": 0, "L1": 0, "L2": 0.3, "L3": 0.7, "L4": 0}}',
                               arm.labels, "stage")
    assert c["answer"] == "L3" and c["prob_status"] == "ok"
    row = K._row(c, it, arm, {}, {}, None)
    assert abs(row["expected_level"] - 2.7) < 1e-12


def test_score_level_limit():
    arm = score_arm(11)
    it = K.Item("r1", "report", "L2", arm.labels, arm.labels, (None,) * 11)
    with pytest.raises(ValueError, match="2-10 levels"):
        K.jev_question(arm, it)
    arm_ok = score_arm(11, jev_score_max_levels=11)
    assert len(K.jev_question(arm_ok, it)["criteria"]) == 11


# ------------------------------------------------------------------------------------------------ timing, store, batch
def test_timing_hold(tmp_path):
    """Config timing.hold stops a live timing sample before any key is read."""
    import run_categorical as RC
    assert "hold" not in ARM.config["timing"]
    held = dataclasses.replace(ARM, config={**ARM.config, "timing": {**ARM.config["timing"], "hold": "not for this arm"}})
    with pytest.raises(SystemExit, match="timing sample on hold"):
        RC.cmd_timing(held, RC.Paths(held, tmp_path), yes=True)


def _completion(content, usage=(400, 150)):
    return {"id": "gen-1", "model": "openai/gpt-5.6-sol-20260709", "provider": "OpenAI",
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def test_answer_reads_the_raw_store(tmp_path):
    store = RawStore(tmp_path)
    it = next(i for i in items() if i.item_id == "rec_01_ac_02")           # presented DCAB: C shows canonical A
    cl = K.CategoricalLLMClient("glm", store, ARM)
    req = cl.request(it)
    store.put(req, _completion('{"choice": "C", "probabilities": {"A": 0.1, "B": 0.1, "C": 0.7, "D": 0.1}}'),
              {"latency_s": 12.5, "route": "standard"})
    r = cl.answer(it)
    assert r["answer"] == "A" and r["valid"] and r["probs"] == {"D": 0.1, "C": 0.1, "A": 0.7, "B": 0.1}
    assert r["latency_s"] == 12.5 and r["tokens_out"] == 150 and r["finish_reason"] == "stop" and r["raw_path"].endswith(".json")
    b = K.CategoricalLLMClient("gpt", store, ARM)                          # batch family, nothing collected yet
    assert b.answer(it)["error"] == BATCH_MISSING and b.answer(it)["route"] == "batch"
    assert r["prob_alias"] is False and b.answer(it)["prob_alias"] is False           # arm 2 has no aliases


def test_answer_aliases_score_arm(tmp_path):
    """A Score arm's reply rule: IVA or IVB counts as IVA-IVB, for the answer and for probability keys,
    and is logged (answer_how 'alias', prob_alias). Jev is unaffected (its Score answers are level indices)."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_arm2_dry_run import score_arm
    arm = score_arm(tmp_path)
    L10, al = arm.labels, arm.config["answer_aliases"]
    assert len(L10) == 10 and al == {"IVA": "IVA-IVB", "IVB": "IVA-IVB"}
    r = K.parse_chatbot_answer('{"stage": "Stage IVA", "probabilities": {"IVA": 0.3, "IVB": 0.2, "IVC": 0.5}}',
                               L10, "stage", al)
    assert (r["answer"], r["answer_how"], r["parse"], r["prob_status"], r["prob_alias"]) == \
        ("IVA-IVB", "alias", "json", "ok", True)
    assert r["probs"]["IVA-IVB"] == pytest.approx(0.5) and r["probs"]["IVC"] == pytest.approx(0.5)
    r = K.parse_chatbot_answer('{"stage": "ivb", "probabilities": {"IVA-IVB": 0.4, "iva": 0.1, "IVC": 0.4}}',
                               L10, "stage", al)                    # level and alias add up before the sum rule
    assert r["answer"] == "IVA-IVB" and r["prob_status"] == "renormalised" and r["prob_alias"]
    assert r["given"]["IVA-IVB"] == pytest.approx(0.5) and r["probs"]["IVA-IVB"] == pytest.approx(0.5 / 0.9)
    r = K.parse_chatbot_answer('{"stage": "IVA", "probabilities": {"IVA": 0.3, "IVC": 0.3}}', L10, "stage", al)
    assert r["answer"] == "IVA-IVB" and r["prob_status"] == "sum_out_of_range"      # the rule still applies
    r = K.parse_chatbot_answer('{"stage": "IVA-IVB", "probabilities": {"IVA-IVB": 0.5, "IVC": 0.5}}', L10, "stage", al)
    assert r["answer_how"] == "exact" and r["prob_alias"] is False
    r = K.parse_chatbot_answer('{"stage": "IVC", "probabilities": {"IVC": 0.5, "ivc": 0.5}}', L10, "stage", al)
    assert r["prob_status"] == "labels_duplicate"                                    # a level named twice still fails
    r = K.parse_chatbot_answer('{"stage": "IVA", "probabilities": {"IVA": 0.5, "IVC": 0.5}}', L10, "stage")
    assert r["answer"] is None and r["parse"] == "json_invalid_answer" and r["prob_status"] == "labels_unknown"
    it = next(i for i in K.load_items(arm) if i.variant == "names")
    store = RawStore(tmp_path / "runs")
    cl = K.CategoricalLLMClient("glm", store, arm)
    store.put(cl.request(it), _completion('{"stage": "IVB", "probabilities": {"IVB": 0.6, "IVC": 0.4}}'),
              {"route": "standard"})
    row = cl.answer(it)
    assert (row["answer"], row["answer_how"], row["prob_alias"], row["valid"]) == ("IVA-IVB", "alias", True, True)
    assert row["probs"]["IVA-IVB"] == pytest.approx(0.6)
    p = tmp_path / "bad.yaml"
    for bad in ({"IVA": "IV"}, {"IVC": "IVA-IVB"}):                  # target must be a level; alias must not be one
        p.write_text(yaml.safe_dump({**arm.config, "answer_aliases": bad}, sort_keys=False), encoding="utf-8")
        with pytest.raises(AssertionError, match="answer_aliases"):
            K.load_arm(p)
    p.write_text(yaml.safe_dump({**CFG, "answer_aliases": {"E": "A"}}, sort_keys=False), encoding="utf-8")
    with pytest.raises(AssertionError, match="Score arms only"):
        K.load_arm(p)


def test_batch_submit_and_collect(tmp_path):
    from jevity import batch as BT
    arm = with_budget()
    store = RawStore(tmp_path / "runs")
    its = items()[:3]
    calls = [K.CatCall(s, it.item_id, r) for s in ("jev", "gpt", "glm", "claude") for it in its for r in (0, 1)]
    todo = K.pending_batch(calls, its, arm, store)
    assert sorted({f for f, _, _ in todo}) == ["claude", "gpt"] and len(todo) == 12   # Jev and GLM synchronous
    sent = []

    def post(payload):
        sent.append(payload)
        return {"id": f"batch-{len(sent)}", "status": "validating"}
    made = K.submit_batch(todo, store, arm, post=post, account=RICH)
    assert len(made) == 2 and {p["model"] for p in sent} == {"openai/gpt-5.6-sol", "anthropic/claude-opus-5.5"}
    body = sent[0]["requests"][0]["body"]
    assert "model" not in body and "provider" not in body and not any(k.startswith("_") for k in body)
    assert body["response_format"]["json_schema"]["name"] == "arm2_answer"
    assert K.pending_batch(calls, its, arm, store) and not K.submit_batch(K.pending_batch(calls, its, arm, store), store, arm, post=post, account=RICH)
    reqs = {**BT.Manifest(store).requests(made[0]["key"]), **BT.Manifest(store).requests(made[1]["key"])}

    def get(url, params=None):
        bid = url.rsplit("/", 1)[1]
        key = next(m["key"] for m in made if m["id"] == bid)
        res = [{"custom_id": cid, "response": {"status_code": 200, "body": _completion(
            '{"choice": "A", "probabilities": {"A": 0.6, "B": 0.4, "C": 0, "D": 0}}')}}
            for cid in BT.Manifest(store).requests(key)]
        return {"id": bid, "status": "completed", "results": res}
    t = BT.collect(store, get=get)
    assert t["written"] == 12 and t["failed"] == 0
    assert not K.pending_batch(calls, its, arm, store)
    r = K.CategoricalLLMClient("gpt", store, arm, repeat=1).answer(its[0])
    assert r["valid"] and r["route"] == "batch" and r["answer"] == its[0].to_canonical("A")
    assert len(reqs) == 12


# ------------------------------------------------------------------------------------------------ keys and budget
def test_keys_load_from_the_given_path_only(tmp_path, monkeypatch):
    monkeypatch.delenv("ARM2_TEST_KEY", raising=False)
    env = tmp_path / "main.env"
    env.write_text("ARM2_TEST_KEY=abc\n", encoding="utf-8")
    arm = dataclasses.replace(ARM, config={**ARM.config, "keys": {"env_file": str(env)}})
    assert K.load_keys(arm) == env and __import__("os").environ["ARM2_TEST_KEY"] == "abc"
    with pytest.raises(SystemExit, match="not found"):
        K.load_keys(dataclasses.replace(ARM, config={**ARM.config, "keys": {"env_file": str(tmp_path / "none")}}))
    assert CFG["keys"]["env_file"] == ".env"
    with pytest.raises(SystemExit) as e:                                              # relative: repository root
        K.load_keys(dataclasses.replace(ARM, config={**ARM.config, "keys": {"env_file": "no_such_dir/.env"}}))
    assert str(K.ROOT / "no_such_dir" / ".env") in str(e.value)


def _manifest(run_dir: Path, entries: list[dict], requests: dict | None = None):
    d = run_dir / "batches"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    for e in entries:
        if requests is not None:
            (d / f"{e['key']}.requests.json").write_text(json.dumps(requests), encoding="utf-8")


def test_open_holds_and_budget_rules(tmp_path):
    req = {"messages": [{"role": "user", "content": "x" * 4000}], "max_tokens": 16000}
    _manifest(tmp_path / "runs" / "eval", [{"key": "k1", "family": "claude", "n": 2, "ingested": False, "status": "in_progress"},
                                           {"key": "k2", "family": "gpt", "n": 5, "ingested": True},
                                           {"key": "k3", "family": "gpt", "n": 5, "ingested": True, "status": "dropped"}],
              {"a": req, "b": req})
    _manifest(tmp_path / "runs" / "older", [{"key": "k4", "family": "gemini", "n": 10, "ingested": False,
                                             "status": "posting"}])                 # unclear: counts; no requests file
    files = K.manifest_files([tmp_path / "runs"], [tmp_path / "runs" / "eval"])
    assert len(files) == 2
    h = K.open_holds(files)
    opus = (1000 * 2.0 + 16000 * 10.0) / 1e6
    assert h[K._slash(tmp_path / "runs" / "eval")] == pytest.approx(2 * opus)
    assert h[K._slash(tmp_path / "runs" / "older")] == pytest.approx(10 * (1300 * 0.375 + 16000 * 1.875) / 1e6)
    cfg = {**ARM.config["budget"], "project_stop_usd": 1000}
    ok = K.budget_check({"key_usage": 500, "balance": 400, "key_limit_remaining": None}, h, 50.0, cfg)
    assert ok["ok"] and ok["other_runs_in_flight"] == sorted(h)
    over = K.budget_check({"key_usage": 960, "balance": 4000, "key_limit_remaining": None}, h, 50.0, cfg)
    assert "project_stop_hard" not in cfg and not over["ok"] and "project stop" in over["reasons"][0]   # default: refuses
    soft = K.budget_check({"key_usage": 960, "balance": 4000, "key_limit_remaining": None}, h, 50.0,
                          {**cfg, "project_stop_hard": False})
    assert soft["ok"] and "project stop" in soft["warnings"][0]                     # warns only
    assert ARM.config["budget"]["project_stop_usd"] is None                           # null: no project stop
    free = K.budget_check({"key_usage": 960, "balance": 4000, "key_limit_remaining": None}, h, 50.0, ARM.config["budget"])
    assert free["ok"] and free["stop"] is None and not free["warnings"]
    hard = {**cfg, "holds_in_balance": False}                                       # the default
    poor = K.budget_check({"key_usage": 10, "balance": 50, "key_limit_remaining": None}, h, 50.0, hard)
    assert not poor["ok"] and "balance" in poor["reasons"][0]                       # must cover the other holds too
    assert cfg["holds_in_balance"] is True                                          # arm 2: holds counted once
    assert K.budget_check({"key_usage": 10, "balance": 50, "key_limit_remaining": None}, h, 50.0, cfg)["ok"]
    short = K.budget_check({"key_usage": 10, "balance": 49, "key_limit_remaining": None}, h, 50.0, cfg)
    assert not short["ok"] and "does not cover this submission" in short["reasons"][0]
    lim = K.budget_check({"key_usage": 10, "balance": 900, "key_limit_remaining": 20}, h, 50.0, cfg)
    assert not lim["ok"] and "limit" in lim["reasons"][0]


def test_account_reads(tmp_path):
    def get(url, params=None):
        return {"data": {"usage": 12.5, "limit": 1000, "limit_remaining": 987.5}} if url.endswith("/key") else \
               {"data": {"total_credits": 300.0, "total_usage": 112.5}}
    a = K.openrouter_account(get)
    assert a == {"key_usage": 12.5, "key_limit": 1000, "key_limit_remaining": 987.5, "balance": 187.5}
    with pytest.raises(SystemExit, match="nothing submitted"):
        K.openrouter_account(lambda url, params=None: {"status": "gone"})


def test_batch_refused_over_the_stop_or_balance(tmp_path):
    arm1 = tmp_path / "main" / "runs"
    req = {"messages": [{"role": "user", "content": "y" * 4000}], "max_tokens": 16000}
    _manifest(arm1 / "eval", [{"key": "e1", "family": "claude", "n": 1000, "ingested": False, "status": "in_progress"}],
              {f"c{i}": req for i in range(1000)})                                  # arm 1 evaluation batch in flight
    arm = with_budget(roots=[arm1], project_stop_usd=1000)
    store = RawStore(tmp_path / "runs")
    its = items()[:20]
    todo = K.pending_batch([K.CatCall("claude", it.item_id) for it in its], its, arm, store)
    posted = []
    arm1_hold = 1000 * (1000 * 2.0 + 16000 * 10.0) / 1e6                         # the evaluation batch's hold
    arm.config["budget"]["holds_in_balance"] = False                               # the default
    with pytest.raises(SystemExit, match="does not cover"):                        # balance covers arm 2 alone only
        K.submit_batch(todo, store, arm, post=posted.append,
                       account=lambda: {"key_usage": 100.0, "key_limit_remaining": None, "balance": 60.0})
    arm.config["budget"]["project_stop_hard"] = True                               # the default
    with pytest.raises(SystemExit, match="project stop"):
        K.submit_batch(todo, store, arm, post=posted.append,
                       account=lambda: {"key_usage": 1000 - arm1_hold, "key_limit_remaining": None, "balance": 5000.0})
    assert not posted and not list((tmp_path / "runs" / "batches").glob("*.requests.json"))
    arm.config["budget"]["project_stop_hard"] = False                              # warns only
    over = K.submit_batch(todo, store, arm, post=lambda p: posted.append(p) or {"id": "b0", "status": "validating"},
                          account=lambda: {"key_usage": 1000 - arm1_hold, "key_limit_remaining": None, "balance": 5000.0})
    assert len(over) == 1 and len(posted) == 1
    posted.clear()
    store = RawStore(tmp_path / "runs2")
    made = K.submit_batch(todo, store, arm, post=lambda p: posted.append(p) or {"id": "b1", "status": "validating"},
                          account=lambda: {"key_usage": 100.0, "key_limit_remaining": None, "balance": arm1_hold + 10})
    assert len(made) == 1 and len(posted) == 1


def test_batch_submit_in_pieces(tmp_path):
    """A submission can go in pieces (--models) so each hold fits the balance and the key limit;
    a later piece skips what is already in flight; a system off the batch route is refused."""
    import run_categorical as RC
    from jevity import batch as BT
    arm = with_budget()
    paths = RC.Paths(arm, tmp_path / "out")
    posted = []
    post = lambda p: posted.append(p) or {"id": f"b{len(posted)}", "status": "validating"}
    first = RC.cmd_batch(arm, paths, "submit", [], yes=True, post=post, account=RICH, systems=["claude"])
    assert len(first) == 1 and len(posted) == 1
    rest = RC.cmd_batch(arm, paths, "submit", [], yes=True, post=post, account=RICH, systems=["gpt", "gemini"])
    assert len(rest) == 2 and len(posted) == 3
    fams = sorted(b["family"] for b in BT.Manifest(paths.store()).items)
    assert fams == ["claude", "gemini", "gpt"]
    assert all(b["n"] == 260 for b in BT.Manifest(paths.store()).items)          # 180 items + 2 x 40 repeats
    design = json.loads((paths.results / "design.json").read_text(encoding="utf-8"))
    assert len(design["systems"]) == len(K.arm_systems(arm))                            # every system, not the piece
    with pytest.raises(SystemExit, match="not batch-route"):
        RC.cmd_batch(arm, paths, "submit", [], yes=True, post=post, account=RICH, systems=["glm"])
    last = RC.cmd_batch(arm, paths, "submit", [], yes=True, post=post, account=RICH)    # the remaining pieces
    assert sorted(b["family"] for b in last) == ["gemini_free", "gpt_free"] and len(posted) == 5


# ------------------------------------------------------------------------------------------------ pacing and workers
def test_any_429_pauses_the_whole_family(monkeypatch):
    """Any 429 sets one pause that every worker of the family waits out (the
    Retry-After header when given, else 5 s doubling); no real network and no real sleep."""
    class R:
        def __init__(self, code, headers=None):
            self.status_code, self.headers = code, headers or {}

        def raise_for_status(self):
            pass

        def json(self):
            return {"id": "ok"}

    seq, posted = [R(429, {"retry-after": "7"}), R(429), R(200)], []

    class Client:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers, json):
            posted.append(url)
            return seq.pop(0)

    now, slept = [1000.0], []
    monkeypatch.setattr(C.httpx, "Client", Client)
    monkeypatch.setattr(K.time, "time", lambda: now[0])

    def sleep(s):
        slept.append(s)
        now[0] += s

    pause = K.FamilyPause()
    assert K.post_paced("u", {}, {}, pause, sleep=sleep) == {"id": "ok"}
    assert len(posted) == 3 and pause.hits == 2 and sum(slept) == pytest.approx(7 + 10) and max(slept) <= 5
    pause.hit()                                                   # another worker of the family now waits too
    t0 = now[0]
    pause.wait(sleep)
    assert now[0] - t0 == pytest.approx(10)
    assert K.family_pause("glm") is K.family_pause("glm") and K.family_pause("glm") is not K.family_pause("muse")


def test_worker_caps():
    import run_categorical as RC
    assert CFG["route"]["workers"] == {} and "glm" not in RC.worker_plan(ARM, K.arm_systems(ARM), 16, 8)   # no cap
    capped = dataclasses.replace(ARM, config={**ARM.config, "route": {**ARM.config["route"], "workers": {"glm": 8}}})
    w = RC.worker_plan(capped, K.arm_systems(capped), 16, 8)
    assert w["glm"] == 8 and w["default"] == 16
    assert RC.worker_plan(capped, K.arm_systems(capped), 4, 8)["glm"] == 4                   # a cap, never a floor


# ------------------------------------------------------------------------------------------------ variant row filter
def test_score_variant_rows_filter(tmp_path):
    """A Score variant may keep only the rows whose column equals a value (as text); a missing column stops with an
    error naming the column; Choice arms refuse the filter."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_arm2_dry_run import score_arm
    arm = score_arm(tmp_path)
    df = pd.read_csv(tmp_path / "reports.csv")
    df["with_definitions"] = ["true" if k < 60 else "false" for k in range(len(df))]
    df.to_csv(tmp_path / "reports.csv", index=False)
    var = {"names": {"definitions": False}, "definitions": {"definitions": True, "rows": {"with_definitions": True}}}
    arm = dataclasses.replace(arm, config={**arm.config, "variants": var})
    items = K.load_items(arm)
    n = pd.Series([i.variant for i in items]).value_counts().to_dict()
    assert n == {"names": 100, "definitions": 60}
    assert all(i.texts[0] for i in items if i.variant == "definitions") and all(not i.texts[0] for i in items if i.variant == "names")
    rep = K.repeat_ids(arm)
    calls = K.plan_calls(items, K.arm_systems(arm), rep, 3, 20260922)
    S = len(K.arm_systems(arm))
    per = pd.Series([c.variant for c in calls]).value_counts().to_dict()
    in_def = sum(r in set(df.report_id[:60]) for r in rep)
    assert per == {"names": S * (100 + 2 * len(rep)), "definitions": S * (60 + 2 * in_def)}
    bad = dataclasses.replace(arm, config={**arm.config, "variants": {**var, "definitions": {"definitions": True,
                                                                                           "rows": {"nope": "x"}}}})
    with pytest.raises(ValueError, match="'nope' missing from the items"):
        K.load_items(bad)
    with pytest.raises(AssertionError, match="Score arms only"):
        K.load_items(dataclasses.replace(ARM, config={**ARM.config, "variants": {"main": {"rows": {"x": 1}}}}))

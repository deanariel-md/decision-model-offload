"""llm_forms_audit.match and coverage on synthetic requests: small, fast, no file or network access."""
import copy
import json

import pandas as pd

from jevity import llm_forms_audit as A

SCHEMA = {"type": "object", "properties": {"stage": {"type": "string", "enum": ["0", "I"]},
                                           "probabilities": {"type": "object"}},
          "required": ["stage", "probabilities"], "additionalProperties": False}


def req(**kw):
    r = {"model": "openai/gpt-5.6-sol",
         "messages": [{"role": "system", "content": "Choose the stage group."},
                      {"role": "user", "content": "Pathology report:\nR1\n\nQuestion: Q\n\nStage groups:\n0\nI\n"}],
         "max_tokens": 16000, "provider": {"only": ["openai"], "allow_fallbacks": False}, "_repeat": 0,
         "seed": 20260922,
         "response_format": {"type": "json_schema",
                             "json_schema": {"name": "arm3_answer", "strict": True, "schema": copy.deepcopy(SCHEMA)}},
         "_arm": "arm3_llm_forms", "_item": "crc001", "_variant": "names"}
    r.update(kw)
    return r


def test_identical_and_cache_keys_ignored():
    stored = req(_arm="arm3", _variant="names", _route="batch")
    ok, diffs = A.match(stored, req())
    assert ok and diffs == []


def test_batch_vs_standard_provider_is_a_note():
    ok, diffs = A.match(req(provider={"only": ["openai"]}, _route="batch"), req())
    assert ok
    assert len(diffs) == 1 and diffs[0].startswith("note: provider: batch block")
    assert A.blocking(diffs) == []


def test_other_provider_difference_blocks():
    ok, diffs = A.match(req(provider={"only": ["azure"], "allow_fallbacks": False}), req())
    assert not ok and any(d.startswith("provider:") for d in diffs)
    ok, _ = A.match(req(provider={"only": ["openai"], "allow_fallbacks": True}), req())
    assert not ok
    ok, _ = A.match(req(provider={"only": ["azure"]}), req())       # batch block, another provider
    assert not ok


def test_schema_name_only_is_a_note():
    stored = req()
    stored["response_format"]["json_schema"]["name"] = "arm3_confuser_answer"
    ok, diffs = A.match(stored, req())
    assert ok and diffs == ["note: schema name 'arm3_confuser_answer' (stored) vs 'arm3_answer' (built)"]


def test_schema_content_or_order_blocks():
    stored = req()
    stored["response_format"]["json_schema"]["schema"]["properties"]["stage"]["enum"] = ["0", "I", "II"]
    assert not A.match(stored, req())[0]
    reordered = req()
    s = reordered["response_format"]["json_schema"]["schema"]
    reordered["response_format"]["json_schema"]["schema"] = {k: s[k] for k in reversed(list(s))}
    ok, diffs = A.match(reordered, req())
    assert not ok and any("different key order" in d for d in diffs)


def test_changed_user_message_blocks():
    built = req()
    built["messages"][1]["content"] = built["messages"][1]["content"].replace("0\nI\n", "0: Tis N0 M0\nI\n")
    ok, diffs = A.match(req(), built)
    assert not ok and len(diffs) == 1 and diffs[0].startswith("user message content differs")


def test_changed_system_message_blocks():
    built = req()
    built["messages"][0]["content"] += " "
    ok, diffs = A.match(req(), built)
    assert not ok and diffs[0].startswith("system message content differs")


def test_changed_max_tokens_blocks():
    ok, diffs = A.match(req(max_tokens=8000), req())
    assert not ok and diffs == ["max_tokens: stored 8000 vs built 16000"]


def test_seed_absent_on_one_side_blocks_absent_on_both_passes():
    stored = req()
    del stored["seed"]
    assert not A.match(stored, req())[0]
    built = req()
    del built["seed"]
    assert A.match(stored, built) == (True, [])


def test_temperature_or_reasoning_blocks_and_no_provider_on_both_passes():
    assert not A.match(req(temperature=0), req())[0]
    assert not A.match(req(), req(reasoning={"effort": "low"}))[0]
    a, b = req(model="google/gemma-3-27b-it:featherless-ai"), req(model="google/gemma-3-27b-it:featherless-ai")
    del a["provider"], b["provider"]
    assert A.match(a, b) == (True, [])


def _index_row(system, item, variant, user, ts, valid=True, raw="r1.json", provider=None):
    r = req(_item=item, _variant=variant)
    if provider is not None:
        r["provider"] = provider
    r["messages"][1]["content"] = user
    return {"system": system, "item_id": item, "stored_variant": variant, "source": "t", "route": "batch",
            "raw_path": raw, "valid": valid, "answer": "I", "top_prob": 0.9, "meta_ts": ts,
            "system_text": r["messages"][0]["content"], "user_text": user,
            "body_rest_json": json.dumps({k: v for k, v in r.items() if k != "messages"})}


def test_coverage_statuses():
    names_user = "Pathology report:\nR1\n\nQuestion: Q\n\nStage groups:\n0\nI\n"
    idx = pd.DataFrame([
        _index_row("gpt", "crc001", "names", names_user, 2.0, raw="late.json", provider={"only": ["openai"]}),
        _index_row("gpt", "crc001", "names", names_user, 1.0, valid=False, raw="early.json"),
    ])

    def build(system, form, rid):
        if form == "definitions":
            r = req(_item=rid, _variant=form)
            r["messages"][1]["content"] = names_user.replace("0\nI\n", "0: Tis N0 M0\nI: T1-T2 N0 M0\n")
            return r
        if form == "documented":
            raise RuntimeError("refused")
        return req(_item=rid, _variant=form)            # names and named_field build the same request

    log = []
    cov = A.coverage(build, idx, systems=["gpt"], forms=["names", "named_field", "definitions", "documented"],
                     report_ids=["crc001"], set_of={"crc001": "main"}, log=log.append)
    st = dict(zip(cov.form, cov.status))
    assert st == {"names": "reused", "named_field": "identical_to_names", "definitions": "gap",
                  "documented": "build_error"}
    names = cov[cov.form == "names"].iloc[0]
    assert names.raw_path == "early.json" and names.n_identical == 2 and not names.stored_valid   # earliest, unusable
    assert len(log) == 1 and "early.json" in log[0]
    nf = cov[cov.form == "named_field"].iloc[0]
    assert nf.raw_path == "early.json"
    gap = cov[cov.form == "definitions"].iloc[0]
    assert json.loads(gap.nearest_diffs)[0].startswith("user message content differs")
    s = A.summary(cov)
    assert s[["reused", "identical_to_names", "gap", "build_error"]].sum().tolist() == [1, 1, 1, 1]

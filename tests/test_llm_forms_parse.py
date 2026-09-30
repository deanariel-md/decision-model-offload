"""Every LLM in every staging form, the reply parser (src/jevity/llm_forms_parse.py), on hand-made replies: the
four-question and single-question replies, response guards, confidence bands and scoring. No stored answer is read, no
network, no key."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevity import categorical as K  # noqa: E402
from jevity import docuse_staging as D  # noqa: E402
from jevity import llm_forms_parse as P  # noqa: E402

TOL = 1e-12
CORE = ("answer", "valid", "top_prob", "parse", "prob_status", "probs", "choice_not_top")


# ------------------------------------------------------------------------------------------------ forms and levels
def test_forms_and_levels():
    assert P.SINGLE_FORMS == ("names", "named_field", "definitions")
    assert P.FOUR_FORMS == ("structure_only", "documented", "documented_no_examples", "documented_notes")
    assert P.LEVELS == tuple(K.load_arm("arm3").labels) == D.LEVELS
    assert P.aliases() == {"IVA": "IVA-IVB", "IVB": "IVA-IVB"}


# ------------------------------------------------------------------------------------------------ four-question replies
def record(content, finish="stop", **resp_extra) -> dict:
    """A stored raw-store record shaped like an OpenRouter chat completion."""
    resp = {"id": "gen-test", "model": "openai/gpt-test", "provider": "OpenAI",
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "cost": 0.0042,
                      "completion_tokens_details": {"reasoning_tokens": 250}}, **resp_extra}
    return {"request": {"_item": "crc999"}, "response": resp, "meta": {"latency_s": 3.5, "route": "standard"}}


def onehot(q: str, choice: str) -> dict:
    return {o: (1.0 if o == choice else 0.0) for o in D.options(q)}


def four(t="T3", n="none", d="absent", m="none", probs=None, drop=()) -> dict:
    """A four-question reply object: each question's chosen option at 1.0, every option listed (as the strict schema
    asks), unless probs gives that question's list."""
    ch = dict(zip(D.QUESTION_IDS, (t, n, d, m)))
    return {q: {"choice": c, "probabilities": (probs or {}).get(q, onehot(q, c))} for q, c in ch.items() if q not in drop}


def row4(content, truth="IIA", form="documented"):
    text = content if isinstance(content, str) or content is None else json.dumps(content)
    return P.llm_row("gpt", form, "crc999", truth, record(text), raw_path="x.json")


def test_four_clean_json_full_confidence():
    r = row4(four())
    assert (r["answer"], r["valid"], r["parse"], r["prob_status"]) == ("IIA", True, "parts", "ok")
    assert r["top_prob"] == 1.0 and r["full_confidence"] and r["confidence_band"] == "1.00" and r["correct"]
    assert json.loads(r["probs"])["IIA"] == 1.0
    parts = json.loads(r["parts"])
    assert list(parts) == list(D.QUESTION_IDS) and parts["t_category"]["choice"] == "T3"
    assert list(r) == P.COLUMNS and r["system"] == "gpt" and r["form"] == "documented" and r["truth"] == "IIA"
    assert (r["provider"], r["model_reported"], r["route"], r["finish_reason"]) == ("OpenAI", "openai/gpt-test", "standard", "stop")
    assert (r["tokens_in"], r["tokens_out"], r["tokens_reasoning"], r["usd_reported"], r["latency_s"]) == (1200, 300, 250, 0.0042, 3.5)
    assert r["raw_path"] == "x.json" and r["error"] is None


def test_four_every_form_same_parse():
    for form in P.FOUR_FORMS:
        r = row4(four("T4b", "none", "present", "M1c"), truth="IVC", form=form)
        assert (r["answer"], r["valid"], r["full_confidence"], r["confidence_band"], r["correct"]) == ("IVC", True, True, "1.00", True)


def test_four_fenced():
    clean = row4(four())
    for text in ("```json\n" + json.dumps(four()) + "\n```", "```\n" + json.dumps(four()) + "\n```",
                 "  ```JSON " + json.dumps(four(), indent=2) + "```  "):
        r = row4(text)
        assert {k: r[k] for k in CORE + ("parts",)} == {k: clean[k] for k in CORE + ("parts",)}
        assert (r["answer"], r["valid"], r["parse"], r["top_prob"]) == ("IIA", True, "parts", 1.0)


def test_four_labels_differing_in_case():
    obj = {"t_category": {"choice": "t4A", "probabilities": {k.upper(): v for k, v in onehot("t_category", "T4a").items()}},
           "regional_nodes": {"choice": "NONE", "probabilities": {k.title(): v for k, v in onehot("regional_nodes", "none").items()}},
           "tumor_deposits": {"choice": "Absent", "probabilities": onehot("tumor_deposits", "absent")},
           "distant_metastasis": {"choice": "none", "probabilities": {k.lower(): v for k, v in onehot("distant_metastasis", "none").items()}}}
    r = row4(obj, truth="IIB")
    assert (r["answer"], r["valid"], r["parse"], r["top_prob"], r["correct"]) == ("IIB", True, "parts", 1.0, True)
    assert json.loads(r["parts"])["t_category"]["choice"] == "T4a"


def test_four_unknown_label():
    obj = four()
    obj["t_category"]["choice"] = "T5"
    r = row4(obj)
    assert (r["answer"], r["valid"], r["parse"], r["correct"]) == (None, False, "choice_invalid", False)
    assert r["top_prob"] == 1.0                                     # the lists are fine ...
    assert r["confidence_band"] is None and not r["full_confidence"]  # ... but an unusable answer has no confidence


def test_four_missing_question():
    r = row4(four(drop=("distant_metastasis",)))
    assert (r["answer"], r["valid"], r["parse"], r["prob_status"]) == (None, False, "choice_invalid", "distant_metastasis:missing")
    assert r["top_prob"] is None and r["confidence_band"] is None and r["correct"] is False


def test_four_sum_095_rescaled():
    p = {**onehot("t_category", "T3"), "T3": 0.6, "T4a": 0.35}
    r = row4(four(probs={"t_category": p}))
    assert (r["answer"], r["valid"], r["prob_status"]) == ("IIA", True, "ok")   # overall status: unstaged mass only
    assert json.loads(r["parts"])["t_category"]["prob_status"] == "renormalised"
    assert abs(r["top_prob"] - 0.6 / 0.95) <= TOL and r["confidence_band"] == "0.50-0.74"
    assert abs(json.loads(r["probs"])["IIB"] - 0.35 / 0.95) <= TOL


def test_four_sum_13_answer_counts_without_confidence():
    p = {**onehot("t_category", "T3"), "T3": 0.8, "T4a": 0.5}
    r = row4(four(probs={"t_category": p}))
    assert (r["answer"], r["valid"], r["correct"]) == ("IIA", True, True)
    assert r["prob_status"] == "t_category:sum_out_of_range" and r["probs"] is None and r["top_prob"] is None
    assert r["confidence_band"] is None and r["full_confidence"] is False


def test_four_omitted_option_zero_mass():
    r = row4(four(probs={"t_category": {"T3": 0.7, "T4a": 0.3}}))
    assert (r["answer"], r["valid"], r["prob_status"]) == ("IIA", True, "ok")
    assert abs(r["top_prob"] - 0.7) <= TOL and r["confidence_band"] == "0.50-0.74"
    lv = json.loads(r["probs"])
    assert abs(lv["IIA"] - 0.7) <= TOL and abs(lv["IIB"] - 0.3) <= TOL and lv["IIC"] == 0.0


def test_four_unstaged_combination():
    r = row4(four("Tis", "1", "absent", "none"), truth="0")
    assert (r["answer"], r["valid"], r["parse"], r["correct"], r["confidence_band"]) == (None, False, "unstaged", False, None)


def test_four_prose_before_json_is_unparsed():
    r = row4("Here is my answer.\n" + json.dumps(four()))
    assert (r["answer"], r["valid"], r["parse"], r["parts"], r["top_prob"]) == (None, False, "unparsed", None, None)


@pytest.mark.parametrize("content", ["", "   \n", None])
def test_four_empty(content):
    r = row4(content)
    assert (r["answer"], r["valid"], r["parse"], r["correct"], r["confidence_band"]) == (None, False, "empty", False, None)


def test_four_json_array():
    r = row4(json.dumps([four()]))
    assert (r["valid"], r["parse"]) == (False, "json_not_object")


def test_four_choice_not_top():
    r = row4(four(probs={"t_category": {**onehot("t_category", "T3"), "T3": 0.4, "T4a": 0.6}}))
    assert (r["answer"], r["valid"], r["choice_not_top"]) == ("IIA", True, True)   # the choice's stage stands
    assert abs(r["top_prob"] - 0.6) <= TOL                                          # product of the four tops


def test_four_confidence_absent_and_extra_keys_ignored():
    obj = four()
    obj["reasoning"] = "T3 N0 M0"
    obj["t_category"]["type"] = "choice"
    r = row4(obj)
    assert (r["answer"], r["valid"], r["parse"], r["top_prob"]) == ("IIA", True, "parts", 1.0)


def test_four_wrapped_object_is_unusable():
    r = row4({"answers": four()})
    assert (r["valid"], r["parse"], r["prob_status"]) == (False, "missing_answer", "t_category:missing")


def test_four_nonnumeric_probabilities_and_numeric_choice():
    obj = four()
    obj["t_category"]["probabilities"] = {k: str(v) for k, v in obj["t_category"]["probabilities"].items()}
    r = row4(obj)
    assert (r["answer"], r["valid"], r["prob_status"], r["top_prob"]) == ("IIA", True, "t_category:values_invalid", None)
    obj = four("T3", "1", "absent", "none")
    obj["regional_nodes"]["choice"] = 1                               # a JSON number, not the label "1"
    r = row4(obj)
    assert (r["valid"], r["parse"]) == (False, "choice_invalid")


@pytest.mark.parametrize("t,n,d,m,level", [
    ("T3", "none", "absent", "none", "IIA"),
    ("Tis", "none", "absent", "none", "0"),
    ("T1", "none", "not stated", "none", "I"),           # deposits not stated: staged as absent
    ("T2", "none", "present", "none", "IIIA"),           # N1c: no positive node, deposits present
    ("T3", "none", "present", "none", "IIIB"),           # N1c
    ("T4b", "none", "present", "none", "IIIC"),          # T4b N1c
    ("T1", "4 to 6", "absent", "none", "IIIA"),          # T1 N2a
    ("T2", "7 or more", "absent", "none", "IIIB"),       # T2 N2b
    ("T4a", "4 to 6", "present", "none", "IIIC"),        # N2a (deposits count only without positive nodes)
    ("T3", "1", "absent", "M1b", "IVA-IVB"),
    ("T4b", "none", "present", "M1c", "IVC"),            # M1c
    ("T2", "2 to 3", "absent", "M1c", "IVC"),
    ("T3", "not stated", "absent", "none", None),        # NX
    ("T3", "none", "absent", "not stated", None),        # M not stated
])
def test_four_stage_from_parts(t, n, d, m, level):
    r = row4(four(t, n, d, m), truth=level or "IIA")
    assert r["answer"] == level and r["valid"] == (level is not None)
    assert r["parse"] == ("parts" if level else "unstaged")
    assert r["correct"] == (level is not None)


# ------------------------------------------------------------------------------------------------ single-question replies
def one(stage, probs=None) -> dict:
    return {"stage": stage, "probabilities": probs if probs is not None else {l: (1.0 if l == stage else 0.0) for l in P.LEVELS}}


def row1(content, truth="IIIB", form="names"):
    text = content if isinstance(content, str) or content is None else json.dumps(content)
    return P.llm_row("claude", form, "crc999", truth, record(text), raw_path="y.json")


def test_single_clean_and_fenced():
    p = {l: 0.0 for l in P.LEVELS} | {"IIIB": 0.8, "IIIC": 0.2}
    for form in P.SINGLE_FORMS:
        r = row1(one("IIIB", p), form=form)
        assert (r["answer"], r["valid"], r["parse"], r["prob_status"], r["answer_how"]) == ("IIIB", True, "json", "ok", "exact")
        assert r["top_prob"] == 0.8 and r["confidence_band"] == "0.75-0.99" and r["correct"] and not r["full_confidence"]
        assert r["parts"] is None and list(r) == P.COLUMNS
    f = row1("```json\n" + json.dumps(one("IIIB", p)) + "\n```")
    assert (f["answer"], f["valid"], f["parse"], f["top_prob"]) == ("IIIB", True, "json", 0.8)
    assert row1(one("IIIB"))["confidence_band"] == "1.00" and row1(one("IIIB"))["full_confidence"]


def test_single_alias():
    r = row1(one("IVB", {l: 0.0 for l in P.LEVELS if l != "IVA-IVB"} | {"IVA": 0.3, "IVB": 0.7}), truth="IVA-IVB")
    assert (r["answer"], r["valid"], r["answer_how"], r["prob_alias"], r["correct"]) == ("IVA-IVB", True, "alias", True, True)
    assert abs(r["top_prob"] - 1.0) <= TOL and json.loads(r["probs"])["IVA-IVB"] == 1.0


def test_single_normalised_label():
    r = row1(one("Stage IIIB.", {l: (1.0 if l == "IIIB" else 0.0) for l in P.LEVELS}))
    assert (r["answer"], r["valid"], r["answer_how"]) == ("IIIB", True, "normalised")


def test_single_unknown_stage():
    r = row1(one("V", {l: 0.1 for l in P.LEVELS}))
    assert (r["answer"], r["valid"], r["parse"], r["correct"], r["confidence_band"]) == (None, False, "json_invalid_answer", False, None)


def test_single_sum_13():
    p = {l: 0.0 for l in P.LEVELS} | {"IIIB": 0.9, "IIIC": 0.4}
    r = row1(one("IIIB", p))
    assert (r["answer"], r["valid"], r["prob_status"], r["probs"], r["top_prob"]) == ("IIIB", True, "sum_out_of_range", None, None)
    assert r["correct"] and r["confidence_band"] is None and not r["full_confidence"]


def test_single_no_embedded_rescue():
    r = row1("The stage is IIIB.\n" + json.dumps(one("IIIB")))
    assert (r["valid"], r["parse"]) == (False, "unparsed")


# ------------------------------------------------------------------------------------------------ guards and bands
def test_response_guards():
    assert P.llm_row("gpt", "names", "a", "I", None)["error"] == "no stored response"
    for resp in ({}, {"choices": []}, {"choices": [None]}, {"choices": [{"message": None, "finish_reason": "length"}]}):
        rec = {"request": {}, "response": resp, "meta": {}}
        for form in ("names", "documented"):
            r = P.llm_row("gpt", form, "a", "I", rec)
            assert (r["valid"], r["parse"], r["correct"]) == (False, "empty", False), (resp, form)
    r = P.llm_row("gpt", "names", "a", "I", {"request": {}, "response": {"choices": [{"message": None, "finish_reason": "length"}]}})
    assert r["finish_reason"] == "length"
    r = P.llm_row("gpt", "documented", "a", "I", record([{"type": "text", "text": "{}"}]))
    assert (r["valid"], r["parse"]) == (False, "content_not_text") and "list" in r["error"]
    r = P.llm_row("gpt", "names", "a", "I", {"request": {}, "response": {"error": {"message": "overloaded", "code": 529}}, "meta": {}})
    assert (r["valid"], r["parse"], r["error"]) == (False, "empty", "response error: overloaded")
    e = P.error_row("gpt", "documented", "a", "I", "RuntimeError('x')")
    assert list(e) == P.COLUMNS and (e["valid"], e["correct"], e["confidence_band"]) == (False, False, None)


@pytest.mark.parametrize("p,b", [(None, None), (float("nan"), None), (0.0, "<0.50"), (0.4999999, "<0.50"), (0.5, "0.50-0.74"),
                                 (0.7499999, "0.50-0.74"), (0.75, "0.75-0.99"), (0.99, "0.75-0.99"),
                                 (1 - 1e-8, "0.75-0.99"), (1 - 1e-9, "1.00"), (1.0, "1.00")])
def test_band_edges(p, b):
    assert P.band(p) == b


def test_score_rules():
    assert P.score({"valid": True, "answer": "I", "truth": "I", "top_prob": 1 - 1e-9}) == \
        {"correct": True, "full_confidence": True, "confidence_band": "1.00"}
    assert P.score({"valid": False, "answer": None, "truth": "I", "top_prob": 1.0}) == \
        {"correct": False, "full_confidence": False, "confidence_band": None}
    assert P.score({"valid": True, "answer": "IIA", "truth": "I", "top_prob": None}) == \
        {"correct": False, "full_confidence": False, "confidence_band": None}

"""Every LLM in every form of the colon cancer staging question that Jev answered: the reply parser and the result
row for the ten LLMs and for Jev's stored answers, on the seven forms. No request is built or sent here; every row comes
from a stored raw-store record {request, response, meta}.

Single-question forms (names, named_field, definitions): categorical.parse_chatbot_answer with arm 3's levels, answer
key and answer_aliases (IVA, IVB -> IVA-IVB), no embedded-JSON reading, then categorical._row: the main arm 3 run's rule.
Jev: categorical.parse_jev_answer on its crc_stage_group answer, as the main run.

Four-question forms (structure_only, documented, documented_no_examples, documented_notes): the reply (a ```json fence
allowed) must be one JSON object; docuse_staging.parse_answers reads it, the function Jev's four answers were scored
with (match_label per question, probability_list with the 0.9-1.1 rescale rule, the stage from the four choices with
crc.n_category / crc.stage_group / crc.level_of, the routing confidence as the product of the four top probabilities),
and the row is docuse_staging.DocuseJevClient.row's. Parse codes: empty, unparsed, json_not_object (read here), and
parse_answers' parts, unstaged, choice_invalid, missing_answer.

Scoring: correct = valid and answer == truth (an unusable answer counts as wrong); full confidence = valid and
top_prob >= 1 - 1e-9; confidence band from top_prob (band); no band for an unusable answer or when top_prob is None
(a probability list that failed the sum rule)."""
from __future__ import annotations

import functools
import json
import math
from typing import Any

from . import categorical as K
from . import docuse_staging as D

SINGLE_FORMS = ("names", "named_field", "definitions")
FOUR_FORMS = ("structure_only", "documented", "documented_no_examples", "documented_notes")
FORMS = SINGLE_FORMS + FOUR_FORMS
FULL_CUT = 1 - 1e-9                     # full confidence (as scripts/docuse/run_structure_only.py and
                                        # scripts/summaries/pool_summary.py)
BANDS = ("<0.50", "0.50-0.74", "0.75-0.99", "1.00")

LEVELS: tuple[str, ...] = D.LEVELS

# Row columns: the calls.parquet columns this module can fill (the runner adds set, source, usd_list,
# excluded, exclude_reason), then the scoring columns, then three columns kept from the shared rows.
COLUMNS = ["system", "form", "item_id", "truth", "answer", "valid", "probs", "prob_status", "top_prob", "parse", "parts",
           "model_reported", "provider", "route", "tokens_in", "tokens_out", "tokens_reasoning", "usd_reported",
           "latency_s", "finish_reason", "raw_path", "error",
           "correct", "full_confidence", "confidence_band",
           "choice_not_top", "answer_how", "prob_alias"]


@functools.lru_cache(maxsize=None)
def arm3() -> K.ArmSpec:
    a = K.load_arm("arm3")
    assert a.labels == LEVELS, (a.labels, LEVELS)
    assert not (a.config.get("parse") or {}).get("embedded_json"), "arm 3 never read embedded JSON"
    return a


@functools.lru_cache(maxsize=None)
def four_arm() -> K.ArmSpec:
    return D.arm_spec()


def aliases() -> dict[str, str]:
    return {str(k): str(v) for k, v in (arm3().config.get("answer_aliases") or {}).items()}


# ------------------------------------------------------------------------------------------------ parsing
def parse_single(content: str | None) -> dict:
    """A single-question reply, exactly as the main arm 3 run read it."""
    return K.parse_chatbot_answer(content, LEVELS, arm3().answer_key, aliases(), embedded=False)


def _unread(code: str) -> dict:
    return {**D.parse_answers(None), "parse": code}


def parse_four(content: str | None) -> dict:
    """A four-question reply: one JSON object (a ```json fence allowed) read by docuse_staging.parse_answers. No
    embedded-JSON rescue: prose around the object leaves the reply unparsed."""
    if content is None or not content.strip():
        return _unread("empty")
    try:
        v = json.loads(K._strip_fence(content))
    except (ValueError, RecursionError):
        return _unread("unparsed")
    if not isinstance(v, dict):
        return _unread("json_not_object")
    return D.parse_answers(v)


# ------------------------------------------------------------------------------------------------ scoring
def _missing(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def band(top_prob: float | None) -> str | None:
    """Confidence band: <0.50; 0.50 to <0.75; 0.75 to <1 - 1e-9; >= 1 - 1e-9 is '1.00'. None for no confidence."""
    if _missing(top_prob):
        return None
    p = float(top_prob)
    if p >= FULL_CUT:
        return "1.00"
    if p >= 0.75:
        return "0.75-0.99"
    if p >= 0.50:
        return "0.50-0.74"
    return "<0.50"


def score(row: dict) -> dict:
    """correct, full_confidence and confidence_band of a row with answer, valid, top_prob and truth."""
    valid, tp, truth = bool(row.get("valid")), row.get("top_prob"), row.get("truth")
    conf = valid and not _missing(tp)
    return {"correct": None if truth is None else bool(valid and row.get("answer") == truth),
            "full_confidence": bool(conf and float(tp) >= FULL_CUT),
            "confidence_band": band(tp) if conf else None}


# ------------------------------------------------------------------------------------------------ rows
def _item(item_id: str, truth: str | None, form: str) -> K.Item:
    return K.Item(str(item_id), "", truth or "", LEVELS, LEVELS, (None,) * len(LEVELS), truth or "", "", form)


def _out(system: str, form: str, item_id: str, truth: str | None, r: dict, parts: dict | None) -> dict:
    row = {"system": system, "form": form, "item_id": str(item_id), "truth": truth,
           "answer": r.get("answer"), "valid": bool(r.get("valid")),
           "probs": json.dumps(r["probs"]) if r.get("probs") else None,
           "prob_status": r.get("prob_status"), "top_prob": r.get("top_prob"), "parse": r.get("parse"),
           "parts": json.dumps(parts) if parts else None,
           **{k: r.get(k) for k in ("model_reported", "provider", "route", "tokens_in", "tokens_out", "tokens_reasoning",
                                    "usd_reported", "latency_s", "finish_reason", "raw_path", "error")},
           "choice_not_top": bool(r.get("choice_not_top")), "answer_how": r.get("answer_how"),
           "prob_alias": bool(r.get("prob_alias"))}
    row.update(score(row))
    return {k: row[k] for k in COLUMNS}


def error_row(system: str, form: str, item_id: str, truth: str | None, error: str, raw_path: str | None = None,
              route: str | None = None) -> dict:
    """An unusable row for a call without a readable stored response (categorical.empty_row's fields)."""
    assert form in FORMS, form
    return _out(system, form, item_id, truth, K.empty_row(str(error)[:300], raw_path, route), None)


def _content(resp: dict) -> tuple[Any, str | None]:
    """(message content, finish_reason) of a chat-completions response; None for any part that is missing."""
    ch = resp.get("choices")
    ch = ch[0] if isinstance(ch, list) and ch and isinstance(ch[0], dict) else {}
    msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
    fin = ch.get("finish_reason")
    return msg.get("content"), (fin if isinstance(fin, str) else None)


def _response_error(resp: dict) -> str | None:
    e = resp.get("error")
    if not e:
        return None
    msg = e.get("message") if isinstance(e, dict) else e
    return f"response error: {msg}"[:300]


def _four_row(parsed: dict, item: K.Item, resp: dict, meta: dict, raw_path: str | None, finish: str | None) -> dict:
    """docuse_staging.DocuseJevClient.row: categorical._row, then the routing confidence (product of the four top
    probabilities) and choice_not_top from parse_answers."""
    r = K._row(parsed, item, four_arm(), resp, meta, raw_path, finish)
    r["top_prob"] = parsed["top_prob"]
    r["choice_not_top"] = parsed["choice_not_top"]
    return r


def llm_row(system: str, form: str, item_id: str, truth: str, cached: dict | None, raw_path: str | None = None) -> dict:
    """The row of one LLM answer from its stored raw-store record {request, response, meta}."""
    assert form in FORMS, form
    if not isinstance(cached, dict) or not isinstance(cached.get("response"), dict):
        return error_row(system, form, item_id, truth, "no stored response", raw_path)
    resp, meta = cached["response"], cached.get("meta") if isinstance(cached.get("meta"), dict) else {}
    item = _item(item_id, truth, form)
    content, finish = _content(resp)
    try:
        if content is not None and not isinstance(content, str):     # unusable (the main run: an exception row)
            parsed = {**(parse_single(None) if form in SINGLE_FORMS else parse_four(None)), "parse": "content_not_text"}
            err = f"message content is {type(content).__name__}, not text"
        else:
            parsed = parse_single(content) if form in SINGLE_FORMS else parse_four(content)
            err = _response_error(resp) if parsed["parse"] == "empty" else None
        r = (K._row(parsed, item, arm3(), resp, meta, raw_path, finish) if form in SINGLE_FORMS
             else _four_row(parsed, item, resp, meta, raw_path, finish))
    except Exception as e:  # noqa: BLE001  recorded as unusable, never dropped (categorical.execute's rule)
        r = {**K.empty_row(repr(e)[:300], raw_path, meta.get("route", "standard")), "finish_reason": finish}
        parsed, err = {"parts": None}, r["error"]
    r["error"] = err
    return _out(system, form, item_id, truth, r, parsed.get("parts") if form in FOUR_FORMS else None)


def jev_row(form: str, cached: dict | None, item_id: str | None = None, truth: str | None = None,
            raw_path: str | None = None) -> dict:
    """The row of one of Jev's stored answers, parsed as its own run parsed it: single-question forms with
    categorical.parse_jev_answer (crc_stage_group, arm 3's spec), four-question forms with docuse_staging.parse_answers
    (docuse_staging.DocuseJevClient.row). item_id defaults to the stored request's _item; correct is None without truth."""
    assert form in FORMS, form
    if item_id is None and isinstance(cached, dict):
        item_id = (cached.get("request") or {}).get("_item")
    if not isinstance(cached, dict) or not isinstance(cached.get("response"), dict):
        return error_row("jev", form, item_id, truth, "no stored response", raw_path)
    resp, meta = cached["response"], cached.get("meta") if isinstance(cached.get("meta"), dict) else {}
    item = _item(item_id, truth, form)
    answers = resp.get("answers")
    if form in SINGLE_FORMS:
        a = arm3()
        ans = answers.get(a.jev_question_id) if isinstance(answers, dict) else None
        parsed = K.parse_jev_answer(ans, a, item)
        r = K._row(parsed, item, a, resp, meta, raw_path)
        return _out("jev", form, item_id, truth, r, None)
    parsed = D.parse_answers(answers)
    return _out("jev", form, item_id, truth, _four_row(parsed, item, resp, meta, raw_path, None), parsed.get("parts"))

"""Categorical answers, arms 2 and 3: Jev's Choice and Score questions and the chatbots' JSON answer with a probability
per option.

The clients subclass those of clients.py: request settings (pinned provider, no fallbacks, no temperature or reasoning
parameter, seed where supported, max_tokens), the raw store (every response saved verbatim before parsing, first write
wins), the transport and the batch bookkeeping all come from clients.py and batch.py. Only the prompt, the question and
the parser differ.

Arm spec (config/<arm>.yaml): kind "choice" (arm 2: four lettered options, shuffled per item, truth in canonical
letters) or "score" (arm 3: ordered levels, names only or names plus definitions). Every answer is stored in canonical
labels, so the analysis never sees the presented order.

Parsing rule, identical for every system: the reply is one JSON object (a ```json fence is allowed) with the answer key
("choice", "stage") and "probabilities" (label -> number in [0, 1]); a label matches exactly, else ignoring case, else
after dropping a leading "option"/"stage"/"answer" word and surrounding brackets or a trailing full stop. An omitted
label carries zero mass. A list summing to 0.9-1.1 is renormalised; any other list leaves that answer out of the
confidence analysis only (the answer still counts). Jev's Choice answer is its `choice`; its Score answer is the
most probable level (ties: the level nearest its `score`, then the lower level), with `score` as its expected level."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from . import clients as C
from .clients import (BATCH_MISSING, HF_ROUTER, MODELS, OPENROUTER, ROOT, JevClient, LLMClient, RawStore, _key,
                      _real01, _sha, active_families, estimate_tokens, split_system)

SEED = 20260922
CHOICE_LABELS = ("A", "B", "C", "D")
JEV_SCORE_MAX_LEVELS = 10     # docs.typesafe.ai/primitives/score: "criteria" holds 2-10 levels
JEV_CHOICE_MAX_OPTIONS = 255  # docs.typesafe.ai/primitives/choice: "up to 255 options"
SUM_RANGE = (0.9, 1.1)        # lists summing to 0.9-1.1 are renormalised


# ------------------------------------------------------------------------------------------------ arm spec and items
@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str                     # "choice" | "score"
    population: str               # passed to the clients; never "eicu"
    instructions: str             # the question, identical for Jev and the chatbots
    answer_key: str               # chatbot JSON key of the chosen label ("choice", "stage")
    jev_question_id: str
    system_prompt: str
    reply_prompt: str
    user_template: str            # {state}, {instructions}, {options}
    labels: tuple[str, ...]       # canonical labels (choice: A-D; score: level names, low to high)
    tiers: tuple[str, ...]
    batch: bool                   # batch route for families with a batch endpoint (as in arm 1)
    config: dict


def load_arm(name_or_path: str | Path) -> ArmSpec:
    p = Path(name_or_path)
    if not p.suffix:
        p = ROOT / "config" / f"{name_or_path}.yaml"
    c = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert c["kind"] in ("choice", "score"), c["kind"]
    assert c["population"] != "eicu", "credentialed eICU rows never go to any model"
    labels = (tuple(str(l) for l in c.get("labels") or CHOICE_LABELS) if c["kind"] == "choice"
              else tuple(str(lv["name"]) for lv in c["levels"]))
    al = c.get("answer_aliases") or {}
    assert c["kind"] == "score" or not al, "answer_aliases: Score arms only (Choice letters are shuffled per item)"
    bad = {a: t for a, t in al.items() if str(t) not in labels or str(a) in labels}
    assert not bad, f"answer_aliases must map a name that is not a level to a level: {bad}"
    pr = c["prompts"]
    return ArmSpec(name=c["arm"], kind=c["kind"], population=c["population"], instructions=c["instructions"],
                   answer_key=c["answer_key"], jev_question_id=c["jev"]["question_id"],
                   system_prompt=pr["system"], reply_prompt=pr["reply"], user_template=pr["user_template"],
                   labels=labels, tiers=tuple(c["systems"]["tiers"]), batch=bool(c.get("route", {}).get("batch")),
                   config=c)


def arm_systems(arm: ArmSpec) -> list[str]:
    """Jev first, then every active family of the arm's tiers in config/models.yaml order."""
    return ["jev"] + active_families(arm.tiers)


@dataclass(frozen=True)
class Item:
    item_id: str
    state: str                        # what Jev receives as state and the chatbots as the message or report
    truth: str                        # canonical label
    presented: tuple[str, ...]        # labels in the order shown
    canonical: tuple[str, ...]        # canonical label shown at each position
    texts: tuple[str | None, ...]     # option text (choice) or level definition (score; None = name only)
    stratum: str = ""                 # arm 2: vignette_type; arm 3: stage group
    group: str = ""                   # arm 2: recommendation_id
    variant: str = "main"

    def to_canonical(self, label: str) -> str:
        return self.canonical[self.presented.index(label)]


def shuffle_order(item_id: str, labels: tuple[str, ...] = CHOICE_LABELS, seed: int = SEED) -> tuple[str, ...]:
    """Canonical labels in presented order for one item: a permutation from a generator seeded with (seed, item id), so
    an item's order depends on nothing else. The same order is used for every system."""
    h = int(hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:8], 16)
    return tuple(labels[i] for i in np.random.default_rng([seed, h]).permutation(len(labels)))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_items(arm: ArmSpec) -> list[Item]:
    """The arm's items for every variant of the arm."""
    return _choice_items(arm) if arm.kind == "choice" else _score_items(arm)


def _item_frame(arm: ArmSpec) -> tuple[pd.DataFrame, dict]:
    ic = arm.config["items"]
    df = pd.read_csv(ROOT / ic["main"], dtype=str, keep_default_na=False)
    assert not df[ic["id_column"]].duplicated().any(), "duplicate item ids"
    return df, ic


def _choice_items(arm: ArmSpec) -> list[Item]:
    """One item per row. Arm 2: the four option texts come from the row's option columns (config items.option_columns:
    canonical letter -> column), shown in the stored seeded order. An arm whose items carry their own option list
    (config items.options_column): _own_option_items."""
    df, ic = _item_frame(arm)
    assert not any("rows" in (v or {}) for v in (arm.config.get("variants") or {}).values()), \
        "variant rows filters are for Score arms only"
    if ic.get("options_column"):
        return _own_option_items(arm, df, ic)
    cols = ic["option_columns"]
    out = []
    for r in df.to_dict("records"):
        iid = r[ic["id_column"]]
        order = shuffle_order(iid, CHOICE_LABELS, int(arm.config["seed"]))
        assert "".join(order) == r[ic["order_column"]], f"{iid}: stored option order differs from the seeded shuffle"
        opts = {c: r[cols[c]] for c in CHOICE_LABELS}
        assert len(set(opts.values())) == 4 and all(opts.values()), f"{iid}: option texts empty or not distinct"
        truth = r[ic["truth_column"]]
        assert truth in CHOICE_LABELS and r["truth_presented"] == CHOICE_LABELS[order.index(truth)], iid
        out.append(Item(iid, r[ic["state_column"]].strip(), truth, CHOICE_LABELS, order, tuple(opts[c] for c in order),
                        r[ic["stratum_column"]], r[ic["group_column"]]))
    return out


def _own_option_items(arm: ArmSpec, df: pd.DataFrame, ic: dict) -> list[Item]:
    """Items that carry their own options (arm 2's second item set): a JSON object letter -> text, shown in the
    dataset's order, the letters the first k of the arm's labels. A variant (config variants) may name other option and
    truth columns: the same questions with another option list."""
    out = []
    for vname, v in (arm.config.get("variants") or {"main": {}}).items():
        oc, tc = (v or {}).get("options_column", ic["options_column"]), (v or {}).get("truth_column", ic["truth_column"])
        for r in df.to_dict("records"):
            iid, opts = r[ic["id_column"]], json.loads(r[oc])
            lab = tuple(opts)
            assert lab == arm.labels[:len(lab)], f"{iid}: letters {lab} are not the first {len(lab)} of {arm.labels}"
            assert 2 <= len(lab) <= JEV_CHOICE_MAX_OPTIONS and all(opts.values()), f"{iid}: {len(lab)} options"
            assert r[tc] in lab, f"{iid}: key {r[tc]!r} is not an option"
            out.append(Item(iid, r[ic["state_column"]].strip(), r[tc], lab, lab, tuple(opts[l] for l in lab),
                            r.get(ic.get("stratum_column", ""), ""), r.get(ic.get("group_column", ""), ""), vname))
    return out


def _score_items(arm: ArmSpec) -> list[Item]:
    df, ic = _item_frame(arm)
    levels = arm.config["levels"]
    out = []
    for vname, v in arm.config["variants"].items():
        texts = tuple((lv.get("definition") if v.get("definitions") else None) for lv in levels)
        rows = df
        for col, val in (v.get("rows") or {}).items():     # a variant may keep only the rows whose column equals a
            if col not in df:                               # value, compared as text
                raise ValueError(f"{arm.name} variant {vname}: rows filter column {col!r} missing from the items")
            want = str(val).lower() if isinstance(val, bool) else str(val)   # YAML true -> "true"
            rows = rows[rows[col].astype(str) == want]
        for r in rows.to_dict("records"):
            truth = r[ic["truth_column"]]
            assert truth in arm.labels, f"{r[ic['id_column']]}: truth {truth!r} is not a level"
            out.append(Item(r[ic["id_column"]], r[ic["state_column"]].strip(), truth, arm.labels, arm.labels, texts,
                            r.get(ic.get("stratum_column", ""), truth) or truth, r.get(ic.get("group_column", ""), ""),
                            vname))
    return out


def repeat_ids(arm: ArmSpec) -> list[str]:
    """The random items asked three times: the stored column when the items file has one, else a draw of config
    repeats.n_items from the sorted ids with the arm seed."""
    df, ic = _item_frame(arm)
    col = ic.get("repeat_column")
    if col and col in df:
        return sorted(df.loc[df[col].str.lower() == "true", ic["id_column"]])
    ids = sorted(df[ic["id_column"]])
    n = min(int(arm.config["repeats"]["n_items"]), len(ids))
    return sorted(np.random.default_rng(int(arm.config["seed"])).choice(ids, size=n, replace=False).tolist())


# ------------------------------------------------------------------------------------------------ prompts and questions
def _flat(s: str) -> str:
    return " ".join(s.split())


def system_text(arm: ArmSpec) -> str:
    return f"{_flat(arm.system_prompt)} {_flat(arm.reply_prompt)}"


def _level_text(arm: ArmSpec, label: str, text: str | None) -> str:
    return label if text is None else arm.config.get("level_format", "{label}: {text}").format(label=label, text=text)


def options_block(arm: ArmSpec, item: Item) -> str:
    if arm.kind == "choice":
        return "\n".join(f"{l}. {t}" for l, t in zip(item.presented, item.texts))
    return "\n".join(_level_text(arm, l, t) for l, t in zip(item.presented, item.texts))


def user_text(arm: ArmSpec, item: Item) -> str:
    return arm.user_template.format(state=item.state, instructions=arm.instructions, options=options_block(arm, item))


def reply_schema(arm: ArmSpec, labels: tuple[str, ...]) -> dict:
    """Strict JSON schema: the chosen label from the list and one probability per label."""
    return {"name": f"{arm.name}_answer", "strict": True, "schema": {
        "type": "object",
        "properties": {
            arm.answer_key: {"type": "string", "enum": list(labels)},
            "probabilities": {"type": "object",
                              "properties": {l: {"type": "number", "minimum": 0, "maximum": 1} for l in labels},
                              "required": list(labels), "additionalProperties": False}},
        "required": [arm.answer_key, "probabilities"], "additionalProperties": False}}


def jev_question(arm: ArmSpec, item: Item) -> dict:
    """Choice: criteria = option texts keyed by presented letter, in presented order. Score: criteria = the ordered
    levels, low to high (names, or names with definitions)."""
    if arm.kind == "choice":
        return {"type": "choice", "instructions": arm.instructions,
                "criteria": {l: t for l, t in zip(item.presented, item.texts)}}
    crit = [_level_text(arm, l, t) for l, t in zip(item.presented, item.texts)]
    cap = int(arm.config.get("jev_score_max_levels", JEV_SCORE_MAX_LEVELS))
    if not 2 <= len(crit) <= cap:
        raise ValueError(f"a Jev Score question takes 2-{cap} levels (docs.typesafe.ai/primitives/score); "
                         f"{arm.name} has {len(crit)}. Confirm the limit with TypeSafe and set jev_score_max_levels in "
                         f"config/{arm.name}.yaml, or change the design.")
    return {"type": "score", "instructions": arm.instructions, "criteria": crit}


# ------------------------------------------------------------------------------------------------ parsing
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S | re.I)
_PREFIX = re.compile(r"^(?:option|stage|answer)\s+", re.I)


def match_label(v: Any, labels: tuple[str, ...]) -> tuple[str | None, str]:
    """(label, how): exact, else ignoring case (when that keeps labels distinct), else after dropping a leading
    option/stage/answer word and surrounding brackets or a trailing full stop."""
    if not isinstance(v, str):
        return None, "not_string"
    s = v.strip()
    if s in labels:
        return s, "exact"
    low = {l.lower(): l for l in labels}
    ci = len(low) == len(labels)
    if ci and s.lower() in low:
        return low[s.lower()], "case"
    t = _PREFIX.sub("", s).strip().rstrip(".").strip("()[] ").strip()
    if t in labels:
        return t, "normalised"
    if ci and t.lower() in low:
        return low[t.lower()], "normalised"
    return None, "unknown_label"


def match_alias(v: Any, labels: tuple[str, ...], aliases: dict[str, str] | None) -> str | None:
    """The level an alias names (arm config answer_aliases), matched with match_label's rules; None if v names a
    label itself or no alias."""
    if not aliases or match_label(v, labels)[0] is not None:
        return None
    a = match_label(v, tuple(aliases))[0]
    return str(aliases[a]) if a is not None else None


def probability_list(raw: Any, labels: tuple[str, ...], keys: dict[str, str] | None = None,
                     aliases: dict[str, str] | None = None) -> tuple[dict | None, str, dict | None]:
    """(renormalised probabilities over labels, status, values as given). `keys` maps raw keys to labels (Jev Score:
    '0'.. -> level names); otherwise keys are matched with match_label, and an alias key's mass is added to its level
    before the sum rule."""
    if not isinstance(raw, dict) or not raw:
        return None, "missing", None
    vals: dict[str, float] = {}
    seen: set[str] = set()
    for k, v in raw.items():
        lab = keys.get(str(k)) if keys is not None else match_label(k, labels)[0]
        via_alias = lab is None and keys is None and (lab := match_alias(k, labels, aliases)) is not None
        if lab is None:
            return None, "labels_unknown", None
        if not via_alias:
            if lab in seen:
                return None, "labels_duplicate", None
            seen.add(lab)
        x = _real01(v)
        if x is None:
            return None, "values_invalid", None
        vals[lab] = vals.get(lab, 0.0) + x
    given = {l: vals.get(l, 0.0) for l in labels}           # an omitted label carries zero mass
    tot = sum(given.values())
    if not SUM_RANGE[0] - 1e-9 <= tot <= SUM_RANGE[1] + 1e-9:
        return None, "sum_out_of_range", given
    return {l: x / tot for l, x in given.items()}, ("ok" if abs(tot - 1) <= 1e-6 else "renormalised"), given


def _strip_fence(content: str) -> str:
    txt = content.strip()
    m = _FENCE.match(txt)
    return m.group(1).strip() if m else txt


def embedded_json(content: str, answer_key: str) -> str | None:
    """The last {...} in a reply that parses as a JSON object holding the answer key and "probabilities" (a reply that
    reasons first and ends with the requested JSON; config parse.embedded_json)."""
    opens = [i for i, ch in enumerate(content) if ch == "{"]
    for end in (i for i in range(len(content) - 1, -1, -1) if content[i] == "}"):
        for start in opens:
            if start >= end:
                break
            try:
                v = json.loads(content[start:end + 1])
            except (ValueError, RecursionError):
                continue
            if isinstance(v, dict) and answer_key in v and "probabilities" in v:
                return content[start:end + 1]
    return None


def parse_chatbot_answer(content: str | None, labels: tuple[str, ...], answer_key: str,
                         aliases: dict[str, str] | None = None, embedded: bool = False) -> dict:
    """{answer (presented label or None), answer_how, probs (renormalised or None), prob_status, given, prob_alias,
    parse}. aliases (arm config answer_aliases, Score arms): an answer naming an alias counts as its level
    (answer_how 'alias'); an alias probability key adds its mass to its level (prob_alias True). embedded (arm config
    parse.embedded_json): a reply that is not one JSON object is read from the last JSON object in it holding the answer
    key and "probabilities" (parse 'json_embedded'); everything else as for a reply that is one object."""
    out = {"answer": None, "answer_how": None, "probs": None, "prob_status": "missing", "given": None,
           "prob_alias": False}
    if content is None or not content.strip():
        return {**out, "parse": "empty"}
    try:
        v = json.loads(_strip_fence(content))
    except (ValueError, RecursionError):
        inner = embedded_json(content, answer_key) if embedded else None
        if inner is None:
            return {**out, "parse": "unparsed"}
        r = parse_chatbot_answer(inner, labels, answer_key, aliases)
        return {**r, "parse": "json_embedded" if r["parse"] == "json" else r["parse"]}
    if not isinstance(v, dict):
        return {**out, "parse": "json_not_object"}
    raw = v.get("probabilities")
    probs, status, given = probability_list(raw, labels, aliases=aliases)
    out.update(probs=probs, prob_status=status, given=given,
               prob_alias=isinstance(raw, dict) and any(match_alias(k, labels, aliases) for k in raw))
    if answer_key not in v:
        return {**out, "parse": "json_no_answer"}
    lab, how = match_label(v[answer_key], labels)
    if lab is None and (al := match_alias(v[answer_key], labels, aliases)) is not None:
        lab, how = al, "alias"
    out.update(answer=lab, answer_how=how)
    return {**out, "parse": "json" if lab is not None else "json_invalid_answer"}


def parse_jev_answer(ans: Any, arm: ArmSpec, item: Item) -> dict:
    """Jev's answer object for one question. Choice: its `choice`, `probabilities` keyed by presented letter and its
    `confidence`. Score: `probabilities` keyed '0'..'n-1', `score` (expected level) and `confidence`; the answer is the
    most probable level."""
    out = {"answer": None, "answer_how": None, "probs": None, "prob_status": "missing", "given": None,
           "jev_confidence": None, "expected_level": None}
    if not isinstance(ans, dict):
        return {**out, "parse": "missing_answer"}
    out["jev_confidence"] = _real01(ans.get("confidence"))
    if arm.kind == "choice":
        probs, status, given = probability_list(ans.get("probabilities"), item.presented)
        lab, how = match_label(ans.get("choice"), item.presented)
        out.update(probs=probs, prob_status=status, given=given, answer=lab, answer_how=how)
        return {**out, "parse": "choice" if lab is not None else "choice_invalid"}
    keys = {str(i): l for i, l in enumerate(item.presented)}
    probs, status, given = probability_list(ans.get("probabilities"), item.presented, keys)
    out.update(probs=probs, prob_status=status, given=given)
    sc = ans.get("score")
    sc = float(sc) if isinstance(sc, (int, float)) and not isinstance(sc, bool) and 0 <= sc <= len(item.presented) - 1 else None
    dist = probs or given
    p = np.array([dist[l] for l in item.presented]) if dist is not None else np.zeros(0)
    if p.sum() <= 0:
        return {**out, "expected_level": sc, "parse": "score_invalid"}
    if sc is None:
        sc = float((np.arange(len(p)) * p).sum() / p.sum())
    top = np.flatnonzero(np.isclose(p, p.max()))
    best = min(top, key=lambda i: (abs(i - sc) if sc is not None else 0, i))
    out.update(answer=item.presented[best], answer_how="argmax", expected_level=sc)
    return {**out, "parse": "score"}


# ------------------------------------------------------------------------------------------------ result rows
def _num(v) -> float | None:
    try:
        return float(v) if v is not None and not isinstance(v, bool) else None
    except (TypeError, ValueError):
        return None


def _row(parsed: dict, item: Item, arm: ArmSpec, resp: dict, meta: dict, raw_path: str | None,
         finish: str | None = None) -> dict:
    """One result in canonical labels. probs: canonical label -> probability (None if the list failed the rule)."""
    probs = parsed.get("probs")
    can = {item.to_canonical(l): p for l, p in probs.items()} if probs else None
    ans = parsed.get("answer")
    top = max(can.values()) if can else None
    not_top = bool(can and ans is not None and can[item.to_canonical(ans)] < top - 1e-12)
    exp = parsed.get("expected_level")
    if exp is None and can and arm.kind == "score":
        exp = float(sum(i * can[l] for i, l in enumerate(arm.labels)))
    u = resp.get("usage") or {}
    return {"answer": item.to_canonical(ans) if ans is not None else None, "answer_presented": ans,
            "valid": ans is not None, "probs": can, "prob_status": parsed.get("prob_status"), "top_prob": top,
            "choice_not_top": not_top, "jev_confidence": parsed.get("jev_confidence"), "expected_level": exp,
            "parse": parsed.get("parse"), "answer_how": parsed.get("answer_how"),
            "prob_alias": bool(parsed.get("prob_alias", False)),
            "provider": resp.get("provider") or meta.get("provider_pinned"), "model_reported": resp.get("model"),
            "route": meta.get("route", "standard"), "finish_reason": finish, "response_id": resp.get("id"),
            "latency_s": _num(meta.get("latency_s")),
            "tokens_in": _num(u.get("prompt_tokens", u.get("input_tokens"))),
            "tokens_out": _num(u.get("completion_tokens", u.get("output_tokens"))),
            "tokens_reasoning": _num((u.get("completion_tokens_details") or {}).get("reasoning_tokens")),  # part of tokens_out
            "usd_reported": _num(u.get("cost")), "raw_path": raw_path, "error": None,
            "simulated": bool(meta.get("simulated", False))}


def empty_row(error: str, raw_path: str | None = None, route: str | None = None) -> dict:
    return {"answer": None, "answer_presented": None, "valid": False, "probs": None, "prob_status": None,
            "top_prob": None, "choice_not_top": False, "jev_confidence": None, "expected_level": None, "parse": None,
            "answer_how": None, "prob_alias": False, "provider": None, "model_reported": None, "route": route, "finish_reason": None,
            "response_id": None, "latency_s": None, "tokens_in": None, "tokens_out": None, "tokens_reasoning": None,
            "usd_reported": None, "raw_path": raw_path, "error": error, "simulated": False}


# ------------------------------------------------------------------------------------------------ clients
Sender = Callable[[str, dict], dict]     # (system, request) -> raw response; simulation only


class FamilyPause:
    """One per family: after any 429 every worker of the family waits until the pause ends (the Retry-After header
    when given, else 5 s doubling to 120 s; halved again after each success)."""

    def __init__(self, first: float = 5.0, cap: float = 120.0):
        self.until, self.delay, self.first, self.cap, self.lock = 0.0, first, first, cap, threading.Lock()
        self.hits = 0

    def wait(self, sleep=time.sleep) -> None:
        while True:
            with self.lock:
                w = self.until - time.time()
            if w <= 0:
                return
            sleep(min(w, 5.0))

    def hit(self, retry_after: float | None = None) -> float:
        with self.lock:
            d = retry_after if retry_after and retry_after > 0 else self.delay
            self.until = max(self.until, time.time() + d)
            self.delay = min(self.delay * 2, self.cap)
            self.hits += 1
            return d

    def ok(self) -> None:
        with self.lock:
            self.delay = max(self.first, self.delay / 2)


_PAUSES: dict[str, FamilyPause] = {}
_PAUSE_GUARD = threading.Lock()


def family_pause(family: str) -> FamilyPause:
    with _PAUSE_GUARD:
        return _PAUSES.setdefault(family, FamilyPause())


def _retry_after(r) -> float | None:
    try:
        return float(r.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def post_paced(url: str, headers: dict, body: dict, pause: FamilyPause, retries: int = 8, sleep=time.sleep) -> dict:
    """clients._post with the family pause: a 429 pauses every worker of the family (and does not use up an attempt
    until 20 have been seen); 500/502/503/529 and transport errors back off on this worker only."""
    delay, attempt, rate_limited = 2.0, 0, 0
    while attempt < retries:
        pause.wait(sleep)
        try:
            with C.httpx.Client(timeout=120) as c:
                r = c.post(url, headers=headers, json=body)
        except C.httpx.TransportError:
            attempt += 1
            if attempt == retries:
                raise
            sleep(delay); delay *= 2
            continue
        if r.status_code == 429:
            pause.hit(_retry_after(r))
            rate_limited += 1
            if rate_limited > 20:
                attempt += 1
            continue
        if r.status_code in (500, 502, 503, 529):
            attempt += 1
            sleep(delay); delay *= 2
            continue
        r.raise_for_status()
        pause.ok()
        return r.json()
    raise RuntimeError("retries exhausted")


def _send_simulated(client, system: str, req: dict) -> dict:
    t0 = time.time()
    resp = client.sender(system, req)
    lat = resp.pop("_sim_latency_s", None)
    meta = {"ts": time.time(), "latency_s": lat if lat is not None else time.time() - t0, "route": "standard",
            "simulated": True, "provider_reported": resp.get("provider"), "model_reported": resp.get("model"),
            "usage": resp.get("usage")}
    if not client.store.put(req, resp, meta):
        return client.store.get(req)
    return {"request": req, "response": resp, "meta": meta}


class CategoricalLLMClient(LLMClient):
    """A chatbot answering one item: LLMClient's settings, pin and route; the arm's prompt; a strict JSON schema where
    the pinned endpoint supports structured outputs."""

    def __init__(self, system: str, store: RawStore, arm: ArmSpec, repeat: int = 0, standard: bool = False,
                 sender: Sender | None = None):
        self.arm, self.sender, self.system = arm, sender, system
        super().__init__(system, store, population=arm.population, repeat=repeat, standard=standard)

    def _batch_spec(self) -> dict | None:
        """Batch route as in arm 1 (the families under `batch` in config/models.yaml), switched per arm by the arm's
        route.batch instead of the population list."""
        b = MODELS.get("batch") or {}
        if not (self.use_batch and self.arm.batch and b.get("enabled") and self.family in (b.get("families") or {})):
            return None
        return b["families"][self.family]

    def request(self, item: Item) -> dict:
        req = super().request(item.state, "")
        req["messages"] = [{"role": "system", "content": system_text(self.arm)},
                           {"role": "user", "content": user_text(self.arm, item)}]
        req.pop("response_format", None)
        if self.spec.get("structured_outputs"):
            req["response_format"] = {"type": "json_schema", "json_schema": reply_schema(self.arm, item.presented)}
        req.update(_arm=self.arm.name, _item=item.item_id, _variant=item.variant)   # cache key only; stripped
        return req

    def _send(self, req: dict) -> dict:
        """LLMClient._send on the standard route, with one pause shared by every worker of the family: any 429 stops
        them all until it clears."""
        if self.sender is not None:
            return _send_simulated(self, self.system, req)
        body = {k: v for k, v in req.items() if not k.startswith("_")}
        if self.population == "eicu":
            raise RuntimeError("credentialed eICU rows never go to any model")
        t0 = time.time()
        if self.transport == "hf_router":
            url, key = f"{HF_ROUTER}/chat/completions", _key("HF_TOKEN")
        else:
            url, key = f"{OPENROUTER}/api/v1/chat/completions", _key("OPENROUTER_API_KEY")
        resp = post_paced(url, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body,
                          family_pause(self.family))
        meta = {"ts": time.time(), "latency_s": time.time() - t0, "family": self.family, "route": "standard",
                "transport": self.transport, "provider_reported": resp.get("provider"),
                "model_reported": resp.get("model"), "usage": resp.get("usage")}
        if not self.store.put(req, resp, meta):
            return self.store.get(req)
        return {"request": req, "response": resp, "meta": meta}

    def answer(self, item: Item) -> dict:
        req = self.request(item)
        cached = self.store.get(req)
        if cached is None and self.batch is not None:
            if self.family in (MODELS.get("batch") or {}).get("fallback_to_standard", []):
                return CategoricalLLMClient(self.system, self.store, self.arm, self.repeat, standard=True,
                                            sender=self.sender).answer(item)
            return empty_row(BATCH_MISSING, str(self.store.path(req)), "batch")
        if cached is None:
            with self.store.lock(req):
                cached = self.store.get(req) or self._send(req)
        resp = cached["response"]
        try:
            content = resp["choices"][0]["message"].get("content")
            finish = resp["choices"][0].get("finish_reason")
        except Exception:
            content, finish = None, None
        parsed = parse_chatbot_answer(content, item.presented, self.arm.answer_key,
                                      self.arm.config.get("answer_aliases"),
                                      bool((self.arm.config.get("parse") or {}).get("embedded_json")))
        return _row(parsed, item, self.arm, resp, cached.get("meta") or {}, str(self.store.path(req)), finish)


class CategoricalJevClient(JevClient):
    """Jev answering one item: JevClient's model, transport and provider block; state = the item; one Choice or
    Score question."""

    def __init__(self, store: RawStore, arm: ArmSpec, repeat: int = 0, transport: str | None = None,
                 sender: Sender | None = None):
        self.arm, self.sender = arm, sender
        super().__init__(store, population=arm.population, repeat=repeat, transport=transport)

    def request(self, item: Item) -> dict:
        req = super().request(item.state, "")
        req["questions"] = {self.arm.jev_question_id: jev_question(self.arm, item)}
        req.update(_arm=self.arm.name, _item=item.item_id, _variant=item.variant)
        return req

    def _send(self, req: dict) -> dict:
        if self.sender is not None:
            return _send_simulated(self, "jev", req)
        return super()._send(req)

    def answer(self, item: Item) -> dict:
        req = self.request(item)
        cached = self.store.get(req)
        if cached is None:
            with self.store.lock(req):
                cached = self.store.get(req) or self._send(req)
        resp = cached["response"]
        ans = (resp.get("answers") or {}).get(self.arm.jev_question_id) if isinstance(resp.get("answers"), dict) else None
        parsed = parse_jev_answer(ans, self.arm, item)
        return _row(parsed, item, self.arm, resp, cached.get("meta") or {}, str(self.store.path(req)))


def make_factory(arm: ArmSpec, store: RawStore, sender: Sender | None = None, standard: bool = False):
    """client_for(system, repeat), one client per (system, repeat). standard=True: synchronous route for every family
    (timing sample, simulation)."""
    lock, cache = threading.Lock(), {}

    def get(system: str, repeat: int = 0):
        with lock:
            if (system, repeat) not in cache:
                cache[(system, repeat)] = (CategoricalJevClient(store, arm, repeat, sender=sender) if system == "jev"
                                           else CategoricalLLMClient(system, store, arm, repeat, standard=standard,
                                                                     sender=sender))
            return cache[(system, repeat)]
    return get


# ------------------------------------------------------------------------------------------------ plan and execution
@dataclass(frozen=True)
class CatCall:
    system: str
    item_id: str
    repeat: int = 0
    variant: str = "main"


def plan_calls(items: list[Item], systems: list[str], rep_ids: list[str] | None = None, n_repeats: int = 3,
               seed: int = SEED, block: int = 50) -> list[CatCall]:
    """Every system on every item and variant (repeat 0); repeats 1..n_repeats-1 on rep_ids; in block order."""
    calls = [CatCall(s, it.item_id, 0, it.variant) for s in systems for it in items]
    reps = set(rep_ids or [])
    calls += [CatCall(s, it.item_id, r, it.variant) for r in range(1, n_repeats) for s in systems for it in items
              if it.item_id in reps]
    return block_order(calls, seed, block)


def block_order(calls: list[CatCall], seed: int = SEED, block: int = 50) -> list[CatCall]:
    """Items in a seeded random order, in blocks of `block`; every call of a block (all systems, variants and repeats)
    precedes the next block, in a seeded random order within it. A spending stop leaves complete blocks."""
    ids = sorted({c.item_id for c in calls})
    rank = {str(p): i for i, p in enumerate(np.random.default_rng(seed).permutation(ids))}
    tie = np.random.default_rng(seed + 1).permutation(len(calls))
    return [calls[i] for i in sorted(range(len(calls)), key=lambda i: (rank[calls[i].item_id] // block, tie[i]))]


def execute(calls: list[CatCall], items: list[Item], client_for, out: Path | None = None, workers: int | dict = 8,
            checkpoint_every: int = 200, arm_name: str = "") -> pd.DataFrame:
    """Run every call (one thread pool per system, all at once); a failed call is recorded, never dropped."""
    by_key = {(it.item_id, it.variant): it for it in items}
    rows: list[dict] = []
    lock = threading.Lock()

    def one(c: CatCall) -> dict:
        try:
            r = client_for(c.system, c.repeat).answer(by_key[(c.item_id, c.variant)])
        except Exception as e:
            r = empty_row(repr(e)[:300])
        r = {"arm": arm_name, "system": c.system, "item_id": c.item_id, "variant": c.variant, "repeat": c.repeat, **r}
        r["probs"] = json.dumps(r["probs"]) if r.get("probs") else None
        return r

    by_system: dict[str, list[CatCall]] = {}
    for c in calls:
        by_system.setdefault(c.system, []).append(c)
    per = workers if isinstance(workers, dict) else {"default": workers}
    pools = {s: ThreadPoolExecutor(max_workers=per.get(s, per["default"]), thread_name_prefix=s) for s in by_system}
    try:
        futs = [pools[s].submit(one, c) for s, cs in by_system.items() for c in cs]
        for i, f in enumerate(as_completed(futs), 1):
            with lock:
                rows.append(f.result())
                if out is not None and i % checkpoint_every == 0:
                    pd.DataFrame(rows).to_parquet(out, index=False)
                    print(f"{i}/{len(calls)} calls", flush=True)
    finally:
        for p in pools.values():
            p.shutdown(wait=True)
    df = pd.DataFrame(rows)
    if out is not None:
        df.to_parquet(out, index=False)
    return df


# ------------------------------------------------------------------------------------------------ batch route
def pending_batch(calls: list[CatCall], items: list[Item], arm: ArmSpec, store: RawStore) -> list[tuple[str, str, dict]]:
    """(family, custom_id, request) for every planned chatbot call on the batch route without a stored response."""
    by_key = {(it.item_id, it.variant): it for it in items}
    out, seen, clients = [], set(), {}
    for c in calls:
        if c.system == "jev":
            continue
        key = (c.system, c.repeat)
        if key not in clients:
            clients[key] = CategoricalLLMClient(c.system, store, arm, c.repeat)
        cl = clients[key]
        if cl.batch is None:
            continue
        req = cl.request(by_key[(c.item_id, c.variant)])
        cid = _sha(req)
        if cid in seen or store.get(req) is not None:
            continue
        seen.add(cid)
        out.append((split_system(c.system)[0], cid, req))
    return out


# ------------------------------------------------------------------------------------------------ keys and batch budget
def _repo_path(p) -> Path:
    """A config path: absolute as given, relative to the repository root otherwise."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_keys(arm: ArmSpec) -> Path:
    """API keys from the .env file named by config keys.env_file (relative to the repository root unless absolute):
    never printed; a variable already set is kept."""
    from dotenv import load_dotenv
    p = _repo_path(arm.config["keys"]["env_file"])
    if not p.exists():
        raise SystemExit(f"key file {p} not found (config/{arm.name}.yaml keys.env_file)")
    load_dotenv(p, override=False)
    return p


DEFAULT_PROMPT_TOKENS = 1300     # prompt tokens assumed when a request cannot be read


def _slash(p) -> str:
    return str(p).replace("\\", "/")


def request_worst_usd(req: dict | None, fam: str) -> float:
    """OpenRouter's worst case for one batch request: prompt tokens at the batch input price plus the whole output cap
    at the batch output price. Prompt tokens are estimated from the request text (4 characters per token)."""
    fams = MODELS["batch"]["families"]
    b = fams.get(fam) or max(fams.values(), key=lambda x: x["price_out"])     # unknown family: the dearest prices
    if req is None:
        tin, cap = DEFAULT_PROMPT_TOKENS, int((MODELS["families"].get(fam) or {}).get("max_tokens", 16000))
    else:
        tin = estimate_tokens(" ".join(str(m.get("content") or "") for m in req.get("messages", [])))
        cap = int(req.get("max_tokens") or 16000)
    return (tin * b["price_in"] + cap * b["price_out"]) / 1e6


def manifest_files(roots: list, extra_dirs: list = ()) -> list[Path]:
    """batches/manifest.json of every run folder under the roots (relative roots: under the repository root) and of the
    extra run folders (this arm's store), each once."""
    out = {}
    for r in roots:
        for f in sorted(_repo_path(r).glob("*/batches/manifest.json")):
            out[f.resolve()] = f
    for d in extra_dirs:
        f = Path(d) / "batches" / "manifest.json"
        if f.exists():
            out[f.resolve()] = f
    return list(out.values())


def open_holds(files: list[Path]) -> dict[str, float]:
    """Worst-case holds of every open submission (neither ingested nor dropped; an unclear 'posting' entry counts: it
    may exist), per run folder. Read only."""
    out: dict[str, float] = {}
    for f in files:
        tot = 0.0
        for b in json.loads(f.read_text(encoding="utf-8")):
            if b.get("ingested") or b.get("status") == "dropped":
                continue
            rf = f.parent / f"{b['key']}.requests.json"
            if rf.exists():
                tot += sum(request_worst_usd(r, b["family"]) for r in json.loads(rf.read_text(encoding="utf-8")).values())
            else:
                tot += int(b.get("n") or 0) * request_worst_usd(None, b["family"])
        out[_slash(f.parent.parent)] = tot
    return out


def openrouter_account(get=None) -> dict:
    """The key's usage and limit (GET /api/v1/key) and the account balance, credits minus usage (GET /api/v1/credits;
    usage includes the holds OpenRouter has booked so far). Account reads only; no model call."""
    from . import batch as BT
    get = get or BT._get
    k = (get(f"{OPENROUTER}/api/v1/key", None) or {}).get("data") or {}
    c = (get(f"{OPENROUTER}/api/v1/credits", None) or {}).get("data") or {}
    if "usage" not in k or "total_credits" not in c:
        raise SystemExit("could not read the key's usage or the account balance: nothing submitted")
    return {"key_usage": float(k["usage"]), "key_limit": k.get("limit"), "key_limit_remaining": k.get("limit_remaining"),
            "balance": float(c["total_credits"]) - float(c["total_usage"])}


def budget_check(account: dict, holds: dict[str, float], new_hold: float, cfg: dict, own: str = "") -> dict:
    """Before every batch submission: (1) when config budget.project_stop_usd is set (null: no stop), the key's total
    usage plus every outstanding hold plus this submission's hold must not exceed it (usage may already include some
    holds, so this errs high); (2) the balance must cover every outstanding hold (other runs' batches in flight
    included) plus this submission's; (3) the key's remaining limit must cover this submission's hold. With
    project_stop_hard: false rule (1) warns and does not refuse. With holds_in_balance: true rule (2) asks the balance
    to cover this submission alone, since the balance OpenRouter reports has already taken off the holds it booked."""
    held = float(sum(holds.values()))
    total = account["key_usage"] + held + new_hold
    stop = cfg.get("project_stop_usd")
    stop = None if stop is None else float(stop)
    reasons, warnings = [], []
    if stop is not None and total > stop:
        (reasons if cfg.get("project_stop_hard", True) else warnings).append(
            f"usage ${account['key_usage']:,.2f} + holds ${held:,.2f} + this submission ${new_hold:,.2f} = "
            f"${total:,.2f}, above the ${stop:,.0f} project stop")
    once = bool(cfg.get("holds_in_balance", False))
    if account["balance"] < (0.0 if once else held) + new_hold:
        reasons.append(f"balance ${account['balance']:,.2f} does not cover this submission ${new_hold:,.2f}" if once else
                       f"balance ${account['balance']:,.2f} does not cover the holds in flight ${held:,.2f} plus this "
                       f"submission ${new_hold:,.2f}")
    lr = account.get("key_limit_remaining")
    if lr is not None and float(lr) < new_hold:
        reasons.append(f"key limit remaining ${float(lr):,.2f} below this submission's hold ${new_hold:,.2f}")
    others = {k: v for k, v in holds.items() if v > 0 and k != own}
    return {"ok": not reasons, "reasons": reasons, "warnings": warnings, "key_usage": account["key_usage"], "balance": account["balance"], "holds": holds, "held": held, "new_hold": new_hold, "total": total,
            "stop": stop, "other_runs_in_flight": sorted(others)}


MIXED_CONFIG = "cannot share an upstream input"   # Google's batch endpoint via OpenRouter


def config_rejections(man) -> set[str]:
    """Keys of the submissions the batch endpoint rejected as a whole because their requests mixed configurations
    (Gemini: "cannot share an upstream input with the requests before it ... Split the requests into one batch per
    configuration"; e.g. 4- and 5-option answer schemas in one batch). No request of such a batch was processed."""
    out = set()
    for b in man.items:
        f = man.dir / f"{b['key']}.result.json"
        if b.get("status") != "failed" or not f.exists():
            continue
        h = json.loads(f.read_text(encoding="utf-8"))
        if MIXED_CONFIG in str((h.get("error") or {}).get("message") or "") and not (h.get("request_counts") or {}).get("completed"):
            out.add(b["key"])
    return out


def batch_attempts(man) -> dict[str, int]:
    """batch.Manifest.attempts without the whole-batch rejections for mixed configurations (config_rejections): those
    requests were never processed, so the rejection is not one of their attempts."""
    skip, out = config_rejections(man), {}
    for b in man.items:
        if b.get("status") != "dropped" and b["key"] not in skip:
            for cid in man.requests(b["key"]):
                out[cid] = out.get(cid, 0) + 1
    return out


def batch_config(req: dict) -> str:
    """What must be shared by every request of one batch (Google's endpoint): the response format (answer schema)."""
    return json.dumps(req.get("response_format"), sort_keys=True)


def submit_batch(todo: list[tuple[str, str, dict]], store: RawStore, arm: ArmSpec, max_requests: int | None = None,
                 dry: bool = False, post=None, max_total: int | None = None, account=None) -> list[dict]:
    """batch.submit for a given list of pending requests (batch.submit builds its list from the arm 1 prompt): the same
    manifest, retry limit, record-before-POST and single POST per chunk; one batch per family and answer schema.
    Before anything is posted: the budget check over every open batch (the run folders under budget.manifest_roots and
    this arm's store). Collect, status, adopt and drop are batch.collect etc. on the arm's store."""
    from . import batch as BT
    post = post or BT._post_once
    man = BT.Manifest(store)
    if man.posting():
        raise SystemExit("A previous submission may or may not exist on OpenRouter: "
                         + ", ".join(f"{b['key']} ({b['family']}, {b['n']} requests)" for b in man.posting())
                         + ". Check openrouter.ai; then adopt <key> <batch_id>, or drop <key> if none was created.")
    busy = man.pending_ids()
    tries, cap = batch_attempts(man), int(MODELS["batch"].get("max_attempts", 3))
    todo = [t for t in todo if t[1] not in busy]
    spent = [t for t in todo if tries.get(t[1], 0) >= cap]
    if spent:
        print(f"{len(spent)} requests reached {cap} submissions without a completed response: not sent again; they "
              "count as unusable.", flush=True)
    todo = [t for t in todo if tries.get(t[1], 0) < cap]
    if max_total is not None:
        todo = todo[:max_total]
    n = max_requests or int(MODELS["batch"].get("max_requests", 1000))
    if todo and not dry:
        bc = arm.config["budget"]
        holds = open_holds(manifest_files(bc.get("manifest_roots") or [], [store.run_dir]))
        acc = (account or openrouter_account)()
        new = sum(request_worst_usd(req, fam) for fam, _, req in todo)
        chk = budget_check(acc, holds, new, bc, _slash(Path(store.run_dir)))
        stop = f"against the ${chk['stop']:,.0f} stop" if chk["stop"] is not None else "(no project stop set)"
        print(f"batch budget: key usage ${chk['key_usage']:,.2f}, balance ${chk['balance']:,.2f}, holds in flight "
              f"${chk['held']:,.2f} ({', '.join(chk['other_runs_in_flight']) or 'no other run'}), this submission "
              f"${new:,.2f} worst case; total ${chk['total']:,.2f} {stop}", flush=True)
        for w in chk["warnings"]:
            print(f"WARNING (submitted anyway, budget.project_stop_hard is false): {w}", flush=True)
        if not chk["ok"]:
            raise SystemExit("BATCH NOT SUBMITTED: " + "; ".join(chk["reasons"]) + ".")
    made = []
    groups: dict[tuple[str, str], list] = {}                  # one batch per family and answer schema (batch_config)
    for fam, cid, req in todo:
        groups.setdefault((fam, batch_config(req)), []).append((cid, req))
    for (fam, _), items in sorted(groups.items()):
        for i in range(0, len(items), n):
            chunk = items[i:i + n]
            if dry:
                made.append({"family": fam, "n": len(chunk)})
                continue
            key = uuid.uuid4().hex[:12]
            (man.dir / f"{key}.requests.json").write_text(json.dumps({cid: req for cid, req in chunk}), encoding="utf-8")
            entry = {"key": key, "id": None, "family": fam, "n": len(chunk), "submitted": time.time(),
                     "provider_pinned": MODELS["batch"]["families"][fam]["provider_only"], "status": "posting",
                     "ingested": False}
            man.items.append(entry); man.save()
            try:
                resp = post(BT.payload_for(fam, chunk))
            except BT.Refused:
                man.items.remove(entry); man.save()
                raise
            except Exception as e:
                man.save()
                raise SystemExit(f"Submission {key} unclear ({e!r:.300}). Nothing more is sent. Check openrouter.ai for a "
                                 f"batch of {len(chunk)} {fam} requests; adopt it or drop {key}.")
            bid = resp.get("id")
            if not bid:
                man.save()
                raise SystemExit(f"Submission {key} returned no id: {json.dumps(resp)[:400]}")
            entry.update(id=bid, status=resp.get("status"))
            man.save(); made.append(entry)
            print(f"submitted {bid}: {fam} {len(chunk)} requests ({resp.get('status')})", flush=True)
    return made

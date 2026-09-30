"""Arm 3, Jev used as documented.

One Jev request per report: the whole report in `state` as a JSON object, four Choice questions in the same request
(T, positive regional nodes in bands, tumor deposits, M; config/docuse_staging.yaml), in two versions (config variants):
documented sends each option's what, not_for and examples; documented_notes sends the same with note_what appended to
what and note_not_for appended to not_for, each joined by one space. The stage is computed in code:
N from the node band and deposits with crc.n_category, the stage group with crc.stage_group over config/crc_ajcc8.yaml
(the functions the key was built with), the scored level with crc.level_of. The ten-level distribution is the sum, over
the answer combinations that map to each level, of the product of the four answer distributions; the routing
confidence is the product of the four top probabilities.

Transport, provider block and raw store come from clients.JevClient and categorical.CategoricalJevClient; rows are
categorical._row rows (the columns of results/arm3/calls.parquet), so categorical_analysis.analyze scores them beside
the chatbots' stored answers. Nothing here sends a request unless a client's answer() finds no stored response.
template(variant) is the one request body of a version with the report as {pathology_report}.
"""
from __future__ import annotations

import functools
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import categorical as K
from . import crc
from .clients import MODELS, ROOT, RawStore, _real01, estimate_tokens

CONFIG_PATH = ROOT / "config" / "docuse_staging.yaml"
SENT_FIELDS = ("what", "not_for", "examples")        # docs.typesafe.ai/primitives/choice criterion object
NOTE_FIELDS = {"what": "note_what", "not_for": "note_not_for"}   # appended in documented_notes, never sent alone
QUESTION_IDS = ("t_category", "regional_nodes", "tumor_deposits", "distant_metastasis")
VARIANTS = ("documented", "documented_notes")
PLACEHOLDER = "{pathology_report}"
# The option of each question that says the report does not give the part (no stage, except deposits: staged as absent).
UNSTATED = {"t_category": "cannot be assessed", "regional_nodes": "not stated", "tumor_deposits": "not stated",
            "distant_metastasis": "not stated"}


def load_config(path: Path = CONFIG_PATH) -> dict:
    c = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert tuple(c["questions"]) == QUESTION_IDS, f"questions must be {QUESTION_IDS}"
    assert tuple(c["variants"]) == VARIANTS, f"variants must be {VARIANTS}"
    q = c["questions"]
    assert q["t_category"]["options"][UNSTATED["t_category"]]["code"] == "TX"
    assert q["regional_nodes"]["options"][UNSTATED["regional_nodes"]]["band"] is None
    assert UNSTATED["tumor_deposits"] in q["tumor_deposits"]["options"]
    assert q["distant_metastasis"]["options"][UNSTATED["distant_metastasis"]]["code"] is None
    return c


CFG = load_config()


def config_sha256(path: Path = CONFIG_PATH) -> str:
    """sha256 (lowercase hex) of the config text with LF line ends, so any checkout gives the same value."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()


def options(qid: str, cfg: dict = CFG) -> tuple[str, ...]:
    return tuple(str(k) for k in cfg["questions"][qid]["options"])


# ------------------------------------------------------------------------------------------------ arm spec
def source_arm() -> K.ArmSpec:
    return K.load_arm(ROOT / CFG["source_arm"])


def arm_spec(cfg: dict = CFG) -> K.ArmSpec:
    """arm 3's spec (levels, analysis settings, repeats, timing) under this arm's name, keys and budget; used by the
    shared clients (rows) and the shared analysis. kind stays 'score' (ten ordered levels)."""
    a3 = source_arm()
    conf = {**a3.config, "arm": cfg["arm"], "keys": cfg["keys"], "budget": cfg["budget"], "docuse": cfg}
    return K.ArmSpec(name=cfg["arm"], kind="score", population=cfg["population"], instructions="",
                     answer_key=a3.answer_key, jev_question_id="", system_prompt="", reply_prompt="", user_template="",
                     labels=a3.labels, tiers=a3.tiers, batch=False, config=conf)


LEVELS: tuple[str, ...] = tuple(crc.LEVELS)


# ------------------------------------------------------------------------------------------------ requests
def criterion(opt: dict, variant: str) -> dict:
    """What Jev receives for one option: the documented fields only (source, code and band stay in the config). In
    documented_notes, note_what is appended to what and note_not_for to not_for, each joined by one space."""
    assert variant in VARIANTS, variant
    out = {k: opt[k] for k in SENT_FIELDS if opt.get(k) not in (None, "", [])}
    assert "what" in out, opt
    if variant == "documented_notes":
        for field, note in NOTE_FIELDS.items():
            if opt.get(note):
                assert field in out, (opt, note)
                out[field] = f"{out[field]} {opt[note]}"
    return out


def questions(variant: str, cfg: dict = CFG) -> dict:
    return {qid: {"type": "choice", "instructions": " ".join(q["instructions"].split()),
                  "criteria": {str(k): criterion(v, variant) for k, v in q["options"].items()}}
            for qid, q in cfg["questions"].items()}


def notes(cfg: dict = CFG) -> list[dict]:
    """Every note field of the config: question, option, the field it is appended to, and its text."""
    return [{"question": qid, "option": str(o), "field": field, "text": v[note]}
            for qid, q in cfg["questions"].items() for o, v in q["options"].items()
            for field, note in NOTE_FIELDS.items() if v.get(note)]


def sent_examples(cfg: dict = CFG) -> list[tuple[str, str, str]]:
    """(question, option, example) for every example sent; the two versions send the same examples."""
    return [(qid, str(o), e) for qid, q in cfg["questions"].items() for o, v in q["options"].items()
            for e in v.get("examples", [])]


def state(report: str, cfg: dict = CFG) -> dict:
    return {cfg["state"]["field"]: report.strip()}


class DocuseJevClient(K.CategoricalJevClient):
    """Jev on one report as documented: JevClient's model slug, transport and provider block (allow_fallbacks false),
    RawStore and single flight (CategoricalJevClient._send); the report as a JSON state and the four Choice
    questions."""

    def request(self, item: K.Item) -> dict:
        req = K.JevClient.request(self, item.state, "")   # model, provider, _repeat exactly as every Jev client
        req["state"] = state(item.state)
        req["questions"] = questions(item.variant)
        req.update(_arm=self.arm.name, _item=item.item_id, _variant=item.variant)
        return req

    def row(self, req: dict, cached: dict, item: K.Item) -> dict:
        resp = cached["response"]
        parsed = parse_answers(resp.get("answers"))
        r = K._row(parsed, item, self.arm, resp, cached.get("meta") or {}, str(self.store.path(req)))
        r["top_prob"] = parsed["top_prob"]               # routing confidence: product of the four top probabilities
        r["choice_not_top"] = parsed["choice_not_top"]
        r["_parts"] = parsed["parts"]
        r["_p_unstaged"] = parsed.get("p_unstaged")
        return r

    def answer(self, item: K.Item) -> dict:
        req = self.request(item)
        cached = self.store.get(req)
        if cached is None:
            with self.store.lock(req):
                cached = self.store.get(req) or self._send(req)
        return self.row(req, cached, item)

    def stored(self, item: K.Item) -> dict | None:
        """The row from the raw store, or None if this request was never answered. Never sends."""
        req = self.request(item)
        cached = self.store.get(req)
        return None if cached is None else self.row(req, cached, item)


def make_factory(arm: K.ArmSpec, store: RawStore, sender: K.Sender | None = None):
    cache: dict[int, DocuseJevClient] = {}

    def get(system: str, repeat: int = 0) -> DocuseJevClient:
        assert system == "jev", "only Jev is called in the documented run"
        if repeat not in cache:
            cache[repeat] = DocuseJevClient(store, arm, repeat, sender=sender)
        return cache[repeat]
    return get


def validate_request(req: dict, item: K.Item) -> list[str]:
    """Problems with one built request (empty list: valid). Checks what the dry run promises."""
    bad = []
    spec = MODELS["jev"]
    if req.get("model") != (spec["slug"] if spec["transport"] == "openrouter" else "jev-1.13.0"):
        bad.append(f"model {req.get('model')}")
    if spec["transport"] == "openrouter" and (req.get("provider") or {}).get("allow_fallbacks") is not False:
        bad.append("provider.allow_fallbacks is not false")
    st = req.get("state")
    if not (isinstance(st, dict) and list(st) == [CFG["state"]["field"]] and st[CFG["state"]["field"]] == item.state.strip()):
        bad.append("state is not the whole report as a one-field JSON object")
    qs = req.get("questions") or {}
    if tuple(qs) != QUESTION_IDS:
        bad.append(f"questions {tuple(qs)}")
    if item.variant not in VARIANTS or req.get("_variant") != item.variant:
        bad.append(f"variant {req.get('_variant')}")
    elif qs != questions(item.variant):
        bad.append(f"questions differ from the {item.variant} template")
    elif body(req) != fill(template(item.variant), item.state):
        bad.append(f"body differs from the {item.variant} template")
    for qid, q in qs.items():
        crit = q.get("criteria") or {}
        if q.get("type") != "choice" or not q.get("instructions"):
            bad.append(f"{qid}: not a Choice question with instructions")
        if tuple(crit) != options(qid) or not 2 <= len(crit) <= K.JEV_CHOICE_MAX_OPTIONS:
            bad.append(f"{qid}: options {tuple(crit)}")
        for k, v in crit.items():
            if not isinstance(v, dict) or set(v) - set(SENT_FIELDS) or not isinstance(v.get("what"), str):
                bad.append(f"{qid}/{k}: criterion is not a what/not_for/examples object")
            elif not all(isinstance(e, str) and e for e in v.get("examples", [])):
                bad.append(f"{qid}/{k}: examples must be non-empty strings")
    if request_tokens(req) > int(spec["context_tokens"]):
        bad.append("request above the context window")
    if not all(k in req for k in ("_repeat", "_arm", "_item", "_variant")):
        bad.append("cache keys missing")
    return bad


def request_tokens(req: dict) -> int:
    """Input tokens estimated as the existing projections do (clients.estimate_tokens: 4 characters per token) over the
    body sent."""
    return estimate_tokens(json.dumps(body(req), ensure_ascii=False))


# ------------------------------------------------------------------------------------------------ one template per version
def body(req: dict) -> dict:
    """The body JevClient._send posts: the request without its cache keys (the keys starting with '_')."""
    return {k: v for k, v in req.items() if not k.startswith("_")}


@functools.lru_cache(maxsize=None)
def _template_text(variant: str) -> str:
    it = K.Item("{template}", PLACEHOLDER, LEVELS[0], LEVELS, LEVELS, (None,) * len(LEVELS), "", "", variant)
    return json.dumps(body(DocuseJevClient(RawStore(ROOT / "runs" / "_unused"), arm_spec()).request(it)),
                      ensure_ascii=False)


def template(variant: str) -> dict:
    """The one request body of a version: the built request with {pathology_report} as the report."""
    return json.loads(_template_text(variant))


def fill(tmpl: dict, report: str) -> dict:
    """A template with the report in place of {pathology_report} (the report stripped, as state() does)."""
    out = json.loads(json.dumps(tmpl))
    f = CFG["state"]["field"]
    assert out["state"][f] == PLACEHOLDER
    out["state"][f] = report.strip()
    return out


def dumps(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def as_template_text(req: dict, report: str) -> str:
    """The body's JSON text with the report's JSON text replaced by {pathology_report}; raises if the report text is
    not in the body exactly once."""
    text, rep = dumps(body(req)), json.dumps(report.strip(), ensure_ascii=False)[1:-1]
    if text.count(rep) != 1:
        raise ValueError(f"the report text occurs {text.count(rep)} times in the body")
    return text.replace(rep, PLACEHOLDER)


def strip_notes(text: str, cfg: dict = CFG) -> str:
    """A documented_notes body's JSON text with every appended note (one space and the note) removed."""
    for t in sorted({n["text"] for n in notes(cfg)}, key=len, reverse=True):
        text = text.replace(" " + json.dumps(t, ensure_ascii=False)[1:-1], "")
    return text


def words(text: str) -> list[str]:
    """Lower-case runs of letters and digits (punctuation and spacing ignored), for the example-overlap check."""
    return re.findall(r"[a-z0-9]+", text.lower())


def example_overlaps(reports: list[str], run: int = 8, cfg: dict = CFG) -> list[dict]:
    """Sent examples found in a report: the example as a case-insensitive substring, the example's words in sequence,
    or any `run` consecutive words of the example. One entry per example that hits, with the report counts."""
    rl = [r.lower() for r in reports]
    rw = [" " + " ".join(words(r)) + " " for r in reports]
    out = []
    for qid, opt, ex in sent_examples(cfg):
        w = words(ex)
        whole = " " + " ".join(w) + " "
        grams = {" " + " ".join(w[i:i + run]) + " " for i in range(len(w) - run + 1)}
        hit = {"substring": sum(ex.lower() in r for r in rl), "words": sum(whole in r for r in rw),
               f"run_of_{run}": sum(any(g in r for g in grams) for r in rw)}
        if any(hit.values()):
            out.append({"question": qid, "option": opt, "example": ex, **hit})
    return out


# ------------------------------------------------------------------------------------------------ parts -> stage
def t_code(opt: str) -> str:
    return CFG["questions"]["t_category"]["options"][opt]["code"]


def m_code(opt: str) -> str:
    return CFG["questions"]["distant_metastasis"]["options"][opt]["code"]


def deposits_code(opt: str) -> int:
    return int(CFG["questions"]["tumor_deposits"]["options"][opt]["code"])


def node_band(opt: str) -> tuple[int, int | None] | None:
    b = CFG["questions"]["regional_nodes"]["options"][opt]["band"]
    return None if b is None else (int(b[0]), None if b[1] is None else int(b[1]))


def n_from_parts(nodes: str, deposits: str) -> str | None:
    """N from the node band and the deposits answer with crc.n_category (the key's function); None for NX."""
    b = node_band(nodes)
    return None if b is None else crc.n_category(b[0], deposits_code(deposits))


def stage_from_parts(t: str, nodes: str, deposits: str, m: str) -> str | None:
    """AJCC 8th stage group from the four answers, or None when the table gives none (NX; Tis with a positive node)."""
    n = n_from_parts(nodes, deposits)
    if n is None:
        return None
    try:
        return crc.stage_group(t_code(t), n, m_code(m))
    except ValueError:
        return None


def level_from_parts(t: str, nodes: str, deposits: str, m: str) -> str | None:
    g = stage_from_parts(t, nodes, deposits, m)
    return None if g is None else crc.level_of(g)


def band_of_count(count: int) -> str:
    """The node option whose band holds a count of involved regional nodes."""
    for opt in options("regional_nodes"):
        b = node_band(opt)
        if b is not None and b[0] <= count and (b[1] is None or count <= b[1]):
            return opt
    raise ValueError(f"no band holds {count}")


def _inverse(qid: str, code: str) -> str:
    hits = [o for o, v in CFG["questions"][qid]["options"].items() if v.get("code") == code]
    if len(hits) != 1:
        raise ValueError(f"{qid}: {len(hits)} options code {code}")
    return str(hits[0])


def parts_of_truth(t: str, nodes_involved: int, tumour_deposits: int, m: str) -> dict[str, str]:
    """The four answers a reader who is right on every part gives, from a report's stored truth."""
    return {"t_category": _inverse("t_category", t), "regional_nodes": band_of_count(int(nodes_involved)),
            "tumor_deposits": "present" if int(tumour_deposits) > 0 else "absent",
            "distant_metastasis": _inverse("distant_metastasis", m)}


def matching_options(qid: str, t: str, nodes_involved: int, tumour_deposits: int, m: str) -> list[str]:
    """Every option of a question that a report's true T, node count, deposit count and M satisfy, read from the
    config alone (code, band) and never from parts_of_truth: T and M by code; nodes by the band holding the count; the
    deposits by code (present: at least one), with the stated options only (every report states its deposits and the
    unstated option is its own answer, see UNSTATED)."""
    opts = CFG["questions"][qid]["options"]
    if qid == "t_category":
        return [str(o) for o, v in opts.items() if v["code"] == t]
    if qid == "distant_metastasis":
        return [str(o) for o, v in opts.items() if v["code"] == m]
    if qid == "regional_nodes":
        return [str(o) for o in opts if (b := node_band(str(o))) is not None
                and b[0] <= int(nodes_involved) and (b[1] is None or int(nodes_involved) <= b[1])]
    return [str(o) for o, v in opts.items() if str(o) != UNSTATED[qid] and int(v["code"]) == int(int(tumour_deposits) > 0)]


def parts_of_cell(t: str, n: str, m: str, deposits_with_nodes: bool = False) -> dict[str, str]:
    """The four answers for an AJCC (T, N, M) cell: N1c is no positive node with deposits; a positive N gives the band
    of its node count (crc_ajcc8.yaml generation.nodes_involved), with or without deposits."""
    lo = crc.AJCC["generation"]["nodes_involved"][n][0]
    dep = n == "N1c" or (deposits_with_nodes and lo > 0)
    return parts_of_truth(t, lo, int(dep), m)


# ------------------------------------------------------------------------------------------------ distributions
@dataclass(frozen=True)
class Combos:
    """Every answer combination (T x nodes x deposits x M, config order) and its level index (-1: no stage)."""
    index: np.ndarray        # (n_combos, 4) option indices
    level: np.ndarray        # (n_combos,) level index or -1


def _combos() -> Combos:
    opts = [options(q) for q in QUESTION_IDS]
    idx = np.array(list(itertools.product(*[range(len(o)) for o in opts])), dtype=int)
    lv = np.array([LEVELS.index(l) if (l := level_from_parts(*(opts[k][i] for k, i in enumerate(row)))) else -1
                   for row in idx], dtype=int)
    return Combos(idx, lv)


COMBOS = _combos()


def level_distribution(parts: dict[str, dict[str, float]]) -> tuple[dict[str, float], float]:
    """(P(level) for the ten levels, P(no stage)) from the four answer distributions (option -> probability). Each
    combination's probability is the product of its four answer probabilities; the level's, the sum over its
    combinations. The ten values and P(no stage) add up to the product of the four lists' totals (1 for lists that add
    up to 1)."""
    arrs = [np.array([float(parts[q].get(o, 0.0)) for o in options(q)]) for q in QUESTION_IDS]
    p = np.prod([arrs[k][COMBOS.index[:, k]] for k in range(4)], axis=0)
    lv = np.bincount(COMBOS.level[COMBOS.level >= 0], weights=p[COMBOS.level >= 0], minlength=len(LEVELS))
    return {l: float(lv[i]) for i, l in enumerate(LEVELS)}, float(p[COMBOS.level < 0].sum())


def parse_answers(answers: Any) -> dict:
    """Jev's four Choice answers -> the parsed fields categorical._row takes, plus the routing confidence and the parts.
    Per question the shared rules: categorical.match_label for the choice, categorical.probability_list for the list
    (an omitted option carries zero mass; a list summing to 0.9-1.1 is renormalised; otherwise it is left out). The
    answer is the level of the four choices; a combination the table does not stage is unusable. The ten-level list is
    conditional on a staged combination (divided by 1 - P(no stage); P(no stage) kept as p_unstaged), so every answer
    with four usable lists keeps a list and its routing confidence in the shared Table."""
    out = {"answer": None, "answer_how": None, "probs": None, "prob_status": "missing", "given": None,
           "jev_confidence": None, "expected_level": None, "top_prob": None, "choice_not_top": False,
           "parse": "missing_answer", "parts": {}}
    if not isinstance(answers, dict):
        return out
    choices, dists, confs, status = {}, {}, [], "ok"
    for qid in QUESTION_IDS:
        a = answers.get(qid)
        labs = options(qid)
        part = {"choice": None, "probs": None, "prob_status": "missing", "confidence": None, "top": None}
        if isinstance(a, dict):
            lab, _ = K.match_label(a.get("choice"), labs)
            probs, st, _ = K.probability_list(a.get("probabilities"), labs)
            part.update(choice=lab, probs=probs, prob_status=st, confidence=_real01(a.get("confidence")),
                        top=max(probs.values()) if probs else None)
        out["parts"][qid] = part
        choices[qid], dists[qid] = part["choice"], part["probs"]
        confs.append(part["confidence"])
        if part["probs"] is None and status == "ok":
            status = f"{qid}:{part['prob_status']}"
    if all(c is not None for c in confs):
        out["jev_confidence"] = float(np.prod(confs))
    if any(isinstance(answers.get(q), dict) for q in QUESTION_IDS):
        out["parse"] = "choice_invalid"
    if all(d is not None for d in dists.values()):
        lv, p_none = level_distribution(dists)
        probs = {l: v / (1 - p_none) for l, v in lv.items()} if p_none < 1 - 1e-12 else None
        st = "unstaged_mass" if probs is None else "ok" if p_none <= 1e-9 else "renormalised"
        out.update(probs=probs, prob_status=st, given=lv, p_unstaged=p_none,
                   top_prob=float(np.prod([max(d.values()) for d in dists.values()])))
        if probs is not None:
            out["expected_level"] = float(sum(i * probs[l] for i, l in enumerate(LEVELS)))
    else:
        out["prob_status"] = status
    if all(c is not None for c in choices.values()):
        lvl = level_from_parts(*(choices[q] for q in QUESTION_IDS))
        if lvl is None:
            out["parse"] = "unstaged"
        else:
            out.update(answer=lvl, answer_how="parts", parse="parts")
            if out["probs"] is not None:
                out["choice_not_top"] = bool(out["probs"][lvl] < max(out["probs"].values()) - 1e-12)
    return out


# ------------------------------------------------------------------------------------------------ items and plan
def load_frames() -> dict[str, pd.DataFrame]:
    ic = CFG["items"]
    return {k: pd.read_csv(ROOT / ic[k], dtype={ic["id_column"]: str}, keep_default_na=False)
            for k in ("main", "confuser")}


def check_variants(variants) -> tuple[str, ...]:
    v = tuple(VARIANTS if variants is None else variants)
    assert v and set(v) <= set(VARIANTS), v
    return tuple(x for x in VARIANTS if x in v)


def load_items(variants=None, cfg: dict = CFG) -> list[K.Item]:
    """The 1,100 reports (1,000 main, 100 misleading-feature) as shared Items, once per variant (default: both, so
    2,200 items keyed by report id and variant); stratum: level (main) or confuser type."""
    ic, out = cfg["items"], []
    for v in check_variants(variants):
        for which, f in load_frames().items():
            for r in f.to_dict("records"):
                truth = str(r[ic["truth_column"]])
                assert truth in LEVELS, (r[ic["id_column"]], truth)
                out.append(K.Item(r[ic["id_column"]], r[ic["state_column"]].strip(), truth, LEVELS, LEVELS,
                                  (None,) * len(LEVELS),
                                  str(r.get("confuser_type", truth)) if which == "confuser" else truth,
                                  str(r.get("substage", "")), v))
    keys = [(i.item_id, i.variant) for i in out]
    assert len(keys) == len(set(keys)), "report ids repeat across the two item files"
    return out


def seeded_ids(items: list[K.Item], seed: int = int(CFG["seed"])) -> list[str]:
    """The report ids in categorical.block_order's seeded order (the order a spending stop leaves complete)."""
    order = K.block_order([K.CatCall("jev", i) for i in sorted({i.item_id for i in items})], seed)
    return [c.item_id for c in order]


def seeded_order(items: list[K.Item], variant: str = VARIANTS[0], seed: int = int(CFG["seed"])) -> list[K.Item]:
    """One variant's items in the seeded report order."""
    by = {i.item_id: i for i in items if i.variant == variant}
    return [by[i] for i in seeded_ids(items, seed) if i in by]


def repeat_ids() -> list[str]:
    """The 40 reports arm 3 asked three times (config/arm3.yaml repeats)."""
    return K.repeat_ids(source_arm())


def n_repeats() -> int:
    return int(source_arm().config["repeats"]["n_repeats"])


def plan(mode: str, items: list[K.Item], variants=None) -> list[K.CatCall]:
    """Per selected variant (default both): full, all 1,100, repeat 0; repeats, repeats 1..n_repeats-1 on arm 3's 40.
    Both variants of a report fall in the same block of the seeded order."""
    vs = check_variants(variants)
    its = [i for i in items if i.variant in vs]
    if mode == "full":
        return K.plan_calls(its, ["jev"], [], 1, int(CFG["seed"]))
    if mode == "repeats":
        reps = set(repeat_ids())
        return [c for c in K.plan_calls([i for i in its if i.item_id in reps], ["jev"], sorted(reps), n_repeats(),
                                        int(CFG["seed"])) if c.repeat > 0]
    raise ValueError(mode)


# ------------------------------------------------------------------------------------------------ calls table
CALL_COLUMNS = ["arm", "system", "item_id", "variant", "repeat", "answer", "answer_presented", "valid", "probs",
                "prob_status", "top_prob", "choice_not_top", "jev_confidence", "expected_level", "parse", "answer_how",
                "prob_alias", "provider", "model_reported", "route", "finish_reason", "response_id", "latency_s",
                "tokens_in", "tokens_out", "tokens_reasoning", "usd_reported", "raw_path", "error", "simulated"]


def calls_table(store: RawStore, arm: K.ArmSpec, items: list[K.Item], calls: list[K.CatCall]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(calls, parts) from the raw store for the planned calls that have a stored response; never sends. calls: the
    columns of results/arm3/calls.parquet (system 'jev', variant documented or documented_notes); parts: the four
    answers per call."""
    by = {(i.item_id, i.variant): i for i in items}
    factory = make_factory(arm, store)
    rows, parts = [], []
    for c in calls:
        r = factory("jev", c.repeat).stored(by[(c.item_id, c.variant)])
        if r is None:
            continue
        p = r.pop("_parts")
        r["probs"] = json.dumps(r["probs"]) if r.get("probs") else None
        rows.append({"arm": arm.name, "system": "jev", "item_id": c.item_id, "variant": c.variant, "repeat": c.repeat, **r})
        prow = {"item_id": c.item_id, "variant": c.variant, "repeat": c.repeat, "p_unstaged": r.pop("_p_unstaged")}
        for q, d in p.items():
            prow.update({f"{q}_choice": d["choice"], f"{q}_top": d["top"], f"{q}_confidence": d["confidence"],
                         f"{q}_prob_status": d["prob_status"], f"{q}_probs": json.dumps(d["probs"]) if d["probs"] else None})
        parts.append(prow)
    df = pd.DataFrame(rows, columns=CALL_COLUMNS)
    return df, pd.DataFrame(parts)


def all_planned(items: list[K.Item]) -> list[K.CatCall]:
    """Every planned call of both variants (full and repeats) whose items are in `items`."""
    vs = sorted({i.variant for i in items}, key=VARIANTS.index)
    return plan("full", items, variants=vs) + plan("repeats", items, variants=vs)


# ------------------------------------------------------------------------------------------------ synthetic answers
def synthetic_answers(parts: dict[str, str], rng: np.random.Generator, p_right: float = 0.8) -> dict:
    """Synthetic Jev answers for tests and --simulate (never real output): per question the right option with
    probability p_right, else another; probabilities on a 0.01 grid adding up to 1."""
    out = {}
    for q in QUESTION_IDS:
        labs = options(q)
        right = labs.index(parts[q])
        pick = right if rng.random() < p_right else int(rng.choice([k for k in range(len(labs)) if k != right]))
        w = rng.dirichlet(np.ones(len(labs)) * 0.3)
        w = 0.5 * w + 0.5 * np.eye(len(labs))[pick]
        c = np.floor(w * 100).astype(int)
        c[pick] += 100 - c.sum()
        probs = {l: round(float(x) / 100, 2) for l, x in zip(labs, c)}
        out[q] = {"choice": labs[pick], "probabilities": probs, "confidence": round(float(c.max()) / 100, 2)}
    return out


def synthetic_sender(truth_parts: dict[str, dict[str, str]], seed: int = 0, p_right: float = 0.8):
    """A categorical.Sender returning synthetic Jev responses keyed by the request's report id."""
    def send(system: str, req: dict) -> dict:
        rng = np.random.default_rng([seed, int(hashlib.sha256(K._sha(req).encode()).hexdigest()[:8], 16)])
        tin = request_tokens(req)
        return {"id": f"sim-{K._sha(req)}", "model": MODELS["jev"]["slug"] + "-sim", "provider": "TypeSafe",
                "answers": synthetic_answers(truth_parts[req["_item"]], rng, p_right),
                "usage": {"input_tokens": tin, "output_tokens": 0}, "_sim_latency_s": float(rng.uniform(0.2, 0.6))}
    return send


def truth_parts_of_frames() -> dict[str, dict[str, str]]:
    out = {}
    for f in load_frames().values():
        for r in f.to_dict("records"):
            out[r["report_id"]] = parts_of_truth(r["t"], r["nodes_involved"], r["tumour_deposits"], r["m"])
    return out


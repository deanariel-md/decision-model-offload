"""Every LLM in every form of the colon cancer staging question that Jev answered: the LLM request builder. Nothing
here sends a request.

Every LLM request is built from Jev's stored request for the same form and report (the raw-store JSON's `request`
without its cache keys): the same report, the same question text and the same options, so the only differences from
Jev are the reply-format system message and the field labels. Base settings come from the shared client code:
categorical.CategoricalLLMClient(system, ..., standard=True).request() (clients.LLMClient.request on the standard
route: model slug or HF model id, max_tokens, provider {"only": ..., "allow_fallbacks": false} on OpenRouter and no
provider block on the HF router, seed where the family lists it, no temperature or reasoning parameter). Then the
messages and response_format are set as below, and the cache keys are _repeat 0, _arm "arm3_llm_forms", _item the
report id and _variant the form.

Jev's stored requests are read from the calls tables its runs write (JEV_TABLES: scripts/run_categorical.py for the
first 1,000 reports and the misleading-feature reports, scripts/heldout/run.py, scripts/docuse/run_staging.py,
scripts/docuse/run_structure_only.py, scripts/nonregional/run_nonregional.py, scripts/arm3_jev_staircase.py and
scripts/arm3_jev_parts.py); the 240 reports are data/arm3_pool/pool240.csv (scripts/make_pool240.py), with each report
set's item file.

Single-question forms (names, named_field, definitions): arm 3's system text and user template (config/arm3.yaml,
byte-identical to the main arm 3 runs on the main reports), with Jev's report, instructions and criteria. Four-question
forms (structure_only, documented, documented_no_examples, documented_notes): FOUR_Q_SYSTEM, and the report followed by
Jev's `questions` object as JSON (indent 2, key order kept); the answer schema lists each question's options in Jev's
order.

build() runs the content assertion (check_content) and the settings check on every request and raises ContentMismatch
on any problem; a form whose Jev questions object differs across reports (template_problems) is refused as a whole.
"""
from __future__ import annotations

import functools
import hashlib
import json
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import categorical as K
from .clients import MODELS, ROOT, RawStore, _sha

# ------------------------------------------------------------------------------------------------ names
SYSTEMS = ("gpt", "claude", "gemini", "muse", "glm", "gpt_free", "claude_free", "gemini_free", "medgemma", "gemma")
FORMS = ("names", "named_field", "definitions", "structure_only", "documented", "documented_no_examples",
         "documented_notes")
QUEUE_ORDER = ("documented", "structure_only", "definitions", "documented_no_examples", "documented_notes", "names",
               "named_field")
SINGLE_FORMS = ("names", "named_field", "definitions")
FOUR_FORMS = ("structure_only", "documented", "documented_no_examples", "documented_notes")
SETS = ("main", "heldout", "misleading_feature", "nonregional_sites")
SET_COUNTS = {"main": 102, "heldout": 18, "misleading_feature": 100, "nonregional_sites": 20}
N_POOL = 240

FOUR_Q_SYSTEM = ("For each question, choose the option that is correct, and give the probability that each option is "
                 "correct, as numbers between 0 and 1 that add up to 1. Always choose one option for each question. "
                 "Reply with a JSON object with one key for each question ID, each holding an object with two keys, "
                 "\"choice\" (the option you choose, written exactly as in the question's criteria) and "
                 "\"probabilities\" (an object with one number for each option), and nothing else.")
SCHEMA_NAME_FOUR = "arm3_parts_answer"
ARM_NAME = "arm3_llm_forms"
STATE_FIELD = "pathology_report"
PLACEHOLDER = "{pathology_report}"
JEV_QUESTION_ID = "crc_stage_group"                 # config/arm3.yaml jev.question_id
FOUR_QUESTION_IDS = ("t_category", "regional_nodes", "tumor_deposits", "distant_metastasis")
FOUR_HEAD, FOUR_SEP = "Pathology report:\n", "\n\nQuestions:\n"
JEV_MODEL = "typesafe/jev-1.13-20260917"            # the model every stored Jev answer should name
CACHE_KEYS = ("_repeat", "_arm", "_item", "_variant")

# ------------------------------------------------------------------------------------------------ files (read only)
POOL_CSV = ROOT / "data" / "arm3_pool" / "pool240.csv"
ITEMS = {"main": ROOT / "data/arm3/items.csv", "heldout": ROOT / "data/arm3_heldout/items.csv",
         "misleading_feature": ROOT / "data/arm3/confuser_items.csv",
         "nonregional_sites": ROOT / "data/arm3_nonregional_sites/items.csv"}
UNUSED_STORE = ROOT / "runs" / "_unused"             # clients need a store; request() never touches it


def _sources() -> dict[str, dict[str, tuple[Path, str | None]]]:
    """form -> report set -> (Jev calls table, value of its `set` column or None). Rows are system jev, repeat 0,
    variant == form, restricted to the set's pool reports."""
    a3, cf = ROOT / "results/arm3/calls.parquet", ROOT / "results/arm3_confuser/calls.parquet"
    hc = ROOT / "results/arm3_heldout/calls.parquet"
    st, parts = ROOT / "results/arm3_jev_staircase/calls.parquet", ROOT / "results/arm3_jev_parts/calls.parquet"
    so = ROOT / "results/arm3_docuse_structure_only"
    dc, dh = ROOT / "results/arm3_docuse/calls.parquet", ROOT / "results/arm3_heldout/documented_calls.parquet"
    dn = ROOT / "results/arm3_nonregional_sites/calls_jev.parquet"
    documented = {"main": (dc, None), "heldout": (dh, None), "misleading_feature": (dc, None),
                  "nonregional_sites": (dn, None)}
    return {
        "names": {"main": (a3, None), "heldout": (hc, None), "misleading_feature": (cf, None),
                  "nonregional_sites": (st, "nonregional")},
        "named_field": {s: (parts, s) for s in SETS},
        "definitions": {"main": (a3, None), "heldout": (hc, None), "misleading_feature": (st, "confuser"),
                        "nonregional_sites": (st, "nonregional")},
        "structure_only": {"main": (so / "main/calls.parquet", None), "heldout": (so / "heldout/calls.parquet", None),
                           "misleading_feature": (so / "confuser/calls.parquet", None),
                           "nonregional_sites": (so / "nonregional/calls.parquet", None)},
        "documented": dict(documented),
        "documented_no_examples": {s: (parts, s) for s in SETS},
        "documented_notes": dict(documented),
    }


JEV_TABLES = _sources()


class ContentMismatch(RuntimeError):
    """A built request that is not Jev's content in the layout above; that form stops."""


# ------------------------------------------------------------------------------------------------ arm 3 and the pool
@functools.lru_cache(maxsize=None)
def arm() -> K.ArmSpec:
    return K.load_arm("arm3")


LEVELS: tuple[str, ...] = K.load_arm("arm3").labels
assert all(MODELS["families"][s].get("structured_outputs") for s in SYSTEMS), "every family sends json_schema"
assert not any(MODELS["families"][s].get("disabled") for s in SYSTEMS)


@functools.lru_cache(maxsize=None)
def _pool() -> pd.DataFrame:
    p = pd.read_csv(POOL_CSV, dtype=str, keep_default_na=False)
    assert list(p.columns) == ["report_id", "set"], list(p.columns)
    assert len(p) == N_POOL and not p.report_id.duplicated().any(), len(p)
    assert p.set.value_counts().to_dict() == SET_COUNTS, p.set.value_counts().to_dict()
    frames = []
    for s in SETS:
        f = pd.read_csv(ITEMS[s], dtype=str, keep_default_na=False)
        assert not f.report_id.duplicated().any(), ITEMS[s]
        f = f.set_index("report_id")
        ids = p.report_id[p.set == s].tolist()
        missing = sorted(set(ids) - set(f.index))
        assert not missing, f"{s}: {len(missing)} pool reports missing from {ITEMS[s]}, e.g. {missing[:3]}"
        frames.append(pd.DataFrame({"report_id": ids, "set": s, "level": f.loc[ids, "level"].tolist(),
                                    "report": [str(r).strip() for r in f.loc[ids, "report"]]}))
    df = pd.concat(frames, ignore_index=True)
    order = {r: i for i, r in enumerate(p.report_id)}
    df = df.iloc[sorted(range(len(df)), key=lambda i: order[df.report_id.iat[i]])].reset_index(drop=True)
    assert len(df) == N_POOL and df.level.isin(LEVELS).all() and (df.report.str.len() > 0).all()
    return df


def pool() -> pd.DataFrame:
    """The 240 reports: report_id, set, level (truth), report (the items file's text, stripped); pool240.csv order."""
    return _pool().copy()


@functools.lru_cache(maxsize=None)
def _reports() -> dict[str, str]:
    return dict(zip(_pool().report_id, _pool().report))


@functools.lru_cache(maxsize=None)
def _set_of() -> dict[str, str]:
    return dict(zip(_pool().report_id, _pool().set))


def report_ids() -> list[str]:
    return list(_pool().report_id)


def report_text(report_id: str) -> str:
    """The items file's report, stripped (the reference for the content assertion)."""
    return _reports()[report_id]


def set_of(report_id: str) -> str:
    return _set_of()[report_id]


# ------------------------------------------------------------------------------------------------ Jev's stored requests
@dataclass(frozen=True)
class JevSource:
    form: str
    set: str
    report_id: str
    table: Path                  # the calls table the row came from
    stored_raw_path: str         # raw_path as the table stores it
    raw_path: Path               # the file read (the stored path, or the copy under ROOT/runs/ when that is missing)
    body: dict                   # the stored request without keys starting with "_"
    cache_keys: dict             # the stored request's "_" keys
    model: str | None            # the Jev response's model


def resolve_raw(p: str) -> Path:
    """A stored raw path; when that file is not there (a table written in another folder), the same
    runs/<store>/<file> under ROOT."""
    f = Path(p)
    if f.exists():
        return f
    q = p.replace("\\", "/")
    if "/runs/" in q:
        g = ROOT / q[q.rindex("/runs/") + 1:]
        if g.exists():
            return g
    raise FileNotFoundError(f"Jev raw file not found: {p}")


@functools.lru_cache(maxsize=None)
def _jev_rows(table: str) -> pd.DataFrame:
    c = pd.read_parquet(table)
    return c[(c.system == "jev") & (c["repeat"] == 0)]


@functools.lru_cache(maxsize=None)
def form_index(form: str) -> dict[str, JevSource]:
    """report id -> Jev's stored request for one form, all 240 reports (built once per process). Stops if a report
    has no row or more than one, if the raw file is missing or if the stored request names another report."""
    if form not in FORMS:
        raise ValueError(form)
    p = _pool()
    out: dict[str, JevSource] = {}
    for s in SETS:
        table, set_filter = JEV_TABLES[form][s]
        c = _jev_rows(str(table))
        c = c[c.variant == form]
        if set_filter is not None:
            c = c[c["set"] == set_filter]
        ids = p.report_id[p.set == s].tolist()
        c = c[c.item_id.isin(ids)]
        n = c.item_id.value_counts()
        missing, doubled = sorted(set(ids) - set(n.index)), sorted(n.index[n > 1])
        assert not missing and not doubled, (f"{form}/{s}: Jev source missing for {len(missing)} reports {missing[:3]}, "
                                             f"doubled for {len(doubled)} {doubled[:3]} ({table})")
        for r in c.itertuples():
            f = resolve_raw(r.raw_path)
            d = json.loads(f.read_text(encoding="utf-8"))
            req = d["request"]
            assert req.get("_item") == r.item_id, (form, r.item_id, req.get("_item"), f)
            out[r.item_id] = JevSource(form, s, r.item_id, Path(table), str(r.raw_path), f,
                                       {k: v for k, v in req.items() if not k.startswith("_")},
                                       {k: v for k, v in req.items() if k.startswith("_")},
                                       (d.get("response") or {}).get("model"))
    assert len(out) == N_POOL, (form, len(out))
    return out


def jev_source(form: str, report_id: str) -> tuple[Path, dict]:
    """(raw file, stored request body without '_' keys) of Jev's answer for this form and report."""
    s = form_index(form)[report_id]
    return s.raw_path, s.body


def jev_model(form: str, report_id: str) -> str | None:
    return form_index(form)[report_id].model


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)       # key order kept (no sort_keys)


@functools.lru_cache(maxsize=None)
def template_problems(form: str) -> tuple[str, ...]:
    """Jev's questions object (single-question forms: its instructions and criteria) must be the same for every report
    of a form; one entry per difference found (empty: one template)."""
    idx = form_index(form)
    groups: dict[str, list[JevSource]] = {}
    for s in idx.values():
        groups.setdefault(_dumps(s.body.get("questions")), []).append(s)
    if len(groups) == 1:
        return ()
    out = []
    for i, (k, g) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        sets = Counter(s.set for s in g)
        out.append(f"{form}: Jev questions variant {i + 1} of {len(groups)} (sha {hashlib.sha256(k.encode()).hexdigest()[:12]}) "
                   f"on {len(g)} reports {dict(sets)}, e.g. {g[0].report_id}")
    return tuple(out)


def source_anomalies(form: str) -> list[str]:
    """Things to report about a form's Jev sources that do not stop it: response models other than JEV_MODEL, and model
    or provider blocks that differ across reports."""
    idx = form_index(form)
    out = []
    models = Counter(s.model for s in idx.values())
    if set(models) != {JEV_MODEL}:
        out.append(f"{form}: Jev response models {dict(models)} (expected {JEV_MODEL})")
    rest = Counter(_dumps({k: v for k, v in s.body.items() if k not in ("state", "questions")}) for s in idx.values())
    if len(rest) > 1:
        out.append(f"{form}: Jev body outside state and questions differs across reports: {dict(rest)}")
    return out


def moved_sources(form: str) -> int:
    """Reports whose stored raw path does not exist and whose file was read from ROOT/runs/."""
    return sum(Path(s.stored_raw_path) != s.raw_path for s in form_index(form).values())


# ------------------------------------------------------------------------------------------------ messages and schema
def system_text_for(form: str) -> str:
    if form in SINGLE_FORMS:
        return K.system_text(arm())
    if form in FOUR_FORMS:
        return FOUR_Q_SYSTEM
    raise ValueError(form)


@functools.lru_cache(maxsize=None)
def _single_pieces() -> tuple[str, str, str, str]:
    """The literal text of arm 3's user template around {state}, {instructions} and {options}."""
    parsed = list(string.Formatter().parse(arm().user_template))
    fields = [f for _, f, _, _ in parsed if f is not None]
    assert fields == ["state", "instructions", "options"], fields
    lits = [lit for lit, _, _, _ in parsed]
    if len(lits) == 3:
        lits.append("")
    assert len(lits) == 4, lits
    return tuple(lits)


def single_user(report: str, question: dict) -> str:
    """config/arm3.yaml user_template with the report, Jev's instructions and Jev's criteria joined by newlines."""
    return arm().user_template.format(state=report, instructions=question["instructions"],
                                      options="\n".join(question["criteria"]))


def four_user(report: str, questions: dict) -> str:
    return FOUR_HEAD + report + FOUR_SEP + json.dumps(questions, ensure_ascii=False, indent=2) + "\n"


def four_schema(questions: dict) -> dict:
    """Strict schema: the four question IDs (all required, no others), each {choice: one of its options in Jev's order,
    probabilities: one number 0-1 per option, all required, no others}."""
    props = {}
    for qid, q in questions.items():
        labs = [str(k) for k in q["criteria"]]
        props[qid] = {"type": "object", "properties": {
            "choice": {"type": "string", "enum": labs},
            "probabilities": {"type": "object",
                              "properties": {lab: {"type": "number", "minimum": 0, "maximum": 1} for lab in labs},
                              "required": labs, "additionalProperties": False}},
            "required": ["choice", "probabilities"], "additionalProperties": False}
    return {"name": SCHEMA_NAME_FOUR, "strict": True, "schema": {
        "type": "object", "properties": props, "required": list(questions), "additionalProperties": False}}


def single_schema() -> dict:
    return K.reply_schema(arm(), arm().labels)


def jev_report(form: str, body: dict) -> tuple[str | None, str | None]:
    """(R, problem): Jev's state for names and definitions; state["pathology_report"] for the other forms."""
    st = body.get("state")
    if form in ("names", "definitions"):
        return (st, None) if isinstance(st, str) else (None, f"Jev's state is a {type(st).__name__}, not the report")
    if isinstance(st, dict) and list(st) == [STATE_FIELD] and isinstance(st[STATE_FIELD], str):
        return st[STATE_FIELD], None
    return None, f"Jev's state is not {{\"{STATE_FIELD}\": <report>}}"


# ------------------------------------------------------------------------------------------------ checks
def check_content(req: dict, jev_body: dict, report: str) -> list[str]:
    """The content assertion for one request (empty list: OK). The report R (from Jev's state) equals the
    items file's report (stripped) and occurs once in the user message; single-question: the system text is arm 3's,
    and the report, Jev's instructions and every criterion string sit exactly where the template puts them; the labels
    are the ten levels; four-question: the system text is FOUR_Q_SYSTEM and json.loads of the text after "Questions:\\n"
    equals Jev's questions (key order included), written with indent 2; the schema lists each question's options in
    Jev's order."""
    bad: list[str] = []
    form = req.get("_variant")
    if form not in FORMS:
        return [f"_variant {form!r} is not a form"]
    R, why = jev_report(form, jev_body)
    if R is None:
        return [why]
    if R != report.strip():
        bad.append("the report in Jev's state differs from the items file's report (stripped)")
    msgs = req.get("messages")
    if not (isinstance(msgs, list) and len(msgs) == 2 and [m.get("role") for m in msgs] == ["system", "user"]
            and all(isinstance(m.get("content"), str) for m in msgs) and all(set(m) == {"role", "content"} for m in msgs)):
        return bad + ["messages are not one system and one user message"]
    sysm, user = msgs[0]["content"], msgs[1]["content"]
    if user.count(R) != 1:
        bad.append(f"the report occurs {user.count(R)} times in the user message")
    q = jev_body.get("questions")
    rf = req.get("response_format")
    if form in SINGLE_FORMS:
        if not (isinstance(q, dict) and list(q) == [JEV_QUESTION_ID]):
            return bad + [f"Jev's questions are not the one Score question {JEV_QUESTION_ID}"]
        jq = q[JEV_QUESTION_ID]
        instr, crit = jq.get("instructions"), jq.get("criteria")
        if jq.get("type") != "score" or not isinstance(instr, str) or not (
                isinstance(crit, list) and len(crit) == len(LEVELS) and all(isinstance(c, str) for c in crit)):
            return bad + ["Jev's question is not a Score question with instructions and ten criteria"]
        labs = list(LEVELS)
        if not all(c == lab or c.startswith(lab + ": ") for c, lab in zip(crit, labs)):
            bad.append("Jev's criteria do not start with the ten level names in order")
        if sysm != K.system_text(arm()):
            bad.append("the system message is not arm 3's system text")
        L0, L1, L2, L3 = _single_pieces()
        if not user.startswith(L0 + R + L1):
            bad.append("the report is not where the template puts it")
        else:
            rest = user[len(L0 + R + L1):]
            if not rest.startswith(instr + L2):
                bad.append("the instructions are not Jev's, where the template puts them")
            else:
                opts = rest[len(instr + L2):]
                if opts != "\n".join(crit) + L3:
                    got = (opts[:len(opts) - len(L3)] if L3 and opts.endswith(L3) else opts).split("\n")
                    diff = [i for i in range(max(len(got), len(crit)))
                            if i >= len(got) or i >= len(crit) or got[i] != crit[i]]
                    bad.append(f"the criteria are not Jev's, one per line where the template puts them "
                               f"(positions {diff[:10]})")
        if user != single_user(R, jq):
            bad.append("the user message is not the template filled with Jev's report, instructions and criteria")
        if rf != {"type": "json_schema", "json_schema": single_schema()}:
            bad.append("response_format is not arm 3's schema over the ten levels")
    else:
        if not (isinstance(q, dict) and tuple(q) == FOUR_QUESTION_IDS
                and all(isinstance(v, dict) and v.get("type") == "choice" and isinstance(v.get("criteria"), dict)
                        for v in q.values())):
            return bad + [f"Jev's questions are not the four Choice questions {FOUR_QUESTION_IDS}"]
        if sysm != FOUR_Q_SYSTEM:
            bad.append("the system message is not FOUR_Q_SYSTEM")
        head = FOUR_HEAD + R + FOUR_SEP
        if not user.startswith(head):
            bad.append("the report is not where the template puts it")
        else:
            tail = user[len(head):]
            try:
                got = json.loads(tail)
            except ValueError:
                bad.append("the text after 'Questions:\\n' is not JSON")
            else:
                if _dumps(got) != _dumps(q):
                    diff = [k for k in sorted(set(got) | set(q)) if _dumps(got.get(k)) != _dumps(q.get(k))] \
                        if isinstance(got, dict) else ["(not an object)"]
                    bad.append(f"the questions after 'Questions:\\n' differ from Jev's (key order included): {diff}")
            if tail != json.dumps(q, ensure_ascii=False, indent=2) + "\n":
                bad.append("the questions are not Jev's written as JSON with indent 2 and a final newline")
        if rf != {"type": "json_schema", "json_schema": four_schema(q)}:
            bad.append("response_format is not the four-question schema over Jev's options in Jev's order")
    return bad


def check_settings(system: str, req: dict) -> list[str]:
    """Base settings as the main runs on the standard route and the cache keys (empty list: OK)."""
    spec = MODELS["families"][system]
    hf = spec.get("transport", "openrouter") == "hf_router"
    bad = []
    if req.get("model") != (spec["hf_model"] if hf else spec["slug"]):
        bad.append(f"model {req.get('model')}")
    if req.get("max_tokens") != spec["max_tokens"]:
        bad.append(f"max_tokens {req.get('max_tokens')}")
    if hf and "provider" in req:
        bad.append("a provider block on the HF router")
    if not hf and req.get("provider") != {"only": list(spec["provider_only"]), "allow_fallbacks": False}:
        bad.append(f"provider {req.get('provider')}")
    want_seed = bool(spec.get("seed")) and MODELS["generation"].get("seed") is not None
    if want_seed != ("seed" in req) or (want_seed and req["seed"] != int(MODELS["generation"]["seed"])):
        bad.append(f"seed {req.get('seed')}")
    for k in ("temperature", "reasoning", "_route"):
        if k in req:
            bad.append(f"{k} present")
    if req.get("_repeat") != 0 or req.get("_arm") != ARM_NAME or req.get("_variant") not in FORMS or not req.get("_item"):
        bad.append(f"cache keys {({k: req.get(k) for k in CACHE_KEYS})}")
    keys = {"model", "messages", "max_tokens", "response_format", *CACHE_KEYS} | ({"seed"} if want_seed else set()) \
        | (set() if hf else {"provider"})
    if set(req) != keys:
        bad.append(f"keys {sorted(set(req) ^ keys)} unexpected or missing")
    return bad


# ------------------------------------------------------------------------------------------------ build
@functools.lru_cache(maxsize=None)
def _client(system: str) -> K.CategoricalLLMClient:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}")
    return K.CategoricalLLMClient(system, RawStore(UNUSED_STORE), arm(), repeat=0, standard=True)


def build(system: str, form: str, report_id: str) -> dict:
    """The request (cache keys included) for one system, form and report; never sends. Raises ContentMismatch if the
    form's Jev template is not one, or if this request fails the content assertion or the settings check."""
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}")
    cl = _client(system)
    tp = template_problems(form)
    if tp:
        raise ContentMismatch(f"{form}: Jev's questions object is not the same for every report: " + "; ".join(tp))
    src = form_index(form)[report_id]
    R, why = jev_report(form, src.body)
    if R is None:
        raise ContentMismatch(f"{system}/{form}/{report_id}: {why}")
    q = src.body.get("questions") or {}
    if not isinstance(q, dict) or (form in SINGLE_FORMS and JEV_QUESTION_ID not in q):
        raise ContentMismatch(f"{system}/{form}/{report_id}: Jev's questions are not this form's")
    item = K.Item(report_id, R, LEVELS[0], LEVELS, LEVELS, (None,) * len(LEVELS), "", "", form)
    req = cl.request(item)                 # model, max_tokens, provider, seed, _repeat: LLMClient's standard route
    req["_arm"] = ARM_NAME
    if form in SINGLE_FORMS:
        user, schema = single_user(R, q[JEV_QUESTION_ID]), single_schema()
    else:
        user, schema = four_user(R, q), four_schema(q)
    req["messages"] = [{"role": "system", "content": system_text_for(form)}, {"role": "user", "content": user}]
    req["response_format"] = {"type": "json_schema", "json_schema": schema}
    bad = check_content(req, src.body, report_text(report_id)) + check_settings(system, req)
    if req["_item"] != report_id or req["_variant"] != form:
        bad.append("cache keys name another report or form")
    if bad:
        raise ContentMismatch(f"{system}/{form}/{report_id}: " + "; ".join(bad))
    return req


def body(req: dict) -> dict:
    """What is sent: the request without its cache keys."""
    return {k: v for k, v in req.items() if not k.startswith("_")}


def request_sha(system: str, form: str, report_id: str) -> str:
    """clients._sha of the full request (cache keys included): the RawStore file name in runs/arm3_llm_forms."""
    return _sha(build(system, form, report_id))


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=None)
def template_text(form: str) -> str:
    """The user message with the report replaced by {pathology_report}: one text per form (asserted over all 240)."""
    out = set()
    for rid in report_ids():
        u = build(SYSTEMS[0], form, rid)["messages"][1]["content"]
        R = report_text(rid)
        assert u.count(R) == 1, (form, rid)
        out.add(u.replace(R, PLACEHOLDER))
    assert len(out) == 1, f"{form}: {len(out)} different templates"
    return out.pop()


def response_format_for(form: str) -> dict:
    return build(SYSTEMS[0], form, report_ids()[0])["response_format"]


def identical_named_field(system: str, report_id: str) -> bool:
    """True if the named_field request equals the names request apart from the _variant cache key (then no call is
    made and the names answer is reused)."""
    a, b = build(system, "named_field", report_id), build(system, "names", report_id)
    strip = lambda r: {k: v for k, v in r.items() if k != "_variant"}     # noqa: E731
    return _dumps(strip(a)) == _dumps(strip(b))

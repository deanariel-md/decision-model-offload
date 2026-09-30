"""Prompt templates (prompts/*.txt): each template, filled with the values of one request, gives exactly the messages,
reply schema and Jev request body the code builds for that request. Synthetic records, messages, questions and reports
only; no data file is read, no request is sent.

Template format (prompts/README.md): the lines before the first "### " line describe the prompt and are not sent. Each
"### <part>" line starts a part, which runs to the next "### " line; blank lines at the end of a part are not part of it,
and a part whose name ends "(ends with a line break)" ends with one line break. {name} marks a value filled per record,
message, question or report. A name that occurs more than once ({option}) takes its values in order; a line (or, in a
JSON part, a member or list element) whose name has no value left is not sent. JSON parts are JSON documents with the
names inside strings: a string that is only a name takes the value as it is (text, or a JSON object)."""
from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from jevity import categorical as K                                                  # noqa: E402
from jevity import docuse_risk as DR                                                 # noqa: E402
from jevity import docuse_staging as D                                               # noqa: E402
from jevity import docuse_structure_only as S                                        # noqa: E402
from jevity import eicu as E                                                         # noqa: E402
from jevity.clients import FillClient, JevClient, LLMClient, PROMPTS, RawStore, active_families   # noqa: E402
from jevity.serialize import annotate, convert_units, to_json, to_prose, to_state   # noqa: E402
from jevity.simulate import make_synthetic_cohort                                    # noqa: E402

TEMPLATES = ROOT / "prompts"
HEAD = re.compile(r"^### (.+)$", re.M)
NAME = re.compile(r"\{([a-z][a-z ,]*[a-z])\}")
ENDS_WITH_BREAK = " (ends with a line break)"
# prompts the Supplementary Notes do not give; their tests skip when the template is not in the folder
NOT_IN_NOTES = {"death_before_discharge", "ten_year_risk_bundle", "ten_year_risk_structured",
                "colon_cancer_stage_structure_only", "colon_cancer_stage_named_field",
                "colon_cancer_stage_structured_no_examples"}
# prompts built by scripts/gapfill/run.py; their tests skip when that script is not in the folder
GAPFILL = {"death_before_discharge_structured"}
HAS_GAPFILL = (ROOT / "scripts" / "gapfill" / "run.py").exists()


# ------------------------------------------------------------------------------------------------ reading and filling
def parts(name: str) -> dict[str, str]:
    """The parts of prompts/<name>.txt: part name -> text."""
    f = TEMPLATES / f"{name}.txt"
    if name in NOT_IN_NOTES and not f.exists():
        pytest.skip(f"prompts/{name}.txt is not in this folder")
    if name in GAPFILL and not HAS_GAPFILL:
        pytest.skip("scripts/gapfill/run.py is not in this folder")
    text = f.read_text(encoding="utf-8")
    heads = list(HEAD.finditer(text))
    out = {}
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[m.end() + 1:end].rstrip("\n")
        part = m.group(1).strip()
        if part.endswith(ENDS_WITH_BREAK):
            part, body = part[:-len(ENDS_WITH_BREAK)], body + "\n"
        assert part not in out, f"{name}: part {part!r} twice"
        out[part] = body
    return out


class Values:
    """name -> value; a list is used in order, one element per occurrence of the name."""

    def __init__(self, values: dict):
        self.v = {k: (list(x) if isinstance(x, list) else x) for k, x in values.items()}
        self.used: set[str] = set()

    def left(self, name: str) -> bool:
        if name not in self.v:
            raise KeyError(f"no value for {{{name}}}")
        return not isinstance(self.v[name], list) or bool(self.v[name])

    def take(self, name: str):
        self.left(name)
        self.used.add(name)
        x = self.v[name]
        return x.pop(0) if isinstance(x, list) else x

    def check_all_used(self) -> None:
        """Every value of a name the part uses has been placed."""
        left = [k for k in self.used if isinstance(self.v[k], list) and self.v[k]]
        assert not left, f"more values than places for {left}"


def fill_text(template: str, values: dict) -> str:
    vals, lines = Values(values), []
    for line in template.split("\n"):
        names = NAME.findall(line)
        if not all(vals.left(n) for n in names):
            continue                                        # a line whose name has no value left is not sent
        lines.append(NAME.sub(lambda m: str(vals.take(m.group(1))), line))
    vals.check_all_used()
    return "\n".join(lines)


_GONE = object()


def fill_json(template: str, values: dict, absent_letters: tuple[str, ...] = ()):
    """The JSON part filled. absent_letters: option letters the question does not offer, left out of the reply
    schema's lists and objects (board-examination questions with four options)."""
    vals = Values(values)

    def walk(x):
        if isinstance(x, str):
            m = NAME.fullmatch(x)
            if m:
                return copy.deepcopy(vals.take(m.group(1))) if vals.left(m.group(1)) else _GONE
            return NAME.sub(lambda m: str(vals.take(m.group(1))), x)
        if isinstance(x, dict):
            out = {k: walk(v) for k, v in x.items() if k not in absent_letters}
            return {k: v for k, v in out.items() if v is not _GONE}
        if isinstance(x, list):
            out = [walk(v) for v in x if v not in absent_letters]
            return [v for v in out if v is not _GONE]
        return x

    out = walk(json.loads(template))
    vals.check_all_used()
    return out


def same_json(a, b) -> bool:
    """Equal values and the same key order."""
    return json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


def body(req: dict) -> dict:
    """What is posted: the request without its cache keys (the keys that start with '_')."""
    return {k: v for k, v in req.items() if not k.startswith("_")}


def names_in(name: str, part_names: tuple[str, ...]) -> set[str]:
    p = parts(name)
    return {n for k in part_names if k in p for n in NAME.findall(p[k])}


def check_chat(name: str, req: dict, values: dict, absent_letters: tuple[str, ...] = ()) -> None:
    """A chatbot request against the template: system message, user message and, when sent, the reply schema."""
    p = parts(name)
    assert set(values) <= names_in(name, ("SYSTEM MESSAGE", "USER MESSAGE", "REPLY SCHEMA")), "a value has no place"
    assert [m["role"] for m in req["messages"]] == ["system", "user"]
    assert req["messages"][0]["content"] == fill_text(p["SYSTEM MESSAGE"], values)
    assert req["messages"][1]["content"] == fill_text(p["USER MESSAGE"], values)
    if "REPLY SCHEMA" in p:
        assert same_json(req["response_format"], fill_json(p["REPLY SCHEMA"], values, absent_letters))
    else:
        assert "response_format" not in req


def check_jev(name: str, req: dict, values: dict) -> None:
    """A Jev request body against the template."""
    jv = {k: v for k, v in values.items() if k in names_in(name, ("JEV REQUEST BODY",))}
    assert same_json(body(req), fill_json(parts(name)["JEV REQUEST BODY"], jv))


# ------------------------------------------------------------------------------------------------ synthetic inputs
ODD = 'He said "stop" \\ then left.\nSecond line, tab\there; en dash – and µg; braces {x} [y].'


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return RawStore(tmp_path_factory.mktemp("store"))


@pytest.fixture(scope="module")
def records():
    """Survey records as the systems saw them: the baseline JSON, an annotated one, units converted, and prose."""
    co = make_synthetic_cohort(n=40, seed=11).to_dict("records")
    st = [to_state(r) for r in co[:4]]
    return [to_json(s) for s in st] + [to_json(annotate(st[0])), to_json(convert_units(st[1])), to_prose(st[2])]


def chatbots() -> list[str]:
    return active_families()


def texts(n: int, stem: str) -> list[str]:
    return [f"{stem} {i}: synthetic text, not from any source." for i in range(n - 1)] + [f"{stem}: {ODD}"]


# ------------------------------------------------------------------------------------------------ ten-year risk
@pytest.mark.parametrize("variant,name", [("raw", "ten_year_risk"), ("M1", "ten_year_risk_disregard_social")])
def test_ten_year_risk_chatbots(variant, name, store, records):
    stmt = PROMPTS["statements"]["nhanes"]
    for fam in chatbots():
        for rec in records:
            check_chat(name, LLMClient(fam, store, "nhanes", variant=variant).request(rec, stmt), {"record": rec})


@pytest.mark.parametrize("name,question,variant", [("ten_year_risk", "primary", "raw"),
                                                   ("ten_year_risk_disregard_social", "primary", "M1"),
                                                   ("ten_year_risk_complement", "complement", "raw"),
                                                   ("ten_year_risk_bands", "risk_bands", "raw"),
                                                   ("ten_year_risk_bundle", "bundle", "raw")])
def test_ten_year_risk_jev(name, question, variant, store, records):
    stmt = PROMPTS["statements"]["nhanes"]
    for rec in records:
        check_jev(name, JevClient(store, "nhanes", question=question, variant=variant).request(rec, stmt), {"record": rec})


def test_ten_year_risk_jev_structured(store):
    co = make_synthetic_cohort(n=40, seed=12).to_dict("records")
    for rec in co[:5]:
        state = DR.state_from_row(rec)
        check_jev("ten_year_risk_structured", DR.make_request(state), {"structured record": state})


def eicu_rows(n: int = 5, seed: int = 13) -> list[dict]:
    """Synthetic rows with the columns of config/eicu.yaml (values are random, not from any database)."""
    r = np.random.default_rng(seed)
    out = []
    for i in range(n):
        rec = {"age": float(r.integers(18, 95)), "sex": ["male", "female"][i % 2], "ethnicity": "Caucasian",
               "admission_dx": "Synthetic diagnosis", "admit_source": "Synthetic source", "unit_type": "Synthetic unit",
               "vent": ["yes", "no"][i % 2], "dialysis": "no", "diabetes": "yes" if i == 2 else None}
        for _, col, _, _ in E.CFG["numeric"]:
            if col != "age" and (i < 3 or col != "lactate_mmol_l"):
                rec[col] = float(r.normal(50, 20))
        out.append(rec)
    return out


def test_death_before_discharge(store):
    stmt = PROMPTS["statements"]["eicu_demo"]
    for rec in eicu_rows():
        text = E.to_json(E.to_state(rec))
        for fam in chatbots():
            check_chat("death_before_discharge", LLMClient(fam, store, "eicu_demo").request(text, stmt), {"record": text})
        check_jev("death_before_discharge", JevClient(store, "eicu_demo").request(text, stmt), {"record": text})


def masked(state: dict, field: str, dumps) -> str:
    """The record with one field's value replaced by "MISSING" (as scripts/exposure_diagnostic.py and
    scripts/eicu_exposure.py write it)."""
    st = dict(state)
    st[field] = {"value": "MISSING", "unit": st[field]["unit"]} if isinstance(st[field], dict) else "MISSING"
    return dumps(st)


def test_masked_field(store):
    co = make_synthetic_cohort(n=40, seed=14).to_dict("records")
    ei = eicu_rows(seed=16)
    cases = [(masked(to_state(co[i]), f, to_json), f, spec["unit"], "nhanes")
             for i, (f, spec) in enumerate(PROMPTS["exposure"]["fields"].items())]
    cases += [(masked(E.to_state(ei[i]), f, E.to_json), f, spec["unit"], "eicu_demo")
              for i, (f, spec) in enumerate(E.CFG["exposure"]["fields"].items())]
    for fam in chatbots():
        for text, field, unit, pop in cases:
            req = FillClient(fam, store, pop).request(text, field, None if pop == "nhanes" else unit)
            check_chat("masked_field", req, {"record": text, "field": field, "unit": unit})


# ------------------------------------------------------------------------------------------------ choice questions
def test_patient_message(store):
    arm = K.load_arm("arm2")
    for i, (msg, order) in enumerate(zip(texts(4, "Message"), ["ABCD", "BDAC", "CADB", "DCBA"])):
        opts = tuple(f"Option for {c}, message {i}" for c in order)
        it = K.Item(f"syn{i}", msg, "A", K.CHOICE_LABELS, tuple(order), opts)
        v = {"patient message": msg, "option": list(opts)}
        for fam in chatbots():
            check_chat("patient_message", K.CategoricalLLMClient(fam, store, arm).request(it), v)
        check_jev("patient_message", K.CategoricalJevClient(store, arm).request(it), v)


@pytest.mark.parametrize("n_options", [4, 5])
def test_board_examination(n_options, store):
    arm = K.load_arm("arm2_ext")
    labels = arm.labels[:n_options]
    for i, q in enumerate(texts(3, "Question")):
        opts = tuple(f"Answer {l} to question {i}" + (" – \"quoted\"" if l == "B" else "") for l in labels)
        it = K.Item(f"syn{i}", q, labels[0], labels, labels, opts, variant=["medhelm", "other_count"][i % 2])
        v = {"question": q, "option": list(opts)}
        absent = tuple(arm.labels[n_options:])
        for fam in chatbots():
            check_chat("board_examination", K.CategoricalLLMClient(fam, store, arm).request(it), v, absent)
        check_jev("board_examination", K.CategoricalJevClient(store, arm).request(it), v)


# ------------------------------------------------------------------------------------------------ ordered levels
def score_items(arm: K.ArmSpec, variant: str, states: list[str]) -> list[K.Item]:
    defs = arm.config["variants"][variant].get("definitions")
    tx = tuple((lv.get("definition") if defs else None) for lv in arm.config["levels"])
    return [K.Item(f"syn{i}", s, arm.labels[0], arm.labels, arm.labels, tx, variant=variant) for i, s in enumerate(states)]


def item_set(name: str, arm: K.ArmSpec) -> dict:
    """{item set} (the reply schema's name is "<item set>_answer") where the template leaves it open."""
    return {"item set": arm.name} if "item set" in names_in(name, ("REPLY SCHEMA",)) else {}


@pytest.mark.parametrize("name,arms,variant", [("colon_cancer_stage", ["arm3", "arm3_confuser"], "names"),
                                               ("colon_cancer_stage_definitions", ["arm3"], "definitions")])
def test_colon_cancer_stage(name, arms, variant, store):
    for a in arms:
        arm = K.load_arm(a)
        for it in score_items(arm, variant, texts(4, "Report")):
            v = {"report": it.state, **item_set(name, arm)}
            for fam in chatbots():
                check_chat(name, K.CategoricalLLMClient(fam, store, arm).request(it), v)
            check_jev(name, K.CategoricalJevClient(store, arm).request(it), v)


@pytest.fixture(scope="module")
def aki_records():
    """Records in the layout scripts/make_eicu_aki.py writes, from synthetic values, with the values that fill
    RECORD LAYOUT."""
    B = pytest.importorskip("make_eicu_aki")
    from jevity import kdigo as KD
    r = np.random.default_rng(15)
    out = []
    for i, (age, sex, wt, dial) in enumerate([("64", "Female", 71.25, None), ("> 89", "Male", float("nan"), 1500),
                                               ("", "", 102.0, 9000), ("40", "Unknown", 55.5, -600)]):
        vals = [(int(t), Decimal(str(round(float(r.uniform(0.4, 6.0)), 2)))) for t in r.integers(-10080, 4320, 3 + i)]
        text = B.record_text(age, sex, wt, vals, dial, B.CFG["kdigo"])
        shown = dial is not None and dial <= int(B.CFG["kdigo"]["record_to_min"])
        lay = {"age": B.age_text(age), "sex": sex.lower() if sex.strip() else "not recorded",
               "admission weight": "not recorded" if wt != wt else f"{wt:.1f} kg",
               "time": [KD.fmt_time(t) for t, _ in sorted(vals, key=lambda x: x[0])],
               "value": [KD.fmt_value(v) for _, v in sorted(vals, key=lambda x: x[0])],
               "dialysis": KD.fmt_time(dial) if shown else "no"}
        out.append((text, lay))
    return out


def test_kidney_injury_record_layout(aki_records):
    p = parts("kidney_injury_stage")
    for text, lay in aki_records:
        line = [fill_text(p["CREATININE LINE"], {"time": t, "value": v}) for t, v in zip(lay["time"], lay["value"])]
        v = {k: lay[k] for k in ("age", "sex", "admission weight", "dialysis")}
        assert text == fill_text(p["RECORD LAYOUT"], {**v, "creatinine values, one per line": "\n".join(line)})


@pytest.mark.parametrize("name,arms,variant", [("kidney_injury_stage", ["eicu_aki", "eicu_aki_side"], "names"),
                                               ("kidney_injury_stage_definitions", ["eicu_aki", "eicu_aki_side"],
                                                "definitions")])
def test_kidney_injury_stage(name, arms, variant, store, aki_records):
    for a in arms:
        arm = K.load_arm(a)
        for it in score_items(arm, variant, [t for t, _ in aki_records] + [ODD]):
            v = {"record": it.state, **item_set(name, arm)}
            for fam in chatbots():
                check_chat(name, K.CategoricalLLMClient(fam, store, arm).request(it), v)
            check_jev(name, K.CategoricalJevClient(store, arm).request(it), v)


# ------------------------------------------------------------------------------------------------ structured input
@pytest.mark.parametrize("name,variant", [("colon_cancer_stage_structured", "documented"),
                                          ("colon_cancer_stage_structured_notes", "documented_notes")])
def test_colon_cancer_stage_structured(name, variant, store):
    cl = D.DocuseJevClient(store, D.arm_spec())
    for i, rep in enumerate(texts(4, "Report")):
        it = K.Item(f"syn{i}", "  " + rep + "\n", D.LEVELS[0], D.LEVELS, D.LEVELS, (None,) * len(D.LEVELS),
                    variant=variant)
        check_jev(name, cl.request(it), {"report": it.state.strip()})


# ------------------------------------------------------------------------------------------------ recommended form, outcome
@pytest.fixture(scope="module")
def gapfill():
    """scripts/gapfill/run.py, whose make_client builds these requests (the calls of `run.py full`)."""
    if not HAS_GAPFILL:
        pytest.skip("scripts/gapfill/run.py is not in this folder")
    spec = importlib.util.spec_from_file_location("gapfill_run", ROOT / "scripts" / "gapfill" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_death_before_discharge_structured(gapfill, store):
    stmt = gapfill.G.statement(gapfill.population("eicu_structured"))
    for rec in eicu_rows(seed=17):
        state = gapfill.G.categorised_state(rec)
        check_jev("death_before_discharge_structured", gapfill.make_client("jev_eicu", store).request(state, stmt),
                  {"structured record": state})


def test_colon_cancer_stage_structure_only(store):
    cl = S.StructureOnlyJevClient(store, D.arm_spec())
    for i, rep in enumerate(texts(4, "Report")):
        it = K.Item(f"syn{i}", "  " + rep + "\n", D.LEVELS[0], D.LEVELS, D.LEVELS, (None,) * len(D.LEVELS),
                    variant=S.VARIANT)
        check_jev("colon_cancer_stage_structure_only", cl.request(it), {"report": it.state.strip()})


# ------------------------------------------------------------------------------------------------ every form, 240 reports
FORM_TEMPLATES = {"names": "colon_cancer_stage", "named_field": "colon_cancer_stage",
                  "definitions": "colon_cancer_stage_definitions", "structure_only": "colon_cancer_stage_structure_only",
                  "documented": "colon_cancer_stage_structured",
                  "documented_no_examples": "colon_cancer_stage_structured_no_examples",
                  "documented_notes": "colon_cancer_stage_structured_notes"}
JEV_TEMPLATES = {**FORM_TEMPLATES, "named_field": "colon_cancer_stage_named_field"}


@pytest.fixture(scope="module")
def forms_tree(tmp_path_factory):
    """Jev's stored requests for every form on synthetic reports (tests/synthetic_llm_forms.py, written with this
    repository's Jev request builders), with jevity.llm_forms reading them."""
    import pandas as pd
    SL = pytest.importorskip("synthetic_llm_forms")
    from jevity import llm_forms as LF
    root = tmp_path_factory.mktemp("forms")
    frames = SL.synthetic_frames()
    pool = pd.concat([pd.DataFrame({"report_id": frames[s].report_id, "set": s}) for s in LF.SETS], ignore_index=True)
    SL.write_tree(root, frames, pool)
    with pytest.MonkeyPatch.context() as mp:
        SL.use(mp, root, pool)
        yield LF
    SL.clear_caches()


@pytest.mark.parametrize("form", list(FORM_TEMPLATES))
def test_colon_cancer_stage_every_form(form, forms_tree):
    """Every chatbot's request in each form of the comparison on the 240 reports (src/jevity/llm_forms.py), and Jev's
    stored request it is built from, against the form's template."""
    LF = forms_tree
    assert set(FORM_TEMPLATES) == set(LF.FORMS)
    name = FORM_TEMPLATES[form]
    item = {"item set": "arm3"} if "item set" in names_in(name, ("REPLY SCHEMA",)) else {}
    for rid in LF.report_ids():
        R = LF.report_text(rid).strip()
        for fam in LF.SYSTEMS:
            req = LF.build(fam, form, rid)
            check_chat(name, req, {"report": R, **item})
            if form == "named_field":
                assert body(req) == body(LF.build(fam, "names", rid))
        check_jev(JEV_TEMPLATES[form], LF.form_index(form)[rid].body, {"report": R})


# ------------------------------------------------------------------------------------------------ the folder
def test_every_template_is_checked_and_every_name_is_known():
    checked = {"ten_year_risk", "ten_year_risk_disregard_social", "ten_year_risk_complement",
               "ten_year_risk_bands", "ten_year_risk_bundle", "ten_year_risk_structured", "death_before_discharge",
               "masked_field", "patient_message", "board_examination", "colon_cancer_stage",
               "colon_cancer_stage_definitions", "kidney_injury_stage", "kidney_injury_stage_definitions",
               "colon_cancer_stage_structured", "colon_cancer_stage_structured_notes",
               "colon_cancer_stage_structure_only", "colon_cancer_stage_named_field",
               "colon_cancer_stage_structured_no_examples"} | GAPFILL
    present = {p.stem for p in TEMPLATES.glob("*.txt")}
    optional = NOT_IN_NOTES | (set() if HAS_GAPFILL else GAPFILL)
    assert present <= checked and checked - optional <= present, sorted(present ^ checked)
    known = {"record", "structured record", "field", "unit", "patient message", "option", "question", "report",
             "item set", "age", "sex", "admission weight", "creatinine values, one per line", "dialysis", "time",
             "value"}
    for name in sorted(present):
        for part, text in parts(name).items():
            assert set(NAME.findall(text)) <= known, (name, part)
            if part in ("REPLY SCHEMA", "JEV REQUEST BODY"):
                json.loads(text)

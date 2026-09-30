"""Arm 3 parity of the Jev and chatbot requests. For one report in each question version, built by the shared clients
(src/jevity/categorical.py): the report, the question sentence and the list of answers (with definitions in that
version) are character-identical in Jev's request and in every chatbot's request. Allowed differences: the field
labels, the trailing line break and the reply-format instruction. The field labels are the user template's three
headings, which label the parts Jev receives as separate API fields (state, instructions, criteria); the reply-format
instruction is the system message and, where the endpoint takes one, the JSON schema. On synthetic reports
(tests/synthetic_items.py). No network."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical as K
from jevity import clients as C
from jevity import crc
from jevity.batch import body_of
import pandas as pd
import synthetic_items as SI

INSTR = ("Assign the pathological stage group for this colorectal cancer according to the AJCC Cancer Staging Manual, "
         "8th edition. Stage IVA-IVB here includes IVA and IVB.")
REPLY_FORMAT = ("Choose the stage group that is correct, and give the probability that each stage group is correct, as "
                "numbers between 0 and 1 that add up to 1. Always choose one stage group. Reply with a JSON object with "
                "two keys, \"stage\" (the stage group you choose, written exactly as in the list) and \"probabilities\" "
                "(an object with one number for each stage group), and nothing else.")
# the allowed differences in the chatbot's user message
FIELD_LABELS = ("Pathology report:\n", "\n\nQuestion: ", "\n\nStage groups:\n")
TRAILING_BREAK = "\n"
TEMPLATE = (FIELD_LABELS[0] + "{state}" + FIELD_LABELS[1] + "{instructions}" + FIELD_LABELS[2] + "{options}"
            + TRAILING_BREAK)
JEV_KEYS = {"model", "state", "questions", "provider"}
CHATBOT_SETTINGS = {"model", "max_tokens", "provider", "seed", "reasoning", "temperature"}   # not shown to the model


def wire(req: dict) -> dict:
    """The body sent: cache-only keys (leading underscore) dropped, as in the clients' _send."""
    return {k: v for k, v in req.items() if not k.startswith("_")}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(C, "_post", boom)
    monkeypatch.setattr(C.httpx, "Client", boom)


@pytest.fixture(scope="module")
def arm():
    return SI.arm("arm3")


@pytest.fixture(scope="module")
def report(arm):
    """The first row of the items file, one item per question version."""
    items = K.load_items(arm)
    rid = items[0].item_id
    return {it.variant: it for it in items if it.item_id == rid}


@pytest.fixture(scope="module")
def built(arm, report, tmp_path_factory):
    """variant -> (Jev body, {chatbot: (request, body)}, {chatbot: client}), every chatbot of the arm."""
    store = C.RawStore(tmp_path_factory.mktemp("raw"))              # request() only builds; nothing is stored
    jev = K.CategoricalJevClient(store, arm)
    bots = {s: K.CategoricalLLMClient(s, store, arm) for s in K.arm_systems(arm)[1:]}
    out = {}
    for v, it in report.items():
        reqs = {s: b.request(it) for s, b in bots.items()}
        out[v] = (wire(jev.request(it)), {s: (r, wire(r)) for s, r in reqs.items()}, bots)
    return out


def test_one_report_in_both_versions(arm, report):
    assert set(report) == {"names", "definitions"} == set(arm.config["variants"])
    rid = {it.item_id for it in report.values()}
    assert rid == {"crc001"}                                          # the first row of the item file
    assert arm.user_template == TEMPLATE and arm.instructions == INSTR


@pytest.mark.parametrize("variant", ["names", "definitions"])
def test_jev_and_every_chatbot_get_the_same_report_question_and_answers(arm, report, built, variant):
    jev, bots, clients = built[variant]
    assert set(jev) == JEV_KEYS and set(jev["questions"]) == {arm.jev_question_id}
    q = jev["questions"][arm.jev_question_id]
    assert set(q) == {"type", "instructions", "criteria"} and q["type"] == "score"
    state, instr, answers = jev["state"], q["instructions"], q["criteria"]

    # Jev's three parts are the sources, character for character
    rows = pd.read_csv(SI.items_path("arm3"), dtype=str, keep_default_na=False).set_index("report_id")
    assert state == rows.loc[report[variant].item_id, "report"].strip()
    assert instr == INSTR
    if variant == "names":
        assert answers == crc.LEVELS
    else:
        assert answers == [f"{lv['name']}: {lv['definition']}" for lv in arm.config["levels"]]
        assert answers == [f"{lv}: {crc.level_definition(lv)}" for lv in crc.LEVELS]

    options = "\n".join(answers)
    user = FIELD_LABELS[0] + state + FIELD_LABELS[1] + instr + FIELD_LABELS[2] + options + TRAILING_BREAK
    assert len(bots) >= 2
    for s, (req, body) in bots.items():
        # the chatbot's user message is Jev's three parts, each character for character, plus the field labels and the
        # trailing line break
        assert [m["role"] for m in body["messages"]] == ["system", "user"], s
        u = body["messages"][1]["content"]
        assert u == user, s
        assert u.count(state) == 1 and u.count(instr) == 1 and u.count(options) == 1
        # the other allowed difference is the reply-format instruction; the rest are request settings
        assert body["messages"][0]["content"] == REPLY_FORMAT == K.system_text(arm), s
        rf = body.get("response_format")
        if clients[s].spec.get("structured_outputs"):
            assert rf == {"type": "json_schema", "json_schema": K.reply_schema(arm, arm.labels)}, s
            assert rf["json_schema"]["schema"]["properties"]["stage"]["enum"] == crc.LEVELS
        else:
            assert rf is None, s
        assert set(body) - {"messages", "response_format"} <= CHATBOT_SETTINGS, s
        if clients[s].batch is not None:                              # the batch route sends the same messages
            assert body_of(req)["messages"] == body["messages"], s


def test_the_versions_differ_only_in_the_definitions(built):
    (jn, bn, _), (jd, bd, _) = built["names"], built["definitions"]
    qn, qd = jn["questions"]["crc_stage_group"], jd["questions"]["crc_stage_group"]
    assert jn["state"] == jd["state"] and qn["instructions"] == qd["instructions"]
    assert [c.split(": ", 1)[0] for c in qd["criteria"]] == qn["criteria"]
    assert {k: v for k, v in jn.items() if k != "questions"} == {k: v for k, v in jd.items() if k != "questions"}
    for s in bn:
        assert bn[s][1]["messages"][0] == bd[s][1]["messages"][0]
        assert {k: v for k, v in bn[s][1].items() if k != "messages"} == {k: v for k, v in bd[s][1].items()
                                                                           if k != "messages"}

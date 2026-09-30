"""Request parity: Jev and the chatbots see the same message, the same question sentence and the same list of answers
(with definitions, in a version that has them), character for character. Allowed differences in arm 2, and nothing
else: the "Message:" and "Question:" labels, the option letters, the trailing line break, and the reply-format
instruction (system message, and the JSON schema that mirrors it where the endpoint takes one). Checked on every arm 2
item (synthetic, tests/synthetic_items.py) for every chatbot, and on both versions of the synthetic ten-level Score arm
(the arm 3 path, under that arm's own template)."""
import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from jevity import categorical as K
from jevity.clients import RawStore
import synthetic_items as SI

ARM = SI.arm("arm2")
ITEM_ID = "rec_01_ac_01"
# The whole of arm 2's allowed framing around the three shared parts: the two labels with the line breaks that set them
# apart, and the trailing line break. Option letters: "A. " before each option text.
ALLOWED_FRAMING = {"arm2": ("Message:\n", "\n\nQuestion: ", "\n\n", "\n")}


def expected_user(arm: K.ArmSpec, state: str, instructions: str, answers: list[str]) -> str:
    """The chatbot's user message rebuilt from Jev's parts and the allowed framing alone (not from the arm's
    template, so a change to the template fails here)."""
    if arm.name in ALLOWED_FRAMING:
        a, b, c, d = ALLOWED_FRAMING[arm.name]
        return f"{a}{state}{b}{instructions}{c}" + "\n".join(answers) + d
    return arm.user_template.format(state=state, instructions=instructions, options="\n".join(answers))


def answer_list(arm: K.ArmSpec, jev: dict) -> list[str]:
    """The answers as Jev receives them, written as the chatbot sees them (Choice: 'letter. text' from Jev's criteria
    keyed by letter; Score: Jev's level strings as they are)."""
    crit = jev["questions"][arm.jev_question_id]["criteria"]
    return [f"{l}. {t}" for l, t in crit.items()] if arm.kind == "choice" else list(crit)


def assert_parity(arm: K.ArmSpec, item: K.Item, chat: dict, jev: dict) -> None:
    q = jev["questions"][arm.jev_question_id]
    # Jev's three parts are the item's own
    assert jev["state"] == item.state and q["instructions"] == arm.instructions
    if arm.kind == "choice":
        assert list(q["criteria"]) == list(item.presented) and list(q["criteria"].values()) == list(item.texts)
    else:
        assert q["criteria"] == [K._level_text(arm, l, t) for l, t in zip(item.presented, item.texts)]
    # the chatbot's user message is exactly those parts with the allowed framing, and nothing else
    answers = answer_list(arm, jev)
    user = chat["messages"][1]
    assert user["role"] == "user" and user["content"] == expected_user(arm, jev["state"], q["instructions"], answers)
    for part in (jev["state"], q["instructions"], *answers):
        assert part in user["content"]
    # the only chatbot-only text: the reply-format instruction (and the schema that mirrors it)
    assert len(chat["messages"]) == 2 and chat["messages"][0] == {"role": "system", "content": K.system_text(arm)}
    assert K.system_text(arm) not in user["content"] and K.system_text(arm) not in jev["state"]
    rf = chat.get("response_format")
    assert rf is None or rf == {"type": "json_schema", "json_schema": K.reply_schema(arm, item.presented)}
    assert set(jev) <= {"model", "state", "questions", "provider", "_repeat", "_arm", "_item", "_variant"}


def requests(arm: K.ArmSpec, item: K.Item, system: str, store: RawStore) -> tuple[dict, dict]:
    return (K.CategoricalLLMClient(system, store, arm).request(item), K.CategoricalJevClient(store, arm).request(item))


def test_arm2_parity_every_item_every_chatbot(tmp_path):
    store = RawStore(tmp_path)
    items = K.load_items(ARM)
    assert {i.variant for i in items} == {"main"}                           # arm 2 has one question version
    chatbots = [s for s in K.arm_systems(ARM) if s != "jev"]
    for it in items:
        jev = K.CategoricalJevClient(store, ARM).request(it)
        msgs = None
        for s in chatbots:
            chat = K.CategoricalLLMClient(s, store, ARM).request(it)
            assert_parity(ARM, it, chat, jev)
            msgs = msgs or chat["messages"]
            assert chat["messages"] == msgs                                  # every chatbot gets the same messages


def test_arm2_framing_is_only_the_allowed_labels(tmp_path):
    a, b, c, d = ALLOWED_FRAMING["arm2"]
    assert ARM.user_template == f"{a}{{state}}{b}{{instructions}}{c}{{options}}{d}"
    store = RawStore(tmp_path)
    it = next(i for i in K.load_items(ARM) if i.item_id == ITEM_ID)
    for extra in ("Message:\n{state}\n\nQuestion: {instructions}\n\n{options}\nAnswer carefully.\n",   # added text
                  "Patient message:\n{state}\n\nQuestion: {instructions}\n\n{options}\n",            # other label
                  "Message:\n{state}\n\nQuestion: {instructions}\n\n{options}"):                     # no final break
        arm = dataclasses.replace(ARM, user_template=extra)
        chat, jev = requests(arm, it, "glm", store)
        with pytest.raises(AssertionError):
            assert_parity(arm, it, chat, jev)


def test_score_arm_parity_both_versions(tmp_path):
    from test_arm2_dry_run import score_arm
    arm = score_arm(tmp_path)
    store = RawStore(tmp_path / "runs")
    items = K.load_items(arm)
    by_version = {}
    for it in items:
        by_version.setdefault(it.variant, it)
    assert set(by_version) == {"names", "definitions"}
    for v, it in by_version.items():
        chat, jev = requests(arm, it, "gpt", store)
        assert_parity(arm, it, chat, jev)
        crit = jev["questions"][arm.jev_question_id]["criteria"]
        assert len(crit) == 10 and all(("criteria for" in c) == (v == "definitions") for c in crit)

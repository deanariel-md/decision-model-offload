"""The arm 3 confuser run (config/arm3_confuser.yaml): arm 3's question and prompts word for word, names only, Jev and
the five primary chatbots, no repeats. No network."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _cfg(name):
    return yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text(encoding="utf-8"))


def test_the_question_is_arm_3s_word_for_word():
    a, c = _cfg("arm3"), _cfg("arm3_confuser")
    for k in ("kind", "population", "seed", "instructions", "answer_key", "answer_aliases", "levels", "prompts", "jev",
              "keys", "route"):
        assert c[k] == a[k], k
    assert c["variants"] == {"names": {"definitions": False}} and c["primary_variant"] == "names"
    assert c["systems"] == {"tiers": ["primary"]} and c["repeats"] == {"n_items": 0, "n_repeats": 1}
    assert c["items"]["main"] == "data/arm3/confuser_items.csv" and c["items"]["stratum_column"] == "confuser_type"
    assert c["arm"] == "arm3_confuser" and c["timing"]["hold"]
    # the same analysis settings, except the stratum and the hybrid test, which is the main run's
    assert "hybrid_share" not in c["analysis"] and a["analysis"]["hybrid_share"] == {"jev_share": 0.5}
    assert {k: v for k, v in c["analysis"].items() if k != "strata_column"} == \
           {k: v for k, v in a["analysis"].items() if k not in ("strata_column", "hybrid_share")}


def test_shared_code_plans_600_calls():
    from jevity import categorical as K
    import run_categorical as RC
    import synthetic_items as SI
    arm = SI.arm("arm3_confuser")                                   # the config with synthetic reports
    items, sys_, rep, calls = RC.design(arm)
    assert sys_ == ["jev", "gpt", "claude", "gemini", "muse", "glm"] and rep == []
    assert len(items) == 100 and {i.variant for i in items} == {"names"} and len(calls) == 600
    assert sorted({i.stratum for i in items}) == ["1", "2", "3", "4", "5"]
    a3 = K.load_arm("arm3")
    it = items[0]
    assert K.user_text(arm, it) == K.user_text(a3, it) and K.system_text(arm) == K.system_text(a3)

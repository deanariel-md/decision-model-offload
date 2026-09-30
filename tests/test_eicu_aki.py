"""The eICU AKI set (config/eicu_aki.yaml, scripts/make_eicu_aki.py): arm 3's skeleton with the AKI question, the
records as written, the key recomputed from exactly the text each record shows, the sample counts and the plan of the
shared code. The item files under data/eicu_aki are built from the eICU-CRD demo by scripts/make_eicu_aki.py; the tests
that read them skip until then. No network, no model call."""
import re
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from jevity import categorical as K
from jevity import kdigo as KD

CFG = yaml.safe_load((ROOT / "config" / "eicu_aki.yaml").read_text(encoding="utf-8"))
RULE = KD.Rule.from_config(CFG["kdigo"])
DEMO = ROOT / "data" / "eicu-collaborative-research-database-demo-2.0.1"
LINE = re.compile(r"^([+-])(\d+) h (\d{2}) min: (\d+\.\d{2})$")
HEAD = ["Age: ", "Sex: ", "Admission weight: ",
        "Serum creatinine (mg/dL), from 7 days before to 72 hours after intensive care unit admission; times are from "
        "unit admission:"]


def _items(which="main"):
    path = {"side": CFG["records"]["side_items"], **{f"unknown_{p}": f for p, f in CFG["records"]["unknown_items"].items()}
            }.get(which) or CFG["items"][which]
    return pd.read_csv(ROOT / path, dtype=str, keep_default_na=False)


def _need_items():
    files = [CFG["items"]["main"], CFG["records"]["side_items"], *CFG["records"]["unknown_items"].values()]
    if not all((ROOT / f).exists() for f in files):
        pytest.skip("data/eicu_aki is built from the eICU-CRD demo by scripts/make_eicu_aki.py")


def parse(record: str):
    """(values, dialysis minute) from the record text alone."""
    lines = record.split("\n")
    assert [l[:len(h)] for l, h in zip(lines, HEAD)] == HEAD and lines[-1].startswith("Dialysis started: ")
    values = []
    for l in lines[len(HEAD):-1]:
        m = LINE.match(l)
        assert m, l
        t = (int(m[2]) * 60 + int(m[3])) * (-1 if m[1] == "-" else 1)
        values.append((t, Decimal(m[4])))
    d = lines[-1].removeprefix("Dialysis started: ")
    if d == "no":
        return values, None
    m = re.fullmatch(r"([+-])(\d+) h (\d{2}) min", d)
    return values, (int(m[2]) * 60 + int(m[3])) * (-1 if m[1] == "-" else 1)


def test_the_question_is_arm_3s_skeleton_with_the_aki_variables():
    arm = K.load_arm("eicu_aki")
    assert arm.kind == "score" and arm.population == "eicu_demo" and arm.labels == KD.LEVELS
    assert arm.instructions == ("By the KDIGO creatinine criteria, what was this patient's highest AKI stage during the "
                                "first 72 hours in the intensive care unit?")
    a3 = K.load_arm("arm3")
    swap = lambda s: s.replace("stage group", "AKI stage")
    assert K.system_text(arm) == swap(K.system_text(a3))
    assert arm.user_template == a3.user_template.replace("Pathology report:", "Patient:").replace(
        "Stage groups:", "AKI stages:")
    assert CFG["variants"] == {"names": {"definitions": False}, "definitions": {"definitions": True}}
    assert CFG["timing"]["hold"] and CFG["route"]["workers"]["medgemma"] + CFG["route"]["workers"]["gemma"] <= 5


def test_every_record_shows_only_the_timeline_and_the_key_follows_from_its_text():
    _need_items()
    for which in ("main", "side"):
        df = _items(which)
        assert df.item_id.is_unique
        for r in df.itertuples():
            values, dialysis = parse(r.record)
            assert len(values) == int(r.n_values) >= 2 and len({t for t, _ in values}) == len(values)
            assert all(-7 * 24 * 60 <= t <= 72 * 60 for t, _ in values) and [t for t, _ in values] == sorted(t for t, _ in values)
            assert str(r.stay_id) not in r.record                       # no identifier in what a model sees
            f = KD.highest_stage(values, dialysis, RULE)
            assert (KD.LEVELS[f.stage], f.criterion) == (r.stage, r.criterion), r.item_id
            if which != "side":                                         # the sensitivity key, from the text too
                alt = KD.LEVELS[KD.highest_stage_either_side(values, dialysis, RULE)]
                assert (alt, str(alt != r.stage).lower()) == (r.stage_either_side, r.baseline_sensitive)


def test_sample_counts_and_disjoint_sets():
    """One stay per patient (rarest stage first); patients on maintenance dialysis and stays with no value in the first
    72 h in the side set."""
    _need_items()
    m, s = _items(), _items("side")
    assert m.stage.value_counts().to_dict() == {"no AKI": 250, "stage 1": 228, "stage 3": 50, "stage 2": 33}
    assert (m.n_values_icu != "0").all()
    assert s.subset.value_counts().to_dict() == {"maintenance_dialysis": 52, "no_value_in_icu": 41}
    assert (s[s.subset == "no_value_in_icu"].n_values_icu == "0").all()
    ids = pd.concat([m.stay_id, s.stay_id])
    assert ids.is_unique
    assert int((m.baseline_sensitive == "true").sum()) == 31
    assert list(m.item_id) == [f"aki{i:04d}" for i in range(1, len(m) + 1)]


def test_shared_code_plans_both_versions_on_every_record():
    import run_categorical as RC
    arm = K.load_arm("eicu_aki")
    _need_items()
    items, sys_, rep, calls, *_ = RC.design(arm)
    assert sys_ == K.arm_systems(arm) and sys_[0] == "jev" and len(sys_) == 11
    per = pd.Series([c.system for c in calls]).value_counts()
    assert (per == 561 * 2 + 40 * 2 * 2).all() and len(rep) == 40
    it = next(i for i in items if i.variant == "definitions")
    assert K.jev_question(arm, it)["criteria"][2] == ("stage 2: a value 2.0 times or more but less than 3.0 times the lowest value in the "
                                                          "preceding 7 days")


def test_confident_half_and_confusion_on_a_hand_built_table():
    import numpy as np
    from types import SimpleNamespace
    import eicu_aki_supplement as E
    # 6 records; Jev's top probabilities rank r1..r3 as its confident half; correct on r1, r2, r4, r5
    items = pd.DataFrame({"item_id": [f"r{i}" for i in range(1, 7)]})
    t = SimpleNamespace(items=items, systems=["jev"], labels=KD.LEVELS, col=lambda s: 0,
                        usable=np.ones((6, 1), bool), top=np.array([[.9], [.8], [.7], [.4], [.3], [.2]]),
                        correct=np.array([[1], [1], [0], [1], [1], [0]], bool),
                        truth=np.array([0, 1, 2, 3, 0, 1]), answer=np.array([[0], [1], [3], [3], [0], [-1]]))
    c = E.confident_half(t, 0.5, np.tile(np.arange(6), (5, 1)), 0.95)
    assert (c["n_confident"], c["correct"], c["wrong"]) == (3, 4, 2)
    assert c["kept"] == 0.5 and c["let_through"] == 0.5                   # 2 of 4 correct kept, 1 of 2 wrong let through
    assert c["accuracy_confident"] == pytest.approx(2 / 3) and c["accuracy_other"] == pytest.approx(2 / 3)
    x = E.confusion(t, "jev")
    assert x["counts"]["stage 2"]["stage 3"] == 1 and x["counts"]["stage 1"]["unusable"] == 1
    assert sum(sum(r.values()) for r in x["counts"].values()) == 6


def test_the_side_set_asks_the_main_question_word_for_word():
    import run_categorical as RC
    side = yaml.safe_load((ROOT / "config" / "eicu_aki_side.yaml").read_text(encoding="utf-8"))
    for k in ("kind", "population", "seed", "instructions", "answer_key", "levels", "variants", "primary_variant",
              "prompts", "jev", "keys", "route", "systems"):
        assert side[k] == CFG[k], k
    assert {k: v for k, v in side["analysis"].items() if k != "strata_column"} == \
           {k: v for k, v in CFG["analysis"].items() if k not in ("strata_column", "hybrid_share")}
    assert side["items"]["main"] == CFG["records"]["side_items"] and side["items"]["stratum_column"] == "subset"
    _need_items()
    items, sys_, rep, calls, *_ = RC.design(K.load_arm("eicu_aki_side"))
    assert rep == [] and len(calls) == 93 * 2 * 11


@pytest.mark.skipif(not DEMO.exists(), reason="needs the unzipped eICU-CRD demo under data/")
def test_the_items_rebuild_from_the_demo_one_stay_per_patient():
    import make_eicu_aki as B
    _need_items()
    B.check_demo(DEMO)
    df, _ = B.build(DEMO)
    m, t, s = B.sample(df)
    assert _items().equals(m.astype(str))
    assert not set(t.stay_id) & set(pd.concat([m.stay_id, s.stay_id]))   # no scored stay among those set aside
    assert _items("side").equals(s.astype(str))
    uf, ul = B.unknown_halves(m, s)
    assert _items("unknown_first").equals(uf.astype(str)) and _items("unknown_last").equals(ul.astype(str))
    pat = df.set_index("stay_id").patient
    every = pd.concat([m.stay_id, t.stay_id, s.stay_id]).map(pat)
    assert every.is_unique                                              # one stay per patient across every set
    chronic = set(df.loc[df.chronic_dialysis, "patient"])
    assert not set(pd.concat([m.stay_id, t.stay_id]).map(pat)) & chronic

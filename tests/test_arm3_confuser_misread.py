"""The confuser misread counts (scripts/arm3_confuser_misread.py) on a small hand-built case. No network."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import arm3_confuser_misread as M


def _case():
    reports = pd.DataFrame({"report_id": ["a", "b", "c", "d"], "level": ["IIA", "IIA", "IIIB", "I"],
                            "confuser_type": [1, 2, 3, 5], "confuser_feature": ["itc", "adhesion", "perforation", "emvi"],
                            "misread_level": ["IIIA", "IIC", "IIIC", None]})
    for t in (1, 2, 3, 4):                                   # every type 1-4 needs a report
        if t not in set(reports.confuser_type):
            reports.loc[len(reports)] = ["e", "IIIB", t, "nonregional_node", "IIIB"]
    conf = pd.DataFrame({"system": ["x"] * 4, "item_id": ["a", "b", "c", "d"], "valid": [True, True, False, True],
                         "answer": ["IIIA", "IIA", None, "IIA"]})
    main = pd.DataFrame({"system": ["x"] * 4, "item_id": ["m1", "m2", "m3", "t6"], "valid": [True] * 4,
                         "answer": ["IIIA", "IIA", "IIIC", "IIIC"], "truth": ["IIA", "IIA", "IIIB", "IIIB"]})
    type6 = pd.DataFrame({"report_id": ["t6"], "level": ["IIIB"], "misread_level": ["IIIC"],
                          "misread_changes_level": ["True"]})
    return conf, reports, main, type6


def test_counts_and_control():
    conf, reports, main, type6 = _case()
    r = M.analyse(conf, reports, main, type6, ["x"], ["x"])
    t1 = r["types_1_4"]["1"]["systems"]["x"]
    assert (t1["n"], t1["correct"], t1["misread"], t1["other"], t1["unusable"]) == (1, 0, 1, 0, 0)
    assert t1["control_same_level_main"] == 0.5              # main run, true IIA: 1 of 2 answered IIIA
    t2 = r["types_1_4"]["2"]["systems"]["x"]
    assert (t2["correct"], t2["misread"]) == (1, 0) and t2["control_same_level_main"] == 0.0
    t3 = r["types_1_4"]["3"]["systems"]["x"]
    assert t3["unusable"] == 1 and t3["control_same_level_main"] == 1.0
    assert r["type_5"]["emvi"]["x"] == {"n": 1, "correct": 0, "misread": 0, "other": 1, "unusable": 0}
    assert r["type_6"]["changes_level"]["x"]["misread"] == 1 and r["type_6"]["level_unchanged"]["x"]["n"] == 0
    md = M.tables(r, ["x"], ["x"])
    assert "| x | 0 of 1 | 1 of 1 | 0 of 1 | 0 of 1 | 50% |" in md

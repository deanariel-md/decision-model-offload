"""Characteristics table (scripts/characteristics.py) on the synthetic cohort: counts add up per set, missing values are
their own row, and an unexpected level stops the script. No network."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import synthetic  # noqa: E402

spec = importlib.util.spec_from_file_location("characteristics", ROOT / "scripts" / "characteristics.py")
CH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CH)


def _cohort():
    c = synthetic.make(n=800, seed=3)
    c.loc[c.index[:150], "split"] = "fit"
    c.loc[c.index[5], "education"] = None
    return c


def test_counts_add_up_and_missing_is_a_row():
    c = _cohort()
    s = CH.summarise(c)
    assert s["sets"]["fit"]["n"] == 150
    for r in s["rows"]:
        for k, v in r["sets"].items():
            n = s["sets"][k]["n"]
            if r["kind"] == "cat":
                assert sum(x["n"] for x in v["levels"].values()) == n
            elif r["kind"] == "num":
                assert v["q1"] <= v["median"] <= v["q3"]
    edu = next(r for r in s["rows"] if r["column"] == "education")
    assert edu["sets"]["fit"]["levels"]["missing"]["n"] == 1
    md = CH.markdown(s)
    assert "Fitting set (n = 150)" in md and "| &nbsp;&nbsp;missing | 1 (0.7%)" in md


def test_unexpected_level_stops():
    c = _cohort()
    c.loc[c.index[0], "smoking"] = "sometimes"
    with pytest.raises(SystemExit):
        CH.summarise(c)

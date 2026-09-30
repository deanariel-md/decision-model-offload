"""Cost, speed and value (scripts/cost_speed.py, scripts/timing_sample.py): dominance, cost per usable estimate at the
billed and list price, and the timing sample's record choice and summary. No network."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CS = _load("cost_speed")
TS = _load("timing_sample")

CFG = {"jev": {"price_per_mtok_input": 0.042},
       "families": {"gpt": {"price_in": 2.0, "price_out": 10.0}, "glm": {"price_in": 1.4, "price_out": 4.4}},
       "batch": {"families": {"gpt": {"price_in": 1.0, "price_out": 5.0}}}}


def test_dominated():
    d = CS.dominated({"a": (1.0, 0.5), "b": (2.0, 0.5), "c": (0.5, 0.9), "d": (2.0, 0.4)})
    assert d["b"] == ["a", "d"]        # a: cheaper, as accurate; d: same cost, more accurate
    assert d["a"] == [] and d["c"] == [] and d["d"] == []
    assert CS.dominated({"x": (1.0, 1.0), "y": (1.0, 1.0)}) == {"x": [], "y": []}    # ties dominate nothing


def test_call_costs_billed_and_list():
    calls = pd.DataFrame({
        "model": ["gpt", "gpt", "gpt", "glm", "jev", "jev_bands"],
        "route": ["batch", "batch", "batch", "standard", "standard", "standard"],
        "tokens_in": [1000, 1000, 1000, 1000, 1000, 1000], "tokens_out": [100, 100, 0, 200, 0, 0],
        "valid": [True, True, False, True, True, True]})
    c = CS.call_costs(calls, CFG)
    assert "jev_bands" not in c
    g = c["gpt"]
    assert g["calls"] == 3 and g["usable"] == 2 and g["failed"] == 1
    assert np.isclose(g["usd_list_total"], (3 * 1000 * 2.0 + 200 * 10.0) / 1e6)
    assert np.isclose(g["usd_billed_total"], (3 * 1000 * 1.0 + 200 * 5.0) / 1e6)
    assert np.isclose(g["usd_billed_per_usable"], g["usd_billed_total"] / 2)       # the failed call counts in the cost
    assert np.isclose(c["glm"]["usd_billed_total"], c["glm"]["usd_list_total"])
    assert np.isclose(c["jev"]["usd_list_total"], 1000 * 0.042 / 1e6)
    # no repeat/annotated/variant columns: every call is on the main pass
    assert g["answers"] == 3 and np.isclose(g["usd_list_per_1000_answers"], g["usd_list_total"] / 3 * 1000)
    assert g["reasoning_tokens_per_answer"] is None and c["jev"]["output_tokens_per_answer"] is None


def test_per_answer_columns_count_the_main_pass_and_reasoning_from_raw(tmp_path):
    import json
    paths = []
    for i, r in enumerate([40, None, 60, 999]):
        f = tmp_path / f"{i}.json"
        u = {"prompt_tokens": 1000, "completion_tokens": 100}
        if r is not None:
            u["completion_tokens_details"] = {"reasoning_tokens": r}
        f.write_text(json.dumps({"request": {}, "response": {"usage": u}, "meta": {}}))
        paths.append(str(f))
    calls = pd.DataFrame({
        "model": ["gpt"] * 4, "route": ["standard"] * 4, "tokens_in": [1000] * 4, "tokens_out": [100, 100, 100, 300],
        "valid": [True, False, True, True], "repeat": [0, 0, 0, 1], "annotated": [False] * 4,
        "variant": ["raw"] * 4, "raw_path": paths})
    cache = tmp_path / "reasoning_tokens.parquet"
    rt = CS.reasoning_by_call(calls, cache)
    assert list(rt.iloc[:3].fillna(-1)) == [40, -1, 60] and np.isnan(rt.iloc[3])     # the repeat is not on the main pass
    assert cache.exists() and CS.reasoning_by_call(calls, cache).equals(rt)           # second read from the cache
    g = CS.call_costs(calls, CFG, reasoning=rt)["gpt"]
    assert g["answers"] == 3 and g["answers_reporting_reasoning"] == 2 and g["reasoning_tokens_per_answer"] == 50
    assert g["output_tokens_per_answer"] == 100 and g["input_tokens_per_answer"] == 1000
    per = (1000 * 2.0 + 100 * 10.0) / 1e6
    assert np.isclose(g["usd_list_per_1000_answers"], per * 1000) and np.isclose(g["usd_list_per_million_answers"], per * 1e6)
    assert np.isclose(g["usd_billed_per_1000_answers"], per * 1000)


def test_value_cheaper_choice_and_dominance():
    res = {"meta": {"models": ["jev", "gpt", "glm"], "tiers": {"jev": "jev", "gpt": "primary", "glm": "primary"}},
           "prediction": {"n_records": 1000, "systems": {"jev": {"log_loss": 0.35, "brier": 0.10},
                                                           "gpt": {"log_loss": 0.34, "brier": 0.10},
                                                           "glm": {"log_loss": 0.36, "brier": 0.11}},
                          "focal_vs_llms": {"pairs": {"jev - gpt": {"diff": 0.01, "ci_simultaneous": [0.0, 0.02],
                                                                    "noninferiority_margin": 0.009, "noninferior": False},
                                                      "jev - glm": {"diff": -0.01, "ci_simultaneous": [-0.02, 0.0],
                                                                    "noninferiority_margin": 0.009, "noninferior": True}}}},
           "per_call_error": {"jev": {"mae_logit": 0.4}, "gpt": {"mae_logit": 0.3}, "glm": {"mae_logit": 0.5}}}
    costs = {"jev": {"usd_list_per_usable": 0.0001, "usd_billed_per_usable": 0.0001},
             "gpt": {"usd_list_per_usable": 0.003, "usd_billed_per_usable": 0.0015},
             "glm": {"usd_list_per_usable": 0.012, "usd_billed_per_usable": 0.012}}
    v = CS.value(res, costs, {"systems": {"jev": {"median_s": 3.5, "p90_s": 4.0, "n": 100}}})
    assert v["jev_vs_primary"]["glm"]["acceptable_cheaper_choice"] is True
    assert v["jev_vs_primary"]["gpt"]["acceptable_cheaper_choice"] is False
    assert np.isclose(v["jev_vs_primary"]["glm"]["cost_ratio_list"], 120)
    assert v["dominated"]["log_loss"]["glm"] == ["gpt", "jev"]
    assert v["systems"]["jev"]["timing_median_s"] == 3.5


def test_timing_first_records_and_summary():
    states = pd.DataFrame({"profile": list(range(300)) * 2, "edit": ["baseline"] * 300 + ["C6_age_plus5"] * 300,
                           "annotated": False, "text": [f"t{i}" for i in range(600)]})
    r1, r2 = TS.first_records(states, 100), TS.first_records(states, 100)
    assert len(r1) == 100 and r1.profile.is_unique and (r1.edit == "baseline").all()
    assert list(r1.profile) == list(r2.profile)                                   # the same records every time
    df = pd.DataFrame({"system": ["jev"] * 10, "i": range(10), "started": np.arange(10.0),
                       "latency_s": np.arange(1.0, 11.0), "valid": [True] * 9 + [False],
                       "provider": ["TypeSafe"] * 10, "tokens_in": [900] * 10, "tokens_out": [80] * 10,
                       "usd_list": [1e-5] * 10, "error": [None] * 10})
    s = TS.summarise(df)["jev"]
    assert s["n"] == 10 and s["usable"] == 9 and np.isclose(s["median_s"], 5.5) and np.isclose(s["p90_s"], 9.1)

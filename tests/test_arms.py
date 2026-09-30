"""Tiers, arms, the reasoning override, the Hugging Face router client and the 200-record subset. No network."""
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import clients as C
from jevity import analysis as A
from jevity.clients import LLMClient, RawStore, arm_systems, split_system, system_tier, active_families
from jevity.runner import plan_split, load_subset

TXT = '{"age": 64, "sex": "female"}'
STMT = "This person will die from any cause within ten years of this examination."
PERT = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "perturbations.yaml").read_text())


def test_tiers_and_arm_systems():
    assert arm_systems("primary") == ["gpt", "claude", "gemini", "muse", "glm"]
    assert set(arm_systems("free")) == {"gpt_free", "claude_free", "gemini_free"}
    assert set(C.MODELS["arms"]) == {"primary", "free", "pair", "reasoning"}
    assert set(arm_systems("pair")) == {"medgemma", "gemma"}
    r = arm_systems("reasoning")
    assert {f"{f}@{e}" for f in ("gpt", "claude", "gemini", "muse", "glm") for e in ("low", "high")} == set(r)
    assert split_system("gpt@low") == ("gpt", "low") and split_system("medgemma") == ("medgemma", None)
    assert system_tier("gpt@high") == "reasoning" and system_tier("claude_free") == "free"
    assert system_tier("jev_bands") == "jev" and system_tier("medgemma") == "pair"


def test_disabled_family_is_never_called(monkeypatch):
    monkeypatch.setitem(C.MODELS["families"], "gpt", {**C.MODELS["families"]["gpt"], "disabled": True})
    assert "gpt" not in active_families()
    with pytest.raises(RuntimeError):
        LLMClient("gpt", None)


def test_effort_override_leaves_default_request_untouched(tmp_path):
    store = RawStore(tmp_path)
    base = LLMClient("gpt", store).request(TXT, STMT)
    assert base["model"] == C.MODELS["families"]["gpt"]["slug"]
    low = LLMClient("gpt@low", store).request(TXT, STMT)
    assert "reasoning" not in base and "temperature" not in base            # provider defaults: nothing sent
    assert low["reasoning"] == {"effort": "low"}
    assert {k: v for k, v in low.items() if k != "reasoning"} == base      # only the override differs
    assert store.path(low) != store.path(base)                             # its own cache entry
    assert LLMClient("claude", store, effort="high").request(TXT, STMT)["reasoning"] == {"effort": "high"}
    with pytest.raises(ValueError):
        LLMClient("gpt", store, effort="medium")
    with pytest.raises(ValueError):
        LLMClient("medgemma", store, effort="low")                         # the reasoning arm is OpenRouter-only


def test_hf_router_client(tmp_path, monkeypatch):
    seen = {}

    def fake_post(url, headers, body, retries=6):
        seen.update(url=url, auth=headers["Authorization"], body=body)
        return {"id": "hf-1", "model": body["model"], "choices": [{"message": {"content": '{"p": 0.31}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 700, "completion_tokens": 9}}

    monkeypatch.setattr(C, "_post", fake_post)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    cl = LLMClient("medgemma", RawStore(tmp_path))
    assert cl.batch is None                                                 # never batched
    r = cl.probability(TXT, STMT)
    assert r["p"] == pytest.approx(0.31) and r["parse"] == "json"
    assert seen["url"] == "https://router.huggingface.co/v1/chat/completions"
    assert seen["auth"] == "Bearer hf_test"
    assert seen["body"]["model"] == "google/medgemma-27b-text-it:featherless-ai"
    assert "provider" not in seen["body"] and "seed" not in seen["body"]
    assert seen["body"]["response_format"]["type"] == "json_schema"
    g = LLMClient("gemma", RawStore(tmp_path)).request(TXT, STMT)
    assert g["model"] == "google/gemma-3-27b-it:featherless-ai"


def _states(n_profiles=60):
    edits = ["baseline", "C1a_hba1c_plus1", "L1_race_nhw_to_nhb", "I1_field_order", "C1b_hba1c_plus2", "M2_race_removed"]
    rows = [{"profile": 1000 + i, "edit": e, "annotated": False, "text": f"rec {i} {e}"} for i in range(n_profiles) for e in edits]
    rows += [{"profile": 1000 + i, "edit": "C1a_hba1c_plus1", "annotated": True, "text": f"rec {i} ann"} for i in range(n_profiles)]
    return pd.DataFrame(rows)


def test_plan_split_arms_and_subset():
    st = _states()
    subset = [1000 + i for i in range(0, 60, 3)]                           # 20 records
    ev = plan_split(st, "eval", PERT, subset=subset)
    for arm in ("reasoning",):
        cs = [c for c in ev if c.model in set(arm_systems(arm))]
        assert cs and {c.profile for c in cs} == set(subset)
        assert all(c.repeat == 0 and c.variant == "raw" and not c.annotated for c in cs)
        assert len(cs) == len(arm_systems(arm)) * len(subset) * 6          # every edit of the subset records, once
    for arm in ("free", "pair"):
        cs = [c for c in ev if c.model in set(arm_systems(arm))]
        assert {c.profile for c in cs} == set(st.profile)
        assert all(c.repeat == 0 and c.variant == "raw" and not c.annotated for c in cs)
    prim = [c for c in ev if c.model == "gpt"]
    assert any(c.repeat > 0 for c in prim) and any(c.variant == "M1" for c in prim)   # primary arm: repeats and M1
    with pytest.raises(SystemExit):
        plan_split(st, "eval", PERT, subset=None)
    wide = plan_split(st[st.edit.isin(["baseline", "M2_race_removed"])], "wide", PERT)
    assert {c.model for c in wide} <= {"jev"} | set(arm_systems("primary"))
    only = plan_split(st, "eval", PERT, subset=subset, systems=["gpt@low", "jev"])
    assert {c.model for c in only if not c.model.startswith("jev")} == {"gpt@low"}


def test_load_subset(tmp_path):
    assert load_subset(tmp_path / "missing.yaml") is None
    f = tmp_path / "s.yaml"
    f.write_text(yaml.safe_dump({"eval200": {"profiles": [3, 1, 2], "seed": 20260922}}))
    assert load_subset(f) == [3, 1, 2]


def test_ordering_and_secondary_intervals_are_marginal():
    assert A.ordered({"gemma", "gpt@low", "jev", "gpt", "claude@high", "jev_bands", "gpt_free"}) == \
        ["jev", "gpt", "gpt_free", "gemma", "gpt@low", "claude@high"]
    assert A.is_primary("jev") and A.is_primary("glm") and not A.is_primary("gpt@low") and not A.is_primary("gemma")
    res = {"critical_value": {"D_logit": 3.1}, "simultaneous_excludes_zero": {}, "n_tests": 5,
           "cells": {"C1a": {"gpt_free": {"D_logit": 0.1, "D_logit_ci": [0, 0.2], "D_logit_ci_wald": [0, 0.2],
                                          "D_logit_ci_simultaneous": [-1, 1], "rules_out_abs_D": {}}}},
           "clinical_slope": {"gpt_free": {"beta": 0.9, "ci": [0.8, 1.0], "ci_simultaneous": [0.7, 1.1], "excludes_one": False}}}
    out = A._marginal_only(res)
    cell = out["cells"]["C1a"]["gpt_free"]
    assert "D_logit_ci_simultaneous" not in cell and "rules_out_abs_D" not in cell and "D_logit_ci_wald" in cell
    assert "critical_value" not in out and "ci_simultaneous" not in out["clinical_slope"]["gpt_free"]


def test_prediction_block_keeps_full_set_with_subset_arm_present():
    """A reasoning-arm system answers only the 200-record subset: it must not shrink the records the prediction block
    takes (every full-set system scored), so n_records is the full common set."""
    import numpy as np
    rng = np.random.default_rng(5)
    prof = list(range(1, 301))
    rows = [{"model": m, "profile": p, "edit": "baseline", "repeat": 0, "annotated": False, "variant": "raw", "valid": True,
             "p": float(rng.uniform(0.05, 0.6))} for m in ("jev", "gpt", "glm", "gpt_free", "medgemma", "gemma") for p in prof]
    rows += [{"model": m, "profile": p, "edit": "baseline", "repeat": 0, "annotated": False, "variant": "raw", "valid": True,
              "p": 0.3} for m in ("gpt@low", "gpt@high") for p in prof[:50]]
    calls = pd.DataFrame(rows)
    y = pd.Series(rng.integers(0, 2, len(prof)), index=prof)
    comps = {"reference_spline_logit": pd.Series(rng.uniform(0.05, 0.6, len(prof)), index=prof),
             "prevalence_only": pd.Series(0.2, index=prof)}
    models = A.ordered(set(calls.model))
    assert "gpt@low" in models                                             # the subset arm is present in the run
    base = A.baseline_predictions(calls, models, prof, comps)
    assert not {"gpt@low", "gpt@high", "prevalence_only"} & set(base)
    assert {"jev", "gpt", "glm", "gpt_free", "medgemma", "gemma", "reference_spline_logit"} <= set(base)
    pb = A.prediction_block(base, y, B=50, focal_versus=["gpt", "glm"])
    assert pb["n_records"] == len(prof)
    assert A.prediction_block({**base, "gpt@low": calls[calls.model == "gpt@low"].set_index("profile")["p"].reindex(prof)},
                              y, B=50)["n_records"] == 50              # a subset-arm system would shrink it
    assert A.full_set_systems(["jev", "gpt", "claude_free", "gemma", "glm@high", "gpt@low"]) == ["jev", "gpt", "claude_free", "gemma"]


def test_per_call_error_interval_covers_point():
    """Refit noise inflates every draw's |z - d|; the Wald-form interval stays centred on the point."""
    import numpy as np
    rng = np.random.default_rng(0)
    prof = np.repeat(np.arange(100), 2)
    tab = pd.DataFrame({"profile": prof, "edit": ["C1a_hba1c_plus1", "L1_race_nhw_to_nhb"] * 100,
                        "klass": ["clinical", "label"] * 100, "z_logit": rng.normal(0.3, 0.2, 200)})
    tab["d_logit"] = 0.3
    idx = {(int(p), e): i for i, (p, e) in enumerate(zip(tab.profile, tab.edit))}
    refits = A.Refits(0.3 + rng.normal(0, 0.3, (200, 50)), np.zeros((200, 50)), idx)
    r = A.per_call_error(tab, refits, A.Draws(range(100), 200, 50), ["C1a_hba1c_plus1", "L1_race_nhw_to_nhb"])
    lo, hi = r["mae_logit_ci"]
    assert lo < r["mae_logit"] < hi and r["mae_logit_draw_mean"] > r["mae_logit"]


def test_execute_runs_systems_in_parallel(tmp_path):
    """Each system gets its own pool and all systems run at once; a system never exceeds its own cap."""
    import threading, time
    from jevity.runner import Call, execute
    live, peak, lock = {}, {"all": 0}, threading.Lock()

    class Slow:
        def __init__(self, m): self.m = m
        def probability(self, text, stmt):
            with lock:
                live[self.m] = live.get(self.m, 0) + 1
                peak[self.m] = max(peak.get(self.m, 0), live[self.m])
                peak["all"] = max(peak["all"], sum(live.values()))
            time.sleep(0.2)
            with lock:
                live[self.m] -= 1
            return {"p": 0.2, "valid": True}

    calls = [Call(m, i, "baseline", False, 0, "x") for m in ("gpt", "glm", "medgemma") for i in range(6)]
    df = execute(calls, lambda m, r, v: Slow(m), "nhanes", tmp_path / "c.parquet", workers={"default": 3, "medgemma": 2})
    assert len(df) == 18 and df.valid.all()
    assert peak["gpt"] == 3 and peak["glm"] == 3 and peak["medgemma"] == 2
    assert peak["all"] == 8                                   # all three systems at once, not one after the other

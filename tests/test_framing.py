"""Framing sensitivity analysis: the system message without its study sentence (variant F1), its call plan (primary
chatbots, the subset of config/subsets.yaml, baseline and the confirmatory family, own store) and its paired analysis.
No network."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import analysis as A
from jevity.clients import JevClient, LLMClient, PROMPTS, _sha, arm_systems, drop_sentence
from jevity.runner import plan_split

ROOT = Path(__file__).resolve().parents[1]
PERT = yaml.safe_load((ROOT / "config" / "perturbations.yaml").read_text())
STMT = PROMPTS["statements"]["nhanes"]
SENTENCE = "You are helping with a research study of mortality risk."


def test_f1_drops_only_the_study_sentence():
    raw = LLMClient("gpt", None).request("R", STMT)
    f1 = LLMClient("gpt", None, variant="F1").request("R", STMT)
    s_raw, s_f1 = raw["messages"][0]["content"], f1["messages"][0]["content"]
    assert s_raw.startswith(SENTENCE + " ") and SENTENCE not in s_f1
    assert s_f1 == s_raw[len(SENTENCE) + 1:]
    assert {k for k in raw if raw[k] != f1[k]} == {"messages"}
    assert raw["messages"][1] == f1["messages"][1]
    assert _sha(raw) != _sha(f1)
    m1 = LLMClient("gpt", None, variant="M1").request("R", STMT)["messages"][0]["content"]
    assert m1.startswith(SENTENCE)                                            # M1 keeps the study sentence


def test_raw_system_message_is_the_configured_text():
    configured = " ".join(" ".join(x.split()) for x in (PROMPTS["llm"]["system"], PROMPTS["llm"]["reply"]))
    for fam in arm_systems("primary"):
        assert LLMClient(fam, None).request("R", STMT)["messages"][0]["content"] == configured


def test_variant_guards():
    with pytest.raises(ValueError):
        drop_sentence("A short text.", SENTENCE)
    with pytest.raises(ValueError):
        LLMClient("gpt", None, variant="X")
    with pytest.raises(ValueError):
        JevClient(None, variant="F1")                                         # Jev has no system message


def _states(n=60):
    edits = ["baseline", "C1a_hba1c_plus1", "L1_race_nhw_to_nhb", "I1_field_order", "C1b_hba1c_plus2", "M2_race_removed"]
    rows = [{"profile": 1000 + i, "edit": e, "annotated": False, "text": f"rec {i} {e}"} for i in range(n) for e in edits]
    rows += [{"profile": 1000 + i, "edit": "C1a_hba1c_plus1", "annotated": True, "text": f"rec {i} ann"} for i in range(n)]
    return pd.DataFrame(rows)


def test_plan_framing():
    st, subset = _states(), [1000 + i for i in range(0, 60, 3)]
    fr = plan_split(st, "framing", PERT, subset=subset)
    assert {c.model for c in fr} == set(arm_systems("primary")) and "jev" not in {c.model for c in fr}
    assert all(c.variant == "F1" and c.repeat == 0 and not c.annotated for c in fr)
    assert {c.profile for c in fr} == set(subset)
    assert {c.edit for c in fr} == {"baseline", "C1a_hba1c_plus1", "L1_race_nhw_to_nhb"}   # family edits only
    assert len(fr) == len(arm_systems("primary")) * len(subset) * 3
    only = plan_split(st, "framing", PERT, subset=subset, systems=["claude"])
    assert {c.model for c in only} == {"claude"}
    ev = plan_split(st, "eval", PERT, subset=subset)
    assert not any(c.variant == "F1" for c in ev)                             # the evaluation plan is unchanged
    with pytest.raises(SystemExit):
        plan_split(st, "framing", PERT, subset=None)


def _synthetic(n=60):
    rng = np.random.default_rng(0)
    eff = {"C1a_hba1c_plus1": 0.3, "C2a_creatinine_x1.5": 0.5, "C3_albumin_minus0.5": 0.4, "L1_race_nhw_to_nhb": 0.0}
    base = rng.normal(-1.5, 0.5, n)
    rows, inst = [], []
    for i in range(n):
        for e in ["baseline"] + list(eff):
            for m in ("gpt", "claude"):
                z = base[i] + eff.get(e, 0.0)
                rows.append({"model": m, "profile": i, "edit": e, "annotated": False, "repeat": 0, "valid": True,
                             "variant": "raw", "p": float(A.expit(z))})
                extra = 0.3 if (m == "claude" and e == "C1a_hba1c_plus1") else 0.0     # claude doubles C1a under F1
                rows.append({"model": m, "profile": i, "edit": e, "annotated": False, "repeat": 0, "valid": True,
                             "variant": "F1", "p": float(A.expit(z + 0.4 + extra))})    # both: +0.4 at every state
            if e != "baseline":
                inst.append({"profile": i, "edit": e, "applicable": True, "supported": True,
                             "klass": "label" if e.startswith("L") else "clinical", "feature": e,
                             "d_logit_spline_logit": 0.2, "d_prob_spline_logit": 0.03})
    calls, inst = pd.DataFrame(rows), pd.DataFrame(inst)
    tab = inst.assign(d_logit=inst.d_logit_spline_logit, d_prob=inst.d_prob_spline_logit)
    return calls, inst, A.Refits.degenerate(tab), list(eff)


def test_framing_sensitivity_analysis():
    calls, inst, refits, fam = _synthetic()
    raw, f1 = calls[calls.variant == "raw"], calls[calls.variant == "F1"]
    r = A.framing_sensitivity(raw, f1, inst, refits, subset=range(60), fam=fam, B=200)
    g, c = r["systems"]["gpt"], r["systems"]["claude"]
    assert g["validity"]["F1"] == 1.0 and g["validity"]["n_F1"] == 60 * 5
    assert g["baseline_shift"]["mean_logit"] == pytest.approx(0.4, abs=1e-6)
    lo, hi = g["baseline_shift"]["mean_logit_ci"]
    assert lo == pytest.approx(0.4, abs=1e-6) and hi == pytest.approx(0.4, abs=1e-6)
    for e in fam:                                      # the shift cancels in every contrast
        assert g["edits"][e]["diff"] == pytest.approx(0.0, abs=1e-6)
    assert g["edits"]["C1a_hba1c_plus1"]["D_logit_raw"] == pytest.approx(0.1, abs=1e-6)
    assert c["edits"]["C1a_hba1c_plus1"]["diff"] == pytest.approx(0.3, abs=1e-6)
    assert c["edits"]["C1a_hba1c_plus1"]["D_logit_F1"] == pytest.approx(0.4, abs=1e-6)
    assert c["summary"]["n_ci_excludes_zero"] == 1 and g["summary"]["n_ci_excludes_zero"] == 0
    assert "clinical_slope" in g and g["clinical_slope"]["diff"] == pytest.approx(0.0, abs=1e-6)
    sub = A.framing_sensitivity(raw, f1, inst, refits, subset=range(30), fam=fam, B=50)
    assert sub["n_subset"] == 30 and sub["systems"]["gpt"]["validity"]["n_F1"] == 30 * 5    # subset only

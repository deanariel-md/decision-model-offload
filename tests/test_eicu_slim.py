"""eICU replication: baseline-only states, the call plan, the batch route and statement for population eicu_demo, the
masked-field text and the descriptive metrics. Synthetic data only; no network."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from jevity import eicu as E, eicu_analysis as EA                          # noqa: E402
from jevity.clients import FillClient, LLMClient, PROMPTS, RawStore        # noqa: E402
from test_eicu import _tables, _write                                      # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _demo_root(tmp_path):
    _write(tmp_path / "src", _tables(300, 1, 100_000, [1, 2, 3]))
    out = tmp_path / "data" / "eicu"
    E.build_demo(tmp_path / "src", out)
    return tmp_path


def test_baseline_states_and_plan(tmp_path, monkeypatch):
    root = _demo_root(tmp_path)
    ms = _load("make_eicu_states")
    st = ms.build_baseline("eval", root)
    demo = pd.read_parquet(root / "data" / "eicu" / "demo_cohort.parquet")
    assert set(st.edit) == {"baseline"} and len(st) == (demo.split == "eval").sum()
    assert st.text.iloc[0] == E.render(demo[demo.SEQN == st.profile.iloc[0]].iloc[0].to_dict())
    run = _load("run_eicu")
    monkeypatch.setattr(run, "D", root / "data" / "eicu")
    calls = run.plan("eval")
    assert run.systems() == ["jev", "gpt", "claude", "gemini", "muse", "glm"]
    assert len(calls) == 6 * len(st) and {c.edit for c in calls} == {"baseline"} and {c.repeat for c in calls} == {0}


def test_eicu_demo_statement_batch_route_and_system_prompt(tmp_path):
    store = RawStore(tmp_path / "runs")
    stmt = PROMPTS["statements"]["eicu_demo"]
    assert stmt == "This patient will die before hospital discharge."
    for fam, batched in (("gpt", True), ("claude", True), ("gemini", True), ("muse", False), ("glm", False)):
        cl = LLMClient(fam, store, "eicu_demo")
        assert (cl.batch is not None) == batched
        req = cl.request("{}", stmt)
        assert req["messages"][1]["content"].rstrip().endswith(f"Statement: {stmt}")
        sysmsg = req["messages"][0]["content"]
        assert "one de-identified record from an intensive care unit database and" in sysmsg
        assert "national health survey" not in sysmsg
    nh = LLMClient("gpt", store, "nhanes")._system()        # NHANES: byte-identical to the configured text
    assert nh == " ".join(" ".join(x.split()) for x in (PROMPTS["llm"]["system"], PROMPTS["llm"]["reply"]))


def test_fill_client_unit_argument_leaves_nhanes_requests_unchanged(tmp_path):
    store = RawStore(tmp_path / "runs")
    cl = FillClient("gpt", store, "nhanes")
    assert cl.request("{}", "total_cholesterol") == cl.request("{}", "total_cholesterol", "mg/dL")
    r = FillClient("gpt", store, "eicu_demo").request("{}", "serum_sodium", "mmol/L")
    assert r["messages"][1]["content"].rstrip().endswith("Missing field: serum_sodium (mmol/L)")


def test_masked_text_and_pick(tmp_path):
    root = _demo_root(tmp_path)
    ex = _load("eicu_exposure")
    demo = pd.read_parquet(root / "data" / "eicu" / "demo_cohort.parquet")
    ev = ex.pick(demo, n=50)
    assert len(ev) == 50 and ev.SEQN.is_unique and (ev.split == "eval").all()
    for f, spec in E.CFG["exposure"]["fields"].items():
        assert ev.loc[ev.field == f, spec["column"]].notna().all()
    r = ev.iloc[0]
    t = ex.masked_text(r.to_dict(), r.field)
    assert '"value": "MISSING"' in t and E.CFG["exposure"]["fields"][r.field]["unit"] in t


def test_metrics_and_block():
    rng = np.random.default_rng(3)
    n = 4000
    p = rng.uniform(0.02, 0.6, n)
    y = (rng.random(n) < p).astype(float)
    m = EA.metrics(p, y)
    assert abs(m["calibration_slope"] - 1) < 0.15 and abs(m["calibration_intercept"]) < 0.15
    assert abs(m["observed_to_expected"] - 1) < 0.08 and 0.6 < m["auroc"] < 0.9
    assert abs(m["brier"] - ((p - y) ** 2).mean()) < 1e-12
    idx = pd.RangeIndex(n)
    preds = {"good": pd.Series(p, index=idx), "flat": pd.Series(np.full(n, y.mean()), index=idx),
             "partial": pd.Series(np.where(np.arange(n) < n - 100, p, np.nan), index=idx)}
    b = EA.baseline_block(preds, pd.Series(y, index=idx), B=200)
    assert b["n_records"] == n - 100 and b["scored_alone"]["good"] == n
    g = b["predictors"]["good"]
    assert g["log_loss_ci"][0] <= g["log_loss"] <= g["log_loss_ci"][1]
    assert b["predictors"]["flat"]["auroc"] == 0.5 and g["log_loss"] < b["predictors"]["flat"]["log_loss"]

"""scripts/llm_forms/analyze.py on synthetic calls tables (analyze.make_stub): no stored answer, no model call."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "summaries"))
sys.path.insert(0, str(ROOT / "scripts" / "llm_forms"))
_spec = importlib.util.spec_from_file_location("llm_forms_analyze", ROOT / "scripts" / "llm_forms" / "analyze.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
import boot  # noqa: E402
from sens_spec import prop  # noqa: E402


def _run(tmp: Path, calls: pd.DataFrame, systems=None) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "calls_STUB.parquet"
    calls.to_parquet(p, index=False)
    return A.run(p, tmp, tmp, systems, None, A.CONFIG, stub=True)


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    c = A.make_stub()
    d = tmp_path_factory.mktemp("full")
    res = _run(d, c)
    return c, res, tmp_path_factory, d


def test_stub_has_the_table_columns(stub):
    c, res, _, _ = stub
    cols = ["system", "form", "set", "item_id", "truth", "answer", "valid", "probs", "prob_status", "top_prob", "parse",
                "parts", "model_reported", "provider", "route", "source", "tokens_in", "tokens_out", "tokens_reasoning",
                "usd_reported", "usd_list", "latency_s", "finish_reason", "raw_path", "error", "excluded", "exclude_reason"]
    assert set(cols) <= set(c.columns)
    assert len(c) == 11 * 7 * 240 and not res["incomplete_cells"] and not res["systems_absent"]
    assert c.groupby("set").item_id.nunique().to_dict() == {"main": 102, "heldout": 18, "misleading_feature": 100,
                                                           "nonregional_sites": 20}


def test_wilson_equals_sens_spec_prop(stub):
    _, res, _, _ = stub
    for s, fs in res["accuracy"].items():
        for f, gs in fs.items():
            for g, a in gs.items():
                ref = prop(a["k"], a["n"])
                assert (a["k"], a["n"], a["p"], a["ci"]) == (ref["k"], ref["n"], ref["p"], ref["ci"])
    a = res["accuracy"]["gpt"]["documented"]
    assert a["pool240"]["n"] == 240 and sum(a[k]["n"] for k in boot.SETS) == 240
    assert a["ordinary"]["n"] == 120 and a["built_to_mislead"]["n"] == 120


def test_correct_computed_when_column_absent(stub, tmp_path):
    c, res, _, _ = stub
    r2 = _run(tmp_path, c.drop(columns=["correct"]))
    assert r2["accuracy"] == res["accuracy"]


def test_bootstrap_reproducible_and_stratified(stub, tmp_path):
    c, res, _, _ = stub
    u = c[["item_id", "set"]].drop_duplicates().sort_values("item_id")
    sets = u.set.to_numpy()
    i1, i2 = boot.stratified_index(sets, boot.SEED, boot.B), boot.stratified_index(sets, boot.SEED, boot.B)
    assert i1.shape == (2000, 240) and (i1 == i2).all()
    assert not (boot.stratified_index(sets, boot.SEED + 1, boot.B) == i1).all()
    col_set = sets[i1[0]]
    assert (sets[i1] == col_set).all()                                   # each column stays within one set
    for k in boot.SETS:                                                   # every resample keeps each set's size
        assert ((sets[i1] == k).sum(1) == (sets == k).sum()).all()
    r2 = _run(tmp_path, c)
    assert r2["resample_index_sha256"] == res["resample_index_sha256"]
    assert r2["paired"] == res["paired"] and r2["pairing_sensitivity"] == res["pairing_sensitivity"]
    assert res["seed"] == 20260929 and res["n_boot"] == 2000


def test_self_comparison_is_zero(stub):
    c, res, _, _ = stub
    rng = np.random.default_rng(0)
    a = rng.random(240) < 0.8
    idx = boot.stratified_index(np.repeat(boot.SETS, [102, 18, 100, 20]))
    assert boot.diff_ci(a, a, idx) == {"est_pts": 0.0, "ci_pts": [0.0, 0.0]}
    for s in ("gpt", "claude"):                                           # stub: named_field identical to names
        r = res["paired"]["form_minus_names"][s]["named_field"]
        assert r["identical_to_names"] and r["est_pts"] == 0 and r["ci_pts"] == [0.0, 0.0]
    assert not res["paired"]["form_minus_names"]["gemini"]["named_field"]["identical_to_names"]


def test_band_counts_add_up(stub):
    _, res, _, _ = stub
    blocks = [res["confidence"][s][f] for s in res["systems"] for f in A.FORMS]
    blocks += [v for g in res["confidence_pooled"].values() for v in g.values()]
    for k in blocks:
        bands = k["bands"]
        assert sum(b["answers"] for b in bands.values()) == k["n"] - k["unusable"] - k["without_probability"]
        assert all(b["correct"] + b["incorrect"] == b["answers"] for b in bands.values())
        assert bands["1.00"]["answers"] == k["full_confidence_n"]
        assert bands["1.00"]["incorrect"] == k["errors_at_full_confidence"]
    assert any(res["cells"][s][f]["part_list_failed"] for s in res["systems"] for f in A.FORMS)


def test_part_list_failure_leaves_confidence(stub):
    c, res, _, _ = stub
    for (s, f), g in c[c.form.isin(["documented", "structure_only"])].groupby(["system", "form"]):
        failed = g.parts.map(lambda p: A.part_failed(A._parts(p))) & g.valid
        nop = (g.valid & g.top_prob.isna()) | failed
        assert res["cells"][s][f]["without_probability"] == int(nop.sum())
        assert res["cells"][s][f]["part_list_failed"] == int(failed.sum())


def test_unstaged_counts(stub):
    c, res, _, _ = stub
    tot = 0
    for (s, f), g in c.groupby(["system", "form"]):
        un = g.parse == "unstaged"
        txm = un & g.parts.map(lambda p: A.tx_nx_m1(A._parts(p)))
        assert res["cells"][s][f]["unstaged"] == int(un.sum())
        assert res["cells"][s][f]["unstaged_tx_or_nx_with_m1"] == int(txm.sum())
        tot += int(txm.sum())
    assert tot > 0
    assert A.tx_nx_m1({"t_category": {"choice": "T3"}, "regional_nodes": {"choice": "not stated"},
                       "distant_metastasis": {"choice": "M1c"}})
    assert not A.tx_nx_m1({"t_category": {"choice": "cannot be assessed"}, "distant_metastasis": {"choice": "none"}})


def test_pairing_equals_llm_alone_without_full_confidence(tmp_path):
    c = A.make_stub(jev_full=False)
    assert not ((c.system == "jev") & (c.top_prob >= 1 - 1e-9)).any()
    res = _run(tmp_path, c)
    for s, r in res["pairing_sensitivity"].items():
        assert r["kept"] == 0 and r["pair_correct"] == r["alone_correct"]
        assert r["est_pts"] == 0 and r["ci_pts"] == [0.0, 0.0]
        assert r["cost_share"] == pytest.approx((r["usd_jev_total"] + r["usd_llm_total"]) / r["usd_llm_total"])
        assert r["usd_llm_passed_on"] == pytest.approx(r["usd_llm_total"])


def test_cost_per_1000_by_hand(stub):
    c, res, _, _ = stub
    price = {"jev": (0.042, 0.0), "claude": (4.0, 20.0), "gemma": (0.1, 0.3), "gpt": (5.0, 30.0)}  # GPT-5.6 Sol at 5/30
    for s, (pi, po) in price.items():
        g = c[(c.system == s) & (c.form == "documented")]
        hand = 1000 * sum((ti * pi + to * po) / 1e6 for ti, to in zip(g.tokens_in, g.tokens_out)) / 240
        assert res["cost"][s]["documented"]["usd_per_1000"] == pytest.approx(hand, rel=1e-12)
        assert res["prices"][s] == [pi, po]
    g = c[(c.system == "gpt") & (c.form == "documented")]
    hand = 1000 * sum((ti * 2 + to * 10) / 1e6 for ti, to in zip(g.tokens_in, g.tokens_out)) / 240   # the sensitivity price, 2/10
    assert res["cost"]["gpt"]["documented"]["sensitivity"]["usd_per_1000"] == pytest.approx(hand, rel=1e-12)
    rep = g.usd_reported.dropna()
    assert res["cost"]["gpt"]["documented"]["reported"]["n_with"] == len(rep)
    assert res["cost"]["gpt"]["documented"]["reported"]["usd_per_1000"] == pytest.approx(1000 * rep.sum() / len(rep))


def test_missing_and_excluded_cells(tmp_path):
    c = A.make_stub(drop=[("gpt", "documented_notes"), ("medgemma", "names")])
    ex = (c.system == "claude") & (c.form == "names") & c.item_id.isin(["stub000", "stub150"])
    c.loc[ex, "excluded"], c.loc[ex, "exclude_reason"] = True, "model 'x', main runs 'y'"
    res = _run(tmp_path, c)
    inc = {(x["system"], x["form"]) for x in res["incomplete_cells"]}
    assert inc == {("gpt", "documented_notes"), ("medgemma", "names"), ("claude", "names")}
    assert res["accuracy"]["claude"]["names"]["pool240"]["n"] == 238
    assert not res["accuracy"]["claude"]["names"]["pool240"]["complete"]
    assert res["cells"]["claude"]["names"]["excluded"] == 2 and len(res["cells"]["claude"]["names"]["excluded_items"]) == 2
    assert res["accuracy"]["gpt"]["documented_notes"]["pool240"]["n"] == 0
    assert "skipped" in res["paired"]["form_minus_names"]["claude"]["documented"]
    assert "skipped" in res["paired"]["jev_minus_llm"]["documented_notes"]["gpt"]
    assert "skipped" in res["paired"]["form_minus_names"]["medgemma"]["definitions"]
    assert "skipped" not in res["pairing_sensitivity"]["gpt"]
    md = (tmp_path / "symmetric_forms_summary.md").read_text(encoding="utf-8")
    assert "**Incomplete.**" in md and "gpt documented_notes 0" in md
    assert (tmp_path / "symmetric_forms_table6_rows.json").exists() and (tmp_path / "analysis.json").exists()


def test_partial_equals_full(stub):
    c, res, tpf, full_dir = stub
    full = json.loads((full_dir / "analysis.json").read_text(encoding="utf-8"))
    d = tpf.mktemp("partial")
    _run(d, c, systems=["gpt", "gemma"])
    assert not (d / "analysis.json").exists()
    for s in ("gpt", "gemma"):
        p = json.loads((d / "partial" / f"{s}.json").read_text(encoding="utf-8"))
        assert p["complete"] and p["accuracy"][s] == full["accuracy"][s] and p["accuracy"]["jev"] == full["accuracy"]["jev"]
        assert p["paired"]["form_minus_names"] == full["paired"]["form_minus_names"][s]
        assert p["paired"]["jev_minus_llm"] == {f: full["paired"]["jev_minus_llm"][f][s] for f in A.FORMS}
        assert p["pairing_sensitivity"] == full["pairing_sensitivity"][s]
        assert p["cost"][s] == full["cost"][s] and p["confidence"][s] == full["confidence"][s]


def test_outputs_read_from_analysis(stub):
    _, res, _, d = stub
    t6 = json.loads((d / "symmetric_forms_table6_rows.json").read_text(encoding="utf-8"))
    an = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
    assert len(t6["rows"]) == 7 * (11 + 2)
    for r in t6["rows"]:
        src = an
        for k in r["trace"].split("."):
            src = src[k]
        assert r["n"] == src["n"] and r["bands"] == src["bands"] and r["accuracy"] == src["accuracy"]
    md = (d / "symmetric_forms_summary.md").read_text(encoding="utf-8")
    a = an["accuracy"]["jev"]["documented"]["pool240"]
    fc = an["full_confidence_by_group"]["jev"]["documented"]["pool240"]["share"]
    assert f"| Both: recommended form | Jev 1.13 | {100 * a['p']:.1f} ({100 * fc:.0f})" in md
    ps = an["pairing_sensitivity"]["claude"]
    assert f"| Claude Opus 5.5 | {ps['alone_correct']} of 240 | {ps['pair_correct']} of 240 |" in md


def test_band_matches_parser():
    try:
        from jevity import llm_forms_parse as LP
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"llm_forms_parse not importable: {e}")
    assert tuple(A.BANDS) == tuple(LP.BANDS)
    assert abs((1 - A.TOL) - LP.FULL_CUT) < 1e-15
    for p in [0.0, 0.25, 0.4999, 0.5, 0.6, 0.7499, 0.75, 0.9, 0.99, 1 - 2e-9, 1 - 1e-9, 1 - 5e-10, 1.0]:
        assert A.band(p) == LP.band(p), p


def test_stub_never_replaces_real_calls():
    with pytest.raises(AssertionError):
        A.main(["--stub", "--calls", str(A.OUT / "calls.parquet")])

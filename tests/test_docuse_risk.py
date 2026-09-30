"""Ten-year risk, Jev used as documented (src/jevity/docuse_risk.py, config/docuse_risk.yaml, scripts/docuse/).
Synthetic records (tests/synthetic.py, in a scratch repository root) and synthetic Jev responses only; every test runs
with the network blocked (clients._post raises)."""
import functools
import importlib.util
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jevity import clients
from jevity import docuse_risk as D
from jevity.serialize import NUMERIC_FIELDS, to_state

ROOT = Path(__file__).resolve().parents[1]
F = D.CFG["fields"]
N_WIDE = 300                              # synthetic records in the wide baseline set


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    """A scratch repository root with the files build_states, wide_profiles and dry_run read: a synthetic cohort
    (tests/synthetic.py), its baseline states as the original run wrote them (serialize.to_state and to_json, one per
    record) and Jev's baseline rows of the wide run (profile and reported input tokens)."""
    from synthetic import make
    from jevity.serialize import to_json
    root = tmp_path_factory.mktemp("root")
    (root / "data").mkdir()
    (root / "results" / "wide").mkdir(parents=True)
    make(n=N_WIDE, seed=0).to_parquet(root / "data" / "cohort.parquet", index=False)
    co = pd.read_parquet(root / "data" / "cohort.parquet")
    rows = co.assign(_p=co.SEQN.astype(int)).set_index("_p")
    sw = pd.DataFrame([{"profile": int(p), "edit": "baseline", "annotated": False,
                        "text": to_json(to_state(rows.loc[p].to_dict()))} for p in rows.index])
    sw.to_parquet(root / "data" / "states_wide.parquet", index=False)
    pd.DataFrame({"model": "jev", "profile": sw.profile, "edit": "baseline", "tokens_in": [len(t) // 4 for t in sw.text]}
                 ).to_parquet(root / "results" / "wide" / "calls.parquet", index=False)
    return root


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network call attempted in a test")
    monkeypatch.setattr(clients, "_post", refuse)


def lab(name, value, sex=None):
    f = F[name]
    bands = f["bands"][sex] if f.get("by_sex") else f["bands"]
    return D.band(value, bands, f.get("scale"), f.get("decimals"))


# --------------------------------------------------------------------------------------------- config and sources
def test_every_field_has_a_source_and_a_catch_all():
    assert D.CFG["question"]["source"] and D.CFG["state"]["source"] and D.CFG["analysis"]["source"]
    for name, f in F.items():
        assert f.get("source"), name
        lists = [f["categories"]] if f.get("kind") == "blood_pressure" else (
            list(f["bands"].values()) if f.get("by_sex") else [f["bands"]])
        for bs in lists:
            assert not ({"below", "at_most", "systolic_at_least"} & set(bs[-1])), f"{name}: last band has a cut point"
            assert len({b["label"] for b in bs}) == len(bs), name


def test_every_original_field_is_kept_or_banded():
    """Every key the original serializer can write is either kept as is or replaced by a band."""
    kept = set(D.CFG["state"]["kept_fields"])
    banded = {s for f in F.values() for s in f["from"]}
    assert {k for k, *_ in NUMERIC_FIELDS} == banded
    assert not kept & banded


def test_question_is_the_original_primary_question():
    req = D.make_request({"sex": "male"})
    orig = clients.JevClient(None, "nhanes").request("any record text", clients.PROMPTS["statements"]["nhanes"])
    assert req["questions"] == orig["questions"] == {"death": {"type": "noul", "instructions":
                                                               "This person will die from any cause within ten years of this examination."}}
    assert req["model"] == orig["model"] == clients.MODELS["jev"]["slug"]
    assert req["provider"] == orig["provider"] and req["provider"]["allow_fallbacks"] is False
    assert req["state"] == {"sex": "male"} and isinstance(req["state"], dict)


def test_config_sha256_ignores_line_ends(tmp_path):
    """The config's sha256 (lowercase hex) is taken over the text with LF line ends."""
    sha = D.config_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    import hashlib
    raw = (ROOT / "config" / "docuse_risk.yaml").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert sha == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    crlf = tmp_path / "crlf.yaml"                                         # a Windows checkout gives the same sha256
    crlf.write_bytes(raw.replace("\n", "\r\n").encode("utf-8"))
    assert D.config_sha256(crlf) == sha
    changed = tmp_path / "changed.yaml"
    changed.write_text(raw + "# one more line\n", encoding="utf-8")
    assert D.config_sha256(changed) != sha


# --------------------------------------------------------------------------------------------- band edges
def test_generic_band_edges():
    """For every configured cut point: a value exactly on a `below` cut goes to the next band, a value exactly on an
    `at_most` cut stays in its band, and the value one display step below a `below` cut stays in its band."""
    dec = {k: d for k, _, _, d in NUMERIC_FIELDS}
    for name, f in F.items():
        if f.get("kind"):
            continue
        step = 10 ** -dec[f["from"][0]] / (f.get("scale") or 1.0)
        for sex, bs in (f["bands"].items() if f.get("by_sex") else [(None, f["bands"])]):
            for i, b in enumerate(bs[:-1]):
                if "below" in b:
                    x = b["below"] / (f.get("scale") or 1.0)
                    assert lab(name, x, sex) == bs[i + 1]["label"], (name, sex, x)
                    assert lab(name, round(x - step, 6), sex) == b["label"], (name, sex, x - step)
                else:
                    x = b["at_most"] / (f.get("scale") or 1.0)
                    assert lab(name, x, sex) == b["label"], (name, sex, x)
                    assert lab(name, round(x + step, 6), sex) == bs[i + 1]["label"], (name, sex, x + step)


BELOW, WITHIN, ABOVE = "below the reference range", "within the reference range", "above the reference range"


@pytest.mark.parametrize("name,value,sex,expected", [
    # five-year age groups
    ("age", 40, None, "40 to 44 years"), ("age", 44, None, "40 to 44 years"), ("age", 45, None, "45 to 49 years"),
    ("age", 75, None, "75 to 79 years"), ("age", 79, None, "75 to 79 years"),
    # WHO BMI: 18.5, 25.0, 30.0, 35.0, 40.0 open the next class
    ("body_mass_index", 18.4, None, "underweight"), ("body_mass_index", 18.5, None, "normal weight"),
    ("body_mass_index", 24.9, None, "normal weight"), ("body_mass_index", 25.0, None, "pre-obesity"),
    ("body_mass_index", 29.9, None, "pre-obesity"), ("body_mass_index", 30.0, None, "obesity class I"),
    ("body_mass_index", 34.9, None, "obesity class I"), ("body_mass_index", 35.0, None, "obesity class II"),
    ("body_mass_index", 39.9, None, "obesity class II"), ("body_mass_index", 40.0, None, "obesity class III"),
    # NCEP ATP III waist: abdominal obesity > 102 cm men, > 88 cm women (at most = no abdominal obesity)
    ("waist_circumference", 101.9, "male", "no abdominal obesity"), ("waist_circumference", 102.0, "male", "no abdominal obesity"),
    ("waist_circumference", 102.1, "male", "abdominal obesity"),
    ("waist_circumference", 87.9, "female", "no abdominal obesity"), ("waist_circumference", 88.0, "female", "no abdominal obesity"),
    ("waist_circumference", 88.1, "female", "abdominal obesity"),
    ("waist_circumference", 95.0, "male", "no abdominal obesity"), ("waist_circumference", 95.0, "female", "abdominal obesity"),
    # ADA HbA1c 5.7 and 6.5
    ("hba1c", 5.6, None, "below the prediabetes range"), ("hba1c", 5.7, None, "prediabetes range"),
    ("hba1c", 6.4, None, "prediabetes range"), ("hba1c", 6.5, None, "diabetes range"),
    # ADA random plasma glucose threshold >= 200 mg/dL
    ("glucose", 199, None, "below the ADA random-glucose threshold for diabetes"),
    ("glucose", 200, None, "at or above the ADA random-glucose threshold for diabetes"),
    ("glucose", 126, None, "below the ADA random-glucose threshold for diabetes"),
    # KDIGO albuminuria: A1 < 30; A2 30-300 inclusive; A3 > 300
    ("albuminuria", 29.9, None, "KDIGO A1, normal to mildly increased"),
    ("albuminuria", 30.0, None, "KDIGO A2, moderately increased"),
    ("albuminuria", 299.9, None, "KDIGO A2, moderately increased"),
    ("albuminuria", 300.0, None, "KDIGO A2, moderately increased"),
    ("albuminuria", 300.1, None, "KDIGO A3, severely increased"),
    # AHA/CDC CRP in mg/L from the record's mg/dL: 1.0 and 3.0 mg/L are average
    ("crp", 0.09, None, "low relative cardiovascular risk"), ("crp", 0.10, None, "average relative cardiovascular risk"),
    ("crp", 0.29, None, "average relative cardiovascular risk"), ("crp", 0.30, None, "average relative cardiovascular risk"),
    ("crp", 0.31, None, "high relative cardiovascular risk"), ("crp", 1.00, None, "high relative cardiovascular risk"),
    ("crp", 1.01, None, "markedly raised, suggesting infection or inflammation"),
    # NCEP total cholesterol 200 and 240; HDL low < 40, high >= 60 (both sexes)
    ("total_cholesterol", 199, None, "desirable"), ("total_cholesterol", 200, None, "borderline high"),
    ("total_cholesterol", 239, None, "borderline high"), ("total_cholesterol", 240, None, "high"),
    ("hdl", 39, "male", "low"), ("hdl", 40, "male", "neither low nor high"), ("hdl", 39, "female", "low"),
    ("hdl", 40, "female", "neither low nor high"), ("hdl", 45, "female", "neither low nor high"),
    ("hdl", 59, "female", "neither low nor high"), ("hdl", 60, "female", "high"), ("hdl", 59, "male", "neither low nor high"),
    ("hdl", 60, "male", "high"),
    # medications: none; 1 to 4; 5 or more (polypharmacy)
    ("medications", 0, None, "none"), ("medications", 1, None, "1 to 4"), ("medications", 4, None, "1 to 4"),
    ("medications", 5, None, "5 or more (polypharmacy)"), ("medications", 20, None, "5 or more (polypharmacy)"),
    # income: NCHS percent-of-poverty categories, ratio cut points 1, 2, 4
    ("income", 0.99, None, "below 100% of the poverty threshold"), ("income", 1.00, None, "100% to 199% of the poverty threshold"),
    ("income", 1.99, None, "100% to 199% of the poverty threshold"), ("income", 2.00, None, "200% to 399% of the poverty threshold"),
    ("income", 3.99, None, "200% to 399% of the poverty threshold"), ("income", 4.00, None, "400% or more of the poverty threshold"),
    # ABIM Laboratory Test Reference Ranges, January 2026: both limits within the range
    ("albumin", 3.4, None, BELOW), ("albumin", 3.5, None, WITHIN), ("albumin", 5.4, None, WITHIN),
    ("albumin", 5.5, None, WITHIN), ("albumin", 5.6, None, ABOVE),
    ("white_cells", 3.9, None, BELOW), ("white_cells", 4.0, None, WITHIN), ("white_cells", 10.9, None, WITHIN),
    ("white_cells", 11.0, None, WITHIN), ("white_cells", 11.1, None, ABOVE),
    ("lymphocytes", 29.9, None, BELOW), ("lymphocytes", 30.0, None, WITHIN), ("lymphocytes", 44.9, None, WITHIN),
    ("lymphocytes", 45.0, None, WITHIN), ("lymphocytes", 45.1, None, ABOVE),
    ("mcv", 79.9, None, BELOW), ("mcv", 80.0, None, WITHIN), ("mcv", 97.9, None, WITHIN),
    ("mcv", 98.0, None, WITHIN), ("mcv", 98.1, None, ABOVE),
    ("rdw", 8.9, None, BELOW), ("rdw", 9.0, None, WITHIN), ("rdw", 14.4, None, WITHIN),
    ("rdw", 14.5, None, WITHIN), ("rdw", 14.6, None, ABOVE),
    ("alkaline_phosphatase", 29, None, BELOW), ("alkaline_phosphatase", 30, None, WITHIN),
    ("alkaline_phosphatase", 119, None, WITHIN), ("alkaline_phosphatase", 120, None, WITHIN),
    ("alkaline_phosphatase", 121, None, ABOVE),
])
def test_band_edges_as_the_sources_state(name, value, sex, expected):
    """Values on each cut point and one display step on either side, at the record's precision (serialize decimals),
    land where the source line says."""
    assert lab(name, value, sex) == expected


def test_band_edges_through_a_record():
    """The same edges reached from an original record (band_field), so the record's value format is exercised."""
    rec = lambda k, v, sex="male": {"sex": sex, k: {"value": v, "unit": "x"}}
    assert D.band_field("glucose", rec("serum_glucose_random", 200)) == "at or above the ADA random-glucose threshold for diabetes"
    assert D.band_field("waist_circumference", rec("waist_circumference", 102.0)) == "no abdominal obesity"
    assert D.band_field("waist_circumference", rec("waist_circumference", 88.1, "female")) == "abdominal obesity"
    assert D.band_field("albuminuria", rec("urine_albumin_creatinine_ratio", 300.0)) == "KDIGO A2, moderately increased"
    assert D.band_field("albumin", rec("serum_albumin", 5.5)) == WITHIN
    assert D.band_field("medications", {"prescription_medicines_past_30_days": 5}) == "5 or more (polypharmacy)"
    assert D.band_field("medications", {"prescription_medicines_past_30_days": 1}) == "1 to 4"


def test_income_top_code_is_the_top_band():
    orig = {"sex": "female", "family_income_to_poverty_ratio": "5 or more"}
    assert D.band_field("income", orig) == "400% or more of the poverty threshold"


@pytest.mark.parametrize("sbp,dbp,expected", [
    (119, 79, "normal"), (120, 79, "elevated"), (129, 79, "elevated"), (130, 79, "stage 1"), (119, 80, "stage 1"),
    (129, 80, "stage 1"), (139, 89, "stage 1"), (140, 89, "stage 2"), (139, 90, "stage 2"), (110, 95, "stage 2"),
    (150, 70, "stage 2"), (125, 85, "stage 1"), (0, 0, "normal"),
])
def test_blood_pressure_aha_acc_2017(sbp, dbp, expected):
    """Table 6 of the 2017 guideline; the higher category of the two readings."""
    assert D.bp_category(sbp, dbp).startswith({"normal": "normal", "elevated": "elevated",
                                               "stage 1": "stage 1", "stage 2": "stage 2"}[expected])


def test_blood_pressure_missing_reading_leaves_the_field_out():
    assert D.band_field("blood_pressure", {"systolic_blood_pressure": {"value": 150, "unit": "mmHg"}}) is None


# --------------------------------------------------------------------------------------------- partition
def intervals(bands):
    """The interval of each band under the first-match rule: (low, low closed, high, high closed, label)."""
    out, lo, lo_closed = [], -math.inf, False
    for b in bands:
        if "below" in b:
            hi, hi_closed = float(b["below"]), False
        elif "at_most" in b:
            hi, hi_closed = float(b["at_most"]), True
        else:
            hi, hi_closed = math.inf, False
        out.append((lo, lo_closed, hi, hi_closed, b["label"]))
        lo, lo_closed = hi, not hi_closed
    return out


def band_lists():
    for name, f in F.items():
        if f.get("kind") == "blood_pressure":
            continue
        for sex, bs in (f["bands"].items() if f.get("by_sex") else [(None, f["bands"])]):
            yield name, sex, bs


def test_labels_are_distinct_within_each_field_and_sex():
    for name, sex, bs in band_lists():
        labs = [b["label"] for b in bs]
        assert len(set(labs)) == len(labs), (name, sex)
        assert all(isinstance(l, str) and l.strip() == l and l for l in labs), (name, sex)
    cats = [c["label"] for c in F["blood_pressure"]["categories"]]
    assert len(set(cats)) == len(cats)
    keys = [f["key"] for f in F.values()]
    assert len(set(keys)) == len(keys) and not set(keys) & set(D.CFG["state"]["kept_fields"])


def test_bands_partition_the_real_line():
    """Each band has at most one cut point, the last none; the intervals run from -inf to +inf, each starting where the
    previous one ends with the edge in exactly one of them, and none is empty. A dense grid then lands in exactly one
    interval, the one band() returns."""
    for name, sex, bs in band_lists():
        assert all(len({"below", "at_most"} & set(b)) <= 1 for b in bs), (name, sex)
        assert not {"below", "at_most"} & set(bs[-1]), (name, sex)
        iv = intervals(bs)
        assert iv[0][0] == -math.inf and iv[-1][2] == math.inf, (name, sex)
        for (lo, lc, hi, hc, lab_), nxt in zip(iv, iv[1:] + [None]):
            assert lo < hi or (lo == hi and lc and hc), (name, sex, lab_, "empty")
            if nxt:
                assert nxt[0] == hi and nxt[1] == (not hc), (name, sex, lab_, "gap or overlap")
        f = F[name]
        cuts = sorted({x for x in (b.get("below", b.get("at_most")) for b in bs) if x is not None})
        grid = sorted(set(np.concatenate([np.linspace(cuts[0] - 10, cuts[-1] + 10, 2001), cuts,
                                          np.nextafter(cuts, -np.inf), np.nextafter(cuts, np.inf)])))
        for v in grid:
            inside = [l for lo, lc, hi, hc, l in iv if (v > lo or (lc and v == lo)) and (v < hi or (hc and v == hi))]
            assert len(inside) == 1, (name, sex, v, inside)
            assert D.band(v, bs) == inside[0], (name, sex, v)


def test_blood_pressure_categories_cover_every_reading_once():
    cats = F["blood_pressure"]["categories"]
    s_cuts = [c["systolic_at_least"] for c in cats[:-1]]
    assert s_cuts == sorted(s_cuts, reverse=True) and not {"systolic_at_least", "diastolic_at_least"} & set(cats[-1])
    labs = [c["label"] for c in cats]
    for sbp in range(60, 240):
        for dbp in range(30, 140, 3):
            got = D.bp_category(sbp, dbp)
            reached = [c["label"] for c in cats[:-1] if sbp >= c["systolic_at_least"]
                       or (c["diastolic_at_least"] is not None and dbp >= c["diastolic_at_least"])]
            assert got == (reached[0] if reached else labs[-1])            # the highest category either reading reaches


# --------------------------------------------------------------------------------------------- CKD-EPI 2021
def egfr_table_form(scr, age, sex):
    """The equation in its four-branch table form (Inker et al., NEJM 2021), written independently of the code."""
    if sex == "female":
        return 142 * 1.012 * (scr / 0.7) ** (-0.241 if scr <= 0.7 else -1.200) * 0.9938 ** age
    return 142 * (scr / 0.9) ** (-0.302 if scr <= 0.9 else -1.200) * 0.9938 ** age


@pytest.mark.parametrize("scr,age,sex,expected", [
    # worked values, mL/min/1.73 m2 rounded to an integer as calculators report them; hand-computed from the
    # published equation
    (1.0, 50, "female", 69), (1.0, 60, "male", 86), (0.8, 40, "male", 115), (0.6, 70, "female", 97),
    (2.5, 75, "male", 26), (0.7, 50, "female", 105), (0.9, 50, "male", 104), (1.2, 65, "female", 50),
])
def test_egfr_worked_examples(scr, age, sex, expected):
    got = D.egfr_ckd_epi_2021(scr, age, sex)
    assert round(got) == expected
    assert math.isclose(got, egfr_table_form(scr, age, sex), rel_tol=1e-12)


def test_ckd_epi_2021_constants_are_the_published_ones():
    """142, kappa 0.7 / 0.9, alpha -0.241 / -0.302, -1.200 above kappa, 0.9938^age, 1.012 female (kidney.org)."""
    eq = F["kidney_function"]["equation"]
    assert eq["constant"] == 142 and eq["age_base"] == 0.9938 and eq["exponent_above"] == -1.200
    assert eq["female"] == {"kappa": 0.7, "alpha": -0.241, "factor": 1.012}
    assert eq["male"] == {"kappa": 0.9, "alpha": -0.302, "factor": 1.0}


def test_egfr_matches_table_form_on_a_grid():
    for sex in ("female", "male"):
        for scr in np.round(np.arange(0.3, 12.0, 0.01), 2):
            for age in (40, 55, 79):
                assert math.isclose(D.egfr_ckd_epi_2021(scr, age, sex), egfr_table_form(scr, age, sex), rel_tol=1e-12)


@pytest.mark.parametrize("egfr,g", [(90.0, "G1"), (89.999, "G2"), (60.0, "G2"), (59.999, "G3a"), (45.0, "G3a"),
                                    (44.999, "G3b"), (30.0, "G3b"), (29.999, "G4"), (15.0, "G4"), (14.999, "G5")])
def test_kdigo_gfr_category_edges(egfr, g):
    assert D.band(egfr, F["kidney_function"]["bands"]).startswith(f"KDIGO {g},")


def test_egfr_category_from_a_record():
    orig = {"age": {"value": 60, "unit": "years"}, "sex": "male", "serum_creatinine": {"value": 1.0, "unit": "mg/dL"}}
    assert D.band_field("kidney_function", orig).startswith("KDIGO G2,")          # 86


# --------------------------------------------------------------------------------------------- records
def test_banded_records_round_trip(synth_root):
    """For every wide record: the original text rebuilds from the cohort row (build_states checks it), banding the
    stored text gives the same state as banding the cohort row, the state survives JSON unchanged and in order, holds
    no number, and a field missing from the original record is missing from the banded one."""
    st = D.build_states(D.wide_profiles(synth_root), root=synth_root)
    assert len(st) == N_WIDE
    source_of = {s: f["key"] for f in F.values() for s in f["from"]}
    for r in st.itertuples(index=False):
        orig = json.loads(r.original_text)
        s = r.state
        assert D.banded_state(orig) == s
        assert list(json.loads(json.dumps(s)).items()) == list(s.items())
        assert not D._numbers_in(s)
        for k, key in source_of.items():
            if k not in orig and k != "diastolic_blood_pressure":
                assert key not in s
            if k in orig:
                assert key in s
        for k in D.CFG["state"]["kept_fields"]:
            assert s.get(k) == orig.get(k)


def test_missing_counts_carry_over(synth_root):
    co = pd.read_parquet(synth_root / "data" / "cohort.parquet")
    prof = D.wide_profiles(synth_root)
    rows = co.assign(_p=co.SEQN.astype(int)).set_index("_p").loc[prof]
    st = D.build_states(prof, root=synth_root)
    have = lambda key: int(st.state.map(lambda s: key in s).sum())
    assert have("body_mass_index_category") == int(rows.bmi.notna().sum())
    assert have("waist_circumference_category") == int(rows.waist_cm.notna().sum())
    assert have("blood_pressure_category") == int((rows.sbp_mmhg.notna() & rows.dbp_mmhg.notna()).sum())
    assert have("urine_albumin_creatinine_ratio_category") == int(rows.uacr_mg_g.notna().sum())
    assert have("family_income_to_poverty_ratio_category") == int(rows.income_poverty_ratio.notna().sum())


def test_dry_run_builds_every_valid_request(synth_root):
    st = D.build_states(D.wide_profiles(synth_root), root=synth_root)
    r = D.dry_run(st, root=synth_root)
    assert r["n_requests"] == N_WIDE and r["n_invalid"] == 0
    assert r["model"] == clients.MODELS["jev"]["slug"] and r["provider"] == {"allow_fallbacks": False}
    assert 0 < r["usd_list_estimated"] < 2.0
    calls = D.plan(st)
    assert sorted(c.profile for c in calls) == D.wide_profiles(synth_root)
    assert {(c.model, c.edit, c.repeat, c.variant) for c in calls} == {("jev", "baseline", 0, "documented")}


@pytest.fixture(scope="module")
def wide_states(synth_root):
    return D.build_states(D.wide_profiles(synth_root), root=synth_root)


def test_template_is_the_original_request_with_the_state_slot():
    t = D.template()
    orig = D.request_body(clients.JevClient(None, "nhanes").request("x", clients.PROMPTS["statements"]["nhanes"]))
    assert t == {**orig, "state": "{state}"} and list(t) == list(orig)
    assert t["model"] == clients.MODELS["jev"]["slug"] and t["provider"]["allow_fallbacks"] is False


def test_every_request_is_the_template_with_its_state(wide_states):
    """Every request: with the state replaced by "{state}", the body is the template exactly (keys, order and values);
    only the state varies. The cache key _repeat is 0 everywhere and never sent."""
    t = D.template()
    tt = json.dumps(t, ensure_ascii=False)
    assert len(wide_states) == N_WIDE
    for r in wide_states.itertuples(index=False):
        req = D.make_request(r.state)
        assert req["_repeat"] == 0
        body = D.request_body(req)
        assert body["state"] is r.state
        swapped = {**body, "state": "{state}"}
        assert swapped == t and json.dumps(swapped, ensure_ascii=False) == tt, r.profile


def test_every_state_has_the_fixed_keys_and_labels_only(wide_states):
    """Keys in the fixed order and only from the allowed list; each banded value exactly one of its field's labels (of
    the record's sex for a by_sex field); kept values equal to the original record's; no original numeric key. The
    state's text is nothing but that fixed wording (allowed keys, config labels, kept original wording) and JSON
    punctuation, so no banded field's original value appears in it except where a config label's own words hold the
    same digits (an age group or a medication count range)."""
    order = D.state_key_order()
    fk = D.field_of_key()
    numeric = {k for k, *_ in NUMERIC_FIELDS}
    kept = set(D.CFG["state"]["kept_fields"])
    assert set(order) == kept | set(fk) and not numeric & set(order)
    for r in wide_states.itertuples(index=False):
        st, orig = r.state, json.loads(r.original_text)
        assert list(st) == [k for k in order if k in st], r.profile
        assert not numeric & set(st), r.profile
        for k, v in st.items():
            if k in fk:
                labs = D.label_lists(fk[k])
                assert v in labs.get(st.get("sex"), labs.get(None, [])), (r.profile, k, v)
            else:
                assert k in kept and v == orig[k], (r.profile, k)
        text = json.dumps(st, ensure_ascii=False)
        fixed = [k for k in order if k in st] + [kk for v in st.values() if isinstance(v, dict) for kk in v]
        fixed += [v for v in st.values() if isinstance(v, str)] + [vv for v in st.values() if isinstance(v, dict)
                                                                   for vv in v.values()]
        rest = text
        for w in sorted(set(fixed), key=len, reverse=True):
            rest = rest.replace(json.dumps(w, ensure_ascii=False), "")
        assert re.fullmatch(r"[\s{}\[\]:,]*", rest), (r.profile, rest[:200])   # nothing but the fixed wording
        assert '"value"' not in text and '"unit"' not in text, r.profile
        for f in F.values():
            for src in f["from"]:
                if src in orig:
                    v = orig[src]["value"] if isinstance(orig[src], dict) else orig[src]
                    assert not re.search(r"(?<![\d.])" + re.escape(str(v)) + r"(?![\d.])", rest), (r.profile, src, v)
        assert not D._numbers_in(st), r.profile


def test_band_distribution_counts_every_record():
    st = small_states(60)
    dist = D.band_distribution(st)
    for name, d in dist.items():
        assert sum(d["counts"].values()) + d["missing"] == 60, name
        assert list(d["counts"]) == list(dict.fromkeys(l for ls in D.label_lists(name).values() for l in ls))


def test_validate_request_catches_problems():
    good = D.make_request({"age_group": "40 to 44 years", "sex": "male"})
    assert D.validate_request(good) == []
    assert any("order" in e for e in D.validate_request(D.make_request({"sex": "male", "age_group": "40 to 44 years"})))
    bad = D.make_request({"sex": "male", "age": 42})
    assert any("unexpected" in e for e in D.validate_request(bad)) and any("numbers" in e for e in D.validate_request(bad))
    assert D.validate_request({**good, "provider": {"allow_fallbacks": True}})
    assert D.validate_request(D.make_request({"age_group": "40 to 45"}))


# --------------------------------------------------------------------------------------------- synthetic responses
def synthetic_response(p, tokens=600):
    return {"id": "gen-synthetic", "model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe",
            "answers": {"death": {"noul": p}}, "usage": {"input_tokens": tokens, "output_tokens": 20}}


def small_states(n=120):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rec = {"age": 40 + i % 40, "sex": "male" if i % 2 else "female", "bmi": float(rng.uniform(17, 45)),
               "creatinine_mg_dl": float(rng.uniform(0.5, 3)), "sbp_mmhg": float(rng.uniform(95, 190)),
               "dbp_mmhg": float(rng.uniform(50, 110)), "rx_count": int(i % 8), "race_ethnicity": "Mexican American"}
        orig = to_state(rec)
        rows.append({"profile": 100000 + i, "state": D.banded_state(orig), "original_text": json.dumps(orig)})
    return pd.DataFrame(rows)


def wide_columns():
    """The columns runner.execute writes (those of results/wide/calls.parquet)."""
    return ["model", "profile", "edit", "annotated", "repeat", "variant", "p", "valid", "provider", "model_reported",
            "error", "raw_path", "route", "parse", "finish_reason", "response_id", "confidence", "latency_s", "p_alive",
            "p_bands", "tokens_in", "tokens_out", "band_probabilities"]


def test_calls_table_from_stored_synthetic_responses(tmp_path):
    st = small_states(30)
    store = clients.RawStore(tmp_path / "store")
    for r in st.itertuples(index=False):
        if r.profile % 3:
            req = D.make_request(r.state)
            assert store.put(req, synthetic_response(round((r.profile % 97) / 100, 2)), {"route": "standard", "latency_s": 1.5})
    df = D.calls_table(st, store, tmp_path / "calls.parquet")
    assert list(df.columns) == wide_columns()
    assert len(df) == 30 and set(df.model) == {"jev"} and set(df.variant) == {"documented"}
    assert set(df.edit) == {"baseline"} and set(df["repeat"]) == {0} and not df.annotated.any()
    ans = df[df.profile % 3 != 0]
    assert ans.valid.all() and (ans.parse == "noul").all() and (ans.provider == "TypeSafe").all()
    assert np.allclose(ans.p, (ans.profile % 97) / 100)
    miss = df[df.profile % 3 == 0]
    assert (~miss.valid).all() and miss.error.str.contains("no stored answer").all()
    assert pd.read_parquet(tmp_path / "calls.parquet").shape == df.shape


def test_run_sends_through_the_repository_transport(tmp_path, monkeypatch):
    """The live path end to end with a synthetic endpoint: clients.JevClient._send posts to OpenRouter's Jev path with
    the state as a JSON object and allow_fallbacks false; answers land in the raw store verbatim and are never
    overwritten; a second run resends nothing."""
    sent = []

    def fake_post(url, headers, body, retries=6):
        sent.append((url, body))
        return synthetic_response(0.25)
    monkeypatch.setattr(clients, "_post", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic")
    st = small_states(120)
    store = clients.RawStore(tmp_path / "store")
    s = D.run(st, store, tmp_path / "progress", stop_usd=5.0, workers=4, chunk_records=50)
    assert s["stopped"] is None and s["calls"] == 120 and s["chunks"] == 3
    uniq = {clients._sha(D.make_request(x)) for x in st.state}
    assert len(sent) == len(uniq)
    url, body = sent[0]
    assert url == clients.OPENROUTER + clients.MODELS["jev"]["openrouter_path"]
    assert isinstance(body["state"], dict) and body["provider"] == {"allow_fallbacks": False}
    assert not any(k.startswith("_") for k in body)
    files = sorted((tmp_path / "store").glob("*.json"))
    before = {f: f.read_bytes() for f in files}
    D.run(st, store, tmp_path / "progress", stop_usd=5.0, workers=4, chunk_records=50)
    assert len(sent) == len(uniq)
    assert {f: f.read_bytes() for f in sorted((tmp_path / "store").glob("*.json"))} == before
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["response"] == synthetic_response(0.25)


def test_run_stops_on_spend_and_on_402(tmp_path, monkeypatch):
    monkeypatch.setattr(clients, "_post", lambda url, headers, body, retries=6: synthetic_response(0.3, tokens=10 ** 6))
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic")
    st = small_states(120)
    s = D.run(st, clients.RawStore(tmp_path / "a"), tmp_path / "p", stop_usd=0.5, workers=2, chunk_records=50)
    assert s["stopped"] and "run stop" in s["stopped"] and s["chunks_run"] == 1

    def refuse(url, headers, body, retries=6):
        raise RuntimeError("HTTP 402 Payment Required")
    monkeypatch.setattr(clients, "_post", refuse)
    s = D.run(st, clients.RawStore(tmp_path / "b"), tmp_path / "p", stop_usd=5.0, workers=2, chunk_records=50)
    assert s["stopped"] and "402" in s["stopped"] and s["chunks_run"] == 1


# --------------------------------------------------------------------------------------------- analysis
def test_fixed_margin_outcomes_from_the_stored_bounds():
    blk = {"n_records": 3, "focal_vs_llms": {"critical_value": 2.0, "pairs": {
        "a - b": {"diff": 0.0, "ci_simultaneous": [-0.01, 0.004], "noninferiority_margin": 0.003, "noninferior": False},
        "a - c": {"diff": 0.0, "ci_simultaneous": [-0.01, 0.006], "noninferiority_margin": 0.007, "noninferior": True}}},
        "systems": {}}
    out = D.fixed_margin_outcomes(blk, 0.005)
    ab, ac = out["focal_vs_llms"]["pairs"]["a - b"], out["focal_vs_llms"]["pairs"]["a - c"]
    assert ab["noninferior_fixed_margin"] is True and ab["noninferior"] is False and ab["noninferiority_margin"] == 0.003
    assert ac["noninferior_fixed_margin"] is False and ac["noninferior"] is True and ac["noninferiority_margin_fixed"] == 0.005


def test_scored_set_is_the_original_set_intersected():
    r = D.scored_set([1, 2, 3, 5, 8], [2, 3, 4, 8, 9])
    assert (r["original"], r["scored"], r["dropped"], r["dropped_ids"], r["_ids"]) == (5, 3, 2, [1, 5], [2, 3, 8])


# --------------------------------------------------------------------------------------------- script live paths
def run_script(monkeypatch, tmp_path, account_usage=10.0, root=None):
    """scripts/docuse/run_risk.py with its outputs under tmp_path, the account reads and the Jev endpoint
    synthetic; with root, its records read from that repository root."""
    from jevity import categorical as K
    spec = importlib.util.spec_from_file_location("run_risk", ROOT / "scripts" / "docuse" / "run_risk.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    monkeypatch.setattr(m, "RESULTS", tmp_path / "results"); monkeypatch.setattr(m, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(m, "STORE", tmp_path / "runs" / "wide_docuse")
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(K, "load_keys", lambda arm: None)
    monkeypatch.setattr(K, "openrouter_account", lambda get=None: {"key_usage": account_usage, "key_limit": None,
                                                                   "key_limit_remaining": None, "balance": 100.0})
    monkeypatch.setattr(K, "manifest_files", lambda roots, extra=(): [])
    monkeypatch.setattr(clients, "_post", lambda url, headers, body, retries=6: synthetic_response(0.4, 900))
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic")
    if root is not None:                  # the script, build_states and dry_run read the records under root
        (root / "scripts").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "scripts" / "timing_sample.py", root / "scripts" / "timing_sample.py")
        monkeypatch.setattr(m, "ROOT", root)
        for f in ("build_states", "dry_run"):
            monkeypatch.setattr(D, f, functools.partial(getattr(D, f), root=root))
    return m


def test_budget_check_refuses_above_the_project_stop(monkeypatch, tmp_path, synth_root):
    assert D.CFG["budget"]["project_stop_usd"] is None                  # null: no project stop
    monkeypatch.setitem(D.CFG["budget"], "project_stop_usd", 1000)
    m = run_script(monkeypatch, tmp_path, account_usage=1000.0, root=synth_root)
    with pytest.raises(SystemExit, match="budget check failed"):
        m.preflight(D.build_states(D.wide_profiles(synth_root)[:50]), True)
    assert not (tmp_path / "runs" / "wide_docuse").exists()


def test_timing_mode(monkeypatch, tmp_path, synth_root):
    m = run_script(monkeypatch, tmp_path, root=synth_root)
    m.timing(5, 0.0, "20260101", True)
    out = json.loads((m.RESULTS / "timing_sample.json").read_text())
    v = out["systems"]["jev_documented"]
    assert out["n_records"] == 5 and v["n"] == 5 and v["usable"] == 5
    assert v["usd_list_total"] == pytest.approx(5 * 900 * 0.042 / 1e6)
    assert len(list((tmp_path / "runs" / "wide_docuse_timing_20260101").glob("*.json"))) == 5

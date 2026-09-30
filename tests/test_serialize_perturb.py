import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevity import serialize as S, perturb as P
from synthetic import make


def test_state_roundtrip_and_renderings():
    rec = make(50).iloc[0].to_dict()
    st = S.to_state(rec)
    txt = S.to_json(st)
    assert '"hba1c"' in txt and json.loads(txt)["hba1c"]["unit"] == "%"
    sh = S.shuffle_fields(st, 3)
    assert set(sh) == set(st) and list(sh) != list(st)
    cu = S.convert_units(st)
    assert cu["serum_creatinine"]["unit"] == "umol/L" and cu["hba1c"] == st["hba1c"]
    prose = S.to_prose(st)
    assert "HbA1c" in prose and str(st["hba1c"]["value"]) in prose
    ann = S.annotate(st)
    assert "category" in ann["hba1c"]


def test_edits_change_exactly_one_column():
    df = make(300)
    rec = df[df.race_ethnicity == "Non-Hispanic White"].iloc[0].to_dict()
    e = P.apply_record_edit(rec, "L1_race_nhw_to_nhb")
    assert e.applicable and e.record["race_ethnicity"] == "Non-Hispanic Black"
    changed = [k for k in rec if rec[k] != e.record[k]]
    assert changed == ["race_ethnicity"]
    e2 = P.apply_record_edit(e.record, "L1_race_nhw_to_nhb")
    assert not e2.applicable
    old = P.apply_record_edit({**rec, "age": 78}, "C6_age_plus5")
    assert not old.applicable
    c = P.apply_record_edit(rec, "C2b_creatinine_x2")
    assert abs(c.record["creatinine_mg_dl"] - 2 * rec["creatinine_mg_dl"]) < 1e-9


def test_support_rule():
    df = make(2000)
    chk = P.SupportChecker(df[df.split == "fit"])
    rec = df[df.split == "eval"].iloc[0].to_dict()
    assert chk.supported(rec)
    far = {**rec, "creatinine_mg_dl": 40.0, "hba1c_pct": 19.0, "albumin_g_dl": 0.5}
    assert not chk.supported(far)


def test_support_rule_label_positivity_and_range():
    df = make(3000)
    fit = df[df.split == "fit"].copy()
    black = fit.race_ethnicity == "Non-Hispanic Black"
    fit.loc[black, "age"] = 45                      # the fitting set holds no older Black adults
    chk = P.SupportChecker(fit)
    rec = {**df[(df.split == "eval")].iloc[0].to_dict(), "race_ethnicity": "Non-Hispanic White", "age": 78}
    assert chk.supported(rec)
    ed = P.apply_record_edit(rec, "L1_race_nhw_to_nhb")
    assert not chk.supported(ed.record, changed="race_ethnicity")        # no comparable Black records
    young = P.apply_record_edit({**rec, "age": 45}, "L1_race_nhw_to_nhb")
    assert chk.supported(young.record, changed="race_ethnicity") == chk.supported(young.record)
    hi = {**rec, "hba1c_pct": float(fit.hba1c_pct.max()) + 1}
    assert not chk.supported(hi, changed="hba1c_pct")


def test_same_information_and_context():
    rec = {**make(50).iloc[0].to_dict(), "rx_names": "METFORMIN; LISINOPRIL", "rx_count": 2}
    txt = S.to_json(S.to_state(rec))
    assert "METFORMIN" not in txt and '"prescription_medicines_past_30_days": 2' in txt     # names never reach a system
    assert S.EXAMINATION_CONTEXT in txt


def test_rare_category_never_supported():
    import pandas as pd
    df = make(2000)
    df["sex"] = "male"
    df.loc[df.index[:10], "sex"] = "female"                  # below min_stratum: no propensity model can be fitted
    ch = P.SupportChecker(df)
    rec = df[df.sex == "male"].iloc[0].to_dict()
    e = P.apply_record_edit(rec, "L5a_sex_male_to_female")
    assert e.applicable and not ch.category_rule(e.record, "sex") and not ch.supported(e.record, changed="sex")
    fb = P.SupportChecker(df, exclude_strata=("sex",))       # the sex fallback drops rule (b) only
    assert fb.category_rule(e.record, "sex") is False and fb.distance(e.record) >= 0


def test_yaml_codes_are_text():
    import yaml, pytest
    root = Path(__file__).resolve().parents[1] / "config"
    for f in ("nhanes_variables.yaml", "nhanes3.yaml"):
        y = yaml.safe_load((root / f).read_text())
        for name, spec in y["variables"].items():
            for v in (spec.get("codes") or {}).values():
                assert isinstance(v, str), (f, name, v)
    from jevity.nhanes import check_categorical_domains
    import pandas as pd
    with pytest.raises(AssertionError):
        check_categorical_domains(pd.DataFrame({"x": [True, False]}), {"x": {"codes": {1: "yes", 2: "no"}}})

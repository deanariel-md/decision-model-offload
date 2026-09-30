"""eICU arm on synthetic eICU-shaped tables: cohort rules, demo exclusion from the fitting set, serialisation, edits,
support and reference contrasts. No real data."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import eicu as E, eicu_reference as ER


def _tables(n: int, seed: int, id0: int, hospitals: list[int]) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(id0, id0 + n)
    upid = np.array([f"{id0 // 1000}-{i // 2}" for i in range(n)])          # two stays per patient
    age = rng.integers(20, 95, n).astype(object); age[rng.random(n) < 0.03] = "> 89"; age[0] = "15"
    lact = rng.lognormal(0.5, 0.6, n); cre = rng.lognormal(0.1, 0.5, n); mapv = rng.normal(70, 15, n)
    eyes, motor, verbal = rng.integers(1, 5, n), rng.integers(1, 7, n), rng.integers(1, 6, n)
    lp = -3 + 0.03 * (np.array([90 if a == "> 89" else int(a) for a in age]) - 60) + 0.5 * np.log(lact) + 0.6 * np.log(cre) \
         - 0.02 * (mapv - 70) - 0.15 * (eyes + motor + verbal - 10)
    dead = rng.random(n) < 1 / (1 + np.exp(-lp))
    offs = rng.integers(300, 20000, n); offs[1] = 600
    patient = pd.DataFrame({
        "patientunitstayid": ids, "patienthealthsystemstayid": ids // 2, "uniquepid": upid,
        "gender": rng.choice(["Male", "Female"], n), "age": age,
        "ethnicity": rng.choice(["Caucasian", "African American", "Hispanic", "Asian", ""], n, p=[.7, .15, .08, .04, .03]),
        "hospitalid": rng.choice(hospitals, n), "apacheadmissiondx": rng.choice(["Sepsis, pulmonary", "CHF, congestive heart failure",
                                                                                "Overdose, sedatives", "CABG alone"], n),
        "hospitaldischargeyear": 2014 + (ids % 2), "hospitaldischargeoffset": offs,
        "hospitaldischargestatus": np.where(dead, "Expired", "Alive"), "unittype": rng.choice(["MICU", "Med-Surg ICU"], n),
        "unitadmitsource": rng.choice(["Emergency Department", "Floor", "Operating Room"], n), "unitvisitnumber": 1,
        "unitdischargeoffset": np.minimum(offs, rng.integers(200, 8000, n)), "unitdischargestatus": "Alive"})
    patient.loc[1, "hospitaldischargestatus"] = "Expired"            # died at 10 h: excluded
    res = pd.DataFrame({"patientunitstayid": np.r_[ids, ids], "apacheversion": ["IV"] * n + ["IVa"] * n,
                        "predictedhospitalmortality": np.r_[rng.random(n), 1 / (1 + np.exp(-lp))]})
    aps = pd.DataFrame({"patientunitstayid": ids, "intubated": rng.integers(0, 2, n), "vent": rng.integers(0, 2, n),
                        "dialysis": 0, "eyes": eyes, "motor": motor, "verbal": verbal, "meds": (rng.random(n) < .1).astype(int),
                        "urine": rng.normal(1500, 600, n).clip(0), "wbc": rng.lognormal(2.3, .4, n), "temperature": rng.normal(37, .8, n),
                        "respiratoryrate": rng.normal(22, 6, n), "sodium": rng.normal(139, 4, n), "heartrate": rng.normal(100, 20, n),
                        "meanbp": mapv, "ph": rng.normal(7.35, .08, n), "hematocrit": rng.normal(33, 6, n), "creatinine": cre,
                        "albumin": np.where(rng.random(n) < .4, -1, rng.normal(3, .6, n)), "pao2": rng.normal(90, 30, n),
                        "pco2": rng.normal(40, 8, n), "bun": rng.lognormal(3, .5, n), "glucose": rng.lognormal(5, .3, n),
                        "bilirubin": rng.lognormal(-.2, .7, n), "fio2": rng.choice([21, 40, 60, 100], n)})
    pv = pd.DataFrame({"patientunitstayid": ids, "electivesurgery": rng.choice([0, 1, np.nan], n),
                       **{c: rng.integers(0, 2, n) * (rng.random(n) < .1) for c in ["aids", "cirrhosis", "hepaticfailure",
                          "immunosuppression", "leukemia", "lymphoma", "metastaticcancer", "diabetes"]}})
    lab = pd.DataFrame({"patientunitstayid": np.r_[ids, ids], "labresultoffset": np.r_[np.full(n, 120), np.full(n, 3000)],
                        "labname": ["lactate"] * (2 * n), "labresult": np.r_[lact, lact + 5]})
    return {"patient": patient, "apachePatientResult": res, "apacheApsVar": aps, "apachePredVar": pv, "lab": lab}


def _write(folder: Path, tabs: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for k, t in tabs.items():
        t.to_csv(folder / f"{k}.csv.gz", index=False)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eicu")
    demo = _tables(700, 1, 100000, [1, 2])
    other = _tables(6000, 2, 200000, [3, 4, 5, 6])
    _write(tmp / "demo", demo)
    _write(tmp / "full", {k: pd.concat([demo[k], other[k]], ignore_index=True) for k in demo})
    out = tmp / "out"
    d = E.build_demo(tmp / "demo", out)
    f = E.build_full(tmp / "full", tmp / "demo", out)
    return d, f, out


def test_cohort_rules(built):
    d, f, _ = built
    assert 100000 not in set(d.SEQN)                                  # age 15 excluded
    assert 100001 not in set(d.SEQN)                                  # died within 24 h excluded
    assert set(d.split) == {"pilot", "eval"} and (d.split == "pilot").sum() == E.CFG["splits"]["n_pilot"]
    assert d.lactate_mmol_l.max() < 20                                # only first-day lactate (the day-3 value is +5)
    assert d.albumin_g_dl.min() > 0                                   # APACHE -1 codes are missing, not values
    assert d.gcs_total.dropna().between(3, 15).all()
    assert (d.age == 90).any() and d.age.min() >= 18


def test_demo_excluded_from_fit(built):
    d, f, _ = built
    assert not set(f.SEQN) & set(E.demo_ids(Path(built[2]).parent / "demo")["stays"])
    assert not set(f.hospitalid) & {1, 2}


def test_state_and_edits(built):
    d, _, _ = built
    rec = d[(d.ethnicity == "Caucasian") & (d.sex == "male") & d.lactate_mmol_l.notna()].iloc[0].to_dict()
    st = E.to_state(rec)
    assert list(st)[:4] == ["record_type", "source", "age", "sex"]
    assert "SEQN" not in E.render(rec) and "death" not in E.render(rec)      # no id or outcome in the text
    base = E.render(rec)
    for name in E.CFG["confirmatory_family"]:
        ed = E.apply_edit(rec, name)
        if not ed.applicable:
            continue
        new = E.render(ed.record)
        changed = [(a, b) for a, b in zip(base.splitlines(), new.splitlines()) if a != b]
        assert len(changed) == 1, (name, changed)                       # exactly one field differs
    assert E.render(rec, "I1_field_order", seed=3) != base and sorted(E.render(rec, "I1_field_order", seed=3)) == sorted(base)
    hi = dict(rec, map_mmhg=120.0)
    assert not E.apply_edit(hi, "EC3_map_minus15").applicable         # hypertensive worst MAP not edited
    old = dict(rec, age=90.0)
    assert E.to_state(old)["age"] == "90 or older" and not E.apply_edit(old, "EC5_age_plus5").applicable


def test_reference_contrasts_and_states(built, monkeypatch):
    d, f, out = built
    import joblib
    ref = ER.fit_spline_logit(f)
    lg = ER.fit_lgbm(f)
    joblib.dump(ref, out / "reference_spline_logit.joblib"); joblib.dump(lg, out / "reference_lightgbm.joblib")
    ev = d[d.split == "eval"]
    B = ev.head(200); X = B.copy(); X["lactate_mmol_l"] = X["lactate_mmol_l"] + 2
    c = ref.contrast(B, X)
    assert c["d_logit"].dropna().mean() > 0                           # lactate raises risk in the synthetic truth
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import make_eicu_states as M
    monkeypatch.setitem(E.CFG, "out_dir", str(out))
    inst, st = M.build("eval", root=Path("/"))
    assert {"baseline", "I1_field_order"} <= set(st.edit)
    assert set(inst.edit) == {e for k in ("irrelevant", "clinical", "label") for e in E.CFG["edits"][k]}
    lac = inst[(inst.edit == "EC1_lactate_plus2") & inst.applicable]
    assert (lac.d_logit_spline_logit != 0).any()
    assert json.loads(inst.iloc[0]["diff"])

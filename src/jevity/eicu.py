"""eICU records for the eICU replication. One builder for two sources (config/eicu.yaml `sources`):
  demo  eICU-CRD Demo v2.0.1, open access (Open Database License): the only eICU records sent to a model.
  full  eICU-CRD v2.0, credentialed access: read on a local machine to fit the outcome references; never sent anywhere.
A record is the first ICU stay of a patient's first hospital stay: adult, APACHE IVa result present, alive 24 h after
unit admission. Outcome: death before hospital discharge. Physiology: APACHE IV worst first-day values (apacheApsVar);
lactate: the highest value in the first ICU day (lab). The profile key is `SEQN` (= patientunitstayid) so the NHANES
tools that join on SEQN work unchanged."""
from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from jevity.perturb import Edit

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "config" / "eicu.yaml").read_text(encoding="utf-8"))
OUTCOME = "death_hosp"
ID = "patientunitstayid"

APS = {"intubated": "intubated", "vent": "vent", "dialysis": "dialysis", "urine": "urine_ml", "wbc": "wbc_10e3_ul",
       "temperature": "temperature_c", "respiratoryrate": "resp_rate", "sodium": "sodium_mmol_l", "heartrate": "heart_rate",
       "meanbp": "map_mmhg", "ph": "ph", "hematocrit": "hematocrit_pct", "creatinine": "creatinine_mg_dl",
       "albumin": "albumin_g_dl", "pao2": "pao2_mmhg", "pco2": "paco2_mmhg", "bun": "bun_mg_dl", "glucose": "glucose_mg_dl",
       "bilirubin": "bilirubin_mg_dl", "fio2": "fio2_pct"}
PRED = {"electivesurgery": "elective_surgery", "aids": "aids", "cirrhosis": "cirrhosis", "hepaticfailure": "hepatic_failure",
        "immunosuppression": "immunosuppression", "leukemia": "leukemia", "lymphoma": "lymphoma",
        "metastaticcancer": "metastatic_cancer", "diabetes": "diabetes"}
YES_NO = ["vent", "intubated", "dialysis", "elective_surgery", "aids", "cirrhosis", "hepatic_failure", "immunosuppression",
          "leukemia", "lymphoma", "metastatic_cancer", "diabetes"]


# ----------------------------------------------------------------------------------------------------- reading
def _find(src: Path, name: str) -> Path:
    want = {f"{name.lower()}.csv.gz", f"{name.lower()}.csv"}
    for p in sorted(Path(src).rglob("*")):
        if p.name.lower() in want:
            return p
    raise FileNotFoundError(f"{name}.csv(.gz) not found under {src}")


def read_table(src: Path, name: str, cols: list[str] | None = None, chunk_filter=None) -> pd.DataFrame:
    """Column names lower-cased. `chunk_filter(df) -> df` reads large tables (lab) in chunks."""
    path = _find(src, name)
    usecols = (lambda c: c.lower() in set(cols)) if cols else None
    if chunk_filter is None:
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        return df
    parts = []
    for ch in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=2_000_000):
        ch.columns = [c.lower() for c in ch.columns]
        parts.append(chunk_filter(ch))
    return pd.concat(parts, ignore_index=True)


def parse_age(x: Any) -> float:
    s = str(x).strip()
    if s in ("", "nan", "None"):
        return math.nan
    if s.startswith(">"):
        return 90.0                          # eICU top-codes ages above 89 as "> 89"
    try:
        return float(s)
    except ValueError:
        return math.nan


# ----------------------------------------------------------------------------------------------------- records
def load_records(src: Path) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Eligible records with the outcome and every serialised field, plus the cohort flow."""
    c = CFG["cohort"]
    flow: list[tuple[str, int]] = []
    pt = read_table(src, "patient")
    flow.append(("ICU stays", len(pt)))
    pt["age"] = pt["age"].map(parse_age)
    pt = pt[pt["age"] >= c["age_min"]]
    flow.append((f"Age {c['age_min']} or older", len(pt)))
    if c.get("first_unit_stay", True):
        pt = pt.sort_values(["uniquepid", "hospitaldischargeyear", "patienthealthsystemstayid", "unitvisitnumber"])
        first_hosp = pt.groupby("uniquepid")["patienthealthsystemstayid"].transform("first")
        pt = pt[(pt["patienthealthsystemstayid"] == first_hosp) & (pt["unitvisitnumber"] == 1)]
        flow.append(("First ICU stay of the first hospital stay per patient", len(pt)))
    res = read_table(src, "apachePatientResult", ["patientunitstayid", "apacheversion", "predictedhospitalmortality"])
    res = res[res["apacheversion"].astype(str).str.strip() == c["apache_version"]]
    res = res.drop_duplicates(ID).rename(columns={"predictedhospitalmortality": "apache_iva_pred"})
    res["apache_iva_pred"] = pd.to_numeric(res["apache_iva_pred"], errors="coerce").where(lambda v: v >= 0)
    pt = pt.merge(res[[ID, "apache_iva_pred"]], on=ID, how="inner")
    flow.append((f"APACHE {c['apache_version']} result present", len(pt)))
    status = pt["hospitaldischargestatus"].astype(str).str.strip()
    pt = pt[status.isin(["Alive", "Expired"])].copy()
    flow.append(("Hospital discharge status recorded", len(pt)))
    died = pt["hospitaldischargestatus"].astype(str).str.strip() == "Expired"
    early = died & (pd.to_numeric(pt["hospitaldischargeoffset"], errors="coerce") < c["alive_minutes"])
    early |= (pt["unitdischargestatus"].astype(str).str.strip() == "Expired") & \
             (pd.to_numeric(pt["unitdischargeoffset"], errors="coerce") < c["alive_minutes"])
    pt = pt[~early].copy()
    pt[OUTCOME] = (pt["hospitaldischargestatus"].astype(str).str.strip() == "Expired").astype(int)
    flow.append((f"Alive {c['alive_minutes'] // 60} h after ICU admission; deaths before discharge = "
                 f"{int(pt[OUTCOME].sum())}", len(pt)))

    aps = read_table(src, "apacheApsVar")
    aps = aps.drop_duplicates(ID)
    for col in list(APS) + ["eyes", "motor", "verbal", "meds"]:
        aps[col] = pd.to_numeric(aps[col], errors="coerce")
        aps.loc[aps[col] < 0, col] = np.nan                # APACHE codes missing as -1
    comp = aps[["eyes", "motor", "verbal"]]
    ok = comp.notna().all(axis=1) & (comp >= 1).all(axis=1) & (aps["meds"].fillna(0) != 1)
    aps["gcs_total"] = comp.sum(axis=1).where(ok)         # not assessable when sedated or a component is missing
    aps = aps.rename(columns=APS)[[ID, "gcs_total"] + list(APS.values())]
    pv = read_table(src, "apachePredVar", [ID] + list(PRED)).drop_duplicates(ID).rename(columns=PRED)
    lo, hi = c["lactate_window_minutes"]
    lab = read_table(src, "lab", [ID, "labresultoffset", "labname", "labresult"],
                     chunk_filter=lambda d: d[d["labname"].astype(str).str.strip().str.lower() == "lactate"])
    lab = lab[pd.to_numeric(lab["labresultoffset"], errors="coerce").between(lo, hi)]
    lac = lab.assign(v=pd.to_numeric(lab["labresult"], errors="coerce")).groupby(ID)["v"].max().rename("lactate_mmol_l")
    df = pt.merge(aps, on=ID, how="left").merge(pv, on=ID, how="left").merge(lac, left_on=ID, right_index=True, how="left")
    for col in YES_NO:
        v = pd.to_numeric(df[col], errors="coerce")
        df[col] = np.where(v == 1, "yes", np.where(v == 0, "no", None))
    df["elective_surgery"] = df["elective_surgery"].where(df["elective_surgery"].notna(), "no")   # APACHE: blank = not elective surgery
    df["sex"] = df["gender"].astype(str).str.strip().str.lower().where(lambda s: s.isin(["male", "female"]))
    eth = df["ethnicity"].astype(str).str.strip()
    df["ethnicity"] = eth.where(~eth.isin(["", "nan", "None"]), "Other/Unknown")
    dx = df["apacheadmissiondx"].astype(str).str.strip()
    df["admission_dx"] = dx.where(~dx.isin(["", "nan", "None"]))
    src_ = df["unitadmitsource"].astype(str).str.strip()
    df["admit_source"] = src_.where(~src_.isin(["", "nan", "None"]))
    df["unit_type"] = df["unittype"].astype(str).str.strip()
    df["SEQN"] = df[ID].astype(int)
    keep = ["SEQN", ID, "uniquepid", "patienthealthsystemstayid", "hospitalid", "hospitaldischargeyear", OUTCOME,
            "apache_iva_pred", "sex", "ethnicity", "admission_dx", "admit_source", "unit_type"] + \
           [col for _, col, _, _ in CFG["numeric"]] + YES_NO
    return df[list(dict.fromkeys(keep))].reset_index(drop=True), flow


def build_demo(src: Path | None = None, out_dir: Path | None = None) -> pd.DataFrame:
    src = Path(src or ROOT / CFG["sources"]["demo"])
    out = Path(out_dir or ROOT / CFG["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    df, flow = load_records(src)
    rng = np.random.default_rng(CFG["seed"])
    df["split"] = "eval"
    pick = rng.permutation(len(df))[:CFG["splits"]["n_pilot"]]
    df.iloc[pick, df.columns.get_loc("split")] = "pilot"
    df["source"] = "demo"
    df.to_parquet(out / "demo_cohort.parquet", index=False)
    _write_flow(out / "demo_flow.md", "eICU-CRD Demo v2.0.1 (open access)", flow, df)
    return df


def demo_ids(src: Path | None = None) -> dict[str, set]:
    """Every stay, patient and hospital in the demo (all stays, not only eligible ones), for fit-set exclusion."""
    pt = read_table(Path(src or ROOT / CFG["sources"]["demo"]), "patient", [ID, "uniquepid", "hospitalid"])
    return {"stays": set(pt[ID].astype(int)), "patients": set(pt["uniquepid"].astype(str)),
            "hospitals": set(pt["hospitalid"].astype(int))}


def build_full(src: Path | None = None, demo_src: Path | None = None, out_dir: Path | None = None) -> pd.DataFrame:
    """Credentialed full eICU on the local machine: the reference fitting set, demo stays, patients and hospitals
    removed (config exclude_from_fit). Reports how many demo ids were found, as the id-compatibility check."""
    src = Path(src or ROOT / CFG["sources"]["full"])
    out = Path(out_dir or ROOT / CFG["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    df, flow = load_records(src)
    ids = demo_ids(demo_src)
    found = {"demo stays found in full eICU": int(df[ID].isin(ids["stays"]).sum()),
             "demo patients found": int(df["uniquepid"].astype(str).isin(ids["patients"]).sum()),
             "demo hospitals found": int(df["hospitalid"].astype(int).isin(ids["hospitals"]).sum())}
    ex = CFG["exclude_from_fit"]
    drop = pd.Series(False, index=df.index)
    if ex.get("demo_stays"):
        drop |= df[ID].isin(ids["stays"])
    if ex.get("demo_patients"):
        drop |= df["uniquepid"].astype(str).isin(ids["patients"])
    if ex.get("demo_hospitals"):
        drop |= df["hospitalid"].astype(int).isin(ids["hospitals"])
    what = ", ".join(k.replace("demo_", "demo ") for k in ("demo_stays", "demo_patients", "demo_hospitals") if ex.get(k))
    flow.append((f"Excluded: {what} (config exclude_from_fit)", int(drop.sum())))
    df = df[~drop].copy()
    df["split"] = "fit"; df["source"] = "full"
    flow.append((f"Reference fitting set; deaths = {int(df[OUTCOME].sum())}", len(df)))
    df.to_parquet(out / "full_fit.parquet", index=False)
    _write_flow(out / "full_flow.md", "eICU-CRD v2.0 (credentialed; local only)", flow, df, extra=found)
    print(json.dumps(found, indent=1))
    if found["demo stays found in full eICU"] == 0:
        print("WARNING: no demo stay id found in the full database; exclusion relies on hospital ids. Check before fitting.")
    return df


def _write_flow(path: Path, title: str, flow, df: pd.DataFrame, extra: dict | None = None) -> None:
    lines = [f"# Cohort flow: {title} (generated by scripts/build_eicu.py)", "", "| Step | n |", "|---|---:|"]
    lines += [f"| {s} | {n:,} |" for s, n in flow]
    if extra:
        lines += ["", "| Check | n |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in extra.items()]
    lines += ["", "## Non-missing share of each field", "", "| Field | share |", "|---|---:|"]
    for col in [c for _, c, _, _ in CFG["numeric"]] + ["sex", "ethnicity", "admission_dx", "admit_source", "unit_type"]:
        lines.append(f"| {col} | {df[col].notna().mean():.3f} |")
    lines += ["", "## Ethnicity", "", df["ethnicity"].value_counts().to_markdown()]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------- serialisation
def _num(x: Any, nd: int):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return round(v, nd) if nd > 0 else int(round(v))


def _present(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _rank(k: str) -> int:
    first, last = CFG["field_order_first"], CFG["field_order_last"]
    nums = [n[0] for n in CFG["numeric"]]
    if k in first:
        return first.index(k)
    if k in last:
        return 1000 + last.index(k)
    return 100 + nums.index(k) if k in nums else 500


def to_state(rec: dict[str, Any]) -> "OrderedDict[str, Any]":
    s: OrderedDict[str, Any] = OrderedDict()
    s["record_type"] = CFG["record"]["record_type"]
    s["source"] = CFG["record"]["source"]
    for key, col in CFG["categorical"]:
        if _present(rec.get(col)):
            s[key] = rec[col]
    for key, col in CFG["flags"]:
        if _present(rec.get(col)):
            s[key] = rec[col]
    for key, col, unit, nd in CFG["numeric"]:
        v = _num(rec.get(col), nd)
        if v is None:
            continue
        if key == "age" and v >= 90:
            s[key] = "90 or older"                       # eICU top code; the age edit requires age <= 84
            continue
        s[key] = {"value": v, "unit": unit} if unit else v
    conds = OrderedDict((key, rec[col]) for key, col in CFG["conditions"] if _present(rec.get(col)))
    if conds:
        s["chronic_conditions"] = conds
    return OrderedDict((k, s[k]) for k in sorted(s, key=_rank))


def to_json(state: dict) -> str:
    return json.dumps(state, ensure_ascii=False, indent=1)


def render(rec: dict[str, Any], irrelevant: str | None = None, seed: int = 0) -> str:
    state = to_state(rec)
    if irrelevant is None:
        return to_json(state)
    op = CFG["edits"]["irrelevant"][irrelevant]["op"]
    if op == "shuffle_fields":
        keys = list(state); random.Random(seed).shuffle(keys)
        return to_json(OrderedDict((k, state[k]) for k in keys))
    raise ValueError(op)


# ----------------------------------------------------------------------------------------------------- edits
def klass_of(name: str) -> str:
    for k in ("irrelevant", "clinical", "label"):
        if name in CFG["edits"][k]:
            return k
    raise KeyError(name)


def apply_edit(rec: dict[str, Any], name: str) -> Edit:
    klass = klass_of(name)
    spec = CFG["edits"][klass][name]
    col = spec["column"]
    base = rec.get(col)
    if not _present(base):
        return Edit(name, klass, dict(rec), applicable=False, reason=f"{col} missing")
    for k, v in spec.get("require", {}).items():
        if rec.get(k) != v:
            return Edit(name, klass, dict(rec), applicable=False, reason=f"requires {k}={v}")
    for k, v in spec.get("require_max", {}).items():
        if not _present(rec.get(k)) or float(rec[k]) > v:
            return Edit(name, klass, dict(rec), applicable=False, reason=f"requires {k}<={v}")
    for k, v in spec.get("require_min", {}).items():
        if not _present(rec.get(k)) or float(rec[k]) < v:
            return Edit(name, klass, dict(rec), applicable=False, reason=f"requires {k}>={v}")
    op, val = spec["op"], spec["value"]
    new = {"add": lambda: float(base) + val, "multiply": lambda: float(base) * val, "set": lambda: val}[op]()
    out = dict(rec)
    out[col] = new
    return Edit(name, klass, out, diff={"column": col, "from": base, "to": new, "op": op})


def support_checker(fit_df: pd.DataFrame):
    from jevity.perturb import SupportChecker
    return SupportChecker(fit_df, cfg=CFG["support"])

"""Record -> serialised state. The baseline is a JSON object with named fields, values as recorded with
units, no derived scores and no interpretation. Every alternative rendering used by the irrelevant class
lives here so that the diff between baseline and edit is exactly one thing."""
from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from typing import Any

# (state key, cohort column, unit, decimals)
NUMERIC_FIELDS = [
    ("age", "age", "years", 0),
    ("body_mass_index", "bmi", "kg/m2", 1),
    ("waist_circumference", "waist_cm", "cm", 1),
    ("systolic_blood_pressure", "sbp_mmhg", "mmHg", 0),
    ("diastolic_blood_pressure", "dbp_mmhg", "mmHg", 0),
    ("family_income_to_poverty_ratio", "income_poverty_ratio", "", 2),
    ("prescription_medicines_past_30_days", "rx_count", "", 0),
    ("serum_albumin", "albumin_g_dl", "g/dL", 1),
    ("serum_creatinine", "creatinine_mg_dl", "mg/dL", 2),
    ("serum_glucose_random", "glucose_mg_dl", "mg/dL", 0),
    ("c_reactive_protein", "crp_mg_dl", "mg/dL", 2),
    ("lymphocyte_percent", "lymphocyte_pct", "%", 1),
    ("mean_corpuscular_volume", "mcv_fl", "fL", 1),
    ("red_cell_distribution_width", "rdw_pct", "%", 1),
    ("alkaline_phosphatase", "alk_phos_u_l", "U/L", 0),
    ("white_blood_cell_count", "wbc_10e3_ul", "x10^3/uL", 1),
    ("hba1c", "hba1c_pct", "%", 1),
    ("total_cholesterol", "total_chol_mg_dl", "mg/dL", 0),
    ("hdl_cholesterol", "hdl_mg_dl", "mg/dL", 0),
    ("urine_albumin_creatinine_ratio", "uacr_mg_g", "mg/g", 1),
]
EXAMINATION_CONTEXT = "United States national health survey, 1999-2008"

CATEGORICAL_FIELDS = [
    ("sex", "sex"), ("race_ethnicity", "race_ethnicity"), ("education", "education"), ("has_health_insurance", "insured"),
    ("smoking_status", "smoking"), ("self_rated_health", "self_rated_health"),
]
CONDITION_FIELDS = [
    ("diabetes", "diabetes"), ("congestive_heart_failure", "chf"), ("coronary_heart_disease", "chd"),
    ("myocardial_infarction", "mi"), ("stroke", "stroke"), ("emphysema", "emphysema"),
    ("chronic_bronchitis", "chronic_bronchitis"), ("cancer", "cancer"),
]

# unit conversions for the irrelevant class (factor, new unit, decimals)
UNIT_CONVERSIONS = {
    "serum_glucose_random": (0.0555, "mmol/L", 2),
    "serum_creatinine": (88.4, "umol/L", 0),
    "total_cholesterol": (0.02586, "mmol/L", 2),
    "hdl_cholesterol": (0.02586, "mmol/L", 2),
    "serum_albumin": (10.0, "g/L", 0),
    "c_reactive_protein": (10.0, "mg/L", 1),
}


def _num(x: Any, nd: int) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), nd) if nd > 0 else int(round(float(x)))


# Baseline field order: context, age and sex, clinical findings, then social fields.
FIELD_ORDER = ["record_type", "source", "age", "sex", "race_ethnicity", "smoking_status", "self_rated_health",
               "body_mass_index", "waist_circumference", "systolic_blood_pressure", "diastolic_blood_pressure"]
FIELD_ORDER_LAST = ["prescription_medicines_past_30_days", "doctor_diagnosed_conditions", "education",
                    "family_income_to_poverty_ratio", "has_health_insurance"]


def _field_rank(k: str) -> int:
    if k in FIELD_ORDER:
        return FIELD_ORDER.index(k)
    if k in FIELD_ORDER_LAST:
        return 1000 + FIELD_ORDER_LAST.index(k)
    return 100 + [f[0] for f in NUMERIC_FIELDS].index(k) if k in [f[0] for f in NUMERIC_FIELDS] else 500


def to_state(rec: dict[str, Any]) -> "OrderedDict[str, Any]":
    """Baseline state. Keys are stable and ordered; a value is {'value': x, 'unit': u} for numbers."""
    s: OrderedDict[str, Any] = OrderedDict()
    s["record_type"] = "adult health examination and interview record"
    s["source"] = EXAMINATION_CONTEXT           # the target population and era, identical for every system
    for key, col in CATEGORICAL_FIELDS:
        v = rec.get(col)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            s[key] = v
    for key, col, unit, nd in NUMERIC_FIELDS:
        v = _num(rec.get(col), nd)
        if v is not None:
            s[key] = {"value": v, "unit": unit} if unit else v
    conds = OrderedDict()
    for key, col in CONDITION_FIELDS:
        v = rec.get(col)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            conds[key] = v
    if conds:
        s["doctor_diagnosed_conditions"] = conds       # NHANES: ever told by a doctor
    inc = s.get("family_income_to_poverty_ratio")
    if isinstance(inc, (int, float)) and inc >= 5:
        s["family_income_to_poverty_ratio"] = "5 or more"   # NHANES top-codes the ratio at 5
    s = OrderedDict((k, s[k]) for k in sorted(s, key=_field_rank))
    # Medication names are never serialised: the outcome references see the count only, and every system must see the
    # same information as the reference. rx_names stays in the cohort for description.
    return s


def to_json(state: dict) -> str:
    return json.dumps(state, ensure_ascii=False, indent=1)


def shuffle_fields(state: dict, seed: int) -> "OrderedDict[str, Any]":
    keys = list(state.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    return OrderedDict((k, state[k]) for k in keys)


def convert_units(state: dict) -> "OrderedDict[str, Any]":
    out = OrderedDict()
    for k, v in state.items():
        if k in UNIT_CONVERSIONS and isinstance(v, dict) and "value" in v:
            f, unit, nd = UNIT_CONVERSIONS[k]
            out[k] = {"value": round(v["value"] * f, nd) if nd else int(round(v["value"] * f)), "unit": unit}
        else:
            out[k] = v
    return out


PROSE_LABELS = {
    "age": "Age", "body_mass_index": "Body mass index", "waist_circumference": "Waist circumference",
    "systolic_blood_pressure": "Systolic blood pressure", "diastolic_blood_pressure": "Diastolic blood pressure",
    "family_income_to_poverty_ratio": "Family income to poverty ratio",
    "prescription_medicines_past_30_days": "Prescription medicines taken in the past 30 days",
    "serum_albumin": "Serum albumin", "serum_creatinine": "Serum creatinine", "serum_glucose_random": "Serum glucose (random)",
    "c_reactive_protein": "C-reactive protein", "lymphocyte_percent": "Lymphocytes",
    "mean_corpuscular_volume": "Mean corpuscular volume", "red_cell_distribution_width": "Red cell distribution width",
    "alkaline_phosphatase": "Alkaline phosphatase", "white_blood_cell_count": "White blood cell count", "hba1c": "HbA1c",
    "total_cholesterol": "Total cholesterol", "hdl_cholesterol": "HDL cholesterol",
    "urine_albumin_creatinine_ratio": "Urine albumin-to-creatinine ratio", "sex": "Sex",
    "race_ethnicity": "Race and ethnicity", "education": "Education", "has_health_insurance": "Health insurance",
    "smoking_status": "Smoking", "self_rated_health": "Self-rated health",
    "doctor_diagnosed_conditions": "Conditions ever diagnosed by a doctor",
    "source": "Source",
}


def to_prose(state: dict) -> str:
    """Prose rendering with identical content, no interpretation (the I3 formatting edit)."""
    parts = []
    for k, v in state.items():
        label = PROSE_LABELS.get(k, k.replace("_", " ").capitalize())
        if k == "record_type":
            parts.append(f"This is an {v}.")
        elif isinstance(v, dict) and "value" in v:
            cat = f" ({v['category']})" if "category" in v else ""
            sep = "" if v["unit"] == "%" else " "
            parts.append(f"{label}: {v['value']}{sep}{v['unit']}{cat}.")
        elif isinstance(v, dict):
            inner = "; ".join(f"{kk.replace('_', ' ')}, {vv}" for kk, vv in v.items())
            parts.append(f"{label}: {inner}.")
        elif isinstance(v, list):
            parts.append(f"{label}: {', '.join(v) if v else 'none'}.")
        else:
            parts.append(f"{label}: {v}.")
    return " ".join(parts)


# annotation variant (a subset of evaluation records): deterministic clinical categories after labs.
ANNOTATIONS = {
    "hba1c": [(5.7, "within usual range"), (6.5, "prediabetic range"), (float("inf"), "diabetic range")],
    "serum_creatinine": [(1.2, "within usual range"), (2.0, "raised"), (float("inf"), "markedly raised")],
    "serum_albumin": [(3.5, "below usual range"), (float("inf"), "within usual range")],
    "systolic_blood_pressure": [(120, "within usual range"), (140, "raised"), (float("inf"), "markedly raised")],
    "c_reactive_protein": [(0.3, "within usual range"), (1.0, "moderately raised"), (float("inf"), "markedly raised")],
}


def annotate(state: dict) -> "OrderedDict[str, Any]":
    out = OrderedDict()
    for k, v in state.items():
        if k in ANNOTATIONS and isinstance(v, dict) and "value" in v:
            label = next(lbl for cut, lbl in ANNOTATIONS[k] if v["value"] < cut)
            out[k] = {"value": v["value"], "unit": v["unit"], "category": label}
        else:
            out[k] = v
    return out


def drop_fields(state: dict, fields) -> "OrderedDict[str, Any]":
    """M2 rendering: the same state without the named fields (e.g. race_ethnicity)."""
    return OrderedDict((k, v) for k, v in state.items() if k not in set(fields))

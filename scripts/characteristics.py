"""Characteristics of the fitting and evaluation sets:
every field the models were shown and the references use, the survey cycle and the ten-year outcome, unweighted.
Counts and percentages for categories (missing values as their own row where any occur), median and IQR for
measurements (with the number missing). Open NHANES data only (data/cohort.parquet); no model output.
Writes results/eval/characteristics.md and results/eval/characteristics.json.
  python scripts/characteristics.py"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SETS = (("fit", "Fitting set"), ("eval", "Evaluation set"))

# (column, label, kind, levels shown): kind "num" (median, IQR), "cat" (every level), "yes" (the levels other than "no")
ROWS = [
    ("cycle", "Survey cycle", "cat", ["1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008"]),
    ("age", "Age, years", "num", None),
    ("sex", "Sex", "cat", ["male", "female"]),
    ("race_ethnicity", "Race and ethnicity", "cat", ["Non-Hispanic White", "Non-Hispanic Black", "Mexican American",
                                                    "Other Hispanic", "Other race including multiracial"]),
    ("education", "Education", "cat", ["less than 9th grade", "9th to 11th grade", "high school graduate or GED",
                                       "some college", "college graduate or above"]),
    ("income_poverty_ratio", "Family income to poverty ratio", "num", None),
    ("insured", "Health insurance", "cat", ["yes", "no"]),
    ("smoking", "Smoking", "cat", ["never", "former", "current"]),
    ("self_rated_health", "Self-rated health", "cat", ["excellent", "very good", "good", "fair", "poor"]),
    ("bmi", "Body mass index, kg/m2", "num", None),
    ("waist_cm", "Waist circumference, cm", "num", None),
    ("sbp_mmhg", "Systolic blood pressure, mmHg", "num", None),
    ("dbp_mmhg", "Diastolic blood pressure, mmHg", "num", None),
    ("rx_count", "Prescription medicines in the past 30 days", "num", None),
    ("diabetes", "Diabetes", "yes", ["yes", "borderline"]),
    ("chf", "Congestive heart failure", "yes", ["yes"]),
    ("chd", "Coronary heart disease", "yes", ["yes"]),
    ("mi", "Myocardial infarction", "yes", ["yes"]),
    ("stroke", "Stroke", "yes", ["yes"]),
    ("emphysema", "Emphysema", "yes", ["yes"]),
    ("chronic_bronchitis", "Chronic bronchitis", "yes", ["yes"]),
    ("cancer", "Cancer", "yes", ["yes"]),
    ("albumin_g_dl", "Serum albumin, g/dL", "num", None),
    ("creatinine_mg_dl", "Serum creatinine, mg/dL", "num", None),
    ("glucose_mg_dl", "Serum glucose (random), mg/dL", "num", None),
    ("crp_mg_dl", "C-reactive protein, mg/dL", "num", None),
    ("lymphocyte_pct", "Lymphocytes, %", "num", None),
    ("mcv_fl", "Mean corpuscular volume, fL", "num", None),
    ("rdw_pct", "Red cell distribution width, %", "num", None),
    ("alk_phos_u_l", "Alkaline phosphatase, U/L", "num", None),
    ("wbc_10e3_ul", "White blood cell count, x10^3/uL", "num", None),
    ("hba1c_pct", "HbA1c, %", "num", None),
    ("total_chol_mg_dl", "Total cholesterol, mg/dL", "num", None),
    ("hdl_mg_dl", "HDL cholesterol, mg/dL", "num", None),
    ("uacr_mg_g", "Urine albumin to creatinine ratio, mg/g", "num", None),
    ("death_10y", "Died within ten years", "yes", [1]),
]
DECIMALS = {"income_poverty_ratio": 2, "creatinine_mg_dl": 2, "crp_mg_dl": 2, "albumin_g_dl": 1, "bmi": 1,
            "waist_cm": 1, "lymphocyte_pct": 1, "mcv_fl": 1, "rdw_pct": 1, "wbc_10e3_ul": 1, "hba1c_pct": 1,
            "uacr_mg_g": 1}


def summarise(cohort: pd.DataFrame) -> dict:
    """Per set and row: n, and per level count and share, or median, quartiles and missing."""
    out = {"sets": {}, "rows": []}
    for key, lab in SETS:
        out["sets"][key] = {"label": lab, "n": int((cohort.split == key).sum())}
    for col, lab, kind, levels in ROWS:
        r = {"column": col, "label": lab, "kind": kind, "sets": {}}
        for key, _ in SETS:
            x = cohort.loc[cohort.split == key, col]
            n = len(x)
            if kind == "num":
                v = x.dropna().astype(float)
                q = np.quantile(v, [0.25, 0.5, 0.75]) if len(v) else [None] * 3
                r["sets"][key] = {"median": float(q[1]), "q1": float(q[0]), "q3": float(q[2]),
                                  "missing": int(x.isna().sum())}
            else:
                lv = {str(l): {"n": int((x == l).sum()), "share": float((x == l).mean())} for l in levels}
                seen = set(levels) | ({"no"} if kind == "yes" else set()) | ({0} if col == "death_10y" else set())
                other = x.notna() & ~x.isin(seen)
                if other.any():
                    raise SystemExit(f"{col}: levels not in the table: {sorted(set(x[other].astype(str)))}")
                lv["missing"] = {"n": int(x.isna().sum()), "share": float(x.isna().mean())}
                r["sets"][key] = {"levels": lv}
        out["rows"].append(r)
    return out


def markdown(s: dict) -> str:
    def num(v, d):
        return f"{v:,.{d}f}"

    head = [s["sets"][k] for k, _ in SETS]
    lines = ["| Characteristic | " + " | ".join(f"{h['label']} (n = {h['n']:,})" for h in head) + " |",
             "|---|" + "---:|" * len(head)]
    for r in s["rows"]:
        col, d = r["column"], DECIMALS.get(r["column"], 0)
        if r["kind"] == "num":
            cells = [f"{num(v['median'], d)} ({num(v['q1'], d)}–{num(v['q3'], d)})" for v in (r["sets"][k] for k, _ in SETS)]
            lines.append(f"| {r['label']}, median (IQR) | " + " | ".join(cells) + " |")
            miss = [r["sets"][k]["missing"] for k, _ in SETS]
            if any(miss):
                lines.append("| &nbsp;&nbsp;missing, n (%) | " + " | ".join(
                    f"{m:,} ({100 * m / s['sets'][k]['n']:.1f}%)" for m, (k, _) in zip(miss, SETS)) + " |")
            continue
        levels = [l for l in r["sets"][SETS[0][0]]["levels"] if l != "missing"]
        if r["kind"] == "yes" and len(levels) == 1:
            cells = [f"{v['levels'][levels[0]]['n']:,} ({100 * v['levels'][levels[0]]['share']:.1f}%)"
                     for v in (r["sets"][k] for k, _ in SETS)]
            lines.append(f"| {r['label']}, n (%) | " + " | ".join(cells) + " |")
        else:
            lines.append(f"| {r['label']}, n (%) |" + " |" * len(head))
            for l in levels:
                cells = [f"{v['levels'][l]['n']:,} ({100 * v['levels'][l]['share']:.1f}%)" for v in (r["sets"][k] for k, _ in SETS)]
                lines.append(f"| &nbsp;&nbsp;{l} | " + " | ".join(cells) + " |")
        miss = [r["sets"][k]["levels"]["missing"] for k, _ in SETS]
        if any(m["n"] for m in miss):
            lines.append("| &nbsp;&nbsp;missing | " + " | ".join(f"{m['n']:,} ({100 * m['share']:.1f}%)" for m in miss) + " |")
    note = ("Unweighted counts of the NHANES 1999–2008 participants aged 40–79 with ten-year mortality follow-up; the "
            "references were fitted on the fitting set, and the systems answered the evaluation set. IQR, interquartile "
            "range; HDL, high-density lipoprotein; HbA1c, glycated haemoglobin.")
    return "\n".join(lines) + "\n\n" + note + "\n"


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    cohort = pd.read_parquet(root / "data" / "cohort.parquet")
    s = summarise(cohort)
    out = root / "results" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "characteristics.json").write_text(json.dumps(s, indent=1))
    (out / "characteristics.md").write_text(markdown(s), encoding="utf-8")
    print(f"{s['sets']['fit']['n']:,} fitting and {s['sets']['eval']['n']:,} evaluation participants; "
          f"wrote {out / 'characteristics.md'} and characteristics.json")

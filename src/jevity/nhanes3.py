"""NHANES III (1988-1994) adults 40-79 with ten-year mortality from the 2019 public-use LMF, harmonised to the
continuous-NHANES cohort columns. Used only as an empirical prior for the reference model.
Column positions come from CDC's SAS layout programs, parsed at runtime (parse_sas_layout)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .nhanes import _fetch, read_lmf as _read_lmf_cont, CONFIG as CONT, check_categorical_domains

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "config" / "nhanes3.yaml").read_text(encoding="utf-8"))
RAW = ROOT / "data" / "raw" / "nhanes3"

_RANGE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s+(\$\s*)?(\d+)\s*-\s*(\d+)(?:\s+\.\d+)?\s*$")
_AT = re.compile(r"@\s*(\d+)\s+([A-Z][A-Z0-9_]*)\s+(\$?)(\d+)\.(\d*)")


def parse_sas_layout(text: str) -> dict[str, tuple[int, int, bool]]:
    """{VAR: (start, end, is_char)} from a SAS INPUT statement, 1-based inclusive. Handles 'VAR start-end',
    'VAR $ start-end', single-column 'VAR col' and '@start VAR informat.' forms."""
    up = text.upper()
    m = re.search(r"\bINPUT\b(.*?);", up, flags=re.S)
    body = m.group(1) if m else up
    out: dict[str, tuple[int, int, bool]] = {}
    for a in _AT.finditer(body):
        start, var, ch, width = int(a.group(1)), a.group(2), bool(a.group(3)), int(a.group(4))
        out[var] = (start, start + width - 1, ch)
    body = _AT.sub(" ", body)
    for var, ch, s, e in re.findall(r"([A-Z][A-Z0-9_]*)\s+(\$\s*)?(\d+)(?:\s*-\s*(\d+))?", body):
        out.setdefault(var, (int(s), int(e) if e else int(s), bool(ch.strip())))
    return out


def read_fixed(name: str, wanted: list[str]) -> pd.DataFrame:
    spec = CFG["files"][name]
    layout_path = _fetch(f"{CFG['base_url']}/{spec['layout']}", RAW / spec["layout"])
    data_path = _fetch(f"{CFG['base_url']}/{spec['data']}", RAW / spec["data"])
    lay = parse_sas_layout(layout_path.read_text(encoding="latin-1"))
    cols = ["SEQN"] + [v for v in wanted if v in lay and v != "SEQN"]
    missing = [v for v in wanted if v not in lay]
    if missing:
        print(f"[nhanes3] {name}: not in layout -> {missing}")
    specs = [(lay[c][0] - 1, lay[c][1]) for c in cols]
    df = pd.read_fwf(data_path, colspecs=specs, names=cols, dtype=str, encoding="latin-1")
    for c in cols:
        if lay[c][2]:
            continue
        raw = df[c].fillna("").str.strip()
        width = lay[c][1] - lay[c][0] + 1
        fill = raw.str.fullmatch(r"8+|9+") & (raw.str.len() == width) & (width > 1)
        df[c] = pd.to_numeric(raw.mask(fill | (raw == "")), errors="coerce")
    return df


def build_nhanes3_cohort(out: Path | None = None) -> pd.DataFrame:
    V = CFG["variables"]
    by_file: dict[str, list[str]] = {}
    for spec in V.values():
        by_file.setdefault(spec["file"], []).extend(spec.get("any_of", [spec.get("var")]))
    frames = {f: read_fixed(f, sorted(set(v))) for f, v in by_file.items()}
    df = frames["adult"]
    for f in ("exam", "lab"):
        df = df.merge(frames[f], on="SEQN", how="left")
    out_df = pd.DataFrame({"SEQN_III": df["SEQN"], "cycle": "1988-1994"})
    for name, spec in V.items():
        if "any_of" in spec:
            cols = [c for c in spec["any_of"] if c in df]
            if not cols:
                out_df[name] = np.nan
                continue
            sub = df[cols]
            yes, no = (sub == 1).any(axis=1), (sub == 2).all(axis=1)
            out_df[name] = np.where(yes, "yes", np.where(no, "no", None))
            continue
        col = spec["var"]
        s = df[col] if col in df else pd.Series(np.nan, index=df.index)
        if "codes" in spec:
            s = s.map({float(k): v for k, v in spec["codes"].items()})
        out_df[name] = s
    yrs = out_df.pop("education_years").where(lambda x: x <= 17)
    out_df["education"] = pd.cut(yrs, bins=[-1, 8, 11, 12, 15, 17],
                                 labels=["less than 9th grade", "9th to 11th grade", "high school graduate or GED",
                                         "some college", "college graduate or above"]).astype(object)
    cal = CFG["creatinine_calibration"]
    out_df["creatinine_mg_dl"] = cal["intercept"] + cal["slope"] * out_df["creatinine_mg_dl"]
    out_df["uacr_mg_g"] = out_df["urine_albumin_mg_l"] * 100.0 / out_df["urine_creatinine_mg_dl"]
    out_df["smoking"] = np.select([out_df.smoked_100.eq("no"), out_df.smoke_now.eq("yes"), out_df.smoke_now.eq("no")],
                                  ["never", "current", "former"], default=None)
    check_categorical_domains(out_df, V)                     # text labels only (YAML yes/no guard)
    out_df["chd"] = np.nan
    out_df["rx_count"] = np.nan
    out_df["rx_names"] = ""
    # mortality
    lmf_path = _fetch(f"{CFG['lmf_url']}/{CFG['lmf_file']}", RAW / CFG["lmf_file"])
    lay = CONT["lmf_layout"]
    lmf = pd.read_fwf(lmf_path, colspecs=[(v[0] - 1, v[1]) for v in lay.values()], names=list(lay.keys()), dtype=str)
    lmf["SEQN_III"] = pd.to_numeric(lmf["seqn"].str.strip(), errors="coerce")
    for c in ("eligstat", "mortstat", "permth_exm"):
        lmf[c] = pd.to_numeric(lmf[c], errors="coerce")
    out_df = out_df.merge(lmf[["SEQN_III", "eligstat", "mortstat", "permth_exm"]], on="SEQN_III", how="left")
    flow = [("NHANES III adults (adult file)", len(out_df))]
    out_df = out_df[out_df.age.between(40, 79)]; flow.append(("Age 40-79", len(out_df)))
    out_df = out_df[out_df.eligstat == 1]; flow.append(("Eligible for linkage", len(out_df)))
    out_df = out_df.dropna(subset=CONT["phenoage_nine"]); flow.append(("Nine PhenoAge biomarkers present", len(out_df)))
    died = (out_df.mortstat == 1) & (out_df.permth_exm <= 120)
    alive = out_df.permth_exm >= 120
    out_df = out_df[died | alive].copy()
    out_df["death_10y"] = died[died | alive].astype(int)
    flow.append((f"Analysis cohort; deaths within 120 months = {int(out_df.death_10y.sum())}", len(out_df)))
    out_df["SEQN"] = out_df["SEQN_III"].astype(int) + int(CFG["seqn_offset"])
    out_df["split"] = "prior"
    out_df["era_nhanes3"] = 1
    out = out or (ROOT / "data" / "cohort_nhanes3.parquet")
    out_df.to_parquet(out, index=False)
    lines = ["# NHANES III prior cohort flow (generated by scripts/build_nhanes3.py)", "", "| Step | n |", "|---|---:|"]
    lines += [f"| {s} | {n:,} |" for s, n in flow]
    lines += ["", "Non-missing share by variable:", "", out_df.notna().mean().round(3).to_frame("share").to_markdown()]
    flow_md = ROOT / "results" / "cohort_flow_nhanes3.md"
    flow_md.parent.mkdir(parents=True, exist_ok=True)
    flow_md.write_text("\n".join(lines), encoding="utf-8")
    return out_df

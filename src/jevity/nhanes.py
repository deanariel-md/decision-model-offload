"""NHANES 1999–2008 cohort with the public-use Linked Mortality File (deaths through 31 Dec 2019).

Runs on a machine with internet access (downloads are cached under data/raw/). Nothing here touches an
API. Every exclusion is counted and written to results/cohort_flow.md by build_cohort().
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config" / "nhanes_variables.yaml").read_text(encoding="utf-8"))
RAW = ROOT / "data" / "raw"


# ----------------------------------------------------------------------------- download / read
def _fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import httpx  # local import so the module loads without network libs in tests

    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
        if r.content[:200].lstrip().lower().startswith((b"<!doctype", b"<html")):   # CDC answers a moved file with a
            raise RuntimeError(f"{url} returned an HTML page, not a data file")      # 404 page and HTTP 200
        dest.write_bytes(r.content)
    return dest


def read_xpt(cycle: str, component: str) -> pd.DataFrame:
    base = CONFIG["files"][component][cycle]
    url = f"{CONFIG['base_url']}/{cycle[:4]}/DataFiles/{base}.xpt"
    path = _fetch(url, RAW / cycle / f"{base}.XPT")
    df = pd.read_sas(path, format="xport", encoding="latin-1")
    df.columns = [c.upper() for c in df.columns]
    return df


def read_lmf(cycle: str) -> pd.DataFrame:
    name = CONFIG["cycles"][cycle]["lmf"]
    path = _fetch(f"{CONFIG['lmf_url']}/{name}", RAW / "lmf" / name)
    lay = CONFIG["lmf_layout"]
    colspecs = [(v[0] - 1, v[1]) for v in lay.values()]
    df = pd.read_fwf(path, colspecs=colspecs, names=list(lay.keys()), dtype=str)
    df["SEQN"] = pd.to_numeric(df["seqn"].str.strip(), errors="coerce")
    for c in ("eligstat", "mortstat", "permth_int", "permth_exm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["SEQN", "eligstat", "mortstat", "ucod_leading", "permth_int", "permth_exm"]]


# ----------------------------------------------------------------------------- harmonise
def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for v in candidates:
        if v in df.columns:
            return df[v]
    return pd.Series(np.nan, index=df.index)


def _bp_mean(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [f"{prefix}{i}" for i in (2, 3, 4) if f"{prefix}{i}" in df.columns]
    if not cols:
        cols = [f"{prefix}1"]
    vals = df[cols].replace(0, np.nan)
    m = vals.mean(axis=1)
    first = df.get(f"{prefix}1")
    return m.where(m.notna(), first)


def _medications(cycle: str) -> pd.DataFrame:
    rx = read_xpt(cycle, "rxq")
    name_col = "RXDDRUG" if "RXDDRUG" in rx.columns else ("RXD240B" if "RXD240B" in rx.columns else None)
    use_col = "RXDUSE" if "RXDUSE" in rx.columns else ("RXD030" if "RXD030" in rx.columns else None)   # RXD030 1999-2002, RXDUSE 2003-08
    taking = rx.get(use_col) if use_col else None
    if name_col is None:
        return pd.DataFrame({"SEQN": rx["SEQN"].unique(), "rx_count": np.nan, "rx_names": None})
    names = rx[name_col].astype(str).str.strip()
    names = names.where(~names.str.upper().isin(["", "NAN", "NONE", "99999", "77777", "55555"]), None)
    rx = rx.assign(_name=names)
    g = rx.groupby("SEQN")
    out = pd.DataFrame({
        "rx_count": g["_name"].apply(lambda s: int(s.dropna().nunique())),
        "rx_names": g["_name"].apply(lambda s: "; ".join(sorted(set(s.dropna())))),
    }).reset_index()
    if taking is not None:
        nt = rx.loc[rx[use_col] == 2, "SEQN"].unique()
        out.loc[out["SEQN"].isin(nt), ["rx_count", "rx_names"]] = [0, ""]
    return out


def load_cycle(cycle: str) -> pd.DataFrame:
    """One harmonised row per SEQN for a cycle, with the LMF merged (before any eligibility rule)."""
    parts = {k: read_xpt(cycle, k) for k in CONFIG["files"] if k != "rxq"}
    df = parts["demo"][["SEQN"]].copy()
    for comp, d in parts.items():
        df = df.merge(d, on="SEQN", how="left", suffixes=("", f"_{comp}"))
    out = pd.DataFrame({"SEQN": df["SEQN"], "cycle": cycle})
    for name, spec in CONFIG["variables"].items():
        s = _first_present(df, spec["vars"])
        if "codes" in spec:
            s = s.map({float(k): v for k, v in spec["codes"].items()})
        out[name] = s
    for w in ("WTMEC2YR", "WTMEC4YR"):       # examination weights (survey-weighted sensitivity only)
        out[w.lower()] = pd.to_numeric(df[w], errors="coerce") if w in df else np.nan
    out["sbp_mmhg"] = _bp_mean(df, "BPXSY")
    out["dbp_mmhg"] = _bp_mean(df, "BPXDI")
    out["uacr_mg_g"] = out["urine_albumin_mg_l"] * 100.0 / out["urine_creatinine_mg_dl"]
    out["smoking"] = np.select(
        [out["smoked_100"].eq("no"), out["smoke_now"].isin(["every day", "some days"]), out["smoke_now"].eq("not at all")],
        ["never", "current", "former"], default=None,
    )
    out = out.merge(_medications(cycle), on="SEQN", how="left")
    out = out.merge(read_lmf(cycle), on="SEQN", how="left")
    corr = CONFIG.get("creatinine_correction", {}).get(cycle)
    if corr:
        out["creatinine_mg_dl"] = corr["intercept"] + corr["slope"] * out["creatinine_mg_dl"]
    return out


def check_categorical_domains(df: pd.DataFrame, variables: dict) -> None:
    """Every coded variable must hold only its declared text labels (or missing). Guards against YAML turning unquoted
    yes/no into booleans, which silently breaks the smoking construction and label edits."""
    for name, spec in variables.items():
        if "codes" not in spec or name not in df:
            continue
        allowed = {v for v in spec["codes"].values()}
        assert all(isinstance(v, str) for v in allowed), f"{name}: non-text code labels {allowed}"
        seen = set(df[name].dropna().unique())
        assert seen <= allowed, f"{name}: values outside the declared labels: {seen - allowed}"


def combined_mec_weight(df: pd.DataFrame) -> pd.Series:
    """Ten-year (1999-2008) examination weight per the NCHS rule for combining cycles: 2/5 x WTMEC4YR for 1999-2002
    (the four-year weight already spans both cycles) and 1/5 x WTMEC2YR for 2003-2008."""
    early = df["cycle"].isin(["1999-2000", "2001-2002"])
    return pd.Series(np.where(early, 0.4 * df["wtmec4yr"], 0.2 * df["wtmec2yr"]), index=df.index, dtype=float)


# ----------------------------------------------------------------------------- cohort
def build_cohort(seed: int = 20260922, n_eval: int = 1000, n_pilot: int = 200,
                 out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = out_dir or (ROOT / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    el = CONFIG["eligibility"]
    nine = CONFIG["phenoage_nine"]
    flow: list[tuple[str, int]] = []
    frames = [load_cycle(c) for c in CONFIG["cycles"]]
    df = pd.concat(frames, ignore_index=True)
    check_categorical_domains(df, CONFIG["variables"])
    assert set(df["smoking"].dropna().unique()) <= {"never", "current", "former"}, "smoking construction failed"
    df["wt_mec_10y"] = combined_mec_weight(df)
    flow.append(("All respondents, 1999–2008", len(df)))
    lmf_check = df.groupby("cycle").agg(linked=("eligstat", lambda s: s.notna().mean()), eligible=("eligstat", lambda s: (s == 1).mean()),
                                        deceased=("mortstat", lambda s: (s == 1).mean()), max_fu=("permth_exm", "max"))
    print("LMF sanity (share linked, share eligible, share deceased, max follow-up months) by cycle:\n", lmf_check.round(3))
    df = df[df["age"].between(el["age_min"], el["age_max"])]
    flow.append((f"Age {el['age_min']}–{el['age_max']} at examination", len(df)))
    df = df[df["eligstat"] == 1]
    flow.append(("Eligible for mortality linkage (eligstat = 1)", len(df)))
    df = df[df["permth_exm"].notna()]
    flow.append(("Examination-based follow-up time present", len(df)))
    n_present = df[nine].notna().sum(axis=1)
    fit_min = int(el.get("fit_min_biomarkers", 9))
    df = df[n_present >= fit_min].copy()
    df["complete9"] = n_present.loc[df.index] == len(nine)
    flow.append((f">= {fit_min} of the nine PhenoAge biomarkers present (fitting set may impute the rest)", len(df)))
    flow.append(("  of whom all nine present (eligible for pilot/evaluation)", int(df["complete9"].sum())))
    h = el["horizon_months"]
    died = (df["mortstat"] == 1) & (df["permth_exm"] <= h)
    alive_h = (df["permth_exm"] >= h) | ((df["mortstat"] == 1) & (df["permth_exm"] > h))
    undetermined = ~(died | alive_h)
    flow.append((f"Vital status undetermined at {h} months (excluded)", int(undetermined.sum())))
    df = df[~undetermined].copy()
    df["death_10y"] = died[~undetermined].astype(int)
    flow.append((f"Analysis cohort; deaths within {h} months = {int(df['death_10y'].sum())}", len(df)))

    rng = np.random.default_rng(seed)
    df["split"] = "fit"
    complete_idx = np.flatnonzero(df["complete9"].to_numpy())
    pick = rng.permutation(complete_idx)
    df.iloc[pick[:n_pilot], df.columns.get_loc("split")] = "pilot"
    df.iloc[pick[n_pilot:n_pilot + n_eval], df.columns.get_loc("split")] = "eval"
    df.to_parquet(out_dir / "cohort.parquet", index=False)
    (out_dir / "splits.json").write_text(json.dumps({
        "seed": seed, "n_pilot": n_pilot, "n_eval": n_eval,
        "counts": df["split"].value_counts().to_dict(),
        "deaths_by_split": df.groupby("split")["death_10y"].sum().to_dict(),
    }, indent=2))
    lines = ["# Cohort flow (generated by scripts/build_cohort.py)", "", "| Step | n |", "|---|---:|"]
    lines += [f"| {s} | {n:,} |" for s, n in flow]
    lines += ["", "## Variable availability by cycle (non-missing fraction among the analysis cohort)", ""]
    avail = df.groupby("cycle")[[c for c in CONFIG["variables"] if c in df.columns] + ["sbp_mmhg", "uacr_mg_g", "rx_count"]].apply(lambda g: g.notna().mean()).round(3)
    lines.append(avail.T.to_markdown())
    flow_md = ROOT / "results" / "cohort_flow.md"
    flow_md.parent.mkdir(parents=True, exist_ok=True)
    flow_md.write_text("\n".join(lines), encoding="utf-8")
    return df

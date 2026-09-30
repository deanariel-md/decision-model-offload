"""Empirical prior (NHANES III) settings shared by fitting, refits and analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def settings(root: Path = ROOT) -> dict:
    cfg = yaml.safe_load((root / "config" / "nhanes3.yaml").read_text(encoding="utf-8"))["prior"]
    sens = {}
    for name, v in cfg.get("sensitivities", {}).items():
        es = v.get("era_specific")
        sens[name] = {"a0": float(v["a0"]), "era_specific": None if es is None else [x for x in es if x != "intercept"]}
    return {"a0": float(cfg["a0"]), "era_specific": [x for x in cfg["era_specific"] if x != "intercept"], "sensitivities": sens}


def load_prior(root: Path = ROOT) -> pd.DataFrame | None:
    f = root / "data" / "cohort_nhanes3.parquet"
    return pd.read_parquet(f) if f.exists() else None

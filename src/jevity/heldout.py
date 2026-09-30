"""Arm 3, held-out reports: colon cancer pathology reports (data/arm3_heldout/items.csv) scored exactly as in arm 3.
The tested systems use arm 3's request builders and the documented Jev client's, unchanged; only the report differs
(heldout_arm: arm 3's config with the held-out items and the standard route, allow_fallbacks false).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import categorical as K
from . import docuse_staging as D
from .clients import ROOT

SEED = 20260928
DATA = ROOT / "data" / "arm3_heldout"
RESULTS = ROOT / "results" / "arm3_heldout"
ITEMS_PATH = DATA / "items.csv"
SYSTEM_STORE = ROOT / "runs" / "arm3_heldout"
MAIN_CHATBOTS = ("gpt", "claude", "gemini", "muse", "glm")


# ------------------------------------------------------------------------------------------------ tested systems
def heldout_arm(items_path: Path = ITEMS_PATH) -> K.ArmSpec:
    """Arm 3's spec (config/arm3.yaml: question, levels, prompts, reply schema, analysis settings, seed) with the
    held-out items and the standard route (provider pinned, allow_fallbacks false). The spec keeps arm 3's name, which
    the reply schema sends ("arm3_answer"), so every request body is arm 3's; the analysis is labelled arm3_heldout
    (config "arm") and the raw replies go to their own store (runs/arm3_heldout)."""
    a3 = K.load_arm("arm3")
    items = {k: v for k, v in a3.config["items"].items() if k not in ("parser_test", "repeat_column")}
    p = Path(items_path)
    items["main"] = (str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)).replace("\\", "/")
    conf = {**a3.config, "arm": "arm3_heldout", "items": items, "route": {**a3.config.get("route", {}), "batch": False}}
    return K.ArmSpec(**{**a3.__dict__, "batch": False, "config": conf})


def load_items_frame(path: Path = ITEMS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"report_id": str, "level": str, "substage": str}, keep_default_na=False)


def arm3_items(items_path: Path = ITEMS_PATH) -> list[K.Item]:
    """The reports as arm 3 Items, names and definitions (categorical.load_items on data/arm3_heldout/items.csv)."""
    return K.load_items(heldout_arm(items_path))


def documented_items(frame: pd.DataFrame | None = None) -> list[K.Item]:
    """The reports as Items of the documented Jev client, both versions (as docuse_staging.load_items)."""
    f = load_items_frame() if frame is None else frame
    return [K.Item(r["report_id"], r["report"].strip(), str(r["level"]), D.LEVELS, D.LEVELS, (None,) * len(D.LEVELS),
                   str(r["level"]), str(r["substage"]), v)
            for v in D.VARIANTS for r in f.to_dict("records")]


def items_frame(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The analysis item frame (as run_categorical.items_frame): item_id, truth, stratum (level), group (substage)."""
    f = load_items_frame() if frame is None else frame
    return (pd.DataFrame({"item_id": f.report_id, "truth": f.level.astype(str), "stratum": f.level.astype(str),
                          "group": f.substage.astype(str), "state": f.report})
            .sort_values("item_id").reset_index(drop=True))


def truth_parts(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    """The four part answers that are right for each report (docuse_staging.parts_of_truth), by report id."""
    return {r["report_id"]: D.parts_of_truth(r["t"], int(r["nodes_involved"]), int(r["tumour_deposits"]), r["m"])
            for r in frame.to_dict("records")}


# ------------------------------------------------------------------------------------------------ projection
def usd(tokens_in: float, tokens_out: float, price_in: float, price_out: float) -> float:
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


def arm3_token_means(path: Path = ROOT / "results" / "arm3" / "calls.parquet") -> dict[tuple[str, str], dict[str, float]]:
    """Arm 3's main-pass mean reported input and output tokens per (system, variant): the projection's basis for the
    tested systems' output tokens."""
    d = pd.read_parquet(path)
    d = d[d["repeat"] == 0]
    return {(s, v): {"tokens_in": float(g.tokens_in.mean()), "tokens_out": float(g.tokens_out.fillna(0).mean())}
            for (s, v), g in d.groupby(["system", "variant"])}

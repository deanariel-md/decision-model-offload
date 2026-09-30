"""Arm 3: the AJCC 8th colorectal stage table and the 10 scored levels.

The stage table (config/crc_ajcc8.yaml) is the answer key; IVA and IVB are scored as one level, IVA-IVB, and IVC as its
own (10 levels, the documented maximum of a Jev Score question)."""
from __future__ import annotations

import itertools
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
AJCC = yaml.safe_load((ROOT / "config" / "crc_ajcc8.yaml").read_text(encoding="utf-8"))
T_CATS: list[str] = AJCC["t_categories"]
N_CATS: list[str] = AJCC["n_categories"]
M_CATS: list[str] = AJCC["m_categories"]
STAGE_GROUPS: list[str] = AJCC["stage_groups"]          # the 11 AJCC substages, stored with every report
LEVELS: list[str] = AJCC["levels"]                       # the 10 scored levels
MAIN_STAGES: list[str] = AJCC["main_stages"]             # 0, I, II, III, IV (secondary measure)


# ---------------------------------------------------------------- stage table

def _expand(v, cats: list[str]) -> list[str]:
    return list(cats) if v == "any" else list(v)


def _build_table(rows: list[dict]) -> dict[tuple[str, str, str], str]:
    table: dict[tuple[str, str, str], str] = {}
    for r in rows:
        assert r["stage"] in STAGE_GROUPS, r
        for cell in itertools.product(_expand(r["t"], T_CATS), _expand(r["n"], N_CATS), _expand(r["m"], M_CATS)):
            if cell in table and table[cell] != r["stage"]:
                raise ValueError(f"rows disagree on {cell}")
            table[cell] = r["stage"]
    return table


STAGE_TABLE = _build_table(AJCC["stage_table"])


def stage_group(t: str, n: str, m: str) -> str:
    """AJCC 8th stage group of a (T, N, M) cell; a cell the table does not stage (Tis with a positive node) raises."""
    try:
        return STAGE_TABLE[(t, n, m)]
    except KeyError:
        raise ValueError(f"AJCC 8th gives no stage group for {t} {n} {m}") from None


def level_of(stage: str) -> str:
    """Scored level of an AJCC stage group: IVA and IVB are IVA-IVB; every other group is its own level."""
    if stage not in STAGE_GROUPS:
        raise ValueError(f"not an AJCC 8th stage group: {stage}")
    return AJCC["level_of_stage_group"].get(stage, stage)


def level_index(level: str) -> int:
    return LEVELS.index(level)


def main_stage(level: str) -> str:
    """Main stage of a scored level: IIA-IIC are II, IIIA-IIIC are III, IVA-IVB and IVC are IV, and 0 and I are
    themselves."""
    try:
        return AJCC["main_stage_of_level"][level]
    except KeyError:
        raise ValueError(f"not a scored level: {level}") from None


def n_category(nodes_involved: int, deposits: int) -> str:
    """N from the count of involved regional nodes; deposits decide only when every node is negative (N1c)."""
    if nodes_involved < 0 or deposits < 0:
        raise ValueError("counts must be non-negative")
    if nodes_involved == 0:
        return "N1c" if deposits > 0 else "N0"
    if nodes_involved == 1:
        return "N1a"
    if nodes_involved <= 3:
        return "N1b"
    return "N2a" if nodes_involved <= 6 else "N2b"


def _range_label(values, cats: list[str], any_label: str) -> str:
    if values == "any":
        return any_label
    idx = [cats.index(v) for v in values]
    assert idx == list(range(idx[0], idx[0] + len(idx))), f"non-contiguous categories {values}"
    return values[0] if len(values) == 1 else f"{values[0]}-{values[-1]}"


def stage_definition(stage: str) -> str:
    """The table rows of a stage group in AJCC notation, e.g. 'T3-T4a N1a-N1c M0; T2-T3 N2a M0; T1-T2 N2b M0'."""
    rows = [r for r in AJCC["stage_table"] if r["stage"] == stage]
    return "; ".join(f"{_range_label(r['t'], T_CATS, 'Any T')} {_range_label(r['n'], N_CATS, 'Any N')} "
                     f"{_range_label(r['m'], M_CATS, 'Any M')}" for r in rows)


def level_definition(level: str) -> str:
    """The definition shown with a level in the definitions version: the wording config gives where it gives one
    (IVA-IVB: 'any T, any N, M1a-M1b'; IVC: 'any T, any N, M1c'), else the level's table rows."""
    if level not in LEVELS:
        raise ValueError(f"not a scored level: {level}")
    return AJCC["level_definitions"].get(level) or stage_definition(level)

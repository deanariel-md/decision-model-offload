"""Synthetic item files for the tests (the repository holds no item file): arm 2 messages, arm 3 reports, the
misleading-feature reports and the non-regional node site reports, with the columns the configs read, written to a
temporary folder that is removed when the test process ends. The truth follows the AJCC stage table
(config/crc_ajcc8.yaml, jevity.crc); the texts are placeholders, not messages or reports.

    arm("arm2") / arm("arm3") / arm("arm3_confuser")   the arm with items.main naming the synthetic file
    arm_yaml(name)                                      that arm's config as a file (for scripts that take a config path)
    items_path(name)                                    the synthetic item file (arm2, arm3, arm3_confuser, nonregional)
    use_in_docuse(monkeypatch)                          config/docuse_staging.yaml's items and source arm, in memory"""
from __future__ import annotations

import atexit
import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical as K  # noqa: E402
from jevity import crc  # noqa: E402

SEED = 20260922
_DIR: Path | None = None
_MADE: dict[str, Path] = {}


def folder() -> Path:
    global _DIR
    if _DIR is None:
        _DIR = Path(tempfile.mkdtemp(prefix="jevity_synthetic_items_"))
        atexit.register(shutil.rmtree, _DIR, True)
    return _DIR


def _write(name: str, df: pd.DataFrame) -> Path:
    p = folder() / f"{name}_items.csv"
    df.to_csv(p, index=False, lineterminator="\n")
    return p


# ------------------------------------------------------------------------------------------------ arm 2
DOMAINS = ("imaging", "prescribing", "screening")


def arm2_frame(seed: int = SEED) -> pd.DataFrame:
    """180 messages on 14 recommendations (rec_01-rec_14; 55 appropriate care, 125 low value), one option set per
    recommendation, the seeded option order and 40 repeats, as config/arm2.yaml reads them."""
    rows = []
    for r in range(1, 15):
        rec = f"rec_{r:02d}"
        opts = {c: f"Option {c} of recommendation {r}: {w}." for c, w in
                zip("ABCD", ("arrange the test now", "wait and review", "start a medicine", "refer at once"))}
        for vt, short, n in (("appropriate_care", "ac", 4 if r < 14 else 3), ("low_value", "lv", 9 if r < 14 else 8)):
            for k in range(1, n + 1):
                iid = f"{rec}_{short}_{k:02d}"
                truth = "A" if vt == "appropriate_care" else "B"
                order = "".join(K.shuffle_order(iid, K.CHOICE_LABELS, seed))
                words = " ".join(["word"] * (8 + (3 * k + r) % 23 + (6 if vt == "appropriate_care" else 0)))
                neg = " I do not know." * ((k + r) % 3 + (1 if vt == "low_value" else 0))
                text = f"Synthetic message {iid}. {words}.{neg} What should I do?"
                rows.append({"item_id": iid, "recommendation_id": rec, "recommendation": f"Recommendation {r}",
                             "vignette_type": vt, "clinical_domain": DOMAINS[r % 3],
                             "intervention": f"synthetic intervention {r}", "evidence_grade": "AB"[r % 2],
                             "truth": truth, "presented_order": order,
                             "truth_presented": K.CHOICE_LABELS[order.index(truth)], "repeat_set": "False",
                             **{f"option_{c}": t for c, t in opts.items()},
                             "prompt_patient": text, "opening_form": "synthetic", "prompt_sent": text})
    df = pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)
    rep = np.random.default_rng(seed).choice(len(df), size=40, replace=False)
    df.loc[rep, "repeat_set"] = "True"
    return df


# ------------------------------------------------------------------------------------------------ arm 3
def _cells(stage: str) -> list[tuple[str, str, str]]:
    """The stage table's cells of a stage group (Tis only in stage 0), in table order."""
    return [c for c, g in crc.STAGE_TABLE.items() if g == stage and (stage == "0" or c[0] != "Tis")]


def _counts(n: str, j: int) -> tuple[int, int, int]:
    """(involved nodes, examined nodes, deposits) that give N category n."""
    lo, hi = crc.AJCC["generation"]["nodes_involved"][n]
    inv = lo + j % (hi - lo + 1)
    td = 1 + j % 3 if n == "N1c" else (0 if n == "N0" or j % 4 else 1 + j % 2)
    assert crc.n_category(inv, td) == n
    return inv, max(12, inv + 3) + j % 7, td


def _text(rid: str, t: str, inv: int, ex: int, td: int, m: str) -> str:
    return (f"SYNTHETIC REPORT {rid}\nPlaceholder text for the tests, not a pathology report.\n"
            f"Depth {t}; {inv} of {ex} nodes; {td} deposits; distant {m}.")


def arm3_frame(seed: int = SEED, per_level: int = 100) -> pd.DataFrame:
    """per_level reports per level (IVA-IVB: half IVA, half IVB), ids crc001 onwards in a seeded order, the first 600
    with_definitions true, 40 of those repeated: the columns config/arm3.yaml and config/docuse_staging.yaml read."""
    recs = []
    for lv in crc.LEVELS:
        groups = [g for g in crc.STAGE_GROUPS if crc.level_of(g) == lv]
        for g in groups:
            cells = _cells(g)
            for j in range(per_level // len(groups)):
                t, n, m = cells[j % len(cells)]
                inv, ex, td = _counts(n, j)
                recs.append({"level": lv, "substage": g, "t": t, "n": n, "m": m, "nodes_involved": inv,
                             "nodes_examined": ex, "tumour_deposits": td, "layout": ("synoptic", "narrative")[j % 2]})
    rng = np.random.default_rng(seed)
    for r, k in zip(recs, rng.permutation(len(recs))):
        r["report_id"] = f"crc{int(k) + 1:03d}"
    df = pd.DataFrame(recs)
    df = df.assign(_n=df.report_id.str[3:].astype(int)).sort_values("_n").drop(columns="_n").reset_index(drop=True)
    first = df.index < min(600, len(df))
    df["with_definitions"] = np.where(first, "true", "false")
    df["repeat"] = "false"
    df.loc[rng.choice(np.flatnonzero(first), size=min(40, int(first.sum())), replace=False), "repeat"] = "true"
    df["report"] = [_text(r.report_id, r.t, r.nodes_involved, r.nodes_examined, r.tumour_deposits, r.m)
                    for r in df.itertuples()]
    cols = ["report_id", "level", "substage", "t", "n", "m", "nodes_involved", "nodes_examined", "tumour_deposits",
            "layout", "with_definitions", "repeat", "report"]
    return df[cols]


# (type, feature, truth cell, the level a reader who falls for the feature gives)
CONFUSERS = {1: ("itc", ("T3", "N0", "M0"), "IIIB"), 2: ("adhesion", ("T3", "N1a", "M0"), "IIIC"),
             3: ("perforation", ("T4a", "N0", "M0"), "IIA"), 4: ("nonregional_node", ("T3", "N1b", "M1a"), "IIIB"),
             5: ("emvi", ("T3", "N2a", "M0"), "")}


def confuser_frame(first: int = 1001, per_type: int = 20, prefix: str = "crc", types=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    """per_type reports per misleading-feature type, ids <prefix><first> onwards: the columns config/arm3_confuser.yaml
    and config/docuse_staging.yaml read."""
    rows, k = [], first
    for ty in types:
        feat, (t, n, m), mis = CONFUSERS[ty]
        g = crc.stage_group(t, n, m)
        for j in range(per_type):
            inv, ex, _ = _counts(n, j)
            rid = f"{prefix}{k}"
            rows.append({"report_id": rid, "level": crc.level_of(g), "substage": g, "t": t, "n": n, "m": m,
                         "nodes_involved": inv, "nodes_examined": ex, "tumour_deposits": 0,
                         "layout": ("synoptic", "narrative")[j % 2], "confuser_type": str(ty),
                         "confuser_feature": feat, "misread_level": mis, "report": _text(rid, t, inv, ex, 0, m)})
            k += 1
    return pd.DataFrame(rows)


def nonregional_frame(sites: list[str]) -> pd.DataFrame:
    """20 type 4 reports (ids nrs1001-nrs1020), the sites in turn: the columns run_nonregional.py reads."""
    df = confuser_frame(1001, 20, "nrs", (4,))
    df["node_site"] = [sites[i % len(sites)] for i in range(len(df))]
    return df


# ------------------------------------------------------------------------------------------------ files and configs
BUILD = {"arm2": arm2_frame, "arm3": arm3_frame, "arm3_confuser": confuser_frame}


def items_path(name: str) -> Path:
    if name not in _MADE:
        _MADE[name] = _write(name, BUILD[name]())
    return _MADE[name]


def config(name: str) -> dict:
    """config/<name>.yaml with items.main naming the synthetic file."""
    c = yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text(encoding="utf-8"))
    c["items"]["main"] = str(items_path(name))
    return c


def arm_yaml(name: str) -> Path:
    p = folder() / f"{name}.yaml"
    if not p.exists():
        p.write_text(yaml.safe_dump(config(name), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def arm(name: str) -> K.ArmSpec:
    a = K.load_arm(name)
    return dataclasses.replace(a, config={**a.config, "items": {**a.config["items"], "main": str(items_path(name))}})


def use_in_docuse(mp) -> None:
    """config/docuse_staging.yaml's item files and source arm (arm 3), in memory, on the synthetic files."""
    from jevity import docuse_staging as D
    mp.setitem(D.CFG["items"], "main", str(items_path("arm3")))
    mp.setitem(D.CFG["items"], "confuser", str(items_path("arm3_confuser")))
    mp.setitem(D.CFG, "source_arm", str(arm_yaml("arm3")))

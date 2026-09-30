"""Jev's stored requests for every form of the colon cancer staging question, written from item files with this
repository's own Jev request builders, in the layout jevity.llm_forms reads (calls tables with raw paths, raw-store
JSON files). No request is sent and no reply is stored: each raw file holds the request and a response naming Jev's
model only. Used by the tests on synthetic reports, and to check stored LLM requests against the builder.

    frames = synthetic_frames()               set -> DataFrame(report_id, level, report, substage), placeholder texts
    write_tree(root, frames, pool)            data/..., data/arm3_pool/pool240.csv, results/... and runs/... under root
    use(monkeypatch, root, pool)              points jevity.llm_forms at root (pool: DataFrame(report_id, set))

Jev's requests per form: names and definitions, categorical.CategoricalJevClient (config/arm3.yaml; the
misleading-feature reports' names, config/arm3_confuser.yaml); documented and documented_notes,
docuse_staging.DocuseJevClient; structure_only, docuse_structure_only.StructureOnlyJevClient; named_field and
documented_no_examples, scripts/arm3_jev_parts.py build_requests() on the names and documented requests written here."""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical as K  # noqa: E402
from jevity import crc  # noqa: E402
from jevity import docuse_staging as D  # noqa: E402
from jevity import docuse_structure_only as S  # noqa: E402
from jevity import llm_forms as LF  # noqa: E402
from jevity.clients import RawStore  # noqa: E402

SEED = 20260922
ITEM_FILES = {s: p.relative_to(LF.ROOT) for s, p in LF.ITEMS.items()}
TABLES = {f: {s: (p.relative_to(LF.ROOT), flt) for s, (p, flt) in v.items()} for f, v in LF.JEV_TABLES.items()}
STORES = {"names": "runs/arm3", "definitions": "runs/arm3", "documented": "runs/arm3_docuse",
          "documented_notes": "runs/arm3_docuse", "structure_only": "runs/arm3_docuse_structure_only",
          "named_field": "runs/arm3_jev_parts", "documented_no_examples": "runs/arm3_jev_parts"}


def synthetic_frames(sizes: dict | None = None, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Placeholder reports (not reports) for each set, the truth drawn over the ten levels; one report per set
    carries quotes, a backslash, a tab, a line break, an en dash, a micro sign and braces."""
    import numpy as np
    sizes = sizes or {"main": 12, "heldout": 3, "misleading_feature": 5, "nonregional_sites": 2}
    rng = np.random.default_rng(seed)
    odd = 'He said "stop" \\ then left.\nSecond line, tab\there; en dash – and µg; braces {x} [y].'
    out = {}
    for s, n in sizes.items():
        rows = []
        for i in range(n):
            lv = crc.LEVELS[int(rng.integers(len(crc.LEVELS)))]
            text = f"Synthetic report {i} of set {s}: placeholder text, not a pathology report."
            rows.append({"report_id": f"syn_{s}_{i:03d}", "level": lv, "substage": lv,
                         "report": text + (" " + odd if i == 0 else "")})
        out[s] = pd.DataFrame(rows)
    return out


def _items(df: pd.DataFrame, variant: str, texts: tuple) -> list[K.Item]:
    return [K.Item(r["report_id"], str(r["report"]).strip(), r["level"], crc.LEVELS, crc.LEVELS, texts,
                   r["level"], str(r.get("substage", "")), variant) for r in df.to_dict("records")]


def _parts_module(root: Path, frames: dict):
    """scripts/arm3_jev_parts.py, reading the tree under root."""
    spec = importlib.util.spec_from_file_location("arm3_jev_parts_synthetic", ROOT / "scripts" / "arm3_jev_parts.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    P.ROOT = root
    P.SETS = {s: root / ITEM_FILES[s] for s in LF.SETS}
    P.N_REPORTS = sum(len(frames[s]) for s in LF.SETS)
    return P


def write_tree(root: Path, frames: dict[str, pd.DataFrame], pool: pd.DataFrame | None = None,
               write_items: bool = True) -> dict:
    """Item files (unless write_items is False: they are there already), the pool file and Jev's stored requests for
    every form and report. pool: DataFrame(report_id, set) (default: every report). Returns form -> report id -> Jev's
    request (cache keys included)."""
    root = Path(root)
    for s, df in frames.items():
        p = root / ITEM_FILES[s]
        p.parent.mkdir(parents=True, exist_ok=True)
        if write_items:
            df.to_csv(p, index=False, lineterminator="\n")
    if pool is None:
        pool = pd.concat([pd.DataFrame({"report_id": frames[s].report_id, "set": s}) for s in LF.SETS])
    pp = root / LF.POOL_CSV.relative_to(LF.ROOT)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pool[["report_id", "set"]].to_csv(pp, index=False, lineterminator="\n")

    a3, a3c = K.load_arm("arm3"), K.load_arm("arm3_confuser")
    defs = tuple(lv.get("definition") for lv in a3.config["levels"])
    none = (None,) * len(crc.LEVELS)
    stores = {k: RawStore(root / v) for k, v in STORES.items()}
    jev = {"names": {"main": a3, "heldout": a3, "misleading_feature": a3c, "nonregional_sites": a3},
           "definitions": {s: a3 for s in LF.SETS}}
    rows: dict[Path, list] = defaultdict(list)
    built: dict[str, dict] = defaultdict(dict)

    def store(form: str, s: str, req: dict):
        st = stores[form]
        st.put(req, {"model": LF.JEV_MODEL, "answers": {}}, {"synthetic": True})
        table, flt = TABLES[form][s]
        rows[root / table].append({"system": "jev", "item_id": req["_item"], "variant": form, "repeat": 0,
                                   "raw_path": str(st.path(req)), "set": flt if flt is not None else s})
        built[form][req["_item"]] = req

    for s in LF.SETS:
        df = frames[s]
        for form, texts in (("names", none), ("definitions", defs)):
            cl = K.CategoricalJevClient(stores[form], jev[form][s])
            for it in _items(df, form, texts):
                store(form, s, cl.request(it))
        dcl = D.DocuseJevClient(stores["documented"], D.arm_spec())
        for form in D.VARIANTS:
            for it in _items(df, form, none):
                store(form, s, dcl.request(it))
        scl = S.StructureOnlyJevClient(stores["structure_only"], D.arm_spec())
        for it in _items(df, S.VARIANT, none):
            store("structure_only", s, scl.request(it))
    _write_tables(rows)
    rows.clear()
    P = _parts_module(root, frames)
    jobs, _ = P.build_requests()
    for s, it, v, req in jobs:
        store(v, s, req)
    _write_tables(rows, append=True)
    return built


def _write_tables(rows: dict, append: bool = False) -> None:
    for path, rs in rows.items():
        df = pd.DataFrame(rs)
        if append and path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def clear_caches() -> None:
    for f in (LF._pool, LF._reports, LF._set_of, LF._jev_rows, LF.form_index, LF.template_problems, LF.template_text):
        f.cache_clear()


def use(monkeypatch, root: Path, pool: pd.DataFrame) -> None:
    """jevity.llm_forms reads the tree under root: its pool file, item files and Jev tables."""
    root = Path(root)
    monkeypatch.setattr(LF, "POOL_CSV", root / LF.POOL_CSV.relative_to(LF.ROOT))
    monkeypatch.setattr(LF, "ITEMS", {s: root / p for s, p in ITEM_FILES.items()})
    monkeypatch.setattr(LF, "JEV_TABLES", {f: {s: (root / p, flt) for s, (p, flt) in v.items()}
                                           for f, v in TABLES.items()})
    monkeypatch.setattr(LF, "N_POOL", len(pool))
    monkeypatch.setattr(LF, "SET_COUNTS", pool.set.value_counts().to_dict())
    clear_caches()

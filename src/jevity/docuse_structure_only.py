"""Arm 3, Jev used as documented: the structure-only ablation.

The documented version changes two things at once against Jev's original staging question: the structure (four Choice
questions in one request, the stage group computed in code from the AJCC 8th table) and the medical context (each
option's what / not_for / examples). structure_only keeps the structure and drops the context. For every report it
sends the documented request (same state, question keys, instructions, option labels in the same order, model and
provider block) with each option's criterion set to null, the label-only form of docs.typesafe.ai/primitives/choice
("null descriptions when option names are self-explanatory"). config/docuse_staging.yaml and the documented and
documented_notes versions are used unchanged. Parsing, the parts-to-stage code (crc.n_category, crc.stage_group,
crc.level_of over config/crc_ajcc8.yaml) and the routing confidence (product of the four top probabilities) are
docuse_staging's.

Every report set the documented version was run on, repeat 0, Jev only (SETS): main (1,000) and confuser (100, the
misleading-feature reports), as run_staging --full; reworded (20, rewritten by hand; run_reworded); nonregional (20,
non-regional node sites the notes do not name; run_nonregional); heldout (200, the held-out reports; scripts/heldout/
run.py). Each set's items are read from the file its documented run read; each set's raw store mirrors its documented
store (first write wins); tables under results/arm3_docuse_structure_only/<set>/. SPEND_CAP_USD, an optional cap on
the reported cost of every call of this variant in every set (stored answers included), is None (no cap) unless
scripts/docuse/run_structure_only.py --max_usd sets it.
"""
from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path

import pandas as pd

from . import categorical as K
from . import docuse_staging as D
from .clients import MODELS, ROOT, RawStore

VARIANT = "structure_only"
BASE_VARIANT = "documented"          # the request whose criteria are emptied
SPEND_CAP_USD: float | None = None   # total US$, every call of this variant in every set (stored answers included);
                                     # None: no cap
RESULTS_REL = "results/arm3_docuse_structure_only"


@dataclasses.dataclass(frozen=True)
class ReportSet:
    """One report set the documented version was run on: its items file, the documented run's raw store and tables
    (read only), this variant's raw store, the documented run's worker count and report order."""
    name: str
    items_csv: Path
    store: Path                  # structure_only raw store
    doc_store: Path              # the documented run's raw store
    doc_calls: Path              # the documented run's calls table (D.calls_table columns)
    doc_parts: Path
    workers: int
    seeded: bool                 # docuse_staging's seeded report order (main, confuser) or the items file's order

    def available(self) -> bool:
        return self.items_csv.exists()


def _set(name, items, store, doc_store, doc_calls, doc_parts, workers, seeded) -> ReportSet:
    return ReportSet(name, Path(items), ROOT / store, Path(doc_store), Path(doc_calls), Path(doc_parts), workers, seeded)


_DOC = ROOT / D.CFG["results"]
SETS: dict[str, ReportSet] = {
    "main": _set("main", ROOT / D.CFG["items"]["main"], "runs/arm3_docuse_structure_only", ROOT / D.CFG["store"],
                 _DOC / "calls.parquet", _DOC / "parts.parquet", int(D.CFG["workers"]), True),
    "confuser": _set("confuser", ROOT / D.CFG["items"]["confuser"], "runs/arm3_docuse_structure_only",
                     ROOT / D.CFG["store"], _DOC / "calls.parquet", _DOC / "parts.parquet", int(D.CFG["workers"]), True),
    "reworded": _set("reworded", ROOT / "data/arm3_reworded/items.csv", "runs/arm3_docuse_reworded_structure_only",
                     ROOT / "runs/arm3_docuse_reworded", ROOT / "results/arm3_docuse_reworded/calls.parquet",
                     ROOT / "results/arm3_docuse_reworded/parts.parquet", 4, False),
    "nonregional": _set("nonregional", ROOT / "data/arm3_nonregional_sites/items.csv",
                        "runs/arm3_nonregional_sites_structure_only", ROOT / "runs/arm3_nonregional_sites",
                        ROOT / "results/arm3_nonregional_sites/calls_jev.parquet",
                        ROOT / "results/arm3_nonregional_sites/parts.parquet", 4, False),
    "heldout": _set("heldout", ROOT / "data/arm3_heldout/items.csv", "runs/arm3_heldout_structure_only",
                    ROOT / "runs/arm3_heldout", ROOT / "results/arm3_heldout/documented_calls.parquet",
                    ROOT / "results/arm3_heldout/parts.parquet", 8, False),
}
SET_NAMES = tuple(SETS)


def label_only(qs: dict) -> dict:
    """The documented questions with every option's criterion replaced by null; keys, their order and every other
    field of each question kept."""
    return {qid: {**q, "criteria": {k: None for k in q["criteria"]}} for qid, q in qs.items()}


def items_frame(rs: ReportSet) -> pd.DataFrame:
    return pd.read_csv(rs.items_csv, dtype={"report_id": str, "level": str, "substage": str}, keep_default_na=False)


def load_items(set_name: str = "main") -> list[K.Item]:
    """A set's reports as Items of this variant, built as its documented run built them (report id, the report
    stripped, level; stratum: confuser type for the confusers, else level; group: substage)."""
    rs = SETS[set_name]
    out = []
    for r in items_frame(rs).to_dict("records"):
        truth = str(r["level"])
        assert truth in D.LEVELS, (r["report_id"], truth)
        out.append(K.Item(r["report_id"], r["report"].strip(), truth, D.LEVELS, D.LEVELS, (None,) * len(D.LEVELS),
                          str(r.get("confuser_type", truth)) if set_name == "confuser" else truth,
                          str(r.get("substage", "")), VARIANT))
    assert len({i.item_id for i in out}) == len(out), set_name
    if set_name == "main":
        assert len(out) == 1000, len(out)
    return out


def truth_parts(set_name: str) -> dict[str, dict[str, str]]:
    """The four answers a reader right on every part gives (docuse_staging.parts_of_truth on the set's truth columns,
    as every documented analysis does)."""
    return {r["report_id"]: D.parts_of_truth(r["t"], int(r["nodes_involved"]), int(r["tumour_deposits"]), r["m"])
            for r in items_frame(SETS[set_name]).to_dict("records")}


def base_item(item: K.Item) -> K.Item:
    return dataclasses.replace(item, variant=BASE_VARIANT)


def ordered_ids(set_name: str, items: list[K.Item]) -> list[str]:
    """main and confuser: docuse_staging's seeded order over all 1,100 reports, restricted to the set; the other sets:
    the items file's order (as their documented runs)."""
    ids = [i.item_id for i in items]
    if not SETS[set_name].seeded:
        return ids
    keep = set(ids)
    return [i for i in D.seeded_ids(D.load_items([BASE_VARIANT])) if i in keep]


class SpendGuard:
    """Refuses a live request when spent + held + this request's estimate would pass the cap (cap None: never
    refuses). spent starts from the stored responses of this variant (usage.cost as reported), so the cap covers
    earlier runs too. Requests already in flight when the cap is reached finish, so the total can pass the cap by at
    most one request's cost per worker (about US$0.00003 each at list price)."""

    def __init__(self, cap: float | None, spent: float = 0.0):
        self.cap = None if cap is None else float(cap)
        self.spent, self.held, self.lock = float(spent), 0.0, threading.Lock()

    def reserve(self, est: float) -> None:
        with self.lock:
            if self.cap is not None and self.spent + self.held + est > self.cap:
                raise RuntimeError(f"spend cap: spent ${self.spent:.4f} + held ${self.held:.4f} + ${est:.6f} "
                                   f"> ${self.cap:.2f}")
            self.held += est

    def settle(self, est: float, actual: float) -> None:
        with self.lock:
            self.held -= est
            self.spent += actual


def list_usd(req: dict) -> float:
    """Projected list cost of one request: body tokens (4 characters per token) at Jev's input price; output free."""
    return D.request_tokens(req) * float(MODELS["jev"]["price_per_mtok_input"]) / 1e6


def stored_spend(store: RawStore | None = None) -> float:
    """Reported cost (usage.cost) of every stored response of this variant: one store, or every set's store."""
    dirs = {store.run_dir} if store is not None else {rs.store for rs in SETS.values()}
    tot = 0.0
    for d in dirs:
        if Path(d).exists():
            for p in Path(d).glob("*.json"):
                u = (json.loads(p.read_text(encoding="utf-8")).get("response") or {}).get("usage") or {}
                tot += float(u.get("cost") or 0.0)
    return tot


class StructureOnlyJevClient(D.DocuseJevClient):
    """The documented request for the report with every criterion null; docuse_staging's transport, raw store,
    parsing and rows. Live sends go through the spend guard."""

    spend: SpendGuard | None = None

    def request(self, item: K.Item) -> dict:
        assert item.variant == VARIANT, item.variant
        req = D.DocuseJevClient.request(self, base_item(item))
        req["questions"] = label_only(req["questions"])
        req["_variant"] = VARIANT
        return req

    def _send(self, req: dict) -> dict:
        if self.sender is not None:
            return super()._send(req)
        if self.spend is None:
            raise RuntimeError("live calls need a spend guard (scripts/docuse/run_structure_only.py)")
        est = list_usd(req)
        self.spend.reserve(est)
        actual = est
        try:
            out = super()._send(req)
            u = (out.get("response") or {}).get("usage") or {}
            actual = float(u.get("cost") if u.get("cost") is not None else est)
            return out
        finally:
            self.spend.settle(est, actual)


def make_factory(arm: K.ArmSpec, store: RawStore, sender: K.Sender | None = None, spend: SpendGuard | None = None):
    cache: dict[int, StructureOnlyJevClient] = {}
    lock = threading.Lock()

    def get(system: str, repeat: int = 0) -> StructureOnlyJevClient:
        assert system == "jev" and repeat == 0, (system, repeat)
        with lock:
            if repeat not in cache:
                c = StructureOnlyJevClient(store, arm, repeat, sender=sender)
                c.spend = spend
                cache[repeat] = c
            return cache[repeat]
    return get


def _paths(obj, pre=()):
    """Every leaf of a JSON object as (path, value)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _paths(v, pre + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _paths(v, pre + (i,))
    else:
        yield pre, obj


def differing_paths(a: dict, b: dict) -> list[tuple]:
    """Leaf paths present in only one body or with different values."""
    pa, pb = dict(_paths(a)), dict(_paths(b))
    return sorted((k for k in set(pa) | set(pb) if pa.get(k, object()) != pb.get(k, object())), key=str)


def validate_request(req: dict, item: K.Item, arm: K.ArmSpec) -> list[str]:
    """Problems with one structure_only request (empty list: valid): the documented request for the report must be
    valid (docuse_staging.validate_request) and equal to this one except that every criterion is null; the option labels
    and their order are the config's."""
    bad = []
    doc_item = base_item(item)
    doc = D.DocuseJevClient(RawStore(ROOT / "runs" / "_unused"), arm).request(doc_item)
    bad += [f"documented: {b}" for b in D.validate_request(doc, doc_item)]
    if req.get("_variant") != VARIANT:
        bad.append(f"variant {req.get('_variant')}")
    exp = {**doc, "questions": label_only(doc["questions"]), "_variant": VARIANT}
    if json.dumps(req, ensure_ascii=False) != json.dumps(exp, ensure_ascii=False):     # key order included
        bad.append("request differs from the documented request with null criteria")
    for qid in D.QUESTION_IDS:
        crit = ((req.get("questions") or {}).get(qid) or {}).get("criteria") or {}
        if tuple(crit) != D.options(qid) or any(v is not None for v in crit.values()):
            bad.append(f"{qid}: criteria are not the config's labels with null descriptions")
    for p in differing_paths(D.body(doc), D.body(req)):
        if not (len(p) >= 4 and p[0] == "questions" and p[2] == "criteria"):
            bad.append(f"body differs outside the criteria at {p}")
    return bad


CALL_COLUMNS = D.CALL_COLUMNS


def documented_request(item: K.Item, arm: K.ArmSpec) -> dict:
    return D.DocuseJevClient(RawStore(ROOT / "runs" / "_unused"), arm).request(base_item(item))


def check_against_stored(item: K.Item, req: dict, arm: K.ArmSpec, set_name: str) -> list[str]:
    """The documented request for the report as the builder makes it must be the one stored by the set's documented
    run (same cache keys, same body, same key order); this variant's body must be that stored body with null criteria."""
    doc = documented_request(item, arm)
    got = RawStore(SETS[set_name].doc_store).get(doc)
    if got is None:
        return ["no stored documented request for this report"]
    sb = D.body(got["request"])
    bad = []
    if json.dumps(sb, ensure_ascii=False) != json.dumps(D.body(doc), ensure_ascii=False):
        bad.append("the stored documented body differs from the builder's")
    exp = {**sb, "questions": label_only(sb["questions"])}
    if json.dumps(exp, ensure_ascii=False) != json.dumps(D.body(req), ensure_ascii=False):
        bad.append("this body is not the stored documented body with null criteria")
    return bad


def calls_table(store: RawStore, arm: K.ArmSpec, items: list[K.Item]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(calls, parts) from the raw store for every report with a stored response; never sends. Same columns as
    results/arm3_docuse/calls.parquet and parts.parquet (docuse_staging.calls_table), variant structure_only."""
    factory = make_factory(arm, store)
    rows, parts = [], []
    for it in items:
        r = factory("jev", 0).stored(it)
        if r is None:
            continue
        p = r.pop("_parts")
        r["probs"] = json.dumps(r["probs"]) if r.get("probs") else None
        rows.append({"arm": arm.name, "system": "jev", "item_id": it.item_id, "variant": VARIANT, "repeat": 0, **r})
        prow = {"item_id": it.item_id, "variant": VARIANT, "repeat": 0, "p_unstaged": r.pop("_p_unstaged")}
        for q, d in p.items():
            prow.update({f"{q}_choice": d["choice"], f"{q}_top": d["top"], f"{q}_confidence": d["confidence"],
                         f"{q}_prob_status": d["prob_status"], f"{q}_probs": json.dumps(d["probs"]) if d["probs"] else None})
        parts.append(prow)
    return pd.DataFrame(rows, columns=CALL_COLUMNS), pd.DataFrame(parts)

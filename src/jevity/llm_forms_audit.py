"""Every LLM in every staging form: which LLM answers already exist. Read only: nothing here sends a request or
writes outside ROOT/results/arm3_llm_forms.

1. build_index(): every stored arm 3 LLM answer (repeat 0, system not jev) on the 240 pool reports, from the calls
   tables listed in SOURCES (raw_path -> the raw-store JSON {request, response, meta}). One row per stored answer: the
   request's messages, response_format, model, max_tokens, seed, provider, the route, and the table's valid / answer /
   top_prob. Written to results/arm3_llm_forms/stored_llm_index.parquet; load_index() reads it back.
2. match(stored_req, built_req): is a stored request character-identical to a built one (cache keys ignored; a schema
   name or a batch-versus-standard provider block is a non-blocking note).
3. coverage(build): every (system, form, report) of 10 x 7 x 240: reused (an identical stored request exists; its
   first stored response is analysed, usable or not), identical_to_names (named_field built equal to names built), gap
   (a new call is needed) or build_error (the builder refused the cell).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

OUT = ROOT / "results" / "arm3_llm_forms"
INDEX = OUT / "stored_llm_index.parquet"
COVERAGE = OUT / "coverage.parquet"
POOL_CSV = ROOT / "data" / "arm3_pool" / "pool240.csv"
ITEMS = {"main": ROOT / "data/arm3/items.csv", "heldout": ROOT / "data/arm3_heldout/items.csv",
         "misleading_feature": ROOT / "data/arm3/confuser_items.csv",
         "nonregional_sites": ROOT / "data/arm3_nonregional_sites/items.csv"}

SYSTEMS = ("gpt", "claude", "gemini", "muse", "glm", "gpt_free", "claude_free", "gemini_free", "medgemma", "gemma")
MAIN_LLMS = SYSTEMS[:5]
FORMS = ("names", "named_field", "definitions", "structure_only", "documented", "documented_no_examples",
         "documented_notes")
SETS = ("main", "heldout", "misleading_feature", "nonregional_sites")
STATUSES = ("reused", "identical_to_names", "gap", "build_error")

# The calls tables that hold stored arm 3 LLM answers on pool reports (label, table, where the raw store lives when a
# recorded raw_path is missing). Rows used: system != jev, repeat == 0, item_id in the pool. arm3_luna_320: GPT-5.6
# Luna with stage names on 320 further staging reports (standard route), a run whose code is not in this repository.
SOURCES: tuple[tuple[str, Path, Path | None], ...] = (
    ("arm3_main", ROOT / "results/arm3/calls.parquet", None),
    ("arm3_confuser", ROOT / "results/arm3_confuser/calls.parquet", None),
    ("arm3_heldout", ROOT / "results/arm3_heldout/calls.parquet", None),
    ("arm3_nonregional_sites", ROOT / "results/arm3_nonregional_sites/calls_chatbots.parquet", None),
    ("arm3_luna_320", ROOT / "results/arm3_luna_320/calls.parquet", ROOT),
)

_MISSING = object()


# ------------------------------------------------------------------------------------------------ small helpers
def sha_text(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def store_key(req: dict) -> str:
    """clients._sha (the RawStore file name), recomputed here so the audit needs no client import."""
    return hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)          # key order kept


def _show(v: Any, n: int = 80) -> str:
    if v is _MISSING:
        return "absent"
    s = _dumps(v)
    return s if len(s) <= n else s[:n] + "..."


def resolve_raw(recorded: str, home: Path | None = None) -> Path:
    """A recorded raw path; if it does not exist, the same runs/<store>/<file> under `home` (the folder that holds
    the store). Raises if neither exists."""
    f = Path(recorded)
    if f.exists():
        return f
    q = recorded.replace("\\", "/")
    if home is not None and "/runs/" in q:
        g = home / q[q.rindex("/runs/") + 1:]
        if g.exists():
            return g
    raise FileNotFoundError(f"raw file not found: {recorded}")


def pool() -> pd.DataFrame:
    p = pd.read_csv(POOL_CSV, dtype=str, keep_default_na=False)
    assert list(p.columns) == ["report_id", "set"] and len(p) == 240 and not p.report_id.duplicated().any()
    return p


def pool_reports() -> dict[str, str]:
    """report id -> the items file's report, stripped (for the user-template check)."""
    p = pool()
    out = {}
    for s in SETS:
        f = pd.read_csv(ITEMS[s], dtype=str, keep_default_na=False).set_index("report_id")
        for rid in p.report_id[p.set == s]:
            out[rid] = str(f.at[rid, "report"]).strip()
    return out


# ------------------------------------------------------------------------------------------------ 1. the index
def _index_row(label: str, table: Path, home: Path | None, r: Any, set_of: dict[str, str]) -> dict:
    raw = resolve_raw(str(r.raw_path), home)
    d = json.loads(raw.read_text(encoding="utf-8"))
    req, resp, meta = d.get("request") or {}, d.get("response") or {}, d.get("meta") or {}
    msgs = req.get("messages")
    assert (isinstance(msgs, list) and len(msgs) == 2 and [m.get("role") for m in msgs] == ["system", "user"]
            and all(set(m) == {"role", "content"} and isinstance(m["content"], str) for m in msgs)), raw
    assert req.get("_item") == r.item_id, (raw, req.get("_item"), r.item_id)
    rf = req.get("response_format")
    js = (rf or {}).get("json_schema") or {}
    choices = resp.get("choices") if isinstance(resp, dict) else None
    known = {"model", "messages", "max_tokens", "provider", "seed", "response_format", "temperature", "reasoning"}
    return {
        "system": r.system, "set": set_of[r.item_id], "item_id": r.item_id, "stored_variant": r.variant,
        "source": label, "source_table": str(table), "stored_arm": req.get("_arm"),
        "raw_path": str(raw), "raw_path_recorded": str(r.raw_path), "raw_moved": str(raw) != str(r.raw_path),
        "request_sha": raw.stem, "sha_ok": store_key(req) == raw.stem,
        "route": r.route, "route_meta": meta.get("route"), "route_key": req.get("_route"),
        "system_text": msgs[0]["content"], "system_sha": sha_text(msgs[0]["content"]),
        "user_text": msgs[1]["content"], "user_sha": sha_text(msgs[1]["content"]),
        "rf_type": (rf or {}).get("type"), "rf_name": js.get("name"), "rf_strict": js.get("strict"),
        "schema_json": _dumps(js.get("schema")) if rf else None,
        "schema_sha": sha_text(_dumps(js.get("schema"))) if rf else None,
        "model": req.get("model"), "max_tokens": req.get("max_tokens"), "seed": req.get("seed"),
        "provider_json": _dumps(req["provider"]) if "provider" in req else None,
        "temperature_json": _dumps(req["temperature"]) if "temperature" in req else None,
        "reasoning_json": _dumps(req["reasoning"]) if "reasoning" in req else None,
        "extra_keys": _dumps(sorted(k for k in req if not k.startswith("_") and k not in known)),
        "cache_keys_json": _dumps({k: v for k, v in req.items() if k.startswith("_")}),
        "body_rest_json": _dumps({k: v for k, v in req.items() if k != "messages"}),
        "model_reported": r.model_reported, "provider_reported": meta.get("provider_reported") or resp.get("provider"),
        "provider": r.provider, "valid": bool(r.valid), "answer": r.answer,
        "top_prob": None if pd.isna(r.top_prob) else float(r.top_prob), "parse": r.parse,
        "finish_reason": r.finish_reason, "table_error": r.error,
        "meta_ts": meta.get("ts"), "batch_id": meta.get("batch_id"),
        "response_has_choices": bool(choices),
    }


def build_index(sources: Iterable[tuple[str, Path, Path | None]] = SOURCES) -> pd.DataFrame:
    p = pool()
    ids, set_of = set(p.report_id), dict(zip(p.report_id, p.set))
    rows = []
    for label, table, home in sources:
        c = pd.read_parquet(table)
        c = c[(c.system != "jev") & (c["repeat"] == 0) & c.item_id.isin(ids)]
        if "simulated" in c.columns:
            assert not c.simulated.any(), f"{table}: simulated rows"
        for r in c.itertuples(index=False):
            rows.append(_index_row(label, table, home, r, set_of))
    df = pd.DataFrame(rows)
    df["seed"] = df["seed"].astype("Int64")
    df["max_tokens"] = df["max_tokens"].astype("Int64")
    dup = df.duplicated(["system", "item_id", "raw_path"])
    assert not dup.any(), f"{int(dup.sum())} stored answers indexed twice"
    return df.sort_values(["system", "set", "item_id", "stored_variant", "source"]).reset_index(drop=True)


def write_index(df: pd.DataFrame, path: Path = INDEX) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return path


def load_index(path: Path = INDEX) -> pd.DataFrame:
    return pd.read_parquet(path)


def stored_request(row: Any) -> dict:
    """The stored request of an index row, rebuilt from its parts (messages were checked to be one system and one
    user message with role and content only)."""
    g = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    req = json.loads(g("body_rest_json"))
    req["messages"] = [{"role": "system", "content": g("system_text")}, {"role": "user", "content": g("user_text")}]
    return req


# ------------------------------------------------------------------------------------------------ 2. match
def _first_diff(a: str, b: str) -> str:
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return (f"lengths {len(a)} vs {len(b)}, first difference at {i}: stored {a[max(0, i - 20):i + 30]!r} "
            f"vs built {b[max(0, i - 20):i + 30]!r}")


def _message_diffs(a: Any, b: Any) -> list[str]:
    if a is _MISSING or b is _MISSING or not isinstance(a, list) or not isinstance(b, list):
        return [] if a is b else [f"messages: stored {_show(a, 40)} vs built {_show(b, 40)}"]
    out = []
    if len(a) != len(b):
        out.append(f"messages: {len(a)} stored vs {len(b)} built")
    for i, (x, y) in enumerate(zip(a, b)):
        name = (x.get("role") if isinstance(x, dict) else None) or f"#{i}"
        if not (isinstance(x, dict) and isinstance(y, dict)):
            out.append(f"messages[{i}] not an object")
            continue
        if x.get("role") != y.get("role"):
            out.append(f"messages[{i}].role: stored {x.get('role')!r} vs built {y.get('role')!r}")
        cx, cy = x.get("content"), y.get("content")
        if cx != cy:
            out.append(f"{name} message content differs ("
                       + (_first_diff(cx, cy) if isinstance(cx, str) and isinstance(cy, str) else "not text") + ")")
        if set(x) != set(y) or any(_dumps(x[k]) != _dumps(y[k]) for k in x if k not in ("role", "content") and k in y):
            out.append(f"messages[{i}] keys: stored {sorted(x)} vs built {sorted(y)}")
    return out


def _response_format_diffs(a: Any, b: Any) -> tuple[list[str], list[str]]:
    if a is _MISSING and b is _MISSING:
        return [], []
    if a is _MISSING or b is _MISSING or not isinstance(a, dict) or not isinstance(b, dict):
        return [f"response_format: stored {_show(a, 40)} vs built {_show(b, 40)}"], []
    diffs, notes = [], []
    for k in sorted(set(a) | set(b)):
        if k == "json_schema":
            continue
        if _dumps(a.get(k, None)) != _dumps(b.get(k, None)) or ((k in a) != (k in b)):
            diffs.append(f"response_format.{k}: stored {_show(a.get(k, _MISSING))} vs built {_show(b.get(k, _MISSING))}")
    ja, jb = a.get("json_schema", _MISSING), b.get("json_schema", _MISSING)
    if ja is _MISSING or jb is _MISSING or not isinstance(ja, dict) or not isinstance(jb, dict):
        if not (ja is _MISSING and jb is _MISSING):
            diffs.append(f"response_format.json_schema: stored {_show(ja, 40)} vs built {_show(jb, 40)}")
        return diffs, notes
    for k in sorted(set(ja) | set(jb)):
        va, vb = ja.get(k, _MISSING), jb.get(k, _MISSING)
        same = (va is _MISSING) == (vb is _MISSING) and _dumps(None if va is _MISSING else va) == _dumps(
            None if vb is _MISSING else vb)
        if same:
            continue
        if k == "name":
            notes.append(f"schema name {va if va is not _MISSING else None!r} (stored) vs "
                         f"{vb if vb is not _MISSING else None!r} (built)")
        elif k == "schema" and isinstance(va, dict) and isinstance(vb, dict) and va == vb:
            diffs.append("response_format.json_schema.schema: same content, different key order")
        else:
            diffs.append(f"response_format.json_schema.{k}: stored {_show(va)} vs built {_show(vb)}")
    return diffs, notes


def _provider_diffs(a: Any, b: Any) -> tuple[list[str], list[str]]:
    if a is _MISSING and b is _MISSING:
        return [], []
    if isinstance(a, dict) and isinstance(b, dict):
        if a == b:
            return [], []
        oa, ob = a.get("only"), b.get("only")
        batch, std = (a, b) if "allow_fallbacks" not in a else (b, a)
        if (set(batch) == {"only"} and set(std) == {"only", "allow_fallbacks"} and std["allow_fallbacks"] is False
                and oa == ob and isinstance(oa, list) and len(oa) == 1):
            side = "stored" if batch is a else "built"
            return [], [f"provider: batch block {_dumps(batch)} ({side}) vs standard block {_dumps(std)} "
                        f"(same provider, route only)"]
    return [f"provider: stored {_show(a)} vs built {_show(b)}"], []


def match(stored_req: dict, built_req: dict) -> tuple[bool, list[str]]:
    """(identical, diffs). Identical: the system and user messages byte-identical; response_format's json_schema
    `schema` identical (key order included) with the same type and strict flag; model, max_tokens and seed identical
    (seed may be absent from both); no temperature and no reasoning in either; the provider block equal, or differing
    only by route (batch {"only": [p]} vs standard {"only": [p], "allow_fallbacks": false}, the same p); no other
    parameter differing. Keys starting with "_" (cache keys) are ignored. diffs lists every blocking difference, then
    the non-blocking ones prefixed "note: " (a schema name, a batch-versus-standard provider block): `identical` can
    be True with notes in diffs."""
    a = {k: v for k, v in stored_req.items() if not str(k).startswith("_")}
    b = {k: v for k, v in built_req.items() if not str(k).startswith("_")}
    diffs: list[str] = []
    notes: list[str] = []
    diffs += _message_diffs(a.pop("messages", _MISSING), b.pop("messages", _MISSING))
    d, n = _response_format_diffs(a.pop("response_format", _MISSING), b.pop("response_format", _MISSING))
    diffs, notes = diffs + d, notes + n
    for k in ("model", "max_tokens", "seed"):
        va, vb = a.pop(k, _MISSING), b.pop(k, _MISSING)
        if (va is _MISSING) != (vb is _MISSING) or (va is not _MISSING and _dumps(va) != _dumps(vb)):
            diffs.append(f"{k}: stored {_show(va)} vs built {_show(vb)}")
    for k in ("temperature", "reasoning"):
        va, vb = a.pop(k, _MISSING), b.pop(k, _MISSING)
        if va is not _MISSING or vb is not _MISSING:
            diffs.append(f"{k} present: stored {_show(va)} vs built {_show(vb)}")
    d, n = _provider_diffs(a.pop("provider", _MISSING), b.pop("provider", _MISSING))
    diffs, notes = diffs + d, notes + n
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, _MISSING), b.get(k, _MISSING)
        if (va is _MISSING) != (vb is _MISSING) or (va is not _MISSING and _dumps(va) != _dumps(vb)):
            diffs.append(f"{k}: stored {_show(va)} vs built {_show(vb)}")
    return not diffs, diffs + [f"note: {x}" for x in notes]


def blocking(diffs: list[str]) -> list[str]:
    return [d for d in diffs if not d.startswith("note: ")]


# ------------------------------------------------------------------------------------------------ 3. coverage
def _body_dumps(req: dict) -> str:
    return _dumps({k: v for k, v in req.items() if not str(k).startswith("_")})


def coverage(build: Callable[[str, str, str], dict], index: pd.DataFrame | None = None,
             systems: Iterable[str] = SYSTEMS, forms: Iterable[str] = FORMS, report_ids: Iterable[str] | None = None,
             set_of: dict[str, str] | None = None, log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """One row per (system, form, report). status: reused (an identical stored request exists; with several, the
    earliest stored meta ts is taken and logged), identical_to_names (named_field whose built request, cache keys
    aside, equals the built names request: the names cell decides), gap (a new call is needed; nearest_diffs gives
    the blocking differences from the closest stored request of the same system and report, if any) or build_error
    (build raised; error holds its message)."""
    idx = load_index() if index is None else index
    if set_of is None:
        p = pool()
        set_of = dict(zip(p.report_id, p.set))
    rids = list(report_ids) if report_ids is not None else list(set_of)
    systems, forms = list(systems), list(forms)
    log = log or (lambda s: None)
    cands: dict[tuple[str, str], list[dict]] = {}
    for r in idx.to_dict("records"):
        cands.setdefault((r["system"], r["item_id"]), []).append(r)
    built: dict[tuple[str, str, str], dict | Exception] = {}

    def get(system: str, form: str, rid: str):
        k = (system, form, rid)
        if k not in built:
            try:
                built[k] = build(system, form, rid)
            except Exception as e:                 # the builder's refusal is recorded, not raised
                built[k] = e
        return built[k]

    def cell(system: str, form: str, rid: str) -> dict:
        row = {"system": system, "form": form, "set": set_of.get(rid), "item_id": rid, "status": None,
               "raw_path": None, "stored_variant": None, "source": None, "route": None, "stored_valid": None,
               "stored_answer": None, "stored_top_prob": None, "n_identical": 0, "notes": "[]",
               "nearest_raw_path": None, "nearest_diffs": "[]", "built_sha": None, "error": None}
        req = get(system, form, rid)
        if isinstance(req, Exception):
            row.update(status="build_error", error=f"{type(req).__name__}: {req}")
            return row
        row["built_sha"] = store_key(req)
        ok, best = [], None
        for c in cands.get((system, rid), []):
            same, diffs = match(stored_request(c), req)
            if same:
                ok.append((c["meta_ts"] if c["meta_ts"] is not None else float("inf"), c, diffs))
            elif best is None or len(blocking(diffs)) < len(blocking(best[1])):
                best = (c, diffs)
        row["n_identical"] = len(ok)
        if ok:
            ok.sort(key=lambda t: t[0])
            ts, c, diffs = ok[0]
            if len(ok) > 1:
                log(f"{system}/{form}/{rid}: {len(ok)} identical stored answers; earliest taken: {c['raw_path']} "
                    f"(others: {[t[1]['raw_path'] for t in ok[1:]]})")
            row.update(status="reused", raw_path=c["raw_path"], stored_variant=c["stored_variant"],
                       source=c["source"], route=c["route"], stored_valid=c["valid"], stored_answer=c["answer"],
                       stored_top_prob=c["top_prob"], notes=_dumps([d[6:] for d in diffs if d.startswith("note: ")]))
        else:
            row["status"] = "gap"
            if best is not None:
                row.update(nearest_raw_path=best[0]["raw_path"], nearest_diffs=_dumps(blocking(best[1])))
        return row

    done: dict[tuple[str, str, str], dict] = {}

    def cached(system: str, form: str, rid: str) -> dict:
        if (system, form, rid) not in done:
            done[(system, form, rid)] = cell(system, form, rid)
        return done[(system, form, rid)]

    out = []
    for system in systems:
        for form in forms:
            for rid in rids:
                if form == "named_field":
                    nf, nm = get(system, "named_field", rid), get(system, "names", rid)
                    if not isinstance(nf, Exception) and not isinstance(nm, Exception) \
                            and _body_dumps(nf) == _body_dumps(nm):
                        base = dict(cached(system, "names", rid))
                        base.update(form="named_field", status="identical_to_names", built_sha=store_key(nf),
                                    notes=_dumps(json.loads(base["notes"]) + [f"names cell: {base['status']}"]))
                        out.append(base)
                        continue
                out.append(dict(cached(system, form, rid)))
    return pd.DataFrame(out)


def summary(cov: pd.DataFrame) -> pd.DataFrame:
    """system x form x set: the number of cells per status (columns STATUSES), in SYSTEMS x FORMS x SETS order."""
    t = cov.groupby(["system", "form", "set", "status"]).size().unstack("status", fill_value=0)
    t = t.reindex(columns=list(STATUSES), fill_value=0).reset_index()
    t.columns.name = None
    order = {"system": {s: i for i, s in enumerate(SYSTEMS)}, "form": {f: i for i, f in enumerate(FORMS)},
             "set": {s: i for i, s in enumerate(SETS)}}
    t = t.sort_values(["system", "form", "set"], key=lambda c: c.map(order[c.name]))
    return t.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ index description
def describe_index(df: pd.DataFrame, reports: dict[str, str] | None = None) -> dict[str, pd.DataFrame | list[str]]:
    """Tables for the notes: counts per system x set x stored variant (routes), request signatures per source, and
    the checks on the index."""
    out: dict[str, Any] = {}
    out["counts"] = (df.groupby(["system", "set", "stored_variant", "route"]).size().rename("n").reset_index())
    sig_cols = ["source", "stored_arm", "route", "rf_name", "schema_sha", "system_sha", "model", "max_tokens", "seed",
                "provider_json", "temperature_json", "reasoning_json", "extra_keys"]
    s = df.assign(schema_sha=df.schema_sha.str[:12], system_sha=df.system_sha.str[:12])
    out["signatures"] = (s.groupby(["system", "stored_variant"] + sig_cols, dropna=False).size().rename("n")
                         .reset_index())
    checks = [f"stored answers indexed: {len(df)}",
              f"raw files exist: {sum(Path(p).exists() for p in df.raw_path)} / {len(df)}",
              f"raw paths read from the copy under the store's folder: {int(df.raw_moved.sum())} "
              f"({', '.join(sorted(df.source[df.raw_moved].unique())) or 'none'})",
              f"file name == RawStore key of the stored request: {int(df.sha_ok.sum())} / {len(df)}",
              f"table route == meta route: {int((df.route == df.route_meta).sum())} / {len(df)}",
              f"responses with choices: {int(df.response_has_choices.sum())} / {len(df)}",
              f"usable stored answers (table valid): {int(df.valid.sum())} / {len(df)}",
              f"table errors: {int(df.table_error.notna().sum())}"]
    if reports is not None:
        tmpl = []
        for r in df.itertuples():
            rep = reports.get(r.item_id)
            n = r.user_text.count(rep) if rep else 0
            tmpl.append(sha_text(r.user_text.replace(rep, "{report}"))[:12] if n == 1 else f"report x{n}")
        df = df.assign(template=tmpl)
        out["templates"] = (df.groupby(["stored_variant", "template", "source"]).agg(
            n=("item_id", "size"), systems=("system", lambda x: ",".join(sorted(set(x))))).reset_index())
        once = int((~df.template.str.startswith("report x")).sum())
        checks.append(f"user message contains the items file's report (stripped) exactly once: {once} / {len(df)}")
    out["checks"] = checks
    return out


"""Every LLM in every staging form: build every LLM request (10 systems x 7 forms x 240 reports = 16,800) from Jev's
stored requests with jevity.llm_forms.build, run every assertion, and write the record
results/arm3_llm_forms/requests.json. Nothing is sent and no key is read.

Per form it prints the number built, the user template's sha256 and size, whether the named_field request equals the
names request for every system and report, the system message and the approximate input characters. A form whose Jev
template is not one, or any request that fails the content assertion or the settings check, stops that form; the
script then exits non-zero after writing what was built, with the failed forms named in the header.

requests.json: {"header": {...}, "requests": [...]}. header: per form the system message, the user template (the
report as {pathology_report}), its sha256, the response_format, the Jev sources per report set, the Jev questions
sha256; per system the base settings; the anomalies found. requests: one compact entry per (system, form, report):
system, form, set, item_id, request_sha (clients._sha of the full request with its cache keys = the RawStore file name
in runs/arm3_llm_forms), body_sha (clients._sha of the body sent), user_sha (sha256 of the user message, UTF-8, first
24 hex), jev_raw_path (the Jev raw file read, relative to the repository root), jev_body_sha (clients._sha of Jev's stored
body), status "built"; named_field entries also same_as_names (the request equals the names request apart from
_variant: no call, the names answer is reused).

  python scripts/llm_forms/build_requests.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevity import llm_forms as LF  # noqa: E402
from jevity.clients import MODELS, _sha  # noqa: E402

OUT = ROOT / "results" / "arm3_llm_forms" / "requests.json"


def rel(p: Path) -> str:
    """A path relative to the repository root, with forward slashes."""
    try:
        return str(Path(p).relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def system_settings() -> dict:
    """Base settings of each system as built (the first report's names request, cache keys and messages left out)."""
    rid = LF.report_ids()[0]
    out = {}
    for s in LF.SYSTEMS:
        b = LF.body(LF.build(s, "names", rid))
        out[s] = {k: v for k, v in b.items() if k not in ("messages", "response_format")}
    return out


def build_form(form: str) -> tuple[list[dict], dict]:
    """Every request of one form (raises ContentMismatch on the first failure) and its header."""
    t0 = time.time()
    entries, chars, same = [], [], Counter()
    idx = LF.form_index(form)
    for s in LF.SYSTEMS:
        for rid in LF.report_ids():
            req = LF.build(s, form, rid)
            src = idx[rid]
            user = req["messages"][1]["content"]
            chars.append(sum(len(m["content"]) for m in req["messages"]))
            e = {"system": s, "form": form, "set": src.set, "item_id": rid, "request_sha": _sha(req),
                 "body_sha": _sha(LF.body(req)), "user_sha": LF.text_sha(user)[:24], "jev_raw_path": rel(src.raw_path),
                 "jev_body_sha": _sha(src.body), "status": "built"}
            if form == "named_field":
                e["same_as_names"] = LF.identical_named_field(s, rid)
                same[e["same_as_names"]] += 1
            entries.append(e)
    tmpl = LF.template_text(form)
    user_shas = Counter(e["user_sha"] for e in entries)
    assert all(v == len(LF.SYSTEMS) for v in user_shas.values()), f"{form}: user message differs across systems"
    sources = {}
    for st in LF.SETS:
        table, filt = LF.JEV_TABLES[form][st]
        sources[st] = {"table": rel(table), "set_filter": filt, "n": sum(1 for x in idx.values() if x.set == st)}
    q0 = idx[LF.report_ids()[0]].body["questions"]
    head = {"kind": "single" if form in LF.SINGLE_FORMS else "four", "n_built": len(entries),
            "system_message": LF.system_text_for(form), "user_template": tmpl,
            "template_sha256": LF.text_sha(tmpl), "template_chars": len(tmpl),
            "response_format": LF.response_format_for(form),
            "jev_questions_sha256": LF.text_sha(json.dumps(q0, ensure_ascii=False)),
            "jev_sources": sources, "jev_models": dict(Counter(x.model for x in idx.values())),
            "jev_raw_read_from_root_runs": LF.moved_sources(form),
            "input_chars_mean": round(sum(chars) / len(chars), 1), "input_chars_min": min(chars),
            "input_chars_max": max(chars), "input_chars_total": sum(chars),
            "seconds": round(time.time() - t0, 1)}
    if form == "named_field":
        head["same_as_names"] = {"yes": same[True], "no": same[False]}
    return entries, head


def main():
    t0 = time.time()
    P = LF.pool()
    print(f"pool: {len(P)} reports {P.set.value_counts().reindex(LF.SETS).to_dict()}; nothing is sent")
    entries, forms, failed, anomalies = [], {}, {}, []
    for form in LF.QUEUE_ORDER:
        try:
            anomalies += list(LF.template_problems(form)) + LF.source_anomalies(form)
            e, h = build_form(form)
        except (LF.ContentMismatch, AssertionError, FileNotFoundError) as ex:
            failed[form] = f"{type(ex).__name__}: {ex}"
            print(f"\n== {form}: STOPPED: {failed[form][:2000]}")
            continue
        entries += e
        forms[form] = h
        nf = (f"named_field == names: {h['same_as_names']['yes']} of {h['n_built']} identical "
              f"({'every' if h['same_as_names']['no'] == 0 else 'NOT every'} system and report)"
              if form == "named_field" else "named_field == names: n/a")
        print(f"\n== {form} ({h['kind']}-question): {h['n_built']} built ({len(LF.SYSTEMS)} systems x {len(P)} reports)")
        print(f"   template sha256 {h['template_sha256']}  ({h['template_chars']:,} characters with the placeholder)")
        print(f"   {nf}")
        print(f"   input characters per request (system + user): mean {h['input_chars_mean']:,.0f}, "
              f"min {h['input_chars_min']:,}, max {h['input_chars_max']:,}; total {h['input_chars_total']:,} "
              f"(about {h['input_chars_total'] / 4 / 1e6:.2f} M tokens at 4 characters per token)")
        print(f"   Jev sources: " + "; ".join(f"{k} {v['n']} from {v['table']}" + (f" [set {v['set_filter']}]" if v['set_filter'] else "")
                                           for k, v in h["jev_sources"].items()))
        print(f"   Jev response models {h['jev_models']}; raw files read from ROOT/runs/ "
              f"(stored path absent): {h['jev_raw_read_from_root_runs']}")
        print(f"   system message: {h['system_message']}")
    header = {"arm": LF.ARM_NAME,
              "paths_relative_to": ".",
              "built_by": "scripts/llm_forms/build_requests.py (jevity.llm_forms.build; nothing sent)",
              "n_requests": len(entries), "systems": list(LF.SYSTEMS), "forms_order": list(LF.FORMS),
              "queue_order": list(LF.QUEUE_ORDER), "pool": {"file": "data/arm3_pool/pool240.csv",
                                                             "sets": P.set.value_counts().reindex(LF.SETS).to_dict()},
              "system_settings": system_settings() if "names" in forms else None,
              "fields": {"request_sha": "clients._sha of the full request incl. cache keys (_repeat, _arm, _item, "
                                        "_variant) = RawStore file name in runs/arm3_llm_forms",
                         "body_sha": "clients._sha of the body sent (no '_' keys)",
                         "user_sha": "sha256 of the user message (UTF-8), first 24 hex",
                         "jev_raw_path": "the Jev raw file read (source of the content), relative to paths_relative_to",
                         "jev_body_sha": "clients._sha of Jev's stored request body (no '_' keys)",
                         "same_as_names": "named_field only: equals the names request apart from _variant"},
              "forms": forms, "failed_forms": failed, "anomalies": anomalies}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = ('{"header": ' + json.dumps(header, ensure_ascii=False, indent=1) + ',\n "requests": [\n'
            + ",\n".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in entries) + "\n]}\n")
    json.loads(text)
    OUT.write_text(text, encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}: {len(entries):,} entries, {OUT.stat().st_size / 1e6:.1f} MB, "
          f"sha256 {LF.text_sha(OUT.read_text(encoding='utf-8'))}")
    print(f"anomalies: {anomalies or 'none'}")
    print(f"done in {time.time() - t0:.0f} s")
    if failed:
        raise SystemExit(f"STOPPED forms: {sorted(failed)}")


if __name__ == "__main__":
    main()

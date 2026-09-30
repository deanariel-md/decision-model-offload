"""Details of the reworded-report robustness check, from the stored responses only; no model call.
  python scripts/docuse/reworded_details.py      # results/arm3_docuse_reworded/details.json and parser_rows.csv
For each of the 40 reworded answers, against the same report's answer in the main documented run (repeat 0):
request parity (every request field equal except the report text), choices, probabilities and confidence, tokens.
Then the report parser of arm 3 (src/jevity/crc_parse.py parse_report) on the reworded and the original text, and a
scan of the reworded text for staging codes."""
from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical as K
from jevity.crc_parse import parse_report

QUESTIONS = ["t_category", "regional_nodes", "tumor_deposits", "distant_metastasis"]
ITEMS = ROOT / "data" / "arm3_reworded" / "items.csv"
MAIN_ITEMS = ROOT / "data" / "arm3" / "items.csv"
STORE = ROOT / "runs" / "arm3_docuse_reworded"
MAIN_STORE = ROOT / "runs" / "arm3_docuse"
RESULTS = ROOT / "results" / "arm3_docuse_reworded"
CODES = re.compile(r"\b(?:p?T(?:is|[0-4][ab]?)|p?N[0-2][abc]?|p?M[01][abc]?|stage\s+(?:0|I{1,3}V?|IV)[ABC]?)\b")


def _load(store: Path) -> dict:
    out = {}
    for f in store.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        rq = d["request"]
        out[(rq["_item"], rq["_variant"], int(rq.get("_repeat", 0)))] = d
    return out


def _state(rq: dict) -> dict:
    return json.loads(rq["state"]) if isinstance(rq["state"], str) else rq["state"]


def answers_vs_original(items: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rw, orig = _load(STORE), _load(MAIN_STORE)
    rows, issues = [], []
    for (rid, v, _), d in sorted(rw.items()):
        o = orig[(items.loc[rid, "source_report_id"], v, 0)]
        rq, oq = d["request"], o["request"]
        for k in sorted(set(rq) | set(oq)):
            if k not in ("state", "_item") and rq.get(k) != oq.get(k):
                issues.append(f"{rid}/{v}: request field {k} differs from the main run")
        if set(_state(rq)) != set(_state(oq)):
            issues.append(f"{rid}/{v}: state fields differ from the main run")
        if _state(rq).get("pathology_report", "").strip() != items.loc[rid, "report"].strip():
            issues.append(f"{rid}/{v}: state is not the reworded report")
        r, ro = d["response"], o["response"]
        row = {"report_id": rid, "source_report_id": items.loc[rid, "source_report_id"], "variant": v,
               "response_id": r["id"], "response_id_original": ro["id"], "model": r.get("model"),
               "provider": r.get("provider"), "tokens_in": r["usage"]["input_tokens"],
               "tokens_in_original": ro["usage"]["input_tokens"]}
        for q in QUESTIONS:
            a, b = r["answers"][q], ro["answers"][q]
            p = a.get("probabilities") or {}
            row[f"{q}_same_choice"] = a["choice"] == b["choice"]
            row[f"{q}_choice_is_top"] = bool(p) and p.get(a["choice"]) == max(p.values())
            row[f"{q}_confidence"], row[f"{q}_confidence_original"] = a.get("confidence"), b.get("confidence")
        rows.append(row)
    return pd.DataFrame(rows), issues


def parser_rows(items: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, it in items.iterrows():
        src = it.source_report_id
        a, b = parse_report(it.report), parse_report(main.loc[src, "report"])
        rows.append({"report_id": rid, "source_report_id": src, "t": main.loc[src, "t"], "n": main.loc[src, "n"],
                     "m": main.loc[src, "m"], "t_reworded": a["t"], "n_reworded": a["n"], "m_reworded": a["m"],
                     "t_original": b["t"], "n_original": b["n"], "m_original": b["m"]})
    df = pd.DataFrame(rows)
    for s in ("reworded", "original"):
        for c in ("t", "n", "m"):
            df[f"{c}_{s}_right"] = df[f"{c}_{s}"] == df[c]
        df[f"tnm_{s}_right"] = df[[f"{c}_{s}_right" for c in ("t", "n", "m")]].all(axis=1)
    return df


def main() -> dict:
    items = pd.read_csv(ITEMS, dtype=str, keep_default_na=False).set_index("report_id")
    main_items = pd.read_csv(MAIN_ITEMS, dtype=str, keep_default_na=False).set_index("report_id")
    ans, issues = answers_vs_original(items)
    pr = parser_rows(items, main_items)
    conf = {q: {"mean": float(ans[f"{q}_confidence"].mean()),
                "mean_original": float(ans[f"{q}_confidence_original"].mean()),
                "lower_than_original": int((ans[f"{q}_confidence"] < ans[f"{q}_confidence_original"]).sum()),
                "higher_than_original": int((ans[f"{q}_confidence"] > ans[f"{q}_confidence_original"]).sum())}
            for q in QUESTIONS}
    joint = ans[[f"{q}_confidence" for q in QUESTIONS]].prod(axis=1)
    joint_o = ans[[f"{q}_confidence_original" for q in QUESTIONS]].prod(axis=1)
    out = {
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "items_sha256": K.file_sha256(ITEMS),
        "answers": {
            "n": len(ans), "request_parity_issues": issues,
            "distinct_response_ids": int(ans.response_id.nunique()),
            "response_ids_shared_with_main_run": len(set(ans.response_id) & set(ans.response_id_original)),
            "providers": ans.provider.value_counts().to_dict(), "models": ans.model.value_counts().to_dict(),
            "same_choice_as_original": {q: int(ans[f"{q}_same_choice"].sum()) for q in QUESTIONS},
            "choice_is_top_probability": {q: int(ans[f"{q}_choice_is_top"].sum()) for q in QUESTIONS},
            "confidence": conf,
            "joint_confidence": {"mean": float(joint.mean()), "mean_original": float(joint_o.mean()),
                                 "min": float(joint.min()), "min_original": float(joint_o.min())},
            "tokens_in_mean": float(ans.tokens_in.mean()),
            "tokens_in_mean_original": float(ans.tokens_in_original.mean()),
        },
        "parser": {
            "parser": "src/jevity/crc_parse.py parse_report (the project's arm 3 report check, written for the "
                      "generator's reports; reads M only from a line starting 'Staging CT')",
            "n_reports": len(pr),
            "tnm_right": {"reworded": int(pr.tnm_reworded_right.sum()), "original": int(pr.tnm_original_right.sum())},
            "t_right": {"reworded": int(pr.t_reworded_right.sum()), "original": int(pr.t_original_right.sum())},
            "n_right": {"reworded": int(pr.n_reworded_right.sum()), "original": int(pr.n_original_right.sum())},
            "m_right": {"reworded": int(pr.m_reworded_right.sum()), "original": int(pr.m_original_right.sum())},
            "t_and_n_right": {s: int((pr[f"t_{s}_right"] & pr[f"n_{s}_right"]).sum()) for s in ("reworded", "original")},
            "t_wrong_reworded": pr.loc[~pr.t_reworded_right, ["report_id", "t", "t_reworded"]].to_dict("records"),
        },
        "staging_codes_in_reworded_text": {rid: CODES.findall(t) for rid, t in items.report.items() if CODES.findall(t)},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    pr.to_csv(RESULTS / "parser_rows.csv", index=False, lineterminator="\n")
    (RESULTS / "details.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("answers", "parser", "staging_codes_in_reworded_text")}, indent=1,
                     default=str))
    return out


if __name__ == "__main__":
    main()

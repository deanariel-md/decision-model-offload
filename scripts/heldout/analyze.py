"""Arm 3 held-out reports, analysis: arm 3's own functions and settings on the reports of data/arm3_heldout/items.csv.
No model call; stored answers only (results/arm3_heldout/calls.parquet, documented_calls.parquet, parts.parquet;
scripts/heldout/run.py).
  - Jev as documented, both versions: the docuse_staging analysis path (scripts/docuse/analyze_staging.py: each version's
    answers beside the chatbots' names answers and beside their definitions answers, categorical_analysis.analyze,
    staging_extra, error_filter, parts_accuracy).
  - Jev as run originally and the five chatbots: categorical_analysis.analyze on each version (names, definitions), with
    analyze_staging.staging_extra for main-stage accuracy and stage III answered 0 to II.
Settings are config/arm3.yaml's (its seed, 2,000 resamples stratified by level, 95% intervals, margin 5 points, Holm
across the five main chatbots). Writes results/arm3_heldout/{analysis,staging}_<jev arm>_<version>.json, parts_*.json and
summary.json (each number with the file and key it comes from); --check re-reads every summary number from its file.
  python scripts/heldout/analyze.py [--n_boot N] [--root DIR]
  python scripts/heldout/analyze.py --check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "docuse"))
from jevity import categorical_analysis as KA
from jevity import docuse_staging as D
from jevity import heldout as H
from jevity.clients import MODELS, system_tier

import analyze_staging as AS

VERSIONS = ("names", "definitions")
REF = re.compile(r"[\w.]+\.json [\w.-]+")          # "file.json dotted.key": where a summary number comes from
JEV_ARMS = ("original",) + D.VARIANTS          # original (arm 3's Score request); documented; documented_notes


def settings(n_boot: int | None) -> dict:
    cfg = H.heldout_arm().config
    return cfg if n_boot is None else {**cfg, "analysis": {**cfg["analysis"], "n_boot": int(n_boot)}}


def chatbots_present(calls: pd.DataFrame, v: str) -> list[str]:
    have = set(calls[(calls.variant == v) & (calls.system != "jev")].system)
    return [s for s in H.MAIN_CHATBOTS if s in have]


def jev_frame(calls: pd.DataFrame, doc: pd.DataFrame, arm: str, v: str) -> pd.DataFrame:
    """The chatbots' answers of version v with one Jev arm's answers as system 'jev' (original: arm 3's Score request
    of the same version; documented versions: analyze_staging.combined)."""
    if arm == "original":
        return calls[calls.variant == v]
    return AS.combined(calls, doc, arm, v)


def run(root: Path = ROOT, n_boot: int | None = None) -> dict:
    res_dir = root / "results" / "arm3_heldout"
    frame = H.items_frame(H.load_items_frame(root / "data" / "arm3_heldout" / "items.csv"))
    calls = pd.read_parquet(res_dir / "calls.parquet")
    doc = pd.read_parquet(res_dir / "documented_calls.parquet")
    parts = pd.read_parquet(res_dir / "parts.parquet")
    cfg = settings(n_boot)
    nb, seed = int(cfg["analysis"]["n_boot"]), int(cfg["seed"])
    summary = {"n_reports": len(frame), "reports_by_level": frame.groupby("stratum").size().to_dict(),
               "settings": {"seed": seed, "n_boot": nb, "margin": cfg["analysis"]["margin"],
                            "holm_family": list(H.MAIN_CHATBOTS), "source": "config/arm3.yaml"},
               "chatbots_missing": {}, "jev_arms": {}, "chatbots": {}, "checker": {}}
    chat_acc = {}
    for arm in JEV_ARMS:
        summary["jev_arms"][arm] = {}
        for v in VERSIONS:
            chat = chatbots_present(calls, v)
            summary["chatbots_missing"][v] = [s for s in H.MAIN_CHATBOTS if s not in chat]
            order = ["jev"] + chat
            tiers = {s: ("jev" if s == "jev" else system_tier(s)) for s in order}
            comb = jev_frame(calls, doc, arm, v)
            res = KA.analyze(comb, frame[["item_id", "truth", "stratum", "group"]], cfg, D.LEVELS, order, tiers, MODELS,
                             None, None, v)
            res.update(jev_arm=arm, chatbots=f"held-out {v} answers",
                       missing_treated_as_unusable={s: c["missing"] for s, c in res["completeness"].items() if c["missing"]})
            st = {"variant": v, "jev_arm": arm,
                  "systems": AS.staging_extra(comb, frame, order, v, nb, seed)}
            if arm != "original":
                st["error_filter"] = AS.error_filter(comb, frame, v)
            fa, fs = f"analysis_{arm}_{v}.json", f"staging_{arm}_{v}.json"
            (res_dir / fa).write_text(json.dumps(res, indent=1), encoding="utf-8")
            (res_dir / fs).write_text(json.dumps(KA.clean(st), indent=1), encoding="utf-8")
            # checker: the chatbots' numbers are the same whichever Jev arm sits beside them
            for s in chat:
                a = res["systems"][s]["exact_accuracy"]
                if (s, v) in chat_acc and abs(chat_acc[(s, v)] - a) > 1e-12:
                    raise SystemExit(f"checker: {s} {v} exact accuracy {a} differs between Jev arms")
                chat_acc[(s, v)] = a
            j = res["systems"]["jev"]
            summary["jev_arms"][arm][v] = {
                "exact_accuracy": [j["exact_accuracy"], f"{fa} systems.jev.exact_accuracy"],
                "exact_accuracy_ci": [j["exact_accuracy_ci"], f"{fa} systems.jev.exact_accuracy_ci"],
                "unusable": [j["unusable"], f"{fa} systems.jev.unusable"],
                "main_stage_accuracy": [st["systems"]["jev"]["main_stage_accuracy"],
                                        f"{fs} systems.jev.main_stage_accuracy"],
                "main_stage_accuracy_ci": [st["systems"]["jev"]["main_stage_accuracy_ci"],
                                           f"{fs} systems.jev.main_stage_accuracy_ci"],
                "stage_iii_to_0_ii": [st["systems"]["jev"]["stage_iii_to_0_ii"], f"{fs} systems.jev.stage_iii_to_0_ii"],
                "stage_iii_n": [st["systems"]["jev"]["stage_iii_n"], f"{fs} systems.jev.stage_iii_n"],
                "usd_list_per_1000": [j["cost"]["usd_list_per_1000_answers"],
                                      f"{fa} systems.jev.cost.usd_list_per_1000_answers"],
                "vs_chatbots": {s: {k: [c[k], f"{fa} comparisons.{s}.{k}"]
                                    for k in ("diff", "ci", "holm_lower_bound", "outcome")}
                                for s, c in res["comparisons"].items() if c["confirmatory"]}}
            if arm == "original":
                summary["chatbots"][v] = {s: {
                    "exact_accuracy": [res["systems"][s]["exact_accuracy"], f"{fa} systems.{s}.exact_accuracy"],
                    "exact_accuracy_ci": [res["systems"][s]["exact_accuracy_ci"], f"{fa} systems.{s}.exact_accuracy_ci"],
                    "unusable": [res["systems"][s]["unusable"], f"{fa} systems.{s}.unusable"],
                    "main_stage_accuracy": [st["systems"][s]["main_stage_accuracy"],
                                            f"{fs} systems.{s}.main_stage_accuracy"],
                    "stage_iii_to_0_ii": [st["systems"][s]["stage_iii_to_0_ii"], f"{fs} systems.{s}.stage_iii_to_0_ii"],
                    "stage_iii_n": [st["systems"][s]["stage_iii_n"], f"{fs} systems.{s}.stage_iii_n"],
                    "usd_list_per_1000": [res["systems"][s]["cost"]["usd_list_per_1000_answers"],
                                          f"{fa} systems.{s}.cost.usd_list_per_1000_answers"]} for s in chat}
    truth = H.truth_parts(H.load_items_frame(root / "data" / "arm3_heldout" / "items.csv"))
    for jv in D.VARIANTS:
        pa = {"jev_variant": jv, "main_pass": AS.parts_accuracy(parts[(parts.variant == jv) & (parts["repeat"] == 0)],
                                                                 truth)}
        (res_dir / f"parts_{jv}.json").write_text(json.dumps(KA.clean(pa), indent=1), encoding="utf-8")
        summary["jev_arms"][jv]["parts_accuracy"] = {q: [pa["main_pass"][q]["accuracy"], f"parts_{jv}.json main_pass.{q}.accuracy"]
                                                     for q in D.QUESTION_IDS}
    summary["checker"] = {"chatbot_accuracy_same_across_jev_arms": True, "summary_values_match_files": check(summary, res_dir)}
    (res_dir / "summary.json").write_text(json.dumps(KA.clean(summary), indent=1), encoding="utf-8")
    return summary


def _resolve(res_dir: Path, ref: str):
    f, key = ref.split(" ", 1)
    v = json.loads((res_dir / f).read_text(encoding="utf-8"))
    for k in key.split("."):
        v = v[k]
    return v


def check(summary: dict, res_dir: Path) -> int:
    """Every [value, "file key"] pair in the summary equals the value at that key of that file; returns the count."""
    n = 0

    def walk(o):
        nonlocal n
        if isinstance(o, list) and len(o) == 2 and isinstance(o[1], str) and REF.fullmatch(o[1]):
            got = _resolve(res_dir, o[1])
            if KA.clean(got) != KA.clean(o[0]):
                raise SystemExit(f"checker: summary value {o[0]!r} differs from {o[1]} ({got!r})")
            n += 1
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(summary)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_boot", type=int, default=None, help="fewer resamples for a quick check only")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--check", action="store_true", help="re-read every summary number from its file")
    a = ap.parse_args(argv)
    res_dir = a.root / "results" / "arm3_heldout"
    if a.check:
        s = json.loads((res_dir / "summary.json").read_text(encoding="utf-8"))
        print(f"checker: {check(s, res_dir)} summary numbers match their files")
        return s
    s = run(a.root, a.n_boot)
    print(json.dumps(KA.clean({arm: {v: {k: x[k][0] for k in ("exact_accuracy", "exact_accuracy_ci", "main_stage_accuracy",
                                                                "usd_list_per_1000")}
                                      for v, x in d.items() if v in VERSIONS}
                                for arm, d in s["jev_arms"].items()}), indent=1))
    return s


if __name__ == "__main__":
    main()

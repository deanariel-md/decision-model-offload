"""Arm 3 supplement tables: repeatability on the repeat set and cost per system. Descriptive; the confirmatory outcomes
stay those of the shared analysis.

Repeatability, per system and question version, on the 40 reports asked three times (repeat 0 is the main pass): the
intraclass correlation ICC(2,1) of the level index (0 for stage 0 to 9 for IVC; two-way random effects, absolute
agreement, single answer; Shrout and Fleiss 1979) with the three answers as the raters, and the share of reports
answered identically all three times. Only reports with three usable answers enter (the shared analysis's stability
block counts the same reports); the others are counted. 95% percentile intervals from resampling reports (the primary
analysis's number of resamples and seed); a resample whose ICC is undefined (no variance at all) is counted, not used.

Cost, per system, on the main pass (the shared analysis's table, jevity.categorical_analysis.Table): input, output and
reasoning tokens per answer (reasoning as reported in the raw reply, where the provider reports it; output includes
it); billed and list cost per 1,000 answers; list cost of one million answers; list cost per correct answer with the
primary analysis's resamples; the extra list cost per extra correct answer against Jev only where the chatbot is more
accurate (otherwise Jev is reported as cheaper and at least as accurate, when it is); the hybrid (Jev answers the share of
reports with its highest top probability, the chatbot the rest; config analysis.hybrid_share, whose test is the shared
analysis's) and its list-cost saving against the chatbot alone; median and 90th-percentile seconds from the timing
sample once it has run.

On the names version: exact accuracy by report format (synoptic, narrative) and the component behind each wrong answer.
The same share handed to Jev at random and the coverage curve come from the shared analysis: its routing table is
results/arm3/table_routing.md, and the coverage table here prints its analysis.json numbers. Pure functions;
scripts/arm3_supplement.py assembles them from the stored calls."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


# ------------------------------------------------------------------------------------------------ repeatability
def icc_2_1(x: np.ndarray) -> float | None:
    """ICC(2,1) of an items x raters matrix without missing values: (BMS - EMS) / (BMS + (k-1) EMS + k (JMS - EMS) / n).
    None when it is undefined (fewer than two items or raters, or no variance at all)."""
    x = np.asarray(x, float)
    n, k = x.shape
    if n < 2 or k < 2:
        return None
    m = x.mean()
    ssr = k * ((x.mean(axis=1) - m) ** 2).sum()
    ssc = n * ((x.mean(axis=0) - m) ** 2).sum()
    sse = ((x - m) ** 2).sum() - ssr - ssc
    bms, jms, ems = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = bms + (k - 1) * ems + k * (jms - ems) / n
    return float((bms - ems) / den) if den > 1e-12 else None


def repeatability(levels: np.ndarray, n_boot: int, seed: int, ci: float = 0.95) -> dict:
    """levels: reports x repeats, the level index of each answer (NaN where the answer was unusable or missing).
    ICC(2,1) and the share answered identically on the reports with every answer usable, with percentile intervals
    from resampling those reports."""
    levels = np.asarray(levels, float)
    full = levels[~np.isnan(levels).any(axis=1)]
    n = len(full)
    out = {"reports": int(len(levels)), "reports_all_usable": n, "reports_with_unusable": int(len(levels) - n),
           "icc_2_1": None, "icc_2_1_ci": None, "identical": None, "identical_ci": None, "resamples_icc_undefined": None}
    if n == 0:
        return out
    same = (full == full[:, :1]).all(axis=1)
    out.update(icc_2_1=icc_2_1(full), identical=float(same.mean()))
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    iccs = np.array([np.nan if (v := icc_2_1(full[i])) is None else v for i in idx])
    ok = ~np.isnan(iccs)
    q = [(1 - ci) / 2, 1 - (1 - ci) / 2]
    out["resamples_icc_undefined"] = int((~ok).sum())
    if ok.any():
        out["icc_2_1_ci"] = [float(v) for v in np.quantile(iccs[ok], q)]
    out["identical_ci"] = [float(v) for v in np.quantile(same[idx].mean(axis=1), q)]
    return out


# ------------------------------------------------------------------------------------------------ tokens and value
def reasoning_tokens(raw: Mapping | None) -> float | None:
    """Reasoning tokens as the raw reply reports them (OpenRouter and OpenAI chat: usage.completion_tokens_details;
    OpenAI responses: usage.output_tokens_details; Gemini: usageMetadata.thoughtsTokenCount); None if not reported."""
    if not isinstance(raw, Mapping):
        return None
    r = raw.get("response", raw)
    if not isinstance(r, Mapping):
        return None
    u = r.get("usage") or {}
    for path in (("completion_tokens_details", "reasoning_tokens"), ("output_tokens_details", "reasoning_tokens"),
                 ("reasoning_tokens",)):
        v = u
        for key in path:
            v = v.get(key) if isinstance(v, Mapping) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    v = (r.get("usageMetadata") or {}).get("thoughtsTokenCount")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def value_verdict(correct_jev: int, correct_chat: int, usd_jev: float, usd_chat: float) -> str:
    """The extra list cost per extra correct answer is priced only where the chatbot is more accurate; otherwise Jev is
    reported as cheaper and at least as accurate (when it is cheaper)."""
    if correct_chat > correct_jev:
        return "chatbot more accurate: extra cost per extra correct answer"
    if usd_jev <= usd_chat:
        return "Jev cheaper and at least as accurate"
    return "Jev at least as accurate, chatbot cheaper"


# ------------------------------------------------------------------------------------------------ report format
def _interval(x: np.ndarray, ci: float) -> list[float]:
    return [float(v) for v in np.quantile(x, [(1 - ci) / 2, 1 - (1 - ci) / 2])]


def accuracy_by_format(correct: np.ndarray, fmt: np.ndarray, idx: np.ndarray, ci: float = 0.95,
                       formats: Sequence[str] = ("synoptic", "narrative")) -> dict:
    """Exact accuracy per report format and the difference (first format minus second), with percentile intervals
    from `idx`, resamples drawn within level x format so each resample keeps every format's size."""
    c, fmt = np.asarray(correct, float), np.asarray(fmt)
    out: dict = {}
    boot = {}
    for f in formats:
        m = fmt == f
        mb = (fmt[idx] == f)
        boot[f] = (c[idx] * mb).sum(axis=1) / np.maximum(mb.sum(axis=1), 1)
        out[f] = {"n": int(m.sum()), "accuracy": float(c[m].mean()) if m.any() else None,
                  "accuracy_ci": _interval(boot[f], ci) if m.any() else None}
    a, b = formats
    out["diff"] = (None if out[a]["accuracy"] is None or out[b]["accuracy"] is None
                   else out[a]["accuracy"] - out[b]["accuracy"])
    out["diff_ci"] = None if out["diff"] is None else _interval(boot[a] - boot[b], ci)
    out["diff_is"] = f"{a} minus {b}"
    return out


# ------------------------------------------------------------------------------------------------ wrong answers
COMPONENTS = ("T", "N", "M")


def components_to_reach(t: str, n: str, m: str, level: str) -> list[str]:
    """The component(s) that must change, from the true T, N and M, to reach the nearest combination the AJCC 8th
    table places in the answered level: the fewest components changed, every tie listed ("T", "N", "M", "T+N", ...;
    ties sorted). Empty if the true cell is already in that level."""
    from jevity import crc
    best: set[str] = set()
    k_best = 4
    for cell, stage in crc.STAGE_TABLE.items():
        if crc.level_of(stage) != level:
            continue
        changed = "+".join(c for c, a, b in zip(COMPONENTS, (t, n, m), cell) if a != b)
        k = 0 if not changed else changed.count("+") + 1
        if k < k_best:
            best, k_best = {changed}, k
        elif k == k_best:
            best.add(changed)
    if k_best == 0:
        return []
    return sorted(best, key=lambda x: [COMPONENTS.index(c) for c in x.split("+")])


def component_label(sets: Sequence[str]) -> str:
    """"T", or "T or N" when two component sets tie."""
    return " or ".join(sets)


def wrong_answer_components(truth_tnm: Sequence[tuple[str, str, str]], answer_level: Sequence[str | None],
                            truth_level: Sequence[str]) -> dict:
    """Counts, over one system's wrong answers, of the component label behind each (components_to_reach); an unusable
    answer (no level) is counted apart."""
    counts: dict[str, int] = {}
    wrong = unusable = 0
    for (t, n, m), ans, tr in zip(truth_tnm, answer_level, truth_level):
        if ans is None:
            unusable += 1
            continue
        if ans == tr:
            continue
        wrong += 1
        lab = component_label(components_to_reach(t, n, m, ans))
        counts[lab] = counts.get(lab, 0) + 1
    order = sorted(counts, key=lambda x: (x.count(" or "), [COMPONENTS.index(c) for c in x.replace(" or ", "+").split("+")]))
    return {"wrong_with_level": wrong, "unusable": unusable, "by_component": {k: counts[k] for k in order}}


# ------------------------------------------------------------------------------------------------ tables
def _ci(v: Sequence | None, fmt: str) -> str:
    return "" if v is None else f" ({fmt.format(v[0])} to {fmt.format(v[1])})"


def _num(v, fmt: str) -> str:
    return "n/a" if v is None else fmt.format(v)


def tables_markdown(results: Mapping[str, Mapping], shared: Mapping[str, Mapping] | None = None,
                    title: str = "Arm 3 supplement tables", script: str = "scripts/arm3_supplement.py",
                    results_dir: str = "results/arm3", unit: str = "reports") -> str:
    """The supplement tables for every question version, from the written results and, for the coverage curve, the
    shared analysis of each version (`shared`, keyed by version; nothing typed by hand). `unit`: what an item is called
    (arm 3 "reports"; the eICU AKI set "records")."""
    L = [f"# {title}", "",
         f"Generated by `{script}` from `{results_dir}/supplement_<version>.json` and, for the coverage "
         "curve, the shared analysis; do not edit by hand.",
         "Descriptive. Intervals are 95% bootstrap percentile intervals.", ""]
    for v, r in results.items():
        sim = " (simulated dry run)" if r.get("simulated") else ""
        L += [f"## Version: {v}{sim}", "",
              f"### Repeatability on the repeat set ({r['repeat_set']} {unit}, three answers each)", "",
              f"| System | {unit.capitalize()} with three usable answers | Answered identically | ICC(2,1), level index |",
              "|---|---|---|---|"]
        for s, x in r["repeatability"].items():
            L.append(f"| {s} | {x['reports_all_usable']} of {x['reports']} | "
                     f"{_num(x['identical'], '{:.1%}')}{_ci(x['identical_ci'], '{:.1%}')} | "
                     f"{_num(x['icc_2_1'], '{:.3f}')}{_ci(x['icc_2_1_ci'], '{:.3f}')} |")
        L += ["", f"### Tokens and cost per answer (main pass, {r['n_items']} {unit})", "",
              "| System | Input tokens | Output tokens | Of which reasoning | Billed USD per 1,000 answers | "
              "List USD per 1,000 answers | List USD per million answers | List USD per correct answer |",
              "|---|---|---|---|---|---|---|---|"]
        for s, x in r["cost"].items():
            rt = x["reasoning_tokens_per_answer"]
            L.append(f"| {s} | {_num(x['input_tokens_per_answer'], '{:,.0f}')} | "
                     f"{_num(x['output_tokens_per_answer'], '{:,.0f}')} | "
                     f"{'not reported' if rt is None else f'{rt:,.0f}'} | "
                     f"{_num(x['usd_billed_per_1000_answers'], '{:.4f}')} | {_num(x['usd_list_per_1000_answers'], '{:.4f}')} | "
                     f"{_num(x['usd_list_per_million_answers'], '{:,.2f}')} | "
                     f"{_num(x['usd_list_per_correct'], '{:.6f}')}{_ci(x['usd_list_per_correct_ci'], '{:.6f}')} |")
        L += ["", "### Each chatbot against Jev (list prices)", "",
              "| Chatbot | Exact accuracy, Jev | Exact accuracy, chatbot | Extra USD per extra correct answer |",
              "|---|---|---|---|"]
        for s, x in r["value"].items():
            cell = (f"{_num(x['usd_per_extra_correct'], '{:.4f}')}{_ci(x['usd_per_extra_correct_ci'], '{:.4f}')}"
                    if x["chatbot_more_accurate"] else x["verdict"])
            L.append(f"| {s} | {x['accuracy_jev']:.3f} | {x['accuracy_chatbot']:.3f} | {cell} |")
        h = r["hybrid"]
        L += ["", f"### Hybrid: Jev answers the {h['n_jev']} {unit} ({h['jev_share']:.0%}) with its highest top "
                  "probability, the chatbot the rest (list prices)", "",
              f"| Chatbot | Exact accuracy, hybrid | Exact accuracy, chatbot alone | USD per 1,000 {unit}, hybrid | "
              f"USD per 1,000 {unit}, chatbot alone | Saving per 1,000 {unit} |", "|---|---|---|---|---|---|"]
        for s, x in h["systems"].items():
            L.append(f"| {s} | {x['accuracy_hybrid']:.3f} | {x['accuracy_chatbot']:.3f} | "
                     f"{x['usd_hybrid_per_1000']:.4f} | {x['usd_chatbot_per_1000']:.4f} | "
                     f"{x['saving_usd_per_1000']:.4f} ({_num(x['saving_share'], '{:.0%}')}) |")
        cov = ((shared or {}).get(v) or {}).get("coverage_curve")
        if cov:                                   # the shared analysis's numbers
            L += ["", f"### Coverage curve: Jev's share of the {unit}, routed by confidence and at random (list prices)", "",
                  "From the shared analysis (`coverage_curve`); the random-half control at the hybrid's share is its "
                  "routing table (`table_routing`).", "",
                  "| Chatbot | Jev's share | Exact accuracy, by confidence | Exact accuracy, at random | "
                  "USD per 1,000 answers, by confidence | USD per 1,000 answers, at random |", "|---|---|---|---|---|---|"]
            for s, pts in cov["systems"].items():
                for x in pts:
                    L.append(f"| {s} | {x['share_jev']:.0%} | {x['exact_accuracy_confidence_routed']:.3f} | "
                             f"{x['exact_accuracy_random_routed']:.3f} | {x['usd_list_per_1000_confidence_routed']:.4f} | "
                             f"{x['usd_list_per_1000_random_routed']:.4f} |")
        if r.get("by_format"):
            L += ["", "### Exact accuracy by report format", "",
                  "| System | Synoptic | Narrative | Difference, synoptic minus narrative |", "|---|---|---|---|"]
            for s, x in r["by_format"].items():
                cells = [f"{_num(x[f]['accuracy'], '{:.3f}')}{_ci(x[f]['accuracy_ci'], '{:.3f}')} (n = {x[f]['n']})"
                         for f in ("synoptic", "narrative")]
                L.append(f"| {s} | {cells[0]} | {cells[1]} | {_num(x['diff'], '{:+.3f}')}{_ci(x['diff_ci'], '{:+.3f}')} |")
        if r.get("components"):
            labels = []
            for x in r["components"].values():
                labels += [k for k in x["by_component"] if k not in labels]
            labels.sort(key=lambda k: (k.count(" or "), k.count("+"), k))
            L += ["", "### Wrong answers by the component behind them", "",
                  "The fewest of T, N and M that must change, from the true T, N and M, to reach a combination the AJCC "
                  "8th table places in the answered stage group. Ties are joined by \"or\".", "",
                  "| System | Wrong answers | " + " | ".join(labels) + " | No usable answer |",
                  "|---|---|" + "---|" * len(labels) + "---|"]
            for s, x in r["components"].items():
                L.append(f"| {s} | {x['wrong_with_level']} | "
                         + " | ".join(str(x["by_component"].get(k, 0)) for k in labels) + f" | {x['unusable']} |")
        if r.get("confident_half"):
            c = r["confident_half"]
            L += ["", f"### Jev's confident half: the {c['n_confident']} {unit} with its highest top probability", "",
                  "Of Jev's correct answers, the share in its confident half (kept by the hybrid); of its wrong answers "
                  "(unusable ones included), the share in its confident half (let through).", "",
                  "| Correct answers | Kept | Wrong answers | Let through | Exact accuracy, confident half | "
                  "Exact accuracy, other half |", "|---|---|---|---|---|---|",
                  f"| {c['correct']} | {_num(c['kept'], '{:.1%}')}{_ci(c['kept_ci'], '{:.1%}')} | {c['wrong']} | "
                  f"{_num(c['let_through'], '{:.1%}')}{_ci(c['let_through_ci'], '{:.1%}')} | "
                  f"{_num(c['accuracy_confident'], '{:.3f}')}{_ci(c['accuracy_confident_ci'], '{:.3f}')} | "
                  f"{_num(c['accuracy_other'], '{:.3f}')}{_ci(c['accuracy_other_ci'], '{:.3f}')} |"]
        if r.get("confusion"):
            L += ["", "### Confusion by stage (rows: the key; columns: the answer)", ""]
            for s, x in r["confusion"].items():
                cols = x["answers"]
                L += [f"**{s}**", "", "| Key | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
                for k, row in x["counts"].items():
                    L.append(f"| {k} | " + " | ".join(str(row[a]) for a in cols) + " |")
                L.append("")
        L += ["", "### Speed (timing sample, standard route, one window)", ""]
        if not r["speed"]:
            L += ["The timing sample has not run yet.", ""]
            continue
        L += ["| System | Answers | Median seconds | 90th percentile seconds |", "|---|---|---|---|"]
        for s, x in r["speed"].items():
            L.append(f"| {s} | {x['n']} | {_num(x['median_s'], '{:.1f}')} | {_num(x['p90_s'], '{:.1f}')} |")
        L.append("")
    return "\n".join(L).rstrip() + "\n"

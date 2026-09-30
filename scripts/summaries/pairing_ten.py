"""Jev kept at a confidence threshold, an LLM otherwise: every task, all ten LLMs, one Bonferroni family of ten.

Built from the stored answers. At each confidence threshold from 0.50 to 1.00 in steps of 0.01, Jev's answer stands
where its top probability reaches the threshold and the LLM's answer is used otherwise. Cost counts Jev on every item
and the LLM on the items passed on, at list price. Differences from the LLM alone come from 2,000 paired bootstrap
resamples, stratified by message type (patient messages), stage level (kidney-injury staging) or report set (cancer
staging); board questions are resampled without strata. The adjusted lower bound is one-sided at 0.025, Bonferroni
over the ten LLMs.

Tasks, in order; the bootstrap seed of task i is [SEED, i]:
  0  Test-or-treat                       results/arm2/calls.parquet
  1  Board                               data/arm2_ext/items.csv, results/arm2_ext/calls.parquet
  2  Board, other count                  as task 1
  3  Kidney, criteria                    data/eicu_aki/items.csv, results/eicu_aki/calls.parquet
  4  Kidney, names                       as task 3
  5  Colon stage 240, recommended form   results/arm3_llm_forms/calls.parquet, data/arm3_pool/pool240.csv
A task whose input files are absent is skipped and written nowhere; the other tasks keep their index and seed.

Writes, under --out (default results/summaries, relative to --root):
  pairing_thresholds.json        {"th": thresholds, "d": {task: {LLM: one row per threshold}}}; each row in tenths of
                                 a point: [difference from the LLM alone, 95% bootstrap interval low, high, adjusted
                                 lower bound]
  pairing_curves.json            {task: {LLM: {"rows": one row per threshold, [threshold, share of items Jev answers,
                                 share of the LLM-only cost saved, added errors per item, difference in points],
                                 "always_jev": [cost saved, added errors per item, 0.0] with Jev answering every item,
                                 "usd_alone_per_1000": the LLM alone, US$ per 1,000 items}}}
  pairing_full_confidence.json   the pairing at full confidence (1.00), per task: label, n, per LLM (accuracy alone
                                 and paired, difference, 95% interval, adjusted lower bound, cost as a share of the
                                 LLM-only cost, US$ per 1,000 items paired and alone), kept, kept_wrong
  pairing_lowest_threshold.json  for the four tasks of Supplementary Fig. 1, the lowest threshold at which the
                                 pairing was non-inferior to each of the ten LLMs, at that threshold and at every
                                 higher one: the adjusted lower bound, rounded to tenths of a point as in
                                 pairing_thresholds.json, above -5 points

    python scripts/summaries/pairing_ten.py [--root .] [--out results/summaries]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260929
TOL = 1e-9
MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
FREE = ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
TEN = MAIN + FREE
NAME = {"gpt": "GPT-5.6 Sol", "claude": "Claude Opus 5.5", "gemini": "Gemini 3.8 Flash", "muse": "Muse Spark 1.1",
        "glm": "GLM-5.3", "gpt_free": "GPT-5.6 Luna", "claude_free": "Claude Sonnet 5",
        "gemini_free": "Gemini 3.5 Flash-Lite", "medgemma": "MedGemma 27B", "gemma": "Gemma 3 27B"}
# list prices, US$ per million input and output tokens
PRICE = {"jev": (0.042, 0.0), "gpt": (5.0, 30.0), "claude": (4.0, 20.0), "gemini": (0.75, 3.75), "muse": (1.25, 4.25),
         "glm": (1.4, 4.4), "gpt_free": (0.20, 1.20), "claude_free": (2.00, 10.00), "gemini_free": (0.30, 2.50),
         "medgemma": (0.265, 0.65), "gemma": (0.10, 0.30)}
TH = np.round(np.arange(0.50, 1.0001, 0.01), 2)
B = 2000
K = len(TEN)
MARGIN = -5.0
FIGURE_TASKS = ["Test-or-treat", "Board", "Colon stage 240, recommended form", "Kidney, criteria"]

MESSAGES_CALLS = "results/arm2/calls.parquet"
BOARD_ITEMS = "data/arm2_ext/items.csv"
BOARD_CALLS = "results/arm2_ext/calls.parquet"
KIDNEY_ITEMS = "data/eicu_aki/items.csv"
KIDNEY_CALLS = "results/eicu_aki/calls.parquet"
STAGING_CALLS = "results/arm3_llm_forms/calls.parquet"
POOL = "data/arm3_pool/pool240.csv"
TASKS = [("Test-or-treat", (MESSAGES_CALLS,)),
         ("Board", (BOARD_ITEMS, BOARD_CALLS)),
         ("Board, other count", (BOARD_ITEMS, BOARD_CALLS)),
         ("Kidney, criteria", (KIDNEY_ITEMS, KIDNEY_CALLS)),
         ("Kidney, names", (KIDNEY_ITEMS, KIDNEY_CALLS)),
         ("Colon stage 240, recommended form", (STAGING_CALLS, POOL))]


def tob(x):
    return x.astype(str).str.lower().eq("true").to_numpy() if x.dtype == object else x.fillna(False).astype(bool).to_numpy()


def sel(c, s, v, ids):
    return c[(c.system == s) & (c.variant == v) & (c["repeat"] == 0)].drop_duplicates("item_id").set_index("item_id").reindex(ids)


def usd(d, s):
    return (pd.to_numeric(d.tokens_in, errors="coerce").fillna(0).to_numpy() * PRICE[s][0]
            + pd.to_numeric(d.tokens_out, errors="coerce").fillna(0).to_numpy() * PRICE[s][1]) / 1e6


def load_tasks(root: Path) -> list:
    """The six tasks in order, as (key, task); task is None where an input file is absent. A task is (label, ids,
    truth, strata, balanced, Jev's answers, each LLM's answers, cost source)."""
    have = {key: all((root / p).exists() for p in paths) for key, paths in TASKS}
    out = {}
    if have["Test-or-treat"]:
        c = pd.read_parquet(root / MESSAGES_CALLS)
        ids = pd.Index(sorted(c[(c.system == "jev") & (c["repeat"] == 0)].item_id.unique()))
        t = pd.Series(np.where(ids.str.contains("_lv_"), "B", "A"), index=ids)
        out["Test-or-treat"] = ("Patient messages (balanced accuracy)", ids, t.to_numpy(), t.to_numpy(), True,
                                sel(c, "jev", "main", ids), {s: sel(c, s, "main", ids) for s in TEN}, None)
    if have["Board"]:
        it = pd.read_csv(root / BOARD_ITEMS, dtype=str).set_index("item_id")
        c = pd.read_parquet(root / BOARD_CALLS)
        for v, tr, lab, key in [("medhelm", it.truth, "Board questions", "Board"),
                                ("other_count", it.truth_alt, "Board questions, other option count", "Board, other count")]:
            ids = tr.index
            out[key] = (lab, ids, tr.to_numpy(), np.zeros(len(ids)), False, sel(c, "jev", v, ids),
                        {s: sel(c, s, v, ids) for s in TEN}, None)
    if have["Kidney, criteria"]:
        it = pd.read_csv(root / KIDNEY_ITEMS, dtype=str).set_index("item_id")
        c = pd.read_parquet(root / KIDNEY_CALLS)
        for v, lab, key in [("definitions", "Kidney-injury staging, with KDIGO criteria", "Kidney, criteria"),
                            ("names", "Kidney-injury staging, stage names", "Kidney, names")]:
            ids = it.index
            out[key] = (lab, ids, it.stage.to_numpy(), it.stage.to_numpy(), False, sel(c, "jev", v, ids),
                        {s: sel(c, s, v, ids) for s in TEN}, None)
    if have["Colon stage 240, recommended form"]:
        # Jev in the recommended form, each LLM with stage names; cost from the stored list-price cost
        cs = pd.read_parquet(root / STAGING_CALLS)
        pool = pd.read_csv(root / POOL, dtype=str)
        ids = pd.Index(pool.report_id)
        sets = pool.set_index("report_id").set.reindex(ids).to_numpy()

        def sel3(s, f):
            return cs[(cs.system == s) & (cs.form == f)].drop_duplicates("item_id", keep="last").set_index("item_id").reindex(ids)

        J3 = sel3("jev", "documented")
        out["Colon stage 240, recommended form"] = ("Cancer staging, recommended form (240 reports)", ids,
                                                    J3.truth.to_numpy(), sets, False, J3,
                                                    {s: sel3(s, "names") for s in TEN}, "store")
    return [(key, out.get(key)) for key, _ in TASKS]


def lowest_threshold(rows: dict) -> float | None:
    """rows: {LLM: its rows in pairing_thresholds.json}, thresholds ascending. The lowest threshold at which every
    LLM's adjusted lower bound, in tenths of a point as written there, lies above the margin, at that threshold and at
    every higher one; None if full confidence fails."""
    lb = np.array([[r[3] for r in x] for x in rows.values()])
    ok = (lb > 10 * MARGIN).all(axis=0)
    i = len(TH)
    while i > 0 and ok[i - 1]:
        i -= 1
    return float(TH[i]) if i < len(TH) else None


def run(root: Path) -> tuple[dict, dict, dict, dict]:
    ni, curves, p7 = {"th": TH.tolist(), "d": {}}, {}, {}
    for ti, (key, task) in enumerate(load_tasks(root)):
        if task is None:
            print(f"{key}: input files absent, skipped")
            continue
        lab, ids, truth, strata, bal, J, Ls, costsrc = task
        N = len(ids)
        jv = tob(J.valid)
        jok = jv & (J.answer.to_numpy() == truth)
        jp = np.where(jv, pd.to_numeric(J.top_prob, errors="coerce").fillna(0).to_numpy(), -1)
        cj = pd.to_numeric(J.usd_list, errors="coerce").to_numpy() if costsrc == "store" else usd(J, "jev")
        rng = np.random.default_rng([SEED, ti])
        groups = [np.where(strata == g)[0] for g in np.unique(strata)]
        W = np.zeros((B, N))
        for g in groups:
            ix = rng.integers(0, len(g), size=(B, len(g)))
            for b in range(B):
                np.add.at(W[b], g[ix[b]], 1)

        def acc_w(P):  # P (T,N) or (N,): accuracy on each resample
            P2 = np.atleast_2d(P).astype(float)
            if bal:
                return np.mean([(W[:, g] @ P2[:, g].T) / W[:, g].sum(1, keepdims=True) for g in groups], axis=0)
            return (W @ P2.T) / N

        def acc_pt(P):
            P2 = np.atleast_2d(P).astype(float)
            return np.mean([P2[:, g].mean(1) for g in groups], axis=0) if bal else P2.mean(1)

        ni["d"][key] = {}
        curves[key] = {}
        p7[key] = {"label": lab, "n": N, "llm": {}}
        for s in TEN:
            L = Ls[s]
            lv = tob(L.valid)
            lok = (lv & (L.answer.to_numpy() == truth)).astype(float)
            cl = pd.to_numeric(L.usd_list, errors="coerce").to_numpy() if costsrc == "store" else usd(L, s)
            P = np.array([np.where(jp >= th - TOL, jok, lok) for th in TH]).astype(float)
            D = 100 * (acc_w(P) - acc_w(lok)[:, :1])            # (B,T)
            pt = acc_pt(P) - acc_pt(lok)[0]
            kept = np.array([(jp >= th - TOL).mean() for th in TH])
            cost = np.array([(cj.sum() + cl[~(jp >= th - TOL)].sum()) / cl.sum() for th in TH])
            err = np.array([((1 - P[i]).sum() - (1 - lok).sum()) / N for i in range(len(TH))])
            ni["d"][key][NAME[s]] = [[int(round(1000 * pt[i])), int(round(10 * np.percentile(D[:, i], 2.5))),
                                      int(round(10 * np.percentile(D[:, i], 97.5))),
                                      int(round(10 * np.percentile(D[:, i], 100 * 0.025 / K)))] for i in range(len(TH))]
            curves[key][NAME[s]] = {"rows": [[float(TH[i]), float(kept[i]), float(1 - cost[i]), float(err[i]), float(100 * pt[i])]
                                             for i in range(len(TH))],
                                    "always_jev": [float(1 - cj.sum() / cl.sum()), float(((1 - jok).sum() - (1 - lok).sum()) / N), 0.0],
                                    "usd_alone_per_1000": float(cl.sum() / N * 1000)}
            i1 = len(TH) - 1
            keep = jp >= 1 - TOL
            p7[key]["llm"][s] = {"alone": float(100 * acc_pt(lok)[0]), "pair": float(100 * acc_pt(P[i1])[0]),
                                 "diff": float(100 * pt[i1]),
                                 "ci": [float(np.percentile(D[:, i1], 2.5)), float(np.percentile(D[:, i1], 97.5))],
                                 "lb": float(np.percentile(D[:, i1], 100 * 0.025 / K)), "cost_pct": float(100 * cost[i1]),
                                 "usd_pair": float((cj.sum() + cl[~keep].sum()) / N * 1000),
                                 "usd_alone": float(cl.sum() / N * 1000)}
        p7[key]["kept"] = int((jp >= 1 - TOL).sum())
        p7[key]["kept_wrong"] = int(((jp >= 1 - TOL) & ~jok).sum())
    lowest = {key: lowest_threshold(ni["d"][key]) for key in FIGURE_TASKS if key in ni["d"]}
    return ni, curves, p7, lowest


def main(root: Path, out: Path) -> None:
    ni, curves, p7, lowest = run(root)
    out.mkdir(parents=True, exist_ok=True)
    for name, obj, indent in [("pairing_thresholds.json", ni, None), ("pairing_curves.json", curves, None),
                              ("pairing_full_confidence.json", p7, 1), ("pairing_lowest_threshold.json", lowest, None)]:
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=indent)
    for key, v in p7.items():
        print(key, "kept", v["kept"], "wrong", v["kept_wrong"])
        for s, x in v["llm"].items():
            print(f"   {s:12s} {x['pair']:.1f}/{x['alone']:.1f} diff {x['diff']:+.1f} ({x['ci'][0]:+.1f} to {x['ci'][1]:+.1f})"
                  f" lb {x['lb']:+.1f} cost {x['cost_pct']:.0f}%")
    print("lowest non-inferior threshold:", lowest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/summaries")
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out)
    main(root, out if out.is_absolute() else root / out)

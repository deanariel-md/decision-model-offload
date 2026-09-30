"""Supplementary Fig. 1 (stored results only): confidence thresholds, accuracy and cost of the pairing (Jev answers the
items whose confidence reaches the threshold, the LLM the rest). Reads pairing_thresholds.json (difference from the LLM
alone and its 95% bootstrap interval at each threshold, per LLM), pairing_curves.json (share answered by Jev and cost
share at each threshold) and pairing_lowest_threshold.json (the lowest threshold non-inferior with each of the ten
LLMs).

    python scripts/figures/supplementary_figures.py [--root .] [--out figures]

Paths are relative to --root. Writes fig_s1_thresholds.png into --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def supp_fig1(root: Path, out: Path):
    S = root / "results" / "summaries"
    ni = load(S / "pairing_thresholds.json"); cv = load(S / "pairing_curves.json"); TH = np.array(ni["th"]); LOW = load(S / "pairing_lowest_threshold.json")
    L = ["GPT-5.6 Sol", "Claude Opus 5.5", "Gemini 3.8 Flash", "Muse Spark 1.1", "GLM-5.3"]
    LF = ["GPT-5.6 Luna", "Claude Sonnet 5", "Gemini 3.5 Flash-Lite", "MedGemma 27B", "Gemma 3 27B"]
    P = [("Patient questions", "Test-or-treat", LOW["Test-or-treat"]), ("Board questions", "Board", LOW["Board"]),
         ("Cancer staging\n(recommended form)", "Colon stage 240, recommended form", LOW["Colon stage 240, recommended form"]), ("Kidney-injury staging\n(with KDIGO criteria)", "Kidney, criteria", LOW["Kidney, criteria"])]
    F = lambda v: '0.0' if abs(v) < 0.05 else f'{v:+.1f}'.replace('-', '−')
    R0 = lambda x: int(x + 0.5); BLUE = "#3f3f3f"; JEV = "#eb6834"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888"})
    fig, axs = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True)
    for j, (lab, k, thr) in enumerate(P):
        a = axs[j]; res = ni["d"][k]
        kept = np.array([100 * r[1] for r in cv[k][L[0]]["rows"]])
        pt = np.array([[x[0] / 10 for x in res[s]] for s in L]); lo = np.array([[x[1] / 10 for x in res[s]] for s in L]); hi = np.array([[x[2] / 10 for x in res[s]] for s in L])
        cost = np.array([[100 * (1 - r[2]) for r in cv[k][s]["rows"]] for s in L])
        o = np.argsort(kept, kind="stable")
        a.axhline(0, color="#333", lw=0.8); a.axhline(-5, color="#555", lw=1, ls="--")
        loF = np.array([[x[1] / 10 for x in res[s]] for s in LF]); hiF = np.array([[x[2] / 10 for x in res[s]] for s in LF])
        a.fill_between(kept[o], loF.min(0)[o], hiF.max(0)[o], color="#a3a3a3", alpha=0.25, lw=0)
        a.fill_between(kept[o], lo.min(0)[o], hi.max(0)[o], color=BLUE, alpha=0.18, lw=0)
        a.plot(kept[o], np.median(pt, 0)[o], color=BLUE, lw=2)

        def box(t, xy, txt, style):
            i = int(np.argmin(abs(TH - t))); x = kept[i]; y = np.median(pt[:, i])
            a.plot(x, y, "o", ms=8.5, mfc=style, mec=JEV, mew=1.8, zorder=5)
            a.annotate(txt.format(t=t, k=R0(x), c0=R0(cost[:, i].min()), c1=R0(cost[:, i].max()), d0=F(pt[:, i].min()), d1=F(pt[:, i].max())), (x, y), xytext=xy, textcoords="axes fraction",
                       fontsize=7.8, va="bottom", arrowprops=dict(arrowstyle="-", color=JEV, lw=0.9), bbox=dict(boxstyle="round,pad=0.3", fc="#fdf5f0", ec="#f0b89c"))
        box(1.00, (0.03, 0.80), "Full confidence (1.00)\nJev answers {k}%\nCost {c0}–{c1}% of the LLM-only cost\nDifference {d0} to {d1}", "white")
        box(thr, (0.03, 0.05), "Lowest non-inferior threshold, {t:.2f}\nJev answers {k}%\nCost {c0}–{c1}% of the LLM-only cost\nDifference {d0} to {d1}", JEV)
        a.set_xlim(0, 100); a.set_ylim(-20, 8); a.set_title(lab, fontsize=10, fontweight="bold", loc="left")
        a.set_xlabel("Cases answered by Jev, %"); a.grid(axis="y", color="#eee", lw=0.6)
    axs[0].set_ylabel("Difference from the LLM alone,\npercentage points")
    fig.text(0.29, 0.965, "Categorical tasks", ha="center", fontsize=10.5, fontweight="bold", color="#333")
    fig.text(0.765, 0.965, "Ordinal tasks", ha="center", fontsize=10.5, fontweight="bold", color="#333")
    fig.tight_layout(rect=(0, 0.0, 1, 0.94)); fig.savefig(out / "fig_s1_thresholds.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="folder holding results/ (default: the current folder)")
    ap.add_argument("--out", default="figures", help="output folder (default: figures)")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    supp_fig1(root, out)
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

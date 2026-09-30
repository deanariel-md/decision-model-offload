"""Figure 2: Jev in the recommended form, and with an LLM, on the 240 pathology reports (stored results only).

  a  cancer staging, the four forms of question (direct stage assignment or four questions with the stage assigned in
     code; without or with medical context), as a heatmap shaded by Jev's accuracy: Jev's accuracy, its share of
     answers at full confidence, and the ranges of the five flagship and the five smaller LLMs given the same form
     (staging_pool240.json)
  b  recommended form: correct and wrong answers by whether they reached full confidence (staging_pool240.json)
  c  Jev when fully confident, each of the ten LLMs otherwise (stage names): accuracy difference from the LLM alone,
     95% bootstrap interval; arrows mark differences beyond the axis (staging_pool240.json)
  d  the same pairings: cost as a share of the LLM-only cost (staging_pool240.json)

    python scripts/figures/figure2.py [--root .] [--out figures]

Reads the result files under --root; writes fig2_main.pdf and fig2_main.png into --out and prints every number drawn.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtrans  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# ---- house style
MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
CHEAP = ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
NAME = {"jev": "Jev 1.13", "gpt": "GPT-5.6 Sol", "claude": "Claude Opus 5.5", "gemini": "Gemini 3.8 Flash",
        "muse": "Muse Spark 1.1", "glm": "GLM-5.3", "gpt_free": "GPT-5.6 Luna", "claude_free": "Claude Sonnet 5",
        "gemini_free": "Gemini 3.5 Flash-Lite", "medgemma": "MedGemma 27B", "gemma": "Gemma 3 27B"}
# colour follows the entity: Jev, flagship LLMs, smaller LLMs
JEV_P, CHAT, CHEAPC = "#eb6834", "#3f3f3f", "#a3a3a3"
INK, INK2, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
    "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False, "legend.fontsize": 6.5,
})
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["font.family"] = "Helvetica"
matplotlib.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

DASH = "–"   # en dash, in Helvetica's standard encoding
TL, AL = 6.0, 6.3
PASS = "#cfcfcf"
LAB = 6.1
DRAWN: list[str] = []   # every number drawn, for the check log


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def note(s: str):
    DRAWN.append(s)


def panel(fig, ax, letter: str, title: str, fx: float, dy: float = 0.035):
    """Panel letter at figure x = fx, title beside it, both just above the axes."""
    y = ax.get_position().y1 + dy
    fig.text(fx, y, letter, fontsize=8, fontweight="bold", va="bottom", ha="left")
    fig.text(fx + 0.018, y, title, fontsize=7, va="bottom", ha="left")


def xgrid(ax):
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def mk_rect(H):
    return lambda x0, y0, x1, y1: [x0 / 183, y0 / H, (x1 - x0) / 183, (y1 - y0) / H]


def style(a_, xlabel=None, grid=True):
    if grid:
        xgrid(a_)
    a_.tick_params(axis="x", labelsize=TL, length=2, pad=1.5)
    a_.tick_params(axis="y", length=0, pad=3)
    if xlabel:
        a_.set_xlabel(xlabel, fontsize=AL, labelpad=2, linespacing=0.95)


def check_extents(fig, name):
    """Every drawn element (each axes with its tick labels, titles and texts; figure texts; legends) inside the figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    items = [(f"axes {i} ({ax.get_xlabel()[:30]!r})", ax.get_tightbbox(r)) for i, ax in enumerate(fig.axes)]
    items += [(f"text {t.get_text()[:30]!r}", t.get_window_extent(r)) for t in fig.texts]
    items += [("figure legend", lg.get_window_extent(r)) for lg in fig.legends]
    bad = [(k, [round(v) for v in (b.x0, b.y0, b.x1, b.y1)]) for k, b in items
           if b.x0 < fb.x0 - 0.5 or b.y0 < fb.y0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y1 > fb.y1 + 0.5]
    mx = min(min(b.x0 for _, b in items) - fb.x0, fb.x1 - max(b.x1 for _, b in items))
    print(f"{name}: " + ("every element inside the figure" if not bad else "OUTSIDE THE FIGURE: " + str(bad))
          + f"; smallest side margin {mx / fig.dpi * 25.4:.1f} mm")
    return not bad


# ---- Figure 2
def fig2(root: Path, out: Path):
    W = root / "results" / "summaries"
    P240 = load(W / "staging_pool240.json")
    H = 126
    R = mk_rect(H)
    fig = plt.figure(figsize=(183 * MM, H * MM))

    # a: 2 x 2 of question forms
    pv = {v: P240["acc"][f"jev|{v}"]["pool240"] for v in ("names", "definitions", "structure_only", "documented")}
    n_st = P240["n"]
    key = {"names": "names", "definitions": "definitions", "structure_only": "structure_only", "structure_definitions": "documented"}
    forms = {k: (pv[v]["p"],) for k, v in key.items()}
    fc = {k: pv[v]["full"] / pv[v]["n"] for k, v in key.items()}
    ax = fig.add_axes(R(36, 72, 90, 110))
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.axis("off")
    HEAT = LinearSegmentedColormap.from_list("jev_heat", ["#fbf3ee", "#f6cdb8", "#ee9e78"])
    tiles = [("names", 0, 1), ("definitions", 0, 0), ("structure_only", 1, 1), ("structure_definitions", 1, 0)]
    for k, cx, cy in tiles:
        col = HEAT(min(1.0, max(0.0, (100 * forms[k][0] - 40) / 55)))   # tile shade = Jev's accuracy (heatmap)
        dark = False   # light ramp: dark text on every tile
        rec = k == "structure_definitions"   # the recommended form, outlined
        ax.add_patch(Rectangle((cx + 0.03, cy + 0.04), 0.94, 0.92, facecolor=col,
                               edgecolor=INK if rec else "none", linewidth=1.0 if rec else 0, zorder=1))
        v, f = 100 * forms[k][0], 100 * fc[k]
        ax.text(cx + 0.5, cy + 0.70, f"{v:.1f}%", fontsize=10, fontweight="bold", ha="center", va="center",
                color="white" if dark else INK)
        ax.text(cx + 0.5, cy + 0.48, f"{f:.0f}% fully confident", fontsize=6, ha="center", va="center",
                color="white" if dark else INK2)
        for _grp, _lab, _y in ((MAIN, "Flagship", 0.30), (CHEAP, "Smaller", 0.15)):
            _lv = [P240["acc"].get(f"{m}|{key[k]}", {}).get("pool240", {}).get("p") for m in _grp]
            _lv = [x for x in _lv if x is not None]
            if _lv:
                _lo, _hi = 100 * min(_lv), 100 * max(_lv)
                _t = f"{_lab} {_lo:.1f}%" if round(_lo, 1) == round(_hi, 1) else f"{_lab} {_lo:.1f}{DASH}{_hi:.1f}%"
                ax.text(cx + 0.5, cy + _y, _t, fontsize=5.5, ha="center", va="center", color="white" if dark else INK2)
                note(f"2a {k} {_lab}: {_t} (n={len(_lv)})")
        note(f"2a {k}: {v:.1f}%; fully confident {f:.1f}%")
    ax.text(0.5, 2.06, "Without task\ndecomposition", fontsize=LAB, ha="center", va="bottom", linespacing=0.95)
    ax.text(1.5, 2.06, "With task\ndecomposition", fontsize=LAB, ha="center", va="bottom", linespacing=0.95)
    ax.text(-0.05, 1.5, "Without\nmedical context", fontsize=LAB, ha="right", va="center", linespacing=0.95)
    ax.text(-0.05, 0.5, "With\nmedical context", fontsize=LAB, ha="right", va="center", linespacing=0.95)
    panel(fig, ax, "a", f"Cancer staging ({n_st}) {DASH} form of question", 0.005, dy=0.085)

    # b: correct and wrong answers by full confidence, recommended form
    dv = pv["documented"]
    n_right, n_wrong = dv["k"], dv["n"] - dv["k"]
    full_right, full_wrong = dv["full"] - dv["full_err"], dv["full_err"]
    z = 1.959963984540054; ns = n_wrong; ks = n_wrong - full_wrong; ph = ks / ns   # Wilson interval, errors short of full confidence
    cen = (ph + z * z / (2 * ns)) / (1 + z * z / ns); half = z * ((ph * (1 - ph) / ns + z * z / (4 * ns * ns)) ** 0.5) / (1 + z * z / ns)
    se = {"p": ph, "ci": [cen - half, min(1.0, cen + half)]}
    bx = fig.add_axes(R(128, 80, 179, 97))
    rows = [(f"Correct\n({n_right:,})", full_right / n_right), (f"Wrong\n({n_wrong})", full_wrong / n_wrong)]
    for i, (lab, share) in enumerate(rows):
        y = 1 - i
        bx.barh(y, 100 * share, height=0.56, color=JEV_P, lw=0, zorder=2)
        bx.barh(y, 100 * (1 - share), left=100 * share, height=0.56, color=PASS, lw=0, zorder=2)
        if share > 0.08:
            bx.text(100 * share / 2, y, f"{100 * share:.0f}%", fontsize=6.2, color="white", ha="center", va="center", fontweight="bold")
        tail = f"{100 * (1 - share):.0f}%"
        if i == 1:
            tail += f" (sensitivity {100 * se['p']:.0f}%, 95% CI {100 * se['ci'][0]:.0f}{DASH}{100 * se['ci'][1]:.0f}%)"
        bx.text(100 * share + 100 * (1 - share) / 2, y, tail, fontsize=6.0, color=INK, ha="center", va="center")
        note(f"2b {lab!r}: full confidence {100 * share:.1f}%")
    bx.set_ylim(-0.55, 1.55); bx.set_yticks([1, 0]); bx.set_yticklabels([r[0] for r in rows], fontsize=LAB, linespacing=0.95)
    bx.set_xlim(0, 100); bx.set_xticks([0, 50, 100])
    for sp_ in ("top", "right", "left"):
        bx.spines[sp_].set_visible(False)
    style(bx, "% of Jev's answers", grid=False)
    bx.legend(handles=[Patch(color=JEV_P, label="Full confidence: Jev answers"),
                       Patch(color=PASS, label="Below full confidence: passed to the LLM")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, fontsize=5.8, handlelength=1.0, handletextpad=0.4,
              columnspacing=1.0, borderaxespad=0.2, frameon=False)
    panel(fig, bx, "b", f"Recommended form {DASH} confidence and correctness",
          0.505, dy=ax.get_position().y1 + 0.085 - bx.get_position().y1)

    # c, d: all ten LLMs, the five flagship LLMs then the five smaller LLMs
    cx_, dx_ = fig.add_axes(R(40, 11, 118, 58)), fig.add_axes(R(128, 11, 179, 58))
    XL, XR = -6.0, 6.5
    groups = [(MAIN, "Flagship LLMs", CHAT), (CHEAP, "Smaller LLMs", CHEAPC)]
    ys, labels, heads = [], [], []
    y = 0.0
    for keys, glab, col in groups:
        two = "\n" in glab
        heads.append((y + (0.0 if two else 0.15), glab)); y -= 1.3 if two else 1.0
        for m in keys:
            r = P240["pairing"][f"documented|{m}|names|pool240"]
            d, lo, hi = r["est"], r["ci"][0], r["ci"][1]
            if lo < XR - 2.5:
                cx_.plot([lo, hi], [y, y], color=col, lw=0.9, alpha=0.85, zorder=2, solid_capstyle="round")
                cx_.scatter(d, y, s=11, color=col, edgecolor="white", linewidth=0.35, zorder=3)
            else:   # beyond the axis: arrow and the difference
                cx_.annotate("", xy=(XR - 0.15, y), xytext=(XR - 1.9, y),
                             arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=col, lw=0.9), zorder=3)
                cx_.text(XR - 2.05, y, f"+{d:.1f}", fontsize=5.8, color=INK, ha="right", va="center")
            cs = 100 * r["cost_share"]
            dx_.scatter(cs, y, s=11, color=col, edgecolor="white", linewidth=0.35, zorder=3)
            if cs > 175:
                dx_.text(cs - 7, y, f"{cs:.0f}%", fontsize=5.8, color=INK, ha="right", va="center")
            else:
                dx_.text(cs + 7, y, f"{cs:.0f}%", fontsize=5.8, color=INK, ha="left", va="center")
            ys.append(y); labels.append(NAME[m])
            note(f"2c/d {m}: diff {d:+.2f} ({lo:+.2f} to {hi:+.2f}); Bonferroni-10 lower bound {r['lb_bonf']:+.2f}; cost {cs:.1f}%")
            y -= 1
        y -= 0.45
    lo_y, hi_y = min(ys) - 0.7, 0.75
    for a_ in (cx_, dx_):
        a_.set_ylim(lo_y, hi_y); a_.set_yticks(ys)
    cx_.set_yticklabels(labels, fontsize=LAB); dx_.set_yticklabels([])
    blend = mtrans.blended_transform_factory(cx_.transAxes, cx_.transData)
    for yy, glab in heads:
        cx_.text(-0.02, yy, glab, transform=blend, fontsize=LAB, fontweight="bold", ha="right", va="center", linespacing=0.95)
    sep = min(ys[:len(MAIN)]) - 0.72
    for a_ in (cx_, dx_):
        a_.axhline(sep, color=GRID, lw=0.6, zorder=0)
    cx_.axvline(0, color=INK2, lw=0.6, zorder=1)
    cx_.axvline(-5, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    cx_.text(-4.85, hi_y - 0.15, "Non-inferiority\nmargin", fontsize=5.6, color=INK2, ha="left", va="top", linespacing=0.9)
    cx_.set_xlim(XL, XR); cx_.set_xticks([-5, -2.5, 0, 2.5, 5])
    cx_.set_xticklabels(["−5", "−2.5", "0", "+2.5", "+5"])
    dx_.axvline(100, color=INK2, lw=0.7, ls=(0, (2, 1.5)), zorder=1)
    dx_.text(104, hi_y - 0.15, "LLM\nalone", fontsize=5.6, color=INK2, ha="left", va="top", linespacing=0.9)
    dx_.set_xlim(0, 225); dx_.set_xticks([0, 100, 200])
    style(cx_, "Accuracy of the hybrid system minus the LLM alone (percentage points)")
    style(dx_, "Hybrid system cost, % of the LLM-only cost")
    panel(fig, cx_, "c", f"Hybrid Jev-LLM system {DASH} accuracy", 0.005, dy=0.055)
    panel(fig, dx_, "d", f"Hybrid Jev-LLM system {DASH} cost", 0.683, dy=0.055)

    ok = check_extents(fig, "fig2")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig2_main.{ext}", dpi=450)
    plt.close(fig)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="folder holding results/ (default: the current folder)")
    ap.add_argument("--out", default="figures", help="output folder (default: figures)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = fig2(Path(a.root), out)
    print("\n".join(DRAWN))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Figure 1: Jev given standard LLM-style inputs, every system on every task (stored results only).

  a  risk of death, AUROC: ten-year risk in NHANES adults (risk_classic.json; the smaller LLMs from
     auroc_free_open_ci.json), with the reference model and age and sex only; death in hospital in intensive care
     (results/eicu/analysis.json: flagship LLMs from the baseline block, smaller LLMs (free-plan and open-weight
     models) from the secondary block), with APACHE IVa
  b  categorical tasks: patient questions (balanced accuracy) and board questions (accuracy)
  c  ordinal tasks: cancer staging on the 240 reports, stage names (staging_pool240.json), and kidney-injury staging
  d  cost per 1,000 answers (five tasks) and median time per answer (four timed tasks), one dot per task

    python scripts/figures/figure1.py [--root .] [--out figures]

Reads the result files under --root; writes fig1_main.pdf and fig1_main.png into --out and prints every number drawn.
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
from matplotlib.lines import Line2D  # noqa: E402

# ---- house style
MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
CHEAP = ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
NAME = {"jev": "Jev 1.13", "gpt": "GPT-5.6 Sol", "claude": "Claude Opus 5.5", "gemini": "Gemini 3.8 Flash",
        "muse": "Muse Spark 1.1", "glm": "GLM-5.3", "gpt_free": "GPT-5.6 Luna", "claude_free": "Claude Sonnet 5",
        "gemini_free": "Gemini 3.5 Flash-Lite", "medgemma": "MedGemma 27B", "gemma": "Gemma 3 27B"}
# colour follows the entity: Jev given the LLMs' text, Jev in the recommended form, flagship LLMs, smaller LLMs
JEV_P, JEV_S, CHAT, CHEAPC = "#eb6834", "#2a78d6", "#3f3f3f", "#a3a3a3"
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
DRAWN: list[str] = []   # every number drawn, for the check log


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def note(s: str):
    DRAWN.append(s)


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


def header(fig, a_, text, dy_mm, H):
    b = a_.get_position()
    fig.text((b.x0 + b.x1) / 2, b.y1 + dy_mm / H, text, fontsize=6.2, ha="center", va="bottom", color=INK, linespacing=0.95)


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


# ---- Figure 1
def fig1(root: Path, out: Path):
    W = root / "results" / "summaries"
    P240 = load(W / "staging_pool240.json")
    H = 152
    R = mk_rect(H)
    rc = load(W / "risk_classic.json")["systems"]
    fo = load(W / "auroc_free_open_ci.json")["systems"]
    ei = load(root / "results" / "eicu" / "analysis.json")["baseline_prediction"]["predictors"]
    es = load(root / "results" / "eicu" / "analysis.json")["secondary_prediction"]["predictors"]
    assert all(abs(es[k]["auroc"] - ei[k]["auroc"]) < 1e-12 for k in ["jev"] + MAIN), "secondary eICU block differs"
    a2 = load(root / "results" / "arm2" / "analysis.json")["systems"]
    bx = load(root / "results" / "arm2_ext" / "analysis.json")
    st = {"n_items": 240, "systems": {k: {"exact_accuracy": P240["acc"][f"{k}|names"]["pool240"]["p"]} for k in ["jev"] + MAIN + CHEAP if f"{k}|names" in P240["acc"]}}
    ak = load(root / "results" / "eicu_aki" / "analysis.json")
    order = ["jev"] + MAIN + CHEAP
    labels = ["Jev"] + [NAME[k] for k in MAIN + CHEAP]
    cols = [  # (title, values {system: v}, xlim, ticks, fmt, reference lines [(v, style, label)])
        (f"Ten-year risk\nof death ({load(W / 'risk_classic.json')['n']:,})",
         {**{k: rc[k]["auroc"] for k in ["jev"] + MAIN}, **{k: fo[k]["auroc"] for k in CHEAP}},
         (0.755, 0.865), [0.76, 0.80, 0.84], "{:.3f}",
         [(rc["reference_spline_logit"]["auroc"], (0, (3, 2)), "Reference model"), (rc["reference_age_sex"]["auroc"], (0, (1, 1.4)), "Age and sex")]),
        (f"Death in hospital,\nintensive care ({load(W / 'eicu_summary.json')['evaluation_stays']:,})",
         {**{k: ei[k]["auroc"] for k in ["jev"] + MAIN}, **{k: es[k]["auroc"] for k in CHEAP}}, (0.78, 0.895), [0.78, 0.81, 0.84, 0.87], "{:.3f}",
         [(ei["apache_iva"]["auroc"], (0, (3, 2)), "APACHE IVa")]),
        (f"Choosing Wisely\npatient questions ({load(root / 'results' / 'arm2' / 'analysis.json')['n_items']})",
         {k: 100 * a2[k]["balanced_accuracy"] for k in order}, (78, 102), [80, 90, 100], "{:.1f}", []),
        (f"Board\nquestions ({bx['n_items']:,})", {k: 100 * bx["systems"][k]["accuracy"] for k in order}, (58, 102), [60, 80, 100], "{:.1f}", []),
        (f"Cancer\nstaging ({st['n_items']:,})", {k: 100 * st["systems"][k]["exact_accuracy"] for k in order if k in st["systems"]}, (0, 104), [0, 50, 100], "{:.1f}", []),
        (f"Kidney-injury\nstaging ({ak['n_items']})", {k: 100 * ak["systems"][k]["exact_accuracy"] for k in order}, (38, 102), [40, 70, 100], "{:.1f}", []),
    ]
    fig = plt.figure(figsize=(183 * MM, H * MM))
    x0, wcol, gap = 30.5, 21.2, 3.2
    xs = []
    x = x0
    for i in range(6):
        if i in (2, 4):
            x += 3.5 if i == 2 else 3.0
        xs.append((x, x + wcol)); x += wcol + gap
    axes = [fig.add_axes(R(a, 84, b, 134)) for a, b in xs]
    ny = len(order)
    for j, (ttl, vals, xl, ticks, fmt, refs) in enumerate(cols):
        a_ = axes[j]
        a_.axhline(ny - 1 - 0.5, color=GRID, lw=0.6, zorder=0)
        a_.axhline(ny - 1 - len(MAIN) - 0.5, color=GRID, lw=0.6, zorder=0)
        for v, ls, _ in refs:
            a_.axvline(v, color=INK2, lw=0.7, ls=ls, zorder=1)
        for i, k in enumerate(order):
            y = ny - 1 - i
            if k not in vals:
                continue
            col = JEV_P if k == "jev" else (CHAT if k in MAIN else CHEAPC)
            a_.scatter(vals[k], y, s=17 if k == "jev" else 11, color=col, edgecolor="white", linewidth=0.4, zorder=3)
            note(f"1 {ttl.splitlines()[0]} {k}: {fmt.format(vals[k])}")
        vj = vals["jev"]
        a_.text(vj, ny - 1 + 0.55, fmt.format(vj).lstrip("0") if fmt == "{:.3f}" else fmt.format(vj), fontsize=5.6,
                color=JEV_P, ha="center", va="bottom", fontweight="bold")
        a_.set_xlim(*xl); a_.set_xticks(ticks)
        a_.set_xticklabels([(f"{t:.2f}".lstrip("0") if fmt == "{:.3f}" else f"{t:g}") for t in ticks])
        a_.set_ylim(-0.6, ny + 0.35)
        a_.set_yticks(range(ny))
        if j == 0:
            a_.set_yticklabels(labels[::-1], fontsize=6.1)
            for t, k in zip(a_.get_yticklabels(), order[::-1]):
                t.set_color(JEV_P if k == "jev" else (INK if k in MAIN else INK2))
                if k == "jev":
                    t.set_fontweight("bold")
        else:
            a_.set_yticklabels([])
        style(a_, "AUROC" if j < 2 else ("Balanced accuracy (%)" if j == 2 else "Accuracy (%)"))
        header(fig, a_, ttl, 1.2, H)
    fig.text(0.005, 142 / H, "a", fontsize=8, fontweight="bold", va="bottom")
    fig.text(0.023, 142 / H, "Risk of death", fontsize=7, va="bottom")
    fig.text(xs[2][0] / 183 - 0.012, 142 / H, "b", fontsize=8, fontweight="bold", va="bottom")
    fig.text(xs[2][0] / 183 + 0.006, 142 / H, "Categorical tasks", fontsize=7, va="bottom")
    fig.text(xs[4][0] / 183 - 0.012, 142 / H, "c", fontsize=8, fontweight="bold", va="bottom")
    fig.text(xs[4][0] / 183 + 0.006, 142 / H, "Ordinal tasks", fontsize=7, va="bottom")
    # d: cost and time per answer on every task with stored cost (five) and every timed task (four), each system
    J_ = lambda p_: load(root / p_)["systems"]
    cs = load(root / "results" / "eval" / "cost_speed.json")["systems"]
    costs = {k: [cs[k]["usd_list_per_1000_answers"]] + [J_(p_)[k]["cost"]["usd_list_per_1000_answers"] for p_ in
             ("results/arm2/analysis.json", "results/arm2_ext/analysis.json", "results/eicu_aki/analysis.json")]
             + ([P240["cost"][f"{k}|names"]] if P240["cost"].get(f"{k}|names") else [])
             for k in order}
    tm = load(W / "timing_summary.json")
    times = {k: [tm[t_][k]["median_s"] for t_ in ("risk", "next_step", "board_exam", "staging")] for k in order}
    cc = [fig.add_axes(R(30.5, 12, 99, 62)), fig.add_axes(R(110, 12, 181, 62))]
    for a_, vals, xl, ticks, tl in ((cc[0], costs, (0.01, 30), [0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"]),
                                     (cc[1], times, (0.5, 60), [0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])):
        a_.set_xscale("log")
        a_.axhline(ny - 1 - 0.5, color=GRID, lw=0.6, zorder=0)
        a_.axhline(ny - 1 - len(MAIN) - 0.5, color=GRID, lw=0.6, zorder=0)
        for i, k in enumerate(order):
            y = ny - 1 - i
            col = JEV_P if k == "jev" else (CHAT if k in MAIN else CHEAPC)
            v = vals[k]
            a_.plot([min(v), max(v)], [y, y], color=col, lw=1.3, alpha=0.45, zorder=2, solid_capstyle="round")
            a_.scatter(v, [y] * len(v), s=9 if k != "jev" else 13, color=col, edgecolor="white", linewidth=0.35, zorder=3)
            note(f"1d {'cost' if a_ is cc[0] else 'time'} {k}: " + ", ".join(f"{x:.3g}" for x in v))
        a_.set_xlim(*xl); a_.set_xticks(ticks); a_.set_xticklabels(tl); a_.minorticks_off()
        a_.set_ylim(-0.6, ny - 0.4)
        a_.set_yticks(range(ny))
        a_.grid(axis="x", color=GRID, lw=0.5, zorder=0); a_.set_axisbelow(True)
    cc[0].set_yticklabels(labels[::-1], fontsize=6.1)
    for t, k in zip(cc[0].get_yticklabels(), order[::-1]):
        t.set_color(JEV_P if k == "jev" else (INK if k in MAIN else INK2))
        if k == "jev":
            t.set_fontweight("bold")
    cc[1].set_yticklabels([])
    style(cc[0], "US$ per 1,000 answers, one dot per task (log scale)", grid=False)
    style(cc[1], "Seconds per answer, median, one dot per task (log scale)", grid=False)
    fig.text(0.005, 70 / H, "d", fontsize=8, fontweight="bold", va="bottom")
    fig.text(0.023, 70 / H, "Cost and time", fontsize=7, va="bottom")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=JEV_P, mec="white", ms=5, label="Jev, standard LLM-style inputs"),
                        Line2D([], [], marker="o", ls="", color=CHAT, mec="white", ms=5, label="Flagship LLMs"),
                        Line2D([], [], marker="o", ls="", color=CHEAPC, mec="white", ms=5, label="Smaller LLMs"),
                        Line2D([], [], color=INK2, lw=0.7, ls=(0, (3, 2)), label="Reference model; APACHE IVa"),
                        Line2D([], [], color=INK2, lw=0.7, ls=(0, (1, 1.4)), label="Age and sex only")],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=6.2, handletextpad=0.3, columnspacing=1.2,
               frameon=False)
    ok = check_extents(fig, "fig1")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_main.{ext}", dpi=450)
    plt.close(fig)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="folder holding results/ (default: the current folder)")
    ap.add_argument("--out", default="figures", help="output folder (default: figures)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = fig1(Path(a.root), out)
    print("\n".join(DRAWN))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Cost-accuracy frontier for arm 2 next-step choice or arm 3 cancer staging, from the stored outputs.

Every single system, Jev-first hybrids (arm 2: Jev keeps its answer when its top probability is 1.00; arm 3: Jev keeps
its most confident half; the chatbot answers the rest), cascades through the cheapest chatbot, and the ceiling of a
perfect router in front of GPT-5.6 Sol, placed by list cost per 1,000 answers and accuracy on the task's main pass
(balanced accuracy for arm 2, exact accuracy for arm 3). Costs are rebuilt per answer from its tokens at list price;
the script stops unless every price below equals the system's standard-tier price in config/models.yaml and each
system's total equals analysis.json's list total. The frontier is the set of options no other option beats on both
cost and accuracy. Also reports the chatbots' tokens per answer, alone and with Jev first.

  python scripts/summaries/cost_frontier.py arm2 [--dir results/arm2] [--out results/summaries]
    -> results/summaries/frontier_arm2.json (arm3: results/summaries/frontier_arm3.json)
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

PRICE = {  # US$ per million tokens (input, output), list price, standard tier
    "jev": (0.042, 0.0), "gpt": (5.0, 30.0), "claude": (4.0, 20.0), "gemini": (0.75, 3.75), "muse": (1.25, 4.25),
    "glm": (1.4, 4.4), "gpt_free": (0.2, 1.2), "claude_free": (2.0, 10.0), "gemini_free": (0.3, 2.5),
    "medgemma": (0.265, 0.65), "gemma": (0.1, 0.3)}  # MedGemma, Gemma: Featherless list prices via the Hugging Face router


def _check_prices(path="config/models.yaml"):
    """Stop unless every chatbot price above equals the first (standard-tier) price_in/price_out in the config."""
    import yaml
    found = {}
    def walk(o, name=None):
        if isinstance(o, dict):
            if "price_in" in o and name not in found:
                found[name] = (float(o["price_in"]), float(o["price_out"]))
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, name)
    walk(yaml.safe_load(Path(path).read_text()))
    for s, pr in PRICE.items():
        if s != "jev":
            assert found.get(s) == pr, (s, pr, found.get(s))
NAME = {"jev": "Jev 1.13", "gpt": "GPT-5.6 Sol", "claude": "Claude Opus 5.5", "gemini": "Gemini 3.8 Flash",
        "muse": "Muse Spark 1.1", "glm": "GLM-5.3", "gpt_free": "GPT-5.6 Luna", "claude_free": "Claude Sonnet 5",
        "gemini_free": "Gemini 3.5 Flash-Lite", "medgemma": "MedGemma 27B", "gemma": "Gemma 3 27B"}
MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
CHEAP = ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
def _arm3_truth(ids):  # the stage level each report was written from (data/arm3/items.csv)
    it = pd.read_csv("data/arm3/items.csv", dtype=str)
    lv = dict(zip(it["report_id"], it["level"]))
    return np.array([lv[i] for i in ids])


TASK = {"arm2": {"truth": lambda ids: np.where(pd.Series(ids).str.contains("_lv_"), "B", "A"),
                 "stratum": lambda ids: pd.Series(ids).str.contains("_lv_").values,
                 "metric": "balanced accuracy", "unit": "messages", "variant": "main", "jev_first": "p1"},
        "arm3": {"truth": _arm3_truth, "stratum": None, "metric": "exact accuracy", "unit": "reports", "variant": "names", "jev_first": "half"}}


def jev_first(P, rule):
    """p1: Jev keeps its answer when its top probability is 1.00 (descriptive cutoff, arm 2).
    half: the tested rule, Jev keeps its most confident 50%, ties by item id (arm 3)."""
    if rule == "p1":
        return P["jev"] >= 1 - 1e-9, "Jev (1.00)"
    order = sorted(P.index, key=lambda i: (-P.at[i, "jev"], i))
    keep = set(order[: len(order) // 2])
    return pd.Series([i in keep for i in P.index], index=P.index), "Jev (top 50%)"


def load(task, d):
    an = json.loads((d / "analysis.json").read_text())
    c = pd.read_parquet(d / "calls.parquet")
    c = c[(c["variant"] == TASK[task]["variant"]) & (c["repeat"] == 0) & c["system"].isin(PRICE)].copy()
    c["usd"] = [(ti * PRICE[s][0] + to * PRICE[s][1]) / 1e6 for s, ti, to in zip(c.system, c.tokens_in.fillna(0), c.tokens_out.fillna(0))]
    for s in PRICE:
        tot = c.loc[c.system == s, "usd"].sum()
        assert abs(tot - an["systems"][s]["cost"]["usd_list_total"]) < 1e-6, (s, tot)
    ids = sorted(c.item_id.unique())
    truth = pd.Series(TASK[task]["truth"](ids), index=ids)
    c["correct"] = c["answer"].values == truth.reindex(c.item_id).values
    piv = lambda v: c.pivot_table(index="item_id", columns="system", values=v, aggfunc="first").reindex(ids)
    c["tok_out"] = c["tokens_out"].fillna(0); c["tok_in"] = c["tokens_in"].fillna(0)
    return an, piv("correct").astype(bool), piv("usd"), piv("top_prob"), (pd.Series(TASK[task]["stratum"](ids), index=ids) if TASK[task]["stratum"] else None), piv("tok_out"), piv("tok_in")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASK))
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default="results/summaries")
    a = ap.parse_args()
    _check_prices()
    d = Path(a.dir or f"results/{a.task}"); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    an, C, U, P, S, TO, TI = load(a.task, d)
    n = len(C)
    acc = (lambda x: float((x[S].mean() + x[~S].mean()) / 2)) if S is not None else (lambda x: float(x.mean()))
    per1000 = lambda u: float(1000 * u.sum() / n)
    pts = []
    for s in ["jev"] + MAIN + CHEAP:
        pts.append({"id": s, "kind": "single", "label": NAME[s], "accuracy": acc(C[s]), "usd_per_1000": per1000(U[s]),
                    "group": "jev" if s == "jev" else ("main" if s in MAIN else "cheap")})
    k1, jlab = jev_first(P, TASK[a.task]["jev_first"])
    for s in MAIN + ["gpt_free"]:
        h = pd.Series(np.where(k1, C["jev"], C[s]), index=C.index)
        pts.append({"id": f"jev1+{s}", "kind": "hybrid", "label": f"{jlab} + {NAME[s]}", "accuracy": acc(h),
                    "usd_per_1000": per1000(U["jev"]) + per1000(U[s].where(~k1, 0)), "partner": s, "share_jev": float(k1.mean())})
    kl = P["gpt_free"] >= 0.95 - 1e-9
    h = pd.Series(np.where(kl, C["gpt_free"], C["gpt"]), index=C.index)
    pts.append({"id": "luna95>gpt", "kind": "cascade", "label": "GPT-5.6 Luna (≥0.95), else GPT-5.6 Sol", "accuracy": acc(h),
                "usd_per_1000": per1000(U["gpt_free"]) + per1000(U["gpt"].where(~kl, 0)), "share_first": float(kl.mean())})
    km = (~k1) & kl
    h = pd.Series(np.where(k1, C["jev"], np.where(km, C["gpt_free"], C["gpt"])), index=C.index)
    pts.append({"id": "jev1>luna95>gpt", "kind": "cascade", "label": f"{jlab}, else Luna (≥0.95), else Sol", "accuracy": acc(h),
                "usd_per_1000": per1000(U["jev"]) + per1000(U["gpt_free"].where(~k1, 0)) + per1000(U["gpt"].where(~(k1 | km), 0))})
    kc = C["jev"]
    h = pd.Series(np.where(kc, True, C["gpt"]), index=C.index)
    pts.append({"id": "oracle+gpt", "kind": "ceiling", "label": "Perfect router: Jev when right, else Sol", "accuracy": acc(h),
                "usd_per_1000": per1000(U["jev"]) + per1000(U["gpt"].where(~kc, 0))})
    real = [p for p in pts if p["kind"] != "ceiling"]
    for p in real:
        p["on_frontier"] = not any((q["accuracy"] >= p["accuracy"] and q["usd_per_1000"] <= p["usd_per_1000"]) and
                                   (q["accuracy"] > p["accuracy"] or q["usd_per_1000"] < p["usd_per_1000"]) for q in real)
    base = {s: next(p for p in pts if p["id"] == s)["usd_per_1000"] for s in MAIN + ["gpt_free"]}
    for p in pts:
        if p["kind"] == "hybrid": p["saving_vs_partner"] = 1 - p["usd_per_1000"] / base[p["partner"]]
        p["saving_vs_gpt"] = 1 - p["usd_per_1000"] / base["gpt"]
        p["usd_per_million"] = 1000 * p["usd_per_1000"]
    tok = {}
    for s_ in MAIN + CHEAP:
        tok[s_] = {"output_per_answer_alone": float(TO[s_].mean()),
                   "output_per_answer_with_jev_first": float(TO[s_].where(~k1, 0).mean()),
                   "input_per_answer": float(TI[s_].mean())}
    tok["jev"] = {"input_per_answer": float(TI["jev"].mean()), "output_per_answer_alone": float(TO["jev"].mean()), "output_billed": False}
    res_tokens = {"note": "chatbot output tokens (including reasoning) per message; with Jev first the chatbot writes only for the messages Jev passes on; tokenizers differ between developers", "systems": tok}
    res = {"task": a.task, "tokens": res_tokens, "metric": TASK[a.task]["metric"], "unit": TASK[a.task]["unit"], "share_jev_at_1": float(k1.mean()), "jev_first_rule": TASK[a.task]["jev_first"], "n_items": n, "cost_basis": "list price, standard tier",
           "note": "descriptive; every option scored on the same main-pass answers", "points": pts,
           "by_id": {p["id"]: p for p in pts}}
    (out / f"frontier_{a.task}.json").write_text(json.dumps(res, indent=1))
    for p in sorted(pts, key=lambda p: p["usd_per_1000"]):
        print(f'{p["label"]:45s} {100*p["accuracy"]:6.1f}  ${p["usd_per_1000"]:.3f}/1000  {"FRONTIER" if p.get("on_frontier") else ""}')


if __name__ == "__main__":
    main()

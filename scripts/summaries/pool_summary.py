"""Pools and the Jev staircase in colon cancer staging. Stored answers only; no model call.

C2. Jev's exact accuracy (95% Wilson interval, sens_spec.prop), share at full confidence (top probability
>= 1 - 1e-9; for the decomposed versions the joint top probability) and errors at full confidence, for pool 240
(data/arm3_pool/pool240.csv), all 1,320, the ordinary 1,200 and the difficult 120, under each version in this order:
names, named field, definitions, structure-only, documented without examples, documented. An unusable answer counts
as wrong and is never at full confidence. The named-field and no-examples versions enter when
results/arm3_jev_parts/calls.parquet exists (scripts/arm3_jev_parts.py).
C3. Recommended-form pairing on pool 240: read from workflow_groups.json (group pool240;
scripts/summaries/workflow_groups.py --pool240).
C4. Drop-in pairing on pool 240 and on all 1,320, the code path of hybrid_all.py (chain.half_rule
picks Jev's most confident half by top probability among its usable answers, ties by report id; hybrid_all.score and
boot_diff; cost = Jev on every report plus the LLM on the reports Jev passes on, at cost_frontier.PRICE): Jev with names,
and Jev with definitions, each main LLM with names otherwise (the main LLMs have no definitions answers on the 120
difficult reports). Difference from the LLM alone, 95% percentile interval and the Bonferroni one-sided lower bound
(alpha 0.025 / 5) from 2,000 paired resamples stratified by report set within the group (hybrid_all's seed), and cost
as a share of the LLM alone.
C5. results/summaries/pool_summary.json and pool_summary.md.

  python scripts/summaries/pool_summary.py      (after workflow_groups.py)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_all import B, SEED, score, boot_diff  # noqa: E402
from chain import half_rule  # noqa: E402
from cost_frontier import PRICE, MAIN  # noqa: E402
from sens_spec import prop  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-9
SETS = ["main", "heldout", "misleading_feature", "nonregional_sites"]
GROUPS = {"pool240": None, "all_1320": SETS, "ordinary_1200": ["main", "heldout"],
          "difficult_120": ["misleading_feature", "nonregional_sites"]}
VERSIONS = ["names", "named_field", "definitions", "structure_only", "documented_no_examples", "documented"]
LABEL = {"names": "Names (drop-in)", "named_field": "Names, report as a named field",
         "definitions": "Definitions (T, N and M in the labels)", "structure_only": "Structure only",
         "documented_no_examples": "Documented, without examples", "documented": "Documented (recommended form)"}
COLS = ["item_id", "answer", "valid", "top_prob", "tokens_in", "tokens_out"]


def truth() -> pd.Series:
    f = {"main": ROOT / "data/arm3/items.csv", "heldout": ROOT / "data/arm3_heldout/items.csv",
         "misleading_feature": ROOT / "data/arm3/confuser_items.csv",
         "nonregional_sites": ROOT / "data/arm3_nonregional_sites/items.csv"}
    t = pd.concat([pd.read_csv(p, dtype=str)[["report_id", "level"]].assign(set=s) for s, p in f.items()])
    assert len(t) == 1320 and not t.report_id.duplicated().any()
    return t.set_index("report_id")


def jev(c: pd.DataFrame, variant: str, **eq) -> pd.DataFrame:
    c = c[(c.system == "jev") & (c.variant == variant) & (c["repeat"] == 0)]
    for k, v in eq.items():
        c = c[c[k] == v]
    return c[COLS]


def versions(branch: Path) -> dict[str, pd.DataFrame]:
    """Jev's main-pass answers per version, all 1,320 reports each (stops if a report is missing or doubled)."""
    rd = pd.read_parquet
    a3, cf = rd(ROOT / "results/arm3/calls.parquet"), rd(ROOT / "results/arm3_confuser/calls.parquet")
    hc = rd(ROOT / "results/arm3_heldout/calls.parquet")
    st = rd(branch / "results/arm3_jev_staircase/calls.parquet")
    so = ROOT / "results/arm3_docuse_structure_only"
    dc, dh = rd(ROOT / "results/arm3_docuse/calls.parquet"), rd(ROOT / "results/arm3_heldout/documented_calls.parquet")
    dn = rd(ROOT / "results/arm3_nonregional_sites/calls_jev.parquet")
    out = {"names": [jev(a3, "names"), jev(hc, "names"), jev(cf, "names"), jev(st, "names", set="nonregional")],
           "definitions": [jev(a3, "definitions"), jev(hc, "definitions"), jev(st, "definitions", set="confuser"),
                           jev(st, "definitions", set="nonregional")],
           "structure_only": [jev(rd(so / k / "calls.parquet"), "structure_only") for k in ("main", "heldout", "confuser", "nonregional")],
           "documented": [jev(dc, "documented"), jev(dh, "documented"), jev(dn, "documented")]}
    parts = branch / "results/arm3_jev_parts/calls.parquet"
    if parts.exists():
        p = rd(parts)
        out["named_field"] = [jev(p, "named_field")]
        out["documented_no_examples"] = [jev(p, "documented_no_examples")]
    res = {}
    for v, fs in out.items():
        d = pd.concat(fs)
        assert len(d) == 1320 and not d.item_id.duplicated().any(), (v, len(d))
        res[v] = d.set_index("item_id")
    return res


def staircase(T: pd.Series, V: dict, groups: dict) -> dict:
    out = {}
    for g, ids in groups.items():
        out[g] = {"n": len(ids), "versions": {}}
        for v in [x for x in VERSIONS if x in V]:
            d = V[v].reindex(ids)
            ok = (d.valid.fillna(False).astype(bool) & (d.answer == T.level.reindex(ids))).to_numpy()
            full = (d.valid.fillna(False).astype(bool) & (d.top_prob.astype(float) >= 1 - TOL)).to_numpy()
            out[g]["versions"][v] = {"accuracy": prop(ok.sum(), len(ids)), "full_confidence_share": float(full.mean()),
                                     "full_confidence_n": int(full.sum()), "errors_at_full_confidence": int((full & ~ok).sum()),
                                     "unusable": int((~d.valid.fillna(False).astype(bool)).sum())}
    return out


def llm_names() -> dict[str, pd.DataFrame]:
    rd = pd.read_parquet
    fs = [rd(ROOT / "results/arm3/calls.parquet"), rd(ROOT / "results/arm3_heldout/calls.parquet"),
          rd(ROOT / "results/arm3_confuser/calls.parquet"), rd(ROOT / "results/arm3_nonregional_sites/calls_chatbots.parquet")]
    out = {}
    for s in MAIN:
        d = pd.concat([f[(f.system == s) & (f.variant == "names") & (f["repeat"] == 0)][COLS] for f in fs])
        assert len(d) == 1320 and not d.item_id.duplicated().any(), (s, len(d))
        out[s] = d.set_index("item_id")
    return out


def usd(d: pd.DataFrame, s: str) -> np.ndarray:
    return ((d.tokens_in.fillna(0) * PRICE[s][0] + d.tokens_out.fillna(0) * PRICE[s][1]) / 1e6).to_numpy()


def dropin(T: pd.Series, V: dict, L: dict, groups: dict, rng) -> dict:
    """The drop-in pairing path of hybrid_all.py on a group of reports; resamples stratified by set, shared across versions and LLMs."""
    out = {}
    for g, ids in groups.items():
        sets = T.set.reindex(ids).to_numpy()
        idx = np.concatenate([np.flatnonzero(sets == k)[rng.integers(0, (sets == k).sum(), size=(B, (sets == k).sum()))]
                              for k in SETS if (sets == k).any()], axis=1)
        truth = T.level.reindex(ids)
        out[g] = {"n": len(ids), "versions": {}}
        for v in ("names", "definitions"):
            j = V[v].reindex(ids)
            jvalid = j.valid.fillna(False).astype(bool).to_numpy()
            keep = half_rule(j.top_prob.astype(float).to_numpy(), jvalid, list(ids))
            jok = (jvalid & (j.answer == truth).to_numpy())
            ju = usd(j, "jev")
            r = {"n_jev": int(keep.sum()), "jev_alone": float(jok.mean()), "llms": {}}
            for s in MAIN:
                b = L[s].reindex(ids)
                bok = (b.valid.fillna(False).astype(bool) & (b.answer == truth)).to_numpy()
                h = np.where(keep, jok, bok)
                bu = usd(b, s)
                d = boot_diff(h, bok, None, idx)
                r["llms"][s] = {"alone": float(score(bok, None)), "pair": float(score(h, None)),
                                "diff_pts": float(100 * (score(h, None) - score(bok, None))),
                                "ci_pts": [float(100 * np.quantile(d, 0.025)), float(100 * np.quantile(d, 0.975))],
                                "lower_bound_bonferroni_one_sided_pts": float(100 * np.quantile(d, 0.025 / len(MAIN))),
                                "cost_share": float((ju.sum() + bu[~keep].sum()) / bu.sum())}
            out[g]["versions"][v] = r
    return out


def md(res: dict) -> str:
    pc = lambda x: f"{100 * x:.1f}"
    L = ["# Pools and the Jev staircase, colon cancer stage (stored answers; descriptive)", "",
         f"Pool 240: {res['pool240']['n']} reports ({', '.join(f'{k} {v}' for k, v in res['pool240']['sets'].items())}); "
         f"seed {res['pool240']['seed']}; `data/arm3_pool/pool240.csv` sha256 {res['pool240']['sha256'][:16]}...", "",
         "## Jev staircase (C2, D3): exact accuracy (95% Wilson interval), share at full confidence, errors at full confidence", ""]
    for g, r in res["staircase"].items():
        L += [f"### {g.replace('_', ' ')} (n = {r['n']:,})", "", "| Version | Right | Accuracy % (95% CI) | Full confidence % | Errors at full confidence | Unusable |",
              "|---|---|---|---|---|---|"]
        for v, x in r["versions"].items():
            a = x["accuracy"]
            L.append(f"| {LABEL[v]} | {a['k']:,}/{a['n']:,} | {pc(a['p'])} ({pc(a['ci'][0])}–{pc(a['ci'][1])}) | "
                     f"{pc(x['full_confidence_share'])} ({x['full_confidence_n']}) | {x['errors_at_full_confidence']} | {x['unusable']} |")
        L.append("")
    w = res["recommended_pool240"]
    L += ["## Recommended form on pool 240 (C3): Jev documented when fully confident, each main LLM with names otherwise", "",
          f"Jev kept {w['kept']} of {w['n']}. Bootstrap stratified by set within the pool (2,000 resamples, seed {res['seeds']['c3']}). "
          "Time share is an estimate from the 100 timed first-1,000 reports.", "",
          "| LLM | Pair right | LLM alone right | Difference, pts (95% CI) | Bonferroni lower bound, pts | Cost share % | Time share % (est.) | Treatment-line crossings, pair / alone |",
          "|---|---|---|---|---|---|---|---|"]
    for s, x in w["llms"].items():
        L.append(f"| {s} | {x['pair_correct']} | {x['alone_correct']} | {x['diff_pts']:.2f} ({x['ci'][0]:.2f} to {x['ci'][1]:.2f}) | "
                 f"{x['lower_bound_bonferroni_one_sided']:.2f} | {pc(x['cost_share'])} | {pc(x['time_share_estimate'])} | "
                 f"{x['pair_crossings']} / {x['alone_crossings']} |")
    L += ["", "## Drop-in pairing (C4): Jev keeps its most confident half, each main LLM (names) answers the rest", "",
          f"The drop-in pairing code path of hybrid_all.py; 2,000 resamples stratified by set, seed {res['seeds']['c4']}.", ""]
    for g, r in res["dropin"].items():
        for v, x in r["versions"].items():
            L += [f"### {g.replace('_', ' ')} (n = {r['n']:,}), Jev with {v} (Jev answers {x['n_jev']}; Jev alone {pc(x['jev_alone'])}%)", "",
                  "| LLM | LLM alone % | Pair % | Difference, pts (95% CI) | Bonferroni lower bound, pts | Cost share % |", "|---|---|---|---|---|---|"]
            for s, y in x["llms"].items():
                L.append(f"| {s} | {pc(y['alone'])} | {pc(y['pair'])} | {y['diff_pts']:.2f} ({y['ci_pts'][0]:.2f} to {y['ci_pts'][1]:.2f}) | "
                         f"{y['lower_bound_bonferroni_one_sided_pts']:.2f} | {pc(y['cost_share'])} |")
            L.append("")
    return "\n".join(L) + "\n"


def main():
    import hashlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default=str(ROOT), help="the folder holding results/ and data/ (default: the repository root)")
    a = ap.parse_args()
    br = Path(a.branch)
    T = truth()
    pool = pd.read_csv(br / "data/arm3_pool/pool240.csv", dtype=str)
    groups = {g: (sorted(pool.report_id) if s is None else sorted(T.index[T.set.isin(s)])) for g, s in GROUPS.items()}
    assert [len(v) for v in groups.values()] == [240, 1320, 1200, 120]
    V, L = versions(br), llm_names()
    wg = json.loads((br / "results/summaries/workflow_groups.json").read_text())
    res = {"note": __doc__.split("\n\n")[0],
           "pool240": {"n": 240, "sets": pool.set.value_counts().reindex(SETS).to_dict(), "seed": 20260929,
                       "sha256": hashlib.sha256((br / "data/arm3_pool/pool240.csv").read_bytes()).hexdigest()},
           "seeds": {"c3": wg["seed"], "c4": SEED}, "versions_order": [v for v in VERSIONS if v in V],
           "staircase": staircase(T, V, groups),
           "recommended_pool240": wg["groups"]["pool240"],
           "dropin": dropin(T, V, L, {g: groups[g] for g in ("pool240", "all_1320")}, np.random.default_rng(SEED))}
    (br / "results/summaries/pool_summary.json").write_text(json.dumps(res, indent=1))
    (br / "results/summaries/pool_summary.md").write_text(md(res), encoding="utf-8")
    print(md(res))


if __name__ == "__main__":
    main()

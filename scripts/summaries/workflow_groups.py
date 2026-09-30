"""Recommended-form pairing (Jev when fully confident, each main LLM otherwise), colon cancer stage, by report group:
ordinary (the first 1,000 reports and the 200 written by another model) and difficult (100 with a misleading feature and
20 with a metastatic node at an unusual site), and all 1,320 together (mixed). Pooled accuracy difference from the LLM alone with a paired bootstrap
stratified by report set (2,000 resamples, seed as workflow_checks.py), 95% percentile interval and the Bonferroni one-sided
lower bound (alpha 0.025 / 5), and cost as a share of the LLM alone. Same item-level code path as workflow_checks.py part C;
every set's counts and costs are asserted against workflow_checks.json. For Fig. 2c-d. Stored answers only; no model call.

  python scripts/summaries/workflow_groups.py --docuse . --heldout . --pool240 data/arm3_pool/pool240.csv
      -> results/summaries/workflow_groups.json (--out to write elsewhere)

--docuse: the folder holding results/arm3_docuse and results/arm3_nonregional_sites; --heldout: the folder holding
data/arm3_heldout/items.csv and results/arm3_heldout. Run from the repository root after workflow_checks.py and
workflow_timing.py.

Group pool240 (--pool240 data/arm3_pool/pool240.csv): the 120 difficult reports and 120 ordinary ones drawn at random
(scripts/make_pool240.py). Same code path; the bootstrap resamples within each set's part of the pool, from its own
generator with the same seed, drawn after the other groups' resamples, and the run asserts that every stored group and
summary value is unchanged (--stored, default the output file). Also, for this group: errors across a treatment line
(workflow_checks.py GRP3: 0-II, III, IV; an unusable answer counts as crossing), for the pair and the LLM alone; and
time as a share of the LLM alone from the timed reports' means (workflow_timing.json: Jev's mean plus the LLM's mean on
the pool's share Jev passes on, over the LLM's mean; the timing sample holds first-1,000 reports only, so this share is
an estimate).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from workflow_checks import PRICE, MAIN, TOL, B, SEED, pts, J, GRP3  # noqa: E402

GROUPS = {"ordinary": ("main", "heldout"), "difficult": ("misleading_feature", "nonregional_sites"),
          "mixed": ("main", "heldout", "misleading_feature", "nonregional_sites")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docuse", required=True); ap.add_argument("--heldout", required=True)
    ap.add_argument("--pool240", default=None); ap.add_argument("--out", default="results/summaries/workflow_groups.json")
    ap.add_argument("--stored", default=None)
    a = ap.parse_args(); W, H = a.docuse, a.heldout
    it = pd.read_csv("data/arm3/items.csv", dtype=str); main_key = dict(zip(it["report_id"], it["level"]))
    cf = pd.read_parquet("results/arm3_confuser/calls.parquet"); cf = cf[(cf["variant"] == "names") & (cf["repeat"] == 0)]
    conf_key = cf[cf["system"] == "gemini"].set_index("item_id")["answer"].to_dict()
    nr = J(f"{W}/results/arm3_nonregional_sites/summary.json")
    hi = pd.read_csv(f"{H}/data/arm3_heldout/items.csv", dtype=str); held_key = dict(zip(hi["report_id"], hi["level"]))
    dc = pd.read_parquet(f"{W}/results/arm3_docuse/calls.parquet"); dc = dc[dc["repeat"] == 0]
    nc = pd.read_parquet(f"{W}/results/arm3_nonregional_sites/calls_jev.parquet"); nc = nc[nc["repeat"] == 0]
    hc = pd.read_parquet(f"{H}/results/arm3_heldout/documented_calls.parquet"); hc = hc[hc["repeat"] == 0]
    nr_key = {i: nr["truth"]["level"] for i in sorted(nc["item_id"].unique())}
    a3 = pd.read_parquet("results/arm3/calls.parquet"); a3 = a3[(a3["variant"] == "names") & (a3["repeat"] == 0)]
    nb = pd.read_parquet(f"{W}/results/arm3_nonregional_sites/calls_chatbots.parquet"); nb = nb[(nb["variant"] == "names") & (nb["repeat"] == 0)]
    hb = pd.read_parquet(f"{H}/results/arm3_heldout/calls.parquet"); hb = hb[(hb["variant"] == "names") & (hb["repeat"] == 0)]
    spec = {"main": (dc, a3, main_key), "heldout": (hc, hb, held_key),
            "misleading_feature": (dc, cf, conf_key), "nonregional_sites": (nc, nb, nr_key)}
    ref = J("results/summaries/workflow_checks.json")["C_structured_workflow"]["documented"]["sets"]
    rng = np.random.default_rng(SEED)
    arrs = {}
    for name, (jc, bc, key) in spec.items():
        ids = sorted(key); n = len(ids); truth = pd.Series(key).reindex(ids)
        j = jc[jc["variant"] == "documented"].set_index("item_id").reindex(ids)
        jans = j["answer"].where(j["valid"].fillna(False).astype(bool))
        keep = j["top_prob"].to_numpy(float) >= 1 - TOL
        ju = (j["tokens_in"].fillna(0) * PRICE["jev"][0] + j["tokens_out"].fillna(0) * PRICE["jev"][1]).to_numpy() / 1e6
        arrs[name] = {"n": n, "ju": ju, "keep": keep, "fok": {}, "bok": {}, "bu": {}, "ids": ids,
                      "truth": truth.to_numpy(), "fin": {}, "bans": {}}
        for s in MAIN:
            b = bc[bc["system"] == s].set_index("item_id").reindex(ids)
            bans = b["answer"].where(b["valid"].fillna(False).astype(bool))
            fok = (pd.Series(np.where(keep, jans, bans), index=ids) == truth).to_numpy(); bok = (bans == truth).to_numpy()
            bu = (b["tokens_in"].fillna(0) * PRICE[s][0] + b["tokens_out"].fillna(0) * PRICE[s][1]).to_numpy() / 1e6
            rc = ref[name]["chatbots"][s]
            assert rc["accuracy"]["k"] == int(fok.sum()) and rc["chatbot_alone_accuracy"]["k"] == int(bok.sum()), (name, s)
            assert abs(rc["usd_per_1000"] - (1000 * (ju.sum() + bu[~keep].sum()) / n)) < 1e-9, (name, s)
            arrs[name]["fok"][s], arrs[name]["bok"][s], arrs[name]["bu"][s] = fok, bok, bu
            arrs[name]["fin"][s], arrs[name]["bans"][s] = np.where(keep, jans, bans), bans.to_numpy()
    idx = {k: rng.integers(0, v["n"], size=(B, v["n"])) for k, v in arrs.items()}
    out = {"note": __doc__.split("\n\n")[0], "B": B, "seed": SEED, "groups": {}}
    for g, sets in GROUPS.items():
        N = sum(arrs[k]["n"] for k in sets)
        r = {"n": N, "sets": list(sets), "kept": int(sum(arrs[k]["keep"].sum() for k in sets)), "llms": {}}
        for s in MAIN:
            hk = sum(int(arrs[k]["fok"][s].sum()) for k in sets); ak = sum(int(arrs[k]["bok"][s].sum()) for k in sets)
            bs = sum(arrs[k]["fok"][s][idx[k]].sum(1) - arrs[k]["bok"][s][idx[k]].sum(1) for k in sets) / N
            cost = sum(arrs[k]["ju"].sum() + arrs[k]["bu"][s][~arrs[k]["keep"]].sum() for k in sets)
            alone = sum(arrs[k]["bu"][s].sum() for k in sets)
            r["llms"][s] = {"diff_pts": pts((hk - ak) / N), "ci": [pts(np.quantile(bs, 0.025)), pts(np.quantile(bs, 0.975))],
                            "lower_bound_bonferroni_one_sided": pts(np.quantile(bs, 0.025 / len(MAIN))),
                            "pair_correct": hk, "alone_correct": ak, "cost_share": float(cost / alone)}
        out["groups"][g] = r
    og, dg = out["groups"]["ordinary"], out["groups"]["difficult"]
    jev_right = {g: sum(int((ref[k]["jev_correct"])) for k in sets) for g, sets in GROUPS.items()}
    out["summary"] = {
        "ordinary_cost_share": {"min": min(v["cost_share"] for v in og["llms"].values()),
                                "max": max(v["cost_share"] for v in og["llms"].values())},
        "ordinary_kept_share": og["kept"] / og["n"],
        "difficult_jev_correct": jev_right["difficult"], "difficult_jev_accuracy": jev_right["difficult"] / dg["n"],
        "ordinary_jev_correct": jev_right["ordinary"],
        "jev_errors": sum(out["groups"][g]["n"] - jev_right[g] for g in ("ordinary", "difficult")),
        "mixed_n": out["groups"]["mixed"]["n"], "mixed_jev_correct": jev_right["mixed"],
        "mixed_jev_accuracy": jev_right["mixed"] / out["groups"]["mixed"]["n"],
        "mixed_kept_share": out["groups"]["mixed"]["kept"] / out["groups"]["mixed"]["n"],
        "mixed_cost_share": {"min": min(v["cost_share"] for v in out["groups"]["mixed"]["llms"].values()),
                             "max": max(v["cost_share"] for v in out["groups"]["mixed"]["llms"].values())},
        "mixed_diff_pts": {"min": min(v["diff_pts"] for v in out["groups"]["mixed"]["llms"].values()),
                           "max": max(v["diff_pts"] for v in out["groups"]["mixed"]["llms"].values())},
        "mixed_lower_bound_min": min(v["lower_bound_bonferroni_one_sided"] for v in out["groups"]["mixed"]["llms"].values())}
    stored = Path(a.stored or a.out)
    if stored.exists():                       # every stored group and summary value must come out unchanged
        old, new = J(stored), json.loads(json.dumps(out))
        bad = [g for g in old["groups"] if g != "pool240" and old["groups"][g] != new["groups"].get(g)]
        assert not bad and old["summary"] == new["summary"], f"stored groups changed: {bad}"
    if a.pool240:
        out["groups"]["pool240"] = pool240(a.pool240, arrs)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(out["summary"])
    for g, r in out["groups"].items():
        print(g, r["n"], "kept", r["kept"], {s: (round(v["diff_pts"], 2), [round(x, 2) for x in v["ci"]], round(v["lower_bound_bonferroni_one_sided"], 2), round(100 * v["cost_share"], 1)) for s, v in r["llms"].items()})


def pool240(path, arrs) -> dict:
    """Group pool240: each set's pool reports, resampled within the set (own generator, same seed)."""
    pool = pd.read_csv(path, dtype=str)
    want = pool.groupby("set")["report_id"].apply(set).to_dict()
    assert set(want) <= set(arrs) and len(pool) == 240, sorted(want)
    sel = {k: np.flatnonzero([i in want[k] for i in arrs[k]["ids"]]) for k in arrs if k in want}
    assert all(len(sel[k]) == len(want[k]) for k in sel), "pool reports missing from a set"
    N = sum(len(v) for v in sel.values())
    rng = np.random.default_rng(SEED)
    idx = {k: rng.integers(0, len(v), size=(B, len(v))) for k, v in sel.items()}
    tm = J("results/summaries/workflow_timing.json")
    kept = int(sum(arrs[k]["keep"][v].sum() for k, v in sel.items()))
    cross = lambda ans, t: int(sum((x is None or (isinstance(x, float) and np.isnan(x)) or GRP3.get(x) != GRP3[y])
                                   for x, y in zip(ans, t)))
    r = {"n": N, "sets": {k: int(len(v)) for k, v in sel.items()}, "kept": kept, "file": Path(path).name, "llms": {}}
    for s in MAIN:
        fok = {k: arrs[k]["fok"][s][v] for k, v in sel.items()}
        bok = {k: arrs[k]["bok"][s][v] for k, v in sel.items()}
        hk, ak = sum(int(x.sum()) for x in fok.values()), sum(int(x.sum()) for x in bok.values())
        bs = sum(fok[k][idx[k]].sum(1) - bok[k][idx[k]].sum(1) for k in sel) / N
        cost = sum(arrs[k]["ju"][v].sum() + arrs[k]["bu"][s][v][~arrs[k]["keep"][v]].sum() for k, v in sel.items())
        alone = sum(arrs[k]["bu"][s][v].sum() for k, v in sel.items())
        jm, lm = tm["jev_recommended_form"]["mean_s"], tm["systems"][s]["llm_mean_s"]
        r["llms"][s] = {"diff_pts": pts((hk - ak) / N), "ci": [pts(np.quantile(bs, 0.025)), pts(np.quantile(bs, 0.975))],
                        "lower_bound_bonferroni_one_sided": pts(np.quantile(bs, 0.025 / len(MAIN))),
                        "pair_correct": hk, "alone_correct": ak, "cost_share": float(cost / alone),
                        "time_share_estimate": float((jm + (1 - kept / N) * lm) / lm),
                        "pair_crossings": sum(cross(arrs[k]["fin"][s][v], arrs[k]["truth"][v]) for k, v in sel.items()),
                        "alone_crossings": sum(cross(arrs[k]["bans"][s][v], arrs[k]["truth"][v]) for k, v in sel.items())}
    return r


if __name__ == "__main__":
    main()

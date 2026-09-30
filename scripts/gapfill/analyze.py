"""Analyses of the answers of scripts/gapfill/run.py. Stored answers only; no call.
  B1 results/summaries/gapfill_eicu_structured.json   Jev's death probability, categorised against drop-in eICU records:
     AUROC with 95% interval (eicu_analysis.baseline_block: 2,000 record resamples, seed 20260922, as results/eicu),
     mean predicted against observed, log-loss (supplement); paired DeLong, categorised minus drop-in.
  B2 results/summaries/gapfill_eicu_hybrid.json       the eICU split on each record form: Jev's death probability on the
     half of stays where its top outcome-option probability is highest (ties by stay id), each main LLM's stored
     probability elsewhere; gapfill_stats.hybrid_descriptive (hybrid.hybrid_block's routing and resamples, B 1,000,
     seed 29; no margin: eICU has no baseline-prediction margin), labelled descriptive.
  B3 results/summaries/gapfill_nhanes_structured_hybrid.json  hybrid.hybrid_block unchanged, as scripts/hybrid_arm1.py,
     with Jev's recommended-form death probability and its recommended-form band confidence.
Inputs. B1 and B2: data/eicu/demo_cohort.parquet (scripts/build_eicu.py demo) and the demo's patient table,
results/eicu/calls_eval.parquet and results/gapfill/eicu_structured.parquet; B2 also results/gapfill/eicu_bands.parquet.
B3: results/wide/{calls.parquet,hybrid.json}, data/cohort.parquet, data/reference_oof.parquet,
results/eval/cost_speed.json, results/gapfill/nhanes_bands.parquet and results/wide_docuse/calls.parquet (in the
--docuse folder, default the repository root).
  python scripts/gapfill/analyze.py [--docuse DIR] [--only NAME ...]
NAME: gapfill_eicu_structured, gapfill_eicu_hybrid or gapfill_nhanes_structured_hybrid (default: all three).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevity import eicu_analysis as EA                                               # noqa: E402
from jevity import gapfill as G                                                      # noqa: E402
from jevity import gapfill_stats as GS                                               # noqa: E402
from jevity import hybrid as H                                                       # noqa: E402
from jevity.analysis import CLIP                                                     # noqa: E402
from jevity.clients import MODELS, active_families                                   # noqa: E402

TABLES = ROOT / "results" / "gapfill"
SUMMARIES = ROOT / "results" / "summaries"
DOCUSE = ROOT                          # the folder holding results/wide_docuse (--docuse)
LLMS = ["gpt", "claude", "gemini", "muse", "glm"]
PRICE = float(MODELS["jev"]["price_per_mtok_input"])


def _valid(df: pd.DataFrame, **eq) -> pd.Series:
    d = df[df.valid.astype(bool)]
    for k, v in eq.items():
        d = d[d[k] == v]
    assert not d.profile.duplicated().any(), eq
    return d.set_index(d.profile.astype(int))


def eicu_inputs():
    ev = G.eval_cohort(ROOT)
    y = ev.set_index(ev.SEQN.astype(int))[G.E.OUTCOME].astype(float)
    ce = pd.read_parquet(ROOT / "results" / "eicu" / "calls_eval.parquet")
    ce = ce[(ce.edit == "baseline") & (ce.variant == "raw")]
    llm = {s: _valid(ce, model=s)["p"].astype(float) for s in LLMS}
    drop = _valid(ce, model="jev")["p"].astype(float)
    cat = _valid(pd.read_parquet(TABLES / "eicu_structured.parquet"), edit="categorised")["p"].astype(float)
    return ev, y, llm, drop, cat, ce


def b1() -> dict:
    ev, y, llm, drop, cat, _ = eicu_inputs()
    both = sorted(set(drop.index) & set(cat.index) & set(y.index))
    yy = y.loc[both].to_numpy(float)
    pc, pd_ = (np.clip(s.loc[both].to_numpy(float), *CLIP) for s in (cat, drop))
    blk = EA.baseline_block({"jev_categorised": cat.loc[both], "jev_drop_in": drop.loc[both]}, y.loc[both])
    dl = GS.delong_paired(yy, pc, pd_)
    allc = sorted(set(both).intersection(*[set(s.index) for s in llm.values()]))
    blk_all = EA.baseline_block({"jev_categorised": cat, "jev_drop_in": drop, **llm}, y.loc[allc])
    tok = pd.read_parquet(TABLES / "eicu_structured.parquet")
    return {"label": "eICU death before hospital discharge, Jev, categorised (recommended form) against drop-in records",
            "n_stays": int(len(y)), "answered_categorised": int(len(cat)), "answered_drop_in": int(len(drop)),
            "n": len(both), "deaths": int(yy.sum()),
            "categorised": blk["predictors"]["jev_categorised"], "drop_in": blk["predictors"]["jev_drop_in"],
            "delong_categorised_minus_drop_in": dl,
            "with_llms_common_records": {"n": blk_all["n_records"], "deaths": blk_all["deaths"],
                                         "predictors": blk_all["predictors"]},
            "settings": {"B": blk["B"], "seed": blk["seed"], "clip": list(CLIP), "delong": "Sun and Xu 2014, clipped probabilities"},
            "cost_usd_list_per_1000": float(tok.tokens_in.mean()) * PRICE / 1e6 * 1000,
            "log_loss_note": "log-loss for the supplement only"}


def _auroc_ci(y: np.ndarray, p: np.ndarray) -> dict:
    auc, _ = GS.delong_components(y, p[None, :])
    return {"auroc": float(auc[0]), "ci95_delong": GS.delong_ci(y, p)}


def _conf(bands: pd.DataFrame) -> pd.Series:
    return bands["band_probabilities"].map(H.top_probability).astype(float)


def _llm_cost(ce: pd.DataFrame) -> dict:
    out = {}
    for s in LLMS:
        f = MODELS["families"][s]
        d = ce[ce.model == s]
        out[s] = float((d.tokens_in.mean() * f["price_in"] + d.tokens_out.mean() * f["price_out"]) / 1e6)
    return out


def b2() -> dict:
    ev, y, llm, drop, cat, ce = eicu_inputs()
    bands = pd.read_parquet(TABLES / "eicu_bands.parquet")
    pt = G.E.read_table(ROOT / G.E.CFG["sources"]["demo"], "patient",
                        [G.E.ID, "unitdischargestatus", "hospitaldischargestatus"])
    pt = pt.set_index(pt[G.E.ID].astype(int))
    out = {"label": "descriptive (no non-inferiority margin: eICU has no baseline-prediction margin)",
           "rule": "Jev's death probability on the half of stays with the highest top outcome-option probability "
                   "(ties by stay id; no usable band answer ranks last), each main LLM's stored probability elsewhere",
           "forms": {}}
    cost = _llm_cost(ce)
    for form, edit, jev in (("drop_in", "baseline", drop), ("categorised", "categorised", cat)):
        b = bands[bands.edit == edit].drop_duplicates("profile").set_index("profile")
        common = sorted(set(jev.index).intersection(*[set(s.index) for s in llm.values()]) & set(y.index))
        conf = _conf(b[b.valid.astype(bool)]).reindex(common).to_numpy(float)
        P = {"jev": jev.loc[common].to_numpy(float), **{s: llm[s].loc[common].to_numpy(float) for s in LLMS}}
        c = {**cost, "jev": float(ce[ce.model == "jev"].tokens_in.mean() if form == "drop_in" else
                                  pd.read_parquet(TABLES / "eicu_structured.parquet").tokens_in.mean()) * PRICE / 1e6,
             "jev_bands": float(b.tokens_in.mean()) * PRICE / 1e6}
        res = GS.hybrid_descriptive(P, y.loc[common].to_numpy(float), conf, np.array(common), LLMS, cost=c)
        # what the outcome Choice itself says (descriptive)
        ok = b[b.valid.astype(bool)]
        truth = pd.Series({i: G.outcome_of(pt.loc[i]) for i in ok.index})
        probs = ok["band_probabilities"].map(lambda j: json.loads(j) if isinstance(j, str) else dict(j))
        top = probs.map(lambda d: max(d, key=d.get))
        opts = list(G.CFG["band_question"]["criteria"])
        res["outcome_question"] = {
            "calls": int(len(b)), "usable": int(len(ok)), "parse": b.parse.value_counts(dropna=False).to_dict(),
            "observed_share": truth.value_counts(normalize=True).reindex(opts).fillna(0).to_dict(),
            "mean_probability": {o: float(probs.map(lambda d: float(d.get(o, 0.0))).mean()) for o in opts},
            "top_option_right": float((top == truth.loc[top.index]).mean()),
            "band_death_probability_auroc": _auroc_ci(y.loc[ok.index].to_numpy(float),
                                                      np.clip(ok["p"].astype(float).to_numpy(), *CLIP))
            if 1 < y.loc[ok.index].sum() < len(ok) - 1 else None,
            "mean_band_death_vs_death_question": {"band": float(ok["p"].astype(float).mean()),
                                                  "death_question": float(jev.reindex(ok.index).mean())}}
        out["forms"][form] = res
    return out


def b3() -> dict:
    calls = pd.read_parquet(ROOT / "results" / "wide" / "calls.parquet")
    co = pd.read_parquet(ROOT / "data" / "cohort.parquet")
    y = co.set_index(co.SEQN.astype(int))["death_10y"].astype(float)
    o = pd.read_parquet(ROOT / "data" / "reference_oof.parquet").set_index("SEQN")
    chat = active_families(("primary",))
    preds = {}
    for m in chat:
        b = calls[(calls.model == m) & (calls.edit == "baseline") & calls.valid & (calls.variant == "raw")]
        preds[m] = b.drop_duplicates("profile").set_index("profile")["p"].astype(float)
    rec = pd.read_parquet(DOCUSE / "results" / "wide_docuse" / "calls.parquet")
    preds["jev"] = _valid(rec)["p"].astype(float)                       # recommended-form death probability
    for col, name in (("p_spline_logit", "reference_spline_logit"), ("p_age_sex", "reference_age_sex")):
        preds[name] = o[col].astype(float)
    common = sorted(set.intersection(*[set(s.dropna().index) for s in preds.values()]) & set(y.index))
    bands = pd.read_parquet(TABLES / "nhanes_bands.parquet")
    bb = bands[bands.valid.astype(bool)].drop_duplicates("profile").set_index("profile")
    conf = bb["band_probabilities"].map(H.top_probability).reindex(common).to_numpy(float)
    own = bb["confidence"].astype(float).reindex(common).to_numpy(float)
    cs = json.loads((ROOT / "results" / "eval" / "cost_speed.json").read_text())["systems"]
    cost = {m: float(cs[m]["usd_list_per_usable"]) for m in chat}
    cost["jev"] = float(rec.tokens_in.mean()) * PRICE / 1e6
    cost["jev_bands"] = float(bands.tokens_in.mean()) * PRICE / 1e6
    P = {k: preds[k].reindex(common).to_numpy(float) for k in ["jev"] + chat + ["reference_spline_logit", "reference_age_sex"]}
    res = H.hybrid_block(P, y.reindex(common).to_numpy(float), conf, np.array(common), chat, conf_own=own, cost=cost)
    bp = bb["p"].astype(float).reindex(common)
    jp = preds["jev"].reindex(common)
    res["bands_vs_primary"] = {"n": int(bp.notna().sum()), "mean_band_implied": float(bp.mean()),
                               "mean_primary": float(jp[bp.notna()].mean()),
                               "mean_abs_difference": float((bp - jp).abs().mean())}
    res["meta"] = {"form": "recommended (config/docuse_risk.yaml records)", "rule": "docs/arm1_hybrid_rule.md, unchanged",
                   "band_calls": int(len(bands)), "band_usable": int(bands.valid.sum()), "cost_list_usd_per_answer": cost,
                   "drop_in_result": "results/wide/hybrid.json"}
    dr = json.loads((ROOT / "results" / "wide" / "hybrid.json").read_text())
    res["drop_in_for_comparison"] = {s: {k: dr["chatbots"][s][k] for k in ("diff", "ci_simultaneous", "outcome")}
                                     for s in chat}
    return res


OUTPUTS = (("gapfill_eicu_structured.json", b1), ("gapfill_eicu_hybrid.json", b2),
           ("gapfill_nhanes_structured_hybrid.json", b3))


def main(argv=None):
    global DOCUSE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docuse", default=str(ROOT), help="the folder holding results/wide_docuse")
    ap.add_argument("--only", nargs="+", default=None, choices=[n[:-len(".json")] for n, _ in OUTPUTS],
                    help="write only these outputs (default: all three)")
    a = ap.parse_args(argv)
    DOCUSE = Path(a.docuse)
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in OUTPUTS:
        if a.only and name[:-len(".json")] not in a.only:
            continue
        res = fn()
        (SUMMARIES / name).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print(f"wrote {SUMMARIES / name}")
        out[name] = res
    return out


if __name__ == "__main__":
    main()

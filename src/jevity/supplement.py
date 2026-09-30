"""Descriptive supplement tables, computed from the inputs of scripts/analyze.py and outside the confirmatory family: no
test, no simultaneous interval, nothing enters the headline tiers.
  reference_change    the reference model's own change for each confirmatory edit (the size the outcomes support)
  extreme_answers     share of answers at exactly 0 or 1
  log_loss_clipping   baseline log-loss with the probability clipped at 0.01, 0.005 (the analysis's bound) and 0.001
  repeat_spread       run-to-run standard deviation from the repeat calls
  output_cap          answers that stopped at the output cap, per system and variant, and by edit
  fill_missing_pairs  the worst-case bound for a system's missing answers (scripts/missing_answer_bound.py): each
                      unanswered supported pair filled with the system's most extreme contrast for that edit
  repeatability       ICC(2,1) and the share answered identically on the repeat set (as arm 3 computes them)
  reasoning_tokens    reasoning tokens as a raw reply reports them (as arm 3 reads them; for scripts/cost_speed.py)"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import CLIP, FAMILY, Draws, Refits, _ci, _variant, expit, logit, ordered

CLIPS = (0.01, CLIP[0], 0.001)          # CLIP[0] = 0.005, the bound of analysis.py
REFERENCE_ORDER = ("spline_logit", "spline_logit_weighted", "lightgbm", "survival_logit")


def first_raw(calls: pd.DataFrame) -> pd.DataFrame:
    """First valid raw answers (repeat 0, unannotated), as analysis.py reads them."""
    return calls[(calls.repeat == 0) & (~calls.annotated) & calls.valid & _variant(calls, "raw")]


def reference_change(inst: pd.DataFrame, refits: dict[str, Refits], profiles, B: int = 2000,
                     edits: list[str] | None = None) -> dict:
    """Per confirmatory edit and reference: the mean change the reference predicts over the supported records (log-odds,
    odds ratio, probability points), its spread across records, and a 95% interval in which each draw resamples records
    and uses refit b mod R (the shared draws of analysis.py: same profiles, same seed)."""
    edits = [e for e in (edits or FAMILY) if e in set(inst.edit)]
    sup = inst[inst.applicable & inst.supported]
    out = {}
    for ref in [r for r in REFERENCE_ORDER if r in refits and f"d_logit_{r}" in inst.columns]:
        rf = refits[ref]
        draws = Draws(profiles, B, rf.R)
        res = {}
        for e in edits:
            t = sup[(sup.edit == e) & sup.profile.isin(draws.pos)]
            if t.empty:
                continue
            dl, dp = t[f"d_logit_{ref}"].to_numpy(float), t[f"d_prob_{ref}"].to_numpy(float) * 100
            rows = rf.rows(t)
            r = {"klass": str(t.klass.iloc[0]), "feature": str(t.feature.iloc[0]), "n_records": int(len(t)),
                 "mean_logit": float(dl.mean()), "odds_ratio": float(np.exp(dl.mean())), "mean_prob_pts": float(dp.mean()),
                 "median_logit": float(np.median(dl)), "iqr_logit": [float(np.percentile(dl, 25)), float(np.percentile(dl, 75))],
                 "p05_p95_logit": [float(np.percentile(dl, 5)), float(np.percentile(dl, 95))],
                 "median_prob_pts": float(np.median(dp)), "iqr_prob_pts": [float(np.percentile(dp, 25)), float(np.percentile(dp, 75))]}
            if (rows >= 0).all():
                W = draws.w(t.profile)
                Ws = np.maximum(W.sum(1), 1e-12)
                ml = (W * rf.d_logit[rows][:, draws.r].T).sum(1) / Ws
                mp = (W * rf.d_prob[rows][:, draws.r].T).sum(1) / Ws * 100
                r |= {"mean_logit_ci": _ci(ml), "odds_ratio_ci": [float(np.exp(x)) if x is not None else None for x in _ci(ml)],
                      "mean_prob_pts_ci": _ci(mp), "refits": int(rf.R)}
            res[e] = r
        out[ref] = res
    return out


def extreme_answers(calls: pd.DataFrame, models) -> dict:
    """Share of first valid raw answers at exactly 0 or exactly 1, over every edit and over baselines alone."""
    c = first_raw(calls)
    out = {}
    for m in models:
        g = c[c.model == m]
        if g.empty:
            continue
        r = {}
        for part, h in (("all_answers", g), ("baseline", g[g.edit == "baseline"].drop_duplicates("profile"))):
            p = h.p.to_numpy(float)
            r[part] = {"n": int(len(p)), "share_0": float(np.mean(p == 0)) if len(p) else None,
                       "share_1": float(np.mean(p == 1)) if len(p) else None,
                       "share_0_or_1": float(np.mean((p == 0) | (p == 1))) if len(p) else None}
        out[m] = r
    return out


def log_loss_clipping(preds: dict[str, pd.Series], y: pd.Series, clips=CLIPS, focal: str = "jev",
                      versus: list[str] | None = None, B: int = 1000, seed: int = 29) -> dict:
    """Mean per-record log-loss on the records every system scored (the records of analysis.prediction_block), with each
    probability clipped to [c, 1 - c] for each c in `clips`. Record resamples reproduce prediction_block's (same seed and
    draw order), so the intervals at analysis.py's bound equal its log_loss_ci. Focal-minus-system differences carry marginal
    percentile intervals only (descriptive)."""
    names = [n for n, s in preds.items() if s is not None]
    common = sorted(set.intersection(*[set(preds[n].dropna().index) for n in names]) & set(y.index))
    yy = y.loc[common].to_numpy(float)
    n = len(common)
    rng = np.random.default_rng(seed)
    rng.integers(0, 2, n)                                    # prediction_block draws its recalibration halves first
    C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)
    raw = {nm: preds[nm].loc[common].to_numpy(float) for nm in names}
    out = {"n_records": n, "deaths": int(yy.sum()), "clips": list(clips), "fixed_clip": CLIP[0], "by_clip": {}}
    for c in clips:
        LL = {}
        blk = {"systems": {}}
        for nm in names:
            p = np.clip(raw[nm], c, 1 - c)
            ll = -(yy * np.log(p) + (1 - yy) * np.log(1 - p))
            LL[nm] = ll
            blk["systems"][nm] = {"log_loss": float(ll.mean()), "log_loss_ci": _ci(C @ ll / n),
                                  "share_clipped": float(np.mean((raw[nm] < c) | (raw[nm] > 1 - c)))}
        if focal in LL:
            blk["focal_minus"] = {}
            for nm in (versus or [x for x in names if x != focal]):
                if nm in LL and nm != focal:
                    d = LL[focal] - LL[nm]
                    blk["focal_minus"][nm] = {"diff": float(d.mean()), "ci_marginal": _ci(C @ d / n)}
        out["by_clip"][str(c)] = blk
    return out


def repeat_spread(calls: pd.DataFrame, models, B: int = 1000, seed: int = 41) -> dict:
    """Run-to-run spread from the repeat calls (the repeat profiles' baseline and representative edits, asked three
    times): per item (profile, edit) with at least two valid raw answers, the sample variance across repeats; the pooled
    SD is the root of the mean variance, on the log-odds scale (clipped as in analysis.py) and in probability
    points. Intervals resample profiles."""
    c = calls[(~calls.annotated) & calls.valid & _variant(calls, "raw")]
    out = {}
    for m in models:
        g = c[c.model == m]
        items = g.loc[g.repeat > 0, ["profile", "edit"]].drop_duplicates()
        if items.empty:
            continue
        g = g.merge(items, on=["profile", "edit"]).drop_duplicates(["profile", "edit", "repeat"])
        g = g.assign(z=logit(g.p.to_numpy(float)), pts=g.p.to_numpy(float) * 100)
        per = g.groupby(["profile", "edit"]).agg(k=("p", "size"), v_logit=("z", "var"), v_pts=("pts", "var"),
                                                  lo=("p", "min"), hi=("p", "max")).reset_index()
        per = per[per.k >= 2]
        if per.empty:
            continue

        def pooled(t: pd.DataFrame) -> dict:
            prof = np.array(sorted(t.profile.unique()))
            pos = {p: i for i, p in enumerate(prof)}
            idx = t.profile.map(pos).to_numpy()
            rng = np.random.default_rng(seed)
            W = np.stack([np.bincount(rng.integers(0, len(prof), len(prof)), minlength=len(prof)) for _ in range(B)])[:, idx].astype(float)
            Ws = np.maximum(W.sum(1), 1e-12)
            vl, vp = t.v_logit.to_numpy(float), t.v_pts.to_numpy(float)
            return {"n_items": int(len(t)), "n_items_three_answers": int((t.k >= 3).sum()), "n_profiles": int(len(prof)),
                    "pooled_sd_logit": float(np.sqrt(vl.mean())), "pooled_sd_logit_ci": _ci(np.sqrt(W @ vl / Ws)),
                    "pooled_sd_pts": float(np.sqrt(vp.mean())), "pooled_sd_pts_ci": _ci(np.sqrt(W @ vp / Ws)),
                    "median_item_sd_logit": float(np.median(np.sqrt(vl))), "median_item_sd_pts": float(np.median(np.sqrt(vp))),
                    "share_items_identical": float(np.mean(t.lo == t.hi))}

        out[m] = pooled(per) | {"by_edit": {e: pooled(t) for e, t in per.groupby("edit")}}
    return out


def systems(calls: pd.DataFrame) -> list[str]:
    """Reporting order; Jev's secondary questions (jev_bands, ...) are not systems."""
    return ordered(set(calls.model))


def output_cap(calls: pd.DataFrame, models) -> dict:
    """Answers that stopped at the output cap (finish reason "length", or output tokens at the configured max_tokens),
    per system and variant over every stored call, with output-token percentiles, the share of unusable answers the cap
    explains, and the share at the cap by edit (first raw answers). Chatbots only: Jev has no output tokens."""
    from .clients import MODELS, split_system
    out = {}
    for m in models:
        fam = split_system(m)[0]
        spec = (MODELS.get("families") or {}).get(fam)
        if m.startswith("jev") or not spec:
            continue
        cap = spec.get("max_tokens")
        g = calls[calls.model == m]
        if "finish_reason" not in g or g.empty:
            continue
        tout = g["tokens_out"].astype(float) if "tokens_out" in g else pd.Series(np.nan, index=g.index)
        hit = (g["finish_reason"] == "length") | (tout >= cap if cap else False)
        r = {"max_tokens": cap, "variants": {}}
        for v, h in g.groupby(g["variant"] if "variant" in g else pd.Series("raw", index=g.index)):
            t, k = tout.loc[h.index].dropna(), hit.loc[h.index]
            bad = ~h.valid.astype(bool)
            r["variants"][str(v)] = {
                "n": int(len(h)), "at_cap": int(k.sum()), "share_at_cap": float(k.mean()),
                "unusable": int(bad.sum()), "unusable_at_cap": int((bad & k).sum()),
                "tokens_out_median": float(t.median()) if len(t) else None,
                "tokens_out_p90": float(t.quantile(0.9)) if len(t) else None,
                "tokens_out_p99": float(t.quantile(0.99)) if len(t) else None,
                "tokens_out_max": float(t.max()) if len(t) else None}
        f = g[(g.repeat == 0) & (~g.annotated)]
        f = f[_variant(f, "raw")] if "variant" in f and (f["variant"] == "raw").any() else f      # framing: F1 only
        if len(f):
            by = hit.loc[f.index].groupby(f.edit)
            r["by_edit_first_answers"] = {e: {"n": int(len(x)), "at_cap": int(x.sum()), "share": float(x.mean())} for e, x in by}
        out[m] = r
    return out


def fill_missing_pairs(tab: pd.DataFrame, inst: pd.DataFrame, calls: pd.DataFrame, model: str, ref: str, profiles,
                       edits, z_fill: dict[str, float]) -> tuple[pd.DataFrame, dict[str, int]]:
    """`tab` (analysis.contrast_table for `model` and reference `ref`) with one row added for every applicable, supported
    record of `edits` among `profiles` that the system did not answer usably (edited answer or baseline missing). The
    added row's contrast is z_fill[edit] on the log-odds scale; its baseline is the system's own valid baseline answer,
    or the median of those answers when the baseline itself is missing (this affects the probability-point D only).
    Edits absent from z_fill are left unfilled. Returns the filled table and the rows added per edit."""
    sup = inst[inst.applicable & inst.supported & inst.edit.isin(list(z_fill)) & inst.edit.isin(list(edits))
               & inst.profile.isin(list(profiles))]
    have = set(zip(tab.profile.astype(int), tab.edit.astype(str)))
    miss = sup[[(int(p), str(e)) not in have for p, e in zip(sup.profile, sup.edit)]]
    if miss.empty:
        return tab, {}
    c = calls[(calls.model == model) & (calls.repeat == 0) & (~calls.annotated) & calls.valid & _variant(calls, "raw")]
    base = c[c.edit == "baseline"].drop_duplicates("profile").set_index("profile")["p"]
    pb = miss.profile.map(base).fillna(float(base.median())).to_numpy(float)
    z = miss.edit.map(z_fill).to_numpy(float)
    pe = expit(logit(pb) + z)
    rows = pd.DataFrame({"profile": miss.profile.to_numpy(), "edit": miss.edit.to_numpy(), "klass": miss.klass.to_numpy(),
                         "feature": miss.feature.to_numpy(), "p_base": pb, "p": pe, "z_logit": z, "z_prob": pe - pb,
                         "d_logit": miss[f"d_logit_{ref}"].to_numpy(float), "d_prob": miss[f"d_prob_{ref}"].to_numpy(float),
                         "q0": miss[f"q0_{ref}"].to_numpy(float) if f"q0_{ref}" in miss else np.nan})
    return pd.concat([tab, rows[tab.columns]], ignore_index=True), {str(e): int(n) for e, n in miss.edit.value_counts().items()}


def icc_2_1(x: np.ndarray) -> float | None:
    """ICC(2,1) of an items x raters matrix without missing values: (BMS - EMS) / (BMS + (k-1) EMS + k (JMS - EMS) / n).
    None when it is undefined (fewer than two items or raters, or no variance at all). The same function as arm 3's
    (jevity.arm3_supplement.icc_2_1), so the three tasks use one method."""
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


def repeatability(calls: pd.DataFrame, models, B: int = 2000, seed: int = 20260922, ci: float = 0.95) -> dict:
    """Repeatability on the repeat set (the repeat profiles' baseline and representative edits, asked three times;
    repeat 0 is the main pass), per system: ICC(2,1) of the answer's log-odds (clipped as in analysis.py) with the
    repeats as the raters, and the share of items answered identically every time (the same probability). Items are
    (profile, edit); only items with a usable answer in every repeat enter, the others are counted. Also on the
    baseline items alone (one per record, as arm 3's reports). 95% percentile intervals from resampling records
    (profiles, with all their items; analysis.py's number of draws and seed); a resample whose ICC is undefined is
    counted, not used."""
    c = calls[(~calls.annotated) & _variant(calls, "raw")]
    q = [(1 - ci) / 2, 1 - (1 - ci) / 2]

    def block(t: pd.DataFrame, k: int) -> dict:
        z = t.pivot_table(index=["profile", "edit"], columns="repeat", values="z", aggfunc="first")
        z = z.reindex(columns=range(k))
        full = z.dropna()
        p = t.pivot_table(index=["profile", "edit"], columns="repeat", values="p", aggfunc="first").reindex(
            index=full.index, columns=range(k))
        out = {"items": int(len(z)), "items_all_usable": int(len(full)), "items_with_unusable": int(len(z) - len(full)),
               "records": int(full.index.get_level_values(0).nunique()), "icc_2_1": None, "icc_2_1_ci": None,
               "identical": None, "identical_ci": None, "resamples_icc_undefined": None}
        if full.empty:
            return out
        X, same = full.to_numpy(float), (p.to_numpy(float) == p.to_numpy(float)[:, :1]).all(axis=1)
        prof = full.index.get_level_values(0).to_numpy()
        ids = np.unique(prof)
        rows = [np.flatnonzero(prof == u) for u in ids]
        out.update(icc_2_1=icc_2_1(X), identical=float(same.mean()))
        rng = np.random.default_rng(seed)
        iccs, idents = np.full(B, np.nan), np.empty(B)
        for b in range(B):
            r = np.concatenate([rows[i] for i in rng.integers(0, len(ids), len(ids))])
            v = icc_2_1(X[r])
            iccs[b] = np.nan if v is None else v
            idents[b] = same[r].mean()
        ok = ~np.isnan(iccs)
        out["resamples_icc_undefined"] = int((~ok).sum())
        if ok.any():
            out["icc_2_1_ci"] = [float(v) for v in np.quantile(iccs[ok], q)]
        out["identical_ci"] = [float(v) for v in np.quantile(idents, q)]
        return out

    res = {}
    for m in models:
        g = c[c.model == m]
        items = g.loc[g.repeat > 0, ["profile", "edit"]].drop_duplicates()
        if items.empty:
            continue
        g = g.merge(items, on=["profile", "edit"]).drop_duplicates(["profile", "edit", "repeat"])
        k = int(g.repeat.max()) + 1
        ok = g.valid.astype(bool).to_numpy()
        pv = g.p.to_numpy(float)
        g = g.assign(z=np.where(ok, logit(np.where(ok, pv, 0.5)), np.nan), p=np.where(ok, pv, np.nan))
        res[m] = {"repeats": k, **block(g, k), "baseline": block(g[g.edit == "baseline"], k)}
    return res


def reasoning_tokens(raw) -> float | None:
    """Reasoning tokens as the raw reply reports them (OpenRouter and OpenAI chat: usage.completion_tokens_details;
    OpenAI responses: usage.output_tokens_details; Gemini: usageMetadata.thoughtsTokenCount); None if not reported.
    The same function as arm 3's (jevity.arm3_supplement.reasoning_tokens), so the three tasks count alike."""
    from collections.abc import Mapping
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

"""Derived metrics that turn the confirmatory family into clinically legible statements. All are descriptive
secondaries, computed from the same shared draws as the family (profile resamples paired with reference refits), so
their intervals carry sampling and reference uncertainty together unless stated otherwise.

  equivalents       every cell's excess expressed in years of age (edit C6) and mmHg of systolic pressure (edit C4);
                    formatting drift and repeat noise expressed the same way
  weights           per system, agreement between the risk-factor weights it implies and the weights outcomes support,
                    across the eleven fields (Spearman rank correlation, slope through the origin)
  consensus         the excess shared by the five language models (mean over systems) and how much the six systems
                    disagree (between-system SD; nearly reference-free because d cancels in differences)
  reclassification  share of people whose ten-year risk category changes after an edit, model against reference
  heterogeneity     does a cell's excess vary with age, sex or baseline reference risk (exploratory)
  dose_response     does doubling the size of a clinical change double the system's response as it does the reference's
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import Draws, Refits, _ci, logit, expit, LLM_ORDER, CRIT_INFLATION

RISK_CUTS = (0.05, 0.10, 0.20, 0.40)          # ten-year mortality categories: <5, 5-10, 10-20, 20-40, >=40%
LLMS = [m for m in LLM_ORDER if m != "jev"]


def _ref_unit_draws(inst: pd.DataFrame, refits: Refits, draws: Draws, ref: str, edit: str, size: float):
    """Reference log-OR per unit of an edit (C6: per year, size 5; C4: per mmHg, size 20), point and per-draw, over every
    applicable supported instance of the edit in the evaluation set."""
    t = inst[(inst.edit == edit) & inst.applicable & inst.supported]
    if not len(t):
        return None, None
    point = float(t[f"d_logit_{ref}"].mean()) / size
    rows = refits.rows(t[["profile", "edit"]])
    ok = rows >= 0
    prof = t.profile.to_numpy()[ok]
    prof_ok = [p for p in prof if int(p) in draws.pos]
    if not prof_ok:
        return point, None
    keep = np.array([int(p) in draws.pos for p in prof])
    Dl = refits.d_logit[rows[ok][keep]][:, draws.r]            # n x B
    W = draws.w(prof_ok)                                        # B x n
    return point, (W * Dl.T).sum(1) / W.sum(1) / size


def equivalents(fam: dict, inst: pd.DataFrame, refits: Refits, draws: Draws, ref: str, drift: dict | None = None) -> dict:
    yr, yr_d = _ref_unit_draws(inst, refits, draws, ref, "C6_age_plus5", 5.0)
    mm, mm_d = _ref_unit_draws(inst, refits, draws, ref, "C4_sbp_plus20", 20.0)
    out = {"reference_log_or_per_year_of_age": yr, "reference_log_or_per_mmHg_sbp": mm, "cells": {}, "suppressed_units": []}
    # a unit whose own interval includes zero would make the ratios explode or flip sign: suppress it
    for name, pt, ud in (("years_of_age", yr, yr_d), ("mmHg_sbp", mm, mm_d)):
        ci = _ci(ud) if ud is not None else [None, None]
        out[f"unit_ci_{name}"] = ci
        if pt is None or ci[0] is None or ci[0] * ci[1] <= 0:
            out["suppressed_units"].append(name)
    if "years_of_age" in out["suppressed_units"]:
        yr = None
    if "mmHg_sbp" in out["suppressed_units"]:
        mm = None
    dr = fam.get("_draws", {})
    for e, cells in fam["cells"].items():
        for m, r in cells.items():
            row = {}
            for unit, pt, ud in (("years_of_age", yr, yr_d), ("mmHg_sbp", mm, mm_d)):
                if not pt:
                    continue
                row[f"excess_in_{unit}"] = r["D_logit"] / pt
                if ud is not None and (e, m) in dr:
                    row[f"excess_in_{unit}_ci"] = _ci(dr[(e, m)]["D_logit"] / ud)
            out["cells"].setdefault(e, {})[m] = row
    if drift:
        out["formatting"] = {}
        for m, d in drift.items():
            f = {}
            for unit, pt in (("years_of_age", yr), ("mmHg_sbp", mm)):
                if pt and d.get("mean_abs_drift_logit") is not None:
                    f[f"formatting_drift_in_{unit}"] = d["mean_abs_drift_logit"] / abs(pt)
                if pt and d.get("mean_abs_repeat_logit") is not None:
                    f[f"repeat_noise_in_{unit}"] = d["mean_abs_repeat_logit"] / abs(pt)
            out["formatting"][m] = f
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def weights(fam: dict) -> dict:
    """Implied risk-factor weights against outcome weights across the family's fields, per system."""
    dr = fam.get("_draws", {})
    out = {}
    for m in fam["models"]:
        es = [e for e in fam["edits"] if m in fam["cells"].get(e, {})]
        if len(es) < 5:
            continue
        z = np.array([np.log(fam["cells"][e][m]["implied_or_model"]) for e in es])
        d = np.array([np.log(fam["cells"][e][m]["or_reference"]) for e in es])
        r = {"fields": es, "implied_log_or": z.tolist(), "reference_log_or": d.tolist(),
             "spearman": _spearman(z, d), "slope_through_origin": float((z * d).sum() / (d * d).sum()),
             "n_fields_wrong_sign": int(((np.sign(z) != np.sign(d)) & (np.abs(d) > 0.02)).sum())}
        if all((e, m) in dr for e in es):
            Z = np.column_stack([dr[(e, m)]["implied_log_or"] for e in es])
            Dd = np.column_stack([dr[(e, m)]["reference_log_or"] for e in es])
            r["spearman_ci"] = _ci([_spearman(Z[b], Dd[b]) for b in range(Z.shape[0])])
            r["slope_through_origin_ci"] = _ci((Z * Dd).sum(1) / (Dd * Dd).sum(1))
        out[m] = r
    return out


def consensus(fam: dict, systems: list[str] | None = None) -> dict:
    """Excess shared by the language models (mean D over systems, max-t over fields) and between-system disagreement."""
    systems = [m for m in (systems or LLMS) if m in fam["models"]]
    dr = fam.get("_draws", {})
    out = {"systems": systems, "fields": {}}
    rows, keys = [], []
    for e in fam["edits"]:
        ms = [m for m in systems if m in fam["cells"].get(e, {}) and (e, m) in dr]
        if len(ms) < 2:
            continue
        pts = np.array([fam["cells"][e][m]["D_logit"] for m in ms])
        M = np.column_stack([dr[(e, m)]["D_logit"] for m in ms])
        mean_d = M.mean(1)
        allm = [m for m in fam["models"] if m in fam["cells"].get(e, {}) and (e, m) in dr]
        A = np.column_stack([dr[(e, m)]["D_logit"] for m in allm])
        out["fields"][e] = {"mean_excess_llms": float(pts.mean()), "mean_excess_llms_ci": _ci(mean_d),
                            "same_sign_all_llms": bool(np.all(np.sign(pts) == np.sign(pts.mean()))),
                            "llms_contributing": ms, "n_llms": len(ms),
                            "between_system_sd": float(np.std([fam["cells"][e][m]["D_logit"] for m in allm], ddof=1)),
                            "between_system_sd_ci": _ci(A.std(1, ddof=1))}
        rows.append(mean_d); keys.append(e)
    if rows:
        M = np.column_stack(rows)
        se = M.std(0, ddof=1) + 1e-12
        crit = CRIT_INFLATION * float(np.quantile(np.max(np.abs(M - M.mean(0)) / se, axis=1), 0.95))
        out["critical_value"] = crit
        for j, e in enumerate(keys):
            p = out["fields"][e]["mean_excess_llms"]
            out["fields"][e]["mean_excess_llms_ci_simultaneous"] = [float(p - crit * se[j]), float(p + crit * se[j])]
    return out


def _cat(p: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(RISK_CUTS), np.asarray(p, float), side="right")


def reclassification(tabs: dict[str, pd.DataFrame], edits: list[str], B: int = 500, seed: int = 41) -> dict:
    """Share of people whose ten-year risk category (<5, 5-10, 10-20, 20-40, >=40%) changes after an edit, with both arms
    anchored on the reference baseline risk q0: the system's change z and the reference's change d are each applied to
    q0, so baseline miscalibration and call-to-call scatter in the baseline do not enter. The same share for formatting
    edits is the noise floor. Profile bootstrap; reference uncertainty not carried (sampling only)."""
    rng = np.random.default_rng(seed)
    out = {"cuts": list(RISK_CUTS), "note": "sampling uncertainty only", "cells": {}}
    for m, t in tabs.items():
        for e in edits:
            s = t[(t.edit == e) & t.q0.notna()].drop_duplicates("profile")
            if len(s) < 20:
                continue
            q0 = s.q0.to_numpy(); q1 = expit(logit(q0) + s.d_logit.to_numpy())
            qm = expit(logit(q0) + s.z_logit.to_numpy())        # the system's change applied to the same baseline risk
            mc = (_cat(qm) != _cat(q0)).astype(float)
            rc = (_cat(q1) != _cat(q0)).astype(float)
            up = (_cat(qm) > _cat(q0)).astype(float)
            idx = [rng.integers(0, len(s), len(s)) for _ in range(B)]
            out["cells"].setdefault(e, {})[m] = {
                "model_share": float(mc.mean()), "reference_share": float(rc.mean()), "excess_share": float(mc.mean() - rc.mean()),
                "excess_share_ci": _ci([mc[i].mean() - rc[i].mean() for i in idx]), "model_share_up": float(up.mean()),
                "n": int(len(s))}
        # noise floor: the same shares for edits that carry no information (I1-I3), applied to each person's baseline risk
        q0p = t[t.q0.notna()].drop_duplicates("profile").set_index("profile").q0
        irr = t[t.edit.isin(["I1_field_order", "I2_units", "I3_prose"]) & t.profile.isin(q0p.index)]
        if len(irr):
            q0i = q0p.loc[irr.profile].to_numpy()
            out.setdefault("formatting_noise_share", {})[m] = float(np.mean(_cat(expit(logit(q0i) + irr.z_logit.to_numpy())) != _cat(q0i)))
    return out


def heterogeneity(tabs: dict[str, pd.DataFrame], cohort: pd.DataFrame, edits: list[str], refits: Refits, draws: Draws,
                  max_draws: int = 300) -> dict:
    """Exploratory: does the per-person discrepancy (z - d, log-odds) vary with age (per decade), female sex and logit
    baseline reference risk? Joint least squares; each draw reweights people by the profile resample AND swaps in that
    draw's reference refit, so the reference's own noise in these interactions enters the interval."""
    co = cohort.set_index(cohort.SEQN.astype(int))
    out = {"note": "exploratory; sampling and reference uncertainty", "cells": {}}
    nb = min(max_draws, draws.B)
    for m, t in tabs.items():
        for e in edits:
            s_ = t[(t.edit == e) & t.q0.notna()].drop_duplicates("profile")
            rows = refits.rows(s_[["profile", "edit"]])
            keep = (rows >= 0) & s_.profile.astype(int).isin(draws.pos.keys()).to_numpy()
            s_, rows = s_[keep], rows[keep]
            if len(s_) < 50:
                continue
            c = co.loc[s_.profile.astype(int)]
            X = np.column_stack([np.ones(len(s_)), (c.age.to_numpy(float) - 60) / 10,
                                 (c.sex.astype(str).to_numpy() == "female").astype(float), logit(s_.q0.to_numpy())])
            names = ["age_per_decade", "female", "logit_baseline_risk"]
            if np.linalg.matrix_rank(X) < X.shape[1]:
                X, names = X[:, [0, 1, 3]], ["age_per_decade", "logit_baseline_risk"]
            z = s_.z_logit.to_numpy()
            pt = np.linalg.lstsq(X, z - s_.d_logit.to_numpy(), rcond=None)[0][1:]
            W = draws.w(s_.profile.astype(int).to_numpy())
            bs = []
            for b in range(nb):
                w = W[b]; y = z - refits.d_logit[rows, draws.r[b]]
                XtW = X.T * w
                bs.append(np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)[0][1:])
            bs = np.array(bs)
            out["cells"].setdefault(e, {})[m] = {n: {"slope": float(pt[j]), "ci": _ci(bs[:, j])} for j, n in enumerate(names)}
    return out


DOSE_PAIRS = {"hba1c": ("C1a_hba1c_plus1", "C1b_hba1c_plus2"), "creatinine": ("C2a_creatinine_x1.5", "C2b_creatinine_x2")}


def dose_response(tabs: dict[str, pd.DataFrame], refits: Refits, draws: Draws) -> dict:
    """Ratio of the mean log-odds response to the larger step over the smaller step, system against reference, on
    profiles with both steps valid and supported. The reference ratio is ~2 for HbA1c (+2 vs +1) and ~1.7 for creatinine
    (x2 vs x1.5 on a log scale); a system whose ratio stays near 1 is registering that a value changed, not by how much.
    Shared draws: profile resamples paired with reference refits."""
    out = {}
    for name, (small, large) in DOSE_PAIRS.items():
        for m, t in tabs.items():
            a = t[t.edit == small].drop_duplicates("profile").set_index("profile")
            b = t[t.edit == large].drop_duplicates("profile").set_index("profile")
            common = sorted(set(a.index) & set(b.index) & set(int(x) for x in draws.pos))
            if len(common) < 30:
                continue
            a, b = a.loc[common], b.loc[common]
            ra = refits.rows(pd.DataFrame({"profile": common, "edit": small}))
            rb = refits.rows(pd.DataFrame({"profile": common, "edit": large}))
            ok = (ra >= 0) & (rb >= 0)
            cm = [c for c, k in zip(common, ok) if k]
            if len(cm) < 30:
                continue
            a, b, ra, rb = a.loc[cm], b.loc[cm], ra[ok], rb[ok]
            W = draws.w(cm); Ws = W.sum(1)
            za, zb = (W * a.z_logit.to_numpy()).sum(1) / Ws, (W * b.z_logit.to_numpy()).sum(1) / Ws
            da = (W * refits.d_logit[ra][:, draws.r].T).sum(1) / Ws
            db = (W * refits.d_logit[rb][:, draws.r].T).sum(1) / Ws
            with np.errstate(divide="ignore", invalid="ignore"):
                rz, rd = zb / za, db / da
            out.setdefault(name, {})[m] = {
                "system_ratio": float(b.z_logit.mean() / a.z_logit.mean()) if a.z_logit.mean() else None,
                "system_ratio_ci": _ci(rz), "reference_ratio": float(b.d_logit.mean() / a.d_logit.mean()),
                "reference_ratio_ci": _ci(rd), "difference_ci": _ci(rz - rd), "n": len(cm)}
    return out


def inversions(fam: dict, min_ref_gap: float = 0.15) -> dict:
    """Pairs (demographic label L, clinical field C) where outcomes rank C above L in magnitude (|log-OR| gap >
    min_ref_gap) but the system moves more, in magnitude, for L than for C. For each: the share of shared draws in which the system's ordering is
    reversed while the reference's is not, and the interval of (D_L - D_C), which carries both kinds of uncertainty."""
    dr = fam.get("_draws", {})
    out = {}
    for m in fam["models"]:
        labs = [e for e in fam["edits"] if e.startswith("L") and m in fam["cells"].get(e, {}) and (e, m) in dr]
        clins = [e for e in fam["edits"] if e.startswith("C") and m in fam["cells"].get(e, {}) and (e, m) in dr]
        for L in labs:
            for C in clins:
                cl, cc = fam["cells"][L][m], fam["cells"][C][m]
                dL, dC = np.log(cl["or_reference"]), np.log(cc["or_reference"])
                zL, zC = np.log(cl["implied_or_model"]), np.log(cc["implied_or_model"])
                if not (abs(dC) - abs(dL) > min_ref_gap and abs(zL) > abs(zC)):     # magnitudes: sex has OR < 1
                    continue
                gL, gC = dr[(L, m)], dr[(C, m)]
                rev = np.mean((np.abs(gL["implied_log_or"]) > np.abs(gC["implied_log_or"]))
                              & (np.abs(gC["reference_log_or"]) > np.abs(gL["reference_log_or"])))
                out.setdefault(m, []).append({"label": L, "clinical": C, "implied_or_label": cl["implied_or_model"],
                                              "implied_or_clinical": cc["implied_or_model"], "reference_or_label": cl["or_reference"],
                                              "reference_or_clinical": cc["or_reference"], "share_draws_reversed": float(rev),
                                              "D_label_minus_D_clinical": float(cl["D_logit"] - cc["D_logit"]),
                                              "D_label_minus_D_clinical_ci": _ci(gL["D_logit"] - gC["D_logit"])})
        if m in out:
            out[m].sort(key=lambda r: -r["share_draws_reversed"])
    return out

"""Full analysis for one split. Pure functions over four inputs:
  calls      results/<split>/calls.parquet   (model, profile, edit, annotated, repeat, p, valid)
  instances  data/instances_<split>.parquet  (profile, edit, klass, feature, applicable, supported, d_* per reference)
  refits     data/refit_d_<ref>_<split>.npz  (d_logit, d_prob: n_instances x R; profile, edit keys)
  cohort     data/cohort.parquet             (SEQN, death_10y, covariates)
Uncertainty: every interval comes from one shared set of profile resamples (counts matrix C, B x n_profiles), and
draw b uses reference refit r = b mod R, so reference uncertainty and sampling uncertainty enter together and all
models are compared on the same draws."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CLIP = (0.005, 0.995)
RACE_EDIT = "L1_race_nhw_to_nhb"
PRIMARY_EDIT = RACE_EDIT          # default edit of per-edit summaries; race is one member of the confirmatory family


def _load_family() -> list[str]:
    """The confirmatory family: one edit per variable, config/perturbations.yaml."""
    try:
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "config" / "perturbations.yaml").read_text(encoding="utf-8"))
        return list(cfg["confirmatory_family"])
    except Exception:
        return ["C1a_hba1c_plus1", "C2a_creatinine_x1.5", "C3_albumin_minus0.5", "C4_sbp_plus20", "C5_smoking_never_to_current",
                "C6_age_plus5", "L1_race_nhw_to_nhb", "L2a_income_low", "L3a_education_low", "L4a_uninsured", "L5a_sex_male_to_female"]


FAMILY = _load_family()


def _load_inflation() -> float:
    """config/analysis.yaml crit_inflation. Fails loudly: a silent 1.0 would change every confirmatory interval."""
    import yaml
    from pathlib import Path
    f = Path(__file__).resolve().parents[2] / "config" / "analysis.yaml"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: crit_inflation must be set explicitly")
    return float(yaml.safe_load(f.read_text())["crit_inflation"])


CRIT_INFLATION = _load_inflation()    # applied to confirmatory, per-edit and consensus max-t; not to the prediction block
PHENOAGE_EDITS = ("C2a_creatinine_x1.5", "C2b_creatinine_x2", "C3_albumin_minus0.5", "C6_age_plus5")   # fields PhenoAge contains
D_KEYS = ("D_logit", "D_prob_pts", "D_logit_recal", "D_prob_pts_recal")
PRIMARY_IRRELEVANT = ["I1_field_order", "I2_units", "I3_prose"]
# primary systems first (Jev, then the confirmatory LLMs), then the secondary tiers; config marks tiers and disabled
# families; reasoning-arm systems ('gpt@low') follow in family order (ordered())
LLM_ORDER = ["jev", "gpt", "claude", "gemini", "muse", "glm",
             "gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
SECONDARY_ARMS = ("free", "pair", "reasoning")
NI_FRACTION = 0.10              # prediction non-inferiority margin: share of the age+sex-to-reference log-loss gap
NULL_BOUNDS = (0.10, 0.20)      # reporting bounds on |D| (log-odds) and |beta - 1|; not equivalence margins


def ordered(names) -> list[str]:
    """calls.model values in reporting order: LLM_ORDER, then reasoning-arm systems by family and effort; Jev's
    secondary questions (jev_bands, ...) are not systems."""
    names = set(names)
    eff = [n for n in names if "@" in n]
    key = lambda n: (LLM_ORDER.index(n.split("@")[0]) if n.split("@")[0] in LLM_ORDER else len(LLM_ORDER), n)
    return [m for m in LLM_ORDER if m in names] + sorted(eff, key=key)


def full_set_systems(models) -> list[str]:
    """Systems asked on every evaluation record: Jev and the primary, free and pair tiers. Systems of an arm run on the
    subset (config `arms` with records: subset, i.e. the reasoning arm 'gpt@low', ...) answer only those records,
    so they are left out of any block that takes the records every system scored (the baseline prediction block)."""
    from .clients import MODELS, system_tier
    subset_arms = {a for a, v in (MODELS.get("arms") or {}).items() if (v or {}).get("records") == "subset"}
    return [m for m in models if "@" not in m and system_tier(m) not in subset_arms]


def baseline_predictions(calls: pd.DataFrame, models, profiles, comparator_preds: dict | None = None) -> dict:
    """Baseline predictions for the prediction block: each full-set system's first valid raw answer per record (indexed
    by `profiles`) and the comparator references (not the prevalence-only comparator)."""
    out = {}
    for m in full_set_systems(models):
        b = calls[(calls.model == m) & (calls.edit == "baseline") & (calls.repeat == 0) & (~calls.annotated) & calls.valid & _variant(calls, "raw")]
        out[m] = b.drop_duplicates("profile").set_index("profile")["p"].reindex(profiles)
    for name, pr in (comparator_preds or {}).items():
        if name != "prevalence_only":
            out[name] = pr.reindex(profiles)
    return out


def is_primary(m: str) -> bool:
    """Jev or a primary-tier family: the only systems in the confirmatory max-t family."""
    from .clients import system_tier
    return system_tier(m) in ("jev", "primary")


def logit(p):
    p = np.clip(np.asarray(p, float), *CLIP)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def wlogit_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, iters: int = 30) -> tuple[float, float]:
    """Weighted logistic regression of y on (1, x) by IRLS: returns (intercept, slope)."""
    w = np.ones_like(x) if w is None else w
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = X @ beta
        p = np.clip(expit(eta), 1e-9, 1 - 1e-9)
        W = w * p * (1 - p)
        z = eta + (y - p) / (p * (1 - p))
        H = X.T @ (W[:, None] * X) + 1e-8 * np.eye(2)
        new = np.linalg.solve(H, X.T @ (W * z))
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return float(beta[0]), float(beta[1])


# ----------------------------------------------------------------------------- tables
def _variant(calls: pd.DataFrame, variant: str) -> pd.Series:
    return (calls["variant"] == variant) if "variant" in calls else pd.Series(variant == "raw", index=calls.index)


def contrast_table(calls: pd.DataFrame, inst: pd.DataFrame, model: str, ref: str, annotated: bool = False,
                   variant: str = "raw") -> pd.DataFrame:
    c = calls[(calls.model == model) & (calls.repeat == 0) & (calls.annotated == annotated) & calls.valid & _variant(calls, variant)]
    base = c[c.edit == "baseline"].drop_duplicates("profile").set_index("profile")["p"]
    ed = c[c.edit != "baseline"].drop_duplicates(["profile", "edit"]).merge(
        inst[inst.applicable & inst.supported], on=["profile", "edit"])
    ed = ed.assign(p_base=ed.profile.map(base)).dropna(subset=["p_base"])
    ed["z_logit"] = logit(ed.p) - logit(ed.p_base)
    ed["z_prob"] = ed.p - ed.p_base
    ed["d_logit"], ed["d_prob"] = ed[f"d_logit_{ref}"], ed[f"d_prob_{ref}"]
    ed["q0"] = ed[f"q0_{ref}"] if f"q0_{ref}" in ed else np.nan
    return ed[["profile", "edit", "klass", "feature", "p_base", "p", "z_logit", "z_prob", "d_logit", "d_prob", "q0"]].reset_index(drop=True)


@dataclass
class Refits:
    d_logit: np.ndarray
    d_prob: np.ndarray
    index: dict = field(default_factory=dict)       # (profile, edit) -> row

    @classmethod
    def load(cls, path) -> "Refits":
        z = np.load(path, allow_pickle=False)
        idx = {(int(p), str(e)): i for i, (p, e) in enumerate(zip(z["profile"], z["edit"]))}
        return cls(z["d_logit"], z["d_prob"], idx)

    @classmethod
    def from_inst(cls, inst: pd.DataFrame, ref: str) -> "Refits":
        """No refits: one column equal to reference `ref` for every applicable, supported instance."""
        t = inst[inst.applicable & inst.supported]
        return cls.degenerate(t.assign(d_logit=t[f"d_logit_{ref}"], d_prob=t[f"d_prob_{ref}"]))

    @classmethod
    def degenerate(cls, tab: pd.DataFrame) -> "Refits":
        """No refits available: one column equal to the primary reference (sampling uncertainty only)."""
        idx = {(int(p), str(e)): i for i, (p, e) in enumerate(zip(tab.profile, tab.edit))}
        return cls(tab.d_logit.to_numpy()[:, None], tab.d_prob.to_numpy()[:, None], idx)

    @property
    def R(self) -> int:
        return self.d_logit.shape[1]

    def rows(self, tab: pd.DataFrame) -> np.ndarray:
        return np.array([self.index.get((int(p), str(e)), -1) for p, e in zip(tab.profile, tab.edit)])


class Draws:
    """Shared profile resamples: counts matrix over the evaluation profiles and the refit used by each draw."""

    def __init__(self, profiles, B: int, R: int, seed: int = 20260922):
        self.profiles = np.asarray(sorted(profiles))
        self.pos = {int(p): i for i, p in enumerate(self.profiles)}
        rng = np.random.default_rng(seed)
        n = len(self.profiles)
        self.C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)
        self.r = np.arange(B) % max(R, 1)
        self.B = B

    def w(self, profiles) -> np.ndarray:
        """B x len(profiles) weights for rows belonging to `profiles`."""
        return self.C[:, [self.pos[int(p)] for p in profiles]]


def _ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))] if len(x) else [None, None]


# ----------------------------------------------------------------------------- primary endpoint
def recal_params_per_draw(base_p: pd.Series, y: pd.Series, halves: pd.Series, draws: Draws) -> dict:
    """Cross-fitted logistic recalibration: for each half h, (a, b) fitted on the OTHER half's baseline predictions,
    refitted inside every draw. Returns arrays a[h], b[h] of length B+1 (last entry = point estimate)."""
    out = {}
    for h in (0, 1):
        other = halves[halves != h].index.intersection(base_p.index)
        x, yy = logit(base_p.loc[other].to_numpy()), y.loc[other].to_numpy().astype(float)
        W = draws.w(other)
        a = np.empty(draws.B + 1); b = np.empty(draws.B + 1)
        for k in range(draws.B):
            a[k], b[k] = wlogit_fit(x, yy, W[k])
        a[-1], b[-1] = wlogit_fit(x, yy)
        out[h] = (a, b)
    return out


def _D_draws(s: pd.DataFrame, profiles: list, edit: str, refits: Refits, draws: Draws, rp: dict | None, halves: pd.Series):
    """Point estimates and draw vectors of D (raw and recalibrated, log-odds and points) for one model on `profiles`."""
    rows = refits.rows(pd.DataFrame({"profile": profiles, "edit": edit}))
    keep = rows >= 0
    profiles = [p for p, k in zip(profiles, keep) if k]
    rows = rows[keep]
    s = s.loc[profiles]
    Dl, Dp = refits.d_logit[rows][:, draws.r], refits.d_prob[rows][:, draws.r]
    W = draws.w(profiles); Ws = W.sum(1)
    zl, zp, pb, pe = s.z_logit.to_numpy(), s.z_prob.to_numpy(), s.p_base.to_numpy(), s.p.to_numpy()
    d_l0, d_p0 = s.d_logit.to_numpy(), s.d_prob.to_numpy()
    out = {"n": len(profiles),
           "point": {"D_logit": float(np.mean(zl - d_l0)), "D_prob_pts": float(np.mean(zp - d_p0) * 100),
                     "implied_log_or_model": float(np.mean(zl)), "log_or_reference": float(np.mean(d_l0))},
           "draws": {"D_logit": ((W * zl).sum(1) - (W * Dl.T).sum(1)) / Ws,
                     "D_prob_pts": (((W * zp).sum(1) - (W * Dp.T).sum(1)) / Ws) * 100,
                     "implied_log_or": (W * zl).sum(1) / Ws, "reference_log_or": (W * Dl.T).sum(1) / Ws}}
    if rp is not None:
        h = halves.loc[profiles].to_numpy()
        a = np.where(h[None, :] == 0, rp[0][0][:-1, None], rp[1][0][:-1, None])
        b = np.where(h[None, :] == 0, rp[0][1][:-1, None], rp[1][1][:-1, None])
        a0 = np.where(h == 0, rp[0][0][-1], rp[1][0][-1]); b0 = np.where(h == 0, rp[0][1][-1], rp[1][1][-1])
        zp_r = expit(a + b * logit(pe)[None, :]) - expit(a + b * logit(pb)[None, :])
        out["point"]["D_logit_recal"] = float(np.mean(b0 * zl - d_l0))
        out["point"]["D_prob_pts_recal"] = float(np.mean(expit(a0 + b0 * logit(pe)) - expit(a0 + b0 * logit(pb)) - d_p0) * 100)
        out["draws"]["D_logit_recal"] = ((W * (b * zl[None, :])).sum(1) - (W * Dl.T).sum(1)) / Ws
        out["draws"]["D_prob_pts_recal"] = (((W * zp_r).sum(1) - (W * Dp.T).sum(1)) / Ws) * 100
    return out


def _recal_setup(models, calls: pd.DataFrame, cohort: pd.DataFrame, draws: Draws, halves_seed: int = 7):
    y = cohort.set_index(cohort.SEQN.astype(int))["death_10y"]
    all_profiles = pd.Index(draws.profiles)
    halves = pd.Series(np.random.default_rng(halves_seed).integers(0, 2, len(all_profiles)), index=all_profiles)
    rps = {}
    for m in models:
        cb = calls[(calls.model == m) & (calls.edit == "baseline") & (calls.repeat == 0) & (~calls.annotated) & calls.valid & _variant(calls, "raw")]
        base_p = cb.drop_duplicates("profile").set_index("profile")["p"]
        rps[m] = recal_params_per_draw(base_p[base_p.index.isin(all_profiles)], y, halves, draws)
    return halves, rps


def confirmatory_family(tabs: dict[str, pd.DataFrame], refits: Refits, draws: Draws, calls: pd.DataFrame, cohort: pd.DataFrame,
                        edits: list[str] | None = None, per_mmhg: float | None = None, pairs: bool = True,
                        halves_seed: int = 7, min_profiles: int = 20, keep_draws: bool = False) -> dict:
    """The confirmatory analysis: D = mean(z - d) for every edit in the family and every model, each on
    the model's own valid, supported profiles, with simultaneous 95% intervals by max-t over ALL edit x model cells
    (one set of shared draws: profile resamples paired with reference refit b mod R). Race is one row of the family."""
    edits = list(FAMILY) if edits is None else list(edits)    # [] is an empty family, never the full one
    models = ordered(tabs)
    halves, rps = _recal_setup(models, calls, cohort, draws, halves_seed)
    cells, keys, D = {}, [], {k: [] for k in D_KEYS}
    sub, cell_draws = {}, {}
    for e in edits:
        for m in models:
            s = tabs[m][tabs[m].edit == e].drop_duplicates("profile").set_index("profile")
            miss = int((refits.rows(pd.DataFrame({"profile": s.index, "edit": e})) < 0).sum())
            if miss:                                  # keys are validated before the 20-pair rule is applied
                raise ValueError(f"{e} x {m}: {miss} supported records have no reference refit row")
            if len(s) < min_profiles:
                continue
            sub[(e, m)] = s
            o = _D_draws(s, sorted(s.index), e, refits, draws, rps[m], halves)
            r = dict(o["point"])
            r.update({f"{k}_ci": _ci(v) for k, v in o["draws"].items()})
            r["implied_or_model"] = float(np.exp(r.pop("implied_log_or_model")))
            r["or_reference"] = float(np.exp(r.pop("log_or_reference")))
            r["mean_model_contrast_pts"] = float(s.z_prob.mean() * 100)
            r["mean_reference_contrast_pts"] = float(s.d_prob.mean() * 100)
            r["n_profiles"] = o["n"]
            if per_mmhg:
                r["excess_in_sbp_mmhg"] = r["D_logit"] / per_mmhg
            cells.setdefault(e, {})[m] = r
            keys.append((e, m))
            for k in D_KEYS:
                D[k].append(o["draws"][k])
            cell_draws[(e, m)] = o["draws"]
    res = {"edits": edits, "models": models, "n_cells": len(keys), "n_tests": len(keys), "critical_value": {}, "cells": cells}
    if keep_draws:
        res["_draws"] = cell_draws     # in memory only; removed before the JSON is written
    if not keys:
        return res
    # clinical response slope per system: beta = sum_e zbar_e dbar_e / sum_e dbar_e^2 over the family's clinical fields
    # (one edit each; each field weighted by dbar_e^2, reported as field_weights), tested against 1 inside the same max-t
    # as the edit x system cells. Per-cell tests rarely detect a system that compresses every clinical change by the same
    # factor; the slope does. The recalibrated slope tests relative weighting only: uniform compression also compresses
    # the baseline, so recalibration undoes it by construction.
    clin = [e for e in edits if e.startswith("C")]
    slope_keys, slope_draws, slope_pts, slope_recal, slope_weights, slope_no_age = [], [], [], {}, {}, {}
    for m in models:
        es = [e for e in clin if (e, m) in keys]
        if len(es) < 3:
            continue
        zbar = np.array([cells[e][m]["implied_or_model"] for e in es]); zbar = np.log(zbar)
        dbar = np.log(np.array([cells[e][m]["or_reference"] for e in es]))
        Zd = np.column_stack([cell_draws[(e, m)]["implied_log_or"] for e in es])
        Dd = np.column_stack([cell_draws[(e, m)]["reference_log_or"] for e in es])
        slope_keys.append(m); slope_pts.append(float((zbar * dbar).sum() / (dbar * dbar).sum()))
        slope_weights[m] = {e: float(w) for e, w in zip(es, dbar ** 2 / (dbar ** 2).sum())}   # each field's share of the slope
        slope_draws.append((Zd * Dd).sum(1) / (Dd * Dd).sum(1))
        na = [i for i, e in enumerate(es) if not e.startswith("C6")]   # secondary: age (C6) carries most of the weight
        if len(na) >= 3:
            slope_no_age[m] = (float((zbar[na] * dbar[na]).sum() / (dbar[na] ** 2).sum()),
                               _ci((Zd[:, na] * Dd[:, na]).sum(1) / (Dd[:, na] ** 2).sum(1)))
        if all("D_logit_recal" in cell_draws[(e, m)] for e in es):   # after recalibration: is it more than miscalibration?
            zr = np.array([cells[e][m]["D_logit_recal"] for e in es]) + dbar
            Zr = np.column_stack([cell_draws[(e, m)]["D_logit_recal"] + cell_draws[(e, m)]["reference_log_or"] for e in es])
            slope_recal[m] = (float((zr * dbar).sum() / (dbar * dbar).sum()), _ci((Zr * Dd).sum(1) / (Dd * Dd).sum(1)))
    for k in D_KEYS:
        M = np.column_stack(D[k])
        pts = np.array([cells[e][m][k] for e, m in keys])
        if k == "D_logit" and slope_keys:
            M = np.column_stack([M] + slope_draws)
            pts = np.r_[pts, slope_pts]
        se = M.std(0, ddof=1) + 1e-12
        raw = float(np.quantile(np.max(np.abs(M - M.mean(0)) / se, axis=1), 0.95))
        crit = CRIT_INFLATION * raw
        res["critical_value"][k] = crit
        res.setdefault("critical_value_raw", {})[k] = raw
        res["crit_inflation"] = CRIT_INFLATION
        for j, (e, m) in enumerate(keys):
            cells[e][m][f"{k}_ci_simultaneous"] = [float(pts[j] - crit * se[j]), float(pts[j] + crit * se[j])]
            # marginal Wald form on the same bootstrap SE, reported beside the percentile interval (`{k}_ci`)
            cells[e][m][f"{k}_ci_wald"] = [float(pts[j] - 1.96 * se[j]), float(pts[j] + 1.96 * se[j])]
        if k == "D_logit":   # reporting bounds for a null result (no equivalence claim): |D| values the interval excludes
            for j, (e, m) in enumerate(keys):
                lo, hi = cells[e][m]["D_logit_ci_simultaneous"]
                cells[e][m]["rules_out_abs_D"] = {str(b): bool(-b < lo and hi < b) for b in NULL_BOUNDS}
        if k == "D_logit" and slope_keys:
            res["clinical_slope"] = {}
            for i, m in enumerate(slope_keys):
                j = len(keys) + i
                lo, hi = float(pts[j] - crit * se[j]), float(pts[j] + crit * se[j])
                res["clinical_slope"][m] = {"beta": float(pts[j]), "ci": _ci(slope_draws[i]), "ci_simultaneous": [lo, hi],
                                            "ci_wald": [float(pts[j] - 1.96 * se[j]), float(pts[j] + 1.96 * se[j])],
                                            "excludes_one": not (lo <= 1 <= hi), "fields": [e for e in clin if (e, m) in keys],
                                            "rules_out_abs_beta_minus_1": {str(b): bool(1 - b < lo and hi < 1 + b) for b in NULL_BOUNDS},
                                            "field_weights": slope_weights.get(m),
                                            "beta_without_age": slope_no_age.get(m, (None, None))[0],
                                            "beta_without_age_ci": slope_no_age.get(m, (None, None))[1],
                                            "beta_recalibrated": slope_recal.get(m, (None, None))[0],
                                            "beta_recalibrated_ci": slope_recal.get(m, (None, None))[1]}
            res["n_tests"] = len(keys) + len(slope_keys)
    res["simultaneous_excludes_zero"] = {e: [m for m, r in cells[e].items()
                                             if not (r["D_logit_ci_simultaneous"][0] <= 0 <= r["D_logit_ci_simultaneous"][1])]
                                         for e in cells}
    if pairs:   # each pair on its own common records (another system's refusals never shrink it); n reported
        res["paired"] = {}
        for e in cells:
            ms = list(cells[e])
            out_e = {}
            for i, a in enumerate(ms):
                for b in ms[i + 1:]:
                    common = sorted(set(sub[(e, a)].index) & set(sub[(e, b)].index))
                    if len(common) < min_profiles:
                        continue
                    da = _D_draws(sub[(e, a)], common, e, refits, draws, rps[a], halves)
                    db = _D_draws(sub[(e, b)], common, e, refits, draws, rps[b], halves)
                    out_e[f"{a} - {b}"] = {"diff": float(da["point"]["D_logit"] - db["point"]["D_logit"]),
                                           "ci": _ci(da["draws"]["D_logit"] - db["draws"]["D_logit"]), "n": len(common)}
            if out_e:
                res["paired"][e] = out_e
    return res


def primary_D(tabs: dict[str, pd.DataFrame], refits: Refits, draws: Draws, calls: pd.DataFrame, cohort: pd.DataFrame,
              edit: str = RACE_EDIT, halves_seed: int = 7) -> dict:
    """Per-edit D with max-t across models only (secondary edits outside the family, M1 detail). Each model's D uses ITS OWN valid, supported profiles (one model's refusals
    never shrink another's sample); paired differences use complete pairs. Raw and cross-fitted logistic recalibration;
    percentile and simultaneous (max-t over shared draws) intervals. Also reports the model-implied and reference
    log odds ratios (mean contrasts on the log-odds scale)."""
    models = ordered(tabs)
    sub = {m: tabs[m][tabs[m].edit == edit].drop_duplicates("profile").set_index("profile") for m in models}
    sub = {m: s for m, s in sub.items() if len(s)}
    models = [m for m in models if m in sub]
    if not models:
        return {"edit": edit, "n_profiles": 0}
    y = cohort.set_index(cohort.SEQN.astype(int))["death_10y"]
    all_profiles = pd.Index(draws.profiles)
    halves = pd.Series(np.random.default_rng(halves_seed).integers(0, 2, len(all_profiles)), index=all_profiles)
    rps = {}
    for m in models:
        cb = calls[(calls.model == m) & (calls.edit == "baseline") & (calls.repeat == 0) & (~calls.annotated) & calls.valid & _variant(calls, "raw")]
        base_p = cb.drop_duplicates("profile").set_index("profile")["p"]
        rps[m] = recal_params_per_draw(base_p[base_p.index.isin(all_profiles)], y, halves, draws)
    common = sorted(set.intersection(*[set(s.index) for s in sub.values()]))
    res = {"edit": edit, "n_profiles": len(common), "n_profiles_by_model": {}, "models": {}}
    own = {m: _D_draws(sub[m], sorted(sub[m].index), edit, refits, draws, rps[m], halves) for m in models}
    for m in models:
        o = own[m]
        r = dict(o["point"])
        r.update({f"{k}_ci": _ci(v) for k, v in o["draws"].items()})
        r["implied_or_model"] = float(np.exp(r.pop("implied_log_or_model")))
        r["or_reference"] = float(np.exp(r.pop("log_or_reference")))
        r["mean_model_contrast_pts"] = float(sub[m].z_prob.mean() * 100)
        r["mean_reference_contrast_pts"] = float(sub[m].d_prob.mean() * 100)
        r["recal_slope_by_half"] = [rps[m][0][1][-1], rps[m][1][1][-1]]
        res["models"][m] = r
        res["n_profiles_by_model"][m] = o["n"]
    for key in ("D_logit", "D_prob_pts", "D_logit_recal", "D_prob_pts_recal"):
        M = np.column_stack([own[m]["draws"][key] for m in models])
        pts = np.array([res["models"][m][key] for m in models])
        se = M.std(0, ddof=1) + 1e-12
        crit = CRIT_INFLATION * float(np.quantile(np.max(np.abs(M - M.mean(0)) / se, axis=1), 0.95))
        for j, m in enumerate(models):
            res["models"][m][f"{key}_ci_simultaneous"] = [float(pts[j] - crit * se[j]), float(pts[j] + crit * se[j])]
        res.setdefault("simultaneous_critical_value", {})[key] = crit
    if common and len(models) > 1:
        pair = {m: _D_draws(sub[m], common, edit, refits, draws, rps[m], halves) for m in models}
        diff = lambda a, b, k: {"diff": float(pair[a]["point"][k] - pair[b]["point"][k]), "ci": _ci(pair[a]["draws"][k] - pair[b]["draws"][k])}
        if "jev" in models:
            res["paired_vs_jev"] = {m: {k: diff(m, "jev", k) for k in ("D_logit", "D_prob_pts", "D_logit_recal")}
                                    for m in models if m != "jev"}
        res["paired_all"] = {f"{a} - {b}": diff(a, b, "D_logit") for i, a in enumerate(models) for b in models[i + 1:]}
    return res


def discrepancy_by_risk(tabs: dict[str, pd.DataFrame], edit: str = PRIMARY_EDIT, B: int = 500, seed: int = 37) -> dict:
    """Secondary: does the discrepancy z - d (log-odds) change with baseline reference risk? Per model, the
    OLS slope of z - d on logit q0 (centred), with a profile-bootstrap interval (sampling uncertainty only; continuous,
    no terciles)."""
    out, rng = {}, np.random.default_rng(seed)
    for m, t in tabs.items():
        t = t[(t.edit == edit) & t.q0.notna()]
        if len(t) < 20:
            continue
        x = logit(t.q0.to_numpy()); x = x - x.mean(); y = (t.z_logit - t.d_logit).to_numpy()
        slope = lambda i: float(np.polyfit(x[i], y[i], 1)[0])
        bs = [slope(rng.integers(0, len(x), len(x))) for _ in range(B)]
        out[m] = {"slope_per_logit_baseline_risk": slope(np.arange(len(x))), "ci": _ci(bs), "n": int(len(x))}
    return out


# ----------------------------------------------------------------------------- descriptive class summaries
def class_beta_draws(tab: pd.DataFrame, refits: Refits, draws: Draws, B_use: int | None = None) -> dict:
    """Edit-level-mean slope per class: point (primary reference) and draws (profile resample + refit)."""
    out = {}
    for klass, t in tab.groupby("klass"):
        if klass == "irrelevant":
            continue
        t = t.reset_index(drop=True)
        edits = sorted(t.edit.unique())
        e_idx = t.edit.map({e: i for i, e in enumerate(edits)}).to_numpy()
        feat = t.groupby("edit")["feature"].first().reindex(edits)
        w_e = (1.0 / feat.map(feat.value_counts())).to_numpy()
        rows = refits.rows(t)
        ok = rows >= 0
        t, e_idx, rows = t[ok], e_idx[ok], rows[ok]
        zl = t.z_logit.to_numpy()
        W = draws.w(t.profile)
        Bn = draws.B if B_use is None else min(B_use, draws.B)

        def beta(wrow, d):
            den = np.bincount(e_idx, weights=wrow, minlength=len(edits)) + 1e-12
            zb = np.bincount(e_idx, weights=wrow * zl, minlength=len(edits)) / den
            db = np.bincount(e_idx, weights=wrow * d, minlength=len(edits)) / den
            keep = den > 1e-9
            return float(np.sum(w_e[keep] * db[keep] * zb[keep]) / np.sum(w_e[keep] * db[keep] ** 2)), zb, db

        b0, zb0, db0 = beta(np.ones(len(zl)), t.d_logit.to_numpy())
        bs = np.array([beta(W[k], refits.d_logit[rows, draws.r[k]])[0] for k in range(Bn)])
        out[klass] = {"beta": b0, "beta_ci": _ci(bs), "draws": bs,
                      "edit_means": [{"edit": e, "z_logit": float(zb0[i]), "d_logit": float(db0[i]),
                                      "z_prob_pts": float(t[t.edit == e].z_prob.mean() * 100),
                                      "d_prob_pts": float(t[t.edit == e].d_prob.mean() * 100),
                                      "n": int((t.edit == e).sum())} for i, e in enumerate(edits)],
                      "mean_abs_discrepancy_pts": float(np.mean(np.abs(t.z_prob - t.d_prob)) * 100)}
    if {"label", "clinical"} <= set(out):
        ratio = out["label"]["draws"] / out["clinical"]["draws"]
        interpretable = out["clinical"]["beta_ci"][0] is not None and out["clinical"]["beta_ci"][0] > 0
        out["rho"] = {"rho": out["label"]["beta"] / out["clinical"]["beta"] if out["clinical"]["beta"] else None,
                      "ci": _ci(ratio) if interpretable else "unbounded (clinical slope interval includes 0)",
                      "interpretable": bool(interpretable)}
    for k in ("label", "clinical"):
        if k in out:
            out[k].pop("draws")
    return out


def drift(calls: pd.DataFrame, tab: pd.DataFrame, model: str, draws: Draws) -> dict:
    irr = tab[tab.edit.isin(PRIMARY_IRRELEVANT)]
    c = calls[(calls.model == model) & (~calls.annotated) & calls.valid & _variant(calls, "raw")]
    first = c[c.repeat == 0].drop_duplicates(["profile", "edit"]).set_index(["profile", "edit"])["p"]
    rep = c[c.repeat > 0]
    rep = rep.assign(p0=[first.get((p, e), np.nan) for p, e in zip(rep.profile, rep.edit)]).dropna(subset=["p0"])
    rep = rep.assign(a_logit=np.abs(logit(rep.p) - logit(rep.p0)), a_prob=np.abs(rep.p - rep.p0))
    out = {"mean_abs_drift_logit": float(np.abs(irr.z_logit).mean()) if len(irr) else None,
           "mean_abs_drift_pts": float(np.abs(irr.z_prob).mean() * 100) if len(irr) else None,
           "mean_abs_repeat_logit": float(rep.a_logit.mean()) if len(rep) else None,
           "mean_abs_repeat_pts": float(rep.a_prob.mean() * 100) if len(rep) else None,
           "identical_repeat_share": float((rep.a_prob == 0).mean()) if len(rep) else None,
           "by_edit": {e: {"mean_abs_pts": float(np.abs(g.z_prob).mean() * 100), "mean_signed_pts": float(g.z_prob.mean() * 100),
                           "n": int(len(g))} for e, g in irr.groupby("edit")}}
    if len(irr) and len(rep):
        Wi, Wr = draws.w(irr.profile), draws.w(rep.profile)
        di = (Wi * np.abs(irr.z_prob.to_numpy())).sum(1) / Wi.sum(1)
        dr = (Wr * rep.a_prob.to_numpy()).sum(1) / np.maximum(Wr.sum(1), 1e-9)
        out["excess_pts"] = float(out["mean_abs_drift_pts"] - out["mean_abs_repeat_pts"])
        out["excess_pts_ci"] = _ci((di - dr) * 100)
        out["exceeds_operational_2pts"] = bool(out["excess_pts_ci"][0] is not None and out["excess_pts_ci"][0] > 2.0)
    return out


def direction_agreement(tab: pd.DataFrame, refits: Refits) -> dict:
    rows = refits.rows(tab)
    ok = rows >= 0
    t = tab[ok]
    sd = refits.d_logit[rows[ok]].std(1)
    indet = np.abs(t.d_logit.to_numpy()) < sd
    agree = np.sign(t.z_logit.to_numpy()) == np.sign(t.d_logit.to_numpy())
    res = {}
    for k in ("clinical", "label"):
        m = (t.klass == k).to_numpy()
        res[k] = {"agree": float(agree[m & ~indet].mean()) if (m & ~indet).any() else None,
                  "indeterminate_share": float(indet[m].mean()) if m.any() else None}
    return res


# ----------------------------------------------------------------------------- baseline performance
def performance(p: np.ndarray, y: np.ndarray, B: int = 500, seed: int = 1) -> dict:
    p = np.clip(np.asarray(p, float), *CLIP); y = np.asarray(y, int)

    def stats(pp, yy, w=None):
        a, b = wlogit_fit(logit(pp), yy.astype(float), w)
        ll = -np.average(yy * np.log(pp) + (1 - yy) * np.log(1 - pp), weights=w)
        return {"auroc": float(roc_auc_score(yy, pp, sample_weight=w)), "calibration_slope": b, "calibration_intercept": a,
                "brier": float(np.average((pp - yy) ** 2, weights=w)), "log_loss": float(ll), "mean_p": float(np.average(pp, weights=w))}

    point = stats(p, y)
    rng = np.random.default_rng(seed)
    bs = [stats(p, y, np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float)) for _ in range(B)]
    return {k: {"est": v, "ci": _ci([b[k] for b in bs])} for k, v in point.items()} | {"n": int(len(y)), "deaths": int(y.sum())}


# ----------------------------------------------------------------------------- Jev-specific diagnostics
def jev_resolution(calls: pd.DataFrame, tab: pd.DataFrame) -> dict:
    c = calls[(calls.model == "jev") & calls.valid & (calls.repeat == 0)]
    on_grid = np.isclose(np.round(c.p * 100), c.p * 100, atol=1e-6)
    return {"share_on_0.01_grid": float(on_grid.mean()), "share_p_equal_0": float((c.p == 0).mean()),
            "share_zero_contrast_by_class": {k: float((g.z_prob == 0).mean()) for k, g in tab.groupby("klass")}}


def output_granularity(calls: pd.DataFrame, tab: pd.DataFrame, model: str) -> dict:
    """How coarse are a system's probabilities? Share on the 0.05 and 0.01 grids, at or below 1%, distinct values, and
    the share of family contrasts that are exactly zero."""
    c = calls[(calls.model == model) & calls.valid & (calls.repeat == 0) & _variant(calls, "raw")]
    p = c.p.to_numpy(float)
    if not len(p):
        return {}
    on = lambda g: float(np.mean(np.isclose(np.round(p / g) * g, p, atol=1e-9)))
    fam = tab[tab.edit.isin(FAMILY)]
    return {"n_outputs": int(len(p)), "share_on_0.05_grid": on(0.05), "share_on_0.01_grid": on(0.01),
            "share_at_or_below_0.01": float(np.mean(p <= 0.01)), "n_distinct_values": int(len(np.unique(np.round(p, 6)))),
            "share_zero_family_contrasts": float(np.mean(fam.z_prob == 0)) if len(fam) else None}


def jev_complement(calls: pd.DataFrame) -> dict:
    d = calls[(calls.model == "jev") & (calls.edit == "baseline") & (calls.repeat == 0) & (~calls.annotated) & calls.valid & _variant(calls, "raw")]
    a = calls[(calls.model == "jev_complement") & calls.valid]
    m = d[["profile", "p"]].merge(a[["profile", "p"]].rename(columns={"p": "p_alive"}), on="profile")
    s = m.p + m.p_alive
    return {"n": int(len(s)), "sum_mean": float(s.mean()) if len(s) else None, "sum_sd": float(s.std()) if len(s) else None,
            "sum_range": [float(s.min()), float(s.max())] if len(s) else None, "share_within_0.05_of_1": float((np.abs(s - 1) <= 0.05).mean()) if len(s) else None}


def cost_latency(calls: pd.DataFrame, models_cfg: dict) -> dict:
    """Billed tokens priced at the configured rate of the route each answer came through, per usable answer (failed and
    unusable calls count in the cost), and synchronous latency. Descriptive."""
    out = {}
    if "tokens_in" not in calls:
        return out
    for m, g in calls.groupby("model"):
        base = "jev" if m.startswith("jev") else m.split("@")[0]      # 'gpt@low': the family's prices and route
        tin, tout = g.tokens_in.fillna(0).to_numpy(float), g.tokens_out.fillna(0).to_numpy(float)
        if base == "jev":
            usd = tin * models_cfg["jev"]["price_per_mtok_input"] / 1e6
        elif base in models_cfg["families"]:
            f = models_cfg["families"][base]
            bt = (models_cfg.get("batch") or {}).get("families", {}).get(base, f)
            batch = (g["route"] == "batch").to_numpy() if "route" in g else np.zeros(len(g), bool)
            usd = np.where(batch, tin * bt["price_in"] + tout * bt["price_out"], tin * f["price_in"] + tout * f["price_out"]) / 1e6
        else:
            continue
        n_ok = int(g.valid.sum())
        lat = g["latency_s"].dropna() if "latency_s" in g else pd.Series(dtype=float)
        out[m] = {"calls": int(len(g)), "usable": n_ok, "usd_total": float(usd.sum()),
                  "usd_per_usable_answer": float(usd.sum() / n_ok) if n_ok else None,
                  "mean_tokens_out": float(tout.mean()) if len(tout) else None,
                  "median_latency_s": float(lat.median()) if len(lat) else None,
                  "routes": g["route"].value_counts().to_dict() if "route" in g else {}}
    j = (out.get("jev") or {}).get("usd_per_usable_answer")
    for m, v in out.items():                  # the economy axis: each system's cost per usable estimate over Jev's
        if j and v.get("usd_per_usable_answer"):
            v["cost_ratio_to_jev"] = float(v["usd_per_usable_answer"] / j)
    return out


def jev_bundle(calls: pd.DataFrame, tol: float = 0.005) -> dict:
    """Question isolation (secondary): the death probability asked alone (primary call) against the same question
    asked in one pass with the complement and the risk bands. Share identical within `tol`, mean absolute change on the
    log-odds scale, the same two quantities for asking the single question twice (the noise floor), and the coherence
    of death and survival answered in one pass. Descriptive; percentile bootstrap over records."""
    base = calls[(calls.model == "jev") & (calls.edit == "baseline") & (~calls.annotated) & calls.valid & _variant(calls, "raw")]
    b0 = base[base.repeat == 0][["profile", "p"]]
    bu = calls[(calls.model == "jev_bundle") & calls.valid]
    m = b0.merge(bu[["profile", "p"] + [c for c in ("p_alive", "p_bands") if c in bu]], on="profile", suffixes=("", "_bundle"))
    if not len(m):
        return {"n": 0}
    rng = np.random.default_rng(31)
    dz = np.abs(np.asarray(logit(np.clip(m.p_bundle, *CLIP))) - np.asarray(logit(np.clip(m.p, *CLIP))))
    same = (np.abs(m.p_bundle - m.p) <= tol).to_numpy().astype(float)
    idx = [rng.integers(0, len(m), len(m)) for _ in range(1000)]
    out = {"n": int(len(m)), "share_identical": float(same.mean()), "share_identical_ci": _ci([same[i].mean() for i in idx]),
           "mean_abs_change_logit": float(dz.mean()), "mean_abs_change_logit_ci": _ci([dz[i].mean() for i in idx])}
    r1 = base[base.repeat == 1][["profile", "p"]].merge(b0, on="profile", suffixes=("_r1", ""))
    if len(r1):
        out["repeat_share_identical"] = float((np.abs(r1.p_r1 - r1.p) <= tol).mean())
        out["repeat_mean_abs_change_logit"] = float(np.abs(logit(np.clip(r1.p_r1, *CLIP)) - logit(np.clip(r1.p, *CLIP))).mean())
    if "p_alive" in m and m.p_alive.notna().any():
        s = (m.p_bundle + m.p_alive).dropna()
        out["bundle_sum_mean"] = float(s.mean()); out["bundle_share_within_0.05_of_1"] = float((np.abs(s - 1) <= 0.05).mean())
    return out


# ----------------------------------------------------------------------------- prediction against outcomes
def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, *CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def prediction_block(preds: dict[str, pd.Series], y: pd.Series, reference: str = "reference_spline_logit",
                     focal: str = "jev", B: int = 1000, seed: int = 29, focal_versus: list[str] | None = None) -> dict:
    """Baseline prediction for every system on the records all of them scored (outcomes directly; no edits).
    Per-record log-loss (clipped to [0.005, 0.995]) is the primary metric, Brier the robustness metric, AUROC and the
    calibration slope/intercept descriptive. Log-loss is split into a calibration component (drop after cross-fitted
    logistic recalibration) and the recalibrated remainder; systems are compared as returned and after recalibration.
    Paired differences share one set of record resamples; max-t simultaneous intervals within each comparison family
    (every system vs the outcome reference; the focal system vs every other system). Magnitudes are also expressed as a
    fraction of the log-loss gap between age + sex and the outcome reference."""
    names = [n for n, s in preds.items() if s is not None]
    common = sorted(set.intersection(*[set(preds[n].dropna().index) for n in names]) & set(y.index))
    yy = y.loc[common].to_numpy(float)
    P = np.column_stack([np.clip(preds[n].loc[common].to_numpy(float), *CLIP) for n in names])
    n = len(common)
    rng = np.random.default_rng(seed)
    halves = rng.integers(0, 2, n)
    Pr = np.empty_like(P)
    for j in range(P.shape[1]):
        for h in (0, 1):
            a, b = wlogit_fit(logit(P[halves != h, j]), yy[halves != h])
            Pr[halves == h, j] = expit(a + b * logit(P[halves == h, j]))
    LL = np.column_stack([_ll(P[:, j], yy) for j in range(P.shape[1])])
    LLr = np.column_stack([_ll(Pr[:, j], yy) for j in range(P.shape[1])])
    BR = (P - yy[:, None]) ** 2
    C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)
    mLL, mLLr, mBR = C @ LL / n, C @ LLr / n, C @ BR / n            # B x S
    out = {"n_records": n, "deaths": int(yy.sum()), "systems": {}}
    for j, nm in enumerate(names):
        a, b = wlogit_fit(logit(P[:, j]), yy)
        out["systems"][nm] = {"log_loss": float(LL[:, j].mean()), "log_loss_ci": _ci(mLL[:, j]),
                              "log_loss_recalibrated": float(LLr[:, j].mean()),
                              "calibration_component": float(LL[:, j].mean() - LLr[:, j].mean()),
                              "brier": float(BR[:, j].mean()), "brier_ci": _ci(mBR[:, j]),
                              "auroc": float(roc_auc_score(yy, P[:, j])), "calibration_slope": b, "calibration_intercept": a,
                              "mean_p": float(P[:, j].mean())}
    gap = None
    if "reference_age_sex" in names and reference in names:
        gap = out["systems"]["reference_age_sex"]["log_loss"] - out["systems"][reference]["log_loss"]
        out["age_sex_to_reference_gap"] = gap

    def family(pairs, key_draws, key_point):
        D = np.column_stack([key_draws[:, names.index(a)] - key_draws[:, names.index(b)] for a, b in pairs])
        pts = np.array([key_point[names.index(a)] - key_point[names.index(b)] for a, b in pairs])
        se = D.std(0, ddof=1) + 1e-12
        crit = float(np.quantile(np.max(np.abs(D - D.mean(0)) / se, axis=1), 0.95))   # records only: no inflation
        res = {}
        for k, (a, b) in enumerate(pairs):
            res[f"{a} - {b}"] = {"diff": float(pts[k]), "ci": _ci(D[:, k]),
                                 "ci_simultaneous": [float(pts[k] - crit * se[k]), float(pts[k] + crit * se[k])],
                                 "fraction_of_age_sex_gap": float(pts[k] / gap) if gap else None}
            if gap:   # non-inferiority of a against b: the upper simultaneous bound of (a - b) below 10% of the gap
                res[f"{a} - {b}"]["noninferiority_margin"] = float(NI_FRACTION * gap)
                res[f"{a} - {b}"]["noninferior"] = bool(pts[k] + crit * se[k] < NI_FRACTION * gap)
        return {"critical_value": crit, "pairs": res}

    pt, ptr = LL.mean(0), LLr.mean(0)
    if reference in names:
        vs_ref = [(nm, reference) for nm in names if nm != reference]
        out["vs_reference"] = family(vs_ref, mLL, pt)
        out["vs_reference_recalibrated"] = family(vs_ref, mLLr, ptr)
    if focal in names:
        vs_focal = [(focal, nm) for nm in names if nm not in (focal, reference) and not nm.startswith("reference_") and nm != "phenoage_10y"
                    and (focal_versus is None or nm in focal_versus)]     # the primary LLMs (non-inferiority family)
        if vs_focal:
            out["focal_vs_llms"] = family(vs_focal, mLL, pt)
            out["focal_vs_llms_recalibrated"] = family(vs_focal, mLLr, ptr)
            out["focal_vs_llms_brier"] = family(vs_focal, mBR, BR.mean(0))
        if "reference_age_sex" in names:     # the Jev interpretive condition: does it beat age + sex on log-loss?
            out["focal_vs_age_sex"] = family([(focal, "reference_age_sex")], mLL, pt)
    return out


def m2_shift(calls: pd.DataFrame, cohort: pd.DataFrame, model: str, profiles, B: int = 1000, seed: int = 31) -> dict:
    """Within-record effect of removing the race field: mean change in log-odds by race group and the Black-minus-White
    difference (how much of the model's race-specific component hiding the field removes)."""
    c = calls[(calls.model == model) & (calls.repeat == 0) & (~calls.annotated) & calls.valid & _variant(calls, "raw")]
    raw = c[c.edit == "baseline"].drop_duplicates("profile").set_index("profile")["p"]
    m2 = c[c.edit == "M2_race_removed"].drop_duplicates("profile").set_index("profile")["p"]
    idx = sorted(set(raw.index) & set(m2.index) & set(profiles))
    if len(idx) < 50:
        return {}
    g = cohort.set_index(cohort.SEQN.astype(int)).loc[idx, "race_ethnicity"].to_numpy()
    d = logit(m2.loc[idx].to_numpy()) - logit(raw.loc[idx].to_numpy())
    rng = np.random.default_rng(seed)
    out = {k: {"n": int((g == k).sum()), "mean_change_logit": float(d[g == k].mean())} for k in GROUPS if (g == k).sum() >= 20}
    w, b_ = np.where(g == "Non-Hispanic White")[0], np.where(g == "Non-Hispanic Black")[0]
    if len(w) >= 20 and len(b_) >= 20:
        diff = d[b_].mean() - d[w].mean()
        bs = [d[rng.choice(b_, len(b_))].mean() - d[rng.choice(w, len(w))].mean() for _ in range(B)]
        out["black_minus_white_logit"] = {"est": float(diff), "ci": _ci(bs)}
    return out


# ----------------------------------------------------------------------------- mitigation and group calibration
GROUPS = ["Non-Hispanic White", "Non-Hispanic Black", "Mexican American"]


def group_calibration(p: pd.Series, cohort: pd.DataFrame, B: int = 1000, seed: int = 3) -> dict:
    """Observed/expected deaths by recorded race group, and the NHB/NHW ratio of O/E, with profile-bootstrap CIs.
    p: predicted ten-year death probability indexed by profile (evaluation split)."""
    co = cohort.set_index(cohort.SEQN.astype(int)).loc[p.index]
    y, g = co["death_10y"].to_numpy(float), co["race_ethnicity"].to_numpy()
    pv = np.clip(p.to_numpy(float), 1e-6, 1 - 1e-6)
    rng = np.random.default_rng(seed)
    idx = {k: np.where(g == k)[0] for k in GROUPS}
    out = {}
    for k, ix in idx.items():
        if len(ix) < 20:
            continue
        oe = y[ix].sum() / pv[ix].sum()
        bs = [(y[b].sum() / pv[b].sum()) for b in (rng.choice(ix, len(ix)) for _ in range(B))]
        out[k] = {"n": int(len(ix)), "deaths": int(y[ix].sum()), "expected": float(pv[ix].sum()), "O_E": float(oe), "O_E_ci": _ci(bs)}
    if "Non-Hispanic White" in out and "Non-Hispanic Black" in out:
        w, b_ = idx["Non-Hispanic White"], idx["Non-Hispanic Black"]
        ratio = (y[b_].sum() / pv[b_].sum()) / (y[w].sum() / pv[w].sum())
        bs = []
        for _ in range(B):
            bw, bb = rng.choice(w, len(w)), rng.choice(b_, len(b_))
            bs.append((y[bb].sum() / pv[bb].sum()) / (y[bw].sum() / pv[bw].sum()))
        out["ratio_NHB_to_NHW"] = {"est": float(ratio), "ci": _ci(bs)}
    return out


def mitigation(calls: pd.DataFrame, inst: pd.DataFrame, cohort: pd.DataFrame, refits: Refits, draws: Draws,
               models: list[str], primary_ref: str, comparator: pd.Series | None = None,
               tabs_raw: dict | None = None) -> dict:
    """M1 (instruction) and M2 (race field removed): D for every label edit under M1, and group calibration at baseline
    for raw, M1 and M2. A mitigation that removes over-weighting but leaves NHB estimates below outcomes shows up as
    D near -(reference contrast) under M1 and an NHB/NHW O/E ratio above 1 under M2."""
    out = {"group_calibration": {}, "D_under_M1": {}}
    ev = set(draws.profiles)
    for m in models:
        gc = {}
        for label, sel in (("raw", (calls.edit == "baseline") & _variant(calls, "raw")),
                           ("M1", (calls.edit == "baseline") & _variant(calls, "M1")),
                           ("M2", (calls.edit == "M2_race_removed") & _variant(calls, "raw"))):
            c = calls[(calls.model == m) & sel & (calls.repeat == 0) & (~calls.annotated) & calls.valid].drop_duplicates("profile")
            c = c[c.profile.isin(ev)]
            if len(c) >= 50:
                gc[label] = group_calibration(c.set_index("profile")["p"], cohort)
        out["group_calibration"][m] = gc
    if comparator is not None:
        out["group_calibration"]["reference"] = {"raw": group_calibration(comparator.loc[sorted(ev)], cohort)}
    tabs_m1 = {m: contrast_table(calls, inst, m, primary_ref, variant="M1") for m in models}
    tabs_m1 = {m: t for m, t in tabs_m1.items() if len(t)}
    m1_edits = [e for e in FAMILY if e.startswith("L")] + ["L1c_race_nhb_to_nhw", "L5b_sex_female_to_male"]
    m1_edits = [e for e in m1_edits if tabs_m1 and any((t.edit == e).any() for t in tabs_m1.values())]
    if m1_edits:
        r = confirmatory_family(tabs_m1, refits, draws, calls, cohort, edits=m1_edits, pairs=False)
        out["D_under_M1"] = {e: {m: {k: v[k] for k in ("D_logit", "D_logit_ci", "D_logit_ci_simultaneous", "D_prob_pts",
                                                        "D_prob_pts_ci", "implied_or_model", "or_reference")}
                                  for m, v in r["cells"][e].items()} for e in r["cells"]}
        if tabs_raw:     # the remedy's effect: M1 minus as returned, same people, shared draws (d cancels: reference-free)
            out["M1_minus_raw"] = {}
            for e in m1_edits:
                for m in models:
                    a = tabs_m1.get(m); b = tabs_raw.get(m)
                    if a is None or b is None:
                        continue
                    a = a[a.edit == e].drop_duplicates("profile").set_index("profile")
                    b = b[b.edit == e].drop_duplicates("profile").set_index("profile")
                    common = sorted(set(a.index) & set(b.index) & set(int(x) for x in draws.pos))
                    if len(common) < 20:
                        continue
                    oa = _D_draws(a, common, e, refits, draws, None, None)
                    ob = _D_draws(b, common, e, refits, draws, None, None)
                    out["M1_minus_raw"].setdefault(e, {})[m] = {
                        "diff_logit": float(oa["point"]["D_logit"] - ob["point"]["D_logit"]),
                        "diff_logit_ci": _ci(oa["draws"]["D_logit"] - ob["draws"]["D_logit"]), "n": len(common)}
    return out


# ----------------------------------------------------------------------------- orchestration
MIN_USABLE, MIN_SUPPORTED = 0.80, 100   # family rules: usable share per system, supported records per edit


def run_all(calls, inst, cohort, refits: dict[str, Refits], split_profiles, B: int = 2000, B_desc: int = 500,
            comparator_preds: dict | None = None, family_rules: bool = True) -> dict:
    primary_ref, second_ref = "spline_logit", "lightgbm"
    models = ordered(set(calls.model))
    prim = [m for m in models if is_primary(m)]      # the confirmatory family and its sensitivity analyses
    R = refits[primary_ref].R
    draws = Draws(split_profiles, B, R)
    tabs = {m: contrast_table(calls, inst, m, primary_ref) for m in models}
    from .clients import system_tier
    out = {"meta": {"models": models, "primary_systems": prim, "tiers": {m: system_tier(m) for m in models}, "B": B, "B_descriptive": B_desc, "reference_refits": R, "crit_inflation": CRIT_INFLATION,
                    "n_eval_profiles": int(len(split_profiles)), "primary_reference": primary_ref, "second_reference": second_ref}}
    out["validity"] = {m: calls[(calls.model == m) & (calls.repeat == 0) & _variant(calls, "raw")].groupby("edit")["valid"].mean().round(4).to_dict() for m in models}
    out["providers_reported"] = {m: calls[calls.model == m]["provider"].value_counts().to_dict() for m in models}
    out["support"] = inst.groupby("edit").apply(lambda g: {"applicable": float(g.applicable.mean()),
                                                              "supported_given_applicable": float(g.supported[g.applicable].mean()) if g.applicable.any() else None}).to_dict()
    sup = inst[inst.applicable & inst.supported]
    c4 = sup[sup.edit == "C4_sbp_plus20"]
    per_mmhg = float(c4[f"d_logit_{primary_ref}"].mean()) / 20.0 if len(c4) else None
    # family rules, decided from data and parse rates alone: a system with < 80% usable outputs and an edit
    # with < 100 supported records are reported descriptively and left out of the simultaneous family
    base_ok = calls[(calls.repeat == 0) & _variant(calls, "raw") & (~calls.annotated)]
    usable = {m: float(base_ok[base_ok.model == m].valid.mean()) for m in models}
    n_sup = sup.groupby("edit").profile.nunique()
    fam_models = [m for m in prim if not family_rules or usable[m] >= MIN_USABLE]
    fam = [e for e in FAMILY if e in set(inst.edit) and (not family_rules or int(n_sup.get(e, 0)) >= MIN_SUPPORTED)]
    out["meta"]["family_rules"] = {"applied": family_rules, "min_usable_share": MIN_USABLE, "min_supported_records": MIN_SUPPORTED,
                                   "usable_share": usable, "supported_records": {e: int(n_sup.get(e, 0)) for e in FAMILY},
                                   "systems_in_family": fam_models, "edits_in_family": fam}
    out["confirmatory"] = confirmatory_family({m: tabs[m] for m in fam_models}, refits[primary_ref], draws, calls, cohort,
                                              edits=fam, per_mmhg=per_mmhg, keep_draws=True)
    out["confirmatory"]["reference_log_or_per_mmHg_sbp"] = per_mmhg
    # model-independent reference odds ratio per edit: every supported instance, whichever model answered
    out["confirmatory"]["reference_or_all_supported"] = {e: float(np.exp(sup.loc[sup.edit == e, f"d_logit_{primary_ref}"].mean()))
                                                         for e in fam if (sup.edit == e).any()}
    out["confirmatory"]["discrepancy_by_baseline_risk"] = {e: discrepancy_by_risk(tabs, edit=e, B=B_desc) for e in fam}
    tabs2 = {m: contrast_table(calls, inst, m, second_ref) for m in prim}
    if family_rules and second_ref not in refits:
        raise ValueError(f"the analysis needs {second_ref} refits")
    ref2 = refits.get(second_ref) or Refits.from_inst(inst, second_ref)
    out["confirmatory_second_reference"] = confirmatory_family(tabs2, ref2, Draws(split_profiles, min(B, 500), ref2.R, seed=11),
                                                               calls, cohort, edits=fam, pairs=False)
    if "survival_logit" in refits and "d_logit_survival_logit" in inst.columns:
        tabs3 = {m: contrast_table(calls, inst, m, "survival_logit") for m in prim}
        out["confirmatory_survival_reference"] = confirmatory_family(
            tabs3, refits["survival_logit"], Draws(split_profiles, min(B, 500), refits["survival_logit"].R, seed=23),
            calls, cohort, edits=fam, pairs=False)
    out["prior_sensitivity"] = {}
    for sens in ("spline_logit_prior", "spline_logit_shared"):
        if f"d_logit_{sens}" in inst.columns:
            ts = {m: contrast_table(calls, inst, m, sens) for m in prim}
            rs = confirmatory_family(ts, Refits.from_inst(inst, sens), Draws(split_profiles, min(B, 500), 1, seed=19), calls, cohort,
                                     edits=fam, pairs=False)
            out["prior_sensitivity"][sens] = {"note": "sampling uncertainty only",
                                              **{e: {m: {k: v[k] for k in ("D_logit", "D_logit_ci", "mean_reference_contrast_pts")}
                                                     for m, v in rs["cells"][e].items()} for e in rs["cells"]}}
    # survey-weighted reference (sensitivity; the cohort is analysed unweighted and makes no national claim)
    out["reference_sensitivity"] = {}
    for sens in ("spline_logit_weighted",):
        if f"d_logit_{sens}" in inst.columns and (inst[f"d_logit_{sens}"] != 0).any():
            ts = {m: contrast_table(calls, inst, m, sens) for m in prim}
            rw = refits.get(sens) or Refits.from_inst(inst, sens)      # paired refits when present (same resamples)
            rs = confirmatory_family(ts, rw, Draws(split_profiles, min(B, 500), rw.R, seed=29), calls, cohort, edits=fam,
                                     pairs=False)
            shift_ci = {}
            rp = refits.get(primary_ref)
            if sens in refits and rp is not None and rp.R == rw.R and rw.R > 1:
                for e in fam:
                    rows = [rw.index[k] for k in zip(sup.profile[sup.edit == e].astype(int), sup.edit[sup.edit == e].astype(str))
                            if k in rw.index and k in rp.index]
                    rows_p = [rp.index[k] for k in zip(sup.profile[sup.edit == e].astype(int), sup.edit[sup.edit == e].astype(str))
                              if k in rw.index and k in rp.index]
                    if rows:
                        shift_ci[e] = _ci(rw.d_logit[rows].mean(0) - rp.d_logit[rows_p].mean(0))
            prim_cells = out["confirmatory"]["cells"]
            same = [np.sign(v["D_logit"]) == np.sign(prim_cells[e][m]["D_logit"]) for e in rs["cells"] for m, v in rs["cells"][e].items()
                    if m in prim_cells.get(e, {})]
            out["reference_sensitivity"][sens] = {
                "note": "survey-weighted primary specification; intervals carry reference refits when present",
                "reference_shift_logit": {e: float(sup.loc[sup.edit == e, f"d_logit_{sens}"].mean()
                                                   - sup.loc[sup.edit == e, f"d_logit_{primary_ref}"].mean())
                                          for e in fam if (sup.edit == e).any()},
                "reference_shift_logit_ci": shift_ci or None,       # paired refits; None when only point fits exist
                "share_cells_same_sign_as_primary": float(np.mean(same)) if same else None,
                "cells": {e: {m: {k: v.get(k) for k in ("D_logit", "D_logit_ci_wald", "D_logit_ci")} for m, v in rs["cells"][e].items()}
                          for e in rs["cells"]},
                "clinical_slope": {m: {k: v.get(k) for k in ("beta", "ci_wald")} for m, v in rs.get("clinical_slope", {}).items()}}
    # PhenoAge (Levine 2018; other authors, NHANES III, published coefficients) as an external reference for the
    # edits to fields it contains. Fixed coefficients: sampling uncertainty only. Triangulation, not a substitute.
    ph_edits = [e for e in PHENOAGE_EDITS if e in set(sup.edit)]
    if "d_logit_phenoage" in inst.columns and ph_edits:
        ts = {m: contrast_table(calls, inst, m, "phenoage") for m in prim}
        rs = confirmatory_family(ts, Refits.from_inst(inst, "phenoage"), Draws(split_profiles, min(B, 500), 1, seed=31), calls,
                                 cohort, edits=ph_edits, pairs=False)
        rng = np.random.default_rng(37)
        agree = {}
        rp = refits.get(primary_ref)
        for e in ph_edits:
            t = sup[sup.edit == e]
            a, b = t[f"d_logit_{primary_ref}"].to_numpy(), t["d_logit_phenoage"].to_numpy()
            keys = list(zip(t.profile.astype(int), t.edit.astype(str)))
            if rp is not None and rp.R > 1 and all(k in rp.index for k in keys):
                A_ = rp.d_logit[[rp.index[k] for k in keys]]           # primary refits carry its uncertainty (PhenoAge is fixed)
                bs = [float((A_[i, j % rp.R] - b[i]).mean()) for j, i in enumerate(rng.integers(0, len(a), len(a)) for _ in range(500))]
            else:
                bs = [float((a[i] - b[i]).mean()) for i in (rng.integers(0, len(a), len(a)) for _ in range(500))]
            agree[e] = {"primary_mean_logit": float(a.mean()), "phenoage_mean_logit": float(b.mean()),
                        "primary_minus_phenoage": float((a - b).mean()), "primary_minus_phenoage_ci": _ci(bs), "n": int(len(a))}
        prim_cells = out["confirmatory"]["cells"]
        out["phenoage_reference"] = {
            "note": "external published reference for fields PhenoAge contains; intervals carry the primary's refits and sampling",
            "reference_agreement": agree,
            "cells": {e: {m: {**{k: v.get(k) for k in ("D_logit", "D_logit_ci_wald")},
                              "D_logit_primary": prim_cells.get(e, {}).get(m, {}).get("D_logit")}
                          for m, v in rs["cells"][e].items()} for e in rs["cells"]}}
    out["edit_secondary"] = {}      # other directions and step sizes: max-t across models within each edit only
    for e in sorted(set(inst[inst.klass.isin(["clinical", "label"])].edit) - set(fam)):
        r = primary_D({m: tabs[m] for m in prim}, refits[primary_ref], Draws(split_profiles, min(B, 500), R, seed=13), calls,
                      cohort, edit=e)
        out["edit_secondary"][e] = {"n_profiles": r.get("n_profiles"),
                                    **{m: {k: r["models"][m][k] for k in ("D_prob_pts", "D_prob_pts_ci", "D_logit", "D_logit_ci")}
                                       for m in r.get("models", {})}}
    ddesc = Draws(split_profiles, B_desc, R, seed=17)
    out["classes"] = {m: class_beta_draws(tabs[m], refits[primary_ref], ddesc) for m in models}
    out["drift"] = {m: drift(calls, tabs[m], m, ddesc) for m in models}
    out["output_granularity"] = {m: output_granularity(calls, tabs[m], m) for m in models}
    from . import headline as H          # secondaries that make the family clinically legible
    fam_res = out["confirmatory"]
    out["headline_metrics"] = {
        "equivalents": H.equivalents(fam_res, inst, refits[primary_ref], draws, primary_ref,
                                     {m: out["drift"][m] for m in prim if m in out["drift"]}),   # headline level: primary only
        "weights": H.weights(fam_res),
        "consensus": H.consensus(fam_res),
        "reclassification": H.reclassification({m: tabs[m] for m in prim}, fam),
        "heterogeneity_exploratory": H.heterogeneity({m: tabs[m] for m in prim}, cohort, fam, refits[primary_ref], draws),
        "dose_response": H.dose_response({m: tabs[m] for m in prim}, refits[primary_ref], draws),
        "inversions": H.inversions(fam_res)}
    fam_res.pop("_draws", None)
    out["direction"] = {m: direction_agreement(tabs[m], refits[primary_ref]) for m in models}
    y = cohort.set_index(cohort.SEQN.astype(int))["death_10y"]
    perf = {}
    for m in models:
        b = calls[(calls.model == m) & (calls.edit == "baseline") & (calls.repeat == 0) & (~calls.annotated) & calls.valid & _variant(calls, "raw")].drop_duplicates("profile")
        b = b[b.profile.isin(split_profiles)]
        perf[m] = performance(b.p.to_numpy(), y.loc[b.profile].to_numpy())
    for name, pr in (comparator_preds or {}).items():
        perf[name] = performance(pr.loc[split_profiles].to_numpy(), y.loc[split_profiles].to_numpy())
    out["baseline_performance"] = perf
    # full-set systems only: a subset-arm system (reasoning arm) would shrink the records every system scored to the
    # 200-record subset
    base_preds = baseline_predictions(calls, models, split_profiles, comparator_preds)
    out["prediction"] = prediction_block(base_preds, y.reindex(split_profiles).dropna(),
                                         focal_versus=[m for m in prim if m != "jev"])
    ann = {}
    for m in models:
        ta = contrast_table(calls, inst, m, primary_ref, annotated=True)
        if len(ta):
            tr = tabs[m][tabs[m].set_index(["profile", "edit"]).index.isin(ta.set_index(["profile", "edit"]).index)]
            ann[m] = {"raw": class_beta_draws(tr, refits[primary_ref], ddesc).get("clinical", {}),
                      "annotated": class_beta_draws(ta, refits[primary_ref], ddesc).get("clinical", {})}
    out["annotation_remedy"] = ann
    if "variant" in calls and (calls["variant"] == "M1").any() or (calls.edit == "M2_race_removed").any():
        out["m2_shift"] = {m: m2_shift(calls, cohort, m, split_profiles) for m in models}
    out["mitigation"] = mitigation(calls, inst, cohort, refits[primary_ref], Draws(split_profiles, min(B, 500), R, seed=23),
                                       prim, primary_ref, (comparator_preds or {}).get("reference_spline_logit"), tabs_raw=tabs)
    if "jev" in models:
        out["jev_resolution"] = jev_resolution(calls, tabs["jev"])
        out["jev_complement"] = jev_complement(calls)
        if "jev_bundle" in set(calls.model):
            out["jev_bundle"] = jev_bundle(calls)
        if "jev_bands" in set(calls.model):
            tb = contrast_table(calls, inst, "jev_bands", primary_ref)
            tn = tabs["jev"][tabs["jev"].set_index(["profile", "edit"]).index.isin(tb.set_index(["profile", "edit"]).index)]
            out["jev_bands_vs_noul"] = {"bands": class_beta_draws(tb, refits[primary_ref], ddesc),
                                        "noul_same_instances": class_beta_draws(tn, refits[primary_ref], ddesc),
                                        "L1_D_pts_bands": float((tb[tb.edit == PRIMARY_EDIT].z_prob - tb[tb.edit == PRIMARY_EDIT].d_prob).mean() * 100) if (tb.edit == PRIMARY_EDIT).any() else None,
                                        "L1_D_pts_noul": float((tn[tn.edit == PRIMARY_EDIT].z_prob - tn[tn.edit == PRIMARY_EDIT].d_prob).mean() * 100) if (tn.edit == PRIMARY_EDIT).any() else None}
    # per-call error (every system, descriptive) and the secondary arms: own bootstrap intervals, outside the max-t family
    out["per_call_error"] = {m: per_call_error(tabs[m], refits[primary_ref], ddesc, fam) for m in models}
    out["secondary_arms"] = secondary_arms(calls, cohort, refits[primary_ref], tabs, models, fam, split_profiles, B_desc)
    out["tier_contrasts"] = tier_contrasts(tabs, refits[primary_ref], ddesc, fam, models)
    out["reasoning_arm"] = reasoning_arm(tabs, refits[primary_ref], split_profiles, fam, B_desc, R)
    from .clients import MODELS as _M
    out["cost_latency"] = cost_latency(calls, _M)
    return out


# ----------------------------------------------------------------------------- secondary arms
_SIMULTANEOUS_KEYS = ("_ci_simultaneous", "rules_out_abs_D")


def _marginal_only(res: dict) -> dict:
    """A secondary arm is outside the max-t family: keep D with its own marginal intervals (percentile and Wald) and
    drop every simultaneous quantity, so nothing from an arm can be read as confirmatory."""
    for k in ("critical_value", "critical_value_raw", "crit_inflation", "simultaneous_excludes_zero", "n_tests"):
        res.pop(k, None)
    for e, cells in res.get("cells", {}).items():
        for m, r in cells.items():
            for k in [k for k in r if k.endswith(_SIMULTANEOUS_KEYS)]:
                r.pop(k)
    for m, r in res.get("clinical_slope", {}).items():
        for k in ("ci_simultaneous", "excludes_one", "rules_out_abs_beta_minus_1"):
            r.pop(k, None)
    res["note"] = "secondary arm: own bootstrap intervals (percentile `_ci`, Wald `_ci_wald`), outside the max-t family"
    return res


def per_call_error(tab: pd.DataFrame, refits: Refits, draws: Draws, edits: list[str]) -> dict:
    """Accuracy of changes, call by call: mean |z - d| on the log-odds scale over the family's supported record-edits,
    with a bootstrap interval (profile resamples paired with reference refits, as for D). Descriptive."""
    t = tab[tab.edit.isin(edits)]
    rows = refits.rows(t)
    t, rows = t[rows >= 0], rows[rows >= 0]
    t = t[t.profile.isin(draws.pos.keys())]
    rows = refits.rows(t)
    if len(t) < 20:
        return {"n_instances": int(len(t))}
    err = np.abs(t.z_logit.to_numpy() - t.d_logit.to_numpy())
    W = draws.w(t.profile)
    E = np.abs(t.z_logit.to_numpy()[:, None] - refits.d_logit[rows][:, draws.r])     # n x B
    dr = (W * E.T).sum(1) / W.sum(1)
    # refit noise inflates |z - d| in every draw, so the draws sit above the point: the interval is Wald-form on the
    # bootstrap SE (point +/- 1.96 SE); the draws' mean is reported beside it
    pt, se = float(err.mean()), float(np.std(dr, ddof=1))
    out = {"mae_logit": pt, "mae_logit_ci": [pt - 1.96 * se, pt + 1.96 * se], "mae_logit_draw_mean": float(dr.mean()),
           "n_instances": int(len(t)), "n_profiles": int(t.profile.nunique()), "by_class": {}}
    for k, g in t.groupby("klass"):
        out["by_class"][str(k)] = {"mae_logit": float(np.abs(g.z_logit - g.d_logit).mean()), "n_instances": int(len(g))}
    return out


def secondary_arms(calls, cohort, refits: Refits, tabs: dict, models: list[str], fam: list[str], split_profiles,
                   B_desc: int) -> dict:
    """D for every edit of the family and the clinical slope, per system of each secondary arm (free, pair, reasoning),
    each arm on its own records with its own bootstrap draws; marginal intervals only."""
    from .clients import system_tier
    out = {}
    for arm in SECONDARY_ARMS:
        ms = [m for m in models if system_tier(m) == arm]
        if not ms:
            continue
        prof = sorted(set(calls.loc[calls.model.isin(ms), "profile"].astype(int)) & set(int(p) for p in split_profiles))
        if len(prof) < 20:
            continue
        d = Draws(prof, B_desc, refits.R, seed=41)
        res = confirmatory_family({m: tabs[m][tabs[m].profile.isin(prof)] for m in ms}, refits, d, calls, cohort,
                                  edits=fam, pairs=False)
        out[arm] = {"n_profiles": len(prof), **_marginal_only(res)}
    return out


def _paired_D(a: pd.DataFrame, b: pd.DataFrame, edit: str, refits: Refits, draws: Draws, min_profiles: int = 20) -> dict | None:
    """D(a) - D(b) for one edit on the records both answered (raw scale), same draws for both."""
    sa = a[a.edit == edit].drop_duplicates("profile").set_index("profile")
    sb = b[b.edit == edit].drop_duplicates("profile").set_index("profile")
    common = sorted(set(sa.index) & set(sb.index) & set(draws.pos))
    if len(common) < min_profiles:
        return None
    da = _D_draws(sa, common, edit, refits, draws, None, None)
    db = _D_draws(sb, common, edit, refits, draws, None, None)
    return {"diff": float(da["point"]["D_logit"] - db["point"]["D_logit"]),
            "ci": _ci(da["draws"]["D_logit"] - db["draws"]["D_logit"]), "n": len(common),
            "_za": da["draws"]["implied_log_or"], "_zb": db["draws"]["implied_log_or"], "_d": da["draws"]["reference_log_or"],
            "_pa": float(da["point"]["implied_log_or_model"]), "_pb": float(db["point"]["implied_log_or_model"]),
            "_pd": float(da["point"]["log_or_reference"])}


def _pair_block(a: pd.DataFrame, b: pd.DataFrame, refits: Refits, draws: Draws, fam: list[str]) -> dict:
    """Per-edit paired D difference, clinical-slope difference and per-call error difference (a minus b), each on
    common records, all from one set of draws. Descriptive."""
    cells, za, zb, dd, pa, pb, pd_ = {}, [], [], [], [], [], []
    for e in fam:
        r = _paired_D(a, b, e, refits, draws)
        if r is None:
            continue
        cells[e] = {k: v for k, v in r.items() if not k.startswith("_")}
        if e.startswith("C"):
            za.append(r["_za"]); zb.append(r["_zb"]); dd.append(r["_d"]); pa.append(r["_pa"]); pb.append(r["_pb"]); pd_.append(r["_pd"])
    out = {"edits": cells}
    if len(dd) >= 3:
        Za, Zb, Dd = np.column_stack(za), np.column_stack(zb), np.column_stack(dd)
        pa, pb, pd_ = np.array(pa), np.array(pb), np.array(pd_)
        sa, sb = (pa * pd_).sum() / (pd_ ** 2).sum(), (pb * pd_).sum() / (pd_ ** 2).sum()
        out["clinical_slope"] = {"a": float(sa), "b": float(sb), "diff": float(sa - sb),
                                 "ci": _ci((Za * Dd).sum(1) / (Dd * Dd).sum(1) - (Zb * Dd).sum(1) / (Dd * Dd).sum(1))}
    ka = a[a.edit.isin(fam)].drop_duplicates(["profile", "edit"]).set_index(["profile", "edit"])
    kb = b[b.edit.isin(fam)].drop_duplicates(["profile", "edit"]).set_index(["profile", "edit"])
    common = ka.index.intersection(kb.index)
    common = common[[int(p) in draws.pos for p, _ in common]]
    if len(common) >= 20:
        ta, tb = ka.loc[common].reset_index(), kb.loc[common].reset_index()
        rows = refits.rows(ta)
        ok = rows >= 0
        ta, tb, rows = ta[ok], tb[ok], rows[ok]
        Dl = refits.d_logit[rows][:, draws.r]
        W = draws.w(ta.profile)
        Ea = np.abs(ta.z_logit.to_numpy()[:, None] - Dl); Eb = np.abs(tb.z_logit.to_numpy()[:, None] - Dl)
        dpt = float(np.abs(ta.z_logit - ta.d_logit).mean() - np.abs(tb.z_logit - tb.d_logit).mean())
        dse = float(np.std((W * (Ea - Eb).T).sum(1) / W.sum(1), ddof=1))     # Wald form, as for per_call_error
        out["per_call_error_diff"] = {"diff": dpt, "ci": [dpt - 1.96 * dse, dpt + 1.96 * dse], "n_instances": int(len(ta))}
    return out


def tier_contrasts(tabs: dict, refits: Refits, draws: Draws, fam: list[str], models: list[str]) -> dict:
    """Descriptive tier contrasts: each free-tier system minus the same developer's primary system, and each
    medical-training pair (config pair_with), on common records."""
    from .clients import MODELS, system_tier
    F = MODELS["families"]
    prim_by_dev = {F[m].get("developer"): m for m in models if system_tier(m) == "primary" and m in F}
    out = {}
    for m in models:
        t = system_tier(m)
        if t == "free":
            p = prim_by_dev.get(F[m].get("developer"))
            if p in tabs:
                out[f"{m} - {p}"] = _pair_block(tabs[m], tabs[p], refits, draws, fam)
        elif t == "pair" and F[m].get("pair_with") in tabs and m < F[m]["pair_with"]:
            out[f"{m} - {F[m]['pair_with']}"] = _pair_block(tabs[m], tabs[F[m]["pair_with"]], refits, draws, fam)
    return out


def reasoning_arm(tabs: dict, refits: Refits, split_profiles, fam: list[str], B_desc: int, R: int) -> dict:
    """Reasoning arm: per primary chatbot, effort high minus effort low on the same subset records (paired), for
    every family edit, the clinical slope and the per-call error. A family whose default is its high effort
    (config arms.reasoning.reuse_default) uses its primary answers on the subset records as 'high'."""
    from .clients import MODELS
    reuse = (MODELS["arms"]["reasoning"].get("reuse_default") or {})
    out = {}
    fams = sorted({m.split("@")[0] for m in tabs if "@" in m}, key=lambda f: LLM_ORDER.index(f) if f in LLM_ORDER else 99)
    for f in fams:
        low = tabs.get(f"{f}@low")
        high = tabs.get(f"{f}@high")
        src = "effort high"
        if high is None and reuse.get(f) == "high" and f in tabs and low is not None:
            high, src = tabs[f][tabs[f].profile.isin(set(low.profile))], "primary answers (default is high)"
        if low is None or high is None:
            continue
        prof = sorted((set(low.profile) | set(high.profile)) & set(int(p) for p in split_profiles))
        if len(prof) < 20:
            continue
        d = Draws(prof, B_desc, R, seed=43)
        out[f] = {"high_source": src, "n_profiles": len(prof), "high_minus_low": _pair_block(high, low, refits, d, fam)}
    return out


def framing_sensitivity(calls_raw: pd.DataFrame, calls_f1: pd.DataFrame, inst: pd.DataFrame, refits: Refits, subset,
                  fam: list[str] | None = None, B: int = 1000, ref: str = "spline_logit") -> dict:
    """Framing sensitivity analysis: per primary chatbot, the system message without its study sentence (variant F1)
    against the full system message (raw) on the same subset records, paired and from one set of draws. Validity under
    each; the baseline shift (mean F1 minus raw, log-odds and points); per confirmatory edit D under each and the
    difference (the reference cancels); the clinical-slope and per-call-error differences (_pair_block). Descriptive."""
    fam = list(fam or FAMILY)
    sub = {int(p) for p in subset}
    models = [m for m in ordered(set(calls_f1.model)) if m != "jev" and "@" not in m]
    out = {"n_subset": len(sub), "edits": fam, "reference": ref, "systems": {}}
    for m in models:
        a = calls_f1[(calls_f1.model == m) & calls_f1.profile.isin(sub) & _variant(calls_f1, "F1") & (calls_f1.repeat == 0)]
        b = calls_raw[(calls_raw.model == m) & calls_raw.profile.isin(sub) & _variant(calls_raw, "raw") & (calls_raw.repeat == 0)
                      & (~calls_raw.annotated) & calls_raw.edit.isin(["baseline"] + fam)]
        if a.empty or b.empty:
            continue
        ta = contrast_table(calls_f1[calls_f1.profile.isin(sub)], inst, m, ref, variant="F1")
        tb = contrast_table(calls_raw[calls_raw.profile.isin(sub)], inst, m, ref)
        prof = sorted(sub & (set(ta.profile) | set(tb.profile) | set(a.profile)))
        if len(prof) < 20:
            continue
        d = Draws(prof, B, refits.R, seed=47)
        r = {"validity": {"F1": float(a.valid.mean()), "raw": float(b.valid.mean()),
                          "n_F1": int(len(a)), "n_raw": int(len(b))}}
        pa = a[(a.edit == "baseline") & a.valid].drop_duplicates("profile").set_index("profile")["p"]
        pb = b[(b.edit == "baseline") & b.valid].drop_duplicates("profile").set_index("profile")["p"]
        common = sorted(set(pa.index) & set(pb.index) & set(d.pos))
        if len(common) >= 20:
            dl = logit(pa.loc[common].to_numpy()) - logit(pb.loc[common].to_numpy())
            dp = (pa.loc[common].to_numpy() - pb.loc[common].to_numpy()) * 100
            W = d.w(common); Ws = W.sum(1)
            r["baseline_shift"] = {"n": len(common), "mean_logit": float(dl.mean()), "mean_logit_ci": _ci((W * dl).sum(1) / Ws),
                                   "mean_prob_pts": float(dp.mean()), "mean_prob_pts_ci": _ci((W * dp).sum(1) / Ws),
                                   "median_abs_logit": float(np.median(np.abs(dl))),
                                   "mean_p_raw": float(pb.loc[common].mean()), "mean_p_F1": float(pa.loc[common].mean())}
        cells = {}
        for e in fam:
            x = _paired_D(ta, tb, e, refits, d)
            if x is None:
                continue
            cells[e] = {"D_logit_F1": x["_pa"] - x["_pd"], "D_logit_raw": x["_pb"] - x["_pd"],
                        "diff": x["diff"], "ci": x["ci"], "n": x["n"]}
        r["edits"] = cells
        blk = _pair_block(ta, tb, refits, d, fam)
        for k in ("clinical_slope", "per_call_error_diff"):
            if k in blk:
                r[k] = blk[k]
        if cells:
            diffs = np.array([c["diff"] for c in cells.values()])
            r["summary"] = {"n_edits": len(cells), "max_abs_diff": float(np.abs(diffs).max()),
                            "n_ci_excludes_zero": int(sum(c["ci"][0] is not None and (c["ci"][0] > 0 or c["ci"][1] < 0)
                                                          for c in cells.values()))}
        out["systems"][m] = r
    return out


def run_wide(calls: pd.DataFrame, cohort: pd.DataFrame, oof: pd.DataFrame, B: int = 1000) -> dict:
    """Wide baseline run (every complete-case record of the fitting and evaluation splits): prediction with
    cross-fitted references, group calibration as returned and with the race field removed, and the within-record M2
    shift."""
    models = full_set_systems(ordered(set(calls.model)))      # the wide plan has no subset arm; kept explicit
    y = cohort.set_index(cohort.SEQN.astype(int))["death_10y"]
    o = oof.set_index("SEQN")
    profiles = sorted(set(o.index))
    preds = {}
    for m in models:
        b = calls[(calls.model == m) & (calls.edit == "baseline") & calls.valid & _variant(calls, "raw")]
        preds[m] = b.drop_duplicates("profile").set_index("profile")["p"].reindex(profiles)
    for col, name in (("p_spline_logit", "reference_spline_logit"), ("p_lightgbm", "reference_lightgbm"),
                      ("p_age_sex", "reference_age_sex"), ("p_phenoage_10y", "phenoage_10y")):
        if col in o:
            preds[name] = o[col].reindex(profiles)
    out = {"meta": {"models": models, "n_records": len(profiles)}, "prediction": prediction_block(preds, y.reindex(profiles).dropna(), B=B)}
    gc = {}
    for m in models:
        gc[m] = {}
        for label, edit in (("raw", "baseline"), ("M2", "M2_race_removed")):
            c = calls[(calls.model == m) & (calls.edit == edit) & calls.valid & _variant(calls, "raw")].drop_duplicates("profile")
            c = c[c.profile.isin(profiles)]
            if len(c) >= 50:
                gc[m][label] = group_calibration(c.set_index("profile")["p"], cohort, B=B)
    if "p_spline_logit" in o:
        gc["reference"] = {"raw": group_calibration(o["p_spline_logit"].reindex(profiles).dropna(), cohort, B=B)}
    out["group_calibration"] = gc
    out["m2_shift"] = {m: m2_shift(calls, cohort, m, profiles, B=B) for m in models}
    return out

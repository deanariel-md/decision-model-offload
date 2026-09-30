"""Synthetic cohort and simulated model behaviours for dry runs of the full pipeline. Nothing here is NHANES and
nothing here is a study result; it exists so that every script and estimator can be exercised end to end, and so that
known behaviours (compression, label over-weighting, format sensitivity, a 0.01 output grid) are recovered by the
analysis without any real call."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import perturb as P

RACES = ["Non-Hispanic White", "Non-Hispanic Black", "Mexican American", "Other Hispanic", "Other race including multiracial"]
EDU = ["less than 9th grade", "9th to 11th grade", "high school graduate or GED", "some college", "college graduate or above"]
COND = ["diabetes", "chf", "chd", "mi", "stroke", "emphysema", "chronic_bronchitis", "cancer"]


def true_logit(rec, race_nhb=0.25, income=-0.10, lab_weight=1.0, age_weight=1.0) -> float:
    g = lambda k: rec[k]
    lab = (0.3 * (g("hba1c_pct") - 5.5) + 0.6 * np.log(g("creatinine_mg_dl")) - 0.8 * (g("albumin_g_dl") - 4.2)
           + 0.15 * np.log(g("crp_mg_dl") + 0.05) + 0.01 * (g("sbp_mmhg") - 130))
    return float(-3.2 + age_weight * 0.06 * (g("age") - 40) + 0.4 * (g("smoking") == "current") + lab_weight * lab
                 + 0.5 * (g("diabetes") == "yes") + 0.6 * (g("chf") == "yes")
                 + race_nhb * (g("race_ethnicity") == "Non-Hispanic Black") + income * (g("income_poverty_ratio") - 2))


def make_synthetic_prior(n=8000, seed=0, intercept_shift=0.35, race_nhb=0.35) -> pd.DataFrame:
    """Synthetic NHANES III-like prior cohort: higher baseline mortality (era), a drifted race coefficient, and the
    variables NHANES III lacks set to missing."""
    df = make_synthetic_cohort(n=n, seed=seed, n_pilot=0, n_eval=0)
    lo = np.array([true_logit(r, race_nhb=race_nhb) for r in df.to_dict("records")]) + intercept_shift
    r = np.random.default_rng(seed + 7)
    df["true_logit"] = lo
    df["death_10y"] = (r.random(len(df)) < 1 / (1 + np.exp(-lo))).astype(int)
    df["SEQN"] = df["SEQN"] + 10_000_000
    # follow-up for the survival reference: deaths within ten years uniform over months 1-120; survivors die in years
    # 11-20 with probability tied to their ten-year risk; everyone else is censored at 300 months (NHANES III to 2019)
    t10 = r.integers(1, 121, len(df)); late = r.random(len(df)) < 1.2 / (1 + np.exp(-lo))
    df["mortstat"] = (df.death_10y.eq(1) | late).astype(int)
    df["permth_exm"] = np.where(df.death_10y.eq(1), t10, np.where(late, r.integers(121, 241, len(df)), 300))
    df["cycle"] = "1988-1994"; df["split"] = "prior"; df["era_nhanes3"] = 1
    df["chd"] = np.nan; df["rx_count"] = np.nan
    return df


def make_synthetic_cohort(n=6000, seed=0, n_pilot=200, n_eval=1000) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "SEQN": np.arange(100000, 100000 + n), "cycle": r.choice(["1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008"], n),
        "age": r.integers(40, 80, n), "sex": r.choice(["male", "female"], n),
        "race_ethnicity": r.choice(RACES, n, p=[0.5, 0.2, 0.15, 0.05, 0.1]), "education": r.choice(EDU, n),
        "insured": r.choice(["yes", "no"], n, p=[0.85, 0.15]), "income_poverty_ratio": np.round(np.clip(r.gamma(2, 1.2, n), 0, 5), 2),
        "bmi": np.round(r.normal(29, 6, n), 1), "waist_cm": np.round(r.normal(100, 15, n), 1),
        "sbp_mmhg": np.round(r.normal(130, 18, n)), "dbp_mmhg": np.round(r.normal(75, 11, n)),
        "smoking": r.choice(["never", "former", "current"], n, p=[0.5, 0.3, 0.2]),
        "self_rated_health": r.choice(["excellent", "very good", "good", "fair", "poor"], n),
        "rx_count": r.poisson(2.5, n), "rx_names": "", "albumin_g_dl": np.round(r.normal(4.2, 0.3, n), 1),
        "creatinine_mg_dl": np.round(np.exp(r.normal(-0.1, 0.25, n)), 2), "glucose_mg_dl": np.round(np.exp(r.normal(4.6, 0.2, n))),
        "crp_mg_dl": np.round(np.exp(r.normal(-1.2, 1.0, n)), 2), "lymphocyte_pct": np.round(r.normal(30, 8, n), 1),
        "mcv_fl": np.round(r.normal(90, 5, n), 1), "rdw_pct": np.round(r.normal(13, 1.2, n), 1),
        "alk_phos_u_l": np.round(r.normal(75, 22, n)), "wbc_10e3_ul": np.round(r.normal(7, 2, n), 1),
        "hba1c_pct": np.round(np.exp(r.normal(1.72, 0.12, n)), 1), "total_chol_mg_dl": np.round(r.normal(200, 40, n)),
        "hdl_mg_dl": np.round(r.normal(52, 15, n)), "uacr_mg_g": np.round(np.exp(r.normal(2.0, 1.0, n)), 1),
    })
    for c in COND:
        df[c] = r.choice(["yes", "no"], n, p=[0.12, 0.88])
    df["true_logit"] = [true_logit(x) for x in df.to_dict("records")]
    df["death_10y"] = (r.random(n) < 1 / (1 + np.exp(-df.true_logit))).astype(int)
    rw = np.random.default_rng(seed + 202)   # separate stream: existing draws are unchanged
    oversampled = df.race_ethnicity.isin(["Non-Hispanic Black", "Mexican American"]).to_numpy() | (df.age.to_numpy() >= 60)
    df["wt_mec_10y"] = np.round(np.exp(rw.normal(np.where(oversampled, 8.6, 9.6), 0.6, n)), 1)   # NHANES-like MEC weights
    df["split"] = "fit"
    idx = r.permutation(n)
    df.loc[df.index[idx[:n_pilot]], "split"] = "pilot"
    df.loc[df.index[idx[n_pilot:n_pilot + n_eval]], "split"] = "eval"
    return df


def _hazard_logit(df: pd.DataFrame, k: np.ndarray | int, race_late: float = 1.0, race_nhb: float = 0.25) -> np.ndarray:
    """Synthetic yearly hazard: logit h_k = true_logit(x) - 2.4 + 0.02 k - 0.05 age_z t_z, with the race term scaled by
    race_late after year ten (1.0 = constant effect, the survival reference's assumption)."""
    base = np.array([true_logit(r, race_nhb=0.0) for r in df.to_dict("records")])
    nhb = (df.race_ethnicity == "Non-Hispanic Black").to_numpy()
    k = np.asarray(k)
    race = race_nhb * nhb * np.where(k >= 10, race_late, 1.0)
    age_z, t_z = (df.age.to_numpy() - 60) / 10, (k - 9.5) / 10
    return base - 2.4 + 0.02 * k - 0.05 * age_z * t_z + race


def true_risk_10y(df: pd.DataFrame, race_late: float = 1.0) -> np.ndarray:
    log_s = np.zeros(len(df))
    for k in range(10):
        log_s += np.log1p(-1 / (1 + np.exp(-_hazard_logit(df, k, race_late))))
    return 1 - np.exp(log_s)


def make_synthetic_survival_cohort(n=14000, seed=0, n_pilot=0, n_eval=1000, race_late: float = 1.0) -> pd.DataFrame:
    """Synthetic cohort with yearly discrete hazards, examination years 1999-2008 and administrative censoring at the
    end of 2019 (follow-up 131-251 months), so death_10y is always determined and later deaths inform a survival fit."""
    df = make_synthetic_cohort(n=n, seed=seed, n_pilot=n_pilot, n_eval=n_eval)
    r = np.random.default_rng(seed + 101)
    start = df.cycle.str[:4].astype(int).to_numpy()
    fu_max = np.floor((2020.0 - (start + r.random(n) * 2)) * 12)
    death = np.full(n, np.inf)
    for k in range(21):
        h = 1 / (1 + np.exp(-_hazard_logit(df, k, race_late)))
        new = np.isinf(death) & (r.random(n) < h)
        death[new] = k * 12 + np.ceil(r.random(new.sum()) * 12)
    df["mortstat"] = (death <= fu_max).astype(int)
    df["permth_exm"] = np.minimum(death, fu_max)
    df["death_10y"] = ((df.mortstat == 1) & (df.permth_exm <= 120)).astype(int)
    df["true_logit"] = np.log(true_risk_10y(df, race_late) / (1 - true_risk_10y(df, race_late)))
    return df


# behaviour of each simulated model: (log-odds slope, race over-weight, income weight, lab weight, noise sd,
#                                     format sd, prose shift, output grid, annotated lab weight)
BEHAVIOURS = {
    "jev":      dict(slope=0.8, race=0.25, income=-0.10, lab=0.4, noise=0.00, fmt=0.05, prose=0.0, grid=0.01, lab_ann=1.0),
    "claude":   dict(slope=1.0, race=0.85, income=-0.10, lab=1.0, noise=0.15, fmt=0.10, prose=0.0, grid=0.001, lab_ann=1.0),
    "gpt":      dict(slope=0.6, race=0.25, income=-0.10, lab=1.0, noise=0.10, fmt=0.05, prose=0.0, grid=0.001, lab_ann=1.0),
    "gemini":   dict(slope=1.0, race=0.00, income=-0.10, lab=1.0, noise=0.10, fmt=0.05, prose=0.0, grid=0.001, lab_ann=1.0),
    "muse":     dict(slope=1.4, race=0.25, income=-0.10, lab=1.0, noise=0.20, fmt=0.10, prose=0.30, grid=0.001, lab_ann=1.0),
    "glm":      dict(slope=1.0, race=0.25, income=-0.40, lab=1.0, noise=0.10, fmt=0.05, prose=0.0, grid=0.001, lab_ann=1.0),
    "gpt_free":       dict(slope=0.8, race=0.25, income=-0.10, lab=1.0, noise=0.15, fmt=0.05, prose=0.0, grid=0.001, lab_ann=1.0),
    "claude_free":    dict(slope=1.0, race=0.60, income=-0.10, lab=1.0, noise=0.15, fmt=0.10, prose=0.0, grid=0.001, lab_ann=1.0),
    "gemini_free":    dict(slope=0.9, race=0.00, income=-0.10, lab=1.0, noise=0.12, fmt=0.05, prose=0.0, grid=0.001, lab_ann=1.0),
    "medgemma":       dict(slope=1.1, race=0.25, income=-0.10, lab=1.0, noise=0.20, fmt=0.10, prose=0.10, grid=0.01, lab_ann=1.0),
    "gemma":          dict(slope=0.7, race=0.25, income=-0.10, lab=0.6, noise=0.25, fmt=0.10, prose=0.10, grid=0.01, lab_ann=1.0),
}


# synthetic cost and speed of each simulated family (mean completion tokens, median seconds per answer): invented, only
# so the dry run exercises the cost and timing outputs (cost_speed.py, timing_sample.py)
SIM_USAGE = {"jev": (0, 0.6), "gpt": (450, 9.0), "claude": (300, 7.0), "gemini": (700, 4.0), "muse": (1500, 18.0),
             "glm": (3000, 40.0),
             "gpt_free": (300, 5.0), "claude_free": (250, 5.0), "gemini_free": (150, 2.0), "medgemma": (10, 3.0),
             "gemma": (10, 3.5)}


@dataclass
class _SimClient:
    owner: "SimulatedModels"
    model: str
    repeat: int
    variant: str = "raw"

    def probability(self, text: str, statement: str) -> dict:
        return self.owner.predict(self.model, self.repeat, text, self.variant)


class SimulatedModels:
    def __init__(self, cohort: pd.DataFrame, states: pd.DataFrame):
        self.cohort = cohort.set_index(cohort.SEQN.astype(int))
        self.lookup = {t: (int(p), e, bool(a)) for t, p, e, a in zip(states.text, states.profile, states.edit, states.annotated)}

    def client(self, model: str, repeat: int, variant: str = "raw"):
        return _SimClient(self, model, repeat, variant)

    @staticmethod
    def _u(*parts) -> float:
        h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
        return int.from_bytes(h[:8], "little") / 2**64

    def _z(self, *parts) -> float:
        u1, u2 = max(self._u(*parts, "a"), 1e-12), self._u(*parts, "b")
        return float(np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2))

    def predict(self, model: str, repeat: int, text: str, variant: str = "raw") -> dict:
        pid, edit, ann = self.lookup[text]
        base_model = "jev" if model.startswith("jev") else model.partition("@")[0]   # 'gpt@low': reasoning arm
        b = dict(BEHAVIOURS[base_model])
        if model.endswith("@low"):   # simulated: less reasoning, slightly compressed clinical response, more noise
            b["slope"], b["noise"] = b["slope"] * 0.9, b["noise"] + 0.05
        if variant == "M1":          # simulated instruction: every model drops race to zero (overshoot)
            b["race"] = 0.0
        if edit == "M2_race_removed":  # race unseen
            b["race"] = 0.0
        rec = self.cohort.loc[pid].to_dict()
        if edit in P.PERT["clinical"] or edit in P.PERT["label"]:
            rec = P.apply_record_edit(rec, edit).record
        lab_w = b["lab_ann"] if ann else b["lab"]
        lo = true_logit(rec, race_nhb=b["race"], income=b["income"], lab_weight=lab_w)
        lo = -1.7 + b["slope"] * (lo + 1.7)
        lo += b["noise"] * self._z(model, pid, edit, ann, repeat, variant)
        if edit in P.PERT["irrelevant"]:
            lo += b["fmt"] * self._z("fmt", model, pid, edit) + (b["prose"] if edit == "I3_prose" else 0.0)
        p = 1 / (1 + np.exp(-lo))
        if model == "jev_bundle":        # simulated: the same answer as the single question, rarely one grid step off
            g = b["grid"]
            p = round(round(p / g) * g + (g if self._u("bundle", pid) < 0.05 else 0.0), 6)
            p = min(p, 1.0)
            return {"p": p, "valid": True, "provider": "sim", "model_reported": model, "raw_path": None,
                    "p_alive": round(1 - p, 2), "p_bands": p}
        if model == "jev_complement":
            p = min(1.0, max(0.0, 1 - p + 0.08 * self._z("comp", pid)))
            return {"p": round(p, 2), "valid": True, "provider": "sim", "model_reported": model, "raw_path": None}
        grid = 0.001 if model == "jev_bands" else b["grid"]
        p = round(round(p / grid) * grid, 6)
        invalid = self._u("invalid", model, pid, edit, repeat) < (0.01 if base_model == "claude" else 0.002)
        out_tok, med_s = SIM_USAGE[base_model]
        tin = len(text) // 4 + (0 if base_model == "jev" else 320)
        usage = ({"input_tokens": tin, "output_tokens": 0} if base_model == "jev" else
                 {"prompt_tokens": tin,
                  "completion_tokens": max(1, int(out_tok * np.exp(0.3 * self._z("tok", model, pid, edit, repeat, variant))))})
        lat = med_s * float(np.exp(0.35 * self._z("lat", model, pid, edit, repeat, variant)))
        return {"p": None if invalid else p, "valid": not invalid, "provider": "sim", "model_reported": model, "raw_path": None,
                "usage": usage, "latency_s": round(lat, 3)}

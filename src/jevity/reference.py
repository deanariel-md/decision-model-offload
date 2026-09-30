"""Outcome-fitted reference models on the same variables the queried models see.

Primary: spline logistic regression (light fixed penalty, Newton solver) with label x age and race x sex terms.
Secondary: LightGBM, and a discrete-time survival model (per-cycle horizon, label effects after year ten separate).
All are fit on the fitting split only. An NHANES III power prior (a0 from config/nhanes3.yaml: 0 for the primary, 0.5
in the sensitivities) keeps the intercept, labels and label interactions era-specific when used. The contrast
d for an edited record is q(edited) - q(baseline) on the probability and (clipped) log-odds scales. Reference
uncertainty comes from refits on bootstrap resamples of the fitting split (and the prior cohort when a prior is used)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.impute import SimpleImputer

NUMERIC = ["age", "bmi", "waist_cm", "sbp_mmhg", "dbp_mmhg", "income_poverty_ratio", "rx_count",
           "albumin_g_dl", "creatinine_mg_dl", "glucose_mg_dl", "crp_mg_dl", "lymphocyte_pct", "mcv_fl",
           "rdw_pct", "alk_phos_u_l", "wbc_10e3_ul", "hba1c_pct", "total_chol_mg_dl", "hdl_mg_dl", "uacr_mg_g"]
CATEGORICAL = ["sex", "race_ethnicity", "education", "insured", "smoking", "self_rated_health",
               "diabetes", "chf", "chd", "mi", "stroke", "emphysema", "chronic_bronchitis", "cancer"]
OUTCOME = "death_10y"
SKEWED = ["creatinine_mg_dl", "crp_mg_dl", "uacr_mg_g", "glucose_mg_dl", "alk_phos_u_l"]   # log before splines
# Contrast references are NOT tuned for prediction (a penalty tuned for prediction shrinks contrasts): the primary
# reference uses a light constant penalty (C = 10 on standardised spline bases).
SPLINE_C = 10.0
# Exact Newton solver: Newton-Cholesky converges to the penalised optimum in a few iterations (a solver that stops early
# acts as hidden shrinkage).
SOLVER = dict(solver="newton-cholesky", max_iter=100)


def _log(x):
    return np.log(np.clip(x, 1e-3, None))


ERA = "era_nhanes3"   # 1 for NHANES III prior rows: an era-specific intercept, so baseline mortality is not borrowed
_LEVELS = {"race_ethnicity": ["Non-Hispanic White", "Non-Hispanic Black", "Mexican American", "Other Hispanic",
                              "Other race including multiracial"],
           "education": ["less than 9th grade", "9th to 11th grade", "high school graduate or GED", "some college",
                         "college graduate or above"],
           "insured": ["yes", "no"], "sex": ["male", "female"]}
_NUM_ERA = {"income_poverty_ratio": 2.0}   # fill for the interaction column only


def _era_specific_default() -> list[str]:
    try:
        cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "config" / "nhanes3.yaml").read_text(encoding="utf-8"))
        return [v for v in cfg["prior"]["era_specific"] if v != "intercept"]
    except Exception:
        return []


def era_columns(era_specific: list[str]) -> list[str]:
    cols = []
    for v in era_specific:
        if v in _LEVELS:
            cols += [f"era_x_{v}_{i}" for i in range(len(_LEVELS[v]))]
        elif v in _NUM_ERA:
            cols.append(f"era_x_{v}")
    return cols


# Effect modification of the tested labels (an additive reference biases D when a label's effect varies with age):
# every label gets a linear age term and race gets a sex term; in the power prior these are era-specific like the
# labels themselves.
IX_LABELS = ["race_ethnicity", "income_poverty_ratio", "education", "insured", "sex"]


def _ix_base(df: pd.DataFrame) -> dict[str, np.ndarray]:
    age_z = (pd.to_numeric(df["age"], errors="coerce").fillna(60.0).to_numpy() - 60.0) / 10.0 if "age" in df else np.zeros(len(df))
    fem = (df["sex"].astype(object).to_numpy() == "female").astype(float) if "sex" in df else np.zeros(len(df))
    out = {}
    for v in IX_LABELS:
        if v in _LEVELS:
            x = df[v].astype(object).to_numpy() if v in df else np.full(len(df), None)
            levs = _LEVELS[v][1:]            # reference level omitted
            for i, lev in enumerate(levs):
                out[f"ix_{v}_{i}_age"] = (x == lev) * age_z
                if v == "race_ethnicity":
                    out[f"ix_{v}_{i}_female"] = (x == lev) * fem
        else:
            x = pd.to_numeric(df[v], errors="coerce").fillna(_NUM_ERA[v]).to_numpy() if v in df else np.full(len(df), _NUM_ERA[v])
            out[f"ix_{v}_age"] = (x - _NUM_ERA[v]) / 1.5 * age_z
    return out


def ix_columns(era_specific: list[str]) -> list[str]:
    names = list(_ix_base(pd.DataFrame({"age": [60.0], "sex": ["male"]})).keys())
    split = [n for n in names if any(n.startswith(f"ix_{v}_") for v in era_specific)]
    return names + [f"era_{n}" for n in split]


def _frame(df: pd.DataFrame, era_specific: list[str] | None = None) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    era = pd.to_numeric(df[ERA], errors="coerce").fillna(0).to_numpy() if ERA in df else np.zeros(len(df))
    X[ERA] = era
    for v in (_era_specific_default() if era_specific is None else era_specific):
        if v in _LEVELS:
            for i, lev in enumerate(_LEVELS[v]):
                X[f"era_x_{v}_{i}"] = era * (df[v].astype(object).to_numpy() == lev) if v in df else 0.0
        elif v in _NUM_ERA:
            X[f"era_x_{v}"] = era * (pd.to_numeric(df[v], errors="coerce").fillna(_NUM_ERA[v]).to_numpy() if v in df else _NUM_ERA[v])
    es = _era_specific_default() if era_specific is None else era_specific
    for n, v in _ix_base(df).items():
        if any(n.startswith(f"ix_{lab}_") for lab in es):     # not borrowed: current and prior eras separate
            X[n], X[f"era_{n}"] = v * (1 - era), v * era
        else:
            X[n] = v
    for c in NUMERIC:
        X[c] = pd.to_numeric(df.get(c), errors="coerce")
    for c in CATEGORICAL:
        X[c] = df.get(c).astype("object").where(df.get(c).notna(), "missing") if c in df else "missing"
    return X


@dataclass
class Reference:
    name: str
    model: object
    era_specific: list | None = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict_proba(_frame(df, self.era_specific))[:, 1], 1e-6, 1 - 1e-6)

    def contrast(self, base: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
        p0, p1 = self.predict(base), self.predict(edited)
        # log-odds on the same clipped scale as model outputs ([0.005, 0.995], analysis.CLIP), so D = z - d never
        # compares a squeezed model contrast with an unsqueezed reference contrast at extreme risks
        lo = lambda p: np.log(np.clip(p, 0.005, 0.995) / (1 - np.clip(p, 0.005, 0.995)))
        return pd.DataFrame({"q0": p0, "q1": p1, "d_prob": p1 - p0, "d_logit": lo(p1) - lo(p0)}, index=base.index)


def _preprocess(spline: bool, era_specific: list[str] | None = None) -> ColumnTransformer:
    lin = [c for c in NUMERIC if c not in SKEWED]
    def num(log: bool):
        steps = [("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
        if log:
            steps.append(("log", FunctionTransformer(_log)))
        steps.append(("sc", StandardScaler()))
        if spline:
            steps.append(("spl", SplineTransformer(n_knots=4, degree=3, knots="quantile", extrapolation="linear")))
        return Pipeline(steps)
    return ColumnTransformer([
        ("num", num(False), lin),
        ("lognum", num(True), SKEWED),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("era", "passthrough", [ERA] + era_columns(_era_specific_default() if era_specific is None else era_specific)),
        ("ix", "passthrough", ix_columns(_era_specific_default() if era_specific is None else era_specific)),
    ])


SURVEY_WEIGHT = "wt_mec_10y"   # combined 1999-2008 examination weight (nhanes.combined_mec_weight)


def survey_weights(fit_df: pd.DataFrame) -> np.ndarray:
    """Examination weights rescaled to mean 1 in the fitting split, so the effective sample size and the prior's a0
    keep their meaning; rows without a weight get the mean."""
    w = pd.to_numeric(fit_df[SURVEY_WEIGHT], errors="coerce").to_numpy(dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    w = np.where(np.isnan(w), np.nanmean(w), w)
    return w / w.mean()


def _pooled(fit_df: pd.DataFrame, prior_df: pd.DataFrame | None, a0: float, weighted: bool = False):
    """Power prior as a weighted likelihood: current rows weight 1 (or their rescaled examination weight when
    `weighted`, a sensitivity analysis), NHANES III rows weight a0, era intercept free."""
    cur = fit_df.assign(**{ERA: 0})
    wc = survey_weights(fit_df) if weighted else np.ones(len(cur))
    if prior_df is None or a0 <= 0:
        return cur, wc
    pri = prior_df.assign(**{ERA: 1})
    both = pd.concat([cur, pri], ignore_index=True)
    return both, np.r_[wc, np.full(len(pri), float(a0))]


def _fit_on_current_basis(pre: ColumnTransformer, est, data: pd.DataFrame, w: np.ndarray, es: list[str], fit_kw: str) -> Pipeline:
    """Fit imputation, scaling and spline knots on CURRENT-era rows only, then the estimator on all (weighted) rows, so
    the basis never changes with a0."""
    X = _frame(data, es)
    cur = (X[ERA].to_numpy() == 0)
    pre.fit(X[cur])
    est.fit(pre.transform(X), data[OUTCOME].astype(int).to_numpy(), **{fit_kw: w})
    return Pipeline([("pre", pre), ("est", est)])


def fit_lgbm(fit_df: pd.DataFrame, seed: int = 0, prior_df: pd.DataFrame | None = None, a0: float = 0.0,
             era_specific: list[str] | None = None, **params) -> Reference:
    import lightgbm as lgb
    p = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, verbose=-1)
    p.update(params)
    data, w = _pooled(fit_df, prior_df, a0)
    es = _era_specific_default() if era_specific is None else era_specific
    model = _fit_on_current_basis(_preprocess(spline=False, era_specific=es), lgb.LGBMClassifier(**p), data, w, es, "sample_weight")
    return Reference("lightgbm", model, es)


def fit_spline_logit(fit_df: pd.DataFrame, C: float = SPLINE_C, prior_df: pd.DataFrame | None = None, a0: float = 0.0,
                     era_specific: list[str] | None = None, weighted: bool = False) -> Reference:
    """era_specific: labels whose coefficients get an era interaction (estimated separately in each era, so they are
    not borrowed); default from config/nhanes3.yaml. Basis (imputer, scaler, knots) fitted on current-era rows.
    weighted: survey-weighted sensitivity (examination weights on the current rows)."""
    data, w = _pooled(fit_df, prior_df, a0, weighted=weighted)
    es = _era_specific_default() if era_specific is None else era_specific
    model = _fit_on_current_basis(_preprocess(spline=True, era_specific=es), LogisticRegression(C=C, **SOLVER),
                                  data, w, es, "sample_weight")
    return Reference("spline_logit", model, es)


# ----------------------------------------------------------------------------- survival sensitivity reference
SURV_INTERVAL = 12    # months per discrete period
SURV_MAX = 240        # administrative truncation: 20 years (1999 examinations reach ~20.9 years by December 2019)
HORIZON = 120         # the estimand stays ten-year risk


def _n_periods(df: pd.DataFrame, interval: int = SURV_INTERVAL, max_months: int = SURV_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Periods at risk and the death indicator per person, with follow-up truncated at a per-cycle horizon that every
    member of the cycle reaches (months from the end of the cycle's last year to 31 December 2019, capped at 240), so
    no one contributes a partial period (deaths in a partial final period would count while its survivors drop out)."""
    t = pd.to_numeric(df["permth_exm"], errors="coerce").to_numpy(float)
    if "cycle" in df:
        end = pd.to_numeric(df["cycle"].astype(str).str[-4:], errors="coerce").to_numpy(float)
        T = np.where(np.isfinite(end), np.minimum((2019 - end) * 12, max_months), max_months)
    else:
        T = np.full(len(df), float(max_months))
    T = np.floor(T / interval) * interval
    dead = (pd.to_numeric(df["mortstat"], errors="coerce").to_numpy() == 1) & (t <= T)
    t = np.minimum(np.nan_to_num(t, nan=0.0), T)
    n = np.where(dead, np.maximum(np.ceil(t / interval), 1), np.floor(t / interval)).astype(int)
    return n, dead


LABELS = ["race_ethnicity", "income_poverty_ratio", "education", "insured", "sex"]   # never borrowed across time
LATE = 10   # periods 0-9 give the ten-year risk; label effects in periods 10+ are estimated separately


def _label_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for v in LABELS:
        if v in _LEVELS:
            x = df[v].astype(object).to_numpy() if v in df else np.full(len(df), None)
            cols += [(x == lev).astype(float) for lev in _LEVELS[v]]
        else:
            x = pd.to_numeric(df[v], errors="coerce").fillna(_NUM_ERA[v]).to_numpy() if v in df else np.full(len(df), _NUM_ERA[v])
            cols.append((x - 2.0) / 1.5)
    return np.column_stack(cols)


def _late_matrix(df: pd.DataFrame, era: np.ndarray | None = None) -> np.ndarray:
    """Label indicators plus the label interaction terms, duplicated by era (current, NHANES III) so that neither era's
    late label effects are shared with the other."""
    M = np.column_stack([_label_matrix(df)] + list(_ix_base(df).values()))
    e = np.zeros(len(df)) if era is None else era
    return np.hstack([M * (1 - e)[:, None], M * e[:, None]])


def _time_design(Z, age: np.ndarray, k: np.ndarray, k_max: int, L: np.ndarray | None = None):
    """Person basis Z (rows already repeated per period) plus period dummies, one age x time term (mortality's age
    gradient flattens with follow-up) and, when L is given, the label indicators again for periods >= 10 only. Clinical
    effects are held constant over follow-up and borrowed from years 11-20; label effects are not (they are free to
    differ after year ten, as the Black-White mortality gap is known to narrow with age), so the ten-year label
    contrasts come from years 1-10 alone."""
    from scipy import sparse
    Pk = sparse.csr_matrix((np.ones(len(k)), (np.arange(len(k)), k)), shape=(len(k), k_max))
    ax = ((np.nan_to_num(age, nan=60.0) - 60.0) / 10.0) * ((k - (k_max - 1) / 2) / 10.0)
    blocks = [sparse.csr_matrix(Z), Pk, sparse.csr_matrix(ax[:, None])]
    if L is not None:
        blocks.append(sparse.csr_matrix(L * (k >= LATE)[:, None]))
    return sparse.hstack(blocks).tocsr()


@dataclass
class SurvivalReference(Reference):
    """Discrete-time (person-period) logistic hazard on the primary basis, fitted on all follow-up to 20 years; predicts
    ten-year risk 1 - prod_{k<10}(1 - h_k). Secondary reference: deaths after year ten sharpen the clinical effects and
    the baseline hazard (assumed constant over follow-up, age excepted); label effects after year ten are separate."""
    k_max: int = SURV_MAX // SURV_INTERVAL
    split_labels: bool = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pre, est = self.model
        Z = pre.transform(_frame(df, self.era_specific))
        age = pd.to_numeric(df["age"], errors="coerce").to_numpy(float)
        L = _late_matrix(df) if self.split_labels else None
        log_s = np.zeros(len(df))
        for k in range(HORIZON // SURV_INTERVAL):
            h = est.predict_proba(_time_design(Z, age, np.full(len(df), k), self.k_max, L))[:, 1]
            log_s += np.log1p(-np.clip(h, 1e-9, 1 - 1e-9))
        return np.clip(1 - np.exp(log_s), 1e-6, 1 - 1e-6)


def fit_survival_logit(fit_df: pd.DataFrame, C: float = SPLINE_C, prior_df: pd.DataFrame | None = None, a0: float = 0.0,
                       era_specific: list[str] | None = None, split_labels: bool = True) -> SurvivalReference:
    data, w = _pooled(fit_df, prior_df, a0)
    ok = pd.to_numeric(data["permth_exm"], errors="coerce").notna().to_numpy()
    data, w = data[ok].reset_index(drop=True), w[ok]
    es = _era_specific_default() if era_specific is None else era_specific
    X = _frame(data, es)
    pre = _preprocess(spline=True, era_specific=es)
    pre.fit(X[X[ERA].to_numpy() == 0])
    Z = pre.transform(X)
    n, dead = _n_periods(data)
    keep = n > 0
    idx = np.repeat(np.flatnonzero(keep), n[keep])
    k = np.concatenate([np.arange(m) for m in n[keep]])
    last = np.r_[np.cumsum(n[keep]) - 1]
    y = np.zeros(len(idx), int); y[last[dead[keep]]] = 1
    k_max = SURV_MAX // SURV_INTERVAL
    L = _late_matrix(data, data[ERA].to_numpy(float))[idx] if split_labels else None
    D = _time_design(Z[idx], pd.to_numeric(data["age"], errors="coerce").to_numpy(float)[idx], k, k_max, L)
    est = LogisticRegression(C=C, **SOLVER).fit(D, y, sample_weight=w[idx])
    return SurvivalReference("survival_logit", (pre, est), es, k_max, split_labels)


def crossfit_predictions(fit_df: pd.DataFrame, targets: pd.DataFrame, k: int = 10, seed: int = 0,
                         prior_df: pd.DataFrame | None = None, a0: float = 0.0) -> pd.DataFrame:
    """Out-of-fold predictions for fitting-set records (targets that belong to fit_df) and full-fit predictions for the
    rest, for the spline reference, LightGBM and age + sex. A record is never scored by a model fitted on it."""
    rng = np.random.default_rng(seed)
    fold = pd.Series(rng.integers(0, k, len(fit_df)), index=fit_df.SEQN.astype(int).to_numpy())
    tid = targets.SEQN.astype(int)
    out = pd.DataFrame(index=tid.to_numpy(), columns=["p_spline_logit", "p_lightgbm", "p_age_sex"], dtype=float)
    in_fit = tid.isin(fold.index).to_numpy()
    groups = [(f, fit_df[fold.to_numpy() != f], targets[in_fit & (fold.reindex(tid).to_numpy() == f)]) for f in range(k)]
    groups.append((None, fit_df, targets[~in_fit]))
    for f, train, score in groups:
        if len(score) == 0:
            continue
        for name, fn in (("p_spline_logit", lambda: fit_spline_logit(train, prior_df=prior_df, a0=a0)),
                         ("p_lightgbm", lambda: fit_lgbm(train, prior_df=prior_df, a0=a0)),
                         ("p_age_sex", lambda: fit_age_sex(train))):
            out.loc[score.SEQN.astype(int).to_numpy(), name] = fn().predict(score)
    out.index.name = "SEQN"
    return out.reset_index()


def fit_age_sex(fit_df: pd.DataFrame) -> Reference:
    """Age + sex logistic comparator (no prior)."""
    model = Pipeline([("pre", ColumnTransformer([("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), ["age"]),
                                                  ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex"])])),
                      ("lr", LogisticRegression(max_iter=1000))])
    model.fit(_frame(fit_df), fit_df[OUTCOME].astype(int))
    return Reference("age_sex", model)


def bootstrap_refits(fit_df: pd.DataFrame, n: int = 300, seed: int = 0, kind: str = "lightgbm",
                     prior_df: pd.DataFrame | None = None, a0: float = 0.0) -> Iterator[Reference]:
    """Outer-loop reference refits: the fitting split and (if used) the NHANES III prior cohort are each resampled
    with replacement, so prior uncertainty enters the intervals too. Yielded one at a time to bound memory."""
    rng = np.random.default_rng(seed)
    for b in range(n):
        f = fit_df.iloc[rng.integers(0, len(fit_df), len(fit_df))].reset_index(drop=True)
        pr = None if prior_df is None else prior_df.iloc[rng.integers(0, len(prior_df), len(prior_df))].reset_index(drop=True)
        if kind == "lightgbm":
            yield fit_lgbm(f, seed=b, prior_df=pr, a0=a0)
        elif kind == "survival_logit":
            yield fit_survival_logit(f, prior_df=pr, a0=a0)
        elif kind == "spline_logit_weighted":     # same seed as spline_logit: the same resamples, so shifts are paired
            yield fit_spline_logit(f, prior_df=pr, a0=a0, weighted=True)
        else:
            yield fit_spline_logit(f, prior_df=pr, a0=a0)


def phenoage_mortality_10y(df: pd.DataFrame) -> np.ndarray:
    """Ten-year mortality probability underlying PhenoAge (Levine et al. 2018, Aging; Gompertz form).
    Units: albumin g/L, creatinine umol/L, glucose mmol/L, ln(CRP mg/dL), lymphocyte %, MCV fL, RDW %, ALP U/L,
    WBC 10^3/uL, age years. Coefficients from the published model."""
    xb = (-19.9067
          - 0.0336 * (df["albumin_g_dl"] * 10.0)
          + 0.0095 * (df["creatinine_mg_dl"] * 88.4)
          + 0.1953 * (df["glucose_mg_dl"] * 0.0555)
          + 0.0954 * np.log(np.clip(df["crp_mg_dl"], 1e-3, None))
          - 0.0120 * df["lymphocyte_pct"]
          + 0.0268 * df["mcv_fl"]
          + 0.3306 * df["rdw_pct"]
          + 0.00188 * df["alk_phos_u_l"]
          + 0.0554 * df["wbc_10e3_ul"]
          + 0.0804 * df["age"])
    gamma, t = 0.0076927, 120.0
    return 1.0 - np.exp(-np.exp(xb) * (np.exp(gamma * t) - 1.0) / gamma)


PHENOAGE_COLUMNS = ["albumin_g_dl", "creatinine_mg_dl", "glucose_mg_dl", "crp_mg_dl", "lymphocyte_pct", "mcv_fl",
                    "rdw_pct", "alk_phos_u_l", "wbc_10e3_ul", "age"]


@dataclass
class PhenoAgeReference:
    """PhenoAge's ten-year mortality model as an external reference for edits to the fields it contains (albumin,
    creatinine, age; glucose, CRP and the blood counts are not edited). Published coefficients, developed by other
    authors on NHANES III (which also enters the primary reference as its prior, so the two are not independent);
    fixed, so it adds no refit uncertainty of its own. Its contrasts condition on the nine
    biomarkers and age, not on the full record, so they are a triangulation of the primary reference, not a substitute.
    Contrasts for edits to other fields are zero by construction and are never analysed."""
    name: str = "phenoage"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(phenoage_mortality_10y(df).to_numpy(dtype=float), 1e-6, 1 - 1e-6)

    def contrast(self, base: pd.DataFrame, edited: pd.DataFrame) -> pd.DataFrame:
        p0, p1 = self.predict(base), self.predict(edited)
        lo = lambda p: np.log(np.clip(p, 0.005, 0.995) / (1 - np.clip(p, 0.005, 0.995)))
        return pd.DataFrame({"q0": p0, "q1": p1, "d_prob": p1 - p0, "d_logit": lo(p1) - lo(p0)}, index=base.index)

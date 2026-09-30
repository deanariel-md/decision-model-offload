"""Outcome references for the eICU replication: death before hospital discharge on the fields the models see.
Fitted on the credentialed full eICU (demo stays and patients excluded, config/eicu.yaml exclude_from_fit) on a local
machine; the fitted objects hold coefficients and bases, never records. Same design as the NHANES primary: spline
logistic regression with a light constant penalty (config reference.spline_C, not tuned for prediction), Newton solver,
missing-value indicators (ICU missingness is informative), label x age terms; LightGBM and XGBoost as secondary
references. Contrasts on the clipped log-odds scale (analysis.CLIP)."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

from jevity.eicu import CFG, OUTCOME, YES_NO
from jevity.reference import Reference, SOLVER

NUMERIC = [c for _, c, _, _ in CFG["numeric"]]
SKEWED = [c for c in CFG["support"]["log_columns"] if c in NUMERIC]
CATEGORICAL = ["sex", "ethnicity", "admit_source", "unit_type"] + YES_NO
DX = "admission_dx"
ETHNICITY = ["Caucasian", "African American", "Hispanic", "Asian", "Native American", "Other/Unknown"]
SPLINE_C = float(CFG["reference"]["spline_C"])


def _log(x):
    return np.log(np.clip(x, 1e-3, None))


def frame(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        X[c] = pd.to_numeric(df.get(c), errors="coerce")
    for c in CATEGORICAL + [DX]:
        X[c] = df[c].astype("object").where(df[c].notna(), "missing") if c in df else "missing"
    age = (pd.to_numeric(df["age"], errors="coerce").fillna(65.0).to_numpy() - 65.0) / 10.0
    for lev in ETHNICITY[1:]:                          # label x age: label effects may differ with age
        X[f"ix_eth_{lev}"] = (df["ethnicity"].astype(object).to_numpy() == lev) * age
    X["ix_female_age"] = (df["sex"].astype(object).to_numpy() == "female") * age
    return X


IX = [f"ix_eth_{lev}" for lev in ETHNICITY[1:]] + ["ix_female_age"]


def _preprocess(spline: bool) -> ColumnTransformer:
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
        ("dx", OneHotEncoder(handle_unknown="infrequent_if_exist",
                             min_frequency=int(CFG["reference"]["dx_min_frequency"])), [DX]),
        ("ix", "passthrough", IX),
    ])


class EicuReference(Reference):
    """Reference.contrast (clipped log-odds and probability differences) on the eICU design matrix."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict_proba(frame(df))[:, 1], 1e-6, 1 - 1e-6)


def fit_spline_logit(fit_df: pd.DataFrame, C: float = SPLINE_C) -> EicuReference:
    m = Pipeline([("pre", _preprocess(spline=True)), ("est", LogisticRegression(C=C, **SOLVER))])
    m.fit(frame(fit_df), fit_df[OUTCOME].astype(int).to_numpy())
    return EicuReference("spline_logit", m)


def fit_lgbm(fit_df: pd.DataFrame, seed: int = 0) -> EicuReference:
    import lightgbm as lgb
    p = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, verbose=-1)
    m = Pipeline([("pre", _preprocess(spline=False)), ("est", lgb.LGBMClassifier(**p))])
    m.fit(frame(fit_df), fit_df[OUTCOME].astype(int).to_numpy())
    return EicuReference("lightgbm", m)


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def fit_xgb(fit_df: pd.DataFrame, seed: int = 0) -> EicuReference:
    """XGBoost beside LightGBM. Constant settings, not tuned, matched to fit_lgbm's where XGBoost has the same setting
    (depth 4 in place of 15 leaves; no minimum leaf size); dense input so zeros are values, not missing."""
    import xgboost as xgb
    p = dict(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
             tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=8)
    m = Pipeline([("pre", _preprocess(spline=False)), ("dense", FunctionTransformer(_dense)), ("est", xgb.XGBClassifier(**p))])
    m.fit(frame(fit_df), fit_df[OUTCOME].astype(int).to_numpy())
    return EicuReference("xgboost", m)


def bootstrap_refits(fit_df: pd.DataFrame, n: int, seed: int = 0, kind: str = "spline_logit") -> Iterator[EicuReference]:
    rng = np.random.default_rng(seed)
    for b in range(n):
        f = fit_df.iloc[rng.integers(0, len(fit_df), len(fit_df))].reset_index(drop=True)
        if kind == "lightgbm":
            yield fit_lgbm(f, seed=b)
        elif kind == "xgboost":
            yield fit_xgb(f, seed=b)
        elif kind == "spline_logit":
            yield fit_spline_logit(f)
        else:
            raise ValueError(f"unknown reference kind: {kind}")

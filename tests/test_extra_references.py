"""PhenoAge as an independent edit reference; survey-weighted sensitivity reference; NCHS weight combination."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import reference as R
from jevity.nhanes import combined_mec_weight
from jevity.simulate import make_synthetic_cohort


def test_combined_weight_rule():
    df = pd.DataFrame({"cycle": ["1999-2000", "2001-2002", "2003-2004", "2007-2008"],
                       "wtmec4yr": [1000.0, 2000.0, np.nan, np.nan], "wtmec2yr": [9.0, 9.0, 3000.0, 5000.0]})
    assert np.allclose(combined_mec_weight(df), [400.0, 800.0, 600.0, 1000.0])


def test_phenoage_reference_fields():
    co = make_synthetic_cohort(n=400, seed=5)
    ph, b = R.PhenoAgeReference(), co.copy()
    up = lambda col, f: ph.contrast(b, b.assign(**{col: f(b[col])})).d_logit
    assert (up("albumin_g_dl", lambda x: x - 0.5) > 0).all()
    assert (up("age", lambda x: x + 5) > 0).all()
    assert (up("creatinine_mg_dl", lambda x: x * 1.5) > 0).all()
    assert np.allclose(up("sbp_mmhg", lambda x: x + 20), 0)          # not a PhenoAge field: never analysed
    assert np.allclose(up("hba1c_pct", lambda x: x + 1), 0)


def test_weighted_reference():
    co = make_synthetic_cohort(n=6000, seed=6, n_pilot=0, n_eval=500)
    fit, ev = co[co.split == "fit"], co[co.split == "eval"]
    w = R.survey_weights(fit)
    assert abs(w.mean() - 1) < 1e-9 and (w > 0).all()
    ru, rw = R.fit_spline_logit(fit, era_specific=[]), R.fit_spline_logit(fit, era_specific=[], weighted=True)
    e = ev.assign(sbp_mmhg=ev.sbp_mmhg + 20)
    du, dw = ru.contrast(ev, e).d_logit.mean(), rw.contrast(ev, e).d_logit.mean()
    assert du > 0 and dw > 0 and abs(du - dw) < 0.1     # correctly specified model: weighting barely moves contrasts

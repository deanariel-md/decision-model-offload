"""The power prior borrows covariate effects, not baseline mortality, and the SAS layout parser reads CDC forms."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import reference as R
from jevity.simulate import make_synthetic_cohort, make_synthetic_prior
from jevity.nhanes3 import parse_sas_layout


def test_era_intercept_not_borrowed():
    cur = make_synthetic_cohort(n=3000, seed=21, n_pilot=0, n_eval=0)
    pri = make_synthetic_prior(n=3000, seed=22, intercept_shift=1.5)      # far higher baseline mortality in the prior era
    m0 = R.fit_spline_logit(cur)
    m1 = R.fit_spline_logit(cur, prior_df=pri, a0=1.0)
    p0, p1 = m0.predict(cur).mean(), m1.predict(cur).mean()
    assert abs(p1 - p0) < 0.02, (p0, p1)                     # current-era predictions keep the current baseline
    assert pri.death_10y.mean() > cur.death_10y.mean() + 0.15


def test_race_not_borrowed_by_default_but_borrowed_when_shared():
    cur = make_synthetic_cohort(n=3000, seed=31, n_pilot=0, n_eval=0)
    pri = make_synthetic_prior(n=6000, seed=32, race_nhb=1.25)             # exaggerated drift so any pull is visible
    b = cur.head(300).assign(race_ethnicity="Non-Hispanic White"); e = b.assign(race_ethnicity="Non-Hispanic Black")
    d0 = R.fit_spline_logit(cur, era_specific=[]).contrast(b, e).d_logit.mean()
    d_era = R.fit_spline_logit(cur, prior_df=pri, a0=0.5).contrast(b, e).d_logit.mean()          # config default: race era-specific
    d_shared = R.fit_spline_logit(cur, prior_df=pri, a0=0.5, era_specific=[]).contrast(b, e).d_logit.mean()
    assert abs(d_era - d0) < 0.1, (d0, d_era)
    assert d_shared > d0 + 0.2, (d0, d_shared)


def test_sas_layout_forms():
    t = "INPUT\n SEQN 1-5\n HSSEX 15\n HFA8R $ 1234-1235\n @1300 DMPPIR 6.3\n;"
    lay = parse_sas_layout(t)
    assert lay["SEQN"] == (1, 5, False) and lay["HSSEX"] == (15, 15, False)
    assert lay["HFA8R"] == (1234, 1235, True) and lay["DMPPIR"] == (1300, 1305, False)

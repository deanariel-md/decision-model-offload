"""Discrete-time survival reference: person-period bookkeeping and recovery of ten-year risk on synthetic data."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import reference as R
from jevity.simulate import make_synthetic_survival_cohort, true_risk_10y


def test_periods():
    df = pd.DataFrame({"permth_exm": [13, 131, 5, 12, 300, 250], "mortstat": [1, 0, 0, 1, 0, 1]})
    n, dead = R._n_periods(df)
    assert n.tolist() == [2, 10, 0, 1, 20, 20]
    assert dead.tolist() == [True, False, False, True, False, False]   # a death after 240 months is censored at 240
    cyc = pd.DataFrame({"permth_exm": [140, 133, 150, 230], "mortstat": [0, 1, 0, 1],
                        "cycle": ["2007-2008", "2007-2008", "2005-2006", "1999-2000"]})
    n, dead = R._n_periods(cyc)
    assert n.tolist() == [11, 11, 12, 19]                  # horizons 132, 132, 156, 228 months
    assert dead.tolist() == [False, False, False, False]   # deaths beyond the cycle horizon are censored there


def test_recovers_ten_year_risk():
    co = make_synthetic_survival_cohort(n=6000, seed=5)
    fit, ev = co[co.split == "fit"], co[co.split == "eval"]
    m = R.fit_survival_logit(fit, era_specific=[])
    p, t = m.predict(ev), true_risk_10y(ev)
    assert abs(p.mean() - t.mean()) < 0.02
    assert np.corrcoef(p, t)[0, 1] > 0.9
    base = ev.reset_index(drop=True)
    assert m.contrast(base, base.assign(sbp_mmhg=base.sbp_mmhg + 20)).d_logit.mean() > 0

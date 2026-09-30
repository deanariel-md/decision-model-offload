"""Confirmatory family and headline metrics on a small synthetic replicate (the machinery, not the statistics)."""
import importlib.util, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import analysis as A

spec = importlib.util.spec_from_file_location("fs", ROOT / "scripts" / "family_simulation.py")
fs = importlib.util.module_from_spec(spec); spec.loader.exec_module(fs)


def test_family_replicate_runs_and_records_factor():
    rows = fs.one_rep(seed=11, n=3000, R_=5, B=100)
    import pandas as pd
    df = pd.DataFrame(rows)
    assert {"null", "planted"} <= set(df.scenario)
    assert (df.edit == "clinical_slope").sum() == 12            # six systems x two scenarios
    fam_rows = df[df.model.isin(fs.MODELS) & (df.edit != "weights_spearman")]
    assert (pd.to_numeric(fam_rows.crit) >= pd.to_numeric(fam_rows.crit_raw) - 1e-12).all()
    assert A.CRIT_INFLATION >= 1.0
    summ = fs.summarise(df, {})
    assert summ["k_hat_95th_percentile"] is not None and "1.05" in summ["by_inflation"]

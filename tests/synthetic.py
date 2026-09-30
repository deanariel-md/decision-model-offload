"""Synthetic cohort for tests (thin wrapper over jevity.simulate)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity.simulate import make_synthetic_cohort


def make(n=4000, seed=0):
    return make_synthetic_cohort(n=n, seed=seed)

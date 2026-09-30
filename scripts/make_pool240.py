"""The 240 staging reports: all 120 difficult staging reports (100 with a misleading feature, 20 with a node at an
unusual site) plus 120 of the 1,200 ordinary ones (the first 1,000 and the 200 written by another model), drawn at
random without replacement with seed 20260929 from the ordinary report ids in sorted order. Set names as in
scripts/summaries/workflow_groups.py (main, heldout, misleading_feature, nonregional_sites).

  python scripts/make_pool240.py      -> data/arm3_pool/pool240.csv (report_id, set)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED, N_ORDINARY = 20260929, 120
SETS = {"main": ROOT / "data/arm3/items.csv", "heldout": ROOT / "data/arm3_heldout/items.csv",
        "misleading_feature": ROOT / "data/arm3/confuser_items.csv",
        "nonregional_sites": ROOT / "data/arm3_nonregional_sites/items.csv"}
SIZE = {"main": 1000, "heldout": 200, "misleading_feature": 100, "nonregional_sites": 20}


def sets() -> pd.DataFrame:
    f = pd.concat([pd.read_csv(p, dtype=str, usecols=["report_id"]).assign(set=s) for s, p in SETS.items()])
    assert f.groupby("set").size().to_dict() == SIZE and not f.report_id.duplicated().any()
    return f


if __name__ == "__main__":
    f = sets()
    ordinary = sorted(f.loc[f.set.isin(["main", "heldout"]), "report_id"])
    pick = set(np.random.default_rng(SEED).choice(ordinary, N_ORDINARY, replace=False).tolist())
    pool = f[f.set.isin(["misleading_feature", "nonregional_sites"]) | f.report_id.isin(pick)]
    order = {s: i for i, s in enumerate(SETS)}
    pool = pool.sort_values(["set", "report_id"], key=lambda c: c.map(order) if c.name == "set" else c)
    assert len(pool) == 240
    out = ROOT / "data" / "arm3_pool" / "pool240.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pool[["report_id", "set"]].to_csv(out, index=False, lineterminator="\n")
    print(f"{out.relative_to(ROOT)}: {len(pool)} reports; {pool.groupby('set', sort=False).size().to_dict()}")

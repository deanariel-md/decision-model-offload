"""Draw the 200-record subset of the reasoning arm from the evaluation split's IDs only (no outcome column is read),
seed 20260922, and write config/subsets.yaml; prints its SHA-256. Refuses to overwrite an existing subset file.
  python scripts/draw_subset.py"""
import hashlib, sys
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity.clients import MODELS

if __name__ == "__main__":
    cfg = MODELS["subset"]
    out = ROOT / cfg["file"]
    if out.exists():
        sys.exit(f"{out} exists: the subset is drawn once and never redrawn")
    ids = pd.read_parquet(ROOT / "data" / "cohort.parquet", columns=["SEQN", "split"])   # IDs and split only
    ev = sorted(ids.loc[ids.split == "eval", "SEQN"].astype(int))
    pick = sorted(int(x) for x in np.random.default_rng(int(cfg["seed"])).choice(ev, size=int(cfg["n"]), replace=False))
    doc = {cfg["key"]: {"profiles": pick, "n": len(pick), "seed": int(cfg["seed"]), "drawn_from": f"evaluation split ({len(ev)} IDs)",
                        "date": date.today().isoformat(), "used_by": ["reasoning"]}}
    out.write_text("# Subset of the reasoning arm, drawn by scripts/draw_subset.py from evaluation IDs only (no outcomes\n"
                   "# read).\n" + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(f"wrote {out}: {len(pick)} of {len(ev)} evaluation IDs, seed {cfg['seed']}")
    print("SHA-256", hashlib.sha256(out.read_bytes()).hexdigest())

"""Run every planned call for a split and population. Prints the cost estimate and asks before spending.
  python scripts/run_calls.py eval                       # every arm (runner.plan_split) on the evaluation split
  python scripts/run_calls.py eval --models gpt,claude   # narrow to these systems (calls.model values, e.g. gpt@low)
  python scripts/run_calls.py eval --simulate            # no API: simulated models (dry run of the whole pipeline)
  python scripts/run_calls.py wide                       # baseline states of the wide run
Batch route (the families under `batch` in config/models.yaml): submit, collect until nothing is waiting, then the
plain command, which reads batch answers from the raw cache and calls the other systems directly.
  python scripts/run_calls.py eval --batch submit
  python scripts/run_calls.py eval --batch collect        # repeat until "waiting 0"; then --batch submit again if any failed
  python scripts/run_calls.py eval
Split "framing" (sensitivity analysis): the primary chatbots with variant F1 on the subset's evaluation states; answers
in runs/framing (batch manifest runs/framing/batches).
Writes results/<split>/calls.parquet."""
import argparse, subprocess, sys
from pathlib import Path
import pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity.clients import RawStore, MODELS
from jevity.runner import plan_split, load_subset, make_client_factory, execute

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["eval", "wide", "framing"])
    ap.add_argument("--population", default="nhanes")
    ap.add_argument("--models", default=None, help="comma list or 'all'")
    ap.add_argument("--workers", type=int, default=16, help="parallel calls per system; every system runs at once")
    ap.add_argument("--hf_workers", type=int, default=8, help="parallel calls per system on the Hugging Face router")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--batch", choices=["submit", "collect", "status", "adopt", "drop"], default=None)
    ap.add_argument("batch_args", nargs="*", help="adopt: <key> <batch_id>; drop: <key>")
    a = ap.parse_args()
    pert = yaml.safe_load((ROOT / "config" / "perturbations.yaml").read_text())
    states = pd.read_parquet(ROOT / "data" / f"states_{'eval' if a.split == 'framing' else a.split}.parquet")
    systems = None if a.models in (None, "all") else a.models.split(",")
    calls = plan_split(states, a.split, pert, subset=load_subset() if a.split in ("eval", "framing") else None, systems=systems)
    counts = pd.Series([f"{c.model}|{c.variant}" for c in calls]).value_counts()
    print("Planned calls by system:")
    print(counts.to_string())
    store_split = "eval" if a.split == "wide" else a.split     # wide reuses the evaluation cache for identical requests
    if a.batch:
        from jevity import batch as BT
        store = RawStore(ROOT / "runs" / store_split)
        if a.batch == "status":
            print(pd.DataFrame(BT.status(store)).to_string()); sys.exit(0)
        if a.batch == "adopt":
            BT.adopt(store, *a.batch_args); sys.exit(0)
        if a.batch == "drop":
            BT.drop(store, *a.batch_args); sys.exit(0)
        if a.batch == "collect":
            t = BT.collect(store)
            print(f"written {t['written']}, failed {t['failed']} (resubmit with --batch submit), waiting {t['waiting_batches']}, "
                  f"errors {t['errors']} (retried next collect)")
            sys.exit(0)
        todo = BT.pending_requests(calls, store, a.population)
        print(f"batch route: {len(todo)} requests not yet answered", pd.Series([f for f, _, _ in todo]).value_counts().to_dict())
        subprocess.run([sys.executable, str(ROOT / "scripts" / "cost_estimate.py"), a.split] + ([a.models] if systems else []))
        if not a.yes and input("Submit? [y/N] ").strip().lower() != "y":
            sys.exit(0)
        BT.submit(calls, store, a.population)
        sys.exit(0)
    if not a.simulate:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "cost_estimate.py"), a.split] + ([a.models] if systems else []))
        if not a.yes and input("Proceed? [y/N] ").strip().lower() != "y":
            sys.exit(0)
        factory = make_client_factory(RawStore(ROOT / "runs" / a.split), a.population)
    else:
        from jevity.simulate import SimulatedModels
        cohort = pd.read_parquet(ROOT / "data" / "cohort.parquet")
        factory = make_client_factory(None, a.population, simulate=SimulatedModels(cohort, states).client)
    out = ROOT / "results" / a.split; out.mkdir(parents=True, exist_ok=True)
    if not a.simulate:
        factory = make_client_factory(RawStore(ROOT / "runs" / store_split), a.population)
    fams = MODELS["families"]
    workers = {"default": a.workers, **{m: a.hf_workers for m in {c.model for c in calls}
                                         if (fams.get(m.split("@")[0]) or {}).get("transport") == "hf_router"}}
    df = execute(calls, factory, a.population, out / "calls.parquet", workers=workers if not a.simulate else 1)
    print(df.groupby("model")["valid"].agg(["size", "mean"]))
    from jevity.clients import BATCH_MISSING
    miss = df[df.error == BATCH_MISSING]
    if len(miss):
        print(f"WARNING {len(miss)} batch-route calls have no collected answer ({miss.model.value_counts().to_dict()}): "
              "run --batch collect / --batch submit and rerun before analysis.")
    print("providers reported:\n", df.groupby(["model", "provider"]).size())

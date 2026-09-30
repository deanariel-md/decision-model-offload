"""Jev on the eICU demo records in the recommended form, and the outcome Choice questions (src/jevity/gapfill.py,
config/gapfill_eicu.yaml). Jev only. Items:
  eicu_structured  Jev's death question on the 1,297 eICU demo evaluation stays, categorised records
  eicu_bands       Jev's eICU outcome Choice on the records as given (drop-in) and the categorised records (2 x 1,297)
  nhanes_bands     the NHANES risk-band question on the 12,910 recommended-form records
  python scripts/gapfill/run.py dry-run                  # builds and checks every request; projects; sends nothing
  python scripts/gapfill/run.py full --item ITEM|all --yes [--max_usd X] [--workers N]
  python scripts/gapfill/run.py analyze                  # results/summaries/gapfill_*.json (scripts/gapfill/analyze.py)
Every live command prints the calls not yet stored and their projected cost. With --max_usd it refuses when the spend
so far (from the stored replies) plus the projection passes that amount, and sends through a guard that refuses any
request once spent plus in flight would pass it; without it there is no stop. Keys come from .env
(OPENROUTER_API_KEY). Raw requests and replies, verbatim, first write wins: runs/gapfill_<item>/; call tables:
results/gapfill/<item>.parquet."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv                                                    # noqa: E402

load_dotenv(ROOT / ".env")
from jevity import docuse_risk as DR                                                # noqa: E402
from jevity import eicu as E                                                        # noqa: E402
from jevity import gapfill as G                                                     # noqa: E402
from jevity.clients import JevClient, RawStore                                      # noqa: E402
from jevity.runner import Call, execute                                             # noqa: E402

RUNS = ROOT / "runs"
STORES = {"eicu_structured": RUNS / "gapfill_eicu_structured", "eicu_bands": RUNS / "gapfill_eicu_bands",
          "nhanes_bands": RUNS / "gapfill_nhanes_bands"}
TABLES = ROOT / "results" / "gapfill"
ITEMS = tuple(STORES)
WORKERS = {"eicu_structured": 8, "eicu_bands": 8, "nhanes_bands": 16}      # parallel requests (--workers)
MAX_USD: float | None = None          # stop on the spend so far plus the projection (--max_usd); None: no stop


# ------------------------------------------------------------------------------------------------ plans
def eicu_drop_in() -> pd.DataFrame:
    """The eICU records as given (drop-in), exactly as the eICU replication sent them (data/eicu/states_eval.parquet),
    demo stays only."""
    ev = G.eval_cohort(ROOT)
    st = pd.read_parquet(ROOT / "data" / "eicu" / "states_eval.parquet")
    st = st[(st.edit == "baseline") & (~st.annotated)]
    if set(st.profile.astype(int)) != set(ev.SEQN.astype(int)):
        raise SystemExit("states_eval.parquet and the evaluation cohort differ")
    return st


def eicu_categorised() -> pd.DataFrame:
    ev = G.eval_cohort(ROOT)
    rows = [{"profile": int(r["SEQN"]), "state": G.categorised_state(r)} for r in ev.to_dict("records")]
    return pd.DataFrame(rows)


def nhanes_states() -> pd.DataFrame:
    return DR.build_states(root=ROOT)


def plan(item: str) -> list[tuple[str, Call]]:
    """(client kind, call) for every request of an item."""
    if item == "eicu_structured":
        return [("jev_eicu", Call("jev", int(r.profile), "categorised", False, 0, r.state))
                for r in eicu_categorised().itertuples(index=False)]
    if item == "eicu_bands":
        a = [("eicu_bands", Call("jev_eicu_bands", int(r.profile), "baseline", False, 0, r.text))
             for r in eicu_drop_in().itertuples(index=False)]
        b = [("eicu_bands", Call("jev_eicu_bands", int(r.profile), "categorised", False, 0, r.state))
             for r in eicu_categorised().itertuples(index=False)]
        return a + b
    if item == "nhanes_bands":
        return [("nhanes_bands", Call("jev_bands", int(r.profile), "recommended", False, 0, r.state))
                for r in nhanes_states().itertuples(index=False)]
    raise ValueError(item)


def client_class(kind: str):
    return {"jev_eicu": G.guarded(JevClient), "eicu_bands": G.guarded(G.EicuBandClient),
            "nhanes_bands": G.guarded(JevClient)}[kind]


def make_client(kind: str, store: RawStore, guard=None):
    cls = client_class(kind)
    if kind == "jev_eicu":
        c = cls(store, G.EICU_POP)
    elif kind == "eicu_bands":
        c = cls(store)
    else:
        c = cls(store, "nhanes", question="risk_bands")
    c.guard = guard
    return c


def population(item: str) -> str:
    return "nhanes" if item == "nhanes_bands" else G.EICU_POP


def requests(item: str) -> list[dict]:
    store = RawStore(STORES[item])
    out = []
    for kind, c in plan(item):
        out.append(make_client(kind, store).request(c.text, G.statement(population(item))))
    return out


# ------------------------------------------------------------------------------------------------ commands
def spent_all() -> float:
    return sum(G.stored_spend(RawStore(d)) for d in STORES.values())


def dry_run(max_usd: float | None = MAX_USD) -> dict:
    out, tot = {}, 0.0
    for item in ITEMS:
        store = RawStore(STORES[item])
        reqs = requests(item)
        new = [r for r in reqs if store.get(r) is None]
        shas = {json.dumps(G.body(r), sort_keys=True, ensure_ascii=False) for r in reqs}
        usd = sum(G.projected_usd(r) for r in new)
        out[item] = {"requests": len(reqs), "distinct": len(shas), "already_stored": len(reqs) - len(new),
                     "projected_usd": usd, "provider_blocks": sorted({json.dumps(r["provider"]) for r in reqs}),
                     "models": sorted({r["model"] for r in reqs}),
                     "question_ids": sorted({q for r in reqs for q in r["questions"]})}
        tot += usd
    spent = spent_all()
    out["_total"] = {"projected_usd": tot, "spent_so_far": spent, "max_usd": max_usd,
                     "over_max": max_usd is not None and spent + tot > max_usd}
    return out


def run(item: str, yes: bool, max_usd: float | None = MAX_USD, workers: int | None = None) -> pd.DataFrame | None:
    store = RawStore(STORES[item])
    pl = plan(item)
    reqs = [make_client(k, store).request(c.text, G.statement(population(item))) for k, c in pl]
    new = [r for r in reqs if store.get(r) is None]
    usd = sum(G.projected_usd(r) for r in new)
    spent = spent_all()
    stop = f"stop ${max_usd:.2f}" if max_usd is not None else "no stop set"
    print(f"{item}: {len(pl)} calls ({len(new)} to send, {len(pl) - len(new)} stored); projected ${usd:.4f}; "
          f"spent so far ${spent:.4f}; {stop}")
    if max_usd is not None and spent + usd > max_usd:
        raise SystemExit(f"spending stop: spent ${spent:.4f} plus projected ${usd:.4f} is above ${max_usd:.2f}; "
                         "nothing sent")
    if not yes:
        print("add --yes to send")
        return None
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set (.env)")
    guard = G.SpendGuard(max_usd, spent)
    clients = {}

    def client_for(model: str, repeat: int, variant: str = "raw"):
        kind = {"jev": "jev_eicu", "jev_eicu_bands": "eicu_bands", "jev_bands": "nhanes_bands"}[model]
        if kind not in clients:
            clients[kind] = make_client(kind, store, guard)
        return clients[kind]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / f"{item}.parquet"
    df = execute([c for _, c in pl], client_for, population(item), out, workers=workers or WORKERS[item])
    print(df.groupby(["edit", "parse"], dropna=False).size().to_string())
    print(df.groupby("provider").size().to_string())
    print(f"spent after: ${spent_all():.4f}; wrote {out}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["dry-run", "full", "analyze"])
    ap.add_argument("--item", default=None, choices=list(ITEMS) + ["all"])
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--max_usd", type=float, default=MAX_USD,
                    help="refuse when the spend so far plus the projection is above this (USD); default: no stop")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel requests (default: 8 for the eICU items, 16 for nhanes_bands)")
    a = ap.parse_args(argv)
    if a.command == "dry-run":
        print(json.dumps(dry_run(a.max_usd), indent=1))
        return
    if a.command == "analyze":
        import analyze as AN
        return AN.main([])
    items = ITEMS if a.item == "all" else (a.item,)
    for it in items:
        run(it, a.yes, a.max_usd, a.workers)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

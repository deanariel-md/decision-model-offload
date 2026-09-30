"""Arm 3 held-out reports: the tested systems on data/arm3_heldout/items.csv (src/jevity/heldout.py). The systems
command prints its calls and projected list cost, reads the key's usage, the account balance and the open batch holds,
checks the spending stops when they are set (--max_usd: this work's spend so far, from the stored replies, plus the
projection; --project_stop_usd: the key's usage plus every open batch hold plus the projection), and needs --yes. Keys
come from the .env file named by config/arm3.yaml keys.env_file.
  python scripts/heldout/run.py systems --yes [--systems jev_documented,jev,gpt,claude,gemini,glm]
  python scripts/heldout/run.py systems --yes --systems muse
  python scripts/heldout/run.py tables            # results/arm3_heldout/calls.parquet, documented_calls.parquet, parts.parquet
Options: --key VAR sends with another .env variable (this process only); --workers N parallel requests per system.
Raw requests and replies, verbatim, first write wins: runs/arm3_heldout/."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical as K
from jevity import docuse_staging as D
from jevity import heldout as H
from jevity.clients import MODELS, RawStore, _key

WORKERS = 8
ARM3_SYSTEMS = ("jev",) + H.MAIN_CHATBOTS
ALL_SYSTEMS = ("jev_documented",) + ARM3_SYSTEMS
BUDGET_ROOTS = [ROOT / "runs"]              # batch folders (read only), for the open holds
MAX_USD: float | None = None                # stop on this work's spend so far plus the projection; None: no stop
PROJECT_STOP_USD: float | None = None       # stop on the key's usage plus open holds plus the projection; None: no stop


def spent_usd() -> dict[str, float]:
    """This work's reported spend so far (usage.cost of every stored reply)."""
    out = {}
    for name, d in (("systems", H.SYSTEM_STORE),):
        tot = 0.0
        for f in (d.glob("*.json") if d.exists() else []):
            try:
                u = (json.loads(f.read_text(encoding="utf-8")).get("response") or {}).get("usage") or {}
                tot += float(u.get("cost") or 0.0)
            except (ValueError, AttributeError):
                continue
        out[name] = tot
    return out


def account(key_var: str = "OPENROUTER_API_KEY") -> dict:
    """The key's usage and limit and the account balance (categorical.openrouter_account; account reads only)."""
    saved = os.environ.get("OPENROUTER_API_KEY")
    try:
        os.environ["OPENROUTER_API_KEY"] = _key(key_var)
        return K.openrouter_account()
    finally:
        if saved is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = saved


def _stop_text(stop: float | None) -> str:
    return f"stop ${stop:,.2f}" if stop is not None else "no stop set"


def spending_check(projection_usd: float, what: str, max_usd: float | None = MAX_USD,
                   project_stop_usd: float | None = PROJECT_STOP_USD) -> dict:
    """The stops before any paid command, each when set: this work's spend so far plus this command against max_usd;
    the key's usage plus every open batch hold plus this command against project_stop_usd. The account balance must
    cover this command. Refuses otherwise."""
    spent = spent_usd()
    acc = account()
    holds = K.open_holds(K.manifest_files(BUDGET_ROOTS))
    held = float(sum(holds.values()))
    total = acc["key_usage"] + held + projection_usd
    ours = sum(spent.values()) + projection_usd
    print(f"{what}: projected ${projection_usd:,.2f} (list)")
    print(f"  this work so far ${sum(spent.values()):,.2f} + this ${projection_usd:,.2f} = ${ours:,.2f} "
          f"({_stop_text(max_usd)})")
    print(f"  key usage ${acc['key_usage']:,.2f} + open holds ${held:,.2f} + this ${projection_usd:,.2f} = ${total:,.2f} "
          f"({_stop_text(project_stop_usd)}); balance ${acc['balance']:,.2f}")
    reasons = []
    if max_usd is not None and ours > max_usd:
        reasons.append(f"this work would reach ${ours:,.2f}, above ${max_usd:,.2f}")
    if project_stop_usd is not None and total > project_stop_usd:
        reasons.append(f"key usage plus holds plus this would be ${total:,.2f}, above ${project_stop_usd:,.2f}")
    if acc["balance"] < projection_usd:
        reasons.append(f"balance ${acc['balance']:,.2f} below this command's ${projection_usd:,.2f}")
    if reasons:
        raise SystemExit("STOP (nothing sent): " + "; ".join(reasons))
    return {"spent": spent, "account": acc, "holds": held, "total": total, "ours": ours}


# ------------------------------------------------------------------------------------------------ tested systems
def system_plan(ids: list[str], systems: tuple[str, ...]) -> tuple[list[K.CatCall], list[K.CatCall]]:
    """(documented Jev calls, arm 3 builder calls): repeat 0 only, every report, both versions."""
    doc = [K.CatCall("jev", i, 0, v) for v in D.VARIANTS for i in ids] if "jev_documented" in systems else []
    a3 = [K.CatCall(s, i, 0, v) for s in ARM3_SYSTEMS if s in systems for v in ("names", "definitions") for i in ids]
    return K.block_order(doc, int(D.CFG["seed"])) if doc else [], K.block_order(a3, H.SEED) if a3 else []


def system_projection(doc: list[K.CatCall], a3: list[K.CatCall]) -> pd.DataFrame:
    """Per system: calls, input tokens from the built requests (4 characters per token), output tokens at arm 3's mean
    per system and version, list price."""
    means = H.arm3_token_means()
    rows = []
    if doc:
        cl = D.DocuseJevClient(RawStore(ROOT / "runs" / "_unused"), D.arm_spec())
        by = {(i.item_id, i.variant): i for i in H.documented_items()}
        tin = sum(D.request_tokens(cl.request(by[(c.item_id, c.variant)])) for c in doc)
        rows.append({"system": "jev_documented", "calls": len(doc), "tokens_in": tin, "tokens_out": 0,
                     "usd": tin * MODELS["jev"]["price_per_mtok_input"] / 1e6})
    if a3:
        arm, items = H.heldout_arm(), H.arm3_items()
        by = {(i.item_id, i.variant): i for i in items}
        for s in ARM3_SYSTEMS:
            cs = [c for c in a3 if c.system == s]
            if not cs:
                continue
            if s == "jev":
                tin = sum(K.estimate_tokens(by[(c.item_id, c.variant)].state
                                            + json.dumps(K.jev_question(arm, by[(c.item_id, c.variant)]))) for c in cs)
                rows.append({"system": s, "calls": len(cs), "tokens_in": tin, "tokens_out": 0,
                             "usd": tin * MODELS["jev"]["price_per_mtok_input"] / 1e6})
                continue
            f = MODELS["families"][s]
            tin = sum(K.estimate_tokens(K.system_text(arm) + K.user_text(arm, by[(c.item_id, c.variant)])) for c in cs)
            tout = sum(means[(s, c.variant)]["tokens_out"] for c in cs)
            rows.append({"system": s, "calls": len(cs), "tokens_in": tin, "tokens_out": tout,
                         "usd": H.usd(tin, tout, f["price_in"], f["price_out"])})
    return pd.DataFrame(rows)


def cmd_systems(yes: bool, systems: tuple[str, ...], key: str | None = None, workers: int = WORKERS,
                max_usd: float | None = MAX_USD, project_stop_usd: float | None = PROJECT_STOP_USD) -> None:
    bad = set(systems) - set(ALL_SYSTEMS)
    assert not bad, f"unknown systems {bad}"
    frame = H.load_items_frame()
    ids = frame.report_id.tolist()
    doc, a3 = system_plan(ids, systems)
    proj = system_projection(doc, a3)
    print(f"tested systems on {len(ids)} reports:")
    print(proj.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    arm = H.heldout_arm()
    K.load_keys(arm)
    spending_check(float(proj.usd.sum()), "tested systems", max_usd, project_stop_usd)
    if not yes:
        raise SystemExit("add --yes to send")
    if key:                             # this process only
        os.environ["OPENROUTER_API_KEY"] = _key(key)
    sstore = RawStore(H.SYSTEM_STORE)
    if doc:
        done = K.execute(doc, H.documented_items(frame), D.make_factory(D.arm_spec(), sstore), None,
                         workers=workers, arm_name="arm3_heldout_documented")
        print("documented Jev:", done.groupby(["variant", "parse"], dropna=False).size().to_dict(),
              "errors", int(done.error.notna().sum()))
    if a3:
        done = K.execute(a3, H.arm3_items(), K.make_factory(arm, sstore, standard=True), None, workers=workers,
                         arm_name=arm.name)
        print(done.groupby(["system", "variant"]).valid.agg(["size", "sum"]).to_string())
        err = done[done.error.notna()]
        if len(err):
            print(f"WARNING {len(err)} calls failed (unusable; the command sends them again):",
                  err.error.str[:120].value_counts().head(5).to_dict())
    cmd_tables()


def stored_rows(calls: list[K.CatCall], items: list[K.Item], factory, arm_name: str) -> pd.DataFrame:
    """Rows of every planned call that has a stored reply, read from the raw store (a call without one is left out;
    nothing is sent)."""
    by = {(i.item_id, i.variant): i for i in items}
    rows = []
    for c in calls:
        cl, it = factory(c.system, c.repeat), by[(c.item_id, c.variant)]
        req = cl.request(it)
        if cl.store.get(req) is None:
            continue
        r = cl.answer(it)
        r = {"arm": arm_name, "system": c.system, "item_id": c.item_id, "variant": c.variant, "repeat": c.repeat, **r}
        r["probs"] = json.dumps(r["probs"]) if r.get("probs") else None
        rows.append(r)
    return pd.DataFrame(rows)


def cmd_tables() -> None:
    frame = H.load_items_frame()
    ids = frame.report_id.tolist()
    doc, a3 = system_plan(ids, ALL_SYSTEMS)
    sstore = RawStore(H.SYSTEM_STORE)
    H.RESULTS.mkdir(parents=True, exist_ok=True)
    dcalls, parts = D.calls_table(sstore, D.arm_spec(), H.documented_items(frame), doc)
    dcalls.to_parquet(H.RESULTS / "documented_calls.parquet", index=False)
    parts.to_parquet(H.RESULTS / "parts.parquet", index=False)
    arm = H.heldout_arm()
    calls = stored_rows(a3, H.arm3_items(), K.make_factory(arm, sstore, standard=True), arm.name)
    calls.to_parquet(H.RESULTS / "calls.parquet", index=False)
    print(f"documented_calls.parquet {len(dcalls)} rows ({int(dcalls.valid.sum()) if len(dcalls) else 0} usable); "
          f"calls.parquet {len(calls)} rows")
    if len(calls):
        print(calls.groupby(["system", "variant"]).valid.agg(["size", "sum"]).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["systems", "tables"])
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--systems", default=",".join(s for s in ALL_SYSTEMS if s != "muse"))
    ap.add_argument("--key", default=None, help="systems: the .env name of the OpenRouter key for this process")
    ap.add_argument("--workers", type=int, default=WORKERS, help="systems: parallel requests per system")
    ap.add_argument("--max_usd", type=float, default=MAX_USD,
                    help="systems: refuse when this work's spend so far plus the projection is above this (USD)")
    ap.add_argument("--project_stop_usd", type=float, default=PROJECT_STOP_USD,
                    help="systems: refuse when the key's usage plus open holds plus the projection is above this (USD)")
    a = ap.parse_args(argv)
    if a.command == "systems":
        return cmd_systems(a.yes, tuple(a.systems.split(",")), a.key, a.workers, a.max_usd, a.project_stop_usd)
    if a.command == "tables":
        return cmd_tables()


if __name__ == "__main__":
    main()

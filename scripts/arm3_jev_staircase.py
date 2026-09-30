"""Jev with stage names and with stage definitions on the difficult staging reports where the arm 3 runs did not ask
them: Jev with definitions on the 100 misleading-feature reports (data/arm3/confuser_items.csv), and Jev with names
and with definitions on the 20 unusual-site reports (data/arm3_nonregional_sites/items.csv). 140 calls.

Requests are built by arm 3's Jev client (jevity.categorical.CategoricalJevClient on config/arm3.yaml, only the items
file replaced). Before any call, every request is compared with the stored Jev request of the same version on the first
1,000 reports (results/arm3/calls.parquet, main pass): only the report text (state) and the item id may differ. One
call per report and version; the first stored reply stands (clients.RawStore, runs/arm3_jev_staircase); only the
client's own transport retries resend. Stops: a reply naming another model or provider; before any call, when set, a
projected list cost above --max-usd or the key's usage plus the projection above --project-stop-usd. Keys from arm 3's
keys.env_file, never printed.

  python scripts/arm3_jev_staircase.py build      # requests and checks; results/arm3_jev_staircase/requests.json
  python scripts/arm3_jev_staircase.py stub       # end to end on simulated replies in a scratch store
  python scripts/arm3_jev_staircase.py run --yes [--max-usd X] [--project-stop-usd Y]   # live: 140 calls
  python scripts/arm3_jev_staircase.py tables     # results/arm3_jev_staircase/calls.parquet from the stored replies
"""
from __future__ import annotations

import argparse, copy, dataclasses, hashlib, json, random, sys, tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical as K  # noqa: E402
from jevity.clients import MODELS, RawStore, _sha  # noqa: E402

STORE = ROOT / "runs" / "arm3_jev_staircase"
OUT = ROOT / "results" / "arm3_jev_staircase"
MODEL, PROVIDER = "typesafe/jev-1.13", "TypeSafe"
STOP_USD: float | None = None       # stop above this projected list cost (--max-usd); None: no stop
CAP_USD: float | None = None        # stop when the key's usage plus the projection exceeds this (--project-stop-usd)
SETS = {"confuser": (ROOT / "data/arm3/confuser_items.csv", ("definitions",)),
        "nonregional": (ROOT / "data/arm3_nonregional_sites/items.csv", ("names", "definitions"))}
N_EXPECTED = 140


def local(p: str) -> Path:
    """A stored raw path as the copy under ROOT/runs/."""
    q = p.replace("\\", "/")
    return ROOT / q[q.index("/runs/") + 1:] if "/runs/" in q else Path(p)


def arm_for(items: Path, variants: tuple) -> K.ArmSpec:
    a3 = K.load_arm("arm3")
    cfg = copy.deepcopy(a3.config)
    cfg["items"]["main"] = str(items)
    cfg["variants"] = {v: a3.config["variants"][v] for v in variants}
    return dataclasses.replace(a3, config=cfg)


def stored_jev() -> dict[str, list[dict]]:
    """Every stored Jev request of the main pass on the first 1,000 reports, by version."""
    c = pd.read_parquet(ROOT / "results/arm3/calls.parquet")
    c = c[(c.system == "jev") & (c.repeat == 0)]
    return {v: [json.loads(local(p).read_text(encoding="utf-8"))["request"] for p in g.raw_path]
            for v, g in c.groupby("variant")}


def build(store_dir: Path = STORE, sender=None) -> list:
    """(set, item, client, request) for every cell; stops on any difference from the stored arm 3 requests."""
    store = RawStore(store_dir)
    ref = stored_jev()
    rest = lambda q: {k: v for k, v in q.items() if k not in ("state", "_item")}
    for v, qs in ref.items():
        assert len(qs) == 1000, (v, len(qs))
        assert all(rest(q) == rest(qs[0]) for q in qs), f"stored {v} requests differ beyond report and item id"
    jobs = []
    for name, (items_path, variants) in SETS.items():
        arm = arm_for(items_path, variants)
        cl = K.CategoricalJevClient(store, arm, repeat=0, sender=sender)
        for it in K.load_items(arm):
            req = cl.request(it)
            want = rest(ref[it.variant][0])
            diff = sorted(k for k in set(rest(req)) | set(want) if rest(req).get(k) != want.get(k))
            if diff:
                raise SystemExit(f"request check stopped: {name} {it.item_id} {it.variant} differs in {diff}")
            if req["state"] != it.state or req["_item"] != it.item_id:
                raise SystemExit(f"request check stopped: {name} {it.item_id}: state or item id not the report's")
            jobs.append((name, it, cl, req))
    counts = pd.Series([f"{n}/{it.variant}" for n, it, _, _ in jobs]).value_counts().to_dict()
    assert len(jobs) == N_EXPECTED and counts == {"confuser/definitions": 100, "nonregional/names": 20,
                                                  "nonregional/definitions": 20}, counts
    return jobs


def projection(jobs) -> float:
    """Input tokens at the stored Jev requests' reported input per character of state, by version; output is free."""
    c = pd.read_parquet(ROOT / "results/arm3/calls.parquet")
    c = c[(c.system == "jev") & (c.repeat == 0)]
    items = pd.read_csv(ROOT / "data/arm3/items.csv", dtype=str, keep_default_na=False).set_index("report_id")["report"]
    tok = 0.0
    for v, g in c.groupby("variant"):
        per_char = g.tokens_in.sum() / sum(len(items[i]) for i in g.item_id)
        fixed = 0.0
        tok += sum(len(r["state"]) * per_char + fixed for _, it, _, r in jobs if it.variant == v)
    return tok * MODELS["jev"]["price_per_mtok_input"] / 1e6 * 1.5          # 1.5: headroom for longer reports


def key_usage() -> float:
    import httpx
    from dotenv import dotenv_values
    key = dotenv_values(ROOT / ".env").get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing from .env")
    r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return float(r.json()["data"]["usage"])


def answer_all(jobs, workers: int) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    stop = threading.Event()

    def one(job):
        name, it, cl, req = job
        if stop.is_set():
            return None
        try:
            r = cl.answer(it)
        except Exception as e:  # noqa: BLE001  recorded, never dropped
            r = K.empty_row(repr(e)[:300])
        bad = r.get("error") is None and not r.get("simulated") and (
            (r.get("model_reported") or "").split("-2026")[0] != MODEL or r.get("provider") != PROVIDER)
        if bad:
            stop.set()
        r = {"arm": "arm3", "system": "jev", "item_id": it.item_id, "variant": it.variant, "repeat": 0, **r,
             "set": name, "stopped_here": bool(bad)}
        r["probs"] = json.dumps(r["probs"]) if r.get("probs") else None
        return r

    with ThreadPoolExecutor(workers) as ex:
        rows = [f.result() for f in as_completed([ex.submit(one, j) for j in jobs])]
    df = pd.DataFrame([r for r in rows if r is not None])
    if stop.is_set():
        raise SystemExit("stopped: a reply named another model or provider:\n"
                         + df[df.stopped_here][["set", "item_id", "model_reported", "provider"]].to_string())
    return df.drop(columns="stopped_here")


def stub_sender(system: str, req: dict) -> dict:
    rng = random.Random(req["state"] + req["_variant"])
    crit = req["questions"]["crc_stage_group"]["criteria"]
    w = [rng.random() ** 4 for _ in crit]
    probs = [round(x / sum(w), 2) for x in w]
    return {"model": MODEL + "-20260917", "provider": PROVIDER, "id": "sim", "_sim_latency_s": 0.0,
            "answers": {"crc_stage_group": {"type": "score", "probabilities": {str(i): p for i, p in enumerate(probs)},
                                            "confidence": 0.5, "score": sum(i * p for i, p in enumerate(probs))}},
            "usage": {"input_tokens": 700, "output_tokens": 19}}


def table(jobs, store_dir: Path) -> pd.DataFrame:
    """The stored replies, with results/arm3/calls.parquet's columns plus `set`."""
    missing = [(n, it.item_id, it.variant) for n, it, _, r in jobs if RawStore(store_dir).get(r) is None]
    if missing:
        raise SystemExit(f"{len(missing)} cells without a stored reply, e.g. {missing[:3]}")
    df = answer_all(jobs, 8)
    cols = list(pd.read_parquet(ROOT / "results/arm3/calls.parquet").columns)
    order = {(n, it.item_id, it.variant): i for i, (n, it, _, _) in enumerate(jobs)}
    df = df.iloc[sorted(range(len(df)), key=lambda i: order[(df.set.iat[i], df.item_id.iat[i], df.variant.iat[i])])]
    return df[cols + ["set"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "stub", "run", "tables"])
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-usd", type=float, default=STOP_USD)
    ap.add_argument("--project-stop-usd", type=float, default=CAP_USD)
    a = ap.parse_args()
    if a.command == "stub":
        with tempfile.TemporaryDirectory() as d:
            jobs = build(Path(d), sender=stub_sender)
            df = answer_all(jobs, a.workers)
            assert len(df) == N_EXPECTED and df.valid.all(), "stub: every simulated reply should parse"
            t = table(jobs, Path(d))
            print(f"stub: {len(t)} simulated replies parsed; columns {len(t.columns)} "
                  f"(arm 3's plus set); by set and version {t.groupby(['set', 'variant']).size().to_dict()}")
        return
    jobs = build()
    OUT.mkdir(parents=True, exist_ok=True)
    man = [{"set": n, "item_id": it.item_id, "variant": it.variant, "sha": _sha(r)} for n, it, _, r in jobs]
    (OUT / "requests.json").write_text(json.dumps(man, indent=1))
    proj = projection(jobs)
    todo = sum(1 for *_, r in jobs if RawStore(STORE).get(r) is None)
    print(f"built {len(jobs)} Jev requests (confuser definitions 100, unusual-site names 20 and definitions 20), "
          f"{todo} without a stored reply; every one equals the stored arm 3 Jev request of its version apart from the "
          f"report text and the item id")
    print(f"requests manifest {(OUT / 'requests.json').relative_to(ROOT)} sha256 "
          f"{hashlib.sha256((OUT / 'requests.json').read_bytes()).hexdigest()}")
    print(f"projected list cost US${proj:.4f}" + (f" (stop above US${a.max_usd:.0f})" if a.max_usd is not None else ""))
    if a.command == "build":
        return
    if a.command == "tables":
        t = table(jobs, STORE)
        t.to_parquet(OUT / "calls.parquet", index=False)
        print(f"{OUT / 'calls.parquet'}: {len(t)} rows, usable {int(t.valid.sum())}; "
              f"{t.groupby(['set', 'variant']).valid.agg(['size', 'sum']).to_dict('index')}; "
              f"models {t.model_reported.value_counts().to_dict()}, providers {t.provider.value_counts().to_dict()}")
        return
    if not a.yes:
        raise SystemExit("run needs --yes")
    if a.max_usd is not None and proj > a.max_usd:
        raise SystemExit("projection over the stop: nothing sent")
    if a.project_stop_usd is not None:
        used = key_usage()
        print(f"key usage US${used:,.2f}; with the projection US${used + proj:,.2f} (cap US${a.project_stop_usd:,.0f})")
        if used + proj > a.project_stop_usd:
            raise SystemExit("over the project cap: nothing sent")
    K.load_keys(K.load_arm("arm3"))
    df = answer_all([j for j in jobs if RawStore(STORE).get(j[3]) is None] or jobs, a.workers)
    print(f"answered {len(df)}; usable {int(df.valid.sum())}; errors {int(df.error.notna().sum())}")


if __name__ == "__main__":
    main()

"""Separating the parts of the cancer-stage recommended form: two Jev versions on all 1,320 staging reports (the first
1,000, the 200 written by another model, the 100 with a misleading feature and the 20 with a node at an unusual site);
2,640 calls.

  named_field             the drop-in request (one Score question, level names) with the state sent as
                          {"pathology_report": <report>}, as every decomposed version sent it. Built from the stored Jev
                          names request of the same report; only `state` may differ, and the report text must equal the
                          stored drop-in text, the items file's report and the documented request's field.
  documented_no_examples  the stored documented request of the same report with every option's `examples` list removed;
                          only those lists may differ.

Answers are parsed by the code that parsed the stored versions (jevity.categorical for the Score answer,
docuse_staging.parse_answers for the four parts, with the joint top probability). One call per report and version; the
first stored reply stands (clients.RawStore, runs/arm3_jev_parts); only the client's own transport retries resend.
Stops: a reply naming another model or provider; before any call, when set, a projected list cost above --max-usd or
the key's usage plus the projection above --project-stop-usd. Keys from .env, never printed.

  python scripts/arm3_jev_parts.py build      # requests and checks; results/arm3_jev_parts/requests.json
  python scripts/arm3_jev_parts.py stub       # end to end on simulated replies in a scratch store
  python scripts/arm3_jev_parts.py run --yes [--max-usd X] [--project-stop-usd Y]   # live: 2,640 calls
  python scripts/arm3_jev_parts.py tables     # results/arm3_jev_parts/calls.parquet from the stored replies
"""
from __future__ import annotations

import argparse, copy, hashlib, json, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import categorical as K  # noqa: E402
from jevity import docuse_staging as DS  # noqa: E402
from jevity.clients import MODELS, RawStore, _sha  # noqa: E402

STORE = ROOT / "runs" / "arm3_jev_parts"
OUT = ROOT / "results" / "arm3_jev_parts"
MODEL, PROVIDER = "typesafe/jev-1.13", "TypeSafe"
STOP_USD: float | None = None       # stop above this projected list cost (--max-usd); None: no stop
CAP_USD: float | None = None        # stop when the key's usage plus the projection exceeds this (--project-stop-usd)
SETS = {"main": ROOT / "data/arm3/items.csv", "heldout": ROOT / "data/arm3_heldout/items.csv",
        "misleading_feature": ROOT / "data/arm3/confuser_items.csv",
        "nonregional_sites": ROOT / "data/arm3_nonregional_sites/items.csv"}
N_REPORTS = 1320


def raw(p: str) -> dict:
    """A stored request from its raw path (the file where it is, else the copy under ROOT/runs/)."""
    f = Path(p)
    if not f.exists():
        q = p.replace("\\", "/")
        f = ROOT / q[q.index("/runs/") + 1:]
    return json.loads(f.read_text(encoding="utf-8"))["request"]


def jev_rows(parquet: Path, variant: str, **eq) -> pd.DataFrame:
    c = pd.read_parquet(parquet)
    c = c[(c.system == "jev") & (c.variant == variant) & (c["repeat"] == 0)]
    for k, v in eq.items():
        c = c[c[k] == v]
    return c


def stored(kind: str) -> dict[str, dict]:
    """Stored main-pass Jev request per report: kind 'names' (drop-in) or 'documented'."""
    if kind == "names":
        fs = [jev_rows(ROOT / "results/arm3/calls.parquet", "names"),
              jev_rows(ROOT / "results/arm3_heldout/calls.parquet", "names"),
              jev_rows(ROOT / "results/arm3_confuser/calls.parquet", "names"),
              jev_rows(ROOT / "results/arm3_jev_staircase/calls.parquet", "names", set="nonregional")]
    else:
        fs = [jev_rows(ROOT / "results/arm3_docuse/calls.parquet", "documented"),
              jev_rows(ROOT / "results/arm3_heldout/documented_calls.parquet", "documented"),
              jev_rows(ROOT / "results/arm3_nonregional_sites/calls_jev.parquet", "documented")]
    c = pd.concat(fs)
    assert len(c) == N_REPORTS and not c.item_id.duplicated().any(), (kind, len(c))
    return {r.item_id: raw(r.raw_path) for r in c.itertuples()}


def flat(o, pre=()) -> dict:
    if isinstance(o, dict) and o:
        return {k: v for key, val in o.items() for k, v in flat(val, pre + (key,)).items()}
    if isinstance(o, list) and o:
        return {k: v for i, val in enumerate(o) for k, v in flat(val, pre + (i,)).items()}
    return {pre: o}


def differing(a: dict, b: dict) -> list[tuple]:
    fa, fb = flat(a), flat(b)
    return sorted((k for k in set(fa) | set(fb) if fa.get(k, KeyError) != fb.get(k, KeyError)), key=str)


def items() -> list[tuple[str, K.Item]]:
    labels = K.load_arm("arm3").labels
    out = []
    for s, p in SETS.items():
        for r in pd.read_csv(p, dtype=str, keep_default_na=False).to_dict("records"):
            out.append((s, K.Item(r["report_id"], r["report"].strip(), r["level"], labels, labels, (None,) * len(labels),
                                  r["level"], r.get("substage", ""), "")))
    assert len(out) == N_REPORTS
    return out


def build_requests() -> tuple[list, dict]:
    """(set, item, version, request) for every call; stops on any difference beyond the stated one."""
    names, doc = stored("names"), stored("documented")
    jobs, removed = [], {}
    for s, it in items():
        n, d = names[it.item_id], doc[it.item_id]
        if not (n["state"] == it.state == d["state"]["pathology_report"]):
            raise SystemExit(f"report text differs between the drop-in, the items file and the documented request: {it.item_id}")
        nf = copy.deepcopy(n)
        nf["state"] = {"pathology_report": n["state"]}
        dp = differing(nf, n)
        if set(dp) != {("state",), ("state", "pathology_report")} or nf["state"]["pathology_report"] != n["state"]:
            raise SystemExit(f"named field {it.item_id}: differs from the stored names request in {dp}")
        ne = copy.deepcopy(d)
        k = 0
        for q in ne["questions"].values():
            for c in q["criteria"].values():
                if isinstance(c, dict) and "examples" in c:
                    k += len(c.pop("examples"))
        dp = differing(ne, d)
        bad = [p for p in dp if not (len(p) == 6 and p[0] == "questions" and p[2] == "criteria" and p[4] == "examples")]
        left = any("examples" in c for q in ne["questions"].values() for c in q["criteria"].values() if isinstance(c, dict))
        if bad or left or not dp:
            raise SystemExit(f"no examples {it.item_id}: differs from the stored documented request in {bad or 'nothing'}")
        removed[it.item_id] = k
        jobs += [(s, it, "named_field", nf), (s, it, "documented_no_examples", ne)]
    assert len(jobs) == 2 * N_REPORTS
    return jobs, removed


class Prebuilt:
    """The request is the checked one; transport, store, gate and parsing are the class's own."""
    reqs: dict = {}

    def request(self, item: K.Item) -> dict:
        return self.reqs[item.item_id]


class NamedFieldClient(Prebuilt, K.CategoricalJevClient):
    pass


class NoExamplesClient(Prebuilt, DS.DocuseJevClient):
    pass


def clients(jobs, store: RawStore, sender=None) -> dict:
    nf = NamedFieldClient(store, K.load_arm("arm3"), 0, sender=sender)
    ne = NoExamplesClient(store, DS.arm_spec(), 0, sender=sender)
    nf.reqs = {it.item_id: r for _, it, v, r in jobs if v == "named_field"}
    ne.reqs = {it.item_id: r for _, it, v, r in jobs if v == "documented_no_examples"}
    return {"named_field": nf, "documented_no_examples": ne}


def answer_all(jobs, cls: dict, workers: int) -> pd.DataFrame:
    stop = threading.Event()

    def one(job):
        s, it, v, req = job
        if stop.is_set():
            return None
        try:
            r = cls[v].answer(it)
        except Exception as e:  # noqa: BLE001  recorded, never dropped
            r = K.empty_row(repr(e)[:300])
        bad = r.get("error") is None and not r.get("simulated") and (
            (r.get("model_reported") or "").split("-2026")[0] != MODEL or r.get("provider") != PROVIDER)
        if bad:
            stop.set()
        r = {k: val for k, val in r.items() if not k.startswith("_")}
        r = {"arm": req["_arm"], "system": "jev", "item_id": it.item_id, "variant": v, "repeat": 0, **r, "set": s,
             "stopped_here": bool(bad)}
        r["probs"] = json.dumps(r["probs"]) if r.get("probs") else None
        return r

    with ThreadPoolExecutor(workers) as ex:
        rows = [f.result() for f in as_completed([ex.submit(one, j) for j in jobs])]
    df = pd.DataFrame([r for r in rows if r is not None])
    if stop.is_set():
        raise SystemExit("stopped: a reply named another model or provider:\n"
                         + df[df.stopped_here][["set", "item_id", "model_reported", "provider"]].to_string())
    return df.drop(columns="stopped_here")


def projection(jobs) -> float:
    """Jev input tokens at the stored versions' reported input per report (names; documented less its examples, whose
    share of the request text is removed), US$0.042 per million, x1.2 headroom."""
    n = pd.concat([jev_rows(ROOT / "results/arm3/calls.parquet", "names"),
                   jev_rows(ROOT / "results/arm3_docuse/calls.parquet", "documented")])
    per = n.groupby("variant").tokens_in.mean()
    usd = N_REPORTS * (per["names"] + 10 + per["documented"]) * MODELS["jev"]["price_per_mtok_input"] / 1e6
    return usd * 1.2


def stub_sender(system: str, req: dict) -> dict:
    qs = req["questions"]
    if "crc_stage_group" in qs:
        crit = qs["crc_stage_group"]["criteria"]
        ans = {"crc_stage_group": {"type": "score", "probabilities": {str(i): (1.0 if i == 2 else 0.0) for i in range(len(crit))},
                                   "confidence": 0.5, "score": 2.0}}
    else:
        pick = {"t_category": "T3", "regional_nodes": "none", "tumor_deposits": "absent", "distant_metastasis": "none"}
        ans = {q: {"type": "choice", "choice": pick[q], "probabilities": {o: (1.0 if o == pick[q] else 0.0) for o in qs[q]["criteria"]},
                   "confidence": 0.9} for q in qs}
    return {"model": MODEL + "-20260917", "provider": PROVIDER, "id": "sim", "_sim_latency_s": 0.0, "answers": ans,
            "usage": {"input_tokens": 900, "output_tokens": 19}}


def key_usage() -> float:
    import httpx
    from dotenv import dotenv_values
    key = dotenv_values(ROOT / ".env").get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing from .env")
    r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return float(r.json()["data"]["usage"])


def table(jobs, store: RawStore) -> pd.DataFrame:
    missing = [(it.item_id, v) for _, it, v, r in jobs if store.get(r) is None]
    if missing:
        raise SystemExit(f"{len(missing)} calls without a stored reply, e.g. {missing[:3]}")
    df = answer_all(jobs, clients(jobs, store, sender=lambda *_: (_ for _ in ()).throw(RuntimeError("no stored reply"))), 16)
    cols = list(pd.read_parquet(ROOT / "results/arm3/calls.parquet").columns)
    order = {(it.item_id, v): i for i, (_, it, v, _) in enumerate(jobs)}
    df = df.iloc[sorted(range(len(df)), key=lambda i: order[(df.item_id.iat[i], df.variant.iat[i])])]
    return df[cols + ["set"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "stub", "run", "tables"])
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-usd", type=float, default=STOP_USD)
    ap.add_argument("--project-stop-usd", type=float, default=CAP_USD)
    a = ap.parse_args()
    jobs, removed = build_requests()
    kinds = pd.Series(removed).value_counts().to_dict()
    print(f"built {len(jobs)} Jev requests on {N_REPORTS} reports: named field (only the state differs from the stored "
          f"names request; report text identical in the drop-in, the items file and the documented request) and "
          f"documented without examples (only the examples lists differ; example strings removed per request: {kinds})")
    if a.command == "stub":
        with tempfile.TemporaryDirectory() as d:
            st = RawStore(Path(d))
            df = answer_all(jobs, clients(jobs, st, sender=stub_sender), a.workers)
            assert len(df) == len(jobs) and df.valid.all(), df.valid.value_counts()
            t = table(jobs, st)
            print(f"stub: {len(t)} simulated replies parsed; {t.groupby('variant').size().to_dict()}; columns arm 3's plus set")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    man = [{"set": s, "item_id": it.item_id, "variant": v, "sha": _sha(r)} for s, it, v, r in jobs]
    (OUT / "requests.json").write_text(json.dumps(man, indent=1))
    proj = projection(jobs)
    todo = sum(1 for *_, r in jobs if RawStore(STORE).get(r) is None)
    print(f"requests manifest {(OUT / 'requests.json').relative_to(ROOT)} sha256 "
          f"{hashlib.sha256((OUT / 'requests.json').read_bytes()).hexdigest()}; {todo} without a stored reply")
    print(f"projected list cost US${proj:.3f}" + (f" (stop above US${a.max_usd:.0f})" if a.max_usd is not None else ""))
    if a.command == "build":
        return
    if a.command == "tables":
        t = table(jobs, RawStore(STORE))
        t.to_parquet(OUT / "calls.parquet", index=False)
        print(f"{OUT / 'calls.parquet'}: {len(t)} rows; usable {t.groupby('variant').valid.agg(['size', 'sum']).to_dict('index')}; "
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
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    todo_jobs = [j for j in jobs if RawStore(STORE).get(j[3]) is None]
    df = answer_all(todo_jobs, clients(jobs, RawStore(STORE)), a.workers)
    print(f"answered {len(df)}; usable {df.groupby('variant').valid.agg(['size', 'sum']).to_dict('index')}; "
          f"errors {int(df.error.notna().sum())}")


if __name__ == "__main__":
    main()

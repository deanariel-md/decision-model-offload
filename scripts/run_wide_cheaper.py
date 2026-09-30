"""The free-plan and medical-training systems on the wide set's baseline records. GPT-5.6 Luna (gpt_free), Claude
Sonnet 5 (claude_free), Gemini 3.5 Flash-Lite (gemini_free), MedGemma 27B (medgemma) and Gemma 3 27B (gemma) answer the
ten-year death question on every baseline record of data/states_wide.parquet with the request every other system gets
on the wide set: the primary prompt, the record text, one pass, no repeats, no edits. The calls are runner.plan_calls
on the wide baseline states for these five systems, in runner.block_order. Answers go to the evaluation store
(runs/eval), which the wide split shares for identical requests (scripts/run_calls.py), so a stored answer to an
identical request is read, not asked again.

Routes (each system on its developer's own provider as config/models.yaml pins it; allow_fallbacks false):
- gpt_free, gemini_free: the batch route (config `batch`): `submit`, then `collect` until nothing is waiting.
- claude_free: the standard route, one request per call (config `batch` gives it no batch route).
- medgemma, gemma: synchronous on the Hugging Face router, on the provider config/models.yaml names. Gemma 3 27B can
  also be sent to a second provider (--provider gemma=deepinfra); the provider is part of the request's model id. A
  call answered by either provider is not asked again, and `build` reads either answer (the configured provider first).

  python scripts/run_wide_cheaper.py submit [--chunk gemini_free=250]   # batch route: submit the pending requests
  python scripts/run_wide_cheaper.py collect             # repeat until "waiting 0"; then submit again if any failed
  python scripts/run_wide_cheaper.py sync [--part 0/2]   # Claude Sonnet 5 on the standard route
  python scripts/run_wide_cheaper.py pair [--only gemma] [--hf_workers medgemma=3,gemma=2] [--provider gemma=deepinfra]
  python scripts/run_wide_cheaper.py build               # results/wide_cheaper/calls.parquet (no call)
  python scripts/run_wide_cheaper.py analyse [--leave_out medgemma]   # results/wide_cheaper/analysis.json (no call)

Keys come from .env: OPENROUTER_API_KEY (batch and standard routes) and HF_TOKEN (Hugging Face router). --key_env and
--hf_token name another variable to send with (this process only); a batch can be read only with the key that
submitted it.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import batch as BT
from jevity.clients import LLMClient, MODELS, PROMPTS, RawStore
from jevity.runner import block_order, execute, make_client_factory, plan_calls

SYSTEMS = ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
BATCH_SYSTEMS = ["gpt_free", "gemini_free"]          # config `batch`
SONNET = "claude_free"
PAIR = ["medgemma", "gemma"]
HF_WORKERS = "medgemma=3,gemma=2"
POPULATION = "nhanes"
OUT = ROOT / "results" / "wide_cheaper"
STORE = RawStore(ROOT / "runs" / "eval")
PROGRESS = ROOT / "runs" / "wide_cheaper_progress"
NOT_STORED = "no stored answer"
STMT = PROMPTS["statements"][POPULATION]
# Hugging Face providers besides the configured one: DeepInfra also serves google/gemma-3-27b-it on the router; MedGemma
# 27B has no other provider. One provider per call; the provider is in the request's model id.
HF_CONFIG = {m: MODELS["families"][m]["hf_model"] for m in PAIR}        # config value, before any in-memory change
HF_ALT = {"gemma": ["deepinfra"]}


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def use_key(name: str, default: str) -> None:
    """Send with the key held in variable `name` instead of `default` (this process only; .env unchanged)."""
    if name != default:
        if not os.environ.get(name):
            raise SystemExit(f"{name} is not set (.env)")
        os.environ[default] = os.environ[name]


# ------------------------------------------------------------------------------------------------ plan and routes
def plan() -> list:
    """Every call for the five systems: the wide baseline states, one pass, in block order (the block order of the
    five systems' calls together, so every step sees the same order)."""
    s = pd.read_parquet(ROOT / "data" / "states_wide.parquet")
    base = s[(s.edit == "baseline") & (~s.annotated)]
    return block_order(plan_calls(base, SYSTEMS, [], [], 1, annotated_models=set()))


def hf_variants(m: str) -> list[str]:
    """Hugging Face providers for a pair system: the configured one first, then the alternatives."""
    return [HF_CONFIG[m].split(":")[1]] + list(HF_ALT.get(m, []))


def with_provider(req: dict, m: str, prov: str) -> dict:
    return {**req, "model": f"{HF_CONFIG[m].split(':')[0]}:{prov}"}


def stored_variant(req: dict, m: str) -> tuple[str | None, dict | None]:
    """The provider whose answer to this request is stored (the configured provider first) and its request."""
    for prov in hf_variants(m):
        r = with_provider(req, m, prov)
        if STORE.path(r).exists():
            return prov, r
    return None, None


# ------------------------------------------------------------------------------------------------ batch route
def submit(a) -> int:
    """gpt_free and gemini_free on the batch route: every request without a stored answer, in block order, in batches
    of --chunk requests per family (default: config `batch` max_requests). A request in an open batch, or one that
    reached config `batch` max_attempts submissions, is not sent again."""
    use_key(a.key_env, "OPENROUTER_API_KEY")
    calls = [c for c in plan() if c.model in BATCH_SYSTEMS]
    todo = BT.pending_requests(calls, STORE, POPULATION)
    print(f"{now()} batch route: {len(todo)} requests without a stored answer {dict(Counter(f for f, _, _ in todo))}",
          flush=True)
    if not a.yes and input("Submit? [y/N] ").strip().lower() != "y":
        return 0
    chunk = {f: int(n) for f, n in (x.split("=") for x in a.chunk.split(",") if x)}   # requests per batch (transport)
    for fam in BATCH_SYSTEMS:
        BT.submit([c for c in calls if c.model == fam], STORE, POPULATION, max_requests=chunk.get(fam))
    return 0


def collect(a) -> int:
    """Ingest every finished batch of the store (safe to repeat)."""
    use_key(a.key_env, "OPENROUTER_API_KEY")
    t = BT.collect(STORE)
    print(f"written {t['written']}, failed {t['failed']} (submit again), waiting {t['waiting_batches']}, "
          f"errors {t['errors']} (retried next collect)")
    return 0


# ------------------------------------------------------------------------------------------------ Hugging Face pair
def pair(a) -> int:
    """MedGemma 27B and Gemma 3 27B, synchronous on the Hugging Face router, --hf_workers calls in flight per system;
    --part i/n takes every n-th call of each system's block order (disjoint parts for parallel processes)."""
    workers = {k: int(v) for k, v in (x.split("=") for x in a.hf_workers.split(","))}
    systems = [m for m in PAIR if m in (a.only.split(",") if a.only else PAIR)]
    provider = {k: v for k, v in (x.split("=") for x in a.provider.split(",") if x)}
    for m, prov in provider.items():             # this process only; config/models.yaml unchanged
        if m not in systems or prov not in HF_ALT.get(m, []):
            raise SystemExit(f"no alternative provider {prov!r} for {m}")
        MODELS["families"][m]["hf_model"] = f"{HF_CONFIG[m].split(':')[0]}:{prov}"
    use_key(a.hf_token, "HF_TOKEN")
    i, n = (int(x) for x in a.part.split("/"))  # part i of n: every n-th call of each system's block order (disjoint)
    print(f"{now()} " + ", ".join(f"{m} on {MODELS['families'][m]['hf_model']} ({workers[m]} in flight)" for m in systems)
          + f", part {i}/{n}, token {a.hf_token}", flush=True)
    calls = [c for m in systems for j, c in enumerate(c for c in plan() if c.model == m) if j % n == i]
    clients = {m: LLMClient(m, STORE, POPULATION) for m in systems}
    PROGRESS.mkdir(parents=True, exist_ok=True)
    factory = make_client_factory(STORE, POPULATION)
    if not a.yes and input(f"Run {', '.join(systems)} synchronously ({workers})? [y/N] ").strip().lower() != "y":
        return 0
    for p in range(1, a.passes + 1):             # a call answered by any provider is not asked again
        todo = [c for c in calls if stored_variant(clients[c.model].request(c.text, STMT), c.model)[1] is None]
        print(f"{now()} pass {p}: " + ", ".join(f"{m} {sum(c.model == m for c in todo)} to ask" for m in systems),
              flush=True)
        if not todo:
            break
        out = PROGRESS / f"pair_{datetime.datetime.now():%Y%m%d_%H%M%S}.parquet"
        df = execute(todo, factory, POPULATION, out, workers={"default": 1, **{m: workers[m] for m in systems}})
        print(df.groupby("model")["valid"].agg(["size", "mean"]).to_string(), flush=True)
        err = df[df.error.notna()]
        if len(err):
            print(f"errors (nothing stored; asked again next pass): {err.groupby('model').size().to_dict()}; e.g. "
                  f"{str(err.error.iloc[0])[:200]}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ Claude Sonnet 5
def sync(a) -> int:
    """Claude Sonnet 5 one request per call on the standard route (config/models.yaml: provider anthropic,
    allow_fallbacks false), --workers calls in flight; --part i/n takes every n-th call of the block order (disjoint
    parts for parallel processes). The requests equal the evaluation run's, so its stored answers are read."""
    use_key(a.key_env, "OPENROUTER_API_KEY")
    cl = LLMClient(SONNET, STORE, POPULATION)
    assert cl.batch is None
    i, n = (int(x) for x in a.part.split("/"))  # part i of n: every n-th call of the block order (disjoint parts)
    plan_sonnet = [c for c in plan() if c.model == SONNET]
    calls = [c for j, c in enumerate(plan_sonnet) if j % n == i]
    todo = [c for c in calls if not STORE.path(cl.request(c.text, STMT)).exists()]
    print(f"{now()} Claude Sonnet 5, standard route, part {i}/{n}: {len(calls)} of {len(plan_sonnet)} calls, "
          f"{len(calls) - len(todo)} stored, {len(todo)} to ask", flush=True)
    if not a.yes and input("Run Claude Sonnet 5 on the standard route? [y/N] ").strip().lower() != "y":
        return 0
    factory = make_client_factory(STORE, POPULATION)
    PROGRESS.mkdir(parents=True, exist_ok=True)
    for p in range(1, a.passes + 1):
        todo = [c for c in calls if not STORE.path(cl.request(c.text, STMT)).exists()]
        print(f"{now()} pass {p}: {len(todo)} to ask", flush=True)
        if not todo:
            break
        out = PROGRESS / f"sonnet_{datetime.datetime.now():%Y%m%d_%H%M%S}.parquet"
        df = execute(todo, factory, POPULATION, out, workers={"default": a.workers})
        print(df.groupby("model")["valid"].agg(["size", "mean"]).to_string(), flush=True)
        err = df[df.error.notna()]
        if len(err):
            print(f"errors (nothing stored; asked again next pass): {len(err)}; e.g. {str(err.error.iloc[0])[:200]}",
                  flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ calls table
class _ReadOnly:
    """A client that reads the store and never sends: a Hugging Face call without a stored answer is a row with error
    'no stored answer'; a batch call without one keeps LLMClient's 'batch result not collected'."""

    def __init__(self, cl, model: str):
        self.cl, self.model = cl, model

    def probability(self, text: str, statement: str) -> dict:
        req = self.cl.request(text, statement)
        if self.cl.transport == "hf_router":         # the stored provider's answer (configured provider first)
            prov, r = stored_variant(req, self.model)
            if r is None:
                return {"p": None, "valid": False, "provider": None, "model_reported": None, "usage": None,
                        "raw_path": str(STORE.path(req)), "error": NOT_STORED}
            cached = STORE.get(r)
            try:
                content = cached["response"]["choices"][0]["message"].get("content")
            except Exception:
                content = None
            out = self.cl._result(r, cached, cached["response"], content)
            out["provider"] = out.get("provider") or prov     # Hugging Face: the provider named in the request
            return out
        if self.cl.batch is None and not STORE.path(req).exists():
            return {"p": None, "valid": False, "provider": None, "model_reported": None, "usage": None,
                    "raw_path": str(STORE.path(req)), "error": NOT_STORED}
        return self.cl.probability(text, statement)


def build(a) -> int:
    """results/wide_cheaper/calls.parquet: one row per planned call (runner.execute's columns), read from the store
    with no call."""
    calls = plan()
    base = make_client_factory(STORE, POPULATION)
    ro = lambda model, repeat, variant="raw": _ReadOnly(base(model, repeat, variant), model)
    OUT.mkdir(parents=True, exist_ok=True)
    df = execute(calls, ro, POPULATION, OUT / "calls.parquet", workers=a.workers, checkpoint_every=10 ** 9)
    print(df.groupby("model")["valid"].agg(["size", "mean"]).to_string())
    print("providers reported:\n", df.groupby(["model", "provider"]).size().to_string())
    return 0


def analyse(a) -> int:
    """The wide analysis with these systems added: analysis.run_wide's prediction and group-calibration code on
    results/wide/calls.parquet (Jev and the primary chatbots) plus results/wide_cheaper/calls.parquet, the way
    analysis.run_eval includes the free and pair tiers: every system in the comparison with the outcome reference, the
    focal comparison (Jev against each system) over the primary chatbots only (focal_versus). The prediction block
    takes the records every included system scored, so a system is analysed only when every planned call has a stored
    answer; --leave_out names systems to leave out. Writes results/wide_cheaper/analysis.json (no call)."""
    import numpy as np
    from jevity import analysis as A
    from jevity.clients import BATCH_MISSING
    leave = {s for s in a.leave_out.split(",") if s}
    cheap = pd.read_parquet(OUT / "calls.parquet")
    systems = [m for m in SYSTEMS if m in set(cheap.model) and m not in leave]
    err = cheap["error"].fillna("").astype(str)
    open_ = cheap[cheap.model.isin(systems) & err.isin([NOT_STORED, BATCH_MISSING])]
    if len(open_):
        raise SystemExit(f"calls without a stored answer: {open_.model.value_counts().to_dict()}; build again when they "
                         f"finish, or --leave_out them")
    wide = pd.read_parquet(ROOT / "results" / "wide" / "calls.parquet")
    calls = pd.concat([wide, cheap[cheap.model.isin(systems)]], ignore_index=True)
    cohort = pd.read_parquet(ROOT / "data" / "cohort.parquet")
    o = pd.read_parquet(ROOT / "data" / "reference_oof.parquet").set_index("SEQN")
    y = cohort.set_index(cohort.SEQN.astype(int))["death_10y"]
    profiles = sorted(set(o.index))
    models = A.full_set_systems(A.ordered(set(calls.model)))
    preds = {}
    for m in models:        # as analysis.run_wide
        b = calls[(calls.model == m) & (calls.edit == "baseline") & calls.valid & A._variant(calls, "raw")]
        preds[m] = b.drop_duplicates("profile").set_index("profile")["p"].reindex(profiles)
    for col, name in (("p_spline_logit", "reference_spline_logit"), ("p_lightgbm", "reference_lightgbm"),
                      ("p_age_sex", "reference_age_sex"), ("p_phenoage_10y", "phenoage_10y")):
        if col in o:
            preds[name] = o[col].reindex(profiles)
    prim = [m for m in models if A.is_primary(m) and m != "jev"]
    yy = y.reindex(profiles).dropna()
    pred = A.prediction_block(preds, yy, B=a.B, focal_versus=prim)
    perf, gc = {}, {}
    for m in systems:       # each system on every record it scored (not only the records all systems scored)
        s = preds[m].dropna()
        s = s[s.index.isin(yy.index)]
        perf[m] = A.performance(s.to_numpy(), yy.loc[s.index].to_numpy())
        gc[m] = {"raw": A.group_calibration(s, cohort, B=a.B)}
    res = {"meta": {"when": now(), "models": models, "cheaper_systems": systems, "left_out": sorted(leave),
                    "n_records_with_outcome": int(len(yy)), "focal_versus": prim, "B": a.B,
                    "calls": ["results/wide/calls.parquet", "results/wide_cheaper/calls.parquet"],
                    "code": "analysis.run_wide (prediction, group calibration) with focal_versus as analysis.run_eval"},
           "prediction": pred, "baseline_performance": perf, "group_calibration": gc}
    write_json(OUT / "analysis.json", json.loads(json.dumps(res, default=lambda v: v.item() if isinstance(v, np.generic) else str(v))))
    ref = json.loads((ROOT / "results" / "wide" / "analysis.json").read_text())["prediction"]
    print(f"{pred['n_records']:,} records ({ref['n_records']:,} in results/wide/analysis.json), {pred['deaths']:,} deaths")
    for k, v in pred["systems"].items():
        same = ref["systems"].get(k, {}).get("log_loss")
        print(f"  {k:24s} log-loss {v['log_loss']:.4f}  recal {v['log_loss_recalibrated']:.4f}  AUROC {v['auroc']:.3f}  "
              f"slope {v['calibration_slope']:.2f}" + (f"  (wide {same:.4f})" if same is not None else ""))
    for k, v in pred.get("vs_reference", {}).get("pairs", {}).items():
        print(f"  {k:40s} diff log-loss {v['diff']:+.4f} {[round(x, 4) for x in v['ci_simultaneous']]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["submit", "collect", "sync", "pair", "build", "analyse"])
    ap.add_argument("--leave_out", default="", help="analyse: systems to leave out, e.g. medgemma")
    ap.add_argument("--B", type=int, default=1000, help="analyse: bootstrap draws (analyze.py wide: 1,000)")
    ap.add_argument("--key_env", default="OPENROUTER_API_KEY",
                    help="submit, collect, sync: the .env variable holding the OpenRouter key to send with")
    ap.add_argument("--part", default="0/1", help="sync, pair: part i/n of the calls (every n-th in block order)")
    ap.add_argument("--hf_token", default="HF_TOKEN",
                    help="pair: the .env variable holding the Hugging Face token to send with")
    ap.add_argument("--provider", default="", help="pair: alternative Hugging Face provider, e.g. gemma=deepinfra")
    ap.add_argument("--workers", type=int, default=16, help="sync: calls in flight; build: parallel store reads")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--chunk", default="",
                    help="submit: requests per batch by family, e.g. gemini_free=250 (default: config batch max_requests)")
    ap.add_argument("--hf_workers", default=HF_WORKERS, help="pair: calls in flight per system")
    ap.add_argument("--only", default=None, help="pair: medgemma or gemma alone")
    ap.add_argument("--passes", type=int, default=3, help="sync, pair: passes over calls without a stored answer")
    a = ap.parse_args()
    return {"submit": submit, "collect": collect, "sync": sync, "pair": pair, "build": build,
            "analyse": analyse}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())

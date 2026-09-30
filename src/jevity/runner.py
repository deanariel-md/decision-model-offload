"""Call plans and a resumable, threaded executor. Every call goes through a client whose RawStore caches the
verbatim response by request hash, so a crashed run resumes where it stopped and a finished run re-reads from disk."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import yaml

from .clients import JevClient, LLMClient, RawStore, PROMPTS, MODELS, ROOT, arm_systems


@dataclass(frozen=True)
class Call:
    model: str            # "jev", "jev_bands", "jev_complement", or a family name
    profile: int
    edit: str
    annotated: bool
    repeat: int
    text: str
    variant: str = "raw"


def plan_calls(states: pd.DataFrame, models: list[str], repeat_profiles: list, rep_edits: list[str], n_repeats: int,
               annotated_models: set[str], band_profiles: list | None = None, complement_profiles: list | None = None,
               m1_edits: list[str] | None = None, bundle_profiles: list | None = None) -> list[Call]:
    """Main pass for every model; repeats (baseline + representative edits) on repeat_profiles; annotated states only
    for annotated_models; Jev secondary questions on their subsets. scripts/run_calls.py then shuffles the plan with a
    blocks of records in a seeded random order (block_order), so a run cut short is a random subset of records with
    every system's calls for those records."""
    calls: list[Call] = []
    for m in models:
        for s in states.itertuples(index=False):
            if s.annotated and m not in annotated_models:
                continue
            calls.append(Call(m, s.profile, s.edit, bool(s.annotated), 0, s.text))
    rep = states[states.profile.isin(repeat_profiles) & states.edit.isin(rep_edits) & (~states.annotated)]
    for r in range(1, n_repeats):
        for m in models:
            for s in rep.itertuples(index=False):
                calls.append(Call(m, s.profile, s.edit, False, r, s.text))
    if band_profiles is not None:
        sub = states[states.profile.isin(band_profiles) & (~states.annotated)]
        calls += [Call("jev_bands", s.profile, s.edit, False, 0, s.text) for s in sub.itertuples(index=False)]
    if complement_profiles is not None:
        sub = states[states.profile.isin(complement_profiles) & (states.edit == "baseline") & (~states.annotated)]
        calls += [Call("jev_complement", s.profile, s.edit, False, 0, s.text) for s in sub.itertuples(index=False)]
    if bundle_profiles is not None:     # the same death question with the complement and the bands in one pass
        sub = states[states.profile.isin(bundle_profiles) & (states.edit == "baseline") & (~states.annotated)]
        calls += [Call("jev_bundle", s.profile, s.edit, False, 0, s.text) for s in sub.itertuples(index=False)]
    if m1_edits:
        sub = states[states.edit.isin(m1_edits) & (~states.annotated)]
        for m in models:
            calls += [Call(m, s.profile, s.edit, False, 0, s.text, "M1") for s in sub.itertuples(index=False)]
    return calls


REP_EDITS = ["baseline", "I1_field_order", "C1b_hba1c_plus2", "L1_race_nhw_to_nhb"]


def load_subset(path: Path | None = None) -> list[int] | None:
    """The 200 evaluation records of the reasoning arm and the framing sensitivity analysis (config/subsets.yaml,
    drawn from evaluation IDs only); None before it exists."""
    sub = MODELS.get("subset") or {}
    p = path or (ROOT / sub.get("file", "config/subsets.yaml"))
    if not p.exists():
        return None
    return [int(x) for x in yaml.safe_load(p.read_text(encoding="utf-8"))[sub.get("key", "eval200")]["profiles"]]


def plan_split(states: pd.DataFrame, split: str, pert: dict, subset: list[int] | None = None,
               systems: list[str] | None = None) -> list[Call]:
    """Every call for a split, in block order. Evaluation split, primary arm (Jev and the primary families): every
    state, with repeats, annotated states for Jev and one family (config annotated_family), Jev's secondary questions
    and M1. Secondary arms (config `arms`), main pass only: free and pair on every state, reasoning on the subset's
    states. Wide split: the primary arm only.
    `systems` narrows the plan to those calls.model values (Jev's questions are kept when 'jev' is listed).
    Split "framing" (a sensitivity analysis on the evaluation states): the primary chatbots with the system message
    without its study sentence (variant F1), on the subset's baselines and confirmatory-family edits."""
    if split == "framing":
        return block_order(_narrow(plan_framing(states, pert, subset), systems))
    profiles = sorted(states.profile.unique())
    if split == "wide":
        n_rep_prof, n_rep, rep_edits = 0, 1, []
    else:
        n_rep_prof, n_rep, rep_edits = 100, 3, REP_EDITS
    rng = np.random.default_rng(20260922)      # draw order: repeats, bands, complement
    rep_profiles = list(rng.choice(profiles, size=min(n_rep_prof, len(profiles)), replace=False)) if n_rep_prof else []
    band_profiles = None if split == "wide" else list(rng.choice(profiles, size=min(300, len(profiles)), replace=False))
    comp_profiles = None if split == "wide" else list(rng.choice(profiles, size=min(200, len(profiles)), replace=False))
    m1 = pert["mitigation"]["M1_instruction"]["edits"] if split == "eval" else None
    calls = plan_calls(states, ["jev"] + arm_systems("primary"), rep_profiles, rep_edits, n_rep,
                       annotated_models=set() if split == "wide" else {"jev", MODELS["annotated_family"]},
                       band_profiles=band_profiles, complement_profiles=comp_profiles, m1_edits=m1,
                       bundle_profiles=comp_profiles if split == "eval" else None)
    arms = [] if split == "wide" else (["free", "pair"] + (["reasoning"] if split == "eval" else []))
    raw = states[~states.annotated]
    for arm in arms:
        if MODELS["arms"][arm]["records"] == "subset":
            if subset is None:
                raise SystemExit(f"arm {arm} needs the subset (config/subsets.yaml, scripts/draw_subset.py)")
            st = raw[raw.profile.isin(set(subset))]
        else:
            st = raw
        calls += plan_calls(st, arm_systems(arm), [], [], 1, annotated_models=set())
    return block_order(_narrow(calls, systems))


def _narrow(calls: list[Call], systems: list[str] | None) -> list[Call]:
    if systems is None:
        return calls
    keep = set(systems)
    return [c for c in calls if c.model in keep or ("jev" in keep and c.model.startswith("jev"))]


def plan_framing(states: pd.DataFrame, pert: dict, subset: list[int] | None) -> list[Call]:
    """Framing sensitivity analysis: every active primary chatbot, variant F1, main pass only, on the subset's baseline
    and confirmatory-family states (evaluation split). Answers go to their own store (runs/framing)."""
    if subset is None:
        raise SystemExit("the framing sensitivity analysis needs the subset (config/subsets.yaml)")
    edits = ["baseline"] + list(pert["confirmatory_family"])
    st = states[(~states.annotated) & states.profile.isin(set(int(p) for p in subset)) & states.edit.isin(edits)]
    return [Call(m, int(s.profile), s.edit, False, 0, s.text, "F1") for m in arm_systems("primary")
            for s in st.itertuples(index=False)]


def _num(v):
    try:
        return float(v) if v is not None and not isinstance(v, bool) else None
    except (TypeError, ValueError):
        return None


def block_order(calls: list, seed: int = 20260922, block: int = 50) -> list:
    """Records in a seeded random order, grouped in blocks of `block`; every call for a block (all systems, versions,
    repeats) precedes the next block, in a seeded random order within it. The batch route submits in this order, so a
    spending stop leaves complete blocks for every system."""
    profiles = sorted({c.profile for c in calls})
    rank = {p: i for i, p in enumerate(np.random.default_rng(seed).permutation(profiles))}
    tie = np.random.default_rng(seed + 1).permutation(len(calls))
    return [calls[i] for i in sorted(range(len(calls)), key=lambda i: (rank[calls[i].profile] // block, tie[i]))]


def make_client_factory(store: RawStore, population: str, simulate: Callable | None = None):
    lock = threading.Lock()
    cache: dict = {}

    def get(model: str, repeat: int, variant: str = "raw"):
        if simulate is not None:
            return simulate(model, repeat, variant)
        key = (model, repeat, variant)
        with lock:
            if key not in cache:
                if model == "jev":
                    cache[key] = JevClient(store, population, repeat=repeat, variant=variant)
                elif model == "jev_bands":
                    cache[key] = JevClient(store, population, repeat=repeat, question="risk_bands")
                elif model == "jev_complement":
                    cache[key] = JevClient(store, population, repeat=repeat, question="complement")
                elif model == "jev_bundle":
                    cache[key] = JevClient(store, population, repeat=repeat, question="bundle")
                else:        # a family, or 'family@effort' for the reasoning arm
                    cache[key] = LLMClient(model, store, population, repeat=repeat, variant=variant)
            return cache[key]
    return get


def execute(calls: list[Call], client_for, population: str, out: Path, workers: int | dict = 8,
            checkpoint_every: int = 200) -> pd.DataFrame:
    """Run every call; `workers` = parallel calls PER SYSTEM (an int, or {system: n} with key 'default')."""
    stmt = PROMPTS["statements"][population]
    rows: list[dict] = []
    lock = threading.Lock()

    def one(c: Call) -> dict:
        try:
            r = client_for(c.model, c.repeat, c.variant).probability(c.text, stmt)
        except Exception as e:  # recorded, never silently dropped
            r = {"p": None, "valid": False, "provider": None, "model_reported": None, "raw_path": None, "error": repr(e)[:300]}
        return {"model": c.model, "profile": c.profile, "edit": c.edit, "annotated": c.annotated, "repeat": c.repeat, "variant": c.variant,
                "p": r.get("p"), "valid": bool(r.get("valid")), "provider": str(r.get("provider")),
                "model_reported": str(r.get("model_reported")), "error": r.get("error"), "raw_path": r.get("raw_path"),
                "route": r.get("route"), "parse": r.get("parse"), "finish_reason": r.get("finish_reason"),
                "response_id": r.get("response_id"), "confidence": _num(r.get("confidence")),
                "latency_s": _num(r.get("latency_s")), "p_alive": _num(r.get("p_alive")), "p_bands": _num(r.get("p_bands")),
                "tokens_in": _num(_usage(r, "prompt_tokens", "input_tokens")),       # Jev reports input_tokens
                "tokens_out": _num(_usage(r, "completion_tokens", "output_tokens")),
                "band_probabilities": json.dumps(r["band_probabilities"]) if r.get("band_probabilities") else None}

    # one pool per system, all running at once: a slow system (long reasoning) never holds the others' slots; within a
    # system the calls keep their seeded block order
    by_system: dict[str, list[Call]] = {}
    for c in calls:
        by_system.setdefault(c.model, []).append(c)
    per = workers if isinstance(workers, dict) else {"default": workers}
    pools = {m: ThreadPoolExecutor(max_workers=per.get(m, per["default"]), thread_name_prefix=m) for m in by_system}
    try:
        futs = [pools[m].submit(one, c) for m, cs in by_system.items() for c in cs]
        for i, f in enumerate(as_completed(futs), 1):
            row = f.result()
            with lock:
                rows.append(row)
                if i % checkpoint_every == 0:
                    _write(pd.DataFrame(rows), out, tries=1)     # a checkpoint that cannot be written is skipped
                    print(f"{i}/{len(calls)} calls")
    finally:
        for p in pools.values():
            p.shutdown(wait=True)
    df = pd.DataFrame(rows)
    _write(df, out)
    return df


def _usage(r: dict, *keys: str):
    u = r.get("usage") or {}
    return next((u[k] for k in keys if u.get(k) is not None), None)


def _write(df: pd.DataFrame, out: Path, tries: int = 10) -> bool:
    """Write calls.parquet; on Windows a write can fail with EINVAL/EACCES while another handle (e.g. on-access scanning)
    holds the file, so the final write is retried. The answers themselves live in the raw store."""
    import time
    for k in range(tries):
        try:
            df.to_parquet(out, index=False)
            return True
        except OSError:
            if k == tries - 1:
                if tries == 1:
                    return False
                raise
            time.sleep(0.5 * (k + 1))
    return False

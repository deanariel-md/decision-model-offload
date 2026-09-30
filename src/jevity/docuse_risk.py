"""Ten-year risk, Jev used as documented.

The same statement and question as Jev's original primary question (config/prompts.yaml, read through JevClient); only
the record changes: it goes to Jev as a JSON object in `state`, with every numeric field turned into a named band in
code (config/docuse_risk.yaml holds every band, cut point and source). The banded record is built from the original
record (serialize.to_state of the same cohort row), and the builder checks that this original record is, character for
character, the text the original run sent (data/states_wide.parquet).

Transport, raw-output store, call plan order and the calls table are the repository's own (clients.JevClient,
clients.RawStore, runner.block_order, runner.execute); scoring is analysis.prediction_block with the settings of
results/wide/analysis.json. No function here sends anything unless a caller passes a live store to `run`."""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import analysis as A
from .clients import MODELS, PROMPTS, ROOT, JevClient, RawStore, estimate_tokens
from .runner import Call, block_order, execute
from .serialize import to_json, to_state

CONFIG_PATH = ROOT / "config" / "docuse_risk.yaml"
POPULATION = "nhanes"


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))


CFG = load_config()
STATEMENT = PROMPTS["statements"][CFG["question"]["statement_key"]]
VARIANT = CFG["variant"]


def config_sha256(path: Path = CONFIG_PATH) -> str:
    """sha256 (lowercase hex) of the config text with LF line ends, so any checkout gives the same value."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------------ banding
def band(value: float, bands: list[dict], scale: float | None = None, decimals: int | None = None) -> str:
    """The label of the first band that matches (`below`: value < x; `at_most`: value <= x; neither: the rest)."""
    v = float(value) * (scale if scale is not None else 1.0)
    if decimals is not None:
        v = round(v, int(decimals))
    if not math.isfinite(v):
        raise ValueError(f"non-finite value {value!r}")
    for b in bands:
        if "below" in b:
            if v < float(b["below"]):
                return b["label"]
        elif "at_most" in b:
            if v <= float(b["at_most"]):
                return b["label"]
        else:
            return b["label"]
    raise ValueError(f"no band for {value!r} (bands need a last band without a cut point)")


def egfr_ckd_epi_2021(creatinine_mg_dl: float, age: float, sex: str, eq: dict | None = None) -> float:
    """Race-free CKD-EPI 2021 creatinine equation (Inker et al., NEJM 2021); constants from config."""
    eq = eq or CFG["fields"]["kidney_function"]["equation"]
    s = eq[sex]
    r = float(creatinine_mg_dl) / s["kappa"]
    return (eq["constant"] * min(r, 1.0) ** s["alpha"] * max(r, 1.0) ** eq["exponent_above"]
            * eq["age_base"] ** float(age) * s["factor"])


def bp_category(systolic: float, diastolic: float, spec: dict | None = None) -> str:
    """AHA/ACC 2017 category from both readings: the highest category either reading reaches."""
    spec = spec or CFG["fields"]["blood_pressure"]
    for c in spec["categories"]:
        s_min, d_min = c.get("systolic_at_least"), c.get("diastolic_at_least")
        if s_min is None and d_min is None:
            return c["label"]
        if (s_min is not None and systolic >= s_min) or (d_min is not None and diastolic >= d_min):
            return c["label"]
    raise ValueError("blood pressure categories need a last category without a cut point")


def _number(v: Any, top_coded: dict | None = None) -> float | None:
    """A value of the original record as a number: {'value': x, 'unit': u}, a bare number, or a top-code text."""
    if v is None:
        return None
    if isinstance(v, dict):
        v = v.get("value")
    if isinstance(v, str):
        if top_coded and v in top_coded:
            return float(top_coded[v])
        raise ValueError(f"text value {v!r} has no top_coded entry")
    if isinstance(v, bool):
        raise ValueError("boolean where a number was expected")
    return float(v)


def field_labels(name: str, cfg: dict = CFG) -> set[str]:
    f = cfg["fields"][name]
    if f.get("kind") == "blood_pressure":
        return {c["label"] for c in f["categories"]}
    if f.get("by_sex"):
        return {b["label"] for bs in f["bands"].values() for b in bs}
    return {b["label"] for b in f["bands"]}


def band_field(name: str, orig: dict, cfg: dict = CFG) -> str | None:
    """The band of one configured field for an original state, or None when an input is missing (the field is then
    left out, as the original serializer leaves out a missing value)."""
    f = cfg["fields"][name]
    kind = f.get("kind")
    if kind == "blood_pressure":
        s, d = (_number(orig.get(k)) for k in f["from"])
        return None if s is None or d is None else bp_category(s, d, f)
    if kind == "egfr":
        scr, age, sex = _number(orig.get(f["from"][0])), _number(orig.get("age")), orig.get("sex")
        if scr is None or age is None or sex not in f["equation"]:
            return None
        return band(egfr_ckd_epi_2021(scr, age, sex, f["equation"]), f["bands"])
    v = _number(orig.get(f["from"][0]), f.get("top_coded"))
    if v is None:
        return None
    bands = f["bands"]
    if f.get("by_sex"):
        if orig.get("sex") not in bands:
            return None
        bands = bands[orig["sex"]]
    return band(v, bands, f.get("scale"), f.get("decimals"))


def banded_state(orig: dict, cfg: dict = CFG) -> "OrderedDict[str, Any]":
    """The documented state from an original state (serialize.to_state): kept fields unchanged, each numeric field
    replaced by its band at the place of its (first) source field. Every original field must be kept or banded."""
    kept = set(cfg["state"]["kept_fields"])
    source_of = {src: name for name, f in cfg["fields"].items() for src in f["from"]}
    out: OrderedDict[str, Any] = OrderedDict()
    done: set[str] = set()
    for k, v in orig.items():
        if k in kept:
            out[k] = v
        elif k in source_of:
            name = source_of[k]
            if name in done:
                continue
            done.add(name)
            lab = band_field(name, orig, cfg)
            if lab is not None:
                out[cfg["fields"][name]["key"]] = lab
        else:
            raise KeyError(f"original field {k!r} is neither kept nor banded (config/docuse_risk.yaml)")
    return out


def state_from_row(rec: dict, cfg: dict = CFG) -> "OrderedDict[str, Any]":
    return banded_state(to_state(rec), cfg)


# ------------------------------------------------------------------------------------------------------ records
def wide_profiles(root: Path = ROOT) -> list[int]:
    """The 12,910 records of the wide baseline run: Jev's baseline rows in results/wide/calls.parquet, checked
    against data/states_wide.parquet."""
    c = pd.read_parquet(root / "results" / "wide" / "calls.parquet", columns=["model", "profile", "edit"])
    prof = sorted(int(p) for p in c.loc[(c.model == "jev") & (c.edit == "baseline"), "profile"].unique())
    s = pd.read_parquet(root / "data" / "states_wide.parquet", columns=["profile", "edit"])
    sp = sorted(int(p) for p in s.loc[s.edit == "baseline", "profile"].unique())
    if prof != sp:
        raise SystemExit("results/wide/calls.parquet baseline records differ from data/states_wide.parquet")
    return prof


def build_states(profiles: list[int] | None = None, root: Path = ROOT, cfg: dict = CFG) -> pd.DataFrame:
    """One row per record: profile, the documented state (dict) and the original text. The original record is
    rebuilt from data/cohort.parquet with serialize.to_state and must equal data/states_wide.parquet's baseline text."""
    cohort = pd.read_parquet(root / "data" / "cohort.parquet")
    sw = pd.read_parquet(root / "data" / "states_wide.parquet")
    base = sw[(sw.edit == "baseline") & (~sw.annotated)].drop_duplicates("profile").set_index("profile")["text"]
    profiles = sorted(base.index.astype(int)) if profiles is None else [int(p) for p in profiles]
    rows = cohort.assign(_p=cohort.SEQN.astype(int)).set_index("_p")
    out, bad = [], []
    for p in profiles:
        orig = to_state(rows.loc[p].to_dict())
        text = to_json(orig)
        if base.get(p) != text:
            bad.append(p)
            continue
        out.append({"profile": p, "state": banded_state(orig, cfg), "original_text": text})
    if bad:
        raise SystemExit(f"{len(bad)} records do not rebuild to the text the original run sent (first {bad[:5]})")
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------------ requests
def client(store: RawStore | None, repeat: int = 0) -> JevClient:
    """Jev exactly as the original run called it: the primary question, config/models.yaml slug and provider block."""
    return JevClient(store, POPULATION, repeat=repeat)


def make_request(state: dict) -> dict:
    return client(None).request(state, STATEMENT)


STATE_SLOT = "{state}"


def request_body(req: dict) -> dict:
    """What is posted (JevClient._send: every key except the cache keys that start with '_')."""
    return {k: v for k, v in req.items() if not k.startswith("_")}


def template() -> dict:
    """The one request every record is sent in, with the record replaced by the string "{state}"."""
    return request_body(make_request(STATE_SLOT))


def state_key_order(cfg: dict = CFG) -> list[str]:
    """Every key a documented state can hold, in its fixed order: the original serializer's order (serialize.to_state,
    _field_rank), each banded field at the place of its first source field."""
    from .serialize import CATEGORICAL_FIELDS, NUMERIC_FIELDS, _field_rank
    orig = sorted(["record_type", "source", *[k for k, _ in CATEGORICAL_FIELDS], *[k for k, *_ in NUMERIC_FIELDS],
                   "doctor_diagnosed_conditions"], key=_field_rank)
    kept = set(cfg["state"]["kept_fields"])
    source_of = {src: f["key"] for f in cfg["fields"].values() for src in f["from"]}
    out: list[str] = []
    for k in orig:
        key = k if k in kept else source_of[k]
        if key not in out:
            out.append(key)
    return out


def field_of_key(cfg: dict = CFG) -> dict[str, str]:
    return {f["key"]: n for n, f in cfg["fields"].items()}


def label_lists(name: str, cfg: dict = CFG) -> dict[str | None, list[str]]:
    """A field's labels in config order: {None: labels}, or {sex: labels} for a by_sex field."""
    f = cfg["fields"][name]
    if f.get("kind") == "blood_pressure":
        return {None: [c["label"] for c in f["categories"]]}
    if f.get("by_sex"):
        return {sex: [b["label"] for b in bs] for sex, bs in f["bands"].items()}
    return {None: [b["label"] for b in f["bands"]]}


def band_distribution(states: pd.DataFrame, cfg: dict = CFG) -> dict:
    """Records per label for every banded field (config order; `missing`: the field is not in the state), and per sex
    for a by_sex field."""
    out = {}
    for name, f in cfg["fields"].items():
        key = f["key"]
        vals = states.state.map(lambda s: s.get(key))
        labs = list(dict.fromkeys(l for ls in label_lists(name, cfg).values() for l in ls))
        d = {"key": key, "counts": {l: int((vals == l).sum()) for l in labs}, "missing": int(vals.isna().sum())}
        if f.get("by_sex"):
            sex = states.state.map(lambda s: s.get("sex"))
            d["by_sex"] = {sx: {l: int(((vals == l) & (sex == sx)).sum()) for l in ls}
                           for sx, ls in label_lists(name, cfg).items()}
        out[name] = d
    return out


def kept_values(states: pd.DataFrame, cfg: dict = CFG) -> dict:
    """The values each kept field takes over the given records (first-seen order sorted by text; objects by sub-key)."""
    out: dict = {}
    for k in cfg["state"]["kept_fields"]:
        vals = [s[k] for s in states.state if k in s]
        if any(isinstance(v, dict) for v in vals):
            sub: dict = {}
            for v in vals:
                for kk, vv in v.items():
                    sub.setdefault(kk, set()).add(vv)
            out[k] = {kk: sorted(vv, key=str) for kk, vv in sub.items()}
        else:
            out[k] = sorted(set(vals), key=str)
    return out


def _numbers_in(x) -> list:
    if isinstance(x, dict):
        return [n for v in x.values() for n in _numbers_in(v)]
    if isinstance(x, (list, tuple)):
        return [n for v in x for n in _numbers_in(v)]
    return [x] if isinstance(x, (int, float)) and not isinstance(x, bool) else []


def validate_request(req: dict, cfg: dict = CFG) -> list[str]:
    """Problems with one documented request (empty list = valid)."""
    errs = []
    ref = client(None).request("x", STATEMENT)          # the original request shape with a text state
    for k in ("model", "questions", "provider", "_repeat"):
        if req.get(k) != ref.get(k):
            errs.append(f"{k} differs from the original Jev request: {req.get(k)!r}")
    if req.get("model") != MODELS["jev"]["slug"]:
        errs.append("model is not config/models.yaml jev.slug")
    if (req.get("provider") or {}).get("allow_fallbacks") is not False:
        errs.append("allow_fallbacks is not false")
    if set(req) != set(ref):
        errs.append(f"request keys {sorted(req)} differ from the original {sorted(ref)}")
    st = req.get("state")
    if not isinstance(st, dict):
        return errs + ["state is not a JSON object"]
    if {**request_body(req), "state": STATE_SLOT} != template():
        errs.append("the request differs from the template in more than the state")
    order = state_key_order(cfg)
    if [k for k in order if k in st] != list(st):
        errs.append("state keys are not in the fixed order")
    kept = set(cfg["state"]["kept_fields"])
    keys = field_of_key(cfg)
    for k, v in st.items():
        if k in keys:
            if v not in field_labels(keys[k], cfg):
                errs.append(f"{k}: {v!r} is not one of its bands")
        elif k not in kept:
            errs.append(f"unexpected state field {k!r}")
    if _numbers_in(st):
        errs.append(f"numbers left in the state: {_numbers_in(st)[:3]}")
    try:
        json.loads(json.dumps({k: v for k, v in req.items() if not k.startswith("_")}, ensure_ascii=False))
    except (TypeError, ValueError) as e:
        errs.append(f"not JSON-serialisable: {e}")
    return errs


def body_tokens(req: dict) -> int:
    """Estimated input tokens of what is sent (clients.estimate_tokens: 4 characters per token)."""
    return estimate_tokens(json.dumps({k: v for k, v in req.items() if not k.startswith("_")}, ensure_ascii=False))


def plan(states: pd.DataFrame) -> list[Call]:
    """One call per record, in the repository's seeded block order (blocks of 50 records)."""
    return block_order([Call("jev", int(r.profile), "baseline", False, 0, r.state, VARIANT)
                        for r in states.itertuples(index=False)])


def dry_run(states: pd.DataFrame, root: Path = ROOT, cfg: dict = CFG) -> dict:
    """Build and validate every request without sending; token and list-cost estimates, calibrated on the original
    run's reported input tokens for the same records (actual / estimated for the original requests)."""
    calls = plan(states)
    reqs = [make_request(c.text) for c in calls]
    errors = {c.profile: e for c, r in zip(calls, reqs) if (e := validate_request(r, cfg))}
    from .clients import _sha
    hashes = {_sha(r) for r in reqs}
    tok = np.array([body_tokens(r) for r in reqs], float)
    orig = states.set_index("profile")["original_text"]
    tok0 = np.array([body_tokens(client(None).request(orig[c.profile], STATEMENT)) for c in calls], float)
    w = pd.read_parquet(root / "results" / "wide" / "calls.parquet", columns=["model", "profile", "edit", "tokens_in"])
    act = w[(w.model == "jev") & (w.edit == "baseline")].drop_duplicates("profile").set_index("profile")["tokens_in"]
    act = act.reindex([c.profile for c in calls])
    ok = act.notna().to_numpy()
    ratio = float(act.to_numpy()[ok].sum() / tok0[ok].sum()) if ok.any() else 1.0
    price = float(MODELS["jev"]["price_per_mtok_input"])
    est = float(tok.sum() * ratio)
    return {"n_requests": len(reqs), "n_unique_requests": len(hashes), "n_invalid": len(errors),
            "invalid": {str(k): v for k, v in list(errors.items())[:20]},
            "tokens_estimated_chars_over_4": float(tok.sum()), "mean_tokens_chars_over_4": float(tok.mean()),
            "original_tokens_reported_over_estimated": ratio, "tokens_estimated_calibrated": est,
            "mean_tokens_calibrated": est / len(reqs), "original_mean_tokens_reported": float(act.mean()),
            "price_per_mtok_input": price, "usd_list_estimated": est * price / 1e6,
            "model": reqs[0]["model"] if reqs else None, "provider": reqs[0].get("provider") if reqs else None,
            "questions": reqs[0]["questions"] if reqs else None, "example_state": dict(calls[0].text) if calls else None,
            "config_sha256": config_sha256(), "template": template(), "band_distribution": band_distribution(states, cfg)}


# ------------------------------------------------------------------------------------------------------ running
class StoredJev(JevClient):
    """Reads answers from the raw store and never sends: a record without a stored answer is an error row."""

    def _send(self, req: dict) -> dict:
        raise LookupError("no stored answer")


def _factory(store: RawStore, cls=JevClient):
    cache: dict = {}

    def get(model: str, repeat: int, variant: str = "raw"):
        assert model == "jev"
        if repeat not in cache:
            cache[repeat] = cls(store, POPULATION, repeat=repeat)
        return cache[repeat]
    return get


def calls_table(states: pd.DataFrame, store: RawStore, out: Path, workers: int = 8) -> pd.DataFrame:
    """The calls table from the stored responses (columns of results/wide/calls.parquet; model jev, variant
    documented, edit baseline, repeat 0), through runner.execute with a client that only reads the store."""
    df = execute(plan(states), _factory(store, StoredJev), POPULATION, Path(out), workers=workers,
                 checkpoint_every=10 ** 9)
    df = df.sort_values("profile").reset_index(drop=True)
    from .runner import _write
    _write(df, Path(out))
    return df


def chunks(calls: list[Call], n_records: int, block: int = 50) -> list[list[Call]]:
    """Consecutive runs of whole 50-record blocks (the plan is in block order), about n_records records each."""
    per = max(block, (n_records // block) * block)
    order = list(dict.fromkeys(c.profile for c in calls))
    rank = {p: i // per for i, p in enumerate(order)}
    out: dict[int, list[Call]] = {}
    for c in calls:
        out.setdefault(rank[c.profile], []).append(c)
    return [out[k] for k in sorted(out)]


def run(states: pd.DataFrame, store: RawStore, progress_dir: Path, stop_usd: float | None, workers: int = 16,
        chunk_records: int = 1000, factory=None) -> dict:
    """Send every planned call not yet answered (answers already in the store are read, not resent), chunk by chunk.
    Stops between chunks when the list spend of the answers so far passes stop_usd (None: no stop) or a call was
    refused for credit (HTTP 402). Returns a summary; the calls table is built afterwards from the store
    (calls_table)."""
    import time
    progress_dir = Path(progress_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)
    price = float(MODELS["jev"]["price_per_mtok_input"])
    factory = factory or _factory(store)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    spent, n_done, stopped = 0.0, 0, None
    parts = chunks(plan(states), chunk_records)
    for k, part in enumerate(parts):
        df = execute(part, factory, POPULATION, progress_dir / f"chunk_{stamp}_{k:03d}.parquet", workers=workers)
        spent += float((df.tokens_in.fillna(0) * price / 1e6).sum())
        n_done += len(df)
        n402 = int(df.error.fillna("").astype(str).str.contains("402").sum())
        print(f"chunk {k + 1}/{len(parts)}: {len(df)} calls, usable {int(df.valid.sum())}, list spend so far ${spent:.4f}",
              flush=True)
        if n402:
            stopped = f"{n402} calls refused for credit (HTTP 402)"
        elif stop_usd is not None and spent > stop_usd:
            stopped = f"list spend ${spent:.4f} above the ${stop_usd:.2f} run stop"
        if stopped:
            print(f"STOP: {stopped}; answers so far stay in the store", flush=True)
            break
    return {"chunks_run": k + 1 if parts else 0, "chunks": len(parts), "calls": n_done, "usd_list": spent,
            "stopped": stopped}


# ------------------------------------------------------------------------------------------------------ analysis
def wide_predictions(calls: pd.DataFrame, oof: pd.DataFrame, jev: pd.Series | None = None,
                     name: str = "jev") -> tuple[dict, list[int]]:
    """Predictions as analysis.run_wide builds them (its systems, then the cross-fitted references); `jev` replaces
    Jev's original answers, under key `name`."""
    models = A.full_set_systems(A.ordered(set(calls.model)))
    o = oof.set_index("SEQN")
    profiles = sorted(set(o.index))
    preds = {}
    for m in models:
        if m == "jev" and jev is not None:
            preds[name] = jev.reindex(profiles)
            continue
        b = calls[(calls.model == m) & (calls.edit == "baseline") & calls.valid & A._variant(calls, "raw")]
        preds[m] = b.drop_duplicates("profile").set_index("profile")["p"].reindex(profiles)
    for col, nm in (("p_spline_logit", "reference_spline_logit"), ("p_lightgbm", "reference_lightgbm"),
                    ("p_age_sex", "reference_age_sex"), ("p_phenoage_10y", "phenoage_10y")):
        if col in o:
            preds[nm] = o[col].reindex(profiles)
    return preds, profiles


def eval_predictions(calls: pd.DataFrame, cohort: pd.DataFrame, oof: pd.DataFrame, root: Path = ROOT,
                     jev: pd.Series | None = None, name: str = "jev") -> tuple[dict, list[int], str]:
    """Baseline predictions on the evaluation split as scripts/analyze.py builds them (analysis.baseline_predictions
    and the fitted references). Without data/reference_*.joblib the references come from data/reference_oof.parquet,
    whose evaluation rows are the full-fit predictions (scripts/crossfit_reference.py); the survival reference then
    has no column."""
    profiles = sorted(cohort.loc[cohort.split == "eval", "SEQN"].astype(int))
    ev = cohort[cohort.split == "eval"].set_index(cohort.loc[cohort.split == "eval", "SEQN"].astype(int))
    comps, source = {}, "joblib"
    names = ("spline_logit", "lightgbm", "survival_logit", "age_sex")
    if all((root / "data" / f"reference_{n}.joblib").exists() for n in names):
        import joblib
        for n in names:
            comps[f"reference_{n}"] = pd.Series(joblib.load(root / "data" / f"reference_{n}.joblib").predict(ev), index=ev.index)
        from .reference import phenoage_mortality_10y
        comps["phenoage_10y"] = pd.Series(np.clip(phenoage_mortality_10y(ev).to_numpy(), 1e-6, 1 - 1e-6), index=ev.index)
    else:
        source = "reference_oof"
        o = oof.set_index("SEQN")
        for col, nm in (("p_spline_logit", "reference_spline_logit"), ("p_lightgbm", "reference_lightgbm"),
                        ("p_age_sex", "reference_age_sex"), ("p_phenoage_10y", "phenoage_10y")):
            comps[nm] = o[col].reindex(profiles)
    preds = A.baseline_predictions(calls, A.ordered(set(calls.model)), profiles, comps)
    if jev is not None:
        preds = {(name if k == "jev" else k): (jev.reindex(profiles) if k == "jev" else v) for k, v in preds.items()}
    return preds, profiles, source


def common_records(preds: dict, y: pd.Series) -> list[int]:
    """The records every system scored (prediction_block's record set)."""
    names = [n for n, s in preds.items() if s is not None]
    return sorted(set.intersection(*[set(preds[n].dropna().index) for n in names]) & set(y.index))


def cheap_family(preds: dict, y: pd.Series, focal: str, versus: list[str], margin: float, B: int, seed: int,
                 label: str = "Jev") -> dict:
    """Focal system against the five cheaper systems on log-loss, the procedure of results/summaries/cheap_family.json
    (risk_evaluation): the records every system scored, per-record log-loss clipped as analysis._ll, one set of B record
    resamples from default_rng(seed), max-t simultaneous intervals around the point differences (standard deviations
    of the resampled differences, ddof 0); non-inferior when the upper simultaneous bound is below the margin, worse
    when the lower bound is above it, inconclusive otherwise. scripts/analyze_risk.py checks that this function
    reproduces the stored block before it scores anything else."""
    common = common_records(preds, y)
    yy = y.loc[common].to_numpy(float)
    ll = {k: A._ll(np.clip(preds[k].loc[common].to_numpy(float), *A.CLIP), yy) for k in [focal] + list(versus)}
    n = len(common)
    rng = np.random.default_rng(seed)
    C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)
    D = np.column_stack([C @ (ll[focal] - ll[k]) / n for k in versus])
    pts = np.array([float((ll[focal] - ll[k]).mean()) for k in versus])
    se = D.std(0)
    crit = float(np.quantile(np.max(np.abs(D - pts) / se, axis=1), 0.95))
    comps = {}
    for j, k in enumerate(versus):
        lo, hi = float(pts[j] - crit * se[j]), float(pts[j] + crit * se[j])
        ni, worse = bool(hi < margin), bool(lo > margin)
        comps[k] = {"diff": float(pts[j]), "ci": A._ci(D[:, j]), "ci_simultaneous": [lo, hi], "noninferior": ni,
                    "worse": worse, "outcome": "non-inferior" if ni else ("worse" if worse else "inconclusive")}
    return {"n_records": n, "deaths": int(yy.sum()), "margin": float(margin), "critical_value": crit, "B": int(B),
            "seed": int(seed), "note": f"log-loss, lower is better; difference {label} minus system", "comparisons": comps}


def documented_answers(calls_doc: pd.DataFrame) -> pd.Series:
    """Documented Jev's usable answer per record (first valid row, as the original analysis takes them)."""
    b = calls_doc[(calls_doc.model == "jev") & (calls_doc.edit == "baseline") & (calls_doc.variant == VARIANT)
                  & (calls_doc.repeat == 0) & calls_doc.valid]
    return b.drop_duplicates("profile").set_index("profile")["p"].astype(float)


def compare(got, ref, tol: float, path: str = "", skip=lambda p: False) -> tuple[list[str], float, bool]:
    """Differences between two JSON trees: structure, text and booleans exactly; numbers within tol x max(1, |ref|).
    Returns (differences, largest absolute numeric difference, bit-exact)."""
    diffs, worst, exact = [], 0.0, True

    def walk(a, b, p):
        nonlocal worst, exact
        if skip(p):
            return
        if isinstance(b, dict):
            keep = lambda d: {k for k in d if not skip(f"{p}/{k}")}
            if not isinstance(a, dict) or keep(a) != keep(b):
                diffs.append(f"{p}: keys differ"); return
            for k in keep(b):
                walk(a[k], b[k], f"{p}/{k}")
        elif isinstance(b, list):
            if not isinstance(a, list) or len(a) != len(b):
                diffs.append(f"{p}: length differs"); return
            for i, (x, z) in enumerate(zip(a, b)):
                walk(x, z, f"{p}[{i}]")
        elif isinstance(b, bool) or isinstance(a, bool) or b is None or isinstance(b, str):
            if a != b:
                diffs.append(f"{p}: {a!r} != {b!r}")
        elif isinstance(b, (int, float)):
            if not isinstance(a, (int, float)):
                diffs.append(f"{p}: {a!r} is not a number"); return
            d = abs(float(a) - float(b))
            worst = max(worst, d)
            exact = exact and a == b
            if d > tol * max(1.0, abs(float(b))):
                diffs.append(f"{p}: {a!r} != {b!r}")
        elif a != b:
            diffs.append(f"{p}: {a!r} != {b!r}")
    walk(got, ref, path)
    return diffs, worst, exact and not diffs


def jsonable(o):
    """JSON-ready copy (numpy scalars and arrays to Python; non-finite floats to null), as scripts/analyze.py."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


MAIN = ["gpt", "claude", "gemini", "muse", "glm"]      # the five main chatbots (active primary families)


def _survival_path(p: str) -> bool:
    """Paths of the stored evaluation block that depend on the survival reference (absent without the joblib files):
    its own entries and, through the family's critical value, every simultaneous bound of the vs-reference families."""
    last = p.rsplit("/", 1)[-1]
    if "reference_survival_logit" in p:
        return True
    return p.startswith("/vs_reference") and (last == "critical_value" or "ci_simultaneous" in p or last == "noninferior")


def inputs(root: Path = ROOT) -> dict:
    return {"wide_calls": pd.read_parquet(root / "results" / "wide" / "calls.parquet"),
            "eval_calls": pd.read_parquet(root / "results" / "eval" / "calls.parquet"),
            "cohort": pd.read_parquet(root / "data" / "cohort.parquet"),
            "oof": pd.read_parquet(root / "data" / "reference_oof.parquet"),
            "wide_json": json.loads((root / "results" / "wide" / "analysis.json").read_text(encoding="utf-8")),
            "eval_json": json.loads((root / "results" / "eval" / "analysis.json").read_text(encoding="utf-8")),
            "cheap_json": json.loads((root / "results" / "summaries" / "cheap_family.json").read_text(encoding="utf-8"))}


def fixed_margin_outcomes(block: dict, margin: float) -> dict:
    """Re-derive every non-inferiority outcome of a prediction_block result at a fixed margin, from its stored bounds:
    beside prediction_block's own `noninferiority_margin` (10% of the gap on the records scored) and `noninferior`,
    each pair gets `noninferiority_margin_fixed` and `noninferior_fixed_margin` (upper simultaneous bound below the
    fixed margin; the same comparison prediction_block makes). The block is changed in place and returned."""
    for fam in block.values():
        if isinstance(fam, dict) and "pairs" in fam:
            for pr in fam["pairs"].values():
                if "noninferiority_margin" in pr:
                    pr["noninferiority_margin_fixed"] = float(margin)
                    pr["noninferior_fixed_margin"] = bool(pr["ci_simultaneous"][1] < margin)
    return block


def _close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def reproduce(inp: dict, root: Path = ROOT, cfg: dict = CFG) -> dict:
    """Run the scoring code on Jev's ORIGINAL answers and check it against the stored results: (1) analysis.run_wide
    against results/wide/analysis.json (whole file); (2) the prediction inputs built here give run_wide's prediction
    block exactly; (3) the evaluation prediction block against results/eval/analysis.json; (4) cheap_family, at the
    fixed evaluation margin, against results/summaries/cheap_family.json risk_evaluation; (5) the margins recomputed on
    these records (10% of the age-and-sex gap) equal the fixed analysis.wide_margin and analysis.eval_margin, and the
    outcomes re-derived at the fixed margins equal the stored ones. Raises AssertionError on any difference beyond the
    tolerance."""
    ac = cfg["analysis"]
    tol = float(ac["reproduction_tolerance"])
    wm, em = float(ac["wide_margin"]), float(ac["eval_margin"])
    cohort, oof = inp["cohort"], inp["oof"]
    y = cohort.set_index(cohort.SEQN.astype(int))["death_10y"]
    out: dict = {"tolerance": tol}
    full = jsonable(A.run_wide(inp["wide_calls"], cohort, oof, B=int(ac["wide_B"])))
    d, worst, exact = compare(json.loads(json.dumps(full)), inp["wide_json"], tol)
    assert not d, f"run_wide does not reproduce results/wide/analysis.json: {d[:5]}"
    out["wide_run_wide"] = {"reproduced": True, "bit_exact": exact, "max_abs_diff": worst}
    preds, profiles = wide_predictions(inp["wide_calls"], oof)
    pb = jsonable(A.prediction_block(preds, y.reindex(profiles).dropna(), B=int(ac["wide_B"])))
    d, _, exact = compare(json.loads(json.dumps(pb)), json.loads(json.dumps(full["prediction"])), 0.0)
    assert not d and exact, f"the prediction inputs built here differ from analysis.run_wide's: {d[:5]}"
    common = common_records(preds, y)
    assert len(common) == inp["wide_json"]["prediction"]["n_records"], "record set differs from the stored block"
    out["wide_prediction_inputs"] = {"identical_to_run_wide": True, "n_records": len(common)}
    ep, eprof, src = eval_predictions(inp["eval_calls"], cohort, oof, root)
    prim = [m for m in A.ordered(set(inp["eval_calls"].model)) if A.is_primary(m) and m != "jev"]
    epb = jsonable(A.prediction_block(ep, y.reindex(eprof).dropna(), B=int(ac["eval_B"]), focal_versus=prim))
    skip = _survival_path if src == "reference_oof" else (lambda p: False)
    d, worst, exact = compare(json.loads(json.dumps(epb)), inp["eval_json"]["prediction"], tol, skip=skip)
    assert not d, f"the evaluation prediction block is not reproduced: {d[:5]}"
    ecommon = common_records(ep, y)
    out["eval_prediction"] = {"reproduced": True, "bit_exact": exact, "max_abs_diff": worst, "references": src,
                              "n_records": len(ecommon), "focal_versus": prim,
                              "not_compared": "entries that depend on the survival reference" if src == "reference_oof" else None}
    margins = {}
    for nm, blk, stored, fixed in (("wide", pb, inp["wide_json"]["prediction"], wm),
                                   ("eval", epb, inp["eval_json"]["prediction"], em)):
        rec = A.NI_FRACTION * blk["age_sex_to_reference_gap"]
        assert _close(rec, fixed, tol), f"{nm}: recomputed margin {rec!r} differs from the fixed {fixed!r}"
        chk = fixed_margin_outcomes(json.loads(json.dumps(blk)), fixed)
        sk = skip if nm == "eval" else (lambda p: False)
        diff = [f"{f}/{k}" for f, fam in chk.items() if isinstance(fam, dict) and "pairs" in fam
                for k, pr in fam["pairs"].items() if "noninferior" in pr and not sk(f"/{f}/pairs/{k}/noninferior")
                and stored[f]["pairs"][k]["noninferior"] != pr["noninferior_fixed_margin"]]
        assert not diff, f"{nm}: outcomes at the fixed margin differ from the stored ones: {diff[:5]}"
        margins[nm] = {"fixed": fixed, "recomputed": rec, "abs_diff": abs(rec - fixed), "bit_exact": rec == fixed,
                       "outcomes_at_fixed_margin_equal_stored": True}
    out["margins"] = margins
    cf = jsonable(cheap_family(ep, y.reindex(ecommon), "jev", ac["cheap_family"], em, int(ac["cheap_B"]),
                               int(ac["cheap_seed"])))
    d, worst, exact = compare(json.loads(json.dumps(cf)), inp["cheap_json"]["sets"]["risk_evaluation"], tol)
    assert not d, f"cheap_family does not reproduce results/summaries/cheap_family.json risk_evaluation: {d[:5]}"
    out["eval_cheap_family"] = {"reproduced": True, "bit_exact": exact, "max_abs_diff": worst, "margin": em}
    out["_state"] = {"wide_common": common, "eval_common": ecommon, "eval_focal_versus": prim}
    return out


def scored_set(original: list[int], answered) -> dict:
    """The records scored: the original analysis's record set intersected with the records documented Jev answered."""
    ans = set(int(p) for p in answered)
    kept = sorted(p for p in original if p in ans)
    dropped = sorted(p for p in original if p not in ans)
    return {"original": len(original), "scored": len(kept), "dropped": len(dropped), "dropped_ids": dropped,
            "_ids": kept}


def score(calls_doc: pd.DataFrame, inp: dict, rep: dict, root: Path = ROOT, cfg: dict = CFG) -> dict:
    """Documented Jev scored with the reproduced code: in place of Jev's original answers, everything else as stored.
    Records: the original analysis's sets (12,697 wide, 983 evaluation) intersected with the records documented Jev
    answered. Non-inferiority at the fixed margins analysis.wide_margin and analysis.eval_margin (fixed_margin_outcomes),
    beside prediction_block's own margin on the records scored."""
    ac = cfg["analysis"]
    wm, em = float(ac["wide_margin"]), float(ac["eval_margin"])
    cohort, oof = inp["cohort"], inp["oof"]
    y = cohort.set_index(cohort.SEQN.astype(int))["death_10y"]
    stt = rep["_state"]
    cols = list(inp["wide_calls"].columns)
    assert list(calls_doc.columns) == cols, f"calls table columns differ from results/wide/calls.parquet: {list(calls_doc.columns)}"
    jd = documented_answers(calls_doc).dropna()
    name = "jev_documented"
    ws, es = scored_set(stt["wide_common"], jd.index), scored_set(stt["eval_common"], jd.index)
    preds, _ = wide_predictions(inp["wide_calls"], oof, jev=jd, name=name)
    yw = y.reindex(ws["_ids"])
    out = {"records": {"wide": {k: v for k, v in ws.items() if k != "_ids"},
                       "eval": {k: v for k, v in es.items() if k != "_ids"}},
           "margins": {"wide_fixed": wm, "eval_fixed": em, "source": "config/docuse_risk.yaml analysis.wide_margin, "
                       "analysis.eval_margin; noninferior_fixed_margin is the outcome at these, noninferior the outcome "
                       "at prediction_block's margin on the records scored"}}
    pr = A.prediction_block(preds, yw, focal=name, B=int(ac["wide_B"]))
    assert pr["n_records"] == ws["scored"], "prediction_block scored a different record set"
    out["prediction"] = fixed_margin_outcomes(pr, wm)
    orig, _ = wide_predictions(inp["wide_calls"], oof)
    two = {name: preds[name], "jev": orig["jev"], "reference_spline_logit": orig["reference_spline_logit"],
           "reference_age_sex": orig["reference_age_sex"]}
    out["documented_vs_original"] = fixed_margin_outcomes(
        A.prediction_block(two, yw, focal=name, B=int(ac["wide_B"]), focal_versus=["jev"]), wm)
    ep, eprof, src = eval_predictions(inp["eval_calls"], cohort, oof, root, jev=jd, name=name)
    ye = y.reindex(es["_ids"])
    epb = A.prediction_block(ep, ye, focal=name, B=int(ac["eval_B"]), focal_versus=stt["eval_focal_versus"])
    assert epb["n_records"] == es["scored"], "the evaluation block scored a different record set"
    out["eval_prediction"] = fixed_margin_outcomes(epb, em)
    out["eval_prediction_references"] = src
    out["eval_cheap_family"] = cheap_family(ep, ye, name, ac["cheap_family"], em, int(ac["cheap_B"]),
                                            int(ac["cheap_seed"]), label="Jev used as documented")
    out["validity"] = validity(calls_doc)
    return jsonable(out)


def validity(calls_doc: pd.DataFrame) -> dict:
    c = calls_doc
    lat = c.latency_s.dropna().astype(float)
    price = float(MODELS["jev"]["price_per_mtok_input"])
    return {"n_calls": int(len(c)), "usable": int(c.valid.sum()), "usable_share": float(c.valid.mean()) if len(c) else None,
            "parse": c["parse"].fillna("none").value_counts().to_dict(),
            "errors": c["error"].dropna().astype(str).str[:80].value_counts().head(10).to_dict(),
            "providers_reported": c["provider"].astype(str).value_counts().to_dict(),
            "models_reported": c["model_reported"].astype(str).value_counts().to_dict(),
            "tokens_in_total": float(c.tokens_in.fillna(0).sum()), "tokens_in_mean": float(c.tokens_in.mean()) if c.tokens_in.notna().any() else None,
            "usd_list_total": float(c.tokens_in.fillna(0).sum() * price / 1e6), "price_per_mtok_input": price,
            "latency_median_s": float(lat.median()) if len(lat) else None,
            "latency_p90_s": float(np.percentile(lat, 90)) if len(lat) else None}

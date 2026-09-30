"""Print call counts and cost before any run, for every arm and billing route. Calls are counted from the same plan
the run uses (runner.plan_split); input tokens are estimated from the request text (4 characters per token) and
--tokens_out output tokens are assumed per call (default 400). Batch-route families are priced at their batch rates,
Jev on input tokens only, Hugging Face router families at their config price. Routes are summed: OpenRouter (standard
and batch) and Hugging Face. No network.
  python scripts/cost_estimate.py eval | wide | framing        # one split, from data/states_<split>.parquet
  python scripts/cost_estimate.py all                          # evaluation run, framing sensitivity analysis and
                                                               # exposure diagnostic; wide shown apart
  python scripts/cost_estimate.py eval gpt,claude              # narrowed to these systems
  python scripts/cost_estimate.py all --stop 500               # also flag a projected total above this many dollars"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity.clients import estimate_tokens, MODELS, PROMPTS, split_system, active_families
from jevity.runner import plan_split, load_subset

ROOT = Path(__file__).resolve().parents[1]
PERT = yaml.safe_load((ROOT / "config" / "perturbations.yaml").read_text())
PROMPT_OVERHEAD = 140
ASSUMED_OUT = 400
WIDE_RECORDS = 11000         # fitting-set complete cases, for the wide projection before its states exist
FILL_PROMPT = 700            # exposure diagnostic prompt tokens
F = MODELS["families"]


def route_of(fam: str, population: str = "nhanes") -> str:
    s = F[fam]
    if s.get("transport") == "hf_router":
        return "huggingface"
    b = MODELS.get("batch") or {}
    if b.get("enabled") and population in b.get("populations", []) and fam in (b.get("families") or {}):
        return "openrouter_batch"
    return "openrouter"


def prices(system: str, route: str) -> tuple[float, float]:
    """(input, output) US$ per million tokens for a calls.model value on a route; Jev bills input only."""
    if system.startswith("jev"):
        return float(MODELS["jev"]["price_per_mtok_input"]), 0.0
    fam = split_system(system.split(":")[0])[0]
    if route == "openrouter_batch" and fam in ((MODELS.get("batch") or {}).get("families") or {}):
        b = MODELS["batch"]["families"][fam]
        return float(b["price_in"]), float(b["price_out"])
    return float(F[fam]["price_in"]), float(F[fam]["price_out"])


def per_call(system: str, text_tokens: float, population: str = "nhanes", standard: bool = False,
             tokens_out: float = ASSUMED_OUT) -> dict:
    """Tokens and price of one call for a calls.model value (family or family@effort); standard=True prices the
    synchronous route even for a batch family (exposure diagnostic, route bridge, timing sample)."""
    fam, _ = split_system(system)
    route = route_of(fam, population)
    if standard and route == "openrouter_batch":
        route = "openrouter"
    pi, po = prices(system, route)
    return {"route": route, "prompt": text_tokens, "completion": tokens_out,
            "usd": (text_tokens * pi + tokens_out * po) / 1e6, "price_assumed": route == "huggingface"}


def estimate(split: str, states: pd.DataFrame, systems: list[str] | None = None, subset: list | None = None,
             quiet: bool = False, tokens_out: float = ASSUMED_OUT) -> dict:
    """Cost of a split's whole plan, by system and route."""
    say = (lambda *a: None) if quiet else print
    calls = plan_split(states, split, PERT, subset=subset, systems=systems)
    tok = float(pd.Series([estimate_tokens(c.text) for c in calls]).mean()) + PROMPT_OVERHEAD if calls else 0
    n = pd.Series([c.model for c in calls]).value_counts()
    rows, routes = {}, {}
    jev_n = int(sum(v for k, v in n.items() if k.startswith("jev")))
    jev_usd = jev_n * tok * MODELS["jev"]["price_per_mtok_input"] / 1e6
    routes["openrouter"] = jev_usd
    say(f"{split}: {len(calls)} calls, ~{tok:.0f} input tokens per call by character count, {tokens_out:.0f} output "
        f"tokens per call assumed")
    say(f"  {'jev (all questions)':24s} {jev_n:7d} calls  ~${jev_usd:8.2f}")
    for m in sorted((k for k in n.index if not k.startswith("jev")), key=lambda k: (split_system(k)[0], k)):
        pc = per_call(m, tok, tokens_out=tokens_out)
        usd = int(n[m]) * pc["usd"]
        rows[m] = {**pc, "calls": int(n[m]), "usd_total": usd}
        routes[pc["route"]] = routes.get(pc["route"], 0.0) + usd
        flag = " (price ASSUMED)" if pc["price_assumed"] else ""
        say(f"  {m:24s} {int(n[m]):7d} calls  ~${usd:8.2f}  ${pc['usd']:.4f}/call  in {pc['prompt']:.0f} out "
            f"{pc['completion']:.0f}  {pc['route']}{flag}")
    total = sum(routes.values())
    say("  by route: " + ", ".join(f"{k} ${v:,.2f}" for k, v in sorted(routes.items())))
    say(f"  TOTAL {split} plan ~${total:,.2f}")
    return {"total": total, "routes": routes, "systems": rows, "jev": {"calls": jev_n, "usd": jev_usd}}


def exposure_cost(quiet: bool = False, tokens_out: float = ASSUMED_OUT) -> float:
    """Masked-field recovery (exposure_diagnostic.py): n profiles x each primary LLM, standard route."""
    n = PROMPTS["exposure"]["n_profiles"]
    fams = active_families(("primary",))
    usd = sum(n * per_call(f, FILL_PROMPT, standard=True, tokens_out=tokens_out)["usd"] for f in fams)
    if not quiet:
        print(f"exposure diagnostic: {n} records x {len(fams)} primary LLMs, standard route, ~${usd:,.2f}")
    return usd


def load_states(split: str, sdir: Path) -> pd.DataFrame:
    f = sdir / f"states_{'eval' if split == 'framing' else split}.parquet"
    if split == "wide" and not f.exists():      # projected from the evaluation baselines' length: baseline + M2
        ev = pd.read_parquet(sdir / "states_eval.parquet")
        txt = list(ev.loc[ev.edit.eq("baseline") & ~ev.annotated, "text"])
        reps = (txt * (2 * WIDE_RECORDS // max(len(txt), 1) + 1))[: 2 * WIDE_RECORDS]
        return pd.DataFrame({"profile": np.repeat(np.arange(-WIDE_RECORDS, 0), 2),
                             "edit": ["baseline", "M2_race_removed"] * WIDE_RECORDS, "annotated": False, "text": reps})
    return pd.read_parquet(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", nargs="?", default="eval", choices=["eval", "wide", "framing", "all"])
    ap.add_argument("systems", nargs="?", default=None, help="comma list of calls.model values")
    ap.add_argument("--states", default=None, help="folder with states_<split>.parquet (default data/)")
    ap.add_argument("--tokens_out", type=float, default=ASSUMED_OUT, help="output tokens assumed per call")
    ap.add_argument("--stop", type=float, default=None, help="flag a projected total above this many dollars")
    ap.add_argument("--json", default=None, help="write the numbers to this file")
    a = ap.parse_args()
    sdir = Path(a.states) if a.states else ROOT / "data"
    systems = a.systems.split(",") if a.systems else None
    subset = load_subset()
    if subset is None and a.split in ("eval", "framing", "all"):
        pro = sorted(pd.read_parquet(sdir / "states_eval.parquet").profile.unique())
        subset = [int(x) for x in np.random.default_rng(20260922).choice(pro, size=min(200, len(pro)), replace=False)]
        print("(config/subsets.yaml not drawn yet: a seeded 200-record stand-in is used for the projection)")
    if a.split != "all":
        r = estimate(a.split, load_states(a.split, sdir), systems, subset, tokens_out=a.tokens_out)
        total = r["total"]
    else:
        e = estimate("eval", load_states("eval", sdir), systems, subset, tokens_out=a.tokens_out)
        print()
        fr = estimate("framing", load_states("framing", sdir), systems, subset, tokens_out=a.tokens_out)
        print()
        mem = exposure_cost(tokens_out=a.tokens_out)
        w = estimate("wide", load_states("wide", sdir), systems, subset, quiet=True, tokens_out=a.tokens_out)
        total = e["total"] + fr["total"] + mem
        routes = {k: e["routes"].get(k, 0.0) + fr["routes"].get(k, 0.0) for k in set(e["routes"]) | set(fr["routes"])}
        routes["openrouter"] = routes.get("openrouter", 0.0) + mem
        print(f"wide run ({WIDE_RECORDS:,} records x baseline + M2 when its states are not built yet): "
              f"~${w['total']:,.0f} (not in the total below)")
        print(f"\nPROJECTED TOTAL (evaluation run + framing sensitivity analysis + exposure diagnostic): ${total:,.0f}   "
              "by route: " + ", ".join(f"{k} ${v:,.0f}" for k, v in sorted(routes.items())))
        r = {"eval": e, "framing": fr, "exposure": mem, "wide": w, "total": total, "routes": routes}
    if a.stop is not None and total > a.stop:
        print(f"ABOVE the --stop limit of ${a.stop:,.0f}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1, default=float))

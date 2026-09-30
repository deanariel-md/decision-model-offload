"""Family-wise error and power of the confirmatory family (11 edits x 6 systems, 66 cells, plus a clinical response
slope per system: 72 tests in one max-t) at NHANES scale, on synthetic cohorts (no API, no network). Each replicate
draws a cohort, fits the primary reference and R bootstrap refits, applies the family edits to 1,000 evaluation records,
simulates six systems and runs the same confirmatory_family() the real analysis uses. The simulated systems carry the
labels of the primary systems (jev, gpt, claude, gemini, muse, glm); every answer is generated here.
  null     every system reports the true contrast plus noise on the log-odds. Noise model "shared" (default): a
           per-profile system deviation u ~ N(0, 0.15) common to the baseline and every edit of that profile, plus
           independent call noise N(0, 0.15) on every call; each edit is compared with the same baseline call, so cells
           of one profile share the baseline's call noise. "legacy": one noise term on the edited call only, the
           baseline noise cancelling from z.
  planted  as null, plus (the clinical slope of each system is tested against 1 in the same max-t): claude race +0.3
           log-odds; glm income +0.3; jev clinical contrasts x 0.7 (compression); gpt albumin x 0.5
Also records, under the null, how often the derived headline metrics look like findings (consensus excess, weight
agreement).
Writes results/family_simulation/<out> (summary, after every replicate) and results/family_simulation/<stem>_rows.csv
(one row per replicate, scenario, edit and system); --merge summarises several row files.
  python scripts/family_simulation.py [--reps 20] [--R 50] [--B 500] [--n 12000] [--seed0 3000] [--noise-model shared]
                                      [--out family_simulation.json]
  python scripts/family_simulation.py --merge <rows.csv> [<rows.csv> ...] --out family_validation.json
The factor crit_inflation in config/analysis.yaml comes from 200 replicates: --seed0 9000 --reps 200 --R 50 --B 500
--n 12000 --noise-model shared (run in parts, e.g. 100 + 84 + 8 + 8 replicates from seeds 9000, 9100, 9184 and 9192,
then --merge); it is the smallest factor on a 0.01 grid whose family-wise error under the global null is at most 5%."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import reference as R
from jevity import perturb as P
from jevity import analysis as A
from jevity import headline as H
from jevity.simulate import make_synthetic_cohort, true_logit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "family_simulation"
MODELS = ["jev", "gpt", "claude", "gemini", "muse", "glm"]
PLANTED = {("claude", "L1_race_nhw_to_nhb"): ("add", 0.3), ("glm", "L2a_income_low"): ("add", 0.3),
           ("gpt", "C3_albumin_minus0.5"): ("scale", 0.5)}
JEV_CLINICAL_SCALE = 0.7


def build_instances(ev: pd.DataFrame, edits: list[str]):
    base, edited, meta = [], [], []
    for rec in ev.to_dict("records"):
        for e in edits:
            ed = P.apply_record_edit(rec, e)
            if not ed.applicable:
                continue
            base.append(rec); edited.append(ed.record)
            meta.append({"profile": int(rec["SEQN"]), "edit": e, "klass": "clinical" if e.startswith("C") else "label",
                         "feature": P.PERT["clinical" if e.startswith("C") else "label"][e]["column"]})
    return pd.DataFrame(base), pd.DataFrame(edited), pd.DataFrame(meta)


def one_rep(seed: int, n: int, R_: int, B: int, noise: float = 0.15, noise_model: str = "shared") -> list[dict]:
    rng = np.random.default_rng(seed)
    co = make_synthetic_cohort(n=n, seed=seed, n_pilot=0, n_eval=1000)
    fit, ev = co[co.split == "fit"], co[co.split == "eval"].reset_index(drop=True)
    edits = list(A.FAMILY)
    Bs, Es, meta = build_instances(ev, edits)
    ref = R.fit_spline_logit(fit, era_specific=[])
    c0 = ref.contrast(Bs, Es)
    DL = np.empty((len(Bs), R_)); DP = np.empty_like(DL)
    for r, m in enumerate(R.bootstrap_refits(fit, n=R_, seed=seed, kind="spline_logit")):
        c = m.contrast(Bs, Es); DL[:, r], DP[:, r] = c.d_logit.to_numpy(), c.d_prob.to_numpy()
    refits = A.Refits(DL, DP, {(p, e): i for i, (p, e) in enumerate(zip(meta.profile, meta.edit))})
    tb = np.array([true_logit(r) for r in Bs.to_dict("records")])
    te = np.array([true_logit(r) for r in Es.to_dict("records")])
    base_lo = {int(r["SEQN"]): true_logit(r) for r in ev.to_dict("records")}
    cohort = ev[["SEQN", "death_10y"]].copy()
    profiles = sorted(base_lo)
    out = []
    for scen in ("null", "planted"):
        tabs, calls = {}, []
        for m in MODELS:
            if noise_model == "legacy":
                nb = {p: rng.normal(0, noise) for p in profiles}        # one baseline call per profile
                u = {p: 0.0 for p in profiles}
            else:
                u = {p: rng.normal(0, 0.15) for p in profiles}          # the system's deviation for this person
                nb = {p: u[p] + rng.normal(0, noise) for p in profiles}  # plus the baseline call's own noise
            lb = np.array([base_lo[p] + nb[p] for p in meta.profile])
            ue = np.array([u[p] for p in meta.profile])
            dz = te - tb
            if scen == "planted":
                if m == "jev":
                    dz = np.where(meta.klass.to_numpy() == "clinical", dz * JEV_CLINICAL_SCALE, dz)
                for (pm, pe), (op, v) in PLANTED.items():
                    if pm == m:
                        sel = meta.edit.to_numpy() == pe
                        dz = np.where(sel, dz + v if op == "add" else dz * v, dz)
            if noise_model == "legacy":
                le = lb + dz + rng.normal(0, noise, len(meta))
            else:                                                       # edited call: same person, its own call noise
                le = tb + ue + dz + rng.normal(0, noise, len(meta))
            pb, pe_ = A.expit(lb), A.expit(le)
            tabs[m] = meta.assign(p_base=pb, p=pe_, z_logit=A.logit(pe_) - A.logit(pb), z_prob=pe_ - pb,
                                  d_logit=c0.d_logit.to_numpy(), d_prob=c0.d_prob.to_numpy(), q0=c0.q0.to_numpy())
            calls.append(pd.DataFrame({"model": m, "profile": profiles, "edit": "baseline", "repeat": 0, "annotated": False,
                                       "valid": True, "variant": "raw", "p": [A.expit(base_lo[p] + nb[p]) for p in profiles]}))
        calls = pd.concat(calls, ignore_index=True)
        draws = A.Draws(profiles, B, R_, seed=seed + 1)
        fam = A.confirmatory_family(tabs, refits, draws, calls, cohort, edits=edits, pairs=False, keep_draws=True)
        cons = H.consensus(fam); w = H.weights(fam)
        for e in edits:
            for m in MODELS:
                r = fam["cells"].get(e, {}).get(m)
                if r is None:
                    continue
                lo, hi = r["D_logit_ci_simultaneous"]
                planted = scen == "planted" and ((m, e) in PLANTED or (m == "jev" and e.startswith("C")))
                out.append({"seed": seed, "scenario": scen, "noise_model": noise_model, "R": R_, "B": B,
                            "edit": e, "model": m, "planted": planted, "D": r["D_logit"],
                            "lo": lo, "hi": hi, "rejects": not (lo <= 0 <= hi), "crit": fam["critical_value"]["D_logit"],
                            "crit_raw": fam["critical_value_raw"]["D_logit"],
                            "lo_marg": r["D_logit_ci"][0], "hi_marg": r["D_logit_ci"][1]})
        for m, v in fam.get("clinical_slope", {}).items():
            lo, hi = v["ci_simultaneous"]
            out.append({"seed": seed, "scenario": scen, "edit": "clinical_slope", "model": m,
                        "planted": scen == "planted" and m in ("jev", "gpt"),     # gpt's albumin plant moves its slope
                        "D": v["beta"], "lo": lo, "hi": hi, "rejects": v["excludes_one"],
                        "crit": fam["critical_value"]["D_logit"], "crit_raw": fam["critical_value_raw"]["D_logit"]})
        for e, v in cons["fields"].items():
            lo, hi = v["mean_excess_llms_ci_simultaneous"]
            out.append({"seed": seed, "scenario": scen, "edit": e, "model": "consensus_llms", "planted": False,
                        "D": v["mean_excess_llms"], "lo": lo, "hi": hi, "rejects": not (lo <= 0 <= hi), "crit": cons["critical_value"]})
        for m, v in w.items():
            out.append({"seed": seed, "scenario": scen, "edit": "weights_spearman", "model": m, "planted": False,
                        "D": v["spearman"], "lo": v["spearman_ci"][0], "hi": v["spearman_ci"][1], "rejects": v["spearman_ci"][0] > 0.9})
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def summarise(df: pd.DataFrame, settings: dict) -> dict:
    """Family-wise error and power at several critical-value factors, and the factor the null replicates call for:
    k = 95th percentile over null replicates of max_j |t_j| / (raw critical value), t_j = (estimate - null) / SE_j."""
    fam = df[df.model.isin(MODELS) & (df.edit != "weights_spearman")].copy()
    for c in ("D", "lo", "hi", "crit", "crit_raw", "lo_marg", "hi_marg"):
        if c not in fam:
            continue
        fam[c] = pd.to_numeric(fam[c], errors="coerce")
    fam["planted"] = fam.planted.astype(str) == "True"
    fam["null_value"] = np.where(fam.edit == "clinical_slope", 1.0, 0.0)
    fam["mid"], fam["se"] = (fam.lo + fam.hi) / 2, (fam.hi - fam.lo) / (2 * fam.crit)
    fam["t"] = (fam.mid - fam.null_value) / fam.se
    nul = fam[fam.scenario == "null"]
    ratio = nul.groupby("seed").apply(lambda g: np.abs(g.t).max() / g.crit_raw.iloc[0])
    out = {"replicates": int(fam.seed.nunique()), "settings": settings,
           "note": "Synthetic cohorts; one row per replicate, scenario, edit and system in the rows files.",
           "mean_critical_value_raw": float(fam.crit_raw.mean()),
           "null_sd_standardised_error_cells": float(nul[nul.edit != "clinical_slope"].t.std()),
           "null_sd_standardised_error_slopes": float(nul[nul.edit == "clinical_slope"].t.std()),
           "k_hat_95th_percentile": float(np.quantile(ratio, 0.95)) if len(ratio) else None,
           "k_hat_bootstrap_95ci": ([float(x) for x in np.quantile([np.quantile(np.random.default_rng(i).choice(ratio.to_numpy(), len(ratio)), 0.95)
                                                                    for i in range(2000)], [0.025, 0.975])] if len(ratio) >= 20 else None),
           "marginal_coverage_null_cells": (float(((c_.lo_marg <= 0) & (c_.hi_marg >= 0)).mean())
                                            if len(c_ := nul[nul.edit != "clinical_slope"].dropna(subset=["lo_marg", "hi_marg"])) else None),
           "clinical_slope_null": {"mean": float(nul[nul.edit == "clinical_slope"].D.mean()),
                                   "sd": float(nul[nul.edit == "clinical_slope"].D.std())},
           "by_inflation": {}}
    for k in (1.0, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.10):
        fam["r"] = np.abs(fam.t) > k * fam.crit_raw
        pl = fam[fam.scenario == "planted"]
        nfw = fam[fam.scenario == "null"].groupby("seed").r.any()
        pw = lambda m, e: float(pl[(pl.model == m) & (pl.edit == e)].r.mean())
        out["by_inflation"][str(k)] = {
            "fwer_global_null_72": float(nfw.mean()),
            "fwer_global_null_72_mc_se": float(np.sqrt(nfw.mean() * (1 - nfw.mean()) / max(len(nfw), 1))),
            "fwer_global_null_72_wilson95": wilson(int(nfw.sum()), int(len(nfw))),
            "fwer_non_planted_in_planted_scenario": float(pl[~pl.planted].groupby("seed").r.any().mean()),
            "power_claude_race_plus0.3": pw("claude", "L1_race_nhw_to_nhb"), "power_glm_income_plus0.3": pw("glm", "L2a_income_low"),
            "power_gpt_albumin_x0.5": pw("gpt", "C3_albumin_minus0.5"), "power_jev_clinical_slope_x0.7": pw("jev", "clinical_slope"),
            "power_jev_clinical_cells_x0.7": {e: pw("jev", e) for e in A.FAMILY if e.startswith("C")}}
    return out


if __name__ == "__main__":
    if "--merge" in sys.argv:    # python scripts/family_simulation.py --merge rows_a.csv rows_b.csv ... --out family_simulation.json
        i = sys.argv.index("--merge"); files = [f for f in sys.argv[i + 1:] if f.endswith(".csv")]
        out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "family_simulation.json"
        df = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in files], ignore_index=True)
        assert df.groupby("seed").scenario.nunique().min() == 2, "incomplete replicate"
        got = lambda c, d: sorted({str(x) for x in df[c].dropna()}) if c in df else d
        res = summarise(df, {"n_cohort": 12000, "n_eval": 1000, "R": got("R", [50]), "B": got("B", [500]), "tests": 72,
                             "noise_model": got("noise_model", ["legacy"]), "source": [Path(f).name for f in files]})
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / out).write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "by_inflation"}, indent=1))
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20); ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--B", type=int, default=500); ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed0", type=int, default=3000); ap.add_argument("--out", default="family_simulation.json")
    ap.add_argument("--noise-model", default="shared", choices=["shared", "legacy"])
    a = ap.parse_args()
    rows, t0 = [], time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in range(a.reps):
        rows += one_rep(a.seed0 + s, a.n, a.R, a.B, noise_model=a.noise_model)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / (Path(a.out).stem + "_rows.csv"), index=False)
        (OUT / a.out).write_text(json.dumps({"reps_done": s + 1, **summarise(df, vars(a))}, indent=1))
        print(f"rep {s + 1}/{a.reps} {time.time() - t0:.0f}s", flush=True)

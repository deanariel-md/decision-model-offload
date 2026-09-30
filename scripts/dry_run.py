"""Dry run of the whole arm 1 pipeline on a synthetic cohort with simulated models (no network, no API).
Copies the code to a scratch folder, so nothing in data/ or results/ is touched.
  python scripts/dry_run.py [--out <folder>] [--R 20] [--B 400] [--sex-uses-category-rule true|false]"""
import argparse, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None); ap.add_argument("--R", type=int, default=20); ap.add_argument("--B", type=int, default=400)
    ap.add_argument("--sex-uses-category-rule", choices=["true", "false"], default="true",
                    help="support.sex_uses_category_rule written into the scratch copy's config")
    a = ap.parse_args()
    work = Path(a.out or tempfile.mkdtemp(prefix="jevity_dry_"))
    for d in ("src", "scripts", "config"):
        shutil.copytree(ROOT / d, work / d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    for d in ("data", "results", "runs"):
        (work / d).mkdir(parents=True, exist_ok=True)
    (work / "config" / "subsets.yaml").unlink(missing_ok=True)     # the dry run draws its own synthetic subset
    import yaml
    pf = work / "config" / "perturbations.yaml"
    pt = pf.read_text(encoding="utf-8")
    line = next(l for l in pt.splitlines() if l.strip().startswith("sex_uses_category_rule:"))
    pf.write_text(pt.replace(line, f"  sex_uses_category_rule: {a.sex_uses_category_rule}"), encoding="utf-8")
    assert yaml.safe_load(pf.read_text(encoding="utf-8"))["support"]["sex_uses_category_rule"] == (a.sex_uses_category_rule == "true")
    sys.path.insert(0, str(work / "src"))
    from jevity.simulate import make_synthetic_survival_cohort, make_synthetic_prior
    co = make_synthetic_survival_cohort(n=6000, seed=3, n_pilot=200, n_eval=1000); co["complete9"] = True
    co.to_parquet(work / "data" / "cohort.parquet", index=False)
    make_synthetic_prior(n=7000, seed=4).to_parquet(work / "data" / "cohort_nhanes3.parquet", index=False)
    shutil.copy(ROOT / "config" / "nhanes3.yaml", work / "config" / "nhanes3.yaml")
    py = sys.executable
    steps = [[py, "scripts/fit_reference.py"],
             [py, "scripts/make_states.py", "eval"],
             [py, "scripts/draw_subset.py"],
             [py, "scripts/refit_references.py", "eval", "--R", str(a.R)],
             [py, "scripts/run_calls.py", "eval", "--simulate"],
             [py, "scripts/analyze.py", "eval", "--B", str(a.B), "--B_desc", "200", "--synthetic"],
             [py, "scripts/run_calls.py", "framing", "--simulate"],
             [py, "scripts/crossfit_reference.py"],
             [py, "scripts/make_states.py", "wide"],
             [py, "scripts/run_calls.py", "wide", "--simulate"],
             [py, "scripts/analyze.py", "wide", "--B", str(a.B)],
             [py, "scripts/timing_sample.py", "--simulate"],
             [py, "scripts/cost_speed.py", "eval"],
             [py, "scripts/exposure_diagnostic.py", "--simulate"]]
    for s in steps:
        print("\n>>>", " ".join(s[1:]), flush=True)
        r = subprocess.run(s, cwd=work)
        if r.returncode != 0:
            sys.exit(f"step failed: {' '.join(s[1:])}")
    print(f"\nDry run complete: {work}")

"""eICU replication, analysis. Descriptive, no non-inferiority test.
  python scripts/analyze_eicu.py eval      # baseline prediction (jevity.eicu_analysis): Jev, the five primary chatbots,
                                           # APACHE IVa's predicted hospital mortality and, when fitted on the full eICU
                                           # by scripts/fit_eicu_reference.py, the spline, LightGBM and XGBoost
                                           # references
  python scripts/analyze_eicu.py eval <root>   # the same under another root folder
Reads results/eicu/calls_eval.parquet and data/eicu/demo_cohort.parquet (open demo); the reference files hold fitted
parameters only. Writes results/eicu/analysis.json (with each system's usable answers, parse types, finish reasons,
providers and routes) and baseline_table.md.

Secondary systems (scripts/run_eicu.py --tiers free,pair): when results/eicu/calls_eval_secondary.parquet exists, the
five (GPT-5.6 Luna, Claude Sonnet 5, Gemini 3.5 Flash-Lite, MedGemma, Gemma) are added beside the six, descriptively.
analysis.json gains the block "secondary_prediction": every predictor (the six, APACHE IVa, the references and the
five) on the stays all of them scored, with the same 2,000 resamples and seed, plus each system's answered and invalid
counts and list cost per 1,000 answers (config/models.yaml standard prices); baseline_table_secondary.md is written
beside it. When analysis.json or baseline_table.md already exists, everything in it (apart from
"secondary_prediction") must be rebuilt unchanged, or nothing is written."""
import json, sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import eicu as E                              # noqa: E402
from jevity import eicu_analysis as EA                    # noqa: E402

D, OUT = ROOT / "data" / "eicu", ROOT / "results" / "eicu"
PRIMARY = ["jev", "gpt", "claude", "gemini", "muse", "glm"]
SECONDARY = ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
NAMES = {"jev": "Jev 1.13", "gpt": "GPT-5.6 Sol", "claude": "Claude Opus 5.5", "gemini": "Gemini 3.8 Flash",
         "muse": "Muse Spark 1.1", "glm": "GLM-5.3", "gpt_free": "GPT-5.6 Luna", "claude_free": "Claude Sonnet 5",
         "gemini_free": "Gemini 3.5 Flash-Lite", "medgemma": "MedGemma 27B", "gemma": "Gemma 3 27B",
         "apache_iva": "APACHE IVa",
         "reference_spline_logit": "Spline logistic reference (full eICU)", "reference_lightgbm": "LightGBM reference (full eICU)",
         "reference_xgboost": "XGBoost reference (full eICU)"}


def parse_check(calls: pd.DataFrame) -> dict:
    out = {}
    for m, g in calls.groupby("model", sort=False):
        out[m] = {"calls": int(len(g)), "usable": int(g.valid.sum()), "share": float(g.valid.mean()),
                  "parse": g["parse"].fillna("none").value_counts().to_dict(),
                  "finish": g["finish_reason"].fillna("none").value_counts().to_dict(),
                  "providers": g["provider"].fillna("none").value_counts().to_dict(),
                  "routes": g["route"].fillna("none").value_counts().to_dict(),
                  "errors": g["error"].dropna().astype(str).str[:80].value_counts().head(5).to_dict()}
    return out


def system_preds(calls: pd.DataFrame, systems: list[str]) -> dict[str, pd.Series]:
    preds = {}
    for m in systems:
        g = calls[(calls.model == m) & (calls.edit == "baseline") & calls.valid]
        if len(g):
            preds[m] = g.drop_duplicates("profile").set_index("profile")["p"].astype(float)
    return preds


def predictors(root: Path, calls: pd.DataFrame, demo: pd.DataFrame) -> dict[str, pd.Series]:
    ev = demo.set_index("SEQN")
    preds = system_preds(calls, PRIMARY)
    preds["apache_iva"] = ev["apache_iva_pred"].astype(float)
    for n in ("spline_logit", "lightgbm", "xgboost"):
        f = root / "data" / "eicu" / f"reference_{n}.joblib"
        if f.exists():
            preds[f"reference_{n}"] = pd.Series(joblib.load(f).predict(ev.reset_index()), index=ev.index)
    return preds


def table(block: dict) -> str:
    def ci(v, k, fmt):
        c = v.get(f"{k}_ci")
        return f"{fmt.format(v[k])} ({fmt.format(c[0])}–{fmt.format(c[1])})" if c and v.get(k) is not None else "–"
    lines = [f"Baseline prediction of death before hospital discharge, eICU-CRD demo evaluation stays "
             f"(n = {block['n_records']:,}, {block['deaths']} deaths; 95% bootstrap intervals, {block['B']:,} resamples)", "",
             "| Predictor | Log-loss | Brier | AUROC | Calibration slope | Calibration intercept | Mean predicted | Observed |",
             "|---|---|---|---|---|---|---|---|"]
    for n, v in block["predictors"].items():
        lines.append(f"| {NAMES.get(n, n)} | {ci(v, 'log_loss', '{:.3f}')} | {ci(v, 'brier', '{:.3f}')} | "
                     f"{ci(v, 'auroc', '{:.3f}')} | {ci(v, 'calibration_slope', '{:.2f}')} | "
                     f"{ci(v, 'calibration_intercept', '{:.2f}')} | {ci(v, 'mean_predicted', '{:.3f}')} | "
                     f"{v['observed']:.3f} |")
    return "\n".join(lines) + "\n"


def answers(calls: pd.DataFrame, systems: list[str]) -> dict:
    """Per system: stays asked, answered (a stored reply), invalid (answered, not parsed), without a reply."""
    out = {}
    for m in systems:
        g = calls[(calls.model == m) & (calls.edit == "baseline")]
        replied = g["error"].isna()
        out[m] = {"calls": int(len(g)), "answered": int(replied.sum()), "invalid": int((replied & ~g.valid).sum()),
                  "no_reply": int((~replied).sum())}
    return out


def list_cost(calls: pd.DataFrame, systems: list[str]) -> dict:
    """List cost of 1,000 answers at config/models.yaml standard prices (Jev: input only, output free), from each
    answer's reported tokens, over the stays the system answered."""
    import yaml
    cfg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    out = {}
    for m in systems:
        g = calls[(calls.model == m) & (calls.edit == "baseline") & calls["error"].isna()]
        pin, pout = ((cfg["jev"]["price_per_mtok_input"], 0.0) if m == "jev"
                     else (cfg["families"][m]["price_in"], cfg["families"][m]["price_out"]))
        usd = (g.tokens_in.fillna(0) * pin + g.tokens_out.fillna(0) * pout) / 1e6
        out[m] = {"usd_per_1000": float(1000 * usd.mean()), "usd_total": float(usd.sum()), "price_in": pin,
                  "price_out": pout, "tokens_in_mean": float(g.tokens_in.mean()), "tokens_out_mean": float(g.tokens_out.mean())}
    return out


def secondary_block(root: Path, calls: pd.DataFrame, sec: pd.DataFrame, demo: pd.DataFrame, y: pd.Series,
                    stored: dict) -> dict:
    """Every predictor on the stays all of them scored, the same resamples and seed. When those stays are the baseline
    block's, the six, APACHE IVa and the references must come out exactly as in that block (`stored`)."""
    block = EA.baseline_block({**predictors(root, calls, demo), **system_preds(sec, SECONDARY)}, y)
    same = block["n_records"] == stored["n_records"] and block["deaths"] == stored["deaths"]
    if same:
        for n, v in stored["predictors"].items():
            assert json.loads(json.dumps(block["predictors"][n], default=float)) == v, f"{n} changed"
    both = pd.concat([calls, sec], ignore_index=True)
    return {"note": "descriptive; the five secondary systems beside the six; every predictor scored on the stays all of "
                    "them scored, the stored 2,000 resamples and seed; cost at list price (config/models.yaml)",
            "systems_added": SECONDARY, "same_stays_as_baseline_prediction": bool(same), **block,
            "answers": answers(both, PRIMARY + SECONDARY), "cost": list_cost(both, PRIMARY + SECONDARY)}


def table_secondary(sb: dict) -> str:
    def ci(v, k, fmt):
        c = v.get(f"{k}_ci")
        return f"{fmt.format(v[k])} ({fmt.format(c[0])}–{fmt.format(c[1])})" if c and v.get(k) is not None else "–"
    lines = [f"Baseline prediction of death before hospital discharge, eICU-CRD demo evaluation stays: the five "
             f"secondary systems beside the six (n = {sb['n_records']:,} stays every predictor scored, {sb['deaths']} "
             f"deaths; 95% bootstrap intervals, {sb['B']:,} resamples; descriptive)", "",
             "| Predictor | AUROC | Mean predicted | Observed | Calibration slope | Log-loss | Answered | Invalid | "
             "US$ per 1,000 answers (list) |", "|---|---|---|---|---|---|---|---|---|"]
    for n, v in sb["predictors"].items():
        a, c = sb["answers"].get(n), sb["cost"].get(n)
        answered = f"{a['answered']:,} of {a['calls']:,}" if a else "–"
        invalid = f"{a['invalid']}" if a else "–"
        usd = f"{c['usd_per_1000']:.2f}" if c else "–"
        lines.append(f"| {NAMES.get(n, n)} | {ci(v, 'auroc', '{:.3f}')} | {ci(v, 'mean_predicted', '{:.3f}')} | "
                     f"{v['observed']:.3f} | {ci(v, 'calibration_slope', '{:.2f}')} | {ci(v, 'log_loss', '{:.3f}')} | "
                     f"{answered} | {invalid} | {usd} |")
    return "\n".join(lines) + "\n"


def unchanged(path: Path, new: dict) -> None:
    """Stop unless every key of the stored file is rebuilt with the same value."""
    if path.exists():
        old = json.loads(path.read_text())
        new = json.loads(json.dumps(new, default=float))
        bad = [k for k in old if k != "secondary_prediction" and old[k] != new.get(k)]
        assert not bad, f"{path.name}: stored values changed in {bad}"


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if split != "eval":
        raise SystemExit("analyze_eicu.py analyses the evaluation stays: python scripts/analyze_eicu.py eval [<root>]")
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT
    out = root / "results" / "eicu"; out.mkdir(parents=True, exist_ok=True)
    calls = pd.read_parquet(out / f"calls_{split}.parquet")
    sec_path = out / f"calls_{split}_secondary.parquet"
    sec = pd.read_parquet(sec_path) if sec_path.exists() else None
    demo = pd.read_parquet(root / "data" / "eicu" / "demo_cohort.parquet")
    demo = demo[demo.split == "eval"]
    y = demo.set_index("SEQN")[E.OUTCOME].astype(float)
    block = EA.baseline_block(predictors(root, calls, demo), y)
    res = {"meta": {"split": split, "population": "eicu_demo", "note": "descriptive; no non-inferiority test",
                    "eligible_demo_stays": int(len(pd.read_parquet(root / "data" / "eicu" / "demo_cohort.parquet"))),
                    "evaluation_stays": int(len(demo)), "outcome": E.OUTCOME},
           "parse": parse_check(calls), "baseline_prediction": block}
    unchanged(out / "analysis.json", res)
    bt = out / "baseline_table.md"
    assert not bt.exists() or bt.read_text(encoding="utf-8").replace("\r\n", "\n") == table(block), "baseline_table.md changed"
    if sec is not None:
        stored = json.loads(json.dumps(block, default=float))
        res["secondary_prediction"] = secondary_block(root, calls, sec, demo, y, stored)
        (out / "baseline_table_secondary.md").write_text(table_secondary(res["secondary_prediction"]), encoding="utf-8")
        print(table_secondary(res["secondary_prediction"]))
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=float))
    bt.write_text(table(block), encoding="utf-8")
    print(table(block))

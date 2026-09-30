"""Step 3. For a split, build every (profile, edit) instance: edited record, serialised text, diff, support flag, and
the reference contrasts d under the primary (spline_logit) and second (lightgbm) references, computed in batch.
Writes data/states_<split>.parquet (texts), data/instances_<split>.parquet (one row per instance, with d) and
data/edited_<split>.parquet (baseline and edited records, for reference refits). No API calls."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import perturb as P
from jevity.serialize import to_state, to_json

ROOT = Path(__file__).resolve().parents[1]
CLASS_OF = {**{k: "irrelevant" for k in P.PERT["irrelevant"]}, **{k: "clinical" for k in P.PERT["clinical"]},
            **{k: "label" for k in P.PERT["label"]}}
FEATURE_OF = {**{k: v["column"] for k, v in P.PERT["clinical"].items()}, **{k: v["column"] for k, v in P.PERT["label"].items()},
              **{k: "rendering" for k in P.PERT["irrelevant"]}}
REFS = ("spline_logit", "lightgbm", "survival_logit")
SENS_REFS = ("spline_logit_prior", "spline_logit_shared",  # present only when the NHANES III cohort is built
             "spline_logit_weighted")                     # present when the cohort carries examination weights


def build(split: str, edits: list, annotated_edits: list, annotated_profiles: set | None, root: Path = ROOT):
    df = pd.read_parquet(root / "data" / "cohort.parquet")
    fit, sub = df[df.split == "fit"], df[df.split == split]
    rule_b = P.PERT["support"].get("sex_uses_category_rule")
    if rule_b is None:
        raise SystemExit("config/perturbations.yaml support.sex_uses_category_rule must be true or false")
    excl = () if rule_b else ("sex",)     # false: sex edits use rules (a) and (c) only
    checker = P.SupportChecker(fit, exclude_strata=excl)
    refs = {n: joblib.load(root / "data" / f"reference_{n}.joblib") for n in REFS + SENS_REFS
            if (root / "data" / f"reference_{n}.joblib").exists()}
    from jevity.reference import PhenoAgeReference
    refs["phenoage"] = PhenoAgeReference()     # independent published reference (fields it contains only)
    inst, states, base_recs, edit_recs = [], [], [], []
    for _, row in sub.iterrows():
        rec = row.to_dict()
        pid = int(rec["SEQN"])
        ann_ok = annotated_profiles is None or pid in annotated_profiles
        states.append({"profile": pid, "edit": "baseline", "annotated": False, "text": to_json(to_state(rec))})
        if split == "eval":   # M2 mitigation: race field removed (analysed by group calibration, not by D)
            states.append({"profile": pid, "edit": "M2_race_removed", "annotated": False,
                           "text": P.render(rec, drop=P.PERT["mitigation"]["M2_race_removed"]["drop_fields"])})
        if annotated_edits and ann_ok:
            states.append({"profile": pid, "edit": "baseline", "annotated": True, "text": P.render(rec, annotated=True)})
        for e in edits:
            klass = CLASS_OF[e]
            if klass == "irrelevant":
                states.append({"profile": pid, "edit": e, "annotated": False, "text": P.render(rec, irrelevant=e, seed=pid)})
                inst.append({"profile": pid, "edit": e, "klass": klass, "feature": "rendering", "applicable": True,
                             "supported": True, "supported_category": True, "diff": json.dumps({"rendering": e})})
                continue
            ed = P.apply_record_edit(rec, e)
            col = ed.diff.get("column")
            sup = bool(ed.applicable and checker.supported(ed.record, changed=col))
            sup_b = bool(ed.applicable and checker.category_rule(ed.record, changed=col))   # rule (b) alone
            inst.append({"profile": pid, "edit": e, "klass": klass, "feature": FEATURE_OF[e], "applicable": ed.applicable,
                         "supported": sup, "supported_category": sup_b, "diff": json.dumps(ed.diff, default=str)})
            if ed.applicable:
                states.append({"profile": pid, "edit": e, "annotated": False, "text": to_json(to_state(ed.record))})
                if e in annotated_edits and ann_ok:
                    states.append({"profile": pid, "edit": e, "annotated": True, "text": P.render(ed.record, annotated=True)})
                base_recs.append({**rec, "_edit": e}); edit_recs.append({**ed.record, "_edit": e})
    inst = pd.DataFrame(inst)
    for n in refs:
        inst[f"d_prob_{n}"] = 0.0; inst[f"d_logit_{n}"] = 0.0; inst[f"q0_{n}"] = np.nan
    if base_recs:
        B, X = pd.DataFrame(base_recs), pd.DataFrame(edit_recs)
        key = pd.MultiIndex.from_arrays([B["SEQN"].astype(int), B["_edit"]])
        for n, m in refs.items():
            c = m.contrast(B, X)
            c.index = key
            idx = pd.MultiIndex.from_arrays([inst.profile, inst.edit])
            has = idx.isin(key)
            inst.loc[has, f"d_prob_{n}"] = c.loc[idx[has], "d_prob"].to_numpy()
            inst.loc[has, f"d_logit_{n}"] = c.loc[idx[has], "d_logit"].to_numpy()
            inst.loc[has, f"q0_{n}"] = c.loc[idx[has], "q0"].to_numpy()
        pd.concat([B.assign(_role="base"), X.assign(_role="edited")]).to_parquet(root / "data" / f"edited_{split}.parquet", index=False)
    st = pd.DataFrame(states)
    inst.to_parquet(root / "data" / f"instances_{split}.parquet", index=False)
    st.to_parquet(root / "data" / f"states_{split}.parquet", index=False)
    summary = inst.groupby(["klass", "edit"])[["applicable", "supported"]].mean().round(3)
    print(summary)
    print(f"states: {len(st)}  mean chars: {st.text.str.len().mean():.0f}")
    return inst, st


def build_wide(root: Path = ROOT, fraction: float = 1.0):
    """Baseline-only states (as recorded, and with the race field removed) for every complete-case record of the
    fitting and evaluation splits: the wide run for prediction and group calibration. Evaluation records reuse the
    identical texts, so their calls come from the raw-response cache at no cost."""
    df = pd.read_parquet(root / "data" / "cohort.parquet")
    keep = df.split.isin(["fit", "eval"]) & (df["complete9"] if "complete9" in df else True)
    if fraction < 1.0:   # a random share of fitting-set records, the same for every system;
        rng = np.random.default_rng(20260922)          # evaluation records are always kept (their calls are cached)
        keep &= (df.split == "eval") | (rng.random(len(df)) < fraction)
    states = []
    for _, row in df[keep].iterrows():
        rec = row.to_dict(); pid = int(rec["SEQN"])
        states.append({"profile": pid, "edit": "baseline", "annotated": False, "text": to_json(to_state(rec))})
        states.append({"profile": pid, "edit": "M2_race_removed", "annotated": False,
                       "text": P.render(rec, drop=P.PERT["mitigation"]["M2_race_removed"]["drop_fields"])})
    st = pd.DataFrame(states)
    st.to_parquet(root / "data" / "states_wide.parquet", index=False)
    print(f"wide states: {len(st):,} ({st.profile.nunique():,} records)")
    return st


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if split == "wide":
        frac = float(sys.argv[sys.argv.index("--fraction") + 1]) if "--fraction" in sys.argv else 1.0
        build_wide(fraction=frac); sys.exit(0)
    if split != "eval":
        sys.exit("usage: make_states.py eval | wide [--fraction f]")
    cohort = pd.read_parquet(ROOT / "data" / "cohort.parquet")
    ev = sorted(cohort.loc[cohort.split == "eval", "SEQN"].astype(int))
    rng = np.random.default_rng(20260922)
    ann = set(int(x) for x in rng.choice(ev, size=min(300, len(ev)), replace=False))
    clinical = list(P.PERT["clinical"])
    build("eval", [e for e in CLASS_OF], clinical, ann)

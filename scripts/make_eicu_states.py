"""eICU replication, step 3. For the demo's evaluation stays, every (record, edit) instance: edited record, serialised
text, support flag and the reference contrasts d (primary spline, LightGBM). Same file schema as the NHANES
make_states, under data/eicu/: states_eval.parquet, instances_eval.parquet, edited_eval.parquet. No API calls.
  python scripts/make_eicu_states.py eval
Baseline prediction only (the states run_eicu.py sends): --baseline writes only the unedited records' states
(data/eicu/states_eval.parquet) and needs neither the fitting set nor a reference.
  python scripts/make_eicu_states.py eval --baseline"""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import eicu as E

ROOT = Path(__file__).resolve().parents[1]
REFS = ("spline_logit", "lightgbm")


def edits_for() -> list[str]:
    ed = E.CFG["edits"]
    return list(ed["irrelevant"]) + list(ed["clinical"]) + list(ed["label"])


def build(split: str, root: Path = ROOT):
    d = root / E.CFG["out_dir"]
    demo = pd.read_parquet(d / "demo_cohort.parquet")
    fit = pd.read_parquet(d / "full_fit.parquet")
    sub = demo[demo.split == split]
    checker = E.support_checker(fit)
    refs = {n: joblib.load(d / f"reference_{n}.joblib") for n in REFS if (d / f"reference_{n}.joblib").exists()}
    if "spline_logit" not in refs:
        raise SystemExit("reference_spline_logit.joblib missing: run scripts/fit_eicu_reference.py first")
    inst, states, base_recs, edit_recs = [], [], [], []
    for _, row in sub.iterrows():
        rec = row.to_dict(); pid = int(rec["SEQN"])
        states.append({"profile": pid, "edit": "baseline", "annotated": False, "text": E.render(rec)})
        for e in edits_for():
            klass = E.klass_of(e)
            if klass == "irrelevant":
                states.append({"profile": pid, "edit": e, "annotated": False, "text": E.render(rec, irrelevant=e, seed=pid)})
                inst.append({"profile": pid, "edit": e, "klass": klass, "feature": "rendering", "applicable": True,
                             "supported": True, "supported_category": True, "diff": json.dumps({"rendering": e})})
                continue
            ed = E.apply_edit(rec, e)
            col = ed.diff.get("column") or E.CFG["edits"][klass][e]["column"]
            sup = bool(ed.applicable and checker.supported(ed.record, changed=col))
            sup_b = bool(ed.applicable and checker.category_rule(ed.record, changed=col))
            inst.append({"profile": pid, "edit": e, "klass": klass, "feature": col, "applicable": ed.applicable,
                         "supported": sup, "supported_category": sup_b, "diff": json.dumps(ed.diff, default=str)})
            if ed.applicable:
                states.append({"profile": pid, "edit": e, "annotated": False, "text": E.render(ed.record)})
                base_recs.append({**rec, "_edit": e}); edit_recs.append({**ed.record, "_edit": e})
    inst = pd.DataFrame(inst)
    for n in refs:
        inst[f"d_prob_{n}"] = 0.0; inst[f"d_logit_{n}"] = 0.0; inst[f"q0_{n}"] = np.nan
    if base_recs:
        B, X = pd.DataFrame(base_recs), pd.DataFrame(edit_recs)
        key = pd.MultiIndex.from_arrays([B["SEQN"].astype(int), B["_edit"]])
        idx = pd.MultiIndex.from_arrays([inst.profile, inst.edit]); has = idx.isin(key)
        for n, m in refs.items():
            c = m.contrast(B, X); c.index = key
            for k in ("d_prob", "d_logit"):
                inst.loc[has, f"{k}_{n}"] = c.loc[idx[has], k].to_numpy()
            inst.loc[has, f"q0_{n}"] = c.loc[idx[has], "q0"].to_numpy()
        pd.concat([B.assign(_role="base"), X.assign(_role="edited")]).to_parquet(d / f"edited_{split}.parquet", index=False)
    st = pd.DataFrame(states)
    inst.to_parquet(d / f"instances_{split}.parquet", index=False)
    st.to_parquet(d / f"states_{split}.parquet", index=False)
    print(inst.groupby(["klass", "edit"])[["applicable", "supported"]].mean().round(3))
    print(f"states: {len(st)}  records: {st.profile.nunique()}  mean chars: {st.text.str.len().mean():.0f}")
    return inst, st


def build_baseline(split: str, root: Path = ROOT) -> pd.DataFrame:
    """The unedited records of a demo split, serialised as the models see them (open demo data only)."""
    d = root / E.CFG["out_dir"]
    demo = pd.read_parquet(d / "demo_cohort.parquet")
    sub = demo[demo.split == split]
    st = pd.DataFrame([{"profile": int(r["SEQN"]), "edit": "baseline", "annotated": False, "text": E.render(r)}
                       for r in (row.to_dict() for _, row in sub.iterrows())])
    st.to_parquet(d / f"states_{split}.parquet", index=False)
    print(f"{split}: {len(st)} baseline states, mean {st.text.str.len().mean():.0f} characters -> {d / f'states_{split}.parquet'}")
    return st


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    split = args[0] if args else "eval"
    if split != "eval":
        raise SystemExit("make_eicu_states.py builds the evaluation stays: python scripts/make_eicu_states.py eval "
                         "[--baseline]")
    build_baseline(split) if "--baseline" in sys.argv else build(split)

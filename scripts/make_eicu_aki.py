"""eICU AKI set: build the records from eICU-CRD Demo v2.0.1, apply the KDIGO key, draw the sample and write the items
(config/eicu_aki.yaml). Demo data only: the source must be the open demo (ODbL licence file, at most
records.demo_stays_max unit stays); nothing from the credentialed full eICU is read. No network, no model call.

    python scripts/make_eicu_aki.py --demo <folder of the unzipped eICU-CRD Demo v2.0.1>
    python scripts/make_eicu_aki.py --demo <same> --check      # rebuild and compare with the written files

A record: age, sex, admission weight, every serum creatinine value from 7 days before to 72 h after unit admission with
its time from unit admission, and the time dialysis started if it did by 72 h (treatment table). Stays with at least
two values. Cleaning, before anything is shown: non-numeric results ("<0.50") dropped; two results at the same minute
reduced to the latest revision (labresultrevisedoffset); values shown with two decimals, and the key uses the shown
values.

Sets: patients on maintenance dialysis (any stay with APACHE's chronic dialysis flag, a dialysis past history or a
dialysis treatment "for chronic renal failure") and stays with no creatinine value in the first 72 h go to the side set,
a test of their own. The main set draws one stay per patient, stage by stage, rarest first (records.stage_order): per
stage a seeded shuffle of the stays of patients not yet drawn, one stay per patient, the first
records.parser_test_per_stage set aside (never scored) and the next records.per_stage to the main set. The side set then
takes one stay per patient from the patients not drawn (maintenance dialysis first). Every main record carries the
stage a baseline read either side would give (kdigo.highest_stage_either_side), for the names version's sensitivity
analysis. The "unknown" version asks every main and side record once, in two seeded halves (unknown_halves).
Writes data/eicu_aki/{items,side_items,unknown_first_items,unknown_last_items}.csv and prints the counts."""
from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import yaml

from jevity import kdigo as KD

CFG = yaml.safe_load((ROOT / "config" / "eicu_aki.yaml").read_text(encoding="utf-8"))
OUT = ROOT / "data" / "eicu_aki"
COLUMNS = ["item_id", "stage", "criterion", "record", "stay_id", "n_values", "n_values_icu", "dialysis_min",
           "stage_either_side", "baseline_sensitive"]
SIDE_COLUMNS = ["item_id", "subset", "stage", "criterion", "record", "stay_id", "n_values", "n_values_icu",
                "dialysis_min"]
UNKNOWN_COLUMNS = ["item_id", "set", "stage", "criterion", "record", "baseline_sensitive"]


def check_demo(demo: Path) -> None:
    lic = demo / "LICENSE.txt"
    if not lic.exists() or "Open Database License" not in lic.read_text(encoding="utf-8", errors="ignore"):
        raise SystemExit(f"{demo}: no ODbL licence file; only the open eICU-CRD demo may be used")
    n = len(pd.read_csv(demo / "patient.csv.gz", usecols=["patientunitstayid"]))
    if n > int(CFG["records"]["demo_stays_max"]):
        raise SystemExit(f"{demo}: {n} unit stays; the demo has 2,520. The credentialed eICU never enters this set")


def read_demo(demo: Path) -> dict[str, pd.DataFrame]:
    p = pd.read_csv(demo / "patient.csv.gz", dtype={"age": str},
                    usecols=["patientunitstayid", "uniquepid", "age", "gender", "admissionweight"])
    lab = pd.read_csv(demo / "lab.csv.gz", usecols=["patientunitstayid", "labresultoffset", "labresultrevisedoffset",
                                                     "labname", "labresult"])
    tr = pd.read_csv(demo / "treatment.csv.gz", usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"])
    aps = pd.read_csv(demo / "apacheApsVar.csv.gz", usecols=["patientunitstayid", "dialysis"])
    ph = pd.read_csv(demo / "pastHistory.csv.gz", usecols=["patientunitstayid", "pasthistorypath"])
    return {"patient": p, "creatinine": lab[lab.labname == "creatinine"].drop(columns="labname"), "treatment": tr,
            "apache": aps, "history": ph}


def creatinine_values(cr: pd.DataFrame, k: dict) -> tuple[pd.DataFrame, dict]:
    """In the record window, numeric, one value per stay and minute (the latest revision); value as shown (2 decimals)."""
    n0 = len(cr)
    w = cr[(cr.labresultoffset >= int(k["record_from_min"])) & (cr.labresultoffset <= int(k["record_to_min"]))]
    n_window = len(w)
    w = w[w.labresult.notna()]
    n_numeric = len(w)
    w = w.reset_index().sort_values(["patientunitstayid", "labresultoffset", "labresultrevisedoffset", "index"])
    w = w.drop_duplicates(["patientunitstayid", "labresultoffset"], keep="last")
    w = w.assign(value=[Decimal(f"{x:.2f}") for x in w.labresult])
    return w[["patientunitstayid", "labresultoffset", "value"]], {
        "creatinine_rows": n0, "in_window": n_window, "non_numeric_dropped": n_window - n_numeric,
        "same_minute_reduced": n_numeric - len(w)}


def dialysis_start(tr: pd.DataFrame, k: dict) -> pd.Series:
    """First dialysis treatment per stay (minutes); every dialysis-like string must be listed in the config."""
    yes, no = set(k["dialysis_treatments"]), set(k["not_dialysis"])
    seen = set(tr.treatmentstring[tr.treatmentstring.str.contains("dialysis", case=False, na=False)])
    unknown = sorted(seen - yes - no)
    if unknown:
        raise SystemExit(f"treatment strings with 'dialysis' not classified in config kdigo: {unknown}")
    d = tr[tr.treatmentstring.isin(yes)]
    return d.groupby("patientunitstayid").treatmentoffset.min()


def chronic_dialysis_patients(d: dict[str, pd.DataFrame]) -> set:
    """Patients (uniquepid) on maintenance dialysis: any of their stays shows one of records.chronic_dialysis."""
    c = CFG["records"]["chronic_dialysis"]
    aps, ph, tr = d["apache"], d["history"], d["treatment"]
    stays = set(aps.loc[aps[c["apache_flag"]] == 1, "patientunitstayid"])
    stays |= set(ph.loc[ph.pasthistorypath.str.contains(c["past_history"], case=False, na=False), "patientunitstayid"])
    stays |= set(tr.loc[tr.treatmentstring.str.contains(c["treatment"], case=False, na=False, regex=False),
                        "patientunitstayid"])
    p = d["patient"]
    return set(p.loc[p.patientunitstayid.isin(stays), "uniquepid"])


def age_text(a) -> str:
    if not isinstance(a, str) or not a.strip():
        return "not recorded"
    a = a.strip()
    return "over 89 years" if a.startswith(">") else f"{int(a)} years"


def record_text(age, sex, weight, values: list[tuple[int, Decimal]], dialysis_min: int | None, k: dict) -> str:
    sx = sex.lower() if isinstance(sex, str) and sex.strip() else "not recorded"
    wt = f"{float(weight):.1f} kg" if pd.notna(weight) else "not recorded"
    lines = [f"Age: {age_text(age)}", f"Sex: {sx}", f"Admission weight: {wt}",
             "Serum creatinine (mg/dL), from 7 days before to 72 hours after intensive care unit admission; times are "
             "from unit admission:"]
    lines += [f"{KD.fmt_time(t)}: {KD.fmt_value(v)}" for t, v in sorted(values, key=lambda x: x[0])]
    shown = dialysis_min is not None and dialysis_min <= int(k["record_to_min"])
    lines.append(f"Dialysis started: {KD.fmt_time(int(dialysis_min)) if shown else 'no'}")
    return "\n".join(lines)


def build(demo: Path) -> tuple[pd.DataFrame, dict]:
    """Every eligible stay with its record, key, sensitivity key and flags; counts."""
    k = CFG["kdigo"]
    rule = KD.Rule.from_config(k)
    d = read_demo(demo)
    vals, counts = creatinine_values(d["creatinine"], k)
    dz = dialysis_start(d["treatment"], k)
    chronic = chronic_dialysis_patients(d)
    by = {s: list(zip(g.labresultoffset.astype(int), g.value)) for s, g in vals.groupby("patientunitstayid")}
    rows = []
    for r in d["patient"].sort_values("patientunitstayid").itertuples():
        v = by.get(r.patientunitstayid, [])
        if len(v) < int(CFG["records"]["min_values"]):
            continue
        dm = dz.get(r.patientunitstayid)
        dm = int(dm) if dm is not None and pd.notna(dm) and int(dm) <= int(k["record_to_min"]) else None
        f = KD.highest_stage(v, dm, rule)
        alt = KD.highest_stage_either_side(v, dm, rule)
        rows.append({"stay_id": int(r.patientunitstayid), "patient": r.uniquepid, "stage": KD.LEVELS[f.stage],
                     "criterion": f.criterion, "record": record_text(r.age, r.gender, r.admissionweight, v, dm, k),
                     "n_values": len(v), "n_values_icu": sum(int(k["assess_from_min"]) <= t <= int(k["assess_to_min"])
                                                             for t, _ in v),
                     "dialysis_min": "" if dm is None else str(dm),
                     "stage_either_side": KD.LEVELS[alt], "baseline_sensitive": str(alt != f.stage).lower(),
                     "chronic_dialysis": r.uniquepid in chronic})
    df = pd.DataFrame(rows)
    counts.update({"unit_stays": len(d["patient"]), "eligible": len(df),
                   "eligible_by_stage": {lv: int((df.stage == lv).sum()) for lv in KD.LEVELS},
                   "chronic_dialysis_stays": int(df.chronic_dialysis.sum()),
                   "no_value_in_icu_stays": int((df.n_values_icu == 0).sum())})
    return df, counts


def sample(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main set, the stays set aside and side set (see the module docstring). Item ids in a seeded order: main
    aki0001..., set aside akitest1..., side akiside001..."""
    rc = CFG["records"]
    rng = np.random.default_rng(int(CFG["seed"]))
    n_test, n_main = int(rc["parser_test_per_stage"]), int(rc["per_stage"])
    pool = df[~df.chronic_dialysis & (df.n_values_icu > 0)]
    used, main, test = set(), [], []
    for lv in rc["stage_order"]:
        c = pool[(pool.stage == lv) & ~pool.patient.isin(used)].sort_values("stay_id")
        c = c.iloc[rng.permutation(len(c))]
        if rc.get("one_stay_per_patient", True):
            c = c.drop_duplicates("patient")
        take = c.iloc[:n_test + n_main]
        used |= set(take.patient)
        test += take.stay_id.iloc[:n_test].tolist()
        main += take.stay_id.iloc[n_test:].tolist()
    side = df[(df.chronic_dialysis | (df.n_values_icu == 0)) & ~df.patient.isin(used)].sort_values("stay_id")
    side = side.assign(subset=np.where(side.chronic_dialysis, "maintenance_dialysis", "no_value_in_icu"))
    side = side.iloc[rng.permutation(len(side))].sort_values("subset", kind="stable").drop_duplicates("patient")
    main = [main[i] for i in rng.permutation(len(main))]
    side_ids = side.stay_id.tolist()
    side_ids = [side_ids[i] for i in rng.permutation(len(side_ids))]
    ix = df.set_index("stay_id")
    m = ix.loc[main].reset_index().assign(item_id=[f"aki{i + 1:04d}" for i in range(len(main))])
    t = ix.loc[test].reset_index().assign(item_id=[f"akitest{i + 1}" for i in range(len(test))])
    s = side.set_index("stay_id").loc[side_ids].reset_index().assign(
        item_id=[f"akiside{i + 1:03d}" for i in range(len(side_ids))])
    return m[COLUMNS], t[COLUMNS], s[SIDE_COLUMNS]


def unknown_halves(m: pd.DataFrame, s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The "unknown" version's two halves: every main and side record once; within each set x stage x
    baseline-sensitive cell a seeded shuffle, half to "unknown" first and half to "unknown" last (an odd cell's extra
    record to a seeded side). Rows in item id order."""
    rng = np.random.default_rng(int(CFG["seed"]) + 1)
    allr = pd.concat([m.assign(set="main"), s.assign(set=s.subset, baseline_sensitive="false")], ignore_index=True)
    first, last = [], []
    for _, g in allr.sort_values("item_id").groupby(["set", "stage", "baseline_sensitive"], sort=True):
        ids = g.item_id.to_numpy()[rng.permutation(len(g))]
        k = len(ids) // 2 + (int(rng.integers(2)) if len(ids) % 2 else 0)
        first += ids[:k].tolist()
        last += ids[k:].tolist()
    ix = allr.set_index("item_id")
    out = []
    for ids in (first, last):
        out.append(ix.loc[sorted(ids)].reset_index()[UNKNOWN_COLUMNS])
    return out[0], out[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", default=str(ROOT / CFG["records"]["demo_dir"]))
    ap.add_argument("--check", action="store_true", help="rebuild and compare with data/eicu_aki; write nothing")
    a = ap.parse_args()
    demo = Path(a.demo)
    check_demo(demo)
    df, counts = build(demo)
    m, t, s = sample(df)
    print(f"unit stays {counts['unit_stays']}; creatinine rows {counts['creatinine_rows']}, in the window "
          f"{counts['in_window']}, non-numeric dropped {counts['non_numeric_dropped']}, same-minute results reduced "
          f"{counts['same_minute_reduced']}")
    print(f"eligible stays (at least {CFG['records']['min_values']} values): {counts['eligible']}; of patients on "
          f"maintenance dialysis {counts['chronic_dialysis_stays']}; with no value in the first 72 h "
          f"{counts['no_value_in_icu_stays']}")
    print("eligible by stage:", counts["eligible_by_stage"])
    print("main set by stage:", m.stage.value_counts().reindex(KD.LEVELS).to_dict(), f"({len(m)} records, "
          f"{len(m)} patients)")
    print("main set, criterion by stage:\n" + pd.crosstab(m.stage, m.criterion).reindex(KD.LEVELS).to_string())
    print("main set, the key against a baseline read either side:\n" +
          pd.crosstab(m.stage, m.stage_either_side).reindex(index=KD.LEVELS, columns=KD.LEVELS, fill_value=0).to_string())
    print(f"baseline-sensitive main records: {int((m.baseline_sensitive == 'true').sum())}")
    print("side set, subset by stage:\n" + pd.crosstab(s.subset, s.stage).to_string())
    uf, ul = unknown_halves(m, s)
    both = pd.concat([uf.assign(pos="first"), ul.assign(pos="last")], ignore_index=True)
    print(f"unknown version: first {len(uf)}, last {len(ul)} records; by set and stage:\n"
          + pd.crosstab([both.set, both.stage], both.pos).to_string())
    files = (("items.csv", m), ("side_items.csv", s), ("unknown_first_items.csv", uf), ("unknown_last_items.csv", ul))
    if a.check:
        same = all(pd.read_csv(OUT / f, dtype=str, keep_default_na=False).equals(x.astype(str)) for f, x in files)
        print("check:", "the written files match" if same else "DIFFERENT from the written files")
        return 0 if same else 1
    OUT.mkdir(parents=True, exist_ok=True)
    for f, x in files:
        x.to_csv(OUT / f, index=False, lineterminator="\n")
    print(f"wrote {', '.join(f for f, _ in files)} under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Supplementary Table 6: answers and correctness by confidence level, for every task, form of question and system.

Reads the stored answers on kidney-injury staging, patient messages and board questions, with their keys, and the
cancer-staging summary on the pool of 240 reports (results/summaries/staging_pool240.json, written by
scripts/summaries/staging_pool240.py). Writes, under --out:
  confidence_table.json         one row per task, form and system: Jev 1.13, each LLM, and each pairing of Jev with an
                                LLM in the same form (system "Jev + <LLM>": Jev's answer where Jev was fully confident,
                                the LLM's answer otherwise; listed where Jev gave any fully confident answer)
  confidence_pairs_other.json   the pairings on kidney injury, patient messages and board questions (every pairing,
                                with "kept", the number of Jev's answers kept)
  confidence_free_other.json    the five free-plan and open-weight models on those three tasks

Each row: n; unusable (no valid answer, or no probability for it); noprob (a valid answer without a probability);
right; the answers at each confidence level, right and wrong (u, below 0.50; s, 0.50 to below 0.75; v, 0.75 to below
1.00; c, 1.00, within 1e-9); auc, the AUROC of confidence for a right answer over the usable answers (None when there
is no right or no wrong answer). Confidence is the probability of the chosen answer (for Jev's four-question staging
forms, the product of the four). Answers: repeat 0, the first stored answer per item.

The cancer-staging rows are taken from the "t6" and "t6_combo" rows of the 240-report summary. The auc of the rows of
Jev 1.13 on its own (every task and form), and of the five main LLMs on their own on kidney injury, patient messages
and board questions, is rounded to 3 decimals; the table prints the auc to 2 decimals from these values. Other auc
values are written in full.

The kidney-injury rows need data/eicu_aki/items.csv, which scripts/make_eicu_aki.py builds from the eICU
Collaborative Research Database demo. Without it they are left out and the other rows are unchanged.

    python scripts/summaries/confidence_table.py [--root .] [--out results/summaries]
        [--pool results/summaries/staging_pool240.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-9
NAME = {"gpt": "GPT-5.6 Sol", "claude": "Claude Opus 5.5", "gemini": "Gemini 3.8 Flash", "muse": "Muse Spark 1.1",
        "glm": "GLM-5.3", "gpt_free": "GPT-5.6 Luna", "claude_free": "Claude Sonnet 5",
        "gemini_free": "Gemini 3.5 Flash-Lite", "medgemma": "MedGemma 27B", "gemma": "Gemma 3 27B"}
MAIN = ["gpt", "claude", "gemini", "muse", "glm"]
FREE = ["gpt_free", "claude_free", "gemini_free", "medgemma", "gemma"]
JEV = "Jev 1.13"

COLON = "Colon cancer stage (240 reports)"
KIDNEY = "Kidney-injury stage (561 stays)"
MESSAGES = "Test-or-treat decisions (180 vignettes)"
BOARD = "Board examination (1,571 questions)"

KIDNEY_ITEMS = "data/eicu_aki/items.csv"
KIDNEY_CALLS = "results/eicu_aki/calls.parquet"
MESSAGES_CALLS = "results/arm2/calls.parquet"
BOARD_ITEMS = "data/arm2_ext/items.csv"
BOARD_CALLS = "results/arm2_ext/calls.parquet"
POOL = "results/summaries/staging_pool240.json"

# forms of the staging question (the 240-report summary's keys) and their labels in the table
FORM = {"definitions": "Each level's T, N and M categories", "structure_only": "Four questions, bare options",
        "documented_no_examples": "Four questions, options defined without examples", "documented": "Recommended form",
        "documented_notes": "Recommended form with AJCC notes"}
FORM_ALL = {"names": "Level names (same input as the LLMs)", "named_field": "Level names, report sent as a named field",
            **FORM}
LLM_NAMES = "Level names"
COUNTS = ["n", "unusable", "noprob", "right", "u_right", "u_wrong", "s_right", "s_wrong", "v_right", "v_wrong",
          "c_right", "c_wrong"]


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def dist(valid, tp, ok):
    """Answers by confidence level: counts, unusable answers and the AUROC of confidence for a right answer."""
    valid = np.asarray(valid, bool)
    tp = np.asarray(tp, float)
    ok = np.asarray(ok, bool)
    noprob = valid & np.isnan(tp)
    unus = (~valid) | noprob
    t = np.nan_to_num(tp)
    bands = {"u": (t < 0.5), "s": (t >= 0.5) & (t < 0.75), "v": (t >= 0.75) & (t < 1 - TOL), "c": t >= 1 - TOL}
    r = {"n": int(len(ok)), "unusable": int(unus.sum()), "noprob": int(noprob.sum()), "right": int((ok & ~unus).sum())}
    for k, m in bands.items():
        m = m & ~unus
        r[k + "_right"] = int((m & ok).sum())
        r[k + "_wrong"] = int((m & ~ok).sum())
    use = ~unus
    r["auc"] = auroc(tp[use], ok[use])
    return r


def tobool(x):
    return x.astype(str).str.lower().eq("true") if x.dtype == object else x.fillna(False).astype(bool)


def sel(c, s, v):
    """One system's answers in one form: repeat 0, the first stored answer per item."""
    return c[(c.system == s) & (c.variant == v) & (c["repeat"] == 0)].drop_duplicates("item_id").set_index("item_id")


def alone(task, form, L, truth, system):
    """One system on its own."""
    L = L.reindex(truth.index)
    lv = tobool(L.valid).to_numpy()
    lt = pd.to_numeric(L.top_prob, errors="coerce").to_numpy()
    return dict(task=task, system=system, form=form, **dist(lv, lt, lv & (L.answer.to_numpy() == truth.to_numpy())))


def paired(task, form, J, L, truth, system):
    """Jev's answer where Jev was fully confident, the LLM's answer otherwise."""
    J = J.reindex(truth.index)
    L = L.reindex(truth.index)
    jv = tobool(J.valid).to_numpy()
    lv = tobool(L.valid).to_numpy()
    jt = pd.to_numeric(J.top_prob, errors="coerce").to_numpy()
    lt = pd.to_numeric(L.top_prob, errors="coerce").to_numpy()
    keep = jv & (np.nan_to_num(jt) >= 1 - TOL)
    tr = truth.to_numpy()
    ok = np.where(keep, J.answer.to_numpy() == tr, lv & (L.answer.to_numpy() == tr))
    r = dict(task=task, system=system, form=form, **dist(np.where(keep, True, lv), np.where(keep, jt, lt), ok))
    r["kept"] = int(keep.sum())
    return r


def other_tasks(root: Path) -> list:
    """(task, stored answers, [(variant, form label, key)]) for kidney injury, patient messages and board questions."""
    out = []
    if (root / KIDNEY_ITEMS).exists():
        it = pd.read_csv(root / KIDNEY_ITEMS, dtype=str).set_index("item_id")
        out.append((KIDNEY, pd.read_parquet(root / KIDNEY_CALLS),
                    [("names", "Stage names", it.stage), ("definitions", "Stage names with KDIGO criteria", it.stage)]))
    else:
        print(f"{KIDNEY_ITEMS} not found: the kidney-injury rows are left out")
    c = pd.read_parquet(root / MESSAGES_CALLS)
    ids = pd.Index(sorted(c[(c.system == "jev") & (c["repeat"] == 0)].item_id.unique()))
    out.append((MESSAGES, c, [("main", "As written", pd.Series(np.where(ids.str.contains("_lv_"), "B", "A"), index=ids))]))
    it = pd.read_csv(root / BOARD_ITEMS, dtype=str).set_index("item_id")
    out.append((BOARD, pd.read_parquet(root / BOARD_CALLS),
                [("medhelm", "Original options", it.truth), ("other_count", "Other option count", it.truth_alt)]))
    return out


def round_auc(r: dict) -> dict:
    r["auc"] = None if r["auc"] is None else round(r["auc"], 3)
    return r


def pick(task, system, form, p):
    return {"task": task, "system": system, "form": form, **{k: p[k] for k in COUNTS}, "auc": p["auc"]}


def main(root: Path, out: Path, pool_path: Path) -> list:
    P = json.loads(pool_path.read_text(encoding="utf-8"))
    t6 = {(r["system"], r["form"]): r for r in P["t6"]}

    # the other three tasks: each system alone (Jev and the five main LLMs), every pairing, the five other models
    base, pairs, free = [], [], []
    for task, c, forms in other_tasks(root):
        for v, lab, tr in forms:
            base.append(round_auc(alone(task, lab, sel(c, "jev", v), tr, JEV)))
            for s in MAIN:
                base.append(round_auc(alone(task, lab, sel(c, s, v), tr, NAME[s])))
            for s in MAIN + FREE:
                pairs.append(paired(task, lab, sel(c, "jev", v), sel(c, s, v), tr, "Jev + " + NAME[s]))
                if s in FREE:
                    a = alone(task, lab, sel(c, s, v), tr, NAME[s])
                    a["kept"] = None
                    free.append(a)

    # cancer staging: Jev in every form, the five main LLMs with level names, then the five main LLMs in the other forms
    colon = [round_auc(pick(COLON, JEV, lab, t6[("jev", fm)])) for fm, lab in FORM_ALL.items()]
    colon += [pick(COLON, NAME[s], LLM_NAMES, t6[(s, "names")]) for s in MAIN if (s, "names") in t6]
    colon += [pick(COLON, NAME[s], lab, t6[(s, fm)]) for fm, lab in FORM.items() for s in MAIN if (s, fm) in t6]
    rows = colon + base

    # pairings and the five other models, each placed after the last row of its task
    extra = [pick(COLON, "Jev + " + NAME[p["system"]], FORM_ALL[p["form"]], p) for p in P.get("t6_combo", []) if p["kept"] != 0]
    extra += [pick(COLON, NAME[p["system"]], LLM_NAMES if p["form"] == "names" else FORM_ALL[p["form"]], p)
              for p in P["t6"] if p["system"] in FREE and p["form"] != "named_field"]
    extra += [pick(p["task"], p["system"], p["form"], p) for p in free]
    extra += [pick(p["task"], p["system"], p["form"], p) for p in pairs if p["kept"] != 0]
    for x in extra:
        last = max(i for i, r in enumerate(rows) if r["task"] == x["task"])
        rows.insert(last + 1, x)

    out.mkdir(parents=True, exist_ok=True)
    for name, obj, indent in [("confidence_table.json", rows, 1), ("confidence_pairs_other.json", pairs, 0),
                              ("confidence_free_other.json", free, 0)]:
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=indent)
    print(f"{len(rows)} rows in {out / 'confidence_table.json'}; {len(pairs)} pairings, {len(free)} other-model rows")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/summaries")
    ap.add_argument("--pool", default=POOL)
    a = ap.parse_args()
    root = Path(a.root)
    out, pool = Path(a.out), Path(a.pool)
    main(root, out if out.is_absolute() else root / out, pool if pool.is_absolute() else root / pool)

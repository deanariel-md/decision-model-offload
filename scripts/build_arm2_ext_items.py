"""Arm 2, second item set: the questions of MedQA (US, 4 options, test set) and Medbullets (5 options), the two MedHELM
clinical decision support benchmarks with case vignettes, in the versions MedHELM loads (crfm-helm MedQAScenario:
data_clean.zip, questions/US/4_options/phrases_no_exclude_test.jsonl; MedBulletsScenario: medbullets_op5.csv). Every
question enters; the next-step, management and treatment questions are a subgroup, labelled from the lead-in only (the
last sentence holding a question mark, else the last sentence) by the patterns in config/arm2_ext.yaml source; the rule
never sees the options, the key or any answer. Medbullets repeats 10 questions word for word (same options and key,
another link); the first copy enters and the repeats are listed. Key = the dataset's answer letter. Stem verbatim;
option texts as HELM builds its references (MedQA: a trailing quote and spaces stripped), in the dataset's order. The
same questions with the other option count sit beside them: MedQA's 5-option US test file (the 4-option file is it with
one wrong option deleted, line for line) and Medbullets op4 (matched on question and link). Source files are downloaded
from the URLs in config/arm2_ext.yaml, cached under data/raw/arm2_ext and checked against the config hashes. A build
writes data/arm2_ext/items_manifest.json with the item file's SHA-256: equal to config/arm2_ext/items_manifest.json, it
confirms the same text (the repeats it lists: config/arm2_ext/duplicates.csv).
  python scripts/build_arm2_ext_items.py --download     # fetch the sources (once), build the items
  python scripts/build_arm2_ext_items.py [--check]      # from the cache; --check: compare with the stored items
Writes data/arm2_ext/items.csv, data/arm2_ext/duplicates.csv and data/arm2_ext/items_manifest.json; prints the counts
per source and seeded random stems inside and outside the subgroup."""
import argparse, hashlib, html, io, json, re, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "arm2_ext"
COLUMNS = ["item_id", "source", "source_index", "meta_info", "link", "decision_question", "lead_in_type",
           "question_sentence", "n_options", "truth", "answer_text", "options", "n_options_alt", "truth_alt",
           "options_alt", "stem"]
_SENT = re.compile(r"(?<=[.!?])\s+|\n+|\s+(?=(?:Which|What)\b)")   # Medbullets runs lab tables into the lead-in


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str) -> bytes:
    """GET with Google Drive's large-file page handled: the page's download form is followed once (the file HELM's
    ensure_file_downloaded fetches)."""
    import httpx
    with httpx.Client(follow_redirects=True, timeout=600) as c:
        r = c.get(url)
        r.raise_for_status()
        if "text/html" in r.headers.get("content-type", "") and "drive" in url:
            form = re.search(r'<form[^>]+action="([^"]+)"', r.text)
            fields = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', r.text))
            if not form:
                raise SystemExit(f"unexpected page from {url}")
            r = c.get(html.unescape(form.group(1)), params=fields)
            r.raise_for_status()
        return r.content


def cached(cfg: dict, spec: dict, download: bool) -> bytes:
    f = ROOT / cfg["source"]["cache_dir"] / spec["file"]
    if not f.exists():
        if not download:
            raise SystemExit(f"{f} missing: run with --download (source {spec['url']})")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(fetch(spec["url"]))
    b = f.read_bytes()
    want = spec.get("zip_sha256") or spec["sha256"]
    if sha256(b) != want:
        raise SystemExit(f"{f}: SHA-256 {sha256(b)} differs from config {want}")
    return b


def source_bytes(cfg: dict, download: bool) -> dict[str, bytes]:
    """The source files as HELM reads them (and the other option count), checked against the config hashes."""
    s = cfg["source"]
    z = zipfile.ZipFile(io.BytesIO(cached(cfg, s["medqa"], download)))
    out = {}
    for key, member, h in (("medqa", s["medqa"]["member"], s["medqa"]["member_sha256"]),
                           ("medqa_alt", s["medqa_alt"]["member"], s["medqa_alt"]["member_sha256"])):
        out[key] = z.read(member)
        assert sha256(out[key]) == h, f"{member}: hash differs from config"
    out["medbullets"] = cached(cfg, s["medbullets"], download)
    out["medbullets_alt"] = cached(cfg, s["medbullets_alt"], download)
    return out


def question_sentence(stem: str) -> str:
    parts = [p.strip() for p in _SENT.split(stem.strip()) if p.strip()]
    q = [p for p in parts if "?" in p]
    return (q or parts)[-1]


def lead_in_type(sentence: str, keep: dict[str, str], drop: dict[str, str]) -> str:
    """The first subgroup pattern the lead-in matches, '' when none does or a drop pattern does."""
    hit = next((k for k, p in keep.items() if re.search(p, sentence, re.I)), "")
    return "" if hit and any(re.search(p, sentence, re.I) for p in drop.values()) else hit


def _norm(s: str) -> str:
    return " ".join(str(s).split())


def records(src: dict[str, bytes]) -> tuple[list[dict], list[dict]]:
    """Every question of both sources as HELM builds its instances, with the other option count beside it; and the
    Medbullets repeats left out."""
    out, dups = [], []
    q4 = [json.loads(l) for l in src["medqa"].decode("utf-8").splitlines()]
    q5 = [json.loads(l) for l in src["medqa_alt"].decode("utf-8").splitlines()]
    assert len(q4) == len(q5)
    for i, (ex, alt) in enumerate(zip(q4, q5)):
        assert len(ex["options"]) == 4 and len(alt["options"]) == 5 and ex["question"] == alt["question"], i
        opts = {k: v.rstrip('"').rstrip() for k, v in ex["options"].items()}
        oalt = {k: v.rstrip('"').rstrip() for k, v in alt["options"].items()}
        assert set(opts.values()) <= set(oalt.values()) and opts[ex["answer_idx"]] == oalt[alt["answer_idx"]], i
        out.append({"item_id": f"medqa_{i:04d}", "source": "medqa", "source_index": i, "meta_info": ex["meta_info"],
                    "link": "", "stem": ex["question"].strip(), "options": opts, "truth": ex["answer_idx"],
                    "answer_text": ex["answer"].rstrip('"').rstrip(), "options_alt": oalt, "truth_alt": alt["answer_idx"]})
    m5 = pd.read_csv(io.BytesIO(src["medbullets"]), dtype=str, keep_default_na=False, encoding="utf-8")
    m4 = pd.read_csv(io.BytesIO(src["medbullets_alt"]), dtype=str, keep_default_na=False, encoding="utf-8")
    alt_by = {(_norm(r["question"]), r["link"]): r for r in m4.to_dict("records")}
    seen: dict[tuple, str] = {}
    for i, r in enumerate(m5.to_dict("records")):
        if not r.get("question") or not r.get("answer_idx") or not r.get("opa"):     # HELM skips these rows
            continue
        opts = {l: r[c] for l, c in zip("ABCDE", ("opa", "opb", "opc", "opd", "ope"))}
        iid = f"medbullets_{i:03d}"
        key = (_norm(r["question"]), tuple(_norm(v) for v in opts.values()), r["answer_idx"].strip())
        if key in seen:
            dups.append({"item_id": iid, "source_index": i, "link": r["link"], "same_as": seen[key]})
            continue
        seen[key] = iid
        a = alt_by[(_norm(r["question"]), r["link"])]
        oalt = {l: a[c] for l, c in zip("ABCD", ("opa", "opb", "opc", "opd"))}
        assert {_norm(v) for v in oalt.values()} <= {_norm(v) for v in opts.values()}, iid
        assert _norm(oalt[a["answer_idx"].strip()]) == _norm(opts[r["answer_idx"].strip()]), iid
        out.append({"item_id": iid, "source": "medbullets", "source_index": i, "meta_info": "", "link": r["link"],
                    "stem": r["question"].strip(), "options": opts, "truth": r["answer_idx"].strip(),
                    "answer_text": r["answer"].strip(), "options_alt": oalt, "truth_alt": a["answer_idx"].strip()})
    for x in out:
        for o, t in (("options", "truth"), ("options_alt", "truth_alt")):
            assert x[t] in x[o] and len(set(x[o].values())) == len(x[o]) and all(x[o].values()), (x["item_id"], o)
    return out, dups


def build(cfg: dict, src: dict[str, bytes]) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = cfg["source"]
    recs, dups = records(src)
    rows = []
    for x in recs:
        q = question_sentence(x["stem"])
        t = lead_in_type(q, s["subgroup_patterns"], s.get("subgroup_exclude") or {})
        rows.append({**x, "question_sentence": q, "lead_in_type": t, "decision_question": bool(t),
                     "n_options": len(x["options"]), "n_options_alt": len(x["options_alt"]),
                     "options": json.dumps(x["options"], ensure_ascii=False),
                     "options_alt": json.dumps(x["options_alt"], ensure_ascii=False)})
    items = pd.DataFrame(rows)[COLUMNS].sort_values("item_id").reset_index(drop=True)
    return items, pd.DataFrame(dups, columns=["item_id", "source_index", "link", "same_as"])


def samples(items: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng([seed, 2])
    ins, out = items[items.decision_question], items[~items.decision_question]
    return (ins.iloc[np.sort(rng.choice(len(ins), size=min(n, len(ins)), replace=False))],
            out.iloc[np.sort(rng.choice(len(out), size=min(n, len(out)), replace=False))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with the stored items")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config" / "arm2_ext.yaml").read_text(encoding="utf-8"))
    src = source_bytes(cfg, a.download)
    items, dups = build(cfg, src)
    if a.check:
        old = pd.read_csv(OUT / "items.csv", dtype=str, keep_default_na=False)
        same = old.equals(items.astype(str))
        print("items.csv matches a rebuild" if same else "items.csv DIFFERS from a rebuild")
        sys.exit(0 if same else 1)
    OUT.mkdir(parents=True, exist_ok=True)
    items.to_csv(OUT / "items.csv", index=False, lineterminator="\n")
    dups.to_csv(OUT / "duplicates.csv", index=False, lineterminator="\n")
    s = cfg["source"]
    g = items.groupby("source")
    man = {"built_by": "scripts/build_arm2_ext_items.py", "config": "config/arm2_ext.yaml",
           "sources": {k: s[k] for k in ("medqa", "medqa_alt", "medbullets", "medbullets_alt")},
           "items_by_source": g.size().to_dict(), "medbullets_repeats_left_out": len(dups),
           "subgroup_by_source": g.decision_question.sum().astype(int).to_dict(),
           "lead_in_type_by_source": items[items.decision_question].groupby(["source", "lead_in_type"]).size()
                                     .unstack(fill_value=0).to_dict("index"),
           "options_by_source": g.n_options.first().astype(int).to_dict(),
           "options_alt_by_source": g.n_options_alt.first().astype(int).to_dict(),
           "truth_by_source": items.groupby(["source", "truth"]).size().unstack(fill_value=0).to_dict("index"),
           "medqa_meta_info": items[items.source == "medqa"].meta_info.value_counts().to_dict(),
           "items_sha256": sha256((OUT / "items.csv").read_bytes()), "n_items": len(items)}
    (OUT / "items_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"items {man['items_by_source']} ({len(items)}; Medbullets repeats left out {len(dups)}); subgroup "
          f"{man['subgroup_by_source']}; lead-in types {man['lead_in_type_by_source']}")
    print(f"truth letters {man['truth_by_source']}; MedQA steps {man['medqa_meta_info']}")
    k, d = samples(items, int(s.get("n_samples", 10)), int(cfg["seed"]))
    for title, df in (("inside the subgroup", k), ("outside the subgroup", d)):
        print(f"\n== {len(df)} random stems {title}")
        for r in df.to_dict("records"):
            print(f"-- {r['item_id']} ({r['lead_in_type'] or 'lead-in: ' + r['question_sentence']}): {r['stem']}")


if __name__ == "__main__":
    main()

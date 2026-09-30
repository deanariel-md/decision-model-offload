"""Arm 3, Jev used as documented: config, the two versions' templates, the examples, parts -> stage, distributions,
calls table, dry run. On synthetic reports (tests/synthetic_items.py: config/docuse_staging.yaml's item files and
source arm, in memory) with synthetic Jev responses and simulated chatbot answers (categorical_sim); no network."""
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "docuse"))

from jevity import categorical as K
from jevity import categorical_analysis as KA
from jevity import crc
from jevity import docuse_staging as D
from jevity.clients import MODELS, RawStore
import synthetic_items as SI

CHATBOTS = ["gpt", "claude", "gemini", "muse", "glm"]


@pytest.fixture(scope="module", autouse=True)
def synthetic_reports():
    with pytest.MonkeyPatch.context() as mp:
        SI.use_in_docuse(mp)
        yield


@pytest.fixture(scope="module")
def arm3_answers(tmp_path_factory):
    """Arm 3's chatbot answers as its run stored them (the shared clients' rows), simulated
    (categorical_sim.SimulatedSender) on the synthetic reports: the five main chatbots, both versions, repeat 0."""
    from jevity.categorical_sim import SimulatedSender
    arm = SI.arm("arm3")
    its = K.load_items(arm)
    fac = K.make_factory(arm, RawStore(tmp_path_factory.mktemp("arm3_store")), sender=SimulatedSender(arm, its),
                         standard=True)
    calls = [K.CatCall(s, i.item_id, 0, i.variant) for s in CHATBOTS for i in its]
    return K.execute(calls, its, fac, None, workers=8, arm_name=arm.name)


@pytest.fixture(scope="module")
def items():
    return D.load_items()


@pytest.fixture(scope="module")
def frames():
    return D.load_frames()


@pytest.fixture(scope="module")
def reports(frames):
    return [r.strip() for f in frames.values() for r in f.report]


def _one(items, v):
    return [i for i in items if i.variant == v]


# ------------------------------------------------------------------------------------------------ config
def test_every_config_text_names_its_source():
    c = D.CFG
    for d in c["design"]:
        assert d["source"].strip()
    for qid, q in c["questions"].items():
        assert q["source"].strip() and q["instructions"].strip(), qid
        for opt, v in q["options"].items():
            assert v.get("source", "").strip(), (qid, opt)
            assert v.get("what", "").strip() and v.get("examples"), (qid, opt)
    assert c["error_filter"]["source"] and c["combination"]["source"] and c["state"]["source"]
    assert "confidence_filter" not in c and "thresholds" not in json.dumps(c["error_filter"])
    assert "variant" not in c and tuple(c["variants"]) == D.VARIANTS


def test_question_options_as_specified():
    assert D.options("t_category") == ("Tis", "T1", "T2", "T3", "T4a", "T4b", "cannot be assessed")
    assert D.options("regional_nodes") == ("none", "1", "2 to 3", "4 to 6", "7 or more", "not stated")
    assert D.options("tumor_deposits") == ("present", "absent", "not stated")
    assert D.options("distant_metastasis") == ("none", "M1a", "M1b", "M1c", "not stated")


NOTE_TERMS = re.compile(r"isolated tumor cells|adhe|non-regional", re.I)   # perforation is in the T4a definition


def test_the_confusions_are_in_the_notes_only():
    """The AJCC notes on isolated tumor cells, adhesions and non-regional nodes reach Jev in documented_notes only;
    documented sends the category definitions alone (T4a's definition includes perforation, so both versions have it)."""
    doc, nts = D.questions("documented"), D.questions("documented_notes")
    assert not NOTE_TERMS.search(json.dumps(doc))
    c = nts["t_category"]["criteria"]
    assert "perforation" in c["T4a"]["what"] and "perforation" in c["T3"]["not_for"]
    cd = doc["t_category"]["criteria"]
    assert "perforation" in cd["T4a"]["what"] and "perforation" in cd["T3"]["not_for"]
    assert "adhesion" in c["T4b"]["what"] and "adhesion" in c["T4b"]["not_for"]
    n = nts["regional_nodes"]["criteria"]
    assert "isolated tumor cells" in n["none"]["what"]
    assert all("isolated tumor cells" in n[o]["what"] and "non-regional" in n[o]["what"]
               for o in ("1", "2 to 3", "4 to 6", "7 or more"))
    m = nts["distant_metastasis"]["criteria"]
    assert "non-regional" in m["M1a"]["what"] and "non-regional" in m["none"]["not_for"]


def test_criteria_sent_are_documented_fields_only():
    for v in D.VARIANTS:
        for q in D.questions(v).values():
            for c in q["criteria"].values():
                assert set(c) <= {"what", "not_for", "examples"}
                assert not any(k.startswith("note") for k in c)


def test_notes_are_appended_with_one_space_and_never_alone():
    for qid, q in D.CFG["questions"].items():
        for opt, v in q["options"].items():
            doc = D.criterion(v, "documented")
            nts = D.criterion(v, "documented_notes")
            assert doc == {k: v[k] for k in ("what", "not_for", "examples") if k in v}
            for field, note in D.NOTE_FIELDS.items():
                want = f"{v[field]} {v[note]}" if v.get(note) else v.get(field)
                assert nts.get(field) == want, (qid, opt, field)
            assert nts["examples"] == doc["examples"]
            if not any(v.get(n) for n in D.NOTE_FIELDS.values()):
                assert nts == doc, (qid, opt)
    assert len(D.notes()) == 9


# ------------------------------------------------------------------------------------------------ one template per version
def test_every_request_is_its_versions_template(items):
    """For each of the 1,100 reports and both versions, the built body with the report text replaced by
    {pathology_report} is that version's template, character for character; only the report varies."""
    arm = D.arm_spec()
    cl = D.DocuseJevClient(RawStore(ROOT / "runs" / "_unused"), arm)
    tmpl = {v: D.dumps(D.template(v)) for v in D.VARIANTS}
    n = {v: 0 for v in D.VARIANTS}
    for it in items:
        req = cl.request(it)
        assert D.as_template_text(req, it.state) == tmpl[it.variant], (it.item_id, it.variant)
        assert D.body(req) == D.fill(D.template(it.variant), it.state)
        n[it.variant] += 1
    assert n == {"documented": 1100, "documented_notes": 1100}
    assert tmpl["documented"].count(D.PLACEHOLDER) == 1 and tmpl["documented_notes"].count(D.PLACEHOLDER) == 1


def test_the_versions_differ_only_in_the_notes(items):
    """Removing the appended notes (one space and the note) from a documented_notes body gives the documented body
    exactly, for the templates and for every report."""
    assert D.strip_notes(D.dumps(D.template("documented_notes"))) == D.dumps(D.template("documented"))
    assert D.dumps(D.template("documented_notes")) != D.dumps(D.template("documented"))
    arm = D.arm_spec()
    cl = D.DocuseJevClient(RawStore(ROOT / "runs" / "_unused"), arm)
    by = {(i.item_id, i.variant): i for i in items}
    for rid in {i.item_id for i in items}:
        a = D.dumps(D.body(cl.request(by[(rid, "documented")])))
        b = D.dumps(D.body(cl.request(by[(rid, "documented_notes")])))
        assert D.strip_notes(b) == a, rid
    # every note is in the notes body once per option that carries it, and nowhere in the documented body
    t_doc, t_notes = D.dumps(D.template("documented")), D.dumps(D.template("documented_notes"))
    for txt in {n["text"] for n in D.notes()}:
        assert txt not in t_doc
        assert t_notes.count(" " + txt) == sum(n["text"] == txt for n in D.notes())


# ------------------------------------------------------------------------------------------------ examples
FORBIDDEN = ["isolated", "itc", "adhes", "adher", "perforat", "ruptur", "non-regional", "nonregional", "para-aortic",
             "aortocaval", "retroperitoneal", "iliac node"]


def test_no_example_describes_a_confusing_feature():
    """No sent example, in either version, names a misleading feature; 'deposit' only in the tumor_deposits
    question."""
    for v in D.VARIANTS:
        for qid, q in D.questions(v).items():
            for opt, c in q["criteria"].items():
                for ex in c["examples"]:
                    low = ex.lower()
                    hits = [w for w in FORBIDDEN if w in low] + (["deposit"] if qid != "tumor_deposits" and "deposit" in low else [])
                    assert not hits, (v, qid, opt, ex, hits)
    assert {e for _, _, e in D.sent_examples()} == {e for v in D.VARIANTS for q in D.questions(v).values()
                                                    for c in q["criteria"].values() for e in c["examples"]}


def test_example_overlap_check_finds_what_it_should():
    fake = ["Tumor extent: Invades the submucosa. Nodes 0/12.", "one two three four five six seven eight nine"]
    cfg = {"questions": {"q": {"options": {"a": {"examples": ["TUMOR EXTENT: invades the submucosa"]},
                                           "b": {"examples": ["zero one two three four five six seven eight ten"]},
                                           "c": {"examples": ["one two three four five six eight"]},
                                           "d": {"examples": ["Four five six"]}}}}}
    got = {h["option"]: h for h in D.example_overlaps(fake, 8, cfg)}
    assert set(got) == {"a", "b", "d"} and got["a"]["substring"] == 1 and got["b"]["run_of_8"] == 1
    assert got["d"]["words"] == 1 and got["b"]["substring"] == 0


# ------------------------------------------------------------------------------------------------ exclusive and exhaustive
def test_node_bands_partition_the_counts():
    """The node bands cover 0 to infinity with no gap and no overlap."""
    bands = sorted(b for o in D.options("regional_nodes") if (b := D.node_band(o)) is not None)
    assert bands[0][0] == 0 and bands[-1][1] is None
    for (lo, hi), (lo2, _) in zip(bands, bands[1:]):
        assert hi is not None and lo <= hi and lo2 == hi + 1
    assert sum(1 for o in D.options("regional_nodes") if D.node_band(o) is None) == 1
    for count in range(0, 200):
        assert sum(1 for o in D.options("regional_nodes") if (b := D.node_band(o)) is not None
                   and b[0] <= count and (b[1] is None or count <= b[1])) == 1


def test_each_option_has_a_distinct_code():
    """Distinct option labels (also case- and space-insensitive, so match_label cannot mix them) in every question;
    distinct T and M codes; distinct node bands. The deposits question has three options on two staging codes by the
    config's design: absent and not stated both stage as no deposits (the config's source line); the check pins that."""
    for qid in D.QUESTION_IDS:
        labs = D.options(qid)
        assert len({" ".join(l.lower().split()) for l in labs}) == len(labs), qid
    for qid in ("t_category", "distant_metastasis"):
        codes = [v["code"] for v in D.CFG["questions"][qid]["options"].values()]
        assert len(set(codes)) == len(codes), qid
    bands = [tuple(v["band"]) if v["band"] else None for v in D.CFG["questions"]["regional_nodes"]["options"].values()]
    assert len(set(bands)) == len(bands)
    dep = {o: v["code"] for o, v in D.CFG["questions"]["tumor_deposits"]["options"].items()}
    assert dep == {"present": 1, "absent": 0, "not stated": 0}
    assert D.t_code("cannot be assessed") == "TX" and D.t_code("Tis") == "Tis"


def test_every_report_truth_maps_to_one_option_per_question(frames):
    """For each of the 1,100 reports, its true T, node count, deposits and M satisfy exactly one option per question
    (read from the config's codes and bands), those are the options parts_of_truth gives, and the four reproduce the
    key: N as stored, the stage group and the level."""
    n = 0
    for f in frames.values():
        for r in f.to_dict("records"):
            args = (r["t"], r["nodes_involved"], r["tumour_deposits"], r["m"])
            got = {q: D.matching_options(q, *args) for q in D.QUESTION_IDS}
            assert all(len(v) == 1 for v in got.values()), (r["report_id"], got)
            one = {q: v[0] for q, v in got.items()}
            assert one == D.parts_of_truth(*args), r["report_id"]
            assert D.n_from_parts(one["regional_nodes"], one["tumor_deposits"]) == r["n"], r["report_id"]
            assert D.stage_from_parts(*(one[q] for q in D.QUESTION_IDS)) == r["substage"], r["report_id"]
            assert D.level_from_parts(*(one[q] for q in D.QUESTION_IDS)) == r["level"], r["report_id"]
            n += 1
    assert n == 1100


def test_unassessable_parts_give_no_stage():
    base = D.parts_of_truth("T3", 2, 0, "M0")
    for q, opt in (("t_category", "cannot be assessed"), ("regional_nodes", "not stated")):
        p = {**base, q: opt}
        assert D.stage_from_parts(*(p[k] for k in D.QUESTION_IDS)) is None
        ans = {k: {"choice": p[k], "probabilities": {p[k]: 1.0}, "confidence": 1.0} for k in D.QUESTION_IDS}
        r = D.parse_answers(ans)
        assert r["answer"] is None and r["parse"] == "unstaged" and r["probs"] is None
    p = {**base, "t_category": "Tis", "regional_nodes": "none", "tumor_deposits": "absent"}
    assert D.level_from_parts(*(p[k] for k in D.QUESTION_IDS)) == "0"
    p = {**base, "t_category": "Tis", "regional_nodes": "none", "tumor_deposits": "present"}   # Tis N1c: no group
    assert D.stage_from_parts(*(p[k] for k in D.QUESTION_IDS)) is None
    assert "other" not in D.options("t_category")


# ------------------------------------------------------------------------------------------------ parts -> stage
def test_every_ajcc_cell_maps_to_the_key_stage():
    for (t, n, m), g in crc.STAGE_TABLE.items():
        for dep in (False, True):
            p = D.parts_of_cell(t, n, m, deposits_with_nodes=dep)
            assert D.stage_from_parts(*(p[q] for q in D.QUESTION_IDS)) == g, (t, n, m, dep, p)


def test_every_answer_combination_follows_the_table():
    """Each of the 432 combinations: None exactly for NX and for cells the table does not stage; otherwise the table's
    group for (T code, crc.n_category(band low, deposits), M code)."""
    for t, nd, dp, m in itertools.product(*[D.options(q) for q in D.QUESTION_IDS]):
        got = D.stage_from_parts(t, nd, dp, m)
        b = D.node_band(nd)
        if b is None:
            assert got is None
            continue
        cell = (D.t_code(t), crc.n_category(b[0], D.deposits_code(dp)), D.m_code(m))
        assert got == crc.STAGE_TABLE.get(cell), (t, nd, dp, m)


def test_node_bands_give_the_key_n_for_every_count():
    for count in range(0, 40):
        for dep in (0, 1, 3):
            opt = D.band_of_count(count)
            assert D.n_from_parts(opt, "present" if dep else "absent") == crc.n_category(count, dep)


def test_true_parts_reproduce_every_key(frames):
    n = 0
    for which, f in frames.items():
        for r in f.to_dict("records"):
            p = D.parts_of_truth(r["t"], r["nodes_involved"], r["tumour_deposits"], r["m"])
            args = [p[q] for q in D.QUESTION_IDS]
            assert D.stage_from_parts(*args) == r["substage"], (r["report_id"], p)
            assert D.level_from_parts(*args) == r["level"], (r["report_id"], p)
            n += 1
    assert n == 1100


# ------------------------------------------------------------------------------------------------ distributions
def _random_parts(rng):
    out = {}
    for q in D.QUESTION_IDS:
        w = rng.dirichlet(np.ones(len(D.options(q))))
        out[q] = dict(zip(D.options(q), w))
    return out


def test_distribution_sums_to_one():
    rng = np.random.default_rng(1)
    for _ in range(200):
        lv, none = D.level_distribution(_random_parts(rng))
        assert abs(sum(lv.values()) + none - 1) < 1e-12
        assert all(v >= 0 for v in lv.values())


def test_distribution_matches_brute_force():
    rng = np.random.default_rng(2)
    parts = _random_parts(rng)
    want = {l: 0.0 for l in D.LEVELS}
    for combo in itertools.product(*[D.options(q) for q in D.QUESTION_IDS]):
        pr = np.prod([parts[q][o] for q, o in zip(D.QUESTION_IDS, combo)])
        lvl = D.level_from_parts(*combo)
        if lvl:
            want[lvl] += pr
    got, _ = D.level_distribution(parts)
    assert all(abs(got[l] - want[l]) < 1e-12 for l in D.LEVELS)


def test_certain_parts_give_a_point_mass_on_the_key():
    p = D.parts_of_truth("T3", 2, 1, "M0")
    lv, none = D.level_distribution({q: {p[q]: 1.0} for q in D.QUESTION_IDS})
    assert lv["IIIB"] == 1.0 and none == 0.0


def test_parse_answers_rules():
    p = D.parts_of_truth("T4a", 0, 2, "M0")                       # IIIB via N1c
    ans = {q: {"choice": p[q], "probabilities": {p[q]: 0.9, **{o: 0.1 / (len(D.options(q)) - 1)
                                                              for o in D.options(q) if o != p[q]}},
               "confidence": 0.8} for q in D.QUESTION_IDS}
    r = D.parse_answers(ans)
    assert r["answer"] == "IIIB" and r["parse"] == "parts"
    assert abs(r["top_prob"] - 0.9 ** 4) < 1e-9
    assert abs(r["jev_confidence"] - 0.8 ** 4) < 1e-9
    assert abs(sum(r["probs"].values()) - 1) < 1e-9
    bad = json.loads(json.dumps(ans))
    bad["regional_nodes"]["choice"] = "not stated"
    assert D.parse_answers(bad)["answer"] is None and D.parse_answers(bad)["parse"] == "unstaged"
    bad = json.loads(json.dumps(ans))
    bad["t_category"]["probabilities"] = {"T1": 0.3}
    r2 = D.parse_answers(bad)
    assert r2["probs"] is None and r2["top_prob"] is None and r2["answer"] == "IIIB"
    part = json.loads(json.dumps(ans))                              # mass on NX: the list is conditional on a stage
    part["regional_nodes"]["probabilities"] = {"none": 0.5, "not stated": 0.5}
    part["t_category"]["probabilities"] = {"T4a": 1.0}                # (Tis with deposits is unstaged too)
    part["distant_metastasis"]["probabilities"] = {"none": 0.9, "M1a": 0.1}   # (M not stated is unstaged too)
    r3 = D.parse_answers(part)
    assert abs(r3["p_unstaged"] - 0.5) < 1e-9 and abs(sum(r3["probs"].values()) - 1) < 1e-9
    assert r3["prob_status"] == "renormalised" and abs(r3["top_prob"] - 0.9 ** 2 * 0.5) < 1e-9
    assert D.parse_answers(None)["parse"] == "missing_answer"


# ------------------------------------------------------------------------------------------------ requests and dry run
def test_request_uses_the_existing_jev_settings(items, tmp_path):
    arm = D.arm_spec()
    req = D.DocuseJevClient(RawStore(tmp_path), arm).request(items[0])
    base = K.CategoricalJevClient(RawStore(tmp_path), SI.arm("arm3")).request(
        next(i for i in K.load_items(SI.arm("arm3")) if i.item_id == items[0].item_id))
    for k in ("model", "provider", "_repeat"):
        assert req[k] == base[k]
    assert req["provider"]["allow_fallbacks"] is False and req["model"] == MODELS["jev"]["slug"]
    assert req["state"] == {"pathology_report": items[0].state}
    assert list(req["questions"]) == list(D.QUESTION_IDS)
    other = next(i for i in items if i.item_id == items[0].item_id and i.variant != items[0].variant)
    req2 = D.DocuseJevClient(RawStore(tmp_path), arm).request(other)
    assert {k: req2[k] for k in ("model", "provider", "_repeat")} == {k: req[k] for k in ("model", "provider", "_repeat")}
    assert K._sha(req) != K._sha(req2), "the two versions are stored under different keys"


def test_dry_run_builds_every_valid_request(tmp_path):
    import run_staging as R
    out = R.dry_run(tmp_path)
    assert out["requests"] == {"full": 2200, "repeats": 160, "timing": 100} and out["invalid"] == 0
    assert out["by_set_and_variant"]["full/documented"]["requests"] == 1100
    assert out["by_set_and_variant"]["full/documented_notes"]["requests"] == 1100
    assert out["by_set_and_variant"]["repeats/documented_notes"]["requests"] == 80
    assert set(k for k in out["by_set_and_variant"] if k.startswith("timing")) == {"timing/documented"}
    assert not list((tmp_path / "runs").rglob("*.json")), "a dry run stores nothing"
    assert out["usd_list_total"] < 1.0
    one = R.dry_run(tmp_path, ["documented_notes"])
    assert one["requests"] == {"full": 1100, "repeats": 80, "timing": 100} and one["invalid"] == 0


def test_plan_both_variants(items):
    full = D.plan("full", items)
    assert len(full) == 2200 and {c.variant for c in full} == set(D.VARIANTS)
    rep = D.plan("repeats", items)
    assert len(rep) == 160 and {c.repeat for c in rep} == {1, 2} and len({c.item_id for c in rep}) == 40
    # both versions of a report fall in the same block of 50 reports
    pos = {}
    ids = D.seeded_ids(items)
    for k, c in enumerate(full):
        pos.setdefault(c.item_id, []).append(k)
    assert all(max(v) // 100 == min(v) // 100 for v in pos.values())
    assert [c.item_id for c in full][::1][:1] and ids[0] in {c.item_id for c in full[:100]}


def test_validate_request_catches_a_changed_template(items, tmp_path):
    arm = D.arm_spec()
    it = items[0]
    req = D.DocuseJevClient(RawStore(tmp_path), arm).request(it)
    assert D.validate_request(req, it) == []
    bad = json.loads(json.dumps(req))
    bad["questions"]["t_category"]["criteria"]["T1"]["what"] += " x"
    assert D.validate_request(bad, it)
    wrong = next(i for i in items if i.item_id == it.item_id and i.variant != it.variant)
    assert D.validate_request(req, wrong)


# ------------------------------------------------------------------------------------------------ calls table and analyze
def test_calls_table_round_trips_through_analyze(items, arm3_answers, tmp_path):
    arm = D.arm_spec()
    store = RawStore(tmp_path / "store")
    truth = D.truth_parts_of_frames()
    base = _one(items, "documented")
    pick = {i.item_id for i in [i for i in base if not i.stratum.isdigit()][:120] + [i for i in base if i.stratum.isdigit()][:20]}
    main = [i for i in items if i.item_id in pick]
    calls = D.plan("full", main)
    K.execute(calls, main, D.make_factory(arm, store, D.synthetic_sender(truth, 3)), workers=2, arm_name=arm.name)
    df, parts = D.calls_table(store, arm, main, calls)
    ref = arm3_answers
    assert list(df.columns) == D.CALL_COLUMNS and set(D.CALL_COLUMNS) - {"response_id"} <= set(ref.columns)
    assert len(df) == len(calls) == 280 and (df.system == "jev").all()
    assert df.variant.value_counts().to_dict() == {"documented": 140, "documented_notes": 140}
    assert parts.groupby("variant").size().to_dict() == {"documented": 140, "documented_notes": 140}
    assert len(parts) == len(df) and df.valid.mean() > 0.2
    # the stored rows parse back identically, and scoring by hand agrees with analyze()
    df2, _ = D.calls_table(store, arm, main, calls)
    pd.testing.assert_frame_equal(df, df2)
    only_main = [i for i in main if not i.stratum.isdigit() and i.variant == "documented_notes"]
    ids = {i.item_id for i in only_main}
    chat = ref[(ref.variant == "names") & (ref.system != "jev") & ref.item_id.isin(ids)]
    jev = df[df.item_id.isin(ids) & (df.variant == "documented_notes")].assign(variant="names")
    assert len(jev) == 120
    frame = pd.DataFrame([{"item_id": i.item_id, "truth": i.truth, "stratum": i.stratum, "group": i.group}
                          for i in only_main]).sort_values("item_id").reset_index(drop=True)
    cfg = {**arm.config, "analysis": {**arm.config["analysis"], "n_boot": 200}}
    systems = ["jev", "gpt", "claude", "gemini", "muse", "glm"]
    res = KA.analyze(pd.concat([chat, jev]), frame, cfg, D.LEVELS, systems,
                     {"jev": "jev", **{s: "primary" for s in systems[1:]}}, MODELS, None, None, "names")
    truth_of = dict(zip(frame.item_id, frame.truth))
    by_hand = float((jev.valid & (jev.answer == jev.item_id.map(truth_of))).mean())
    assert abs(res["systems"]["jev"]["exact_accuracy"] - by_hand) < 1e-12
    assert res["completeness"]["jev"]["missing"] == 0
    j = jev.set_index("item_id")
    top = j.top_prob.where(j.valid.astype(bool), -1.0)       # an unusable answer ranks last (hybrid_split)
    use = KA.hybrid_split(KA.Table(pd.concat([chat, jev]), frame, systems, D.LEVELS, MODELS, "names"), 0.5)
    chosen = frame.item_id[use]
    assert top.loc[chosen].min() >= top.drop(chosen).fillna(-1).max() - 1e-12   # Jev keeps its most confident half


def test_analysis_helpers_on_synthetic(items, arm3_answers, tmp_path):
    import analyze_staging as A
    arm = D.arm_spec()
    store = RawStore(tmp_path / "store")
    truth = D.truth_parts_of_frames()
    calls = D.plan("full", items)
    K.execute(calls, items, D.make_factory(arm, store, D.synthetic_sender(truth, 5, 0.9)), workers=4,
              arm_name=arm.name)
    df, parts = D.calls_table(store, arm, items, calls)
    ref = arm3_answers
    for jv in D.VARIANTS:
        for cv in ("names", "definitions"):
            comb = A.combined(ref, df, jv, cv)
            assert (comb[comb.system == "jev"].variant == cv).all() and (comb.system == "jev").sum() == 1100
            assert (comb[comb.system != "jev"].variant == cv).all()
    comb = A.combined(ref, df, "documented", "names")
    frame = A.main_items_frame("names")
    ex = A.staging_extra(comb, frame, ["jev", "gpt"], n_boot=100)
    assert set(ex["jev"]) >= {"main_stage_accuracy", "stage_iii_to_0_ii", "ivc_to_iva_ivb", "confusion",
                              "stage_iii_iv", "stage_iv"}
    assert sum(map(sum, ex["jev"]["confusion"]["counts"])) == ex["jev"]["confusion"]["n_usable"]
    ef = A.error_filter(comb, frame)
    assert ef["n_kept"] == 500 and ef["share"] == 0.5
    t = A.table(comb, frame, ["jev"], "names")
    keep = KA.hybrid_split(t, 0.5)
    conf = np.where(t.usable[:, 0] & ~np.isnan(t.top[:, 0]), t.top[:, 0], -np.inf)
    assert conf[keep].min() >= conf[~keep].max()           # the most confident half by the routing confidence
    assert abs(ef["accuracy_kept"] - t.correct[keep, 0].mean()) < 1e-12
    pa = A.parts_accuracy(parts[(parts.repeat == 0) & (parts.variant == "documented")], truth)
    assert set(pa) == set(D.QUESTION_IDS) and pa["t_category"]["n"] == 1100
    mis = A.misread_table(df[df.variant == "documented"], None, A.confuser_items(), A.main_items_frame("names"), comb,
                          ["jev"])
    assert set(mis) == {"note", "types_1_4", "type_5", "type_6"}
    assert sum(v["systems"]["jev"]["n"] for v in mis["types_1_4"].values()) == 80
    assert sum(v["jev"]["n"] for v in mis["type_5"].values()) == 20

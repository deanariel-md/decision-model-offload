import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevity import analysis as A
from jevity.clients import parse_probability


def test_wlogit_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    y = (rng.random(20000) < A.expit(-1 + 0.7 * x)).astype(float)
    a, b = A.wlogit_fit(x, y)
    assert abs(a + 1) < 0.06 and abs(b - 0.7) < 0.06


def test_draws_shared_and_weighted():
    d = A.Draws(range(100), B=50, R=7)
    assert d.C.shape == (50, 100) and np.allclose(d.C.sum(1), 100) and d.r.max() == 6
    assert d.w([5, 5, 7]).shape == (50, 3)


def test_parse_probability():
    ok = {'{"p": 0.123}': 0.123, '```json\n{"p": 0.2}\n```': 0.2, "0.35": 0.35, "p: 0.4": 0.4, '{"p": 1}': 1.0,
          "12%": 0.12, "p = .05": 0.05, "0": 0.0, "1e-3": 0.001}
    for txt, want in ok.items():
        assert parse_probability(txt) == want, txt
    bad = ['{"p": 1.7}', "p: 1.7", "p: 12%", "p: 1e-3", "true", '{"p": true}', '{"p": "0.2"}', "NaN",
           '{"p": NaN}', "no idea", "About 12% chance",
           "I cannot give a probability from 0 to 1 for this person.", "Between 0.1 and 0.2", "", None, "[0.2]"]
    for txt in bad:
        assert parse_probability(txt) is None, txt


def test_empty_family_and_missing_refit_keys():
    import pandas as pd, pytest
    tab = pd.DataFrame({"profile": range(30), "edit": "C4_sbp_plus20", "klass": "clinical", "feature": "sbp_mmhg",
                        "p_base": 0.2, "p": 0.25, "z_logit": 0.29, "z_prob": 0.05, "d_logit": 0.2, "d_prob": 0.03, "q0": 0.2})
    calls = pd.DataFrame({"model": "jev", "profile": range(30), "edit": "baseline", "repeat": 0, "annotated": False,
                          "valid": True, "variant": "raw", "p": 0.2})
    cohort = pd.DataFrame({"SEQN": range(30), "death_10y": [0, 1] * 15})
    ref = A.Refits(np.full((30, 4), 0.2), np.full((30, 4), 0.03), {(i, "C4_sbp_plus20"): i for i in range(30)})
    d = A.Draws(range(30), B=50, R=4)
    assert A.confirmatory_family({"jev": tab}, ref, d, calls, cohort, edits=[], pairs=False)["n_cells"] == 0
    assert A.confirmatory_family({"jev": tab}, ref, d, calls, cohort, edits=["C4_sbp_plus20"], pairs=False)["n_cells"] == 1
    short = A.Refits(ref.d_logit[:29], ref.d_prob[:29], {(i, "C4_sbp_plus20"): i for i in range(29)})
    with pytest.raises(ValueError):
        A.confirmatory_family({"jev": tab}, short, d, calls, cohort, edits=["C4_sbp_plus20"], pairs=False)


def test_block_order_keeps_records_together():
    from jevity.runner import Call, block_order
    calls = [Call(m, p, e, False, 0, "") for p in range(200) for m in ("jev", "gpt") for e in ("baseline", "C4")]
    out = block_order(calls, block=50)
    first = {c.profile for c in out[:200]}                    # the first block: 50 records, every call for each
    assert len(first) == 50 and sorted(c.profile for c in out[:200]) == sorted([p for p in first for _ in range(4)])


def test_jev_bundle_and_band_parsing(tmp_path):
    from jevity import clients as C
    store = C.RawStore(tmp_path)
    j = C.JevClient(store, question="bundle")
    req = j.request('{"age": 60}', "This person will die from any cause within ten years of this examination.")
    qid = C.PROMPTS["jev"]["risk_bands"]["question_id"]
    assert set(req["questions"]) == {"death", "alive", qid}
    assert req["questions"][qid]["criteria"]["alive_at_10_years"] == "Alive ten years after this examination."
    bands = {"died_within_2_years": 0.05, "died_2_to_5_years": 0.05, "died_5_to_10_years": 0.1, "alive_at_10_years": 0.8}
    store.put(req, {"answers": {"death": {"noul": 0.14}, "alive": {"noul": 0.83},
                                qid: {"probabilities": bands, "confidence": 0.6}}}, {})
    r = j.probability('{"age": 60}', "This person will die from any cause within ten years of this examination.")
    assert r["valid"] and r["p"] == 0.14 and abs(r["p_bands"] - 0.2) < 1e-9 and r["p_alive"] == 0.83
    assert C._risk_band_mean({"probabilities": {**bands, "alive_at_10_years": 0.5}})[1] == "bands_mass"
    m1 = C.JevClient(store, variant="M1").request('{"age": 60}', "S.")
    raw = C.JevClient(store).request('{"age": 60}', "S.")
    assert m1["questions"] == raw["questions"] and m1["state"].startswith("Instruction: ") and m1["state"].endswith('{"age": 60}')
    assert C._risk_band_mean({"probabilities": {"new_label": 1.0}})[1] == "bands_labels"
    assert C._real01(True) is None


def test_cost_latency_counts_failures():
    import pandas as pd
    from jevity.clients import MODELS
    calls = pd.DataFrame({"model": ["gpt"] * 4, "valid": [True, True, False, True], "route": ["batch"] * 3 + ["standard"],
                          "tokens_in": [500.0] * 4, "tokens_out": [400.0] * 4, "latency_s": [None, None, None, 2.0]})
    out = A.cost_latency(calls, MODELS)["gpt"]
    b, f = MODELS["batch"]["families"]["gpt"], MODELS["families"]["gpt"]
    want = (3 * (500 * b["price_in"] + 400 * b["price_out"]) + 500 * f["price_in"] + 400 * f["price_out"]) / 1e6
    assert abs(out["usd_total"] - want) < 1e-12 and out["usable"] == 3 and out["median_latency_s"] == 2.0


def test_jev_bundle_summary():
    import pandas as pd
    rows = []
    for p_ in range(40):
        rows += [{"model": "jev", "profile": p_, "edit": "baseline", "repeat": 0, "annotated": False, "valid": True,
                  "variant": "raw", "p": 0.10, "p_alive": None},
                 {"model": "jev", "profile": p_, "edit": "baseline", "repeat": 1, "annotated": False, "valid": True,
                  "variant": "raw", "p": 0.10 if p_ % 2 else 0.11, "p_alive": None},
                 {"model": "jev_bundle", "profile": p_, "edit": "baseline", "repeat": 0, "annotated": False,
                  "valid": True, "variant": "raw", "p": 0.10 if p_ % 4 else 0.12, "p_alive": 0.88}]
    out = A.jev_bundle(pd.DataFrame(rows))
    assert out["n"] == 40 and abs(out["share_identical"] - 0.75) < 1e-9 and abs(out["repeat_share_identical"] - 0.5) < 1e-9
    assert out["bundle_share_within_0.05_of_1"] == 1.0
    c = A.jev_complement(pd.DataFrame(rows + [{"model": "jev_complement", "profile": 0, "edit": "baseline", "repeat": 0,
                                               "annotated": False, "valid": True, "variant": "raw", "p": 0.85,
                                               "p_alive": None}]))
    assert c["n"] == 1 and abs(c["sum_mean"] - 0.95) < 1e-9

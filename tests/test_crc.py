"""Arm 3: every cell of the AJCC 8th stage table, the 10 scored levels, the report parser (src/jevity/crc_parse.py) and
the arm spec read by the shared categorical code (config/arm3.yaml). No network."""
import itertools
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import crc
from jevity.crc_parse import parse_report

T = ["Tis", "T1", "T2", "T3", "T4a", "T4b"]
N = ["N0", "N1a", "N1b", "N1c", "N2a", "N2b"]
M = ["M0", "M1a", "M1b", "M1c"]
CHATBOT_SYSTEM = ("Choose the stage group that is correct, and give the probability that each stage group is correct, as "
                  "numbers between 0 and 1 that add up to 1. Always choose one stage group. Reply with a JSON object with "
                  "two keys, \"stage\" (the stage group you choose, written exactly as in the list) and \"probabilities\" "
                  "(an object with one number for each stage group), and nothing else.")
# AJCC 8th, M0, transcribed from the manual's table (independent of config/crc_ajcc8.yaml)
GRID_M0 = {
    "T1":  ["I",   "IIIA", "IIIA", "IIIA", "IIIA", "IIIB"],
    "T2":  ["I",   "IIIA", "IIIA", "IIIA", "IIIB", "IIIB"],
    "T3":  ["IIA", "IIIB", "IIIB", "IIIB", "IIIB", "IIIC"],
    "T4a": ["IIB", "IIIB", "IIIB", "IIIB", "IIIC", "IIIC"],
    "T4b": ["IIC", "IIIC", "IIIC", "IIIC", "IIIC", "IIIC"],
    "Tis": ["0",   None,   None,   None,   None,   None],
}


def expected_stage(t, n, m):
    if m != "M0":
        return {"M1a": "IVA", "M1b": "IVB", "M1c": "IVC"}[m]          # Any T, any N
    return GRID_M0[t][N.index(n)]


@pytest.mark.parametrize("t,n,m", list(itertools.product(T, N, M)))
def test_every_cell_of_the_stage_table(t, n, m):
    exp = expected_stage(t, n, m)
    if exp is None:
        with pytest.raises(ValueError):
            crc.stage_group(t, n, m)
    else:
        assert crc.stage_group(t, n, m) == exp


def test_table_shape_and_categories():
    assert (crc.T_CATS, crc.N_CATS, crc.M_CATS) == (T, N, M)
    assert crc.STAGE_GROUPS == ["0", "I", "IIA", "IIB", "IIC", "IIIA", "IIIB", "IIIC", "IVA", "IVB", "IVC"]
    assert len(crc.STAGE_TABLE) == 144 - 5                                  # only Tis with a positive node is unstaged
    assert set(itertools.product(T, N, M)) - set(crc.STAGE_TABLE) == {("Tis", n, "M0") for n in N[1:]}


@pytest.mark.parametrize("inv,td,n", [(0, 0, "N0"), (0, 1, "N1c"), (0, 4, "N1c"), (1, 0, "N1a"), (1, 3, "N1a"),
                                      (2, 0, "N1b"), (3, 2, "N1b"), (4, 0, "N2a"), (6, 1, "N2a"), (7, 0, "N2b"),
                                      (14, 4, "N2b")])
def test_n_category_deposits_count_only_when_nodes_negative(inv, td, n):
    assert crc.n_category(inv, td) == n


def _expand_definition(text):
    """'T3-T4a N1a-N1c M0; T2-T3 N2a M0' -> cells, reading the AJCC notation independently of crc."""
    def rng(tok, cats, anyword):
        if tok == anyword:
            return cats
        a, _, b = tok.partition("-")
        return cats[cats.index(a):cats.index(b or a) + 1]
    cells = set()
    for part in text.split("; "):
        part = part.replace("Any T", "AnyT").replace("Any N", "AnyN")
        t, n, m = part.split()
        cells |= set(itertools.product(rng(t, T, "AnyT"), rng(n, N, "AnyN"), rng(m, M, "AnyM")))
    return cells


def test_definitions_match_the_table():
    for g in crc.STAGE_GROUPS:
        assert _expand_definition(crc.stage_definition(g)) == {c for c, s in crc.STAGE_TABLE.items() if s == g}
    assert crc.stage_definition("IIIB") == "T3-T4a N1a-N1c M0; T2-T3 N2a M0; T1-T2 N2b M0"
    assert crc.stage_definition("IVC") == "Any T Any N M1c"


def test_ten_levels():
    assert crc.LEVELS == ["0", "I", "IIA", "IIB", "IIC", "IIIA", "IIIB", "IIIC", "IVA-IVB", "IVC"]
    assert [crc.level_of(g) for g in crc.STAGE_GROUPS] == crc.LEVELS[:8] + ["IVA-IVB", "IVA-IVB", "IVC"]
    for lv in crc.LEVELS[:8]:
        assert crc.level_definition(lv) == crc.stage_definition(lv)
    assert crc.level_definition("IVA-IVB") == "any T, any N, M1a-M1b"
    assert crc.level_definition("IVC") == "any T, any N, M1c"
    for lv, ms in (("IVA-IVB", ["M1a", "M1b"]), ("IVC", ["M1c"])):                    # each is exactly its definition
        assert {c for c, s in crc.STAGE_TABLE.items() if crc.level_of(s) == lv} == set(itertools.product(T, N, ms))
    with pytest.raises(ValueError):
        crc.level_of("IVD")
    for bad in ("IVA", "IVB", "IV"):
        with pytest.raises(ValueError):
            crc.level_definition(bad)
    assert crc.MAIN_STAGES == ["0", "I", "II", "III", "IV"]
    assert [crc.main_stage(lv) for lv in crc.LEVELS] == ["0", "I", "II", "II", "II", "III", "III", "III", "IV", "IV"]
    for bad in ("IVB", "IV"):
        with pytest.raises(ValueError):
            crc.main_stage(bad)


def test_arm_spec_for_the_shared_code():
    """config/arm3.yaml, read by jevity.categorical: the 10 levels in order with the table's definitions, the
    instruction and reply rule, the two versions and the item file's columns."""
    cfg = yaml.safe_load((ROOT / "config" / "arm3.yaml").read_text(encoding="utf-8"))
    assert cfg["kind"] == "score" and cfg["answer_key"] == "stage" and cfg["seed"] == 20260922
    assert cfg["instructions"] == ("Assign the pathological stage group for this colorectal cancer according to the "
                                   "AJCC Cancer Staging Manual, 8th edition. Stage IVA-IVB here includes IVA and IVB.")
    assert cfg["answer_aliases"] == {"IVA": "IVA-IVB", "IVB": "IVA-IVB"}                 # the reply rule
    system = f"{' '.join(cfg['prompts']['system'].split())} {' '.join(cfg['prompts']['reply'].split())}"
    assert system == CHATBOT_SYSTEM                                                     # the wording, exactly
    assert [lv["name"] for lv in cfg["levels"]] == crc.LEVELS
    assert all(lv["definition"] == crc.level_definition(lv["name"]) for lv in cfg["levels"])
    assert len(cfg["levels"]) == 10 and "jev_score_max_levels" not in cfg                # Score's maximum; no override
    assert cfg["variants"] == {"names": {"definitions": False},                          # both: all 1,000
                               "definitions": {"definitions": True}}
    assert cfg["primary_variant"] == "names"
    ic = cfg["items"]
    assert (ic["id_column"], ic["state_column"], ic["truth_column"], ic["stratum_column"], ic["group_column"],
            ic["repeat_column"]) == ("report_id", "report", "level", "level", "substage", "repeat")


def test_parser_imports_no_project_code():
    import ast
    tree = ast.parse((ROOT / "src" / "jevity" / "crc_parse.py").read_text(encoding="utf-8"))
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {f"{n.module}.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not any("crc" in m or "yaml" in m or "jevity" in m for m in mods), mods
    doc = ast.get_docstring(tree)
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value != doc]
    assert not any("crc_templates" in s or "crc_ajcc8" in s or "config" in s for s in strings)


def _report(depth, nodes, deposits, ct):
    return ("SURGICAL PATHOLOGY REPORT\nSpecimen: Sigmoid colon, sigmoid colectomy\n\nFINAL DIAGNOSIS\n"
            f"Adenocarcinoma of the sigmoid colon, 3.5 cm. {depth} Lymphovascular invasion is present. Margins are clear.\n\n"
            f"{nodes} {deposits}\n\nStaging CT chest, abdomen and pelvis: {ct}")


@pytest.mark.parametrize("depth,nodes,deposits,ct,want", [
    ("Tumor invades into the subserosa.", "Lymph nodes: 2/19 positive.", "No tumor deposits.",
     "No distant disease.", ("T3", "N1b", "M0")),
    ("Gross perforation through the tumor is present.", "Metastatic carcinoma in 0 of 21 lymph nodes.",
     "Tumor deposits: 3.", "Enlarged para-aortic lymph nodes consistent with metastasis; no other distant disease.",
     ("T4a", "N1c", "M1a")),
    ("The tumor invades the urinary bladder (confirmed histologically).", "Nine of 20 lymph nodes positive (9/20).",
     "Tumor deposits: none.", "Liver and adrenal metastases.", ("T4b", "N2b", "M1b")),
    ("Carcinoma in situ; no invasion of the submucosa.", "Lymph nodes negative (0/14).", "Tumor deposits: absent.",
     "Omental caking.", ("Tis", "N0", "M1c")),
    ("Invasion is into but not through the muscularis propria.", "Lymph nodes: 1 of 16 involved.",
     "One tumor deposit is also present (1).", "Unremarkable; no metastases.", ("T2", "N1a", "M0")),
])
def test_parser_reads_free_wording(depth, nodes, deposits, ct, want):
    p = parse_report(_report(depth, nodes, deposits, ct))
    assert (p["t"], p["n"], p["m"]) == want


def test_parser_returns_none_when_unreadable():
    p = parse_report("SURGICAL PATHOLOGY REPORT\nAdenocarcinoma of the colon.\nLymph nodes: see addendum.")
    assert p["t"] is None and p["n"] is None and p["m"] is None

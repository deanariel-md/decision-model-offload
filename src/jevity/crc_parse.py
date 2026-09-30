"""Arm 3 check: read a colorectal resection report and recover its T, N and M.

Written apart from the code that wrote the reports: this module never imports jevity.crc and reads no config.
It knows the anatomy of the bowel wall, how pathology reports state node and deposit counts, how a staging CT reports
distant disease, and the AJCC 8th rules that turn those into categories. It does not know the templates. The counts it
read are returned with the categories so a miss can be traced."""
from __future__ import annotations

import re

import pandas as pd

_NEG = re.compile(r"\b(?:no|not|none|without|free|negative|absent|clear|intact|uninvolved|unremarkable)\b", re.I)
_CT = re.compile(r"^\s*staging ct\b[^:\n]*:\s*(?P<body>.+)$", re.I | re.M)
# a clause ends at punctuation or where a contrast or a negation starts ("..., but not into", "... with no extension")
_CLAUSE = re.compile(r"[;:,()]|\s(?=(?:but|without|with no|and the|whereas)\b)", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Deepest layer first. A clause names the deepest layer the tumor reaches unless it is negated.
_DEPTH = [
    ("T4b", re.compile(r"\badjacent (?:organ|structure)s?\b|\ben bloc\b|\bsmall bowel\b|\babdominal wall\b|"
                       r"\b(?:urinary )?bladder\b|\buterus\b|\bvagina\b|\bprostate\b|\bstomach\b|\bspleen\b|"
                       r"\bduodenum\b|\bureter\b|\bkidney\b|\bpancreas\b", re.I)),
    ("T4a", re.compile(r"\bvisceral peritoneum\b|\bserosal surface\b|\bserosa\b|\bperitoneal surface\b", re.I)),
    ("T3", re.compile(r"\b(?:through|beyond) the (?:full thickness of the )?muscularis propria\b|"
                      r"\bperi(?:colic|colonic|colorectal|rectal)\b|\bsubserosa\w*|\bmesorect\w*", re.I)),
    ("T2", re.compile(r"\bmuscularis propria\b", re.I)),
    ("T1", re.compile(r"\bsubmucosa\b|\bthrough the muscularis mucosae?\b", re.I)),
    ("Tis", re.compile(r"\blamina propria\b|\bintramucosal\b|\bin situ\b|\bconfined to the mucosa\b", re.I)),
]
_T_ORDER = ["Tis", "T1", "T2", "T3", "T4a", "T4b"]
# the idiom "into but not through <layer>" places the tumor in that layer
_INTO_NOT_THROUGH = re.compile(r"\binto,?\s+but not through,?\s+the ", re.I)
# statements about other elements; the words "invasion" or "pericolic" in them say nothing about depth
_NOT_DEPTH = re.compile(r"lymph node|\bnodes?\b|deposit|margin|perforat|lymphovascular|perineural|mismatch|specimen",
                        re.I)

_SITES = {
    "peritoneum": re.compile(r"periton\w*|\bomentum\b|\bomental\b|carcinomatosis", re.I),
    "liver": re.compile(r"\bliver\b|\bhepatic\b", re.I),
    "lung": re.compile(r"\blungs?\b|\bpulmonary\b", re.I),
    "bone": re.compile(r"\bbones?\b|\bosseous\b|\bvertebra\w*|\blytic\b|\bskeleton\b", re.I),
    # not "retroperitoneal": the pericolic nodes of the ascending and descending colon lie there and are regional
    "distant nodes": re.compile(r"non-?regional (?:lymph )?nodes?|(?:para-?aortic|aortocaval|supraclavicular) "
                                r"(?:lymph )?nodes?", re.I),
    "other organ": re.compile(r"\bbrain\b|\badrenal\b|\bovar(?:y|ies)\b", re.I),
}


# AJCC 8th and CAP (colon and rectum 4.4.0.1) rules that the confuser set tests: isolated tumor cells (clusters of
# 0.2 mm or less) in a node leave it negative (N0); an adhesion with no tumor in it microscopically is not T4b, T follows
# the depth of invasion; a metastasis in a non-regional node is distant metastasis (M1a, or M1b beside a second site),
# not a regional node.
_ITC = re.compile(r"isolated tumou?r cells", re.I)
_ADHESION = re.compile(r"\badhe(?:sion|sions|rent|rence)\b", re.I)
_DISTANT_NODE = _SITES["distant nodes"]
# an adhesion is T4b only where the report says tumor is in it; "adherent to <organ>" names the organ without saying the
# tumor invades it
_IN_ADHESION = re.compile(r"\b(?:tumou?r|carcinoma)\b.*\b(?:in|within|into) the adhesion|"
                          r"\badhesions?\b.*\b(?:contains?|shows?|involved by|with)\b.*\b(?:tumou?r|carcinoma)\b", re.I)
_ADHERENT = re.compile(r"\badheren(?:t|ce)\b[^,;.()]*", re.I)
# a statement's parts where a contrast starts ("..., but not through ...") or at a semicolon; a negation in one part
# says nothing about the other
_CONTRAST = re.compile(r";|,?\s+(?=(?:but|whereas|although)\b)", re.I)
_POSITIVE = re.compile(r"metasta|carcinoma|involved|positive|tumou?r", re.I)
_COUNT = re.compile(r"\b(\d+)\s*/\s*(\d+)\b|\b(\d+)\s+(?:involved\s+)?of\s+(?:the\s+)?(\d+)\b")
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                                                   "nine", "ten", "eleven", "twelve"])}


def _statements(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out += [s.strip() for s in _SENTENCE.split(line) if s.strip()]
    return out


def _clauses(statement: str) -> list[str]:
    return [c.strip() for c in _CLAUSE.split(statement) if c and c.strip()]


def _depth(statements: list[str]) -> tuple[str | None, str]:
    """Deepest layer stated positively in the depth statements; gross perforation through the tumor counts as T4a."""
    best, evidence = -1, ""

    def take(cat: str, s: str) -> None:
        nonlocal best, evidence
        if _T_ORDER.index(cat) > best:
            best, evidence = _T_ORDER.index(cat), s

    for s in statements:
        if _ADHESION.search(s):
            # T4b only if a clause says tumor is in the adhesion; otherwise the wall layers the rest of the statement
            # names decide, and an organ named as adherent is not an invaded one
            if any(_IN_ADHESION.search(c) and not _NEG.search(c) for c in _clauses(s)):
                take("T4b", s)
                continue
            for c in _clauses(_INTO_NOT_THROUGH.sub("into the ", _ADHERENT.sub("", s))):
                if _NEG.search(c) or _ADHESION.search(c):
                    continue
                cat = next((cat for cat, rx in _DEPTH if cat != "T4b" and rx.search(c)), None)
                if cat:
                    take(cat, s)
            continue
        if re.search(r"perforat", s, re.I):
            if any(re.search(r"perforat", part, re.I) and not _NEG.search(part) for part in _CONTRAST.split(s)):
                take("T4a", s)
            continue
        if _NOT_DEPTH.search(s):
            continue
        for c in _clauses(_INTO_NOT_THROUGH.sub("into the ", s)):
            if _NEG.search(c):
                continue
            cat = next((cat for cat, rx in _DEPTH if rx.search(c)), None)
            if cat:
                take(cat, s)
    return (_T_ORDER[best] if best >= 0 else None), evidence


def _node_counts(statement: str) -> tuple[int | None, int | None]:
    """(involved, examined) from a lymph node statement: 'x/y', 'x of y', 'examined: y; involved: x',
    'y examined, none involved', 'y nodes examined, x with metastatic carcinoma' and the like."""
    s = re.sub(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b(?=\s+(?:involved\s+)?of\b)", lambda w: _NUMBER_WORDS[w.group(1)],
               statement.lower())
    m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", s) or re.search(r"\b(\d+)\s+(?:involved\s+)?of\s+(?:the\s+)?(\d+)\b", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    ex = (re.search(r"examined:?\s*(\d+)", s)
          or re.search(r"\b(\d+)\s+(?:regional\s+)?(?:lymph\s+)?nodes?\b[^.;]*?\bexamined\b", s)
          or re.search(r"\b(\d+)\s+examined\b", s))
    inv = (re.search(r"(?:involved|positive):?\s*(\d+)", s)
           or re.search(r"\b(\d+)\s+(?:with|containing|positive for|involved by)\s+metasta", s))
    involved = int(inv.group(1)) if inv else (0 if _NEG.search(s) else None)
    return involved, (int(ex.group(1)) if ex else None)


def _deposit_count(statement: str) -> int | None:
    m = re.search(r"\b(\d+)\b", statement)
    if m:
        return int(m.group(1))
    return 0 if _NEG.search(statement) else None


def _n_from_counts(involved: int | None, deposits: int | None) -> str | None:
    """AJCC 8th: N follows the involved node count; deposits give N1c only when every node is negative."""
    if involved is None:
        return None
    if involved == 0:
        if deposits is None:
            return None
        return "N1c" if deposits > 0 else "N0"
    if involved == 1:
        return "N1a"
    if involved <= 3:
        return "N1b"
    return "N2a" if involved <= 6 else "N2b"


def _m_from_ct(line: str | None, pathology_sites: list[str] | tuple = ()) -> tuple[str | None, list[str]]:
    """M from the staging CT line and any distant site the pathology proves (a non-regional node): peritoneal disease
    is M1c; otherwise one distant organ or site is M1a and two or more are M1b; a line that only denies distant disease
    is M0."""
    if line is None and not pathology_sites:
        return None, []
    sites: list[str] = []
    denied = False
    for c in re.split(r"[.;]", line or ""):
        c = c.strip()
        if not c:
            continue
        if _NEG.search(c):
            denied = True
            continue
        sites += [k for k, rx in _SITES.items() if rx.search(c) and k not in sites]
    sites += [k for k in pathology_sites if k not in sites]
    if "peritoneum" in sites:
        return "M1c", sites
    if sites:
        return ("M1a" if len(sites) == 1 else "M1b"), sites
    return ("M0" if denied else None), sites


def _node_statement(statement: str) -> str | None:
    """A regional node statement without its isolated-tumor-cell parts (None when nothing about the nodes is left).
    A focus stated as larger than 0.2 mm is a micrometastasis, a positive node, whatever it is called."""
    if not _ITC.search(statement) or any(float(x) > 0.2 for x in re.findall(r"(\d+(?:\.\d+)?)\s*mm\b", statement)):
        return statement
    rest = " ".join(p for p in _CONTRAST.split(statement) if not _ITC.search(p)).strip()
    return rest if re.search(r"lymph node|\bnodes\b", rest, re.I) else None


def _distant_node_positive(statement: str) -> bool:
    """A non-regional node statement reports a metastasis: an involved count above zero, or with no count, a part that
    names carcinoma or involvement and is not negated ("..., no other sites" negates nothing about the node)."""
    m = _COUNT.search(statement)
    if m:
        return int(m.group(1) or m.group(3)) > 0
    return any(_POSITIVE.search(p) and not _NEG.search(p) for p in re.split(r"[,;]|\s(?=but\b)", statement))


def parse_report(text: str) -> dict:
    """T, N and M recovered from the report text, with the counts and CT sites they rest on (None where unreadable)."""
    ct = _CT.search(text)
    body = text[:ct.start()] + text[ct.end():] if ct else text
    statements = _statements(body)
    t, t_evidence = _depth(statements)
    distant = [s for s in statements if _DISTANT_NODE.search(s)]         # separately submitted non-regional nodes
    node_st = [x for x in (_node_statement(s) for s in statements if re.search(r"lymph node|\bnodes\b", s, re.I)
                           and not re.search(r"deposit", s, re.I) and s not in distant) if x]
    involved, examined = _node_counts(node_st[0]) if len(node_st) == 1 else (None, None)
    dep_st = [s for s in statements if re.search(r"deposit", s, re.I)]
    deposits = _deposit_count(dep_st[0]) if len(dep_st) == 1 else None
    m, sites = _m_from_ct(ct.group("body") if ct else None,
                          ["distant nodes"] if any(_distant_node_positive(s) for s in distant) else [])
    return {"t": t, "n": _n_from_counts(involved, deposits), "m": m, "nodes_involved": involved,
            "nodes_examined": examined, "tumour_deposits": deposits, "ct_sites": ";".join(sites),
            "t_evidence": t_evidence}


def parse_reports(reports: pd.DataFrame) -> pd.DataFrame:
    """parse_report on every row (columns report_id and text); one row per report."""
    return pd.DataFrame([{"report_id": rid, **parse_report(txt)} for rid, txt in zip(reports["report_id"], reports["text"])])

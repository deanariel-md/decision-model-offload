"""Step 0. Download the public NHANES files and the public-use Linked Mortality Files into data/raw/ and check each
against config/nhanes_sha256.txt. Files already present are not downloaded again. No API key is needed.
  python scripts/download_nhanes.py             # NHANES 1999-2008 + LMF, and NHANES III + LMF (reference prior)
  python scripts/download_nhanes.py --list      # print every URL and local path; download nothing
  python scripts/download_nhanes.py --verify    # check the files already in data/raw/; download nothing
Exit code 1 if a file is missing (with --verify) or its checksum differs."""
import argparse, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevity import nhanes as N
from jevity import nhanes3 as N3

SUMS = ROOT / "config" / "nhanes_sha256.txt"


def nhanes_files() -> list[tuple[str, Path]]:
    """(url, local path) for every file the cohort builders read, in the order they read them, each once."""
    c, c3 = N.CONFIG, N3.CFG
    out: dict[Path, str] = {}
    for cycle in c["cycles"]:
        for comp in c["files"]:
            base = c["files"][comp][cycle]
            out.setdefault(N.RAW / cycle / f"{base}.XPT", f"{c['base_url']}/{cycle[:4]}/DataFiles/{base}.xpt")
        name = c["cycles"][cycle]["lmf"]
        out.setdefault(N.RAW / "lmf" / name, f"{c['lmf_url']}/{name}")
    for spec in c3["files"].values():
        for key in ("layout", "data"):
            out.setdefault(N3.RAW / spec[key], f"{c3['base_url']}/{spec[key]}")
    out.setdefault(N3.RAW / c3["lmf_file"], f"{c3['lmf_url']}/{c3['lmf_file']}")
    return [(u, p) for p, u in out.items()]


def rel(p: Path) -> str:
    return p.relative_to(ROOT / "data").as_posix()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected() -> dict[str, str]:
    out = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, path = line.split(maxsplit=1)
            out[path.strip()] = digest
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    files = nhanes_files()
    if a.list:
        for url, p in files:
            print(f"{url}\n    -> data/{rel(p)}")
        sys.exit(0)
    want = expected()
    bad = 0
    for url, p in files:
        if not a.verify:
            N._fetch(url, p)
        if not p.exists():
            print(f"MISSING  data/{rel(p)}"); bad += 1; continue
        d, e = sha256(p), want.get(rel(p))
        status = "ok" if d == e else ("NO CHECKSUM" if e is None else "CHECKSUM DIFFERS")
        bad += status != "ok"
        print(f"{status:16s} data/{rel(p)}")
    print(f"{len(files) - bad} of {len(files)} files match config/nhanes_sha256.txt")
    sys.exit(1 if bad else 0)

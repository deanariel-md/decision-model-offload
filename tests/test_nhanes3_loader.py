"""End-to-end run of the NHANES III loader on tiny fixed-width files written from CDC-style SAS layouts."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jevity.nhanes3 as N3


def _write(tmp: Path, name: str, layout: dict, rows: list[dict]):
    width = max(e for _, e in layout.values())
    sas = "DATA X; INFILE 'x'; INPUT\n" + "\n".join(f"  {v} {s}-{e}" if s != e else f"  {v} {s}" for v, (s, e) in layout.items()) + "\n;\n"
    (tmp / f"{name}.sas").write_text(sas)
    lines = []
    for r in rows:
        buf = [" "] * width
        for v, (s, e) in layout.items():
            txt = str(r.get(v, "")).rjust(e - s + 1)[: e - s + 1]
            buf[s - 1:e] = list(txt)
        lines.append("".join(buf))
    (tmp / f"{name}.dat").write_text("\n".join(lines) + "\n")


def test_loader(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 60
    adult_lay = {"SEQN": (1, 5), "HSAGEIR": (6, 7), "HSSEX": (8, 8), "DMARETHN": (9, 9), "HFA8R": (10, 11), "DMPPIR": (12, 17),
                 "HFB2": (18, 18), "HFB7": (19, 19), "HFB9": (20, 20), "HFB11": (21, 21), "HAR1": (22, 22), "HAR3": (23, 23),
                 "HAC1C": (24, 24), "HAF10": (25, 25), "HAC1D": (26, 26), "HAC1G": (27, 27), "HAC1F": (28, 28), "HAC1N": (29, 29),
                 "HAC1O": (30, 30), "HAD1": (31, 31), "HAB1": (32, 32)}
    exam_lay = {"SEQN": (1, 5), "BMPBMI": (6, 11), "BMPWAIST": (12, 17), "PEPMNK1R": (18, 20), "PEPMNK5R": (21, 23)}
    lab_lay = {"SEQN": (1, 5), "AMP": (6, 9), "CEP": (10, 14), "SGP": (15, 18), "CRP": (19, 23), "LMPPCNT": (24, 28),
               "MVPSI": (29, 33), "RWP": (34, 38), "APPSI": (39, 42), "WCP": (43, 47), "GHP": (48, 51), "TCP": (52, 54),
               "HDP": (55, 57), "UBP": (58, 62), "URP": (63, 66)}
    ad, ex, lb, lmf = [], [], [], []
    for i in range(n):
        sq = 3 + i
        ad.append({"SEQN": sq, "HSAGEIR": int(rng.integers(35, 85)), "HSSEX": 1 + i % 2, "DMARETHN": 1 + i % 4, "HFA8R": 88 if i == 0 else 12,
                   "DMPPIR": "888888" if i == 1 else "1.234", "HFB2": 2, "HFB7": 2, "HFB9": 2, "HFB11": 1 if i % 3 else 2,
                   "HAR1": 1, "HAR3": 2, "HAC1C": 2, "HAF10": 2, "HAC1D": 2, "HAC1G": 2, "HAC1F": 2, "HAC1N": 2, "HAC1O": 1 if i == 5 else 2,
                   "HAD1": 2, "HAB1": 3})
        ex.append({"SEQN": sq, "BMPBMI": "27.4", "BMPWAIST": "98.2", "PEPMNK1R": 128, "PEPMNK5R": 78})
        lb.append({"SEQN": sq, "AMP": "4.2", "CEP": "1.1", "SGP": "8888" if i == 2 else "95", "CRP": "0.21", "LMPPCNT": "30.1",
                   "MVPSI": "90.2", "RWP": "12.9", "APPSI": "88", "WCP": "6.8", "GHP": "5.4", "TCP": 210, "HDP": 48, "UBP": "10.0", "URP": "120"})
        lmf.append(f"{sq:<14d}1{int(rng.random() < 0.3)}".ljust(42) + f"{300:>3d}{int(rng.integers(20, 300)):>3d}")
    for name, lay, rows in (("adult", adult_lay, ad), ("exam", exam_lay, ex), ("lab", lab_lay, lb)):
        _write(tmp_path, name, lay, rows)
    (tmp_path / "NHANES_III_MORT_2019_PUBLIC.dat").write_text("\n".join(lmf) + "\n")
    monkeypatch.setattr(N3, "_fetch", lambda url, dest: tmp_path / Path(url).name)
    monkeypatch.setattr(N3, "ROOT", tmp_path)
    df = N3.build_nhanes3_cohort(out=tmp_path / "c.parquet")
    assert len(df) > 0 and set(df.split) == {"prior"} and (df.SEQN >= 10_000_000).all()
    assert df.age.between(40, 79).all()
    assert df.insured.isin(["yes", "no"]).all()
    assert "APPSI" not in df and (df.alk_phos_u_l == 88).all()          # a real 88 survives the fill rule
    r3 = df[df.SEQN == 10_000_000 + 3 + 2]
    assert r3.empty or r3.glucose_mg_dl.isna().all()                     # 8888 fill masked -> excluded or missing
    assert abs(df.creatinine_mg_dl.iloc[0] - (0.960 * 1.1 - 0.184)) < 1e-9

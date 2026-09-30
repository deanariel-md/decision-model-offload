"""Recording and file-write robustness: Jev's token fields are recorded, a checkpoint that cannot be written is
skipped while the final write is retried, and the batch manifest's replace is retried while a reader holds the file.
No network."""
import json
from pathlib import Path

import pandas as pd
import pytest

from jevity import batch as BT
from jevity import runner as R
from jevity.clients import RawStore


def test_usage_reads_jev_and_llm_token_fields():
    assert R._usage({"usage": {"prompt_tokens": 900, "completion_tokens": 12}}, "prompt_tokens", "input_tokens") == 900
    assert R._usage({"usage": {"input_tokens": 1118, "output_tokens": 86}}, "prompt_tokens", "input_tokens") == 1118
    assert R._usage({"usage": {"input_tokens": 1118, "output_tokens": 86}}, "completion_tokens", "output_tokens") == 86
    assert R._usage({"usage": None}, "prompt_tokens", "input_tokens") is None


def test_execute_records_jev_tokens(tmp_path):
    calls = [R.Call("jev", 1, "baseline", False, 0, "t")]

    class Fake:
        def probability(self, text, stmt):
            return {"p": 0.2, "valid": True, "provider": "TypeSafe", "usage": {"input_tokens": 1000, "output_tokens": 80}}

    df = R.execute(calls, lambda m, r, v: Fake(), "nhanes", tmp_path / "calls.parquet", workers=1)
    assert df.tokens_in.iloc[0] == 1000 and df.tokens_out.iloc[0] == 80
    assert (tmp_path / "calls.parquet").exists()


def test_write_skips_a_failed_checkpoint_and_retries_the_final(tmp_path, monkeypatch):
    n = {"k": 0}
    real = pd.DataFrame.to_parquet

    def flaky(self, *a, **k):
        n["k"] += 1
        if n["k"] <= 2:
            raise OSError(22, "Invalid argument")
        return real(self, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    df = pd.DataFrame({"a": [1]})
    assert R._write(df, tmp_path / "x.parquet", tries=1) is False       # checkpoint: skipped, no exception
    assert R._write(df, tmp_path / "x.parquet") is True                 # final: retried until it lands
    assert pd.read_parquet(tmp_path / "x.parquet").a.iloc[0] == 1


def test_manifest_save_retries_while_the_file_is_held(tmp_path, monkeypatch):
    man = BT.Manifest(RawStore(tmp_path))
    man.items = [{"key": "k1", "status": "completed", "ingested": True}]
    real = Path.replace
    n = {"k": 0}

    def held(self, target):
        n["k"] += 1
        if n["k"] <= 3:
            raise PermissionError(13, "Access is denied")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", held)
    monkeypatch.setattr(BT.time, "sleep", lambda s: None)
    man.save()
    assert json.loads(man.path.read_text())[0]["key"] == "k1" and n["k"] == 4

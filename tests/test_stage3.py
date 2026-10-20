from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from caqec.evaluation.protocol import file_sha256, git_state, single_use_guard, verify_split_checksum
from caqec.plotting.decoder_plots import plot_failure_rates, plot_paired_differences
from caqec.simulation.dataset import DatasetConfig, generate_split, write_card

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "exp05_stage3_test_evaluation.py"


def test_single_use_guard(tmp_path):
    target = tmp_path / "results.json"
    single_use_guard(target)                       # nothing there yet: fine
    target.write_text("{}")
    with pytest.raises(RuntimeError):
        single_use_guard(target)
    single_use_guard(target, allow_rerun=True)     # explicit override


def test_split_checksum_verification(tmp_path):
    cfg = DatasetConfig()
    splits = {n: generate_split(cfg, n, 200, np.random.SeedSequence(i)) for i, n in enumerate(("train", "val", "test"))}
    write_card(tmp_path / "dataset_card.json", cfg, splits)
    assert len(verify_split_checksum(splits["test"], tmp_path / "dataset_card.json", "test")) == 64
    tampered = {k: v.copy() for k, v in splits["test"].items()}
    tampered["y"][0] ^= 1
    with pytest.raises(ValueError):
        verify_split_checksum(tampered, tmp_path / "dataset_card.json", "test")


def test_file_hash_and_git_state(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    assert file_sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert set(git_state(tmp_path)) >= {"head", "uncommitted"}  # tmp_path is not a repo: head is None


def _metrics(rate, half=0.002):
    return {"failure_rate": rate, "ci_low": rate - half, "ci_high": rate + half}


def test_plots_are_written(tmp_path):
    groups = ["all", "0.02"]
    by_decoder = {"lookup": {g: _metrics(0.07) for g in groups}, "tree": {g: _metrics(0.06) for g in groups}}
    paired = {g: {"tree": {"diff": 0.01, "ci_low": 0.005, "ci_high": 0.015}} for g in groups}
    plot_failure_rates(by_decoder, groups, tmp_path / "a.png", "t")
    plot_paired_differences(paired, ["tree"], groups, tmp_path / "b.png", "t")
    for name in ("a.png", "b.png"):
        assert (tmp_path / name).stat().st_size > 2000


def test_stage3_script_guards_come_before_the_test_split_is_read():
    text = SCRIPT.read_text()
    guard, read, verify = (text.index(s) for s in (
        "single_use_guard(results_path", 'load_split(data_dir / "test.npz")', "verify_split_checksum(test"))
    assert guard < read < verify
    assert "fit(" not in text and "candidate_grid" not in text   # nothing is trained or selected here
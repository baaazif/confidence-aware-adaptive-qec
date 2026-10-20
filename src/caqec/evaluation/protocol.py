"""Evaluation-protocol helpers: single-use test guard, provenance, checksums."""
import hashlib
import json
import subprocess
from pathlib import Path

from caqec.simulation.dataset import checksum


def single_use_guard(results_path, allow_rerun: bool = False) -> None:
    """Refuse to evaluate the test split again once a results file exists.
    A discipline mechanism, not security: re-running after seeing test results invites tuning on it."""
    if Path(results_path).exists() and not allow_rerun:
        raise RuntimeError(
            f"{results_path} already exists, so the test split was already evaluated. "
            "Evaluating again after seeing the results invites tuning on the test set. "
            "Use --allow-rerun only with a documented reason.")


def verify_split_checksum(arrays: dict, card_path, name: str) -> str:
    """Check the loaded split against the checksum recorded when the dataset was generated."""
    expected = json.loads(Path(card_path).read_text())["sha256"][name]
    actual = checksum(arrays)
    if actual != expected:
        raise ValueError(f"{name} split does not match dataset_card.json (file changed or regenerated)")
    return actual


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state(root) -> dict:
    """Commit hash and uncommitted files, recorded as provenance (never required to be clean)."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return {"head": run("rev-parse", "HEAD"), "uncommitted": run("status", "--porcelain").splitlines()}
    except Exception as exc:  # git missing or not a repository
        return {"head": None, "uncommitted": None, "error": str(exc)}
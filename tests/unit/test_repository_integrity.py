"""Every file the pipeline needs must actually ship.

A file that exists on a developer machine but never reaches the cluster produces a
confusing runtime failure far from its cause -- `slurm/dls/cuda.sh` going missing shows up
as GPU jobs silently falling back to a CPU backend. `.gitignore` is the usual way that
happens, so no pipeline-critical path may be ignored.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = [
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    "README.md",
    "slurm/common.sh",
    "slurm/dls/cuda.sh",
    "scripts/submit.sh",
    "scripts/cluster_preflight.sh",
    "src/crystal_field/cli.py",
    "configs/1ubq/default.yaml",
    "configs/1ubq/prior_grid.yaml",
    ".github/workflows/ci.yml",
]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="not a git checkout",
)


def _ignored(path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


@pytest.mark.parametrize("relative", REQUIRED)
def test_required_file_exists(relative):
    assert (ROOT / relative).is_file(), f"{relative} is missing from the checkout"


@pytest.mark.parametrize("relative", REQUIRED)
def test_required_file_is_not_gitignored(relative):
    assert not _ignored(relative), f"{relative} is gitignored and would never reach the cluster"


@pytest.mark.parametrize(
    "pattern", ["slurm/*.sbatch", "slurm/*.sh", "slurm/dls/*.sh", "scripts/*.sh", "src/crystal_field/**/*.py"]
)
def test_no_pipeline_file_is_gitignored(pattern):
    matched = sorted(ROOT.glob(pattern))
    assert matched, f"{pattern} matched nothing"
    ignored = [str(p.relative_to(ROOT)) for p in matched if _ignored(str(p.relative_to(ROOT)))]
    assert ignored == [], f"gitignored pipeline files would never ship: {ignored}"


def test_experimental_data_and_run_products_are_ignored():
    """The inverse: data must NOT ship."""
    for relative in ["data/1ubq/1ubq-sf.cif", "runs/1UBQ/shared/reflections.npz", ".venv/bin/python"]:
        assert _ignored(relative), f"{relative} should be gitignored"

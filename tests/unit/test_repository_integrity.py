"""Every file the pipeline needs must actually ship.

A file that exists on a developer machine but never reaches the cluster produces a
confusing runtime failure far from its cause -- `slurm/dls/cuda.sh` going missing shows up
as GPU jobs silently falling back to a CPU backend. `.gitignore` is the usual way that
happens, so no pipeline-critical path may be ignored.
"""

import re
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


def test_the_python_range_matches_what_ci_actually_validates():
    """An unbounded requires-python let a clean rebuild land on 3.14, where the JAX CUDA
    wheels had no matching build. The bound is only meaningful if it tracks CI."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    requires = re.search(r'requires-python = "([^"]+)"', pyproject).group(1)
    upper = re.search(r"<\s*3\.(\d+)", requires)
    assert upper, f"requires-python must have an upper bound, got {requires!r}"
    lower = re.search(r">=\s*3\.(\d+)", requires)
    assert lower, f"requires-python must have a lower bound, got {requires!r}"

    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    tested = sorted(int(m) for m in set(re.findall(r'"3\.(\d+)"', workflow)))
    assert tested, "could not find the CI python matrix"

    assert int(lower.group(1)) <= tested[0], "requires-python excludes the oldest python CI tests"
    # Every version CI exercises must be allowed, and nothing beyond it: an upper bound
    # looser than the matrix is exactly how a rebuild silently reached 3.14.
    assert int(upper.group(1)) == tested[-1] + 1, (
        f"CI tests up to 3.{tested[-1]} but requires-python stops at <3.{upper.group(1)}"
    )


def test_the_cuda_extra_guidance_is_not_a_bare_site_default():
    """`ptxas too old` on a V100 is really `CUDA 13 dropped compute capability 7.0`.
    The guidance must key on the GPU, not on whatever `module load cuda` provides."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "cuda12" in pyproject and "cuda13" in pyproject
    assert "compute" in pyproject.lower(), "extras must be documented by compute capability"

    cuda_sh = (ROOT / "slurm/dls/cuda.sh").read_text()
    assert "compute_cap" in cuda_sh, "cuda.sh must query the GPU's compute capability"
    assert "cuda12" in cuda_sh, "the mismatch message must name the fix"

    preflight = (ROOT / "scripts/cluster_preflight.sh").read_text()
    assert "compute_cap" in preflight, "preflight must check the plugin against the GPU"


WORKSPACE_VARS = (
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "FIELDX_DATA_ROOT",
    "FIELDX_RUNS_ROOT",
    "FIELDX_JAX_CACHE",
)


def _workspace_env(tmp_path, script_args="", pre_exports=""):
    """Source workspace-env.sh in a clean shell and report what it exported."""
    body = f"{pre_exports}\nsource '{ROOT}/scripts/workspace-env.sh' {script_args} 2>/dev/null\n"
    body += "\n".join(f'printf "%s=%s\\n" {v} "${v}"' for v in ("FIELDX_WORKSPACE", *WORKSPACE_VARS))
    proc = subprocess.run(["bash", "-c", body], capture_output=True, text=True, cwd=ROOT)
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


def test_one_workspace_path_fills_in_every_location(tmp_path):
    """Setting five paths by hand is five chances to put one on the home quota."""
    workspace = tmp_path / "ws"
    env = _workspace_env(tmp_path, script_args=f"'{workspace}'")

    assert env["FIELDX_WORKSPACE"] == str(workspace)
    for var in WORKSPACE_VARS:
        assert env[var].startswith(str(workspace)), f"{var} escaped the workspace: {env[var]}"
        assert Path(env[var]).is_dir(), f"{var} was not created: {env[var]}"


def test_an_explicit_setting_beats_the_workspace_default(tmp_path):
    """A workspace default that silently overrode a deliberate choice would be worse
    than no default at all -- someone pointing the cache at fast local scratch must keep
    it. Without this the test above passes whether or not precedence is respected."""
    chosen = tmp_path / "chosen-cache"
    env = _workspace_env(
        tmp_path,
        script_args=f"'{tmp_path / 'ws'}'",
        pre_exports=f"export UV_CACHE_DIR='{chosen}'",
    )
    assert env["UV_CACHE_DIR"] == str(chosen)
    # The others still come from the workspace, so precedence is per-variable.
    assert env["FIELDX_DATA_ROOT"].startswith(str(tmp_path / "ws"))


def test_nothing_is_exported_without_a_workspace(tmp_path):
    """Sourcing the script with no workspace must not invent paths for a user who
    manages their own; an empty FIELDX_WORKSPACE is the untouched default."""
    env = _workspace_env(tmp_path)
    assert env["FIELDX_WORKSPACE"] == ""
    for var in WORKSPACE_VARS:
        assert env[var] == "", f"{var} was set without a workspace: {env[var]}"

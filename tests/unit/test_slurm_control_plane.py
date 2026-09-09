"""SLURM control-plane tests (plan section 20H).

These encode the bugs the refactor removed, so they cannot come back:

  * an sbatch script resolving repository files through "$0" (SLURM stages the script
    under /var/spool/slurm/..., so "$0" does not point into the repository);
  * a required CFI_UV_ENV variable or a generated environment/cluster.env;
  * a generic prior-grid fallback onto the 6O2H dataset.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SBATCH_SCRIPTS = sorted(ROOT.glob("slurm/*.sbatch"))
SHELL_SCRIPTS = sorted([*ROOT.glob("scripts/*.sh"), *ROOT.glob("slurm/*.sh"), *ROOT.glob("slurm/dls/*.sh")])
GPU_JOBS = {
    "30_numerics",
    "40_gpu_tests",
    "45_atomic_benchmark",
    "50_prior_candidate",
    "60_final_fit",
    "70_information",
    "80_evaluate_free",
}


def test_the_expected_jobs_exist():
    names = {p.stem for p in SBATCH_SCRIPTS}
    assert names == {
        "00_inspect",
        "10_prepare",
        "20_make_rho0",
        "25_scaling_train",
        "30_numerics",
        "40_gpu_tests",
        "45_atomic_benchmark",
        "50_prior_candidate",
        "52_select_prior",
        "55_scaling_work",
        "60_final_fit",
        "70_information",
        "75_freeze_model",
        "80_evaluate_free",
    }


@pytest.mark.parametrize("script", [*SBATCH_SCRIPTS, *SHELL_SCRIPTS], ids=lambda p: p.name)
def test_shell_syntax_is_valid(script):
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", SBATCH_SCRIPTS, ids=lambda p: p.name)
def test_sbatch_never_resolves_paths_from_dollar_zero(script):
    text = script.read_text()
    assert 'dirname "$0"' not in text
    assert "dirname $0" not in text
    assert not re.search(r"\$\{?0\}?\b", text.replace('"$0" must never be used', "")), (
        "an sbatch script must not use $0 to locate repository files"
    )


@pytest.mark.parametrize("script", SBATCH_SCRIPTS, ids=lambda p: p.name)
def test_sbatch_resolves_the_repository_through_slurm_submit_dir(script):
    text = script.read_text()
    assert 'cd "$SLURM_SUBMIT_DIR"' in text
    assert 'source "$SLURM_SUBMIT_DIR/slurm/common.sh"' in text


@pytest.mark.parametrize("script", SBATCH_SCRIPTS, ids=lambda p: p.name)
def test_gpu_jobs_load_cuda_and_cpu_jobs_do_not(script):
    text = script.read_text()
    wants_gpu = "--gres=gpu" in text
    assert (script.stem in GPU_JOBS) == wants_gpu, f"{script.stem} GPU allocation disagrees with the expected set"
    assert ("fieldx_load_cuda" in text) == wants_gpu, "every GPU job must load CUDA itself; no CPU job may require it"


@pytest.mark.parametrize("script", [*SBATCH_SCRIPTS, *SHELL_SCRIPTS], ids=lambda p: p.name)
def test_no_active_script_requires_the_retired_environment(script):
    text = script.read_text()
    assert "CFI_UV_ENV" not in text
    assert "environment/cluster.env" not in text
    assert "bootstrap_cluster" not in text


@pytest.mark.parametrize("script", SHELL_SCRIPTS + SBATCH_SCRIPTS, ids=lambda p: p.name)
def test_no_dataset_specific_fallback(script):
    """6O2H is an example dataset, never the generic default."""
    text = script.read_text().lower()
    assert "6o2h" not in text, "no pipeline script may reference a specific dataset"
    assert "1ubq" not in text


def test_submit_requires_a_dataset_prior_grid_and_never_substitutes_one():
    text = (ROOT / "scripts/submit.sh").read_text()
    assert "Prior grid not found" in text
    assert "prior_grid.yaml" in text
    # The retired behaviour: [[ -f "$GRID" ]] || GRID="$ROOT/configs/6o2h/prior_grid.yaml"
    assert "configs/6o2h" not in text


def test_only_the_dls_file_carries_site_specific_paths():
    """Section 7: the generic scientific code must not contain DLS-specific paths."""
    offenders = []
    for path in [*ROOT.glob("src/**/*.py"), *SBATCH_SCRIPTS, ROOT / "slurm/common.sh", *ROOT.glob("scripts/*.sh")]:
        text = path.read_text()
        # "slurm/dls/cuda.sh" is a repository path, not a site path; "/dls/data" and
        # "dls_sw" are the real Diamond filesystem locations.
        if "dls_sw" in text or "/dls/data" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"DLS paths leaked outside slurm/dls/: {offenders}"


def test_dls_cuda_init_loads_the_cuda_module():
    text = (ROOT / "slurm/dls/cuda.sh").read_text()
    assert "module load cuda" in text


BASH = shutil.which("bash") or "/bin/bash"


def _run_common(env, cwd):
    script = 'source "$1/slurm/common.sh"; echo "ROOT=$PROJECT_ROOT"; echo "UV=$UV"; echo "ENV=$UV_PROJECT_ENVIRONMENT"'
    return subprocess.run(
        [BASH, "-c", script, "bash", str(ROOT)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
    )


def test_common_sh_resolves_the_repository_from_a_fake_slurm_environment(tmp_path):
    """The staged-script scenario: cwd is a spool directory, SLURM_SUBMIT_DIR is the repo."""
    spool = tmp_path / "var" / "spool" / "slurm" / "job4242"
    spool.mkdir(parents=True)
    env = {**os.environ, "SLURM_SUBMIT_DIR": str(ROOT), "SLURM_JOB_ID": "4242"}
    result = _run_common(env, spool)
    assert result.returncode == 0, result.stderr
    assert f"ROOT={ROOT}" in result.stdout
    assert f"ENV={ROOT}/.venv" in result.stdout


def test_common_sh_rejects_a_submit_dir_that_is_not_the_repository(tmp_path):
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    env = {**os.environ, "SLURM_SUBMIT_DIR": str(fake_repo), "SLURM_JOB_ID": "1"}
    # SLURM_SUBMIT_DIR is not a FieldX checkout, so resolution falls back to the real
    # location of common.sh rather than silently using the wrong tree.
    result = _run_common(env, tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"ROOT={ROOT}" in result.stdout


def test_common_sh_honours_an_explicit_uv_executable(tmp_path):
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\necho fake\n")
    fake_uv.chmod(0o755)
    env = {**os.environ, "SLURM_SUBMIT_DIR": str(ROOT), "FIELDX_UV": str(fake_uv)}
    result = _run_common(env, tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"UV={fake_uv}" in result.stdout


def test_common_sh_fails_when_no_uv_is_available(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PATH"}
    env.update({"SLURM_SUBMIT_DIR": str(ROOT), "PATH": str(tmp_path), "FIELDX_UV": str(tmp_path / "absent")})
    result = _run_common(env, tmp_path)
    assert result.returncode != 0
    assert "no usable uv executable" in result.stderr


def test_common_sh_points_the_uv_cache_at_a_configured_location(tmp_path):
    cache = tmp_path / "uv-cache"
    env = {**os.environ, "SLURM_SUBMIT_DIR": str(ROOT), "FIELDX_UV_CACHE": str(cache)}
    result = subprocess.run(
        [BASH, "-c", f'source "{ROOT}/slurm/common.sh"; echo "CACHE=$UV_CACHE_DIR"'],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert f"CACHE={cache}" in result.stdout


def test_submit_refuses_a_missing_prior_grid(tmp_path):
    config = tmp_path / "default.yaml"
    config.write_text("run: {}\n")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit.sh"), str(config)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 2
    assert "Prior grid not found" in result.stderr


def test_submit_rejects_a_missing_config(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit.sh"), str(tmp_path / "absent.yaml")],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 2
    assert "Configuration not found" in result.stderr


@pytest.mark.skipif(shutil.which("sbatch") is not None, reason="a real SLURM host would submit jobs")
def test_submit_stops_when_sbatch_is_absent(tmp_path):
    (tmp_path / "prior_grid.yaml").write_text("candidates: [{name: a, prior: {}}]\n")
    shutil.copy(ROOT / "configs/1ubq/default.yaml", tmp_path / "default.yaml")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit.sh"), str(tmp_path / "default.yaml")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 2
    assert "sbatch not found" in result.stderr


def test_free_evaluation_is_not_wired_into_the_dependency_chain():
    """Section 8: 80_evaluate_free must never be submitted automatically."""
    text = (ROOT / "scripts/submit.sh").read_text()
    submitted = re.findall(r"submit\s+(\d\d_\w+)\.sbatch", text)
    assert "80_evaluate_free" not in submitted
    assert "50_prior_candidate.sbatch" in text
    # It is only printed, as the exact command the operator must run by hand.
    assert text.count("80_evaluate_free.sbatch") == 1
    assert "sbatch ${CFI_SBATCH_ARGS:-}" in text


def test_dependency_chain_is_complete_and_ordered():
    text = (ROOT / "scripts/submit.sh").read_text()
    for dependency in [
        '--dependency=afterok:"$j_inspect"',
        '--dependency=afterok:"$j_prepare"',
        '--dependency=afterok:"$j_rho0"',
        '--dependency=afterok:"$j_scaletr"',
        '--dependency=afterok:"$j_numerics"',
        '--dependency=afterok:"$j_gputest"',
        '--dependency=afterok:"$j_atomic"',
        '--dependency=afterok:"$j_priors"',
        '--dependency=afterok:"$j_select"',
        '--dependency=afterok:"$j_scalewk"',
        '--dependency=afterok:"$j_final"',
    ]:
        assert dependency in text, dependency
    # The freeze waits for both the final fit and the information spectrum.
    assert '--dependency=afterok:"$j_final":"$j_info"' in text


def test_submit_exports_only_the_minimum_runtime_metadata():
    text = (ROOT / "scripts/submit.sh").read_text()
    exported = set()
    for block in re.findall(r'--export="([^"]*)"', text) + re.findall(r'_ENV="([^"]*)"', text):
        exported.update(re.findall(r"\b(CFI_[A-Z_]+)=", block))
    assert exported <= {"CFI_CONFIG", "CFI_PROJECT_ROOT", "CFI_PRIOR_MANIFEST", "CFI_SELECTED_CONFIG"}


def test_prior_candidates_are_submitted_as_an_array():
    text = (ROOT / "scripts/submit.sh").read_text()
    assert '--array="0-$((N-1))%$LIMIT"' in text
    assert "CFI_ARRAY_LIMIT" in text


def test_dls_cuda_init_points_xla_at_the_site_toolkit(tmp_path):
    """The ptxas fix, pinned.

    XLA prefers the ptxas bundled in the pip CUDA wheels, which fails with "ptxas too
    old" when it predates the GPU. The site init must redirect XLA at the toolkit that
    `module load cuda` provides.
    """
    fake = tmp_path / "cuda"
    (fake / "bin").mkdir(parents=True)
    ptxas = fake / "bin" / "ptxas"
    ptxas.write_text("#!/bin/sh\necho 'release 13.4, V13.4.0'\n")
    ptxas.chmod(0o755)

    script = (
        f'export CUDA_HOME="{fake}"; source "{ROOT}/slurm/dls/cuda.sh"; echo "FLAGS=$XLA_FLAGS"; echo "DIR=$CUDA_DIR"'
    )
    result = subprocess.run([BASH, "-c", script], capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert f"FLAGS=--xla_gpu_cuda_data_dir={fake}" in result.stdout
    assert f"DIR={fake}" in result.stdout


def test_dls_cuda_init_survives_a_host_with_no_toolkit(tmp_path):
    """Off-cluster it must warn and continue, not abort a job."""
    script = (
        f'unset CUDA_HOME CUDA_ROOT CUDA_PATH CUDA_DIR; PATH=/nonexistent; source "{ROOT}/slurm/dls/cuda.sh"; echo DONE'
    )
    result = subprocess.run([BASH, "-c", script], capture_output=True, text=True, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "DONE" in result.stdout
    assert "ptxas too old" in result.stderr or "no CUDA toolkit" in result.stderr


def test_cuda_init_is_only_referenced_through_the_override():
    """slurm/common.sh must honour FIELDX_CUDA_INIT so other sites can substitute one."""
    text = (ROOT / "slurm/common.sh").read_text()
    assert "FIELDX_CUDA_INIT" in text
    assert "slurm/dls/cuda.sh" in text

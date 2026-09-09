"""Command-line surface: the login-node commands must work without an accelerator."""

import json

import pytest
from typer.testing import CliRunner

import crystal_field.analysis.decomposition as decomposition_module
from crystal_field.cli import app

runner = CliRunner()


def _json(result):
    assert result.exit_code == 0, result.output
    return json.loads(result.output.replace("\n", ""))


def test_help_lists_the_pipeline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["config-check", "init-pdb", "prepare", "make-rho0", "check-rho0", "fit-map", "evaluate-free"]:
        assert command in result.output


def test_config_check_succeeds_on_a_complete_dataset(tiny_dataset):
    result = runner.invoke(app, ["config-check", str(tiny_dataset["config_path"])])
    assert result.exit_code == 0
    assert '"ok": true' in result.output.replace("\n", "").replace(" ", " ")


def test_config_check_fails_on_a_broken_config(tmp_path):
    broken = tmp_path / "broken.yaml"
    broken.write_text("run: {output_dir: out}\n")
    assert runner.invoke(app, ["config-check", str(broken)]).exit_code != 0


def test_inspect_reports_columns_and_the_derived_grid(tiny_dataset):
    result = runner.invoke(app, ["inspect", str(tiny_dataset["config_path"])])
    assert result.exit_code == 0
    assert "suggested_grid_shape" in result.output
    assert "column_suggestions" in result.output


def test_estimate_memory_runs_before_preparation(tiny_dataset):
    result = runner.invoke(app, ["estimate-memory", str(tiny_dataset["config_path"])])
    assert result.exit_code == 0
    assert "estimated_gpu_GiB_information" in result.output


def test_inspect_h5_requires_a_configured_file(tiny_dataset):
    result = runner.invoke(app, ["inspect-h5", str(tiny_dataset["config_path"])])
    assert result.exit_code != 0


def test_evaluate_free_refuses_without_a_model_lock(tiny_dataset):
    result = runner.invoke(app, ["evaluate-free", str(tiny_dataset["config_path"])])
    assert result.exit_code != 0
    assert isinstance(result.exception, (RuntimeError, SystemExit))


@pytest.mark.parametrize("command", ["prepare", "make-rho0", "fit-map", "decompose"])
def test_pipeline_commands_reject_a_missing_config(command, tmp_path):
    assert runner.invoke(app, [command, str(tmp_path / "absent.yaml")]).exit_code != 0


def test_decompose_is_listed_in_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "decompose" in result.output


def test_decompose_invokes_the_analysis_with_the_requested_options(tiny_dataset, monkeypatch):
    seen = {}

    def fake(cfg, basis=None, n_trials=None):
        seen["basis"] = basis
        seen["n_trials"] = n_trials
        return {"targets": {}}

    # decompose_cmd imports run_decomposition inside its body (the codebase convention
    # for every analysis-invoking command), so the name to patch is where it is looked
    # up at call time -- the source module -- not an attribute of the cli module itself.
    monkeypatch.setattr(decomposition_module, "run_decomposition", fake)
    result = runner.invoke(app, ["decompose", str(tiny_dataset["config_path"]), "--basis", "full", "--n-trials", "3"])
    assert result.exit_code == 0, result.output
    assert seen == {"basis": "full", "n_trials": 3}


@pytest.mark.parametrize("value", ["0", "-1"])
def test_decompose_rejects_a_non_positive_trial_count(tiny_dataset, monkeypatch, value):
    """The flag must enforce the same `ge=1` the configuration does.

    Unvalidated, `--n-trials 0` is falsy and silently falls back to the configured default
    -- the run reports 8 trials while the operator asked for none. `--n-trials -1` is worse:
    `range(-1)` is empty, the control's mean is a mean over nothing, and `json.dumps` writes
    the resulting NaN as a bare `NaN` token that no RFC-8259 parser will read back.
    """

    def fail(*args, **kwargs):
        raise AssertionError("the analysis must not be reached with an invalid trial count")

    monkeypatch.setattr(decomposition_module, "run_decomposition", fail)
    result = runner.invoke(app, ["decompose", str(tiny_dataset["config_path"]), "--n-trials", value])
    assert result.exit_code != 0

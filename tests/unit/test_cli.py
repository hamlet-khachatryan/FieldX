"""Command-line surface: the login-node commands must work without an accelerator."""

import json

import pytest
from typer.testing import CliRunner

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


@pytest.mark.parametrize("command", ["prepare", "make-rho0", "fit-map"])
def test_pipeline_commands_reject_a_missing_config(command, tmp_path):
    assert runner.invoke(app, [command, str(tmp_path / "absent.yaml")]).exit_code != 0

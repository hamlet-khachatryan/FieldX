"""Atomic representability benchmark (plan section 20F).

Ordinary refinement directions -- coordinate, occupancy, isotropic B and anisotropic ADP
perturbations -- must stay representable at a sane latent cost. This is a diagnostic on
the prior, not a restriction on what v3 may discover: a meaningful density feature is
allowed to lie outside this tangent space entirely.
"""

import gemmi
import numpy as np
import pytest
import yaml
from conftest import config_payload, synthetic_reflections, write_mtz, write_tiny_model

from crystal_field.config import AppConfig, resolve_payload_paths
from crystal_field.crystallography.atomic_benchmark import run_atomic_benchmark
from crystal_field.crystallography.io import make_model_density, make_solvent_mask, prepare_reflections
from crystal_field.crystallography.scaling import fit_baseline_scaling

pytestmark = pytest.mark.integration

ANISOU_PDB = """CRYST1   20.000   24.000   28.000  90.00  90.00  90.00 P 1           1
ATOM      1  CA  ALA A   1       5.200   5.400   6.300  1.00 11.00           C
ANISOU    1  CA  ALA A   1     1400   1300   1500     50    -30     20       C
ATOM      2  CB  ALA A   1       5.000   6.700   7.100  1.00 14.00           C
ANISOU    2  CB  ALA A   1     1800   1700   1900    -60     40    -10       C
ATOM      3  S   MET A   2      15.000  18.000  20.000  1.00 20.00           S
ANISOU    3  S   MET A   2     2500   2400   2600     30     20     10       S
END
"""


def _dataset(tmp_path, model_text=None, prior=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model.pdb"
    if model_text is None:
        write_tiny_model(model)
    else:
        model.write_text(model_text)
    hkls, observed, sigma, free = synthetic_reflections(model)
    mtz = write_mtz(tmp_path / "refl.mtz", hkls, observed, sigma, free)
    payload = config_payload(tmp_path, model, mtz, atomic_benchmark={"n_atoms": 3})
    if prior:
        payload["prior"].update(prior)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    cfg = AppConfig.model_validate(resolve_payload_paths(dict(payload), tmp_path))
    prepare_reflections(cfg)
    make_model_density(cfg)
    make_solvent_mask(cfg)
    fit_baseline_scaling(cfg)
    return cfg


def test_coordinate_occupancy_and_b_perturbations_are_all_measured(tmp_path):
    result = run_atomic_benchmark(_dataset(tmp_path))
    assert set(result["perturbation_kinds"]) == {"x", "b_iso", "occupancy"}
    assert result["n_atoms"] == 3
    for row in result["rows"]:
        assert np.isfinite(row["latent_rms"]) and row["latent_rms"] > 0
        assert row["delta_rms"] > 0, "the perturbation must actually change the density"
        assert 0.0 < row["bandlimited_fraction"] <= 1.0


def test_anisotropic_adp_perturbations_are_measured_when_present(tmp_path):
    result = run_atomic_benchmark(_dataset(tmp_path, ANISOU_PDB))
    assert "adp_aniso" in result["perturbation_kinds"]
    adp_rows = [row for row in result["rows"] if row["kind"] == "adp_aniso"]
    assert len(adp_rows) == 3
    assert all(row["delta_rms"] > 0 for row in adp_rows)


def test_the_benchmark_leaves_the_model_untouched(tmp_path):
    """Every perturbation must be undone, or later stages fit a different model."""
    cfg = _dataset(tmp_path, ANISOU_PDB)
    before = gemmi.read_structure(str(cfg.input.model))
    run_atomic_benchmark(cfg)
    after = gemmi.read_structure(str(cfg.input.model))
    for a, b in zip(before[0].all(), after[0].all(), strict=True):
        assert (a.atom.pos.x, a.atom.pos.y, a.atom.pos.z) == (b.atom.pos.x, b.atom.pos.y, b.atom.pos.z)
        assert a.atom.b_iso == b.atom.b_iso
        assert a.atom.occ == b.atom.occ


def test_a_narrower_prior_charges_more_for_the_same_perturbation(tmp_path):
    """The prior's cost for elementary refinement directions is what is being screened."""
    broad = run_atomic_benchmark(_dataset(tmp_path / "broad", prior={"correlation_length_angstrom": 0.4, "alpha": 2.0}))
    narrow = run_atomic_benchmark(
        _dataset(tmp_path / "narrow", prior={"correlation_length_angstrom": 3.0, "alpha": 3.0})
    )
    assert narrow["summary"]["max_latent_rms"] > broad["summary"]["max_latent_rms"]


def test_the_result_is_written_for_the_record(tmp_path):
    cfg = _dataset(tmp_path)
    run_atomic_benchmark(cfg)
    assert (cfg.run.output_dir / "atomic_benchmark.json").exists()

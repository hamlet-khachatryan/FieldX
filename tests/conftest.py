"""Synthetic crystallographic fixtures.

Every test builds its own tiny dataset. Nothing in the test suite depends on a
downloaded reflection file or on a particular PDB entry, so the generic pipeline is
exercised without a dataset-specific path.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

TINY_PDB = """CRYST1   20.000   24.000   28.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A   1       4.000   5.000   6.000  1.00 12.00           N
ATOM      2  CA  ALA A   1       5.200   5.400   6.300  1.00 11.00           C
ATOM      3  C   ALA A   1       6.100   4.300   6.900  1.00 13.00           C
ATOM      4  O   ALA A   1       7.300   4.500   7.100  1.00 15.00           O
ATOM      5  CB  ALA A   1       5.000   6.700   7.100  1.00 14.00           C
ATOM      6  N   GLY A   2      11.000  12.000  13.000  1.00 16.00           N
ATOM      7  CA  GLY A   2      12.200  12.400  13.300  1.00 15.00           C
ATOM      8  C   GLY A   2      13.100  11.300  13.900  1.00 17.00           C
ATOM      9  O   GLY A   2      14.300  11.500  14.100  1.00 18.00           O
ATOM     10  S   MET A   3      15.000  18.000  20.000  1.00 20.00           S
END
"""

CELL = (20.0, 24.0, 28.0, 90.0, 90.0, 90.0)
SPACEGROUP = "P 1"
D_MIN = 2.6

# A space group whose symmetry operations are not the identity. In `P 1` every
# symmetrization is a no-op, which makes it impossible to tell from a passing test whether
# the production code symmetrized at all. TINY_PDB and SPACEGROUP are pinned by dozens of
# tests and must not change, so this is a companion rather than a replacement.
SYMMETRIC_SPACEGROUP = "P 21 21 21"


def _cryst1(spacegroup, z_value):
    a, b, c, alpha, beta, gamma = CELL
    return f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} {spacegroup:<11}{z_value:>4}"


SYMMETRIC_PDB = "\n".join([_cryst1(SYMMETRIC_SPACEGROUP, 4), *TINY_PDB.splitlines()[1:]]) + "\n"


def write_tiny_model(path):
    path.write_text(TINY_PDB)
    return path


def write_symmetric_model(path):
    """The same atoms as `write_tiny_model`, declared in P 21 21 21."""
    path.write_text(SYMMETRIC_PDB)
    return path


def synthetic_reflections(model_path, d_min=D_MIN, free_fraction=0.08, seed=0):
    """Structure factors from the tiny model, with sigmas, noise and free flags."""
    import gemmi

    st = gemmi.read_structure(str(model_path))
    st.setup_entities()
    calculator = gemmi.StructureFactorCalculatorX(st.cell)
    rng = np.random.default_rng(seed)

    hkls, amplitudes = [], []
    limit = 12
    for h in range(-limit, limit + 1):
        for k in range(-limit, limit + 1):
            for m in range(-limit, limit + 1):
                if (h, k, m) == (0, 0, 0):
                    continue
                if st.cell.calculate_d((h, k, m)) < d_min:
                    continue
                hkls.append((h, k, m))
                amplitudes.append(abs(calculator.calculate_sf_from_model(st[0], (h, k, m))))

    hkls = np.asarray(hkls, dtype=np.int32)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    sigma = np.maximum(0.03 * amplitudes, 0.05 * amplitudes.mean())
    observed = np.abs(amplitudes + sigma * rng.standard_normal(amplitudes.shape))
    free = (rng.random(len(hkls)) < free_fraction).astype(np.int32)
    # Guarantee both classes exist regardless of the draw.
    free[0], free[1] = 1, 0
    return hkls, observed, sigma, free


def write_mtz(path, hkls, observed, sigma, free, cell=CELL, spacegroup=SPACEGROUP):
    import reciprocalspaceship as rs

    ds = rs.DataSet(
        {
            "H": hkls[:, 0].astype(np.int32),
            "K": hkls[:, 1].astype(np.int32),
            "L": hkls[:, 2].astype(np.int32),
            "FP": observed.astype(np.float32),
            "SIGFP": sigma.astype(np.float32),
            "FREE": free.astype(np.int32),
        },
        cell=cell,
        spacegroup=spacegroup,
    )
    ds = ds.set_index(["H", "K", "L"]).infer_mtz_dtypes()
    ds.merged = True
    ds.write_mtz(str(path))
    return path


def config_payload(tmp_path, model, reflections, **overrides):
    payload = {
        "run": {"output_dir": str(tmp_path / "run"), "shared_data_dir": str(tmp_path / "shared"), "seed": 7},
        "input": {
            "reflections": str(reflections),
            "model": str(model),
            "scattering": "xray",
            "observation_kind": "amplitude",
            "columns": {"observation": "FP", "sigma": "SIGFP", "free": "FREE"},
            "free_test_value": 1,
            "expected_free_fraction_min": 0.02,
            "expected_free_fraction_max": 0.20,
        },
        "split": {"strategy": "existing_free_then_hash", "tune_fraction_of_work": 0.2},
        "resolution": {"d_min_angstrom": D_MIN, "d_max_angstrom": None},
        "grid": {"shape": None, "samples_per_dmin": 3.0},
        "prior": {"kernel": "matern", "tau_density": 0.05, "correlation_length_angstrom": 0.8, "alpha": 2.5},
        "optimizer": {"max_iterations": 3, "checkpoint_every": 1},
        "information": {"n_modes": 2},
        "atomic_benchmark": {"n_atoms": 2},
    }
    for section, values in overrides.items():
        payload.setdefault(section, {}).update(values)
    return payload


def build_dataset(tmp_path, write_model=write_tiny_model, spacegroup=SPACEGROUP):
    """A complete synthetic dataset: model, MTZ and a validated configuration."""
    from crystal_field.config import AppConfig, resolve_payload_paths

    model = write_model(tmp_path / "tiny.pdb")
    hkls, observed, sigma, free = synthetic_reflections(model)
    mtz = write_mtz(tmp_path / "tiny.mtz", hkls, observed, sigma, free, spacegroup=spacegroup)
    payload = config_payload(tmp_path, model, mtz)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    cfg = AppConfig.model_validate(resolve_payload_paths(dict(payload), tmp_path))
    return {
        "cfg": cfg,
        "config_path": config_path,
        "payload": payload,
        "model": model,
        "reflections": mtz,
        "hkls": hkls,
        "tmp_path": tmp_path,
    }


def prepare_dataset(dataset):
    """A dataset from `build_dataset` after prepare + rho0 + solvent mask + train scaling."""
    from crystal_field.crystallography.io import make_model_density, make_solvent_mask, prepare_reflections
    from crystal_field.crystallography.scaling import fit_baseline_scaling

    cfg = dataset["cfg"]
    dataset["metadata"] = prepare_reflections(cfg)
    dataset["rho0_stats"] = make_model_density(cfg)
    dataset["mask_stats"] = make_solvent_mask(cfg)
    dataset["scaling"] = fit_baseline_scaling(cfg)
    return dataset


@pytest.fixture
def tiny_dataset(tmp_path):
    """A complete synthetic dataset: model, MTZ and a validated configuration."""
    return build_dataset(tmp_path)


@pytest.fixture
def prepared_dataset(tiny_dataset):
    """`tiny_dataset` after prepare + rho0 + solvent mask + train scaling."""
    return prepare_dataset(tiny_dataset)


@pytest.fixture
def symmetric_prepared_dataset(tmp_path):
    """`prepared_dataset`'s equivalent in P 21 21 21, where symmetrization does something."""
    return prepare_dataset(build_dataset(tmp_path, write_symmetric_model, SYMMETRIC_SPACEGROUP))


@pytest.fixture
def jax_x64():
    """Run a test in double precision.

    Some numerical identities (grid independence of the prior, exact spectral mixing)
    are checked to a tolerance that float32 cannot reach. JAX's x64 mode is global, so
    it is toggled per test rather than for the whole session.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", False)


def problem_arrays(observation=(1.0, 2.0, 3.0), split=(0, 1, 2), shape=(6, 6, 6), volume=1000.0, d_min=2.0):
    """A minimal in-memory ProblemArrays, with no file I/O."""
    import jax.numpy as jnp

    from crystal_field.inference.problem import ProblemArrays

    n = len(observation)
    hkls = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]][:n], dtype=np.int32)
    rng = np.random.default_rng(11)
    return ProblemArrays(
        rho0=jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=jnp.float32),
        solvent_mask=jnp.zeros(shape, dtype=jnp.float32),
        hkls=jnp.asarray(hkls, dtype=jnp.int32),
        observation=jnp.asarray(observation, dtype=jnp.float32),
        sigma=jnp.ones(n, dtype=jnp.float32),
        split=jnp.asarray(split, dtype=jnp.int8),
        reciprocal_metric=jnp.eye(3, dtype=jnp.float32) / 100.0,
        symmetry_rotations=jnp.eye(3, dtype=jnp.int32)[None, ...],
        symmetry_translations=jnp.zeros((1, 3), dtype=jnp.float32),
        overall_scale=jnp.ones(n, dtype=jnp.float32),
        solvent_scale=jnp.zeros(n, dtype=jnp.float32),
        unit_cell_volume=volume,
        d_min_angstrom=d_min,
        observation_kind="amplitude",
    )


def problem_config(tmp_path, **overrides):
    """A schema-valid configuration for in-memory problems; its paths are never read."""
    from crystal_field.config import AppConfig

    payload = {
        "run": {"output_dir": str(tmp_path / "run"), "shared_data_dir": str(tmp_path / "shared"), "seed": 5},
        "input": {
            "reflections": str(tmp_path / "unused.mtz"),
            "model": str(tmp_path / "unused.pdb"),
            "columns": {"observation": "FP", "sigma": "SIGFP", "free": "FREE"},
            "free_test_value": 1,
        },
        "split": {"strategy": "hash"},
        "resolution": {"d_min_angstrom": 2.0},
        "prior": {"kernel": "matern", "tau_density": 0.05, "correlation_length_angstrom": 1.0, "alpha": 2.5},
        "baseline": {"bulk_solvent": {"enabled": False}, "scaling": {"enabled": False}},
        "likelihood": {"global_scale": "fixed"},
    }
    for section, values in overrides.items():
        payload.setdefault(section, {}).update(values)
    return AppConfig.model_validate(payload)


def write_mmcif(pdb_path, out_path):
    """Convert the tiny PDB model to mmCIF, the format RCSB actually serves."""
    import gemmi

    st = gemmi.read_structure(str(pdb_path))
    st.setup_entities()
    out_path.write_text(st.make_mmcif_document().as_string())
    return out_path


def write_sf_cif(mtz_path, out_path):
    """Convert an MTZ to a deposited-style structure-factor mmCIF."""
    import gemmi

    converter = gemmi.MtzToCif()
    converter.spec_lines = [
        "H H index_h",
        "K H index_k",
        "L H index_l",
        "FP F F_meas_au",
        "SIGFP Q F_meas_sigma_au",
        "FREE I status S",
    ]
    out_path.write_text(converter.write_cif_to_string(gemmi.read_mtz_file(str(mtz_path))))
    return out_path


def write_multi_dataset_mtz(path, datasets, cell=CELL, spacegroup=SPACEGROUP, seed=0):
    """An MTZ carrying several datasets, as MAD/SAD files do.

    `datasets` is a list of (dataset_name, wavelength, {label: type}). Labels may repeat
    across datasets -- that reuse is exactly what makes implicit selection unsafe.
    Returns (path, {(dataset_name, label): first_value}) so tests can assert which
    dataset's numbers came back.
    """
    import gemmi

    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.SpaceGroup(spacegroup)
    mtz.set_cell_for_all(gemmi.UnitCell(*cell))
    hkl = np.array([[h, k, m] for h in range(1, 6) for k in range(1, 6) for m in range(1, 6)], dtype=float)
    rng = np.random.default_rng(seed)

    columns, truth = [], {}
    for name, wavelength, labels in datasets:
        dataset = mtz.add_dataset(name)
        dataset.wavelength = wavelength
        for label, mtz_type in labels.items():
            mtz.add_column(label, mtz_type, dataset_id=dataset.id)
            values = rng.random(len(hkl)) * 100 if mtz_type != "I" else (rng.random(len(hkl)) < 0.1).astype(float)
            columns.append(values)
            truth[(name, label)] = float(values[0])
    mtz.set_data(np.hstack([hkl, np.column_stack(columns)]))
    mtz.write_to_file(str(path))
    return path, truth

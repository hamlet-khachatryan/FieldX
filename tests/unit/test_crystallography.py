"""Crystallographic primitives (plan section 20C)."""

import gemmi
import numpy as np
import pytest

from crystal_field.crystallography.density import (
    direct_structure_factors,
    model_density_on_grid,
    model_electron_count,
)
from crystal_field.crystallography.geometry import (
    cell_tuple,
)
from crystal_field.crystallography.symmetry import symmetry_arrays

# Every crystal system and every lattice centring (P, A, C, I, F, R), plus both
# rhombohedral settings. Symmetry bugs hide in the centred and trigonal groups.
SPACE_GROUPS = [
    "P 1",
    "P -1",  # triclinic
    "P 21",
    "C 2",
    "C 2/c",  # monoclinic
    "P 21 21 21",
    "C 2 2 21",
    "I 2 2 2",
    "F 2 2 2",  # orthorhombic
    "P 41",
    "P 43 21 2",
    "I 4",
    "I 41/a",  # tetragonal
    "P 31",
    "R 3 :H",
    "R 3 :R",
    "R 32 :H",  # trigonal / rhombohedral
    "P 63",
    "P 63 2 2",  # hexagonal
    "P 21 3",
    "I 2 3",
    "F 4 3 2",
    "F m -3 m",  # cubic
]


@pytest.mark.parametrize("name", SPACE_GROUPS)
def test_rotational_and_full_operator_counts_agree(name):
    """symmetry_arrays() returns rotational operators; the full group adds centring."""
    from crystal_field.crystallography.symmetry import centring_translations, group_order

    sg = gemmi.SpaceGroup(name)
    rotations, translations = symmetry_arrays(sg)
    full_rotations, _ = symmetry_arrays(sg, include_centring=True)
    n_cen = len(centring_translations(sg))

    assert rotations.shape == (len(list(sg.operations().sym_ops)), 3, 3)
    assert translations.shape == (rotations.shape[0], 3)
    assert full_rotations.shape[0] == group_order(sg) == rotations.shape[0] * n_cen


@pytest.mark.parametrize("name", SPACE_GROUPS)
def test_operators_are_integral_orthogonal_and_in_the_unit_cell(name):
    rotations, translations = symmetry_arrays(gemmi.SpaceGroup(name), include_centring=True)
    assert rotations.dtype == np.int32
    for r in rotations:
        assert abs(round(float(np.linalg.det(r)))) == 1
    assert np.all((translations >= 0.0) & (translations < 1.0))


@pytest.mark.parametrize("name", SPACE_GROUPS)
def test_first_operator_is_the_identity(name):
    rotations, translations = symmetry_arrays(gemmi.SpaceGroup(name))
    np.testing.assert_array_equal(rotations[0], np.eye(3, dtype=np.int32))
    np.testing.assert_allclose(translations[0], np.zeros(3), atol=1e-12)


@pytest.mark.parametrize("name", SPACE_GROUPS)
def test_centring_factor_is_one_on_every_reflection_the_filter_keeps(name):
    """The precondition that lets the projection drop centring operators.

    (1/n_cen) sum_j exp(-2 pi i h.c_j) is always 0 or 1. It is 0 exactly for reflections
    forbidden by the *lattice* centring -- not for every systematic absence, since glide
    planes and screw axes forbid reflections through the rotational operators instead
    (C 2/c and I 41/a have both kinds). What matters here is the converse: any reflection
    that survives systematic_absences() has centring factor exactly 1, so on the filtered
    set the centring operators contribute nothing.
    """
    from crystal_field.crystallography.symmetry import centring_translations

    sg = gemmi.SpaceGroup(name)
    centring = centring_translations(sg)
    rng = np.random.default_rng(2)
    hkls = rng.integers(-6, 7, size=(400, 3)).astype(np.int32)
    factor = np.exp(-2j * np.pi * (hkls.astype(np.float64) @ centring.T)).mean(axis=1)

    magnitude = np.abs(factor)
    assert np.all(np.isclose(magnitude, 0.0, atol=1e-12) | np.isclose(magnitude, 1.0, atol=1e-12))

    kept = ~sg.operations().systematic_absences(hkls)
    np.testing.assert_allclose(factor[kept], 1.0, atol=1e-12)


@pytest.mark.parametrize("name", SPACE_GROUPS)
def test_projection_without_centring_matches_the_full_group(name):
    """The exactness claim, checked against the full-group projection itself."""
    jnp = pytest.importorskip("jax.numpy")

    from crystal_field.crystallography.symmetry import centring_translations
    from crystal_field.forward.diffraction import symmetry_projected_fcalc

    sg = gemmi.SpaceGroup(name)
    rng = np.random.default_rng(5)
    shape = (24, 24, 24)
    fgrid = jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=jnp.complex64)
    hkls = rng.integers(-6, 7, size=(400, 3)).astype(np.int32)

    def project(indices, include_centring):
        rotations, translations = symmetry_arrays(sg, include_centring=include_centring)
        return np.asarray(
            symmetry_projected_fcalc(
                fgrid,
                jnp.asarray(indices, dtype=jnp.int32),
                jnp.asarray(rotations, dtype=jnp.int32),
                jnp.asarray(translations, dtype=jnp.float32),
            )
        )

    kept = hkls[~sg.operations().systematic_absences(hkls)]
    assert len(kept) > 0
    np.testing.assert_allclose(project(kept, False), project(kept, True), rtol=2e-5, atol=2e-5)

    # On lattice-forbidden reflections the full group cancels to zero while the reduced
    # projection does not -- which is exactly why the absence filter is a precondition.
    centring = centring_translations(sg)
    if len(centring) > 1:
        factor = np.exp(-2j * np.pi * (hkls.astype(np.float64) @ centring.T)).mean(axis=1)
        forbidden = hkls[np.abs(factor) < 0.5]
        assert len(forbidden) > 0
        np.testing.assert_allclose(project(forbidden, True), 0.0, atol=2e-5)
        assert np.linalg.norm(project(forbidden, False)) > 1e-2, "the reduced projection needs the absence filter"


@pytest.mark.parametrize(
    ("name", "cell", "shape", "ok"),
    [
        ("P 41", (30, 30, 38, 90, 90, 90), (48, 48, 60), True),
        ("P 41", (30, 30, 38, 90, 90, 90), (48, 54, 60), False),
        ("P 63", (30, 30, 38, 90, 90, 120), (48, 48, 60), True),
        ("P 63", (30, 30, 38, 90, 90, 120), (48, 54, 60), False),
        ("P 21 21 21", (30, 34, 38, 90, 90, 90), (48, 54, 60), True),
        ("F m -3 m", (40, 40, 40, 90, 90, 90), (48, 48, 48), True),
        ("F m -3 m", (40, 40, 40, 90, 90, 90), (48, 54, 60), False),
    ],
)
def test_grid_shape_is_validated_against_symmetry(name, cell, shape, ok):
    """Gemmi requires equal sampling along symmetry-related axes; catch it in config."""
    from crystal_field.crystallography.symmetry import validate_grid_shape

    sg = gemmi.SpaceGroup(name)
    unit_cell = gemmi.UnitCell(*cell)
    if ok:
        assert validate_grid_shape(shape, unit_cell, sg) == shape
    else:
        with pytest.raises(ValueError, match="incompatible with space group"):
            validate_grid_shape(shape, unit_cell, sg)


def test_systematic_absences_are_detected():
    """P 21 21 21 extinguishes odd h00, 0k0 and 00l."""
    operations = gemmi.SpaceGroup("P 21 21 21").operations()
    hkls = np.array([[1, 0, 0], [2, 0, 0], [0, 3, 0], [0, 4, 0], [0, 0, 5], [1, 1, 1]], dtype=np.int32)
    absent = operations.systematic_absences(hkls)
    assert list(absent) == [True, False, True, False, True, False]


def test_model_density_honours_the_declared_grid(tmp_path):
    """put_model_density_on_grid() silently re-derives the grid size; we must not.

    rho0, the solvent mask and the Nyquist guard all have to live on one grid, or the
    same (h,k,l) gathers a different Fourier bin from each.
    """
    from conftest import write_tiny_model

    st = gemmi.read_structure(str(write_tiny_model(tmp_path / "tiny.pdb")))
    st.setup_entities()
    sg = gemmi.SpaceGroup("P 1")
    for shape in [(24, 30, 36), (40, 48, 56), (20, 24, 28)]:
        rho, _ = model_density_on_grid(st[0], st.cell, sg, shape, 2.6, 1e-6)
        assert rho.shape == shape
        assert float(rho.max()) > 0.0


def test_integrated_density_equals_the_model_electron_count(tmp_path):
    from conftest import write_tiny_model

    st = gemmi.read_structure(str(write_tiny_model(tmp_path / "tiny.pdb")))
    st.setup_entities()
    sg = gemmi.SpaceGroup("P 1")
    shape = (40, 48, 56)
    rho, _ = model_density_on_grid(st[0], st.cell, sg, shape, 1.5, 1e-7)
    integrated = float(rho.sum()) * st.cell.volume / rho.size
    assert integrated == pytest.approx(model_electron_count(st[0], sg), rel=0.01)


SYMMETRIC_PDB = """CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} {sg:<11}{z:4d}
ATOM      1  N   ALA A   1       4.000   5.000   6.000  1.00 12.00           N
ATOM      2  CA  ALA A   1       5.200   5.400   6.300  1.00 11.00           C
ATOM      3  C   ALA A   1       6.100   4.300   6.900  1.00 13.00           C
ATOM      4  O   ALA A   1       7.300   4.500   7.100  1.00 15.00           O
ATOM      5  S   MET A   2      11.000  12.000  13.000  1.00 20.00           S
END
"""

HKLS = np.array([[1, 0, 0], [2, 1, 0], [0, 2, 3], [3, -2, 1], [-1, 4, 2], [5, 3, 2]], dtype=np.int32)

# (space group, cell, FFT shape) -- each cell and grid is metrically compatible with its
# space group; tetragonal and hexagonal groups need a == b and matching grid dimensions.
SYMMETRIC_CASES = [
    ("P 1", (30.0, 34.0, 38.0, 90.0, 90.0, 90.0), (48, 54, 60)),
    ("P 21 21 21", (30.0, 34.0, 38.0, 90.0, 90.0, 90.0), (48, 54, 60)),
    ("C 2", (30.0, 34.0, 38.0, 90.0, 100.0, 90.0), (48, 54, 60)),
    ("P 41", (30.0, 30.0, 38.0, 90.0, 90.0, 90.0), (48, 48, 60)),
    ("P 63", (30.0, 30.0, 38.0, 90.0, 90.0, 120.0), (48, 48, 60)),
]


def _symmetric_model(tmp_path, sg_name, cell_params):
    sg = gemmi.SpaceGroup(sg_name)
    a, b, c, al, be, ga = cell_params
    path = tmp_path / "sym.pdb"
    path.write_text(SYMMETRIC_PDB.format(a=a, b=b, c=c, al=al, be=be, ga=ga, sg=sg_name, z=len(sg.operations())))
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    # Rebuilt from parameters, exactly as _model_for_metadata() rebuilds it from
    # metadata.json -- such a cell carries no symmetry images until it is set up.
    return st, sg, gemmi.UnitCell(*cell_params)


def _grid_amplitudes(rho, volume, hkls):
    # jnp.fft.fftn uses exp(-2 pi i h.r); Gemmi is exp(+2 pi i h.r), hence the conjugate.
    fgrid = np.fft.fftn(rho.astype(np.float64)) * (volume / rho.size)
    idx = np.mod(hkls, np.asarray(rho.shape))
    return np.abs(np.conj(fgrid[idx[:, 0], idx[:, 1], idx[:, 2]]))


@pytest.mark.parametrize(("sg_name", "cell_params", "shape"), SYMMETRIC_CASES, ids=[c[0] for c in SYMMETRIC_CASES])
def test_fft_of_rho0_matches_direct_summation(tmp_path, sg_name, cell_params, shape):
    """An FFT-free reference for the starting density, in symmetric space groups too.

    Gemmi applies symmetry through `UnitCell.images`, and a cell rebuilt from parameters
    has none. Without setting them up the direct sum silently covers only the asymmetric
    unit while rho0 covers the whole cell. In P1 the two coincide, so only a symmetric
    space group catches it -- which is why every case below is parametrized.
    """
    st, sg, cell = _symmetric_model(tmp_path, sg_name, cell_params)
    assert len(cell.images) == 0, "a cell built from parameters must start with no images"

    rho, _ = model_density_on_grid(st[0], cell, sg, shape, 2.0, 1e-7)
    direct = direct_structure_factors(st[0], cell, sg, HKLS, 2.0)
    measured = _grid_amplitudes(rho, cell.volume, HKLS)
    reference = np.abs(direct)
    relative = np.linalg.norm(measured - reference) / max(np.linalg.norm(reference), 1e-30)
    assert relative < 0.05, f"{sg_name}: relative amplitude error {relative:.1%}"


@pytest.mark.parametrize(("sg_name", "cell_params", "shape"), SYMMETRIC_CASES, ids=[c[0] for c in SYMMETRIC_CASES])
def test_direct_summation_covers_every_symmetry_copy(tmp_path, sg_name, cell_params, shape):
    """F(000) is the whole cell's electron count: ASU electrons times the group order."""
    st, sg, cell = _symmetric_model(tmp_path, sg_name, cell_params)
    per_asu = model_electron_count(st[0], gemmi.SpaceGroup("P 1"))
    f000 = direct_structure_factors(st[0], cell, sg, np.array([[0, 0, 0]], dtype=np.int32), 2.0)[0]
    assert float(f000.real) == pytest.approx(per_asu * len(sg.operations()), rel=0.02)


def test_grid_shape_follows_cell_dmin_and_sampling(tiny_dataset):
    """The computational grid is derived, never assumed."""
    from crystal_field.crystallography.io import choose_grid_shape, load_reflections

    cfg = tiny_dataset["cfg"]
    ds = load_reflections(cfg)
    coarse = choose_grid_shape(
        ds, cfg.model_copy(update={"grid": cfg.grid.model_copy(update={"samples_per_dmin": 2.0})})
    )
    fine = choose_grid_shape(ds, cfg.model_copy(update={"grid": cfg.grid.model_copy(update={"samples_per_dmin": 4.0})}))
    assert all(f > c for f, c in zip(fine, coarse, strict=True))
    # Real-space sampling must be at least d_min / samples_per_dmin along every axis.
    for length, n in zip(cell_tuple(ds.cell)[:3], coarse, strict=True):
        assert length / n <= cfg.resolution.d_min_angstrom / 2.0 + 1e-9


def test_explicit_grid_shape_is_honoured(tiny_dataset):
    from crystal_field.crystallography.io import choose_grid_shape, load_reflections

    cfg = tiny_dataset["cfg"]
    cfg = cfg.model_copy(update={"grid": cfg.grid.model_copy(update={"shape": (32, 32, 32)})})
    assert choose_grid_shape(load_reflections(cfg), cfg) == (32, 32, 32)


def test_nyquist_guard_rejects_a_grid_too_small_for_the_reflections(tiny_dataset):
    """A grid that cannot hold the retained HKLs aliases them; prepare must refuse."""
    from crystal_field.crystallography.io import prepare_reflections

    cfg = tiny_dataset["cfg"]
    cfg = cfg.model_copy(update={"grid": cfg.grid.model_copy(update={"shape": (8, 8, 8)})})
    with pytest.raises(ValueError, match="too small for retained HKLs"):
        prepare_reflections(cfg)

"""Construction of the starting atomistic density rho0.

rho0 comes from the refined atomic model through Gemmi's crystallographic density
calculator. It is the physical baseline of the v3 field

    rho = rho0 + delta_rho,

so it must be an unblurred physical density on exactly the declared FFT grid --
not a Refmac-style blurred density, whose Fourier normalization would have to be
undone before the correction field meant anything.
"""

from __future__ import annotations

import gemmi
import numpy as np

DENSITY_CALCULATORS = {
    "xray": gemmi.DensityCalculatorX,
    "electron": gemmi.DensityCalculatorE,
    "neutron": gemmi.DensityCalculatorN,
}

STRUCTURE_FACTOR_CALCULATORS = {
    "xray": gemmi.StructureFactorCalculatorX,
    "electron": gemmi.StructureFactorCalculatorE,
    "neutron": gemmi.StructureFactorCalculatorN,
}


def model_density_on_grid(model, cell, spacegroup, shape, d_min, cutoff, scattering="xray"):
    """Put the atomic model density on exactly `shape`, symmetrized over the space group.

    put_model_density_on_grid() calls initialize_grid() internally, which re-derives the
    grid size from d_min and dc.rate and silently discards any earlier set_size(). That
    would place rho0 on a different grid than the declared shape and the solvent mask, so
    the same (h,k,l) would gather a different Fourier bin from each. Gemmi's own sequence
    is expanded here instead, with the size overridden in the middle of it; set_size()
    zero-fills, so this is the identical computation carried out on the declared grid.
    """
    shape = tuple(int(n) for n in shape)
    dc = DENSITY_CALCULATORS[scattering]()
    dc.d_min = float(d_min)
    dc.cutoff = float(cutoff)
    dc.grid.spacegroup = spacegroup
    dc.grid.set_unit_cell(cell)
    dc.initialize_grid()
    dc.grid.set_size(*shape)
    dc.add_model_density_to_grid(model)
    dc.grid.symmetrize_sum()
    rho = np.array(dc.grid.array, dtype=np.float32, copy=True, order="C")
    if rho.shape != shape:
        raise RuntimeError(f"Gemmi produced a {rho.shape} density grid but {shape} was requested")
    return rho, dc


def model_electron_count(model, spacegroup) -> float:
    """Total scattering electrons in the whole unit cell, summed over symmetry copies."""
    per_asu = sum(cra.atom.occ * cra.atom.element.atomic_number for cra in model.all())
    return float(per_asu * len(spacegroup.operations()))


def direct_structure_factors(model, cell, spacegroup, hkls, d_min, scattering="xray"):
    """Structure factors by direct summation over atoms -- independent of any FFT grid.

    Gemmi applies crystallographic symmetry through `UnitCell.images`, which a cell
    constructed as `gemmi.UnitCell(a, b, c, ...)` does NOT carry: it has zero images, so
    the sum would cover the asymmetric unit only while rho0 covers the whole cell. In P1
    those agree, which is why it has to be set up explicitly rather than assumed.
    setup_cell_images() populates them from the space group.
    """
    st = gemmi.Structure()
    st.cell = cell
    st.spacegroup_hm = spacegroup.xhm()
    st.add_model(model)
    st.setup_entities()
    st.setup_cell_images()
    if len(st.cell.images) + 1 != len(spacegroup.operations()):
        raise RuntimeError(
            f"Expected {len(spacegroup.operations())} symmetry operations for {spacegroup.xhm()}, "
            f"but the unit cell carries {len(st.cell.images) + 1}. Structure-factor summation "
            "would cover the wrong number of symmetry copies."
        )
    calculator = STRUCTURE_FACTOR_CALCULATORS[scattering](st.cell)
    calculator.addends.clear()
    return np.array(
        [calculator.calculate_sf_from_model(st[0], (int(h), int(k), int(l_))) for h, k, l_ in hkls],
        dtype=np.complex128,
    )

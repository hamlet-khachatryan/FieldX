# Tangent decomposition: known limitations and follow-ups

Recorded at the end of the `tangent-decomposition` branch. Every item here was found by
review, judged real, and deliberately not fixed in that branch. None is load-bearing for
the correctness of what the stage reports today on the synthetic fixtures; the ordering is
by what would bite first on real data.

**The stage has never been run on a real structure.** Everything below is reasoning about
code and cost, not a measurement from a real dataset.

## Before the first real run

**Float32 normal equations may overstate rank at scale.** `_data_supported_target` computes
column structure factors in `arrays.rho0.dtype`, which is float32 by default
(`run.enable_x64: false`), then forms float64 normal equations and solves with
`rcond=None` — a float64 tolerance of about 1e-13 relative. Because these are *normal*
equations the singular values are squared, so a basis direction at 1e-4 of the largest
appears at 1e-8 in the gram, below float32's noise floor of about 1e-7. `lstsq` can then
keep pure-noise directions, overstate `rank`, and drive the residual negative — where it is
clamped, reporting `explained_fraction == 1.0`.

This could not be demonstrated at synthetic scale: a run under float32 and one under
`enable_x64: true` agreed to eight digits (`data_supported.explained_fraction`
0.04684827651 vs 0.04684827809, `rank` 30, condition number 26.23 both ways). But 30
well-conditioned columns is not the ~1,980 strongly-overlapping columns of a 1UBQ run.

*Mitigation:* set `run.enable_x64: true` for this CPU stage, or cast the grid to float64
inside `structure_factors`. Cheap either way. **On the first real run, check `rank` and
`explained_fraction`: a rank equal to the column count, or a fraction of exactly 1.0, means
this happened.**

**Catastrophic cancellation is silently clamped.** `solve_normal_equations` computes the
residual in the expanded form `||t||^2 - 2 a.rhs + a.Ga`, which loses roughly
`log10(cond(gram))` digits; with a condition number plausibly 1e8-1e10 at 1UBQ scale, a
genuine explained fraction of 0.999 can be reported as exactly 1.0 rather than as a failure.
`residual = max(residual, 0.0)` and `np.clip(explained, 0, 1)` hide it.

For `decompose_target` the expansion is unnecessary — `unexplained` is already materialised,
so `1 - ||unexplained||^2/||target||^2` is exact and free. Keep the expansion only in
`_data_supported_target`, where the residual is not materialised.

**No memory guard.** The repository's own idiom (`InformationConfig.memory_budget_gib`:
"Refuse to launch a job whose estimate exceeds this rather than discovering it as an OOM
hours in") has no counterpart here. At 1UBQ the numbers are comfortable — with
`basis: full`, about 318 MB for the stacked structure factors with a transient peak near
640 MB during `np.stack`, plus an 87 MB gram, against `--mem=32G` — but `gram` grows as
`n_columns^2` and `stacked` as `n_reflections x n_columns`. A larger structure should fail
at submit time, not two hours in.

## Correctness in cases not yet exercised

**`_truncate`'s distance metric assumes an orthogonal cell.** It treats
`d_frac_i x cell.{a,b,c}` as Cartesian components and sums their squares, omitting the cross
terms for monoclinic and triclinic cells. Reachable only via `box_radius_angstrom`, and the
`MIN_NORM_FRACTION` guard would catch gross over-truncation, so the failure mode is a
wrongly-shaped kept region that still passes the guard. Use `cell.orthogonalize` on the
fractional offset.

**Column symmetry-invariance is not tested.** Spec section 9.2 asks for it; the existing
test only counts non-zeros. One line would close it:
`assert_allclose(symmetrize_grid(column, cell, sg), column)`.

## Documentation accuracy

**`PARAMETER_STEPS` hardcodes `atomic_benchmark`'s defaults, not its configuration.** The
comment claims the two diagnostics "perturb atoms identically", but
`AtomicBenchmarkConfig` exposes `coordinate_delta_angstrom`, `b_delta_angstrom2` and
`occupancy_delta`. Change any of them and the comment is silently false. Either read the
config or soften the comment to say *default* steps.

**The capacity control is matched in operator and norm, but not in spectrum.** It draws
white `z` through `L`, whereas the real `u = L z_fit` has a data-driven `z_fit` whose
spectral content is not white. This follows spec section 6.1 exactly and is not an
implementation defect, but "matched random fields" in `docs/METHODS.md` reads stronger than
what is computed. Worth one qualifying clause.

## Not a defect (recorded so it is not re-investigated)

The `PydanticSerializationUnexpectedValue` warning in the test suite comes from
`tests/unit/test_prior_grid.py` using `model_copy(update=...)`, which does not validate, so
raw dicts land in a field declared `list[PriorComponent]`. Production code never takes that
path (`grep -rn "model_copy" src/` returns nothing) and `model_dump()` output is identical
either way, so the YAML a user writes or reads is unaffected. Test-only and cosmetic.

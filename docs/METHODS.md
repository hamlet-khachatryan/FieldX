# Methods specification (v3)

## 1. Scientific target

The v3 target is not a lower work-set residual by itself. Success requires a simultaneous reduction in held-out and fitted-data disagreement after conventional atomistic refinement has already provided a strong starting model.

For the final locked model:

- `R_work(field) < R_work(baseline)`;
- `R_free(field) < R_free(baseline)`;
- the change in `R_free - R_work` is reported explicitly.

The free set is used for nothing: not optimization, not prior selection, not
hyperparameter selection, not early stopping, not grid selection, not nuisance
calibration, not model interpretation. `G = R_free - R_work` is a diagnostic, never
an optimization target.

v3 uses Bragg observations only. Diffuse scattering is a separate, later formulation.

## 2. Density parameterization

`rho = rho0 + L z`, with a real-valued, experimentally band-limited correction field. `rho0` is produced from the coordinate model using Gemmi density calculators, which honor atomic coordinates, occupancies and ADPs.

`rho0` is unblurred. A Refmac-style blurred density is not used as the physical baseline,
because the correction field would then be a correction to something whose Fourier
normalization has been altered. `check-rho0` verifies the generated density against the
crystallographic model in two independent ways: the integrated density must equal the
model's electron count, and the FFT of `rho0` must reproduce structure factors obtained
by direct summation over atoms.

The prior is deliberately generic. It expresses only that coherent multi-voxel density
changes are plausible while isolated single-voxel excursions are not; it does not require
a density feature to correspond to an atom, a coordinate shift, an ADP change or a
predefined physical mode. A meaningful feature may be unfamiliar and non-atomistic.

The field therefore includes density directions corresponding locally to
coordinate/occupancy/ADP perturbations, while admitting additional non-atomistic
directions. The atomic representability benchmark is a diagnostic on that first
property: a candidate prior that makes ordinary refinement directions excessively
improbable is not accepted merely because it improves the tuning residual. It is not a
restriction on the second.

## 3. Spectral kernels

All kernels are cut off at `1/d_min`. Their transfer functions are normalized so that for
white unit-variance latent coefficients the real-space RMS correction is `tau_density`.

Everything is expressed in physical reciprocal-space units: the spectral geometry uses
the reciprocal metric tensor and Miller indices, so `correlation_length_angstrom` and the
band limit mean the same thing on any FFT grid. Changing `grid.samples_per_dmin` does not
silently change the physical correlation length. This is asserted directly in
`tests/unit/test_priors.py::test_prior_is_grid_independent`, which also documents why a
grid too small to contain the band-limited sphere is rejected by the Nyquist guard.

Supported kernels:

- `matern`: `(kappa^2 + q^2)^(-alpha/2)`;
- `squared_exponential`: `exp(-ell^2 q^2 / 4)`;
- `bandlimited_white`: constant spectral amplitude inside the observed band;
- `multiscale_matern`: square root of a weighted sum of Matérn spectral powers.

Latent penalties:

- Gaussian: `0.5 sum(z^2)`;
- Student-t: heavy-tailed robust innovations;
- smoothed Laplace: sparse-innovation alternative;
- Cauchy: the heaviest tail offered.

The first experiment on a new dataset uses Gaussian latents only. The pilot grid is four
spatial candidates — Matern at three correlation lengths plus a squared-exponential
control — so the experiment tests the existing v3 formulation rather than confounding it
with additional prior flexibility.

## 4. Baseline solvent term

Gemmi `SolventMasker` generates a Refmac-compatible solvent mask by default. Its reciprocal contribution is

`F_sol = k_sol exp(-B_sol s^2/4) F_mask`.

The default `k_sol=0.35` and `B_sol=46 A^2` are conventional starting values. They are frozen within each candidate; they can later be included in the internal
work-set model comparison without touching free reflections.

## 5. Likelihood and scaling

The default likelihood is sigma-weighted Gaussian residuals in the configured observation domain. A Student-t alternative is included for robustness testing. A positive global scale is profiled using only the active fit scope.

Candidate fits use train reflections. The selected model is refit using all non-free reflections.

## 5b. Reflection data selection

An MTZ may carry several datasets that reuse column labels -- MAD/SAD wavelengths, a
native plus a derivative. The dataset is resolved explicitly: `input.mtz_dataset` names
it by id or by name, and resolution proceeds without it only when the choice is forced,
either because a single data-bearing dataset exists or because the configured labels
occur exactly once and in one dataset. Anything else is an error listing the candidates.

The chosen dataset id, dataset name and wavelength are written into the prepared
metadata, which `MODEL_LOCK.json` hashes, so a locked model records the data it was
refined against. Reflection CIFs carry a single dataset and are unaffected.

## 5c. Symmetry projection

The latent field is unconstrained, so the forward model projects F onto the
symmetry-invariant subspace by averaging over the space group. That average factorises
into a rotational part and a centring part, because Gemmi composes a centred group as
`(R_s, t_s + c_j)`:

    (1/|G|) sum_{s,j} exp(-2 pi i h.(t_s + c_j)) F(h R_s)
        = [ (1/n_sym) sum_s exp(-2 pi i h.t_s) F(h R_s) ] x [ (1/n_cen) sum_j exp(-2 pi i h.c_j) ]

The centring factor is exactly 1 for a reflection allowed by the lattice and exactly 0
for one it forbids. Since `prepare_reflections` removes every systematically absent
reflection, the retained set carries a centring factor of one, and the projection is
computed over the rotational operators alone. This is exact rather than approximate, but
it is conditional on the absence filter, so both forms are retained and their equivalence
is asserted across 22 space groups spanning all seven crystal systems and all centrings.

Note that centring is not the only source of absence: glide planes and screw axes forbid
reflections through the rotational operators themselves (`C 2/c`, `I 41/a` have both).
The property relied on is the converse -- any reflection surviving the filter has
centring factor one.

Measured effect on the projection, 20000 reflections on a 48^3 grid:

| group | operators | reduced | speed-up |
|---|---:|---:|---:|
| `F m -3 m` | 192 | 48 | 5.1x |
| `F 4 3 2` | 96 | 24 | 4.5x |
| `I 4` | 8 | 4 | 1.9x |
| `P 21 21 21` | 4 | 4 | 1.0x |

Primitive groups are unaffected; the electron-count check and the direct-summation
reference continue to use the full group order, which counts every copy in the cell.

## 5d. Capacity control: refinement from an empty cell

`baseline.starting_density: zero` replaces rho0 with an empty cell. This is not a phasing
method. With amplitudes alone the problem is phase-degenerate, and the run is not
expected to recover a structure.

Its purpose is calibration. The field carries far more degrees of freedom than there are
reflections -- for 1UBQ, 324000 voxels against 6029 reflections, roughly 54 parameters per
observation -- so only the prior stands between the model and arbitrary fitting. The
control measures how far R_work can be driven down with no structural information at all,
which is the number against which a model-start R_work must be read.

Three settings become mandatory and are enforced by the schema: `baseline.scaling` and
`baseline.bulk_solvent` are disabled (both are defined relative to a model density), and
`optimizer.init` must be `random`, because |F| is not differentiable at F = 0 and the
gradient at z = 0 is NaN.

Judge the control on the tune subset, with `fit_scope: train`. It should fit the training
amplitudes and fail to generalise; if it ever generalised, R_work would have lost its
evidential value entirely. Do not spend a free-set read on it.

## 5e. Declaring a prior sweep

A prior grid may enumerate `candidates:` explicitly, declare a `sweep:` over axes, or
both. A sweep takes the Cartesian product of

    kernel, tau_density, correlation_length_angstrom, alpha,
    latent_distribution, student_t_df, laplace_softening, cauchy_scale

and then collapses combinations that describe the same operator. alpha does not reach a
squared-exponential kernel, neither alpha nor correlation length reaches band-limited
white noise, and a latent shape parameter does not reach a distribution that does not use
it. Declaring 3 kernels x 2 tau x 3 correlation lengths x 2 alpha is 36 combinations but
only 20 distinct operators, and only those 20 are fitted.

`multiscale_matern` cannot be swept: it is defined by a component list rather than a
scalar axis, so it must be given under `candidates:`.

Every candidate is an independent GPU job, so an expansion above `max_candidates`
(default 64) is refused; raising it has to be written into the grid file.

Each candidate's effective parameters are recorded in the manifest and in the fitted
`metrics.json` under `prior`, so a result states what it depended on without reference to
its configuration file.

## 5f. Maps

Two families are written. Calculated maps -- the starting density, the inferred
correction, their sum -- show what the model is, and are smooth by construction.
Coefficient maps show what the data say: unweighted 2Fo-Fc and Fo-Fc, and sigma-A
weighted 2mFo-DFc and mFo-DFc, each with both FieldX and atomic-model phases so the two
can be compared like for like.

Sigma-A weighting follows Read (1986). Per resolution shell, D is the phase-free
least-squares scale <|Fo||Fc|>/<|Fc|^2> and the residual variance is
<|Fo|^2> - D^2<|Fc|^2>, floored positive; the figure of merit is I1(X)/I0(X) for acentric
reflections and tanh(X) for centric ones, with X carrying the epsilon factor. Centric
2mFo-DFc coefficients are m|Fo|, not 2m|Fo| - D|Fc|.

The estimation set is recorded. Conventionally sigma-A is estimated on the free set,
because a model fitted to the work set explains it optimistically; FieldX defaults to the
work set to avoid spending the held-out reflections on interpretation, and permits the
free set only after the one-shot evaluation has already consumed it.

The forward model uses exp(-2 pi i h.r) while Gemmi's map transform expects
exp(+2 pi i h.r), so coefficients are conjugated on the way out, and only the asymmetric
unit is passed to the transform since it expands by symmetry. Getting either wrong drops
the correlation between an Fc map and the density it came from from about 0.87 to 0.12,
which is what the map test asserts.

## 5g. Tangent decomposition

A read-only diagnostic testing the claim in section 1 directly: is the inferred density
reachable by ordinary refinement? The correction is projected onto the atomic tangent
space, spanned by the density derivatives with respect to atomic coordinates and,
optionally, B factors and occupancies.

Columns are built by central-differencing single-atom Gemmi densities. Density is additive
over atoms, so a single-atom structure yields the exact column, and Gemmi's density cutoff
makes each column sparse, which is what makes a few thousand columns affordable.

Two targets are reported: the symmetrized correction on the grid, and its data-supported
part on the measured work reflections. Only the symmetric component of the correction
reaches `F_calc`, and every column is symmetric, so the antisymmetric residue is reported
separately rather than counted as unexplained.

With thousands of free parameters the basis fits a substantial fraction of anything, so
an explained fraction is never reported alone: matched random fields drawn through the
same prior operator are decomposed onto the same basis and reported alongside. The
comparison, not the number, is the result.

Both targets get their own control, computed from the same draws. A control is a property
of the target as much as of the basis — the grid target and the work-set `Delta_F` target
have different floors — so neither fraction may be read against the other's control, and
the report keeps each next to its own.

## 6. R factors

For amplitude data the reported R value is

`sum |Fobs - k Fcalc| / sum |Fobs|`

over the requested split. For intensity likelihoods, diagnostic R values are reported in amplitude space using square roots of non-negative intensities.

## 7. Model selection

The deposited free flags define the final set when the entry provides usable ones. When
they do not — a constant or absent flag column, as in 1UBQ — the free set is a
deterministic Friedel-paired hash holdout. That set is genuinely untouched and a valid
cross-validation statistic, but it is not the deposited crystallographic R-free and every
report must say so; `evaluate-free` records which case applies in its `free_set_kind`
field.

A third case exists for completeness: `split.strategy: none` declares that no held-out
set is available. All retained reflections become work, split into train and tune so that
prior selection and early stopping still have something to rank on, and the sole target
is `R_work`. No stage errors in this mode, but no cross-validated claim can be made from
it either: with nothing held out, a reduction in `R_work` measures capacity rather than
information. The model lock and every evaluation artifact record `has_free_set: false`
and `target: R_work only`. The remaining work reflections are deterministically partitioned by a seeded HKL hash into train and tune subsets, with Friedel pairs kept together for non-anomalous data.

Candidate selection minimizes tune-set mean chi-square. Ties are broken by tune R. The winner is refit on the complete work set.

## 8. Lock and free evaluation

After final fitting, `freeze-model` hashes:

- selected YAML;
- prepared reflection arrays;
- metadata;
- starting density and solvent mask;
- work-set nuisance scaling;
- final latent field;
- final work metrics;
- the information spectrum.

Including the information spectrum means the pipeline cannot be short-circuited into
freezing a model that was never characterised.

`evaluate-free` verifies every hash, then evaluates both `z=0` (the atomic baseline under
identical nuisance scaling) and `z=z_map` on the same free reflections in one invocation.
It reports `R_work_atomic`, `R_work_field`, `R_free_atomic`, `R_free_field`, the two gaps
and the three differences, and optimizes nothing.

The evaluation is enforced as one-shot by an append-only ledger stored beside the
prepared reflections, so re-freezing a revised model into a fresh output directory does
not restore a consumed free set.

## 9. Numerical gates

Before fitting:

1. the starting-density check against the crystallographic model;
2. the Gemmi-to-JAX FFT convention check;
3. the finite-difference directional derivative check;
4. the JVP/VJP adjoint identity check;
5. the accelerator test suite;
6. the atomic representability benchmark.

Fitting is blocked if any of these fail; each raises and breaks the `afterok` chain.

The finite-difference step is chosen adaptively as `cbrt(u |f| / |f'|)`, where `u` is the
unit roundoff. The objective is a chi-squared sum over thousands of reflections and is
orders of magnitude larger than its directional derivative, so a fixed small step makes
the central difference pure cancellation noise in float32.

## 10. Information analysis

The leading modes of the work-set `J^T J` operator are computed matrix-free with JAX
LOBPCG: `J v` is a JVP and `J^T w` a VJP, and no dense Jacobian or `N_vox x N_vox` matrix
is ever formed. Reported effective-dimension and information values are lower bounds
because only leading modes are retained.

The solve is guarded by a memory estimate against `information.memory_budget_gib` before
LOBPCG allocates its `3k`-wide basis of full latent fields.

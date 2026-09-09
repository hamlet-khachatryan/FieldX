# Tangent decomposition: is the inferred density reachable by ordinary refinement?

Design specification. 2026-09-09.

**Status: design only. Nothing in this document is implemented.**

## 1. Purpose

FieldX v3 infers a correction `u = L·z` to the refined atomic density. The manuscript's
central claim is that experimentally supported density remains *outside* the local atomic
coordinate/occupancy/ADP model class. That claim is currently untested: a v3 run reports
that `R_free` improved, but not whether the density responsible could simply have been
found by moving atoms.

This specification defines an analysis that answers it directly. Given a completed v3
fit, decompose the inferred correction onto the atomic tangent space

```
Phi = span{ d(rho)/dx_i, d(rho)/dy_i, d(rho)/dz_i, d(rho)/dB_i, d(rho)/dq_i }
```

and report what fraction it explains, what fraction it does not, and -- critically --
whether that fraction exceeds what the same basis explains of pure noise.

The unexplained remainder is the object the project exists to find.

## 2. Relation to v3 and v4

`docs/V4_DESIGN.md` defines v4 as the formulation

```
rho(r) = rho0(r) + Phi a + L_res z
```

This specification covers two increments.

**Increment 1 -- tangent decomposition (implement now).** A read-only analysis. It loads a
completed v3 run, builds `Phi`, solves a least-squares problem and writes a report. It
changes nothing in the model, prior, likelihood, optimizer or forward calculation. The
formulation remains `rho = rho0 + L·z` throughout.

**Increment 2 -- joint refinement (specified, not implemented).** Optimization over
`(a, z)` with `Phi a` in the forward model. Specified here so increment 1's operator is
built with the right interface. Implemented only if increment 1 shows `Phi` explains
anything worth refining.

**Naming.** Increment 1 is *not* v4 and must not be called v4. It does not change the
formulation, and v4 is defined as a formulation. Calling a diagnostic v4 would blur
exactly the distinction the manuscript needs to keep sharp. Increment 1 ships as a v3
analysis stage named `decompose`; `v4` is reserved for increment 2 and would introduce a
`prior.modes` config namespace and a version bump. `docs/V4_DESIGN.md` gains a section
describing the decomposition as the diagnostic that gates v4.

## 3. What is decomposed

Two targets, reported side by side.

**Full correction.** `symmetrize(u)` on the computational grid, where `u = L·z` is
recovered from the fitted `z_map.npy` by applying the same transfer operator the fit used.

The symmetrization is not optional. `z` is unconstrained, so `u` has an antisymmetric
component, but only the symmetric part reaches `F_calc` through
`symmetry_projected_fcalc`. Every `Phi` column is a difference of two symmetrized
densities and is therefore symmetric, so the antisymmetric part of `u` is orthogonal to
`Phi` by construction and would otherwise appear as permanently unexplained density that
no basis could ever reach. The discarded antisymmetric fraction is reported: if it is
large, that is a fact about the fit worth knowing.

**Data-supported part.** `Delta_F(h) = F(rho0 + u)(h) - F(rho0)(h)` on the measured work
reflections, decomposed against the corresponding structure factors of the `Phi` columns.

The two differ substantially and the gap is informative. For 1UBQ the correction has
324,000 voxels, band-limited to roughly 45,000 non-zero Fourier coefficients, of which
6,029 correspond to measured reflections. Everything else is prior interpolation. The
difference between the two explained fractions says how much of what v3 "found" was
supported by data rather than filled in by the prior.

## 4. The Phi operator

### 4.1 Construction

Each column is a central difference of Gemmi densities:

```
d(rho)/d(theta_i)  ~  [ rho(theta_i + h) - rho(theta_i - h) ] / (2h)
```

computed with the same `model_density_on_grid` path the pipeline already uses for `rho0`,
so the columns are consistent by construction with the density they perturb.

Step sizes follow the existing `atomic_benchmark` configuration: 0.02 Angstrom for
coordinates, 0.5 Angstrom squared for B, 0.02 for occupancy.

### 4.2 Locality

An atom's density is truncated at Gemmi's `baseline.density_cutoff`, giving a radius of
roughly 3 Angstrom. Each column is therefore computed and stored only within a local box
around the atom, together with one box per symmetry copy, since `rho0` is
`symmetrize_sum`-ed.

For 1UBQ at approximately 0.58 Angstrom grid spacing this is about 1,100 voxels per copy
and 4 copies, so roughly 4,400 non-zeros per column. At the default basis of 1,980
columns, `Phi` occupies about 70 MB as (value, index) pairs and `Phi^T Phi` is
1980 x 1980 dense, about 16 MB. Both are small enough that a direct solve is appropriate
and no iterative solver is required.

The locality is an optimization and must be validated, not assumed. See section 8.

### 4.3 Basis options

Configurable through `decomposition.basis`:

| value | columns per unit | 1UBQ count | note |
|---|---|---|---|
| `coordinates` | 3 per atom | 1,980 | **default** |
| `coordinates_b` | 4 per atom | 2,640 | adds isotropic B |
| `full` | 5 per atom | 3,300 | adds occupancy |
| `residue_rigid` | 6 per multi-atom residue, 3 per single-atom | 630 | 76 x 6 + 58 x 3 |

Hydrogens and zero-occupancy atoms are excluded, matching `atomic_benchmark`'s existing
atom selection. For 1UBQ that leaves 660 atoms in 134 groups: 76 multi-atom residues and
58 ordered waters.

`coordinates` is the default because coordinate error is the dominant under-modelled
quantity at 1.8 Angstrom and it is the best-conditioned of the meaningful choices. Under
`residue_rigid`, single-atom groups such as ordered waters contribute translation only;
their rotation modes are degenerate and are omitted rather than included as null columns.

## 5. The solve

`numpy.linalg.lstsq` on the normal equations, with optional ridge regularization
(`decomposition.ridge`, default 0.0). SVD-based, so rank deficiency is resolved rather
than producing a spurious solution; rank and condition number are reported.

Explained fraction:

```
explained = 1 - ||target - Phi a||^2 / ||target||^2
```

reported for both targets of section 3, together with a per-parameter-type breakdown
(how much is coordinate-like, B-like, occupancy-like).

## 6. Guards

### 6.1 Capacity control

With 1,980 free parameters, `Phi` will explain a substantial fraction of almost anything.
An explained fraction reported on its own is uninterpretable.

Draw `decomposition.n_capacity_trials` (default 8) random fields through the *same*
operator `L` used by the fit, scale each to `||u||`, symmetrize, and decompose onto the
same `Phi`. Report mean and standard deviation.

The headline result is the pair. If a random field scores 0.70, a real score of 0.75 is
not a finding. Trials are seeded from `run.seed` and are reproducible.

This mirrors the empty-field capacity control already in the repository
(`configs/1ubq/empty_field_control.yaml`) and exists for the same reason.

### 6.2 Restricted basis

The default basis is the smallest that answers the question. The capacity control
measures whether even that is over-parameterised: moving from `coordinates` to `full`
must raise the control's explained fraction, since the larger basis fits noise better.
That relationship is asserted in the test suite (section 9), which is what makes the
control a measurement rather than a decoration.

## 7. Data flow

### 7.1 DAG placement

A new **CPU** stage, `slurm/72_decompose.sbatch`. Everything it does is Gemmi density
calls, sparse linear algebra and FFTs; no accelerator is required, so it does not compete
for GPU allocation and does not call `fieldx_load_cuda`.

```
60_final_fit --+--> 70_information --> 75_freeze_model
               +--> 72_decompose       (leaf)
```

It depends on `afterok:$j_final` and runs concurrently with stage 70.

**The model lock is not changed.** `75_freeze_model` continues to depend on stages 60 and
70 only, and `MODEL_LOCK.json` does not hash the decomposition. The decomposition is
read-only, re-derivable, and makes no claim about held-out data, so locking it buys
nothing while forcing the freeze to wait on a stage it does not depend on. The existing
dependency chain is untouched.

Suggested resources: `--cpus-per-task=8 --mem=32G --time=02:00:00`.

### 7.2 Free-set rule

The full-correction target reads no reflections at all. The data-supported target reads
work reflections only, the same rule the map export follows. The stage runs before the
freeze and never touches the free set.

### 7.3 Configuration

A new top-level block. No `v4` namespace is introduced.

```yaml
decomposition:
  enabled: true
  basis: coordinates          # coordinates | coordinates_b | full | residue_rigid
  box_radius_angstrom: null   # null -> derived from Gemmi's density cutoff
  ridge: 0.0
  n_capacity_trials: 8
  write_maps: true
```

### 7.4 Artifacts

Owned by the stage rather than written into another stage's directory:

```
runs/<ID>/final/decomposition/
    decomposition.json
    explained.ccp4        Phi a       - what ordinary refinement could have produced
    unexplained.ccp4      u - Phi a   - what it could not
```

`decomposition.json` records: the basis and its column count, rank and condition number,
explained fraction for both targets, the per-parameter-type breakdown, the antisymmetric
fraction of `u`, capacity-control mean and standard deviation with the trial count, the
`starting_density` mode of the run being analysed, and a verdict comparing the real
explained fraction against the control.

`maps.json` gains a cross-reference so the full map set remains discoverable from one
place without cross-stage writes.

### 7.5 Command line

```
fieldrefine decompose CONFIG [--basis BASIS] [--trials N]
```

Flags override configuration for exploration without editing files.

## 8. Error handling

- **Missing `z_map.npy`** -- report and point at `fieldrefine fit-map CONFIG`, following
  the idiom `export_maps` already uses.
- **Box truncation** -- the locality optimization is valid only if the local column
  captures the whole derivative. For a sample of atoms, compare the truncated column's
  norm against the full-grid difference computed by the existing `atomic_benchmark` path.
  If it captures less than 99.9%, fail and instruct the user to raise
  `box_radius_angstrom`. This is the one place where the performance shortcut could
  silently corrupt the result.
- **Rank deficiency** -- reported, not treated as an error. A rank below the column count
  is a fact about the basis.
- **`starting_density: zero`** -- permitted. Decomposing the empty-field control's density
  is a legitimate question, and the report records which case was analysed because the
  interpretation differs.
- **Empty basis** -- no atoms surviving selection is an error naming the selection rule.

## 9. Testing

### 9.1 The two tests that make the result trustworthy

**Ground truth.** Displace one atom by a known 0.05 Angstrom, compute the resulting
density difference, and decompose it. The recovered amplitude must match the displacement
and the explained fraction must approach 1. This validates construction, symmetry
handling, the solve and the reporting against an answer known in advance.

**The control must detect over-parameterisation.** A random field through `L` must score
lower than a genuine atomic perturbation; and moving from `coordinates` to `full` must
*raise* the control's explained fraction. If it does not, the control is not measuring
what it claims and the guard is decorative.

### 9.2 Supporting tests

- The local column agrees with the full-grid reference from `atomic_benchmark`,
  validating the locality shortcut against the existing implementation.
- Columns are invariant under the space group's operations.
- The reported antisymmetric fraction matches a deliberately antisymmetric `u`.
- `lstsq` recovers exact coefficients on a well-conditioned synthetic case.
- Capacity trials are seeded and reproducible across runs.
- End-to-end on the synthetic dataset: all artifacts present, maps readable, and
  `explained + unexplained == symmetrize(u)` to numerical tolerance. Note the target is
  the symmetrized correction, per section 3; the parts do not sum to `u` itself unless
  its antisymmetric component is zero.

### 9.3 Leakage

Following the existing pattern in `tests/unit/test_statistical_separation.py`: mutate a
free observation and assert every reported number is unchanged.

### 9.4 The span property

`docs/V4_DESIGN.md` section 2 states that v3 is the special case `Phi = 0`. Pinned now:
with the basis disabled, the analysis must report `explained = 0` and `unexplained = u`.
Trivial for increment 1, but it is the invariant increment 2 must preserve, so it belongs
in the suite from the start.

### 9.5 Control-plane test updates

Two existing tests pin properties this change alters, and will fail until updated:

- `test_the_expected_jobs_exist` pins the exact set of sbatch stems; add `72_decompose`.
- `test_gpu_jobs_load_cuda_and_cpu_jobs_do_not` pins which jobs allocate a GPU; add
  `72_decompose` as a **non-GPU** job, which that test then enforces.

## 10. Increment 2: joint refinement (specified, not implemented)

Implemented only if increment 1 shows `Phi` explains a meaningful fraction above its
capacity control.

**Formulation.** `rho = rho0 + Phi a + L_res z`, optimized jointly over `(a, z)`. Optax
operates on pytrees, so the parameter becomes a mapping rather than a single array; the
optimizer loop, checkpointing and the `z_map.npy` output all need corresponding changes.

**Deflation.** `range(Phi)` is projected out of `L_res` so amplitudes and residual are not
redundant. Without it the two overlap by construction, the amplitudes are unidentifiable
and the optimizer wanders along a flat direction. The projection must be orthogonal with
an explicit adjoint so the existing derivative and adjoint gates apply unchanged.

**Priors.** An independent `tau_res` for the residual, so its scale is not tied to the
mode amplitudes. A sparse prior on `a` -- where sparsity has a meaning, few modes active
and each well determined -- and Gaussian on `z`, where a heavy tail would buy localized
spikes, the opposite of the coherence assumption.

**The cost property.** `docs/V4_DESIGN.md` section 2 requires that for a density change
lying outside `range(Phi)`, the v4 penalty must not exceed the v3 penalty by more than the
cost of the zero amplitudes. This is the property that keeps v4 a superset of v3 rather
than a physical-mode model wearing v3's name. It must be unit-tested before increment 2
is used on data.

**Reporting.** `||Phi a||`, `||L_res z||` and `delta_R_free` separately. A result where
`||L_res z||` dominates is not a failure; it is the most interesting outcome available.

## 11. Risks and limitations

- **The basis can explain noise.** Mitigated by the capacity control and the restricted
  default basis, but the mitigation is a comparison, not a proof. Any reported explained
  fraction must be quoted alongside its control.
- **Finite-difference error.** Columns are approximations. The step sizes are inherited
  from `atomic_benchmark` and the ground-truth test bounds the resulting error, but the
  columns are not exact derivatives.
- **Collinearity.** Neighbouring atoms' derivatives overlap and `d/dB` is near-degenerate
  with `d/dq`. The condition number is reported; a very large one means the per-type
  breakdown should not be over-interpreted even when the total explained fraction is
  sound.
- **A high explained fraction is ambiguous.** It could mean the field found nothing new,
  or that the refinement which produced `rho0` had not fully converged. Distinguishing
  these requires re-refining the atomic model against the amplitudes, which is out of
  scope here and would be the natural follow-up.
- **This does not test v4.** It tests whether v4's premise is worth acting on.

## 12. Decisions taken during design

Recorded so the reasoning is not lost:

1. **Staged rather than full v4 now** -- the diagnostic is a strict subset of the work
   needed for joint refinement and carries none of the identifiability risk.
2. **Atomic tangent space rather than physical modes** -- it answers the manuscript's
   actual claim, needs no external input such as TLS groups, and reuses existing code.
3. **Capacity control plus a restricted basis** rather than cross-validation or a
   regularization path -- cheapest decisive guard, consistent with the idiom already in
   the repository.
4. **Both targets reported** -- the gap between the full correction and its data-supported
   part is itself a result.
5. **Sparse real-space `Phi` with dense normal equations** rather than analytic
   reciprocal-space derivatives or a JAX reimplementation of Gemmi -- locality makes the
   simple approach cheap enough that the sophisticated ones buy nothing.
6. **The model lock is unchanged** -- the decomposition is re-derivable and makes no
   held-out claim.

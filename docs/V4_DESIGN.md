# v4 design

**Status: design only.** v4 does not replace v3 and nothing here is implemented in the
v3 code path. The purpose of this document is to fix the formulation before any code is
written, and — just as importantly — to record what v4 must *not* do to v3.

## 1. Mathematical definition

v3 infers

```
rho(r) = rho0(r) + L_theta z,          z ~ N(0, I)
```

v4 keeps that and adds an explicit, optional decomposition:

```
rho(r) = rho0(r) + Phi a + L_res z
```

- `rho0` — the density of the refined atomic model, as in v3;
- `Phi` — a dictionary of *coherent physical modes*: columns are density perturbations
  produced by a physically meaningful collective change (rigid-group displacement, TLS
  libration, a normal mode, a concerted side-chain rotamer swap, a bulk-solvent
  redistribution). `a` are their amplitudes;
- `L_res z` — a **generic correlated residual field**, the same operator class as v3.

Writing the field this way is a change of basis with an explicit interpretable subspace,
not a narrowing of the hypothesis space, *provided* `L_res` is retained at full v3
generality.

## 2. Relation to v3

v3 is the special case `Phi = 0`. Two properties must hold for v4 to be a superset:

1. **Span.** `range(Phi) + range(L_res)` must contain `range(L_theta)`. If `Phi` is
   dropped, v4 must reduce exactly to v3.
2. **Cost.** For a density change that lies in neither `range(Phi)` nor near it, the v4
   penalty must not exceed the v3 penalty by more than the cost of the (zero) mode
   amplitudes. A v4 prior that makes non-modal density expensive has quietly become a
   physical-mode model.

Property 2 is the whole reason this document exists. **The temptation in v4 is to make
`Phi` mandatory** — to require every density feature to be an atom, a motion, or a
predefined mode — and that would discard v3's central claim. The motivating intuition is
only:

```
one physical feature or motion   ->  coherent multi-voxel pattern
isolated single-voxel spike      ->  likely noise
```

Coherence is a *prior*, and an *interpretation aid*. It is not a requirement that useful
density correspond to a template. Real experimentally supported density may be less
structured than any explicit atomic or physical model, and v4 must still be able to find
it through `L_res`.

## 3. Atomic tangent-space benchmark

The v3 benchmark (`fieldrefine atomic-benchmark`) already measures the latent cost of
elementary refinement directions:

```
d rho / d x_i,  d rho / d B_i,  d rho / d q_i,  d rho / d U_i
```

v4 extends it in two directions:

- the same perturbations are projected onto `range(Phi)`, giving the fraction of an
  ordinary refinement direction that the mode dictionary already explains;
- the residual after that projection is costed under `L_res`, which is the quantity that
  says whether v4 has accidentally become restrictive.

A v4 configuration is unusable if an elementary coordinate perturbation is expensive
under `L_res` *and* poorly represented by `Phi`.

## 4. Coherent physical modes

Candidate constructions for `Phi`, in increasing order of commitment:

| Construction | Source | Interpretability | Risk |
|---|---|---|---|
| Rigid-group translations/rotations | chain and domain definitions | high | too coarse to explain much |
| TLS-like libration modes | the deposited TLS groups | high | inherits the deposited partitioning |
| Elastic-network normal modes | coordinates alone | medium | arbitrary cutoff and force constants |
| Alternate-conformer differences | rebuilt rotamers | high | requires model building |
| Data-driven modes | leading eigenvectors of `J^T J` from v3 | low | not physically labelled |

The last row is the honest bridge from v3: the information spectrum already computed in
stage 70 identifies the density directions the data actually constrains. Using those as
`Phi` makes v4 a re-parameterisation of what v3 measured, rather than an imported
assumption.

## 5. Generic residual field

`L_res` should be the v3 operator, unchanged, with two adjustments:

- optional deflation of `range(Phi)` from `L_res` so amplitudes and residual are not
  redundant (a projection, not a penalty);
- an independent `tau_res`, so the residual's scale is not tied to the mode amplitudes.

If deflation is used it must be an orthogonal projection with an explicit adjoint, so
the derivative and adjoint gates in `crystal_field/validation/derivatives.py` still
apply unchanged.

## 6. Sparsity and heavy tails

Heavy-tailed latent statistics (Student-t, Laplace, Cauchy) already exist in the v3
`PriorConfig` schema and are deliberately excluded from the first pilot. In v4 they
become substantively useful on the **mode amplitudes** `a`, where sparsity has a
meaning: few modes active, each well determined. On the residual field `z` a heavy tail
mainly buys localized spikes, which is the opposite of the coherence assumption.

Proposed default: sparse prior on `a`, Gaussian on `z`. Any other combination must be
justified against the tune subset, never the free set.

## 7. Discovery and interpretation

v4's reporting should separate three quantities:

```
||Phi a||          density explained by interpretable coherent modes
||L_res z||        density that is coherent but not modal
delta_R_free       whether either was worth anything predictively
```

A result where `||L_res z||` dominates is *not* a failure of v4. It is the most
interesting outcome available: experimentally supported density that no physical
template explains. A result where `||Phi a||` dominates is the reassuring outcome, and
also the one most at risk of being an artifact of how `Phi` was built.

## 8. Risks and limitations

- **Restriction creep.** Every mode added to `Phi` is a modelling assumption. If mode
  selection is tuned aggressively, v4 stops testing v3's question. Mitigation: the span
  and cost properties in section 2 must be unit-tested before v4 is used on data.
- **Redundancy.** `Phi` and `L_res` overlap by construction; without deflation the
  amplitudes are unidentifiable and the optimizer wanders along a flat direction.
- **Circularity.** Building `Phi` from the v3 information spectrum of the same dataset
  and then evaluating on the same free set is not a clean test. If `Phi` is derived from
  data, it must be derived from the work set only, before the lock.
- **Cost.** `Phi` is dense in the voxel basis. Storing it explicitly defeats the
  matrix-free design; it must be applied as an operator, like everything else.
- **Diffuse scattering is still out of scope here.** Bragg data constrains the mean
  periodic density. A fluctuation/covariance likelihood around a frozen Bragg-derived
  mean field is a separate formulation again, and it should not be conflated with v4's
  mode decomposition of the mean.

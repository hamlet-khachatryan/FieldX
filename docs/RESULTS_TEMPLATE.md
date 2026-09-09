# Results template

One copy per dataset per locked model. Fill every field from the machine-readable
artifacts; do not paste numbers that no artifact contains.

```
dataset          <PDBID>
config sha256    <MODEL_LOCK.json: config_sha256>
grid shape       <metadata.json: grid_shape>
n voxels
samples_per_dmin
d_min / d_max
space group / cell
```

## Free-set provenance

```
split strategy   existing_free_then_hash | hash
free_set_kind    <FREE_EVALUATION.json: free_set_kind>
```

A `hash` holdout is a genuine cross-validation statistic but is **not** the deposited
crystallographic R-free. State which one this is before quoting any number below.

## Starting density (stage 20)

```
integrated electrons / model electrons
electron-count relative error
amplitude relative L2 error
pass
```

## Numerical gates (stage 30)

```
FFT relative L2 error, direct
FFT relative L2 error, conjugated
matched relation / ambiguous
finite-difference step
directional derivative relative error
adjoint relative error
accelerator, precision
```

## Atomic representability (stage 45)

```
per-kind max latent RMS      x | b_iso | occupancy | adp_aniso
min band-limited fraction
```

## Prior comparison (stage 50 / 52)

Ranked on the tune subset only.

| candidate | kernel | tau | correlation length | latent | chi2_tune / n_tune | r_tune | r_train | iterations |
|---|---|---:|---:|---|---:|---:|---:|---:|
| | | | | | | | | |

```
winner
selection criterion    mean tune chi-square, ties broken by tune R
```

## Locked fit (stage 60 / 70)

```
work reflections     n_train + n_tune
free reflections     n_free (unread at this point)
final work R
objective components
convergence           iterations, final gradient norm
leading information eigenvalues
d_eff lower bound
information lower bound (nats)
estimated GPU GiB
```

## One-shot free evaluation (manual stage 80)

| metric | atomic | field | change |
|---|---:|---:|---:|
| R_work | `r_work_atomic` | `r_work_field` | `delta_r_work` |
| R_free | `r_free_atomic` | `r_free_field` | `delta_r_free` |
| R_free − R_work | `gap_atomic` | `gap_field` | `delta_gap` |

```
primary_criterion_met    <true|false>
evaluation_index         <must be 1 for a one-shot result>
ledger entries
```

## Capacity control (`starting_density: zero`)

Run separately, judged on tune only, never on free.

| | `r_train` | `r_tune` |
|---|---:|---:|
| model start | | |
| empty start | | |

```
parameters per observation = n_voxels / n_reflections
```

If the empty start reaches a comparable `r_train`, then `R_work` alone carries no
evidence and only the held-out comparison counts. If it also reaches a comparable
`r_tune`, the prior is too weak and the main result cannot be interpreted.

## Interpretation

- Did **both** R_work and R_free fall? A fall in R_work alone is overfitting, not
  evidence of missing density.
- What happened to the gap, and is that consistent with the R changes?
- Where is the inferred density change concentrated — is it spatially coherent, and does
  it sit in well-informed information modes?
- Are fine atomic-like corrections preserved, per the representability benchmark?
- What density change is present that no atomic or physical template explains? This is
  the outcome v3 exists to look for, and it should be described rather than explained
  away.
- What remains unexplained, and what would distinguish the competing explanations?

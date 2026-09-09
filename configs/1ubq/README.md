# 1UBQ -- first v3 experiment

Ubiquitin, `P 21 21 21`, cell `50.84 42.77 28.95 90 90 90`, 1.80 A, 6029 merged
amplitudes, 660 non-hydrogen atoms. Small and fast: the derived FFT grid at
`samples_per_dmin = 3.0` is `90 x 72 x 50` (324k voxels), so a full pilot fits
comfortably on one GPU.

Fetch the data (the repository never commits experimental reflections):

```bash
uv run fieldrefine init-pdb 1UBQ
```

## The free set is a hash holdout, not a deposited R-free

1UBQ's deposited `FreeR_flag` column is constant -- every one of the 6029 reflections
carries the value 1 -- so the entry defines no held-out set. `configs/1ubq/default.yaml`
therefore uses `split.strategy: hash`, a deterministic Friedel-paired holdout of 10%.

That set is still genuinely untouched by fitting, prior selection, early stopping and
nuisance calibration, so it remains a valid cross-validation statistic. It is **not**
the deposited crystallographic R-free, and every report must say so. `evaluate-free`
records this in its `free_set_kind` field.

## Prior grid

`prior_grid.yaml` holds the four v3 pilot candidates: Matern at three correlation
lengths plus a squared-exponential control, all Gaussian. Heavy-tailed latents and
multiscale mixtures are excluded on purpose -- the first experiment tests the existing
v3 formulation, not extra prior flexibility.

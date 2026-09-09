# HPC and SLURM guide

## Policy

Login node: configuration, inspection, prior-grid expansion, lightweight tests, `sbatch`
submission, queue monitoring.

SLURM GPU node: all production density calculation, FFTs, autodiff, optimization, prior
comparison, the information spectrum and large GPU tests.

An interactive GPU node is never required for normal operation. The login node has no
GPU and correctly reports a CPU JAX device; that is not a failure.

## Environment

One environment, created by uv, living in the repository:

```
FieldX/.venv
```

There is no `pip install --user`, no `python -m venv`, no generated
`environment/cluster.env`, and no variable that must be exported to find the environment.

```bash
uv sync --locked --extra cuda13 --group dev     # GPU cluster
uv sync --locked --extra cpu    --group dev     # laptop, login node, CI
```

Compute jobs use `uv run --frozen --no-sync`, so **dependency resolution never happens
on a compute node**.

### Keep uv off the home quota

The `cuda12`/`cuda13` extras pull several GB of NVIDIA wheels. If uv's cache sits on a
small home quota the sync dies mid-extraction:

```
failed to create directory `/home/<user>/.cache/uv/.tmp.../nvidia/nvshmem/include`:
Disk quota exceeded (os error 122)
```

Export uv's own variables *before* the first sync — `FIELDX_UV_CACHE` is only read once
`slurm/common.sh` has been sourced, so it does not cover a bare `uv sync`:

```bash
export UV_CACHE_DIR=/dls/data2temp01/hamlet/workspace/uv-cache
export UV_PYTHON_INSTALL_DIR=/dls/data2temp01/hamlet/workspace/uv-python
uv cache dir        # confirm, then sync
```

Recover from a partial download with `uv cache clean`. Putting the cache on the same
filesystem as the repository also lets uv hardlink into `.venv` instead of copying.
`scripts/cluster_preflight.sh` fails on a submission host if either directory is under
`$HOME`.

### Optional environment variables

| Variable | Effect |
|---|---|
| `UV_CACHE_DIR` | uv's package cache. **Set this before the first `uv sync` on a cluster** |
| `UV_PYTHON_INSTALL_DIR` | Where uv puts managed interpreters |
| `FIELDX_UV` | Absolute path to a `uv` executable, when uv is neither on `PATH` nor in `.venv/bin` |
| `FIELDX_UV_CACHE` | Alias that `common.sh` maps onto `UV_CACHE_DIR`, for jobs that export only FieldX variables |
| `FIELDX_UV_PYTHON_DIR` | Alias that `common.sh` maps onto `UV_PYTHON_INSTALL_DIR` |
| `FIELDX_DATA_ROOT` | Where `init-pdb` puts datasets (default `./data`) |
| `FIELDX_RUNS_ROOT` | Where `init-pdb` points run outputs (default `./runs`) |
| `FIELDX_JAX_CACHE` | JAX compilation cache directory (default `<repo>/.jax-cache`) |
| `FIELDX_CUDA_INIT` | Site CUDA init script (default `slurm/dls/cuda.sh`) |
| `CFI_SBATCH_ARGS` | Extra `sbatch` flags: account, partition, QoS |
| `CFI_ARRAY_LIMIT` | Concurrent prior-candidate array tasks (default 4) |

`slurm/common.sh` resolves `PROJECT_ROOT`, the uv executable and the environment; it is
sourced by every SLURM job and by the login-node helper scripts.

## Path resolution under SLURM

`sbatch` may stage the submitted script under a spool directory such as
`/var/spool/slurm/job12345/slurm_script`. `$0` therefore does **not** point into the
repository, and

```bash
source "$(dirname "$0")/common.sh"     # WRONG: breaks under SLURM
```

fails at runtime. Every job instead begins with

```bash
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/slurm/common.sh"
```

`scripts/submit.sh` invokes `sbatch` from the repository root, so `SLURM_SUBMIT_DIR` is
the checkout. `slurm/common.sh` validates it (it must contain `pyproject.toml` and
`slurm/common.sh`) and falls back to `CFI_PROJECT_ROOT`, then to its own location.

`tests/unit/test_slurm_control_plane.py` enforces this: it greps every `.sbatch` file for
`$0` usage, constructs a fake SLURM environment with the working directory in a spool
path, and checks that `common.sh` still resolves the repository.

## CUDA

Every GPU job loads CUDA itself. A batch job does not inherit an interactive shell's
module environment:

```bash
fieldx_load_cuda        # sources $FIELDX_CUDA_INIT, default slurm/dls/cuda.sh
```

`slurm/dls/cuda.sh` is the only site-specific file in the repository. On DLS it runs
`module load cuda`, which provides `/dls_sw/apps/cuda/13.3.1` with CUPTI at
`extras/CUPTI/lib64/libcupti.so`, and prepends CUPTI to `LD_LIBRARY_PATH` if the module
exports only `CUDA_HOME`. JAX 0.10+ with the CUDA 13 plugin works on a GPU node after
that.

CPU jobs never call `fieldx_load_cuda`, and no login-node command requires CUDA. The
control-plane tests assert both directions: every job with `--gres=gpu` loads CUDA, and
no job without one does.

Porting to another cluster means writing one file and pointing `FIELDX_CUDA_INIT` at it.

## Memory planning

```bash
uv run fieldrefine estimate-memory configs/1ubq/default.yaml
```

works before anything has been prepared, deriving the grid from the model cell, `d_min`
and `samples_per_dmin`. It reports the cost of a single real field, a complex FFT buffer,
the resident set, and the totals for Adam, L-BFGS and the LOBPCG information solve,
including a 1.35x allowance for XLA and cuFFT workspaces.

The information spectrum is guarded: `information.memory_budget_gib` (default 40) is
checked *before* LOBPCG allocates its 3k-wide basis of full latent fields, so an
oversized configuration fails in seconds instead of OOM-ing hours in. Stage 30 and stage
70 both print the estimate.

Nothing in the implementation forms a dense Jacobian or an `N_vox x N_vox` matrix. `J v`
is a JVP, `J^T w` is a VJP, the prior is applied by FFT, and LOBPCG is matrix-free.

## Memory and cache environment

`slurm/common.sh` sets:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=0.80          # only applies when preallocation is on
JAX_COMPILATION_CACHE_DIR=<repo>/.jax-cache  # or $FIELDX_JAX_CACHE
```

**Preallocation is off by default, deliberately.** Reserving most of the device up front
leaves CUDA no room *outside* the arena to load compiled modules, and the job dies as

```
RESOURCE_EXHAUSTED: Failed to load in-memory CUBIN (compiled for a different GPU?):
CUDA_ERROR_OUT_OF_MEMORY [executable_name='jit_copy']
```

which is easy to misread as a data-size problem: it can happen while copying a single
scalar to the host, on a problem needing well under a GiB. The "compiled for a different
GPU?" clause is XLA's generic hint, not the cause.

FieldX grids are modest and `fieldrefine estimate-memory` sizes them in advance, so
on-demand allocation is the safer default. Turn preallocation back on — with a fraction
that leaves headroom, 0.7 to 0.8, never 0.9 — only if a large information solve shows
fragmentation:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
```

`fieldrefine device-report` prints `memory` (`bytes_in_use`, `peak_bytes_in_use`,
`bytes_limit`) and the preallocation settings in force, so compare `bytes_limit` against
the card's total before changing anything. Stage 30 and stage 60 both run it.

Put the compilation cache on a fast shared filesystem: prior candidates share shapes and
precision and so reuse compiled executables. If the DAG has run across GPUs of different
architectures, clear `.jax-cache` — a cached CUBIN from another card is unloadable.

## First run

Keep the pilot small — one GPU, one prior, 50–100 optimizer iterations, 8 information
modes — and measure compilation time, time per iteration, GPU memory and objective
convergence before running the four-prior comparison. `configs/1ubq/default.yaml` already
carries those settings. Do not open with a large production sweep.

## Candidate concurrency

```bash
export CFI_ARRAY_LIMIT=4
```

Each prior candidate is an independent GPU job in a SLURM array. Start conservatively.

## Free-set protection

Stage 75 writes `MODEL_LOCK.json`, hashing the configuration, prepared reflections,
starting density, solvent mask, work scaling, fitted latent field and information
spectrum. Stage 80 is **never submitted automatically**; `scripts/submit.sh` prints the
exact command instead.

`evaluate-free` verifies the lock before reading a single free reflection and reports the
atomic baseline and the field model on the same reflections in the same run. Every read
appends to `FREE_SET_LEDGER.json` beside the prepared reflections, after which both
`evaluate-free` and `freeze-model` refuse without an explicit override. The overrides
`--allow-repeat-free-evaluation` and `--allow-after-free-evaluation` exist for
deliberate, reportable exceptions: they forfeit the held-out status, write repeats to
`FREE_EVALUATION_repeat_<n>.json` rather than overwriting, and are themselves recorded.

## Apptainer

Optional, and not part of the normal DLS workflow (`module load cuda` + the repository
`.venv` + SLURM). Set `CFI_APPTAINER_IMAGE` and optionally `CFI_APPTAINER_BINDS`;
`slurm/common.sh` then routes `run_cfi` and `run_uv` through `apptainer exec --nv`.
`containers/Apptainer.def` builds the image on a site that permits container creation.

## Transfer to a disconnected cluster

```bash
scripts/package_transfer.sh /path/to/fieldx-transfer.tar.gz
```

Ships source, tests, configs, manuscript and SLURM jobs, with `uv.lock`, but no
experimental data, run products or local environments. Build a wheelhouse on a networked
node if the target has no package mirror; do not resolve packages from compute nodes.

"""Declarative configuration schema for FieldX v3.

The v3 model is

    rho(r) = rho0(r) + L_theta z,      z ~ N(0, I)

where rho0 is the atomistically generated starting density and L_theta is a
spatially correlated, band-limited density-field operator. Every quantity below
is expressed in physical crystallographic units (Angstrom, Angstrom^-2); nothing
is expressed in voxels, so changing the FFT grid never changes the prior.

Relative paths in a configuration file are resolved against the directory that
contains that file, so a config can be committed with repository-relative paths
and still resolve identically on a laptop and on a cluster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Path-valued fields, as (section, key) pairs. Resolution is explicit rather than
# inferred from the annotation so that adding a path field is a deliberate act.
PATH_FIELDS = (
    ("run", "output_dir"),
    ("run", "shared_data_dir"),
    ("run", "compilation_cache_dir"),
    ("input", "reflections"),
    ("input", "model"),
    ("input", "diffuse_h5"),
)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(Strict):
    output_dir: Path
    shared_data_dir: Path | None = None
    seed: int = 20260907
    enable_x64: bool = False
    compilation_cache_dir: Path | None = None

    @property
    def data_dir(self) -> Path:
        return self.shared_data_dir or (self.output_dir / "shared")


class ColumnConfig(Strict):
    observation: str
    sigma: str
    free: str | None = None


class InputConfig(Strict):
    reflections: Path
    model: Path
    # 6O2H has an associated diffuse-scattering map. v3 never reads it; the path is
    # recorded so `fieldrefine inspect-h5` can describe it and v4 can consume it.
    diffuse_h5: Path | None = None
    scattering: Literal["xray", "electron", "neutron"] = "xray"
    observation_kind: Literal["amplitude", "intensity"] = "amplitude"
    columns: ColumnConfig
    # Which dataset inside a multi-dataset MTZ supplies the columns above: a dataset id
    # or name. Required whenever the file carries several data-bearing datasets and the
    # configured labels do not single one out (MAD/SAD wavelengths reuse labels).
    # Reflection CIFs have no dataset concept and ignore this.
    mtz_dataset: int | str | None = None
    free_test_value: int | None = None
    expected_free_fraction_min: float = Field(0.02, ge=0.0, le=1.0)
    expected_free_fraction_max: float = Field(0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def free_fraction_bounds(self):
        if self.expected_free_fraction_min >= self.expected_free_fraction_max:
            raise ValueError("expected_free_fraction_min must be < expected_free_fraction_max")
        return self


class SplitConfig(Strict):
    """How reflections are divided into train, tune and free.

    `existing_free_then_hash` uses the deposited free flags. `hash` builds a
    deterministic Friedel-paired holdout when the entry deposits none -- still a valid
    cross-validation statistic, though not the deposited crystallographic R-free.

    `none` declares that no held-out set exists at all. The whole dataset becomes work,
    the only target is R_work, and every free-set stage degrades to a clearly labelled
    no-op instead of failing. Use it only when a holdout is genuinely impossible: with
    nothing held out, a lower R_work is not evidence of anything, because a field model
    with this much capacity can always reduce it.
    """

    strategy: Literal["existing_free_then_hash", "hash", "none"] = "existing_free_then_hash"
    tune_fraction_of_work: float = Field(0.10, gt=0.0, lt=0.5)
    final_fraction_if_hash: float = Field(0.05, gt=0.0, lt=0.5)
    pair_friedel_when_nonanomalous: bool = True
    anomalous: bool = False


class ResolutionConfig(Strict):
    d_min_angstrom: float = Field(gt=0.0)
    d_max_angstrom: float | None = Field(default=None, gt=0.0)


class GridConfig(Strict):
    """Computational FFT grid.

    The physical unit cell and the computational grid are distinct concepts. When
    `shape` is null the grid is derived from the cell, the experimental d_min and
    `samples_per_dmin`, rounded up to FFT-friendly dimensions by Gemmi.
    """

    shape: tuple[int, int, int] | None = None
    samples_per_dmin: float = Field(3.0, ge=2.0, le=6.0)

    @model_validator(mode="after")
    def validate_shape(self):
        if self.shape is not None:
            if any(int(n) < 4 for n in self.shape):
                raise ValueError(f"grid.shape entries must each be at least 4; got {self.shape}")
            if any(int(n) % 2 for n in self.shape):
                raise ValueError(
                    f"grid.shape entries must be even so the Nyquist guard is well defined; got {self.shape}"
                )
        return self


class PriorComponent(Strict):
    correlation_length_angstrom: float = Field(gt=0.0)
    alpha: float = Field(2.5, gt=0.0)
    weight: float = Field(gt=0.0)


class PriorConfig(Strict):
    """Correlated, band-limited density-field prior.

    The purpose of the correlation is that coherent multi-voxel density changes are
    plausible while isolated single-voxel excursions are strongly disfavored. It is
    deliberately generic: a meaningful density feature may be unfamiliar or
    non-atomistic, and the prior must not force every change into an atom.
    """

    kernel: Literal[
        "matern",
        "squared_exponential",
        "bandlimited_white",
        "multiscale_matern",
    ] = "matern"
    tau_density: float = Field(0.05, gt=0.0)
    correlation_length_angstrom: float = Field(1.0, gt=0.0)
    alpha: float = Field(2.5, gt=0.0)
    components: list[PriorComponent] = Field(default_factory=list)
    remove_mean: bool = True
    latent_distribution: Literal["gaussian", "student_t", "laplace", "cauchy"] = "gaussian"
    student_t_df: float = Field(4.0, gt=2.0)
    laplace_softening: float = Field(1e-3, gt=0.0)
    cauchy_scale: float = Field(1.0, gt=0.0)

    @model_validator(mode="after")
    def validate_prior(self):
        if self.kernel == "matern" and self.alpha <= 1.5:
            raise ValueError("For a 3D Matern field, alpha must exceed 1.5 or the field has no finite variance")
        if self.kernel == "multiscale_matern":
            if not self.components:
                raise ValueError("multiscale_matern requires at least one component")
            for component in self.components:
                if component.alpha <= 1.5:
                    raise ValueError("Each multiscale_matern component needs alpha > 1.5")
        elif self.components:
            raise ValueError(f"prior.components is only meaningful for multiscale_matern, not {self.kernel!r}")
        return self


class BulkSolventConfig(Strict):
    enabled: bool = True
    radii: Literal["refmac", "cctbx", "van_der_waals"] = "refmac"
    k_sol: float = Field(0.35, ge=0.0)
    b_sol_angstrom2: float = Field(46.0, ge=0.0)


class ScalingConfig(Strict):
    enabled: bool = True
    fit_isotropic_b_first: bool = True


class BaselineConfig(Strict):
    # "model" is the v3 experiment: refine a correction to the refined atomic density.
    # "zero" starts from an empty cell and is a CAPACITY CONTROL, not a phasing method:
    # with amplitudes alone the problem is phase-degenerate, so a low R_work reached this
    # way measures how much the field class can fit without structural information.
    starting_density: Literal["model", "zero"] = "model"
    density_cutoff: float = Field(1e-6, gt=0.0)
    bulk_solvent: BulkSolventConfig = Field(default_factory=BulkSolventConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)


class LikelihoodConfig(Strict):
    global_scale: Literal["profile", "fixed"] = "profile"
    fixed_scale: float = Field(1.0, gt=0.0)
    sigma_floor: float = Field(1e-6, gt=0.0)
    loss: Literal["gaussian", "student_t"] = "gaussian"
    student_t_df: float = Field(6.0, gt=2.0)


class OptimizerConfig(Strict):
    method: Literal["lbfgs", "adam"] = "lbfgs"
    fit_scope: Literal["train", "work"] = "train"
    # |F| is not differentiable at F = 0, so an empty starting density must not be
    # optimized from z = 0: the gradient is NaN. "random" draws z ~ init_scale * N(0, I).
    init: Literal["zero", "random"] = "zero"
    init_scale: float = Field(1.0, gt=0.0)
    max_iterations: int = Field(300, ge=1)
    tolerance: float = Field(5e-5, gt=0.0)
    lbfgs_memory: int = Field(8, ge=1, le=50)
    adam_learning_rate: float = Field(3e-3, gt=0.0)
    checkpoint_every: int = Field(10, ge=1)
    validation_patience: int = Field(8, ge=1)
    validation_min_delta: float = Field(1e-5, ge=0.0)


class InformationConfig(Strict):
    n_modes: int = Field(16, ge=1)
    max_iterations: int = Field(80, ge=1)
    tolerance: float = Field(1e-4, gt=0.0)
    # LOBPCG carries a 3k-wide basis of full latent fields. Refuse to launch a job
    # whose estimate exceeds this rather than discovering it as an OOM hours in.
    memory_budget_gib: float = Field(40.0, gt=0.0)


class AtomicBenchmarkConfig(Strict):
    enabled: bool = True
    n_atoms: int = Field(8, ge=1)
    coordinate_delta_angstrom: float = Field(0.02, gt=0.0)
    b_delta_angstrom2: float = Field(0.5, gt=0.0)
    occupancy_delta: float = Field(0.02, gt=0.0, lt=0.5)


class AppConfig(Strict):
    run: RunConfig
    input: InputConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    resolution: ResolutionConfig
    grid: GridConfig = Field(default_factory=GridConfig)
    prior: PriorConfig = Field(default_factory=PriorConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    likelihood: LikelihoodConfig = Field(default_factory=LikelihoodConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    information: InformationConfig = Field(default_factory=InformationConfig)
    atomic_benchmark: AtomicBenchmarkConfig = Field(default_factory=AtomicBenchmarkConfig)

    @property
    def has_free_set(self) -> bool:
        """False when the configuration declares that no held-out set exists."""
        return self.split.strategy != "none"

    @model_validator(mode="after")
    def validate_global(self):
        if self.split.strategy == "existing_free_then_hash":
            if not self.input.columns.free:
                raise ValueError("columns.free is required for existing_free_then_hash")
            if self.input.free_test_value is None:
                raise ValueError("input.free_test_value is required for existing_free_then_hash")
        if self.baseline.scaling.enabled and self.input.observation_kind != "amplitude":
            raise ValueError("Gemmi baseline scaling currently requires amplitude observations")
        if self.baseline.starting_density == "zero":
            # An empty cell makes the model-derived nuisance terms meaningless and the
            # zero start numerically undefined. Require the experiment to say so
            # explicitly rather than silently reinterpreting the configuration.
            problems = []
            if self.baseline.scaling.enabled:
                problems.append("baseline.scaling.enabled must be false (a zero density has no scale to fit)")
            if self.baseline.bulk_solvent.enabled:
                problems.append("baseline.bulk_solvent.enabled must be false (there is no model to mask)")
            if self.likelihood.global_scale != "profile":
                problems.append("likelihood.global_scale must be 'profile' (it is the only source of scale)")
            if self.optimizer.init != "random":
                problems.append("optimizer.init must be 'random' (the gradient at z=0 is NaN when rho0=0)")
            if problems:
                raise ValueError("baseline.starting_density='zero' requires:\n  - " + "\n  - ".join(problems))
        if self.split.anomalous:
            raise ValueError(
                "v3 models a real-valued mean density and does not support anomalous differences. "
                "Use non-anomalous merged amplitudes or intensities."
            )
        if (
            self.resolution.d_max_angstrom is not None
            and self.resolution.d_max_angstrom < self.resolution.d_min_angstrom
        ):
            raise ValueError("resolution.d_max_angstrom must be >= d_min_angstrom")
        return self


def resolve_payload_paths(payload: dict, base_dir: Path) -> dict:
    """Make every relative path in a raw config payload absolute against `base_dir`."""
    base_dir = Path(base_dir).resolve()
    for section, key in PATH_FIELDS:
        block = payload.get(section)
        if not isinstance(block, dict):
            continue
        value = block.get(key)
        if value in (None, ""):
            continue
        path = Path(value)
        block[key] = str(path if path.is_absolute() else (base_dir / path).resolve())
    return payload


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return AppConfig.model_validate(resolve_payload_paths(payload, path.parent))


def dump_config(cfg: AppConfig, path: str | Path) -> None:
    """Write a config with absolute paths, so a dumped config is location-independent."""
    payload = cfg.model_dump(mode="json", exclude_none=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def check_config(path: str | Path) -> dict:
    """Schema validation plus the input checks that are cheap on a login node."""
    cfg = load_config(path)
    problems = []
    for label, target in (("input.reflections", cfg.input.reflections), ("input.model", cfg.input.model)):
        if not target.is_file():
            problems.append(f"{label} does not exist: {target}")
    if cfg.input.diffuse_h5 is not None and not cfg.input.diffuse_h5.is_file():
        problems.append(f"input.diffuse_h5 does not exist: {cfg.input.diffuse_h5}")
    if cfg.split.strategy == "existing_free_then_hash" and not cfg.input.columns.free:
        problems.append("split.strategy=existing_free_then_hash needs input.columns.free")
    # Crystallographic checks need the model's cell and space group, which is cheap to
    # read and catches at submission time what would otherwise fail inside stage 20.
    crystal = {}
    if cfg.input.model.is_file():
        import gemmi

        from crystal_field.crystallography.symmetry import group_order, validate_grid_shape

        structure = gemmi.read_structure(str(cfg.input.model))
        spacegroup = gemmi.SpaceGroup(structure.spacegroup_hm) if structure.spacegroup_hm else None
        if spacegroup is not None:
            if not structure.cell.is_compatible_with_spacegroup(spacegroup):
                problems.append(
                    f"unit cell {structure.cell} is not metrically compatible with space group {spacegroup.xhm()}"
                )
            if cfg.grid.shape is not None:
                try:
                    validate_grid_shape(cfg.grid.shape, structure.cell, spacegroup)
                except ValueError as exc:
                    problems.append(str(exc))
            crystal = {
                "spacegroup": spacegroup.xhm(),
                "symmetry_copies_in_cell": group_order(spacegroup),
            }

    if problems:
        raise ValueError("Configuration is not usable:\n  - " + "\n  - ".join(problems))
    return {
        "config": str(Path(path).resolve()),
        "output_dir": str(cfg.run.output_dir),
        "data_dir": str(cfg.run.data_dir),
        "reflections": str(cfg.input.reflections),
        "model": str(cfg.input.model),
        "d_min_angstrom": cfg.resolution.d_min_angstrom,
        "samples_per_dmin": cfg.grid.samples_per_dmin,
        "split_strategy": cfg.split.strategy,
        "mtz_dataset": cfg.input.mtz_dataset,
        "has_free_set": cfg.has_free_set,
        "target": "R_work and R_free" if cfg.has_free_set else "R_work only (no held-out set)",
        "starting_density": cfg.baseline.starting_density,
        "prior_kernel": cfg.prior.kernel,
        "latent_distribution": cfg.prior.latent_distribution,
        "fit_scope": cfg.optimizer.fit_scope,
        **crystal,
        "ok": True,
    }

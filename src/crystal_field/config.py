from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class RunConfig(BaseModel):
    output_dir: Path
    shared_data_dir: Path | None = None
    seed: int = 20260906
    enable_x64: bool = False
    compilation_cache_dir: Path | None = None

    @property
    def data_dir(self) -> Path:
        return self.shared_data_dir or (self.output_dir / "shared")


class ColumnConfig(BaseModel):
    observation: str
    sigma: str
    free: str | None = None


class InputConfig(BaseModel):
    reflections: Path
    model: Path
    diffuse_h5: Path | None = None
    scattering: Literal["xray", "electron", "neutron"] = "xray"
    observation_kind: Literal["amplitude", "intensity"] = "amplitude"
    columns: ColumnConfig
    free_test_value: int | None = None
    expected_free_fraction_min: float = Field(0.02, ge=0.0, le=1.0)
    expected_free_fraction_max: float = Field(0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def free_fraction_bounds(self):
        if self.expected_free_fraction_min >= self.expected_free_fraction_max:
            raise ValueError("expected_free_fraction_min must be < expected_free_fraction_max")
        return self


class SplitConfig(BaseModel):
    strategy: Literal["existing_free_then_hash", "hash"] = "existing_free_then_hash"
    tune_fraction_of_work: float = Field(0.10, gt=0.0, lt=0.5)
    final_fraction_if_hash: float = Field(0.05, gt=0.0, lt=0.5)
    pair_friedel_when_nonanomalous: bool = True
    anomalous: bool = False


class ResolutionConfig(BaseModel):
    d_min_angstrom: float = Field(gt=0.0)
    d_max_angstrom: float | None = Field(default=None, gt=0.0)


class GridConfig(BaseModel):
    shape: tuple[int, int, int] | None = None
    samples_per_dmin: float = Field(3.0, ge=2.0, le=6.0)


class PriorComponent(BaseModel):
    correlation_length_angstrom: float = Field(gt=0.0)
    alpha: float = Field(2.5, gt=0.0)
    weight: float = Field(gt=0.0)


class PriorConfig(BaseModel):
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
            raise ValueError("For a 3D Matérn field, alpha should be > 1.5")
        if self.kernel == "multiscale_matern" and not self.components:
            raise ValueError("multiscale_matern requires at least one component")
        return self


class BulkSolventConfig(BaseModel):
    enabled: bool = True
    radii: Literal["refmac", "cctbx", "van_der_waals"] = "refmac"
    k_sol: float = Field(0.35, ge=0.0)
    b_sol_angstrom2: float = Field(46.0, ge=0.0)


class ScalingConfig(BaseModel):
    enabled: bool = True
    fit_isotropic_b_first: bool = True


class BaselineConfig(BaseModel):
    refmac_compatible_density_blur: bool = False
    density_cutoff: float = Field(1e-6, gt=0.0)
    bulk_solvent: BulkSolventConfig = Field(default_factory=BulkSolventConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)


class LikelihoodConfig(BaseModel):
    global_scale: Literal["profile", "fixed"] = "profile"
    fixed_scale: float = Field(1.0, gt=0.0)
    sigma_floor: float = Field(1e-6, gt=0.0)
    loss: Literal["gaussian", "student_t"] = "gaussian"
    student_t_df: float = Field(6.0, gt=2.0)


class OptimizerConfig(BaseModel):
    method: Literal["lbfgs", "adam"] = "lbfgs"
    fit_scope: Literal["train", "work"] = "train"
    max_iterations: int = Field(300, ge=1)
    tolerance: float = Field(5e-5, gt=0.0)
    lbfgs_memory: int = Field(8, ge=1, le=50)
    adam_learning_rate: float = Field(3e-3, gt=0.0)
    checkpoint_every: int = Field(10, ge=1)
    validation_patience: int = Field(8, ge=1)
    validation_min_delta: float = Field(1e-5, ge=0.0)


class InformationConfig(BaseModel):
    n_modes: int = Field(16, ge=1)
    max_iterations: int = Field(80, ge=1)
    tolerance: float = Field(1e-4, gt=0.0)


class AtomicBenchmarkConfig(BaseModel):
    enabled: bool = True
    n_atoms: int = Field(8, ge=1)
    coordinate_delta_angstrom: float = Field(0.02, gt=0.0)
    b_delta_angstrom2: float = Field(0.5, gt=0.0)
    occupancy_delta: float = Field(0.02, gt=0.0, lt=0.5)


class AppConfig(BaseModel):
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

    @model_validator(mode="after")
    def validate_global(self):
        if self.split.strategy == "existing_free_then_hash":
            if not self.input.columns.free:
                raise ValueError("columns.free is required for existing_free_then_hash")
            if self.input.free_test_value is None:
                raise ValueError("input.free_test_value is required for existing_free_then_hash")
        if self.baseline.scaling.enabled and self.input.observation_kind != "amplitude":
            raise ValueError("Gemmi baseline scaling currently requires amplitude observations")
        if self.baseline.refmac_compatible_density_blur:
            raise ValueError(
                "The inferred field must use an unblurred physical density grid. "
                "Keep baseline.refmac_compatible_density_blur=false for v1."
            )
        if self.split.anomalous:
            raise ValueError(
                "v1 models a real-valued mean density and does not support anomalous differences. "
                "Use non-anomalous merged amplitudes/intensities for v1."
            )
        if self.resolution.d_max_angstrom is not None and self.resolution.d_max_angstrom < self.resolution.d_min_angstrom:
            raise ValueError("resolution.d_max_angstrom must be >= d_min_angstrom")
        return self


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    return AppConfig.model_validate(payload)


def dump_config(cfg: AppConfig, path: str | Path) -> None:
    payload = cfg.model_dump(mode="json", exclude_none=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_pdb_template(path: Path, entry: dict, refl: dict) -> None:
    payload = {
        "run": {"output_dir": str(path.parent.parent.parent / "runs" / entry["pdb_id"]), "seed": 20260907, "enable_x64": False},
        "input": {
            "reflections": str(Path(entry["reflections"]).resolve()),
            "model": str(Path(entry["model"]).resolve()),
            "scattering": "xray",
            "observation_kind": "amplitude",
            "columns": {"observation": refl["observation"], "sigma": refl["sigma"], "free": refl["free"]},
            "free_test_value": refl["free_test_value"],
        },
        "split": {
            "strategy": "existing_free_then_hash" if refl["free"] and refl["free_test_value"] is not None else "hash",
            "tune_fraction_of_work": 0.10,
            "final_fraction_if_hash": 0.10,
            "pair_friedel_when_nonanomalous": True,
            "anomalous": False,
        },
        "resolution": {"d_min_angstrom": float(refl["d_min"]), "d_max_angstrom": float(refl["d_max"])},
        "grid": {"shape": None, "samples_per_dmin": 3.0},
        "prior": {"kernel": "matern", "tau_density": 0.03, "correlation_length_angstrom": 0.75, "alpha": 2.5, "components": [], "remove_mean": True, "latent_distribution": "gaussian", "student_t_df": 4.0, "laplace_softening": 1e-3, "cauchy_scale": 1.0},
        "baseline": {"refmac_compatible_density_blur": False, "density_cutoff": 1e-6, "bulk_solvent": {"enabled": True, "radii": "refmac", "k_sol": 0.35, "b_sol_angstrom2": 46.0}, "scaling": {"enabled": True, "fit_isotropic_b_first": True}},
        "likelihood": {"global_scale": "profile", "fixed_scale": 1.0, "sigma_floor": 1e-6, "loss": "gaussian", "student_t_df": 6.0},
        "optimizer": {"method": "lbfgs", "fit_scope": "train", "max_iterations": 250, "tolerance": 5e-5, "lbfgs_memory": 8, "adam_learning_rate": 3e-3, "checkpoint_every": 10, "validation_patience": 8, "validation_min_delta": 1e-5},
        "information": {"n_modes": 16, "max_iterations": 80, "tolerance": 1e-4},
        "atomic_benchmark": {"enabled": True, "n_atoms": 8, "coordinate_delta_angstrom": 0.02, "b_delta_angstrom2": 0.5, "occupancy_delta": 0.02},
    }
    payload["input"]["expected_free_fraction_min"] = 0.02
    payload["input"]["expected_free_fraction_max"] = 0.15
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

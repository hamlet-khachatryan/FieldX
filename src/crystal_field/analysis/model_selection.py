from __future__ import annotations

import copy
import datetime
import hashlib
import itertools
import json
from pathlib import Path

import yaml

from crystal_field.config import AppConfig, dump_config, load_config
from crystal_field.model.prior import effective_parameters


def _slug(value):
    return str(value).replace(".", "p").replace("-", "m").replace(" ", "_")


# Axes a sweep may vary. multiscale_matern is excluded: its behaviour is set by a list
# of components, which is not a scalar axis, so it belongs under `candidates:`.
SWEEPABLE = (
    "kernel",
    "tau_density",
    "correlation_length_angstrom",
    "alpha",
    "latent_distribution",
    "student_t_df",
    "laplace_softening",
    "cauchy_scale",
)

KERNEL_ABBREVIATION = {"matern": "matern", "squared_exponential": "rbf", "bandlimited_white": "white"}

# Each candidate is an independent GPU job, so an unconstrained product is a way to
# accidentally queue hundreds of them. Exceeding this requires saying so in the grid file.
DEFAULT_MAX_CANDIDATES = 64


def _sweep_name(effective: dict) -> str:
    parts = [KERNEL_ABBREVIATION.get(effective["kernel"], effective["kernel"])]
    for key, prefix in (
        ("tau_density", "t"),
        ("correlation_length_angstrom", "l"),
        ("alpha", "a"),
    ):
        if effective.get(key) is not None:
            parts.append(f"{prefix}{_slug(effective[key])}")
    if effective["latent_distribution"] != "gaussian":
        parts.append(effective["latent_distribution"])
    return "_".join(parts)


def expand_sweep(sweep: dict, base_prior: dict) -> list[dict]:
    """Cartesian product over the declared axes, minus combinations that mean the same.

    alpha does nothing for a squared-exponential kernel, correlation length does nothing
    for band-limited white noise, and student_t_df does nothing under a Gaussian latent.
    Combinations differing only in such an inert parameter describe one operator and are
    collapsed to a single candidate, so "compute every mode" does not mean "fit the same
    prior repeatedly".
    """
    unknown = sorted(set(sweep) - set(SWEEPABLE))
    if unknown:
        raise ValueError(f"Cannot sweep {unknown}; sweepable axes are {list(SWEEPABLE)}")
    axes = {key: (value if isinstance(value, list) else [value]) for key, value in sweep.items()}
    for key, values in axes.items():
        if not values:
            raise ValueError(f"Sweep axis {key!r} is empty")

    keys = list(axes)
    candidates, seen = [], set()
    for combination in itertools.product(*(axes[key] for key in keys)):
        override = dict(zip(keys, combination, strict=True))
        prior = {**base_prior, **override}
        if prior.get("kernel") == "multiscale_matern":
            raise ValueError(
                "multiscale_matern cannot be swept: it is defined by a component list. "
                "Give it explicitly under `candidates:`."
            )
        effective = effective_parameters(prior)
        signature = json.dumps(effective, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append({"name": _sweep_name(effective), "prior": effective})
    return candidates


def expand_prior_grid(base_config: Path, grid_file: Path, output_dir: Path):
    """One candidate configuration per prior-grid entry, all fitted on train only."""
    base = load_config(base_config)
    grid_path = Path(grid_file)
    if not grid_path.is_file():
        raise FileNotFoundError(f"Prior grid not found: {grid_path}")
    spec = yaml.safe_load(grid_path.read_text()) or {}
    candidates = list(spec.get("candidates", []) or [])
    base_prior = base.prior.model_dump(mode="json")
    if spec.get("sweep"):
        candidates += expand_sweep(spec["sweep"], base_prior)
    if not candidates:
        raise ValueError(f"{grid_path} contains no candidates and no sweep")

    limit = int(spec.get("max_candidates", DEFAULT_MAX_CANDIDATES))
    if len(candidates) > limit:
        raise ValueError(
            f"{grid_path} expands to {len(candidates)} candidates, above max_candidates={limit}. "
            "Each candidate is an independent GPU job. Narrow the sweep, or raise "
            "max_candidates in the grid file deliberately."
        )
    names = [item.get("name") for item in candidates if item.get("name")]
    if len(set(names)) != len(names):
        raise ValueError(f"{grid_path} has duplicate candidate names: {sorted(names)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, item in enumerate(candidates):
        name = item.get("name") or f"candidate_{i:03d}"
        payload = base.model_dump(mode="json", exclude_none=True)
        prior_override = copy.deepcopy(item.get("prior", {}))
        # components only mean anything for multiscale_matern; a candidate that switches
        # away from it must not inherit the base configuration's component list.
        if prior_override.get("kernel") not in (None, "multiscale_matern") and "components" not in prior_override:
            prior_override["components"] = []
        payload["prior"].update(prior_override)
        payload["likelihood"].update(copy.deepcopy(item.get("likelihood", {})))
        payload["baseline"].update(copy.deepcopy(item.get("baseline", {})))
        payload["optimizer"]["fit_scope"] = "train"
        payload["run"]["shared_data_dir"] = str(base.run.data_dir)
        payload["run"]["output_dir"] = str(base.run.output_dir / "candidates" / name)
        cfg = AppConfig.model_validate(payload)
        path = output_dir / f"{i:03d}_{_slug(name)}.yaml"
        dump_config(cfg, path)
        manifest.append(
            {
                "index": i,
                "name": name,
                "config": str(path),
                "output_dir": str(cfg.run.output_dir),
                "prior": effective_parameters(cfg.prior),
            }
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"n_candidates": len(manifest), "manifest": str(manifest_path)}


def select_prior(base_config: Path, manifest_path: Path, selected_path: Path):
    base = load_config(base_config)
    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    for item in manifest:
        metrics_path = Path(item["output_dir"]) / "fit" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing candidate metrics: {metrics_path}")
        m = json.loads(metrics_path.read_text())
        score = float(m["chi2_tune"]) / max(float(m["n_tune"]), 1.0)
        rows.append(
            {
                **item,
                "score": score,
                "r_tune": float(m["r_tune"]),
                "r_train": float(m["r_train"]),
            }
        )
    rows.sort(key=lambda x: (x["score"], x["r_tune"]))
    winner = rows[0]
    winner_cfg = load_config(winner["config"])
    payload = winner_cfg.model_dump(mode="json", exclude_none=True)
    payload["optimizer"]["fit_scope"] = "work"
    payload["run"]["output_dir"] = str(base.run.output_dir / "final")
    payload["run"]["shared_data_dir"] = str(base.run.data_dir)
    selected = AppConfig.model_validate(payload)
    dump_config(selected, selected_path)
    report = {"winner": winner, "ranking": rows, "selected_config": str(selected_path)}
    report_path = selected_path.with_suffix(".selection.json")
    report_path.write_text(json.dumps(report, indent=2))
    return report


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# The free set is consumable: once it has been read, no amount of re-freezing restores it.
# The ledger therefore lives beside the prepared reflections rather than in a run output
# directory, so that evaluating, tweaking a hyperparameter, re-freezing into a fresh
# output_dir and re-evaluating is still caught. It is append-only.
FREE_SET_LEDGER_NAME = "FREE_SET_LEDGER.json"


def free_set_ledger_path(cfg) -> Path:
    return cfg.run.data_dir / FREE_SET_LEDGER_NAME


def read_free_set_ledger(cfg):
    path = free_set_ledger_path(cfg)
    if not path.exists():
        return []
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise RuntimeError(f"{path} is corrupt: expected a list of evaluation records")
    return entries


def append_free_set_ledger(cfg, entry):
    path = free_set_ledger_path(cfg)
    entries = read_free_set_ledger(cfg)
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))
    return entries


def _consumed_message(entries, action):
    first = entries[0]
    return (
        f"The deposited free set has already been read {len(entries)} time(s) for this prepared "
        f"split; first at {first.get('timestamp_utc')} under lock {first.get('lock_config_sha256', '')[:12]}. "
        f"The free-set evaluation is one-shot, so {action} would invalidate the held-out result. "
        f"See {FREE_SET_LEDGER_NAME}. Re-preparing reflections into a fresh data_dir creates a "
        "genuinely new split; overriding instead must be reported as a repeated evaluation."
    )


def freeze_model(config_path: Path, allow_after_free_evaluation: bool = False):
    cfg = load_config(config_path)
    if cfg.optimizer.fit_scope != "work":
        raise ValueError("Only a work-refit configuration can be frozen")
    consumed = read_free_set_ledger(cfg)
    if consumed and not allow_after_free_evaluation:
        raise RuntimeError(_consumed_message(consumed, "re-freezing a revised model"))
    required = [
        cfg.run.data_dir / "metadata.json",
        cfg.run.data_dir / "reflections.npz",
        cfg.run.data_dir / "rho0.npy",
        cfg.run.data_dir / "solvent_mask.npy",
        cfg.run.output_dir / "fit" / "z_map.npy",
        cfg.run.output_dir / "fit" / "metrics.json",
        # The information spectrum is part of the frozen record, so the DAG cannot be
        # short-circuited into freezing a model that was never characterised.
        cfg.run.output_dir / "info_spectrum.json",
    ]
    if cfg.baseline.scaling.enabled:
        required.append(cfg.run.data_dir / "scaling_work.npz")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    lock = {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "artifacts": {str(p): sha256_file(p) for p in required},
        # Authoritative record of whether the free set has been read; this file is only
        # written at freeze time, so it can never report a later evaluation itself.
        "free_set_ledger": str(free_set_ledger_path(cfg)),
        "free_set_used_at_freeze": bool(consumed),
        "has_free_set": cfg.has_free_set,
        "target": "R_work and R_free" if cfg.has_free_set else "R_work only (no held-out set)",
        "note": "Freeze created before free-set evaluation. Do not alter model or hyperparameters after this point.",
    }
    lock_path = cfg.run.output_dir / "MODEL_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2))
    return {"lock": str(lock_path), "config_sha256": lock["config_sha256"]}


def verify_lock(config_path: Path, allow_consumed: bool = False):
    cfg = load_config(config_path)
    lock_path = cfg.run.output_dir / "MODEL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError("MODEL_LOCK.json is required before free-set evaluation")
    lock = json.loads(lock_path.read_text())
    if lock["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("Configuration changed after MODEL_LOCK.json was created")
    for p, digest in lock["artifacts"].items():
        path = Path(p)
        if not path.exists() or sha256_file(path) != digest:
            raise RuntimeError(f"Frozen artifact changed after lock: {path}")
    consumed = read_free_set_ledger(cfg)
    if consumed and not allow_consumed:
        raise RuntimeError(_consumed_message(consumed, "another free-set evaluation"))
    lock["free_set_evaluations"] = consumed
    return lock


def record_free_evaluation(cfg, lock, result_path: Path, summary):
    """Append a free-set read to the append-only ledger. Called only after the result is written."""
    entries = read_free_set_ledger(cfg)
    entry = {
        "free_set_used": True,
        "evaluation_index": len(entries) + 1,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "lock_config_sha256": lock["config_sha256"],
        "config": str(lock["config"]),
        "result_file": str(result_path),
        "result_sha256": sha256_file(result_path),
        "reported": summary,
    }
    append_free_set_ledger(cfg, entry)
    return entry

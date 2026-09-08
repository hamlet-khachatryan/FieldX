from __future__ import annotations

import csv
import json

import numpy as np


def _scalar_dict(d):
    return {k: float(v) for k, v in d.items()}


def run_map_fit(cfg, arrays, objective, metrics, density_from_z):
    import jax
    import jax.numpy as jnp
    import optax

    out = cfg.run.output_dir / "fit"
    out.mkdir(parents=True, exist_ok=True)
    z = jnp.zeros_like(arrays.rho0)
    history = []
    objective_jit = jax.jit(objective)
    metrics_jit = jax.jit(metrics)

    if cfg.optimizer.method == "adam":
        opt = optax.adam(cfg.optimizer.adam_learning_rate)
        state = opt.init(z)
        value_grad = jax.jit(jax.value_and_grad(objective))
    else:
        opt = optax.lbfgs(memory_size=cfg.optimizer.lbfgs_memory)
        state = opt.init(z)
        value_grad = optax.value_and_grad_from_state(objective_jit)

    best_tune = np.inf
    best_z = None
    best_iteration = 0
    stale = 0

    for it in range(cfg.optimizer.max_iterations):
        if cfg.optimizer.method == "adam":
            value, grad = value_grad(z)
            updates, state = opt.update(grad, state, z)
        else:
            value, grad = value_grad(z, state=state)
            updates, state = opt.update(
                grad, state, z, value=value, grad=grad, value_fn=objective_jit
            )
        z = optax.apply_updates(z, updates)
        grad_norm = optax.tree.norm(grad)

        should_checkpoint = (
            it % cfg.optimizer.checkpoint_every == 0
            or it + 1 == cfg.optimizer.max_iterations
            or float(grad_norm) < cfg.optimizer.tolerance
        )
        if should_checkpoint:
            m = _scalar_dict(metrics_jit(z))
            row = {"iteration": it + 1, "value": float(value), "grad_norm": float(grad_norm), **m}
            history.append(row)
            np.save(out / "z_checkpoint.npy", np.asarray(jax.device_get(z), dtype=np.float32))

            # During prior tuning, use only the internal tune subset for early stopping.
            if cfg.optimizer.fit_scope == "train":
                tune_score = m["chi2_tune"] / max(m["n_tune"], 1.0)
                if tune_score < best_tune - cfg.optimizer.validation_min_delta:
                    best_tune = tune_score
                    best_z = np.asarray(jax.device_get(z), dtype=np.float32)
                    best_iteration = it + 1
                    stale = 0
                    np.save(out / "z_best_tune.npy", best_z)
                else:
                    stale += 1
                    if stale >= cfg.optimizer.validation_patience:
                        break

        if float(grad_norm) < cfg.optimizer.tolerance:
            break

    if cfg.optimizer.fit_scope == "train" and best_z is not None:
        z = jnp.asarray(best_z, dtype=arrays.rho0.dtype)

    final_metrics = _scalar_dict(metrics_jit(z))
    rho = density_from_z(z)
    np.save(out / "z_map.npy", np.asarray(jax.device_get(z), dtype=np.float32))
    np.save(out / "rho_map_raw.npy", np.asarray(jax.device_get(rho), dtype=np.float32))
    final_metrics["best_tune_iteration"] = int(best_iteration) if cfg.optimizer.fit_scope == "train" else None
    final_metrics["fit_scope"] = cfg.optimizer.fit_scope
    final_metrics["prior_kernel"] = cfg.prior.kernel
    final_metrics["latent_distribution"] = cfg.prior.latent_distribution
    (out / "metrics.json").write_text(json.dumps(final_metrics, indent=2))

    if history:
        with (out / "history.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
    return final_metrics

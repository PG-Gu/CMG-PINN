"""Shared infrastructure for the low-memory forward-PDE PINN tasks."""

from __future__ import annotations
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable
import deepxde as dde
import numpy as np
import torch
from src.activations.activation_baselines import (
    activation_options_from_config,
    is_activation_baseline,
)
from src.training.muon_common import build_muon_with_aux_adam
from src.training.task_train_common import (
    GLLFTracker,
    LossTracker,
    ReferenceMetricTracker,
    TrueTrainLossTracker,
    count_params,
    get_aux_param_groups,
    lbfgs_state_stats,
    save_selected_model_snapshot,
    set_seed,
    snapshot_activation_state,
    snapshot_gllf,
    temporary_lbfgs_options,
)
from src.models.network import GLLF_FNN
from src.models.pirate_mlp import PirateNetFNN

DEFAULT_PRECISION = "float32"
DEFAULT_LBFGS_ITERS = 15000
REFERENCE_INTERVAL = 100
TRUE_TRAIN_LOSS_INTERVAL = 100


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_npz_write(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _sampling_paths(task_key, seed, cfg):
    directory = Path(cfg["sampling_root"]) / task_key / f"seed_{seed}"
    return (directory / "points.npz", directory / "sampling.json")


def _sampling_spec(task_key: str, seed: int, cfg: dict) -> dict:
    spec = {
        "task_key": task_key,
        "seed": int(seed),
        "precision": cfg.get("precision", DEFAULT_PRECISION),
        "sampling_version": cfg.get("sampling_version", "low_memory_forward_v1"),
        "domain": cfg.get("sampling_domain", task_key),
        "point_counts": dict(cfg.get("point_counts", {})),
        "train_distribution": cfg.get("train_distribution", "Hammersley"),
    }
    replay_mode = cfg.get("sampling_replay_mode", "anchors")
    if replay_mode != "anchors":
        spec["sampling_replay_mode"] = replay_mode
    return spec


def _sampling_arrays(data) -> dict[str, np.ndarray]:
    arrays = {
        "train_x_all": np.asarray(data.train_x_all),
        "train_x_bc": np.asarray(data.train_x_bc),
        "train_x": np.asarray(data.train_x),
        "num_bcs": np.asarray(data.num_bcs, dtype=np.int64),
    }
    operator_layout = getattr(data, "low_memory_operator_layout_arrays", None)
    if callable(operator_layout):
        operator_layout = operator_layout()
    for name, value in (operator_layout or {}).items():
        arrays[f"operator_layout__{name}"] = np.asarray(value)
    return arrays


def _assert_sampling_layout(data, arrays: dict[str, np.ndarray]) -> None:
    actual = _sampling_arrays(data)
    for name, expected in arrays.items():
        if not np.array_equal(actual[name], expected):
            raise RuntimeError(
                f"Persisted DeepXDE sampling layout did not reproduce exactly: {name} differs. Refusing to train with a changed collocation set."
            )


def create_persisted_data(task_key, cfg, seed, data_factory):
    """Generate or replay the task's fixed collocation arrays."""
    dde.config.set_default_float("float32")
    points_path, spec_path = _sampling_paths(task_key, seed, cfg)
    spec = _sampling_spec(task_key, seed, cfg)
    if points_path.is_file() and spec_path.is_file():
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError(
                "Sampling settings changed; choose another output directory."
            )
        with np.load(points_path) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
        set_seed(seed)
        anchors = (
            None
            if cfg.get("sampling_replay_mode", "anchors") == "regenerate"
            else arrays["train_x_all"]
        )
        data = data_factory(anchors)
        _assert_sampling_layout(data, arrays)
    else:
        set_seed(seed)
        data = data_factory(None)
        arrays = _sampling_arrays(data)
        points_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_npz_write(points_path, arrays)
        _atomic_json_write(spec_path, spec)
    data.low_memory_sampling_manifest = {"spec": spec}
    return data


def prepare_sampling_config(task_key, cfg, seed, data_factory):
    create_persisted_data(task_key, cfg, seed, data_factory)
    return {}


def create_standard_network(
    cfg: dict,
    *,
    output_transform: Callable | None = None,
    feature_transform: Callable | None = None,
):
    """Instantiate the existing FNN/PirateNet implementations for a task."""
    activation_name = cfg["activation_name"]
    if activation_name == "fourier_features":
        from src.models.fourier_feature_network import FourierFeatureFNN

        net = FourierFeatureFNN(
            layer_sizes=cfg["layer_sizes"],
            kernel_initializer=cfg.get("kernel_initializer", "Glorot normal"),
            fourier_scales=cfg.get("fourier_scales", (1.0, 10.0)),
            fourier_seed=cfg.get("fourier_seed", cfg.get("seed")),
            fourier_features_per_branch=cfg.get("fourier_features_per_branch"),
            input_lower=cfg.get("fourier_input_lower"),
            input_upper=cfg.get("fourier_input_upper"),
        )
    elif activation_name == "pirate_mlp":
        net = PirateNetFNN(
            layer_sizes=cfg["layer_sizes"],
            kernel_initializer=cfg.get("kernel_initializer", "Glorot normal"),
        )
    else:
        net = GLLF_FNN(
            layer_sizes=cfg["layer_sizes"],
            activation_name=activation_name,
            kernel_initializer=cfg.get("kernel_initializer", "Glorot normal"),
            sharing=cfg.get("sharing", "none"),
            bounds_mode=cfg.get("bounds_mode", "none"),
            norm_mode=cfg.get("norm_mode", "none"),
            mu_init=cfg.get("mu_init", "N/A"),
            I_init=cfg.get("I_init", "N/A"),
            gllf_mode=cfg.get("gllf_mode", "all"),
            eps=cfg.get("eps", 1e-06),
            activation_options=activation_options_from_config(cfg),
        )
    if feature_transform is not None:
        net.apply_feature_transform(feature_transform)
    if output_transform is not None:
        net.apply_output_transform(output_transform)
    return net.float()


def _cuda_peaks(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return (None, None)
    torch.cuda.synchronize(device)
    return (
        float(torch.cuda.max_memory_allocated(device) / 1024**3),
        float(torch.cuda.max_memory_reserved(device) / 1024**3),
    )


def _mean_abs(value) -> float:
    if isinstance(value, (list, tuple)):
        pieces = [np.ravel(np.asarray(item)) for item in value]
        value = np.concatenate(pieces) if pieces else np.empty(0)
    return float(np.mean(np.abs(np.asarray(value))))


def _predict_in_chunks(model, X, *, operator=None, chunk_size=None):
    X = np.asarray(X)
    if not chunk_size or len(X) <= int(chunk_size):
        return model.predict(X, operator=operator)
    pieces = []
    for start in range(0, len(X), int(chunk_size)):
        pieces.append(
            model.predict(X[start : start + int(chunk_size)], operator=operator)
        )
    if isinstance(pieces[0], (list, tuple)):
        return [
            np.concatenate([np.asarray(piece[index]) for piece in pieces], axis=0)
            for index in range(len(pieces[0]))
        ]
    return np.concatenate([np.asarray(piece) for piece in pieces], axis=0)


def _metric_bundle(
    model,
    X_ref,
    y_ref,
    pde_operator,
    prediction_selector=None,
    chunk_size=None,
    metric_scales=None,
) -> tuple[dict, np.ndarray]:
    y_pred = _predict_in_chunks(model, X_ref, chunk_size=chunk_size)
    scored_prediction = (
        prediction_selector(y_pred) if prediction_selector is not None else y_pred
    )
    scored_reference = y_ref
    if metric_scales is not None:
        scales = np.asarray(metric_scales, dtype=np.float32).reshape(1, -1)
        if scales.shape[1] != scored_prediction.shape[1] or np.any(scales <= 0):
            raise ValueError(
                "metric_scales must contain one positive scale per scored output."
            )
        scored_prediction = scored_prediction / scales
        scored_reference = y_ref / scales
    residual = (
        None
        if pde_operator is None
        else _predict_in_chunks(
            model, X_ref, operator=pde_operator, chunk_size=chunk_size
        )
    )
    return (
        {
            "l2_relative_error": float(
                dde.metrics.l2_relative_error(scored_reference, scored_prediction)
            ),
            "test_mse": float(np.mean((scored_prediction - scored_reference) ** 2)),
            "max_abs_error": float(
                np.max(np.abs(scored_prediction - scored_reference))
            ),
            "mean_abs_pde_residual": None if residual is None else _mean_abs(residual),
        },
        y_pred,
    )


class _SelectedReferenceMetricTracker(dde.callbacks.Callback):
    """Reference tracker for mixed-form tasks whose primary output is a subset."""

    def __init__(self, X_ref, y_ref, prediction_selector, interval=100):
        super().__init__()
        self.X_ref = X_ref
        self.y_ref = y_ref
        self.prediction_selector = prediction_selector
        self.interval = int(interval)
        self.mse_records = []
        self.l2_records = []

    def on_epoch_end(self):
        step = int(self.model.train_state.step)
        if step % self.interval != 0 and step != 1:
            return
        prediction = self.prediction_selector(self.model.predict(self.X_ref))
        mse = float(np.mean((prediction - self.y_ref) ** 2))
        l2 = float(dde.metrics.l2_relative_error(self.y_ref, prediction))
        self.mse_records.append((step, mse))
        self.l2_records.append((step, l2))


def run_forward_task(
    *,
    task_key: str,
    cfg: dict,
    seed: int,
    create_data: Callable[[dict, int], object],
    create_network: Callable[[dict], torch.nn.Module],
    reference_bundle: Callable[[], dict],
    pde_operator: Callable,
    residual_evaluator: Callable[[object, object], float] | None = None,
    extra_metrics: (
        Callable[[object, np.ndarray, np.ndarray, np.ndarray], dict] | None
    ) = None,
    prediction_selector: Callable[[np.ndarray], np.ndarray] | None = None,
    loss_weights_resolver: Callable[[object, dict], list[float]] | None = None,
    verbose: bool = True,
    track_reference_metrics: bool = False,
    reference_interval: int = REFERENCE_INTERVAL,
    track_true_train_loss: bool = False,
    true_train_loss_interval: int = TRUE_TRAIN_LOSS_INTERVAL,
) -> dict:
    """Train one low-memory task under the fixed first-order then L-BFGS protocol."""
    if cfg.get("precision", DEFAULT_PRECISION) != DEFAULT_PRECISION:
        raise ValueError("Low-memory forward tasks require precision='float32'.")
    dde.config.set_default_float(DEFAULT_PRECISION)
    weight_lr = float(cfg.get("weight_lr", 0.001))
    activation_lr = float(cfg.get("activation_lr", weight_lr))
    first_order_iterations = int(cfg["first_order_iterations"])
    if tuple(cfg.get("adam_betas", (0.9, 0.999))) != (0.9, 0.999):
        raise ValueError(
            "The controlled suite currently supports only the recorded Adam betas (0.9, 0.999)."
        )
    set_seed(seed)
    data = create_data(cfg, seed)
    set_seed(seed)
    net = create_network(cfg).float()
    params = count_params(net)
    auxiliary_param_count = sum(
        (
            parameter.numel()
            for parameter in get_aux_param_groups(net, weight_lr, activation_lr)[1][
                "params"
            ]
        )
    )
    model = dde.Model(data, net)
    external_trainable_variables = list(
        getattr(data, "low_memory_external_trainable_variables", ())
    )
    effective_pde_operator = getattr(data, "low_memory_pde_operator", pde_operator)
    if not getattr(data, "supports_posthoc_residual_metric", True):
        effective_pde_operator = None
    reference = reference_bundle()
    X_ref = np.asarray(reference["X"], dtype=np.float32)
    y_ref = np.asarray(reference["y"], dtype=np.float32)
    metric_scales = reference.get("metric_scales")
    evaluation_chunk_size = int(
        reference.get("evaluation_chunk_size", cfg.get("evaluation_chunk_size", 0)) or 0
    )
    callbacks = [LossTracker(interval=100)]
    loss_tracker = callbacks[0]
    true_loss_tracker = None
    if track_true_train_loss or cfg.get("track_true_train_loss", False):
        true_loss_tracker = TrueTrainLossTracker(
            interval=int(cfg.get("true_train_loss_interval", true_train_loss_interval))
        )
        callbacks.append(true_loss_tracker)
    activation_tracker = None
    activation_name = cfg["activation_name"]
    if activation_name == "gllf" or is_activation_baseline(activation_name):
        activation_tracker = GLLFTracker(
            net,
            interval=100,
            state_fn=(
                snapshot_gllf
                if activation_name == "gllf"
                else snapshot_activation_state
            ),
        )
        callbacks.append(activation_tracker)
    resolved_loss_weights = cfg.get("loss_weights")
    if loss_weights_resolver is not None:
        resolved_loss_weights = list(loss_weights_resolver(model, cfg))
    if resolved_loss_weights is not None:
        resolved_loss_weights = [float(value) for value in resolved_loss_weights]
        if (
            not resolved_loss_weights
            or not np.isfinite(resolved_loss_weights).all()
            or any((value <= 0 for value in resolved_loss_weights))
        ):
            raise ValueError(
                "Resolved loss weights must be finite and strictly positive."
            )
    cfg["resolved_loss_weights"] = resolved_loss_weights
    reference_tracker = None
    if track_reference_metrics or cfg.get("track_reference_metrics", False):
        if prediction_selector is None:
            reference_tracker = ReferenceMetricTracker(
                X_ref, y_ref, interval=reference_interval
            )
        else:
            reference_tracker = _SelectedReferenceMetricTracker(
                X_ref, y_ref, prediction_selector, interval=reference_interval
            )
        callbacks.append(reference_tracker)
    compile_kwargs = {
        "loss_weights": resolved_loss_weights,
        "external_trainable_variables": external_trainable_variables,
    }
    optimizer = build_muon_with_aux_adam(
        net,
        weight_lr,
        activation_lr,
        extra_aux_params=external_trainable_variables,
        extra_aux_lr=weight_lr,
        matrix_policy=cfg.get("muon_matrix_policy", "backbone"),
    )
    model.compile(optimizer, **compile_kwargs)
    device = next(net.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    first_started = started
    _, train_state = model.train(
        iterations=first_order_iterations,
        callbacks=callbacks,
        display_every=1000 if verbose else first_order_iterations + 1,
    )
    first_wall = time.perf_counter() - first_started
    first_loss = float(np.sum(train_state.loss_train))
    first_order_consumed = int(train_state.step)
    first_allocated, first_reserved = _cuda_peaks(device)
    lbfgs_wall = 0.0
    lbfgs_loss = first_loss
    lbfgs_target = None
    lbfgs_consumed = None
    lbfgs_reason = None
    lbfgs_evals = None
    lbfgs_allocated = None
    lbfgs_reserved = None
    lbfgs_target = int(cfg.get("lbfgs_fixed_budget_target_iters", DEFAULT_LBFGS_ITERS))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with temporary_lbfgs_options(
        maxiter=lbfgs_target,
        iter_per_step=lbfgs_target,
        maxfun=max(lbfgs_target * 50, lbfgs_target + 1),
        fun_per_step=max(lbfgs_target * 50, lbfgs_target + 1),
        gtol=-1.0,
        ftol=-1.0,
    ):
        model.compile("L-BFGS", **compile_kwargs)
        previous_step = model.train_state.step
        lbfgs_started = time.perf_counter()
        _, train_state = model.train(
            callbacks=callbacks, display_every=1000 if verbose else 100000
        )
        lbfgs_wall = time.perf_counter() - lbfgs_started
    lbfgs_allocated, lbfgs_reserved = _cuda_peaks(device)
    state_iterations, lbfgs_evals = lbfgs_state_stats(model.opt)
    lbfgs_consumed = (
        int(state_iterations)
        if state_iterations is not None
        else int(train_state.step - previous_step)
    )
    lbfgs_reason = (
        "budget_exhausted" if lbfgs_consumed >= lbfgs_target else "ended_early"
    )
    lbfgs_loss = float(np.sum(train_state.loss_train))
    if cfg.get("require_exact_budget", False):
        if first_order_consumed != first_order_iterations:
            raise RuntimeError(
                f"First-order budget incomplete: consumed {first_order_consumed}, expected {first_order_iterations}."
            )
        if True and lbfgs_consumed != lbfgs_target:
            raise RuntimeError(
                f"L-BFGS budget incomplete: consumed {lbfgs_consumed}, expected {lbfgs_target}."
            )
    metrics, y_pred = _metric_bundle(
        model,
        X_ref,
        y_ref,
        effective_pde_operator,
        prediction_selector=prediction_selector,
        chunk_size=evaluation_chunk_size,
        metric_scales=metric_scales,
    )
    if residual_evaluator is not None:
        metrics["mean_abs_pde_residual"] = float(residual_evaluator(model, data))
    if extra_metrics is not None:
        metrics.update(extra_metrics(model, X_ref, y_ref, y_pred))
    task_extra_metrics = getattr(data, "low_memory_extra_metrics", None)
    if task_extra_metrics is not None:
        metrics.update(task_extra_metrics(model, X_ref, y_ref, y_pred))
    task_metric_override = getattr(data, "low_memory_metric_override", None)
    if task_metric_override is not None:
        metrics.update(task_metric_override(model, X_ref, y_ref, y_pred))
    sampling_manifest = data.low_memory_sampling_manifest
    final_mu = final_I = final_activation_state = None
    if activation_name == "gllf":
        snapshot = snapshot_gllf(net)
        final_mu = {name: state["mu"] for name, state in snapshot.items()}
        final_I = {name: state["I"] for name, state in snapshot.items()}
    elif is_activation_baseline(activation_name):
        final_activation_state = snapshot_activation_state(net)
    snapshot_external_parameters = {
        name: parameter
        for name, parameter in zip(
            getattr(data, "low_memory_external_parameter_names", []),
            external_trainable_variables,
        )
    }
    selected_model_snapshot = save_selected_model_snapshot(
        net, cfg, external_parameters=snapshot_external_parameters
    )
    all_allocated = [
        value for value in (first_allocated, lbfgs_allocated) if value is not None
    ]
    all_reserved = [
        value for value in (first_reserved, lbfgs_reserved) if value is not None
    ]
    result = {
        **metrics,
        "params": params,
        "auxiliary_params": auxiliary_param_count,
        "external_trainable_params": int(
            sum((p.numel() for p in external_trainable_variables))
        ),
        "external_trainable_parameter_names": list(
            getattr(data, "low_memory_external_parameter_names", [])
        ),
        "task_key": task_key,
        "method_tag": cfg.get("method_tag"),
        "setting_tag": cfg.get("setting_tag"),
        "optimizer_tag": cfg.get("optimizer_tag"),
        "optimizer_name": cfg.get("optimizer_name", cfg.get("optimizer_tag")),
        "architecture": list(cfg["layer_sizes"]),
        "kernel_initializer": cfg.get("kernel_initializer"),
        "point_counts": dict(cfg.get("point_counts", {})),
        "loss_weights": resolved_loss_weights,
        "loss_component_names": list(cfg.get("loss_component_names", [])),
        "final_loss_components": {
            name: float(value)
            for name, value in zip(
                cfg.get("loss_component_names", []),
                np.asarray(train_state.loss_train).ravel(),
            )
        },
        "primary_metric": cfg.get("primary_metric", "l2_relative_error"),
        "metric_identity": cfg.get(
            "metric_identity", cfg.get("primary_metric", "l2_relative_error")
        ),
        "precision": DEFAULT_PRECISION,
        "wall_time_sec": round(time.perf_counter() - started, 2),
        "first_order_wall_sec": round(first_wall, 2),
        "muon_wall_sec": round(first_wall, 2),
        "lbfgs_wall_sec": round(lbfgs_wall, 2),
        "first_order_final_loss": first_loss,
        "muon_final_loss": first_loss,
        "lbfgs_final_loss": lbfgs_loss,
        "first_order_optimizer": "muon",
        "first_order_budget_target_iters": first_order_iterations,
        "first_order_budget_consumed_iters": first_order_consumed,
        "post_first_order_mode": "lbfgs_fixed_budget",
        "lbfgs_budget_target_iters": lbfgs_target,
        "lbfgs_budget_consumed_iters": lbfgs_consumed,
        "lbfgs_budget_stop_reason": lbfgs_reason,
        "lbfgs_func_evals": lbfgs_evals,
        "lbfgs_single_optimizer": True,
        "lbfgs_restart_count": 1,
        "budget_complete": first_order_consumed == first_order_iterations
        and (False or lbfgs_consumed == lbfgs_target),
        "first_order_peak_cuda_allocated_gb": first_allocated,
        "first_order_peak_cuda_reserved_gb": first_reserved,
        "lbfgs_peak_cuda_allocated_gb": lbfgs_allocated,
        "lbfgs_peak_cuda_reserved_gb": lbfgs_reserved,
        "peak_cuda_allocated_gb": max(all_allocated) if all_allocated else None,
        "peak_cuda_reserved_gb": max(all_reserved) if all_reserved else None,
        "loss_history": loss_tracker.records,
        "final_mu": final_mu,
        "final_I": final_I,
        "fourier_feature_manifest": (
            net.implementation_manifest()
            if hasattr(net, "implementation_manifest")
            else None
        ),
        "final_activation_state": final_activation_state,
        "selected_model_snapshot": selected_model_snapshot,
        "sampling_manifest_spec": sampling_manifest["spec"],
        "analytic_reference_id": cfg.get("analytic_reference_id", task_key),
        "config": dict(cfg),
        "seed": int(seed),
    }
    return result

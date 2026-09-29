from __future__ import annotations
import time
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
    TrueTrainLossTracker,
    count_params,
    lbfgs_state_stats,
    save_selected_model_snapshot,
    set_seed,
    snapshot_gllf,
    snapshot_activation_state,
    temporary_lbfgs_options,
)
from src.models.network import GLLF_FNN
from src.models.pirate_mlp import PirateNetFNN

DEFAULT_FIRST_ORDER_ITERATIONS = 50000
DEFAULT_LBFGS_ITERS = 15000
DEFAULT_PRECISION = "float32"
REFERENCE_GRID_SIZE = 100
REFERENCE_INTERVAL = 100
TRUE_TRAIN_LOSS_INTERVAL = 100
TRUE_DIFFUSIVITY = 1.0
INITIAL_DIFFUSIVITY = 2.0
OBSERVATION_COUNT = 10


def _solution(x):
    return np.exp(-x[:, 1:2]) * np.sin(np.pi * x[:, 0:1])


def _set_precision(cfg):
    precision = cfg.get("precision", DEFAULT_PRECISION)
    if precision not in {"float32", "float64"}:
        raise ValueError(f"Unsupported precision: {precision!r}")
    dde.config.set_default_float(precision)
    return precision


def _dtype_for_precision(precision):
    return np.float64 if precision == "float64" else np.float32


def _cast_network(net, precision):
    return net.double() if precision == "float64" else net.float()


def _cast_data_arrays(data, precision):
    dtype = _dtype_for_precision(precision)
    for name in (
        "train_x",
        "train_y",
        "test_x",
        "test_y",
        "train_x_all",
        "train_x_bc",
        "train_aux_vars",
        "test_aux_vars",
    ):
        value = getattr(data, name, None)
        if isinstance(value, np.ndarray):
            setattr(data, name, value.astype(dtype, copy=False))
        elif isinstance(value, (list, tuple)):
            converted = [
                item.astype(dtype, copy=False) if isinstance(item, np.ndarray) else item
                for item in value
            ]
            setattr(data, name, type(value)(converted))
    return data


def create_data(diffusivity, *, precision=DEFAULT_PRECISION):
    """Create inverse-diffusion data with fixed noiseless observations."""
    dtype = _dtype_for_precision(precision)
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        source = torch.exp(-x[:, 1:2]) * (
            torch.sin(torch.pi * x[:, 0:1])
            - torch.pi**2 * torch.sin(torch.pi * x[:, 0:1])
        )
        return dy_t - diffusivity * dy_xx + source

    bc = dde.icbc.DirichletBC(
        geomtime, lambda x: 0.0, lambda _, on_boundary: on_boundary
    )
    ic = dde.icbc.IC(
        geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
    )
    observation_x = np.hstack(
        (geom.random_points(OBSERVATION_COUNT), np.ones((OBSERVATION_COUNT, 1)))
    ).astype(dtype)
    observation_y = _solution(observation_x).astype(dtype)
    observation_bc = dde.icbc.PointSetBC(observation_x, observation_y, component=0)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic, observation_bc],
        num_domain=40,
        num_boundary=20,
        num_initial=10,
        num_test=10000,
    )
    return (data, pde, observation_x, observation_y)


def create_network(cfg):
    if cfg["activation_name"] == "fourier_features":
        from src.models.fourier_feature_network import FourierFeatureFNN

        return FourierFeatureFNN(
            cfg["layer_sizes"],
            kernel_initializer=cfg.get("kernel_initializer", "Glorot uniform"),
            fourier_scales=cfg.get("fourier_scales", (1.0, 10.0)),
            fourier_seed=cfg.get("fourier_seed", cfg.get("seed")),
            fourier_features_per_branch=cfg.get("fourier_features_per_branch"),
            input_lower=cfg.get("fourier_input_lower"),
            input_upper=cfg.get("fourier_input_upper"),
        )
    residual_mode = "none"
    if cfg["activation_name"] == "pirate_mlp":
        return PirateNetFNN(
            layer_sizes=cfg["layer_sizes"],
            kernel_initializer=cfg.get("kernel_initializer", "Glorot uniform"),
        )
    return GLLF_FNN(
        layer_sizes=cfg["layer_sizes"],
        activation_name=cfg["activation_name"],
        kernel_initializer=cfg.get("kernel_initializer", "Glorot uniform"),
        sharing=cfg.get("sharing", "neuron"),
        bounds_mode=cfg.get("bounds_mode", "tanh_wrapper"),
        mu_init=cfg.get("mu_init", 0.5),
        I_init=cfg.get("I_init", 0.5),
        gllf_mode=cfg.get("gllf_mode", "all"),
        eps=cfg.get("eps", 1e-06),
        activation_options=activation_options_from_config(cfg),
    )


def _reference_grid(precision):
    dtype = _dtype_for_precision(precision)
    x = np.linspace(-1, 1, REFERENCE_GRID_SIZE, dtype=dtype)
    t = np.linspace(0, 1, REFERENCE_GRID_SIZE, dtype=dtype)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    X = np.column_stack((xx.ravel(), tt.ravel())).astype(dtype)
    return (X, _solution(X).astype(dtype))


class _InverseReferenceTracker(dde.callbacks.Callback):
    """Track coefficient, solution-grid, and fixed-observation metrics."""

    def __init__(
        self, X_ref, y_ref, observation_x, observation_y, diffusivity, interval=100
    ):
        super().__init__()
        self.X_ref = X_ref
        self.y_ref = y_ref
        self.observation_x = observation_x
        self.observation_y = observation_y
        self.diffusivity = diffusivity
        self.interval = interval
        self.solution_l2_records = []
        self.coefficient_l2_records = []
        self.mse_records = []
        self.observation_mse_records = []
        self.observation_l2_records = []

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.interval != 0 and step != 1:
            return
        prediction = self.model.predict(self.X_ref)
        solution_l2 = float(dde.metrics.l2_relative_error(self.y_ref, prediction))
        self.mse_records.append(
            (int(step), float(np.mean((prediction - self.y_ref) ** 2)))
        )
        observation_prediction = self.model.predict(self.observation_x)
        observation_mse = float(
            np.mean((observation_prediction - self.observation_y) ** 2)
        )
        observation_l2 = float(
            dde.metrics.l2_relative_error(self.observation_y, observation_prediction)
        )
        recovered = float(self.diffusivity.detach().cpu().item())
        coefficient_error = abs(recovered - TRUE_DIFFUSIVITY) / abs(TRUE_DIFFUSIVITY)
        self.solution_l2_records.append((int(step), solution_l2))
        self.coefficient_l2_records.append((int(step), float(coefficient_error)))
        self.observation_mse_records.append((int(step), observation_mse))
        self.observation_l2_records.append((int(step), observation_l2))


def run_single(
    cfg,
    seed,
    verbose=True,
    track_reference_metrics=False,
    reference_interval=REFERENCE_INTERVAL,
    track_true_train_loss=False,
    true_train_loss_interval=TRUE_TRAIN_LOSS_INTERVAL,
):
    weight_lr = float(cfg.get("weight_lr", 0.001))
    activation_lr = float(cfg.get("activation_lr", weight_lr))
    first_order_iterations = int(
        cfg.get("first_order_iterations", DEFAULT_FIRST_ORDER_ITERATIONS)
    )
    precision = _set_precision(cfg)
    set_seed(seed)
    diffusivity = dde.Variable(
        float(cfg.get("initial_diffusivity", INITIAL_DIFFUSIVITY))
    )
    data, pde, observation_x, observation_y = create_data(
        diffusivity, precision=precision
    )
    data = _cast_data_arrays(data, precision)
    set_seed(seed)
    net = _cast_network(create_network(cfg), precision)
    network_params = count_params(net)
    total_params = dict(network_params)
    total_params["total"] += 1
    total_params["trainable"] += 1
    total_params["external_trainable"] = 1
    model = dde.Model(data, net)
    X_ref, y_ref = _reference_grid(precision)
    callbacks = [LossTracker(interval=100)]
    true_loss_tracker = None
    if track_true_train_loss or cfg.get("track_true_train_loss", False):
        true_loss_tracker = TrueTrainLossTracker(interval=true_train_loss_interval)
        callbacks.append(true_loss_tracker)
    gllf_tracker = None
    if cfg["activation_name"] == "gllf" or is_activation_baseline(
        cfg["activation_name"]
    ):
        gllf_tracker = GLLFTracker(
            net,
            interval=100,
            state_fn=(
                snapshot_gllf
                if cfg["activation_name"] == "gllf"
                else snapshot_activation_state
            ),
        )
        callbacks.append(gllf_tracker)
    reference_tracker = None
    if track_reference_metrics or cfg.get("track_reference_metrics", False):
        reference_tracker = _InverseReferenceTracker(
            X_ref,
            y_ref,
            observation_x,
            observation_y,
            diffusivity,
            interval=reference_interval,
        )
        callbacks.append(reference_tracker)
    compile_kwargs = {"external_trainable_variables": [diffusivity]}
    optimizer = build_muon_with_aux_adam(
        net,
        weight_lr,
        activation_lr,
        extra_aux_params=[diffusivity],
        extra_aux_lr=weight_lr,
        matrix_policy=cfg.get("muon_matrix_policy", "backbone"),
    )
    model.compile(optimizer, **compile_kwargs)
    total_start = time.perf_counter()
    first_order_start = total_start
    _, train_state = model.train(
        iterations=first_order_iterations,
        callbacks=callbacks,
        display_every=1000 if verbose else first_order_iterations + 1,
    )
    first_order_wall = time.perf_counter() - first_order_start
    first_order_final_loss = float(np.sum(train_state.loss_train))
    muon_wall = first_order_wall
    lbfgs_wall = 0.0
    lbfgs_final_loss = first_order_final_loss
    lbfgs_budget_target_iters = None
    lbfgs_budget_consumed_iters = None
    lbfgs_budget_stop_reason = None
    lbfgs_func_evals = None
    lbfgs_single_optimizer = None
    lbfgs_restart_count = None
    lbfgs_budget_target_iters = int(
        cfg.get("lbfgs_fixed_budget_target_iters", DEFAULT_LBFGS_ITERS)
    )
    lbfgs_single_optimizer = True
    lbfgs_restart_count = 1
    with temporary_lbfgs_options(
        maxiter=lbfgs_budget_target_iters,
        iter_per_step=lbfgs_budget_target_iters,
        maxfun=max(lbfgs_budget_target_iters * 50, lbfgs_budget_target_iters + 1),
        fun_per_step=max(lbfgs_budget_target_iters * 50, lbfgs_budget_target_iters + 1),
        gtol=-1.0,
        ftol=-1.0,
    ):
        model.compile("L-BFGS", **compile_kwargs)
        previous_step = model.train_state.step
        lbfgs_start = time.perf_counter()
        _, train_state = model.train(
            callbacks=callbacks, display_every=1000 if verbose else 100000
        )
        lbfgs_wall = time.perf_counter() - lbfgs_start
    state_n_iter, lbfgs_func_evals = lbfgs_state_stats(model.opt)
    lbfgs_budget_consumed_iters = (
        int(state_n_iter)
        if state_n_iter is not None
        else int(train_state.step - previous_step)
    )
    lbfgs_budget_stop_reason = (
        "budget_exhausted"
        if lbfgs_budget_consumed_iters >= lbfgs_budget_target_iters
        else "ended_early"
    )
    lbfgs_final_loss = float(np.sum(train_state.loss_train))
    prediction = model.predict(X_ref)
    solution_l2 = float(dde.metrics.l2_relative_error(y_ref, prediction))
    observation_prediction = model.predict(observation_x)
    observation_mse = float(np.mean((observation_prediction - observation_y) ** 2))
    observation_l2 = float(
        dde.metrics.l2_relative_error(observation_y, observation_prediction)
    )
    recovered = float(diffusivity.detach().cpu().item())
    coefficient_error = abs(recovered - TRUE_DIFFUSIVITY) / abs(TRUE_DIFFUSIVITY)
    residual = model.predict(X_ref, operator=pde)
    final_mu = None
    final_I = None
    final_activation_state = None
    if cfg["activation_name"] == "gllf":
        snapshot = snapshot_gllf(net)
        final_mu = {key: value["mu"] for key, value in snapshot.items()}
        final_I = {key: value["I"] for key, value in snapshot.items()}
    elif is_activation_baseline(cfg["activation_name"]):
        final_activation_state = snapshot_activation_state(net)
    selected_model_snapshot = save_selected_model_snapshot(
        net, cfg, external_parameters={"diffusivity": diffusivity}
    )
    return {
        "fourier_feature_manifest": (
            net.implementation_manifest()
            if hasattr(net, "implementation_manifest")
            else None
        ),
        "selected_model_snapshot": selected_model_snapshot,
        "l2_relative_error": float(coefficient_error),
        "diffusivity_relative_error": float(coefficient_error),
        "recovered_diffusivity": recovered,
        "true_diffusivity": TRUE_DIFFUSIVITY,
        "solution_l2_relative_error": solution_l2,
        "observation_mse": observation_mse,
        "observation_l2_relative_error": observation_l2,
        "test_mse": float(np.mean((prediction - y_ref) ** 2)),
        "mean_abs_pde_residual": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(prediction - y_ref))),
        "params": total_params,
        "network_params": network_params,
        "external_trainable_params": 1,
        "observation_count": OBSERVATION_COUNT,
        "observation_noise_std": 0.0,
        "observation_x": observation_x.tolist(),
        "observation_y": observation_y.tolist(),
        "wall_time_sec": round(time.perf_counter() - total_start, 2),
        "first_order_wall_sec": round(first_order_wall, 2),
        "muon_wall_sec": round(muon_wall, 2),
        "lbfgs_wall_sec": round(lbfgs_wall, 2),
        "first_order_final_loss": first_order_final_loss,
        "muon_final_loss": first_order_final_loss,
        "lbfgs_final_loss": lbfgs_final_loss,
        "first_order_optimizer": "muon",
        "post_first_order_mode": "lbfgs_fixed_budget",
        "lbfgs_budget_target_iters": lbfgs_budget_target_iters,
        "lbfgs_budget_consumed_iters": lbfgs_budget_consumed_iters,
        "lbfgs_budget_stop_reason": lbfgs_budget_stop_reason,
        "lbfgs_func_evals": lbfgs_func_evals,
        "lbfgs_single_optimizer": lbfgs_single_optimizer,
        "lbfgs_restart_count": lbfgs_restart_count,
        "loss_history": callbacks[0].records,
        "solution_l2_history": (
            reference_tracker.solution_l2_records if reference_tracker else None
        ),
        "observation_mse_history": (
            reference_tracker.observation_mse_records if reference_tracker else None
        ),
        "observation_l2_history": (
            reference_tracker.observation_l2_records if reference_tracker else None
        ),
        "final_mu": final_mu,
        "final_I": final_I,
        "final_activation_state": final_activation_state,
        "config": dict(cfg),
        "seed": seed,
    }

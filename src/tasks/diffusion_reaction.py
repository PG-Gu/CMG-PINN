import time
import deepxde as dde
import numpy as np
import torch
from src.activations.activation_baselines import (
    activation_options_from_config,
    is_activation_baseline,
)
from src.training.task_train_common import (
    GLLFTracker,
    LossTracker,
    ReferenceMetricTracker,
    TrueTrainLossTracker,
    count_params,
    evaluate_reference_suite,
    lbfgs_state_stats,
    save_selected_model_snapshot,
    set_seed,
    snapshot_gllf,
    snapshot_activation_state,
    temporary_lbfgs_options,
)
from src.training.muon_common import build_muon_with_aux_adam
from src.models.network import GLLF_FNN
from src.models.pirate_mlp import PirateNetFNN

DEFAULT_FIRST_ORDER_ITERATIONS = 20000
DEFAULT_LBFGS_ITERS = 15000
REFERENCE_INTERVAL = 100
TRUE_TRAIN_LOSS_INTERVAL = 100
PI = np.pi


def _forcing_torch(x):
    return torch.exp(-x[:, 1:2]) * (
        3 * torch.sin(2 * x[:, 0:1]) / 2
        + 8 * torch.sin(3 * x[:, 0:1]) / 3
        + 15 * torch.sin(4 * x[:, 0:1]) / 4
        + 63 * torch.sin(8 * x[:, 0:1]) / 8
    )


def _pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - dy_xx - _forcing_torch(x)


def exact_solution(x):
    return np.exp(-x[:, 1:2]) * (
        np.sin(x[:, 0:1])
        + np.sin(2 * x[:, 0:1]) / 2
        + np.sin(3 * x[:, 0:1]) / 3
        + np.sin(4 * x[:, 0:1]) / 4
        + np.sin(8 * x[:, 0:1]) / 8
    )


def create_data():
    geom = dde.geometry.Interval(-PI, PI)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    return dde.data.TimePDE(
        geomtime, _pde, [], num_domain=320, solution=exact_solution, num_test=80000
    )


def _output_transform(inputs, outputs):
    x = inputs[:, 0:1]
    t = inputs[:, 1:2]
    return (
        t * (PI**2 - x**2) * outputs
        + torch.sin(x)
        + torch.sin(2 * x) / 2
        + torch.sin(3 * x) / 3
        + torch.sin(4 * x) / 4
        + torch.sin(8 * x) / 8
    )


def create_reference_bundle(nx=401, nt=201):
    x = np.linspace(-PI, PI, nx, dtype=np.float64)
    t = np.linspace(0, 1, nt, dtype=np.float64)
    xx, tt = np.meshgrid(x, t)
    X = np.column_stack([xx.ravel(), tt.ravel()])
    y = exact_solution(X)
    return {"x": x, "t": t, "X": X, "y": y}


def create_network(cfg):
    act = cfg["activation_name"]
    if act == "fourier_features":
        from src.models.fourier_feature_network import FourierFeatureFNN

        net = FourierFeatureFNN(
            cfg["layer_sizes"],
            kernel_initializer=cfg.get("kernel_initializer", "Glorot uniform"),
            fourier_scales=cfg.get("fourier_scales", (1.0, 10.0)),
            fourier_seed=cfg.get("fourier_seed", cfg.get("seed")),
            fourier_features_per_branch=cfg.get("fourier_features_per_branch"),
            input_lower=cfg.get("fourier_input_lower"),
            input_upper=cfg.get("fourier_input_upper"),
        )
        net.apply_output_transform(_output_transform)
        return net
    residual_mode = "none"
    if act == "pirate_mlp":
        net = PirateNetFNN(
            layer_sizes=cfg["layer_sizes"],
            kernel_initializer=cfg.get("kernel_initializer", "Glorot uniform"),
        )
    else:
        net = GLLF_FNN(
            layer_sizes=cfg["layer_sizes"],
            activation_name=act,
            kernel_initializer=cfg.get("kernel_initializer", "Glorot uniform"),
            sharing=cfg.get("sharing", "neuron"),
            bounds_mode=cfg.get("bounds_mode", "tanh_wrapper"),
            mu_init=cfg.get("mu_init", 0.5),
            I_init=cfg.get("I_init", 0.5),
            gllf_mode=cfg.get("gllf_mode", "all"),
            eps=cfg.get("eps", 1e-06),
            activation_options=activation_options_from_config(cfg),
        )
    net.apply_output_transform(_output_transform)
    return net


def run_single(
    cfg,
    seed,
    verbose=True,
    track_reference_metrics=False,
    reference_interval=REFERENCE_INTERVAL,
    track_true_train_loss=False,
    true_train_loss_interval=TRUE_TRAIN_LOSS_INTERVAL,
):
    weight_lr = cfg.get("weight_lr", 0.001)
    activation_lr = cfg.get("activation_lr", weight_lr)
    first_order_iterations = cfg.get("first_order_iterations", DEFAULT_FIRST_ORDER_ITERATIONS)
    track_true_train_loss = track_true_train_loss or cfg.get(
        "track_true_train_loss", False
    )
    true_train_loss_interval = cfg.get(
        "true_train_loss_interval", true_train_loss_interval
    )
    set_seed(seed)
    data = create_data()
    set_seed(seed)
    net = create_network(cfg)
    params = count_params(net)
    model = dde.Model(data, net)
    ref_bundle = create_reference_bundle()
    X_ref, y_ref = (ref_bundle["X"], ref_bundle["y"])
    callbacks = []
    loss_tracker = LossTracker(interval=100)
    callbacks.append(loss_tracker)
    true_loss_tracker = None
    if track_true_train_loss:
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
    ref_tracker = None
    if track_reference_metrics:
        ref_tracker = ReferenceMetricTracker(X_ref, y_ref, interval=reference_interval)
        callbacks.append(ref_tracker)
    optimizer = build_muon_with_aux_adam(
        net, weight_lr, activation_lr, matrix_policy=cfg.get("muon_matrix_policy", "backbone")
    )
    model.compile(optimizer, metrics=["l2 relative error"])
    t0 = time.perf_counter()
    first_order_start = t0
    _, train_state = model.train(
        iterations=first_order_iterations,
        callbacks=callbacks,
        display_every=1000 if verbose else first_order_iterations + 1,
    )
    first_order_wall = time.perf_counter() - first_order_start
    first_order_final_loss = float(np.sum(train_state.loss_train))
    muon_wall = first_order_wall
    muon_final_loss = first_order_final_loss
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
        model.compile("L-BFGS", metrics=["l2 relative error"])
        prev_step = model.train_state.step
        lbfgs_start = time.perf_counter()
        _, train_state = model.train(
            display_every=1000 if verbose else 100000, callbacks=callbacks
        )
        lbfgs_wall = time.perf_counter() - lbfgs_start
    state_n_iter, lbfgs_func_evals = lbfgs_state_stats(model.opt)
    lbfgs_budget_consumed_iters = (
        int(state_n_iter)
        if state_n_iter is not None
        else int(train_state.step - prev_step)
    )
    lbfgs_budget_stop_reason = (
        "budget_exhausted"
        if lbfgs_budget_consumed_iters >= lbfgs_budget_target_iters
        else "ended_early"
    )
    lbfgs_final_loss = float(np.sum(train_state.loss_train))
    metrics = evaluate_reference_suite(model, X_ref, y_ref, _pde)
    final_mu = None
    final_I = None
    final_activation_state = None
    if cfg["activation_name"] == "gllf":
        snap = snapshot_gllf(net)
        final_mu = {key: value["mu"] for key, value in snap.items()}
        final_I = {key: value["I"] for key, value in snap.items()}
    elif is_activation_baseline(cfg["activation_name"]):
        final_activation_state = snapshot_activation_state(net)
    selected_model_snapshot = save_selected_model_snapshot(net, cfg)
    return {
        "fourier_feature_manifest": (
            net.implementation_manifest()
            if hasattr(net, "implementation_manifest")
            else None
        ),
        "selected_model_snapshot": selected_model_snapshot,
        "l2_relative_error": metrics["l2_relative_error"],
        "test_mse": metrics["test_mse"],
        "mean_abs_pde_residual": metrics["mean_abs_pde_residual"],
        "max_abs_error": metrics["max_abs_error"],
        "params": params,
        "wall_time_sec": round(time.perf_counter() - t0, 2),
        "first_order_wall_sec": round(first_order_wall, 2),
        "muon_wall_sec": round(muon_wall, 2),
        "lbfgs_wall_sec": round(lbfgs_wall, 2),
        "first_order_final_loss": first_order_final_loss,
        "muon_final_loss": muon_final_loss,
        "lbfgs_final_loss": lbfgs_final_loss,
        "first_order_optimizer": "muon",
        "post_first_order_mode": "lbfgs_fixed_budget",
        "lbfgs_budget_target_iters": lbfgs_budget_target_iters,
        "lbfgs_budget_consumed_iters": lbfgs_budget_consumed_iters,
        "lbfgs_budget_stop_reason": lbfgs_budget_stop_reason,
        "lbfgs_func_evals": lbfgs_func_evals,
        "lbfgs_single_optimizer": lbfgs_single_optimizer,
        "lbfgs_restart_count": lbfgs_restart_count,
        "loss_history": loss_tracker.records,
        "final_mu": final_mu,
        "final_I": final_I,
        "final_activation_state": final_activation_state,
        "config": dict(cfg),
        "seed": seed,
    }

from src.paths import DATA_DIR as BUNDLED_DATA
import time
import deepxde as dde
import numpy as np
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
    evaluate_reference_suite,
    lbfgs_state_stats,
    save_selected_model_snapshot,
    set_seed,
    snapshot_gllf,
    snapshot_activation_state,
    temporary_lbfgs_options,
)
from src.models.network import GLLF_FNN
from src.models.pirate_mlp import PirateNetFNN

POISSON_LSHAPE_NPZ = BUNDLED_DATA / "Poisson_Lshape.npz"
DEFAULT_FIRST_ORDER_ITERATIONS = 50000
DEFAULT_LBFGS_ITERS = 15000
DEFAULT_PRECISION = "float32"
REFERENCE_INTERVAL = 100
TRUE_TRAIN_LOSS_INTERVAL = 100


def _pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx - dy_yy - 1


def _boundary(_, on_boundary):
    return on_boundary


def create_data():
    geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, _boundary)
    return dde.data.PDE(
        geom, _pde, bc, num_domain=1200, num_boundary=120, num_test=1500
    )


def load_reference_bundle():
    if not POISSON_LSHAPE_NPZ.exists():
        raise FileNotFoundError(f"Missing reference data: {POISSON_LSHAPE_NPZ}")
    data = np.load(POISSON_LSHAPE_NPZ)
    X = data["X_test"]
    y = data["y_ref"]
    mask = np.isfinite(y[:, 0])
    return {"X": X[mask], "y": y[mask], "finite_mask_count": int(np.sum(mask))}


def create_network(cfg):
    act = cfg["activation_name"]
    if act == "fourier_features":
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
    if act == "pirate_mlp":
        return PirateNetFNN(
            layer_sizes=cfg["layer_sizes"],
            kernel_initializer=cfg.get("kernel_initializer", "Glorot uniform"),
        )
    return GLLF_FNN(
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


def _set_precision(cfg):
    precision = cfg.get("precision", DEFAULT_PRECISION)
    if precision not in {"float32", "float64"}:
        raise ValueError(f"Unsupported precision: {precision!r}")
    dde.config.set_default_float(precision)
    return precision


def _cast_network(net, precision):
    if precision == "float64":
        return net.double()
    return net.float()


def _cast_data_arrays(data, precision):
    dtype = np.float64 if precision == "float64" else np.float32
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
    first_order_iterations = int(cfg.get("first_order_iterations", DEFAULT_FIRST_ORDER_ITERATIONS))
    track_true_train_loss = track_true_train_loss or cfg.get(
        "track_true_train_loss", False
    )
    true_train_loss_interval = cfg.get(
        "true_train_loss_interval", true_train_loss_interval
    )
    precision = _set_precision(cfg)
    set_seed(seed)
    data = create_data()
    data = _cast_data_arrays(data, precision)
    set_seed(seed)
    net = _cast_network(create_network(cfg), precision)
    params = count_params(net)
    model = dde.Model(data, net)
    ref_bundle = load_reference_bundle()
    X_ref, y_ref = (ref_bundle["X"], ref_bundle["y"])
    callbacks = [LossTracker(interval=100)]
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
    model.compile(optimizer)
    t0 = time.perf_counter()
    _, train_state = model.train(
        iterations=first_order_iterations,
        callbacks=callbacks,
        display_every=1000 if verbose else first_order_iterations + 1,
    )
    first_order_wall = time.perf_counter() - t0
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
        model.compile("L-BFGS")
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
        "poisson_lshape_reference_points": ref_bundle["finite_mask_count"],
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
        "loss_history": callbacks[0].records,
        "final_mu": final_mu,
        "final_I": final_I,
        "final_activation_state": final_activation_state,
        "config": dict(cfg),
        "seed": seed,
    }

import math
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
    TrueTrainLossTracker,
    count_params,
    lbfgs_state_stats,
    save_selected_model_snapshot,
    set_seed,
    snapshot_gllf,
    snapshot_activation_state,
    temporary_lbfgs_options,
)
from src.training.muon_common import build_muon_with_aux_adam
from src.training.low_memory_forward_common import (
    create_persisted_data,
    prepare_sampling_config,
)
from src.models.network import GLLF_FNN
from src.models.pirate_mlp import PirateNetFNN

TASK_KEY = "kovasznay_flow"
DEFAULT_FIRST_ORDER_ITERATIONS = 30000
DEFAULT_LBFGS_ITERS = 15000
DEFAULT_PRECISION = "float32"
REFERENCE_INTERVAL = 100
TRUE_TRAIN_LOSS_INTERVAL = 100
REYNOLDS = 20.0
NU = 1.0 / REYNOLDS
LAMBDA = 1.0 / (2.0 * NU) - math.sqrt(1.0 / (4.0 * NU**2) + 4.0 * math.pi**2)


def _u_exact(x):
    return 1.0 - np.exp(LAMBDA * x[:, 0:1]) * np.cos(2.0 * np.pi * x[:, 1:2])


def _v_exact(x):
    return (
        LAMBDA
        / (2.0 * np.pi)
        * np.exp(LAMBDA * x[:, 0:1])
        * np.sin(2.0 * np.pi * x[:, 1:2])
    )


def _p_exact(x):
    return 0.5 * (1.0 - np.exp(2.0 * LAMBDA * x[:, 0:1]))


def load_reference_bundle(precision=DEFAULT_PRECISION):
    rng = np.random.default_rng(0)
    dtype = np.float64 if precision == "float64" else np.float32
    x = rng.uniform([-0.5, -0.5], [1.0, 1.5], size=(100000, 2)).astype(dtype)
    y = np.hstack([_u_exact(x), _v_exact(x), _p_exact(x)]).astype(dtype)
    return {"X": x, "y": y}


def _pde(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    momentum_x = u * u_x + v * u_y + p_x - 1.0 / REYNOLDS * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - 1.0 / REYNOLDS * (v_xx + v_yy)
    continuity = u_x + v_y
    return [momentum_x, momentum_y, continuity]


def _boundary_outflow(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1.0)


def create_data(anchors=None):
    geom = dde.geometry.Rectangle([-0.5, -0.5], [1.0, 1.5])
    bcs = [
        dde.icbc.DirichletBC(
            geom, _u_exact, lambda x, on_boundary: on_boundary, component=0
        ),
        dde.icbc.DirichletBC(
            geom, _v_exact, lambda x, on_boundary: on_boundary, component=1
        ),
        dde.icbc.DirichletBC(geom, _p_exact, _boundary_outflow, component=2),
    ]
    return dde.data.PDE(
        geom,
        _pde,
        bcs,
        num_domain=0 if anchors is not None else 2601,
        num_boundary=0 if anchors is not None else 400,
        anchors=anchors,
        num_test=100000,
    )


def prepare_job_config(cfg, seed):
    return prepare_sampling_config(TASK_KEY, cfg, seed, create_data)


def create_network(cfg):
    residual_mode = "none"
    if cfg["activation_name"] == "fourier_features":
        from src.models.fourier_feature_network import FourierFeatureFNN

        return FourierFeatureFNN(
            layer_sizes=cfg["layer_sizes"],
            kernel_initializer=cfg.get("kernel_initializer", "Glorot normal"),
            fourier_scales=cfg.get("fourier_scales", (1.0, 10.0)),
            fourier_seed=cfg.get("fourier_seed", cfg.get("seed")),
            fourier_features_per_branch=cfg.get("fourier_features_per_branch"),
            input_lower=cfg.get("fourier_input_lower"),
            input_upper=cfg.get("fourier_input_upper"),
        )
    if cfg["activation_name"] == "pirate_mlp":
        return PirateNetFNN(
            layer_sizes=cfg["layer_sizes"],
            kernel_initializer=cfg.get("kernel_initializer", "Glorot normal"),
        )
    return GLLF_FNN(
        layer_sizes=cfg["layer_sizes"],
        activation_name=cfg["activation_name"],
        kernel_initializer=cfg.get("kernel_initializer", "Glorot normal"),
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
    return net.double() if precision == "float64" else net.float()


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


def _cuda_memory_peaks(device):
    if device.type != "cuda":
        return (None, None)
    torch.cuda.synchronize(device)
    return (
        torch.cuda.max_memory_allocated(device) / 1024**3,
        torch.cuda.max_memory_reserved(device) / 1024**3,
    )


def _component_l2s(y_true, y_pred):
    return {
        "l2_u": float(dde.metrics.l2_relative_error(y_true[:, 0:1], y_pred[:, 0:1])),
        "l2_v": float(dde.metrics.l2_relative_error(y_true[:, 1:2], y_pred[:, 1:2])),
        "l2_p": float(dde.metrics.l2_relative_error(y_true[:, 2:3], y_pred[:, 2:3])),
    }


def _primary_metric(component_metrics):
    return float(
        np.mean(
            [
                component_metrics["l2_u"],
                component_metrics["l2_v"],
                component_metrics["l2_p"],
            ]
        )
    )


class _KovasznayReferenceTracker(dde.callbacks.Callback):

    def __init__(self, X_ref, y_ref, interval=100):
        super().__init__()
        self.X_ref = X_ref
        self.y_ref = y_ref
        self.interval = interval
        self.mse_records = []
        self.l2_records = []
        self.component_records = []
        self.latest_step = None
        self.latest_metric = None

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.interval != 0 and step != 1:
            return
        y_pred = self.model.predict(self.X_ref)
        component = _component_l2s(self.y_ref, y_pred)
        primary_l2 = _primary_metric(component)
        self.mse_records.append((int(step), float(np.mean((y_pred - self.y_ref) ** 2))))
        self.l2_records.append((int(step), primary_l2))
        self.component_records.append((int(step), component))
        self.latest_step = int(step)
        self.latest_metric = float(primary_l2)


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
    data = create_persisted_data(TASK_KEY, cfg, seed, create_data)
    data = _cast_data_arrays(data, precision)
    set_seed(seed)
    net = _cast_network(create_network(cfg), precision)
    params = count_params(net)
    model = dde.Model(data, net)
    ref_bundle = load_reference_bundle(precision)
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
    if track_reference_metrics or False:
        ref_tracker = _KovasznayReferenceTracker(
            X_ref, y_ref, interval=reference_interval
        )
        callbacks.append(ref_tracker)
    best_model_saver = None
    optimizer = build_muon_with_aux_adam(
        net, weight_lr, activation_lr, matrix_policy=cfg.get("muon_matrix_policy", "backbone")
    )
    model.compile(optimizer)
    memory_device = next(net.parameters()).device
    if memory_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(memory_device)
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
    first_order_peak_allocated_gb, first_order_peak_reserved_gb = _cuda_memory_peaks(
        memory_device
    )
    lbfgs_wall = 0.0
    lbfgs_final_loss = first_order_final_loss
    lbfgs_budget_target_iters = None
    lbfgs_budget_consumed_iters = None
    lbfgs_budget_stop_reason = None
    lbfgs_func_evals = None
    lbfgs_single_optimizer = None
    lbfgs_restart_count = None
    lbfgs_peak_allocated_gb = None
    lbfgs_peak_reserved_gb = None
    lbfgs_budget_target_iters = int(
        cfg.get("lbfgs_fixed_budget_target_iters", DEFAULT_LBFGS_ITERS)
    )
    lbfgs_single_optimizer = True
    lbfgs_restart_count = 1
    if memory_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(memory_device)
    with temporary_lbfgs_options(
        maxiter=lbfgs_budget_target_iters,
        iter_per_step=lbfgs_budget_target_iters,
        maxfun=max(lbfgs_budget_target_iters * 50, lbfgs_budget_target_iters + 1),
        fun_per_step=max(lbfgs_budget_target_iters * 50, lbfgs_budget_target_iters + 1),
        gtol=-1.0,
        ftol=-1.0,
    ):
        model.compile("L-BFGS")
        if best_model_saver is not None:
            best_model_saver.phase_name = "lbfgs_fixed_budget"
        prev_step = model.train_state.step
        lbfgs_start = time.perf_counter()
        _, train_state = model.train(
            display_every=1000 if verbose else 100000, callbacks=callbacks
        )
        lbfgs_wall = time.perf_counter() - lbfgs_start
    lbfgs_peak_allocated_gb, lbfgs_peak_reserved_gb = _cuda_memory_peaks(memory_device)
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
    total_wall = time.perf_counter() - t0
    y_pred = model.predict(X_ref)
    f_pred = model.predict(X_ref, operator=_pde)
    component = _component_l2s(y_ref, y_pred)
    primary_l2 = _primary_metric(component)
    test_mse = float(np.mean((y_pred - y_ref) ** 2))
    residual = float(np.mean(np.abs(f_pred)))
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
    sampling_manifest = data.low_memory_sampling_manifest
    return {
        "l2_relative_error": primary_l2,
        "l2_u": component["l2_u"],
        "l2_v": component["l2_v"],
        "l2_p": component["l2_p"],
        "test_mse": test_mse,
        "mean_abs_pde_residual": residual,
        "max_abs_error": float(np.max(np.abs(y_pred - y_ref))),
        "params": params,
        "wall_time_sec": round(total_wall, 2),
        "first_order_wall_sec": round(first_order_wall, 2),
        "muon_wall_sec": round(muon_wall, 2),
        "lbfgs_wall_sec": round(lbfgs_wall, 2),
        "first_order_final_loss": first_order_final_loss,
        "muon_final_loss": muon_final_loss,
        "lbfgs_final_loss": lbfgs_final_loss,
        "first_order_optimizer": "muon",
        "first_order_budget_target_iters": first_order_iterations,
        "first_order_budget_consumed_iters": first_order_iterations,
        "post_first_order_mode": "lbfgs_fixed_budget",
        "lbfgs_budget_target_iters": lbfgs_budget_target_iters,
        "lbfgs_budget_consumed_iters": lbfgs_budget_consumed_iters,
        "lbfgs_budget_stop_reason": lbfgs_budget_stop_reason,
        "lbfgs_func_evals": lbfgs_func_evals,
        "lbfgs_single_optimizer": lbfgs_single_optimizer,
        "lbfgs_restart_count": lbfgs_restart_count,
        "budget_complete": True
        and lbfgs_budget_consumed_iters == lbfgs_budget_target_iters,
        "task_key": cfg.get("task_key", TASK_KEY),
        "method_tag": cfg.get("method_tag"),
        "setting_tag": cfg.get("setting_tag"),
        "optimizer_tag": cfg.get("optimizer_tag"),
        "optimizer_name": cfg.get("optimizer_name", cfg.get("optimizer_tag")),
        "sampling_manifest_spec": sampling_manifest["spec"],
        "analytic_reference_id": cfg.get("analytic_reference_id"),
        "metric_identity": cfg.get("metric_identity", "mean_relative_l2_u_v_p"),
        "first_order_peak_cuda_allocated_gb": first_order_peak_allocated_gb,
        "first_order_peak_cuda_reserved_gb": first_order_peak_reserved_gb,
        "lbfgs_peak_cuda_allocated_gb": lbfgs_peak_allocated_gb,
        "lbfgs_peak_cuda_reserved_gb": lbfgs_peak_reserved_gb,
        "loss_history": loss_tracker.records,
        "component_l2_history": ref_tracker.component_records if ref_tracker else None,
        "final_mu": final_mu,
        "final_I": final_I,
        "final_activation_state": final_activation_state,
        "selected_model_snapshot": selected_model_snapshot,
        "fourier_feature_manifest": (
            net.implementation_manifest()
            if hasattr(net, "implementation_manifest")
            else None
        ),
        "config": dict(cfg),
        "seed": seed,
    }

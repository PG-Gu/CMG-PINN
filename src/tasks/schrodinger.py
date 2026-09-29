from src.paths import DATA_DIR as BUNDLED_DATA
from src.paths import DATA_DIR
import time
import urllib.request
import deepxde as dde
import numpy as np
from scipy.io import loadmat
from src.activations.activation_baselines import (
    activation_options_from_config,
    is_activation_baseline,
)
from src.training.muon_common import build_muon_with_aux_adam
from src.training.low_memory_forward_common import (
    create_persisted_data,
    prepare_sampling_config,
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
from src.models.network import GLLF_FNN
from src.models.pirate_mlp import PirateNetFNN

DATA_DIR = BUNDLED_DATA
NLS_MAT = DATA_DIR / "NLS.mat"
TASK_KEY = "schrodinger"
NLS_URL = (
    "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/NLS.mat"
)
DEFAULT_FIRST_ORDER_ITERATIONS = 10000
DEFAULT_LBFGS_ITERS = 15000
DEFAULT_PRECISION = "float32"
REFERENCE_INTERVAL = 100
TRUE_TRAIN_LOSS_INTERVAL = 100
X_LOWER = -5.0
X_UPPER = 5.0
T_LOWER = 0.0
T_UPPER = np.pi / 2


def ensure_reference_data():
    if NLS_MAT.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading NLS.mat to {NLS_MAT} ...")
        urllib.request.urlretrieve(NLS_URL, NLS_MAT)
        print("Done.")
    except Exception as exc:
        raise FileNotFoundError(
            f"Missing {NLS_MAT}. Place NLS.mat in ./data or ensure the fallback download can reach {NLS_URL}."
        ) from exc


def load_reference_bundle():
    ensure_reference_data()
    data = loadmat(NLS_MAT)
    x = np.ravel(data["x"]).astype(np.float64)
    t = np.ravel(data["tt"]).astype(np.float64)
    exact = np.asarray(data["uu"])
    xx, tt = np.meshgrid(x, t)
    X = np.column_stack([xx.ravel(), tt.ravel()])
    u = np.real(exact.T).reshape(-1, 1)
    v = np.imag(exact.T).reshape(-1, 1)
    h = np.sqrt(u**2 + v**2)
    y = np.hstack([u, v])
    return {"x": x, "t": t, "X": X, "u": u, "v": v, "h": h, "y": y}


def _pde(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    v_t = dde.grad.jacobian(y, x, i=1, j=1)
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
    f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u
    return [f_u, f_v]


def _init_cond_u(x):
    return 2.0 / np.cosh(x[:, 0:1])


def _init_cond_v(x):
    return np.zeros((len(x), 1), dtype=x.dtype if hasattr(x, "dtype") else np.float32)


def create_data(anchors=None):
    space_domain = dde.geometry.Interval(X_LOWER, X_UPPER)
    time_domain = dde.geometry.TimeDomain(T_LOWER, T_UPPER)
    geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)
    bc_u_0 = dde.icbc.PeriodicBC(
        geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0
    )
    bc_u_1 = dde.icbc.PeriodicBC(
        geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=0
    )
    bc_v_0 = dde.icbc.PeriodicBC(
        geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=1
    )
    bc_v_1 = dde.icbc.PeriodicBC(
        geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=1
    )
    ic_u = dde.icbc.IC(
        geomtime, _init_cond_u, lambda _, on_initial: on_initial, component=0
    )
    ic_v = dde.icbc.IC(
        geomtime, _init_cond_v, lambda _, on_initial: on_initial, component=1
    )
    return dde.data.TimePDE(
        geomtime,
        _pde,
        [bc_u_0, bc_u_1, bc_v_0, bc_v_1, ic_u, ic_v],
        num_domain=0 if anchors is not None else 10000,
        num_boundary=0 if anchors is not None else 20,
        num_initial=0 if anchors is not None else 200,
        anchors=anchors,
        train_distribution="pseudo",
    )


def prepare_job_config(cfg, seed):
    return prepare_sampling_config(TASK_KEY, cfg, seed, create_data)


def create_network(cfg):
    act = cfg["activation_name"]
    residual_mode = "none"
    if act == "fourier_features":
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
    if act == "pirate_mlp":
        return PirateNetFNN(
            layer_sizes=cfg["layer_sizes"],
            kernel_initializer=cfg.get("kernel_initializer", "Glorot normal"),
        )
    return GLLF_FNN(
        layer_sizes=cfg["layer_sizes"],
        activation_name=act,
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


def _schrodinger_metrics(model, ref_bundle):
    X_ref = ref_bundle["X"]
    pred = model.predict(X_ref)
    u_pred = pred[:, 0:1]
    v_pred = pred[:, 1:2]
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
    f_pred = model.predict(X_ref, operator=_pde)
    f_concat = np.hstack(f_pred) if isinstance(f_pred, (list, tuple)) else f_pred
    h_err = h_pred - ref_bundle["h"]
    return {
        "l2_u": float(dde.metrics.l2_relative_error(ref_bundle["u"], u_pred)),
        "l2_v": float(dde.metrics.l2_relative_error(ref_bundle["v"], v_pred)),
        "l2_h": float(dde.metrics.l2_relative_error(ref_bundle["h"], h_pred)),
        "test_mse": float(np.mean(h_err**2)),
        "mean_abs_pde_residual": float(np.mean(np.abs(f_concat))),
        "max_abs_error": float(np.max(np.abs(h_err))),
    }


class SchrodingerReferenceMetricTracker(dde.callbacks.Callback):

    def __init__(self, ref_bundle, interval=100):
        super().__init__()
        self.ref_bundle = ref_bundle
        self.interval = interval
        self.mse_records = []
        self.l2_records = []

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.interval == 0 or step == 1:
            metrics = _schrodinger_metrics(self.model, self.ref_bundle)
            self.mse_records.append((int(step), metrics["test_mse"]))
            self.l2_records.append((int(step), metrics["l2_h"]))


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
    ref_bundle = load_reference_bundle()
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
        ref_tracker = SchrodingerReferenceMetricTracker(
            ref_bundle, interval=reference_interval
        )
        callbacks.append(ref_tracker)
    optimizer = build_muon_with_aux_adam(
        net, weight_lr, activation_lr, matrix_policy=cfg.get("muon_matrix_policy", "backbone")
    )
    model.compile(optimizer, loss="MSE")
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
    metrics = _schrodinger_metrics(model, ref_bundle)
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
        "l2_relative_error": metrics["l2_h"],
        "l2_u": metrics["l2_u"],
        "l2_v": metrics["l2_v"],
        "l2_h": metrics["l2_h"],
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
        "metric_identity": cfg.get("metric_identity", "amplitude_l2_relative_error"),
        "loss_history": callbacks[0].records,
        "final_mu": final_mu,
        "final_I": final_I,
        "final_activation_state": final_activation_state,
        "fourier_feature_manifest": (
            net.implementation_manifest()
            if hasattr(net, "implementation_manifest")
            else None
        ),
        "selected_model_snapshot": selected_model_snapshot,
        "config": dict(cfg),
        "seed": seed,
    }

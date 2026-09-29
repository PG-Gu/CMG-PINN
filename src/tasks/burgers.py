from src.paths import DATA_DIR as BUNDLED_DATA
from src.paths import DATA_DIR
import random
import time
import urllib.request
from contextlib import contextmanager
import numpy as np
import torch
import deepxde as dde
from deepxde.optimizers import config as dde_optim_config
from src.activations.activation_baselines import (
    activation_options_from_config,
    activation_state,
    is_activation_baseline,
    is_fixed_activation,
    module_auxiliary_parameters,
)
from src.activations.gllf_activation import GLLFActivation
from src.models.network import GLLF_FNN
from src.training.muon_common import build_muon_with_aux_adam
from src.training.task_train_common import save_selected_model_snapshot
from src.models.pirate_mlp import PirateNetFNN

DATA_DIR = BUNDLED_DATA
BURGERS_NPZ = DATA_DIR / "Burgers.npz"
BURGERS_NPZ_URL = (
    "https://github.com/lululxvi/deepxde/raw/master/examples/dataset/Burgers.npz"
)
LBFGS_FIXED_BUDGET_TARGET_ITERS = 15000
LBFGS_FIXED_BUDGET_MAXFUN_MULTIPLIER = 50
BURGERS_NU = 0.01 / np.pi


def set_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dde.config.set_random_seed(seed)


def load_reference_bundle():
    """Load Burgers.npz and return grid-friendly arrays for evaluation."""
    if not BURGERS_NPZ.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Burgers.npz to {BURGERS_NPZ} ...")
        urllib.request.urlretrieve(BURGERS_NPZ_URL, BURGERS_NPZ)
        print("Done.")
    data = np.load(BURGERS_NPZ)
    t = np.ravel(data["t"])
    x = np.ravel(data["x"])
    exact = data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return {"t": t, "x": x, "u": exact, "X": X, "y": y}


def _reference_mse_l2(model, X_ref, y_ref):
    """Evaluate reference-grid MSE and L2 relative error for the current model."""
    y_pred = model.predict(X_ref)
    mse = float(np.mean((y_pred - y_ref) ** 2))
    l2 = float(dde.metrics.l2_relative_error(y_ref, y_pred))
    return (mse, l2)


def _ux_operator(x, y):
    return dde.grad.jacobian(y, x, i=0, j=0)


def _make_boundary_eval_points(t_ref, x_value):
    return np.column_stack(
        [
            np.full_like(t_ref, fill_value=x_value, dtype=np.float64),
            t_ref.astype(np.float64),
        ]
    )


def _make_initial_eval_points(x_ref):
    return np.column_stack(
        [x_ref.astype(np.float64), np.zeros_like(x_ref, dtype=np.float64)]
    )


def _evaluate_reference_suite(model, ref_bundle):
    """Compute final reference-grid and physics metrics for the current model."""
    X_ref = ref_bundle["X"]
    y_ref = ref_bundle["y"]
    x_ref = ref_bundle["x"]
    t_ref = ref_bundle["t"]
    nx = len(x_ref)
    nt = len(t_ref)
    y_pred = model.predict(X_ref)
    f_pred = model.predict(X_ref, operator=_pde)
    l2_err = float(dde.metrics.l2_relative_error(y_ref, y_pred))
    test_mse = float(np.mean((y_pred - y_ref) ** 2))
    mean_res = float(np.mean(np.abs(f_pred)))
    max_abs_error = float(np.max(np.abs(y_pred - y_ref)))
    X_bc_left = _make_boundary_eval_points(t_ref, x_ref[0])
    X_bc_right = _make_boundary_eval_points(t_ref, x_ref[-1])
    bc_left = model.predict(X_bc_left)
    bc_right = model.predict(X_bc_right)
    bc_errors = np.vstack((bc_left, bc_right))
    bc_test_mse = float(np.mean(np.square(bc_errors)))
    X_ic = _make_initial_eval_points(x_ref)
    ic_true = -np.sin(np.pi * x_ref)[:, None]
    ic_pred = model.predict(X_ic)
    ic_test_mse = float(np.mean(np.square(ic_pred - ic_true)))
    y_pred_grid = y_pred.reshape(nt, nx)
    mass = np.trapz(y_pred_grid, x_ref, axis=1)
    mass_dt = np.gradient(mass, t_ref, edge_order=2 if nt >= 3 else 1)
    ux_left = model.predict(X_bc_left, operator=_ux_operator).reshape(-1)
    ux_right = model.predict(X_bc_right, operator=_ux_operator).reshape(-1)
    u_left = bc_left.reshape(-1)
    u_right = bc_right.reshape(-1)
    flux_left = 0.5 * np.square(u_left) - BURGERS_NU * ux_left
    flux_right = 0.5 * np.square(u_right) - BURGERS_NU * ux_right
    mass_balance = mass_dt + flux_right - flux_left
    mean_abs_mass_balance_error = float(np.mean(np.abs(mass_balance)))
    return {
        "y_pred": y_pred,
        "f_pred": f_pred,
        "l2_relative_error": l2_err,
        "test_mse": test_mse,
        "mean_abs_pde_residual": mean_res,
        "max_abs_error": max_abs_error,
        "bc_test_mse": bc_test_mse,
        "ic_test_mse": ic_test_mse,
        "mean_abs_mass_balance_error": mean_abs_mass_balance_error,
    }


@contextmanager
def _temporary_lbfgs_options(**updates):
    """Temporarily override DeepXDE L-BFGS options in-place."""
    original = dict(dde_optim_config.LBFGS_options)
    dde_optim_config.LBFGS_options.update(updates)
    try:
        yield
    finally:
        dde_optim_config.LBFGS_options.clear()
        dde_optim_config.LBFGS_options.update(original)


def _uses_gllf_parameters(cfg):
    """Return True when the config includes either hidden or input GLLF params."""
    return cfg.get("activation_name") == "gllf"


def _uses_auxiliary_parameters(cfg):
    """Return True when separate auxiliary LR parameters are present."""
    return (
        _uses_gllf_parameters(cfg)
        or bool(False)
        or cfg.get("activation_name") == "pirate_mlp"
        or is_activation_baseline(cfg.get("activation_name"))
        or bool(cfg.get("residual_enabled", False))
    )


def _pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx


def create_burgers_data():
    """Build the TimePDE problem (geometry + PDE + BC/IC)."""
    geom = dde.geometry.Interval(-1, 1)
    timeDom = dde.geometry.TimeDomain(0, 0.99)
    geomtime = dde.geometry.GeometryXTime(geom, timeDom)
    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(
        geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
    )
    data = dde.data.TimePDE(
        geomtime, _pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160
    )
    return data


def create_network(cfg):
    """Create a GLLF_FNN (or tanh FNN) from a config dict."""
    act = cfg["activation_name"]
    if act == "fourier_features":
        from src.models.fourier_feature_network import FourierFeatureFNN

        return FourierFeatureFNN(
            cfg["layer_sizes"],
            kernel_initializer=cfg.get("kernel_initializer", "Glorot normal"),
            fourier_scales=cfg.get("fourier_scales", (1.0, 10.0)),
            fourier_seed=cfg.get("fourier_seed", cfg.get("seed")),
            fourier_features_per_branch=cfg.get("fourier_features_per_branch"),
            input_lower=cfg.get("fourier_input_lower"),
            input_upper=cfg.get("fourier_input_upper"),
        )
    sizes = cfg["layer_sizes"]
    residual_mode = "none"
    cmg_implementation = cfg.get("cmg_implementation", "reference")
    if act == "tanh":
        return GLLF_FNN(
            sizes,
            activation_name="tanh",
            kernel_initializer="Glorot normal",
            norm_mode=cfg.get("norm_mode", "none"),
            cmg_implementation=cmg_implementation,
        )
    elif act == "sin":
        return GLLF_FNN(
            sizes,
            activation_name="sin",
            kernel_initializer="Glorot normal",
            cmg_implementation=cmg_implementation,
        )
    elif act == "gllf":
        return GLLF_FNN(
            sizes,
            activation_name=act,
            kernel_initializer="Glorot normal",
            sharing=cfg["sharing"],
            bounds_mode=cfg["bounds_mode"],
            norm_mode=cfg.get("norm_mode", "none"),
            mu_init=cfg.get("mu_init", "uniform"),
            I_init=cfg.get("I_init", 0.5),
            gllf_mode=cfg.get("gllf_mode", "all"),
            eps=cfg.get("eps", 1e-06),
            cmg_implementation=cmg_implementation,
        )
    elif act == "cmg_gelu":
        return GLLF_FNN(
            sizes,
            activation_name=act,
            kernel_initializer="Glorot normal",
            sharing=cfg.get("sharing", "layer"),
            bounds_mode=cfg.get("bounds_mode", "tanh_wrapper_fast"),
            norm_mode=cfg.get("norm_mode", "none"),
            mu_init=cfg.get("mu_init", 0.5),
            I_init=cfg.get("I_init", 0.5),
            gllf_mode=cfg.get("gllf_mode", "all"),
            eps=cfg.get("eps", 1e-06),
            cmg_implementation=cmg_implementation,
        )
    elif is_activation_baseline(act) or is_fixed_activation(act):
        return GLLF_FNN(
            sizes,
            activation_name=act,
            kernel_initializer="Glorot normal",
            norm_mode=cfg.get("norm_mode", "none"),
            activation_options=activation_options_from_config(cfg),
        )
    elif act == "pirate_mlp":
        return PirateNetFNN(sizes, kernel_initializer="Glorot normal")
    else:
        raise ValueError(f"Unknown activation_name: {act!r}")


def _snapshot_gllf(net):
    """Return a dict of current mu / I values per GLLF layer (detached)."""
    snap = {}
    idx = 0
    for m in net.modules():
        if isinstance(m, GLLFActivation):
            snap[f"layer_{idx}"] = {
                "mu": m.mu.detach().cpu().tolist(),
                "I": m.I_param.detach().cpu().tolist(),
            }
            idx += 1
    return snap


def _count_params(net):
    total = sum((p.numel() for p in net.parameters()))
    gllf = 0
    activation = 0
    residual_alpha = 0
    ln = 0
    bn = 0
    for m in net.modules():
        if isinstance(m, GLLFActivation):
            gllf += m.mu_raw.numel() + m.I_raw.numel()
        activation += sum((p.numel() for p in module_auxiliary_parameters(m)))
        if isinstance(m, torch.nn.LayerNorm):
            ln += sum((p.numel() for p in m.parameters()))
        if isinstance(m, torch.nn.BatchNorm1d):
            bn += sum((p.numel() for p in m.parameters()))
    if hasattr(net, "residual_parameter_count"):
        residual_alpha = net.residual_parameter_count()
    return {
        "total": total,
        "gllf": gllf,
        "activation": activation,
        "residual_alpha": residual_alpha,
        "layernorm": ln,
        "batchnorm": bn,
        "weights": total - gllf - 0 - 0 - activation - 0 - residual_alpha - ln - bn,
    }


def _lbfgs_state_stats(opt):
    """Return (n_iter, func_evals) from a PyTorch LBFGS optimizer state."""
    if opt is None:
        return (None, None)
    state_dict = opt.state_dict().get("state", {})
    if not state_dict:
        return (None, None)
    first_state = next(iter(state_dict.values()))
    return (first_state.get("n_iter"), first_state.get("func_evals"))


class _LossTracker(dde.callbacks.Callback):
    """Records per-component losses every *interval* steps."""

    def __init__(self, interval=100):
        super().__init__()
        self.interval = interval
        self.records = []

    def on_train_begin(self):
        pass

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.interval == 0 or step == 1:
            losses = self.model.train_state.loss_train
            self.records.append((int(step), [float(l) for l in losses]))


class _TrueTrainLossTracker(dde.callbacks.Callback):
    """Records fresh train losses every *interval* steps."""

    def __init__(self, interval=100):
        super().__init__()
        self.interval = interval
        self.records = []

    def on_train_begin(self):
        pass

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.interval == 0 or step == 1:
            _, losses = self.model._outputs_losses(
                True,
                self.model.train_state.X_train,
                self.model.train_state.y_train,
                self.model.train_state.train_aux_vars,
            )
            self.records.append((int(step), [float(l) for l in losses]))


class _GLLFTracker(dde.callbacks.Callback):
    """Records GLLF mu/I values every *interval* steps."""

    def __init__(self, net, interval=100, state_fn=_snapshot_gllf):
        super().__init__()
        self.net = net
        self.interval = interval
        self.state_fn = state_fn
        self.records = []

    def on_train_begin(self):
        pass

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.interval == 0 or step == 1:
            self.records.append((int(step), self.state_fn(self.net)))


class _ReferenceMetricTracker(dde.callbacks.Callback):
    """Records reference-grid MSE and L2 every *interval* steps."""

    def __init__(self, X_ref, y_ref, interval=100):
        super().__init__()
        self.X_ref = X_ref
        self.y_ref = y_ref
        self.interval = interval
        self.mse_records = []
        self.l2_records = []

    def on_train_begin(self):
        pass

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.interval == 0 or step == 1:
            mse, l2 = _reference_mse_l2(self.model, self.X_ref, self.y_ref)
            self.mse_records.append((int(step), mse))
            self.l2_records.append((int(step), l2))


def run_single(
    cfg,
    seed,
    verbose=True,
    track_reference_metrics=False,
    reference_interval=100,
    track_true_train_loss=False,
    true_train_loss_interval=100,
):
    """Run one Burgers experiment and return a result dict."""
    weight_lr = cfg.get("weight_lr", 0.001)
    activation_lr = cfg.get("activation_lr", weight_lr)
    first_order_iterations = cfg.get("first_order_iterations", 15000)
    lbfgs_mode = cfg.get("lbfgs_mode", "all_params")
    if cfg.get("track_true_train_loss"):
        track_true_train_loss = True
    true_train_loss_interval = cfg.get(
        "true_train_loss_interval", true_train_loss_interval
    )
    set_seed(seed)
    data = create_burgers_data()
    set_seed(seed)
    net = create_network(cfg)
    has_gllf_params = _uses_gllf_parameters(cfg)
    has_aux_params = _uses_auxiliary_parameters(cfg)
    param_counts = _count_params(net)
    if verbose:
        print(f"\n{'=' * 60}")
        label = cfg["activation_name"]
        if cfg["activation_name"] == "gllf":
            label += f" ({cfg['sharing']}, {cfg['bounds_mode']})"
        elif cfg.get("norm_mode") not in (None, "none"):
            label += f" ({cfg['norm_mode']})"
        print(f"Config: {label}  |  seed={seed}  |  params={param_counts['total']}")
        print(f"{'=' * 60}")
    model = dde.Model(data, net)
    ref_bundle = None
    X_ref, y_ref = (None, None)
    if track_reference_metrics or False:
        ref_bundle = load_reference_bundle()
        X_ref, y_ref = (ref_bundle["X"], ref_bundle["y"])
    loss_tracker = _LossTracker(interval=100)
    callbacks = [loss_tracker]
    true_loss_tracker = None
    gllf_tracker = None
    ref_tracker = None
    if track_true_train_loss:
        true_loss_tracker = _TrueTrainLossTracker(interval=true_train_loss_interval)
        callbacks.append(true_loss_tracker)
    if has_gllf_params or is_activation_baseline(cfg["activation_name"]):
        gllf_tracker = _GLLFTracker(
            net,
            interval=100,
            state_fn=_snapshot_gllf if has_gllf_params else activation_state,
        )
        callbacks.append(gllf_tracker)
    if track_reference_metrics:
        ref_tracker = _ReferenceMetricTracker(X_ref, y_ref, interval=reference_interval)
        callbacks.append(ref_tracker)
    optimizer = build_muon_with_aux_adam(
        net, weight_lr, activation_lr, matrix_policy=cfg.get("muon_matrix_policy", "backbone")
    )
    model.compile(optimizer)
    t0 = time.perf_counter()
    first_order_start = t0
    losshistory, train_state = model.train(
        iterations=first_order_iterations,
        callbacks=callbacks,
        display_every=1000 if verbose else first_order_iterations + 1,
    )
    first_order_end = time.perf_counter()
    first_order_wall = first_order_end - first_order_start
    first_order_final_loss = float(np.sum(train_state.loss_train))
    lbfgs_wall = 0.0
    lbfgs_final_loss = first_order_final_loss
    post_first_order_extra_iters = 0
    post_first_order_stop_reason = None
    post_first_order_final_loss = None
    lbfgs_budget_target_iters = None
    lbfgs_budget_consumed_iters = None
    lbfgs_restart_count = None
    lbfgs_budget_stop_reason = None
    lbfgs_func_evals = None
    lbfgs_single_optimizer = None
    lbfgs_budget_target_iters = int(
        cfg.get("lbfgs_fixed_budget_target_iters", LBFGS_FIXED_BUDGET_TARGET_ITERS)
    )
    fun_budget = int(
        cfg.get(
            "lbfgs_fixed_budget_maxfun",
            max(
                lbfgs_budget_target_iters * LBFGS_FIXED_BUDGET_MAXFUN_MULTIPLIER,
                lbfgs_budget_target_iters + 1,
            ),
        )
    )
    lbfgs_restart_count = 1
    lbfgs_single_optimizer = True
    with _temporary_lbfgs_options(
        maxiter=lbfgs_budget_target_iters,
        iter_per_step=lbfgs_budget_target_iters,
        maxfun=fun_budget,
        fun_per_step=fun_budget,
        gtol=-1.0,
        ftol=-1.0,
    ):
        model.compile("L-BFGS")
        prev_step = model.train_state.step
        lbfgs_start = time.perf_counter()
        losshistory, train_state = model.train(
            display_every=1000 if verbose else 100000, callbacks=callbacks
        )
        lbfgs_wall += time.perf_counter() - lbfgs_start
    state_n_iter, lbfgs_func_evals = _lbfgs_state_stats(model.opt)
    consumed_from_steps = int(train_state.step - prev_step)
    lbfgs_budget_consumed_iters = (
        int(state_n_iter) if state_n_iter is not None else consumed_from_steps
    )
    if lbfgs_budget_consumed_iters >= lbfgs_budget_target_iters:
        lbfgs_budget_stop_reason = "budget_exhausted"
    else:
        lbfgs_budget_stop_reason = "ended_early"
    lbfgs_final_loss = float(np.sum(train_state.loss_train))
    total_wall = time.perf_counter() - t0
    if ref_bundle is None:
        ref_bundle = load_reference_bundle()
    metrics = _evaluate_reference_suite(model, ref_bundle)
    l2_err = metrics["l2_relative_error"]
    test_mse = metrics["test_mse"]
    mean_res = metrics["mean_abs_pde_residual"]
    if verbose:
        print(f"  L2 relative error : {l2_err:.6e}")
        print(f"  Mean |PDE residual|: {mean_res:.6e}")
        print(f"  BC test MSE       : {metrics['bc_test_mse']:.6e}")
        print(f"  IC test MSE       : {metrics['ic_test_mse']:.6e}")
        print(f"  Max |error|       : {metrics['max_abs_error']:.6e}")
        print(f"  Mean |mass bal.|  : {metrics['mean_abs_mass_balance_error']:.6e}")
        print(
            f"  Wall time         : {total_wall:.1f}s  (Muon {first_order_wall:.1f}s + L-BFGS {lbfgs_wall:.1f}s)"
        )
    final_mu, final_I = (None, None)
    final_activation_state = None
    if has_gllf_params:
        snap = _snapshot_gllf(net)
        final_mu = {k: v["mu"] for k, v in snap.items()}
        final_I = {k: v["I"] for k, v in snap.items()}
    elif is_activation_baseline(cfg["activation_name"]):
        final_activation_state = activation_state(net)
    selected_model_snapshot = save_selected_model_snapshot(net, cfg)
    result = {
        "fourier_feature_manifest": (
            net.implementation_manifest()
            if hasattr(net, "implementation_manifest")
            else None
        ),
        "selected_model_snapshot": selected_model_snapshot,
        "l2_relative_error": l2_err,
        "mean_abs_pde_residual": mean_res,
        "params": param_counts,
        "wall_time_sec": round(total_wall, 2),
        "first_order_wall_sec": round(first_order_wall, 2),
        "lbfgs_wall_sec": round(lbfgs_wall, 2),
        "first_order_final_loss": first_order_final_loss,
        "first_order_optimizer": "muon",
        "lbfgs_final_loss": lbfgs_final_loss,
        "post_first_order_mode": "lbfgs_fixed_budget",
        "post_first_order_extra_iters": post_first_order_extra_iters,
        "post_first_order_stop_reason": post_first_order_stop_reason,
        "post_first_order_final_loss": post_first_order_final_loss,
        "lbfgs_budget_target_iters": lbfgs_budget_target_iters,
        "lbfgs_budget_consumed_iters": lbfgs_budget_consumed_iters,
        "lbfgs_restart_count": lbfgs_restart_count,
        "lbfgs_budget_stop_reason": lbfgs_budget_stop_reason,
        "lbfgs_func_evals": lbfgs_func_evals,
        "lbfgs_single_optimizer": lbfgs_single_optimizer,
        "test_mse": test_mse,
        "bc_test_mse": metrics["bc_test_mse"],
        "ic_test_mse": metrics["ic_test_mse"],
        "max_abs_error": metrics["max_abs_error"],
        "mean_abs_mass_balance_error": metrics["mean_abs_mass_balance_error"],
        "loss_history": loss_tracker.records,
        "final_mu": final_mu,
        "final_I": final_I,
        "final_activation_state": final_activation_state,
        "config": {k: v if not isinstance(v, list) else v for k, v in cfg.items()},
        "seed": seed,
    }
    return result

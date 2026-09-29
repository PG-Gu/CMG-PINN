"""Shared training/evaluation helpers for non-Burgers PINN tasks."""

import random
from contextlib import contextmanager
from pathlib import Path
import deepxde as dde
import numpy as np
import torch
from deepxde.optimizers import config as dde_optim_config
from src.activations.activation_baselines import (
    activation_state,
    module_auxiliary_parameters,
)
from src.activations.gllf_activation import GLLFActivation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dde.config.set_random_seed(seed)


def reference_mse_l2(model, X_ref, y_ref):
    y_pred = model.predict(X_ref)
    mse = float(np.mean((y_pred - y_ref) ** 2))
    l2 = float(dde.metrics.l2_relative_error(y_ref, y_pred))
    return (mse, l2)


def save_selected_model_snapshot(net, cfg, *, external_parameters=None):
    """Save terminal weights and inverse parameters when requested."""
    if not cfg.get("save_selected_model_snapshot", False):
        return None
    path = Path(cfg["selected_model_snapshot_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "external_trainable_parameters": {
                k: v.detach().cpu() for k, v in (external_parameters or {}).items()
            },
        },
        path,
    )
    return {"path": str(path), "snapshot_kind": "terminal_state"}


def evaluate_reference_suite(model, X_ref, y_ref, pde_operator):
    y_pred = model.predict(X_ref)
    f_pred = model.predict(X_ref, operator=pde_operator)
    return {
        "y_pred": y_pred,
        "f_pred": f_pred,
        "l2_relative_error": float(dde.metrics.l2_relative_error(y_ref, y_pred)),
        "test_mse": float(np.mean((y_pred - y_ref) ** 2)),
        "mean_abs_pde_residual": float(np.mean(np.abs(f_pred))),
        "max_abs_error": float(np.max(np.abs(y_pred - y_ref))),
    }


def snapshot_gllf(net):
    snap = {}
    idx = 0
    for m in net.modules():
        if isinstance(m, GLLFActivation):
            idx += 1
            snap[f"layer_{idx}"] = {
                "mu": _to_list(m.mu.detach().cpu().flatten()),
                "I": _to_list(m.I_param.detach().cpu().flatten()),
            }
    return snap


def snapshot_activation_state(net):
    """Return effective activation parameters in layer order."""
    return activation_state(net)


def count_params(net):
    total = sum((p.numel() for p in net.parameters()))
    trainable = sum((p.numel() for p in net.parameters() if p.requires_grad))
    gllf = 0
    activation = 0
    residual_alpha = 0
    for m in net.modules():
        if isinstance(m, GLLFActivation):
            for p in m.parameters(recurse=False):
                gllf += p.numel()
        else:
            activation += sum((p.numel() for p in module_auxiliary_parameters(m)))
    if hasattr(net, "residual_parameter_count"):
        residual_alpha = net.residual_parameter_count()
    return {
        "total": int(total),
        "trainable": int(trainable),
        "gllf": int(gllf),
        "activation": int(activation),
        "residual_alpha": int(residual_alpha),
        "weights": int(trainable - gllf - 0 - 0 - activation - 0 - residual_alpha),
    }


def get_aux_param_groups(model, weight_lr, aux_lr, *, extra_weight_params=()):
    aux_ids = set()
    aux_params = []
    net = model if not hasattr(model, "net") else model.net
    for m in net.modules():
        if isinstance(m, GLLFActivation):
            for p in m.parameters(recurse=False):
                aux_params.append(p)
                aux_ids.add(id(p))
        else:
            for p in module_auxiliary_parameters(m):
                aux_params.append(p)
                aux_ids.add(id(p))
    if hasattr(net, "residual_parameters"):
        for p in net.residual_parameters():
            aux_params.append(p)
            aux_ids.add(id(p))
    weight_params = [p for p in model.parameters() if id(p) not in aux_ids]
    known_ids = {id(p) for p in weight_params}
    known_ids.update(aux_ids)
    for param in extra_weight_params:
        if getattr(param, "requires_grad", False) and id(param) not in known_ids:
            weight_params.append(param)
            known_ids.add(id(param))
    return [
        {"params": weight_params, "lr": weight_lr},
        {"params": aux_params, "lr": aux_lr},
    ]


def lbfgs_state_stats(opt):
    for state in opt.state.values():
        if "n_iter" in state or "func_evals" in state:
            return (state.get("n_iter"), state.get("func_evals"))
    return (None, None)


@contextmanager
def temporary_lbfgs_options(**updates):
    original = dict(dde_optim_config.LBFGS_options)
    dde_optim_config.LBFGS_options.update(updates)
    try:
        yield
    finally:
        dde_optim_config.LBFGS_options.clear()
        dde_optim_config.LBFGS_options.update(original)


class LossTracker(dde.callbacks.Callback):

    def __init__(self, interval=100):
        super().__init__()
        self.interval = interval
        self.records = []

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.interval == 0 or step == 1:
            losses = self.model.train_state.loss_train
            self.records.append((int(step), [float(loss) for loss in losses]))


class TrueTrainLossTracker(dde.callbacks.Callback):

    def __init__(self, interval=100):
        super().__init__()
        self.interval = interval
        self.records = []

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.interval == 0 or step == 1:
            _, losses = self.model._outputs_losses(
                True,
                self.model.train_state.X_train,
                self.model.train_state.y_train,
                self.model.train_state.train_aux_vars,
            )
            self.records.append((int(step), [float(loss) for loss in losses]))


class GLLFTracker(dde.callbacks.Callback):

    def __init__(self, net, interval=100, state_fn=snapshot_gllf):
        super().__init__()
        self.net = net
        self.interval = interval
        self.state_fn = state_fn
        self.records = []

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.interval == 0 or step == 1:
            self.records.append((int(step), self.state_fn(self.net)))


class ReferenceMetricTracker(dde.callbacks.Callback):

    def __init__(self, X_ref, y_ref, interval=100):
        super().__init__()
        self.X_ref = X_ref
        self.y_ref = y_ref
        self.interval = interval
        self.mse_records = []
        self.l2_records = []

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.interval == 0 or step == 1:
            mse, l2 = reference_mse_l2(self.model, self.X_ref, self.y_ref)
            self.mse_records.append((int(step), mse))
            self.l2_records.append((int(step), l2))


def _to_list(arr):
    out = arr.tolist()
    return out if isinstance(out, list) else [out]

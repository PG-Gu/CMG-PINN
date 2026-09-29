from __future__ import annotations
import deepxde as dde
import numpy as np
from src.training.low_memory_forward_common import (
    create_persisted_data,
    create_standard_network,
    prepare_sampling_config,
    run_forward_task,
)

TASK_KEY = "underdamped_vibration"
POINT_COUNTS = {"domain": 1000, "initial_conditions": 2}


def _exact_solution(t):
    omega = 3 * np.sqrt(3) / 2
    return np.exp(-1.5 * t[:, 0:1]) * (
        np.cos(omega * t[:, 0:1]) + np.sin(omega * t[:, 0:1]) / np.sqrt(3)
    )


def _pde(t, y):
    u_t = dde.grad.jacobian(y, t, i=0, j=0)
    u_tt = dde.grad.hessian(y, t, i=0, j=0)
    return u_tt + 3 * u_t + 9 * y


def _data_factory(anchors):
    timedomain = dde.geometry.TimeDomain(0, 1)
    displacement = dde.icbc.DirichletBC(
        timedomain,
        lambda t: 1.0,
        lambda t, on_boundary: on_boundary and dde.utils.isclose(t[0], 0.0),
    )
    velocity = dde.icbc.OperatorBC(
        timedomain,
        lambda t, y, _: dde.grad.jacobian(y, t, i=0, j=0),
        lambda t, on_boundary: on_boundary and dde.utils.isclose(t[0], 0.0),
    )
    count = POINT_COUNTS["domain"] if anchors is None else 0
    boundary_count = POINT_COUNTS["initial_conditions"] if anchors is None else 0
    return dde.data.PDE(
        timedomain,
        _pde,
        [displacement, velocity],
        num_domain=count,
        num_boundary=boundary_count,
        anchors=anchors,
        train_distribution="uniform",
    )


def create_data(cfg, seed):
    return create_persisted_data(TASK_KEY, cfg, seed, _data_factory)


def prepare_job_config(cfg, seed):
    return prepare_sampling_config(TASK_KEY, cfg, seed, _data_factory)


def create_network(cfg):
    return create_standard_network(cfg)


def _reference_bundle():
    X = np.linspace(0.0, 1.0, 10000, dtype=np.float32).reshape(-1, 1)
    return {"X": X, "y": _exact_solution(X).astype(np.float32)}


def _extra_metrics(model, X_ref, y_ref, y_pred):
    initial = np.asarray([[0.0]], dtype=np.float32)
    velocity = model.predict(
        initial, operator=lambda t, y: dde.grad.jacobian(y, t, i=0, j=0)
    )
    return {
        "initial_displacement_mse": float(np.mean((model.predict(initial) - 1.0) ** 2)),
        "initial_velocity_mse": float(np.mean(velocity**2)),
    }


def run_single(
    cfg,
    seed,
    verbose=True,
    track_reference_metrics=False,
    reference_interval=100,
    track_true_train_loss=False,
    true_train_loss_interval=100,
):
    return run_forward_task(
        task_key=TASK_KEY,
        cfg=cfg,
        seed=seed,
        create_data=create_data,
        create_network=create_network,
        reference_bundle=_reference_bundle,
        pde_operator=_pde,
        extra_metrics=_extra_metrics,
        verbose=verbose,
        track_reference_metrics=track_reference_metrics,
        reference_interval=reference_interval,
        track_true_train_loss=track_true_train_loss,
        true_train_loss_interval=true_train_loss_interval,
    )

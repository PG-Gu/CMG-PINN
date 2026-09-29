from __future__ import annotations
from src.paths import DATA_DIR as BUNDLED_DATA
import deepxde as dde
import numpy as np
from scipy.io import loadmat
import torch
from src.training.low_memory_forward_common import (
    create_persisted_data,
    create_standard_network,
    prepare_sampling_config,
    run_forward_task,
)

TASK_KEY = "kdv"
POINT_COUNTS = {"domain": 8000, "periodic_pairs": 200}
REFERENCE_PATH = BUNDLED_DATA / "KdV.mat"


def _verify_reference():
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(REFERENCE_PATH)


def _load_reference_grid():
    _verify_reference()
    data = loadmat(REFERENCE_PATH)
    x = np.asarray(data["x"], dtype=np.float32).reshape(-1)
    t = np.asarray(data["tt"], dtype=np.float32).reshape(-1)
    u = np.asarray(data["uu"], dtype=np.float32).T
    x = np.append(x, np.float32(1.0))
    u = np.concatenate((u, u[:, :1]), axis=1)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    return (
        np.column_stack((xx.ravel(), tt.ravel())).astype(np.float32),
        u.reshape(-1, 1),
    )


def _pde(x, y):
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_xx = dde.grad.hessian(y, x, i=0, j=0)
    u_xxx = dde.grad.jacobian(u_xx, x, i=0, j=0)
    return u_t + y * u_x + 0.0025 * u_xxx


def _is_left_boundary(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], -1.0)


def _hard_initial_condition(inputs, outputs):
    return torch.cos(torch.pi * inputs[:, 0:1]) + inputs[:, 1:2] * outputs


def _anchors():
    interior = np.column_stack(
        (
            np.random.uniform(-1.0, 1.0, POINT_COUNTS["domain"]),
            np.random.uniform(0.0, 1.0, POINT_COUNTS["domain"]),
        )
    )
    boundary = np.column_stack(
        (
            np.full(POINT_COUNTS["periodic_pairs"], -1.0),
            np.linspace(
                1 / POINT_COUNTS["periodic_pairs"], 1.0, POINT_COUNTS["periodic_pairs"]
            ),
        )
    )
    return np.vstack((interior, boundary)).astype(np.float32)


def _data_factory(stored_anchors):
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    bcs = [
        dde.icbc.PeriodicBC(geomtime, 0, _is_left_boundary, derivative_order=0),
        dde.icbc.PeriodicBC(geomtime, 0, _is_left_boundary, derivative_order=1),
    ]
    anchors = _anchors() if stored_anchors is None else stored_anchors
    return dde.data.TimePDE(
        geomtime,
        _pde,
        bcs,
        num_domain=0,
        num_boundary=0,
        num_initial=0,
        anchors=anchors,
    )


def create_data(cfg, seed):
    return create_persisted_data(TASK_KEY, cfg, seed, _data_factory)


def prepare_job_config(cfg, seed):
    return prepare_sampling_config(TASK_KEY, cfg, seed, _data_factory)


def create_network(cfg):
    return create_standard_network(cfg, output_transform=_hard_initial_condition)


def _reference_bundle():
    X, y = _load_reference_grid()
    return {"X": X, "y": y.astype(np.float32)}


def _extra_metrics(model, X_ref, y_ref, y_pred):
    times = np.linspace(0.0, 1.0, 201, dtype=np.float32).reshape(-1, 1)
    left = np.column_stack((np.full_like(times, -1.0), times))
    right = np.column_stack((np.full_like(times, 1.0), times))
    left_x = model.predict(
        left, operator=lambda x, y: dde.grad.jacobian(y, x, i=0, j=0)
    )
    right_x = model.predict(
        right, operator=lambda x, y: dde.grad.jacobian(y, x, i=0, j=0)
    )
    return {
        "periodic_value_mse": float(
            np.mean((model.predict(left) - model.predict(right)) ** 2)
        ),
        "periodic_derivative_mse": float(np.mean((left_x - right_x) ** 2)),
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

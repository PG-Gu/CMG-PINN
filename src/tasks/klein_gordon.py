from __future__ import annotations
import deepxde as dde
import numpy as np
import torch
from src.training.low_memory_forward_common import (
    create_persisted_data,
    create_standard_network,
    prepare_sampling_config,
    run_forward_task,
)

TASK_KEY = "klein_gordon"
POINT_COUNTS = {"domain": 1000, "boundary_each": 100, "initial": 100}


def _exact(x):
    xx, tt = (x[:, 0:1], x[:, 1:2])
    return xx * torch.cos(5 * torch.pi * tt) + (xx * tt) ** 3


def _exact_np(x):
    xx, tt = (np.asarray(x)[:, 0:1], np.asarray(x)[:, 1:2])
    return xx * np.cos(5 * np.pi * tt) + (xx * tt) ** 3


def _forcing(x):
    xx, tt = (x[:, 0:1], x[:, 1:2])
    u = _exact(x)
    u_tt = -25 * torch.pi**2 * xx * torch.cos(5 * torch.pi * tt) + 6 * xx**3 * tt
    u_xx = 6 * xx * tt**3
    return u_tt - u_xx + u**3


def _pde(x, y):
    u_tt = dde.grad.hessian(y, x, i=1, j=1)
    u_xx = dde.grad.hessian(y, x, i=0, j=0)
    return u_tt - u_xx + y**3 - _forcing(x)


def _sample_sets():
    interior = np.column_stack(
        (
            np.random.uniform(0.0, 1.0, POINT_COUNTS["domain"]),
            np.random.uniform(0.0, 1.0, POINT_COUNTS["domain"]),
        )
    )
    times = np.linspace(0.0, 1.0, POINT_COUNTS["boundary_each"], dtype=np.float32)
    left = np.column_stack((np.zeros_like(times), times))
    right = np.column_stack((np.ones_like(times), times))
    initial_x = np.linspace(0.0, 1.0, POINT_COUNTS["initial"], dtype=np.float32)
    initial = np.column_stack((initial_x, np.zeros_like(initial_x)))
    return tuple((a.astype(np.float32) for a in (interior, left, right, initial)))


def _data_factory(anchors):
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    sampled_interior, left, right, initial = _sample_sets()
    interior = (
        sampled_interior if anchors is None else np.asarray(anchors, dtype=np.float32)
    )
    initial_velocity = (
        -5 * np.pi * initial[:, 0:1] * np.sin(5 * np.pi * initial[:, 1:2])
    )
    bcs = [
        dde.icbc.PointSetBC(left, _exact_np(left), component=0),
        dde.icbc.PointSetBC(right, _exact_np(right), component=0),
        dde.icbc.PointSetBC(initial, _exact_np(initial), component=0),
        dde.icbc.PointSetOperatorBC(
            initial,
            initial_velocity.astype(np.float32),
            lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
        ),
    ]
    return dde.data.TimePDE(
        geomtime,
        _pde,
        bcs,
        num_domain=0,
        num_boundary=0,
        num_initial=0,
        anchors=interior,
        train_distribution="uniform",
    )


def create_data(cfg, seed):
    return create_persisted_data(TASK_KEY, cfg, seed, _data_factory)


def prepare_job_config(cfg, seed):
    return prepare_sampling_config(TASK_KEY, cfg, seed, _data_factory)


def create_network(cfg):
    return create_standard_network(cfg)


def _reference_bundle():
    x = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    t = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    X = np.column_stack((xx.ravel(), tt.ravel())).astype(np.float32)
    return {"X": X, "y": _exact_np(X).astype(np.float32)}


def _extra_metrics(model, X_ref, y_ref, y_pred):
    return {}


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

from __future__ import annotations
import deepxde as dde
import numpy as np
from src.training.low_memory_forward_common import (
    create_persisted_data,
    create_standard_network,
    prepare_sampling_config,
    run_forward_task,
)

TASK_KEY = "heat_equation"
POINT_COUNTS = {"domain": 2540, "boundary": 80, "initial": 160}


def _exact_solution(x):
    return np.exp(-0.4 * np.pi**2 * x[:, 1:2]) * np.sin(np.pi * x[:, 0:1])


def _pde(x, y):
    return dde.grad.jacobian(y, x, i=0, j=1) - 0.4 * dde.grad.hessian(y, x, i=0, j=0)


def _data_factory(anchors):
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(
        geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
    )
    counts = (
        POINT_COUNTS if anchors is None else {"domain": 0, "boundary": 0, "initial": 0}
    )
    return dde.data.TimePDE(
        geomtime,
        _pde,
        [bc, ic],
        num_domain=counts["domain"],
        num_boundary=counts["boundary"],
        num_initial=counts["initial"],
        anchors=anchors,
        solution=_exact_solution,
        num_test=2540,
    )


def create_data(cfg, seed):
    return create_persisted_data(TASK_KEY, cfg, seed, _data_factory)


def prepare_job_config(cfg, seed):
    return prepare_sampling_config(TASK_KEY, cfg, seed, _data_factory)


def create_network(cfg):
    return create_standard_network(cfg)


def _reference_bundle():
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    t = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    X = np.column_stack((xx.ravel(), tt.ravel())).astype(np.float32)
    return {"X": X, "y": _exact_solution(X).astype(np.float32)}


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
        verbose=verbose,
        track_reference_metrics=track_reference_metrics,
        reference_interval=reference_interval,
        track_true_train_loss=track_true_train_loss,
        true_train_loss_interval=true_train_loss_interval,
    )

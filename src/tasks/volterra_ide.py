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

TASK_KEY = "volterra_ide"
POINT_COUNTS = {"domain": 10, "boundary": 2, "quadrature_degree": 20}


def _exact(x):
    return np.exp(-x[:, 0:1]) * np.cosh(x[:, 0:1])


def _kernel(x, t):
    return np.exp(t - x)


def _pde(x, y, int_mat):
    derivative = dde.grad.jacobian(y, x, i=0, j=0)
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        indices, values, shape = int_mat
        sparse = torch.sparse_coo_tensor(
            torch.as_tensor(indices, dtype=torch.long),
            torch.as_tensor(values, dtype=y.dtype, device=y.device),
            tuple((int(v) for v in np.asarray(shape).reshape(-1))),
            device=y.device,
        ).coalesce()
        integral = torch.sparse.mm(sparse, y)
    else:
        integral = torch.as_tensor(int_mat, dtype=y.dtype, device=y.device) @ y
    n_physical = integral.shape[0]
    return derivative[:n_physical] + y[:n_physical] - integral


def _data_factory(_anchors):
    geom = dde.geometry.Interval(0, 5)
    bc = dde.icbc.DirichletBC(
        geom, lambda x: 1.0, lambda x, _: dde.utils.isclose(x[0], 0.0)
    )
    data = dde.data.IDE(
        geom,
        _pde,
        [bc],
        POINT_COUNTS["quadrature_degree"],
        kernel=_kernel,
        num_domain=POINT_COUNTS["domain"],
        num_boundary=POINT_COUNTS["boundary"],
        train_distribution="uniform",
        solution=_exact,
        num_test=100,
    )
    data.supports_posthoc_residual_metric = False
    data.low_memory_operator_layout_arrays = lambda: {
        "train_matrix": data.get_int_matrix(True),
        "test_matrix": data.get_int_matrix(False),
    }
    return data


def create_data(cfg, seed):
    return create_persisted_data(TASK_KEY, cfg, seed, _data_factory)


def prepare_job_config(cfg, seed):
    return prepare_sampling_config(TASK_KEY, cfg, seed, _data_factory)


def create_network(cfg):
    return create_standard_network(cfg)


def _reference_bundle():
    X = np.linspace(0.0, 5.0, 100, dtype=np.float32).reshape(-1, 1)
    return {"X": X, "y": _exact(X).astype(np.float32)}


def _extra_metrics(model, X_ref, y_ref, y_pred):
    return {}


def _residual_metric(model, data):
    int_mat = data.get_int_matrix(False)
    residual = model.predict(data.test_x, operator=lambda x, y: _pde(x, y, int_mat))
    return float(np.mean(np.abs(np.asarray(residual))))


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
        residual_evaluator=_residual_metric,
        extra_metrics=_extra_metrics,
        verbose=verbose,
        track_reference_metrics=track_reference_metrics,
        reference_interval=reference_interval,
        track_true_train_loss=track_true_train_loss,
        true_train_loss_interval=true_train_loss_interval,
    )

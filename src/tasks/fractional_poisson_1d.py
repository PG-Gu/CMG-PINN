from __future__ import annotations
from math import gamma
import deepxde as dde
import numpy as np
import torch
from src.training.low_memory_forward_common import (
    create_persisted_data,
    create_standard_network,
    prepare_sampling_config,
    run_forward_task,
)

TASK_KEY = "fractional_poisson_1d"
POINT_COUNTS = {"fractional_resolution": 101}
ALPHA = 1.5


def _solution(x):
    return x[:, 0:1] ** 3 * (1.0 - x[:, 0:1]) ** 3


def _solution_np(x):
    xx = np.asarray(x)[:, 0:1]
    return xx**3 * (1.0 - xx) ** 3


def _matmul(int_mat, y):
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        indices, values, shape = int_mat
        indices = torch.as_tensor(indices, dtype=torch.long)
        values = torch.as_tensor(values, dtype=y.dtype, device=y.device)
        shape = tuple((int(v) for v in np.asarray(shape).reshape(-1)))
        sparse = torch.sparse_coo_tensor(indices, values, shape, device=y.device)
        return torch.sparse.mm(sparse.coalesce(), y)
    return torch.as_tensor(int_mat, dtype=y.dtype, device=y.device) @ y


def _pde(x, y, int_mat):
    xx = x[:, 0:1]
    rhs = (
        gamma(4) / gamma(4 - ALPHA) * (xx ** (3 - ALPHA) + (1 - xx) ** (3 - ALPHA))
        - 3
        * gamma(5)
        / gamma(5 - ALPHA)
        * (xx ** (4 - ALPHA) + (1 - xx) ** (4 - ALPHA))
        + 3
        * gamma(6)
        / gamma(6 - ALPHA)
        * (xx ** (5 - ALPHA) + (1 - xx) ** (5 - ALPHA))
        - gamma(7) / gamma(7 - ALPHA) * (xx ** (6 - ALPHA) + (1 - xx) ** (6 - ALPHA))
    )
    lhs = _matmul(int_mat, y)
    return lhs - rhs[: lhs.shape[0]]


def _data_factory(_anchors):
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.DirichletBC(
        geom, lambda x: _solution_np(x), lambda _, on_boundary: on_boundary
    )
    data = dde.data.FPDE(
        geom,
        _pde,
        ALPHA,
        [bc],
        [POINT_COUNTS["fractional_resolution"]],
        meshtype="static",
        solution=lambda x: _solution_np(x),
    )
    data.supports_posthoc_residual_metric = False

    def operator_layout():
        data.test()
        arrays = {}
        for phase, matrix in (
            ("train", data.get_int_matrix(True)),
            ("test", data.get_int_matrix(False)),
        ):
            if isinstance(matrix, (list, tuple)) and len(matrix) == 3:
                arrays[f"{phase}_indices"] = matrix[0]
                arrays[f"{phase}_values"] = matrix[1]
                arrays[f"{phase}_shape"] = matrix[2]
            else:
                arrays[f"{phase}_matrix"] = matrix
        return arrays

    data.low_memory_operator_layout_arrays = operator_layout
    return data


def create_data(cfg, seed):
    return create_persisted_data(TASK_KEY, cfg, seed, _data_factory)


def prepare_job_config(cfg, seed):
    return prepare_sampling_config(TASK_KEY, cfg, seed, _data_factory)


def create_network(cfg):
    return create_standard_network(cfg, output_transform=lambda x, y: x * (1.0 - x) * y)


def _reference_bundle():
    X = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(-1, 1)
    return {"X": X, "y": _solution_np(X).astype(np.float32)}


def _extra_metrics(model, X_ref, y_ref, y_pred):
    boundary = np.asarray([[0.0], [1.0]], dtype=np.float32)
    return {"boundary_value_mse": float(np.mean(model.predict(boundary) ** 2))}


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

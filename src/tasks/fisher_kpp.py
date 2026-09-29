from __future__ import annotations
from functools import lru_cache
import deepxde as dde
import numpy as np
from scipy.integrate import solve_ivp
from src.training.low_memory_forward_common import (
    create_persisted_data,
    create_standard_network,
    prepare_sampling_config,
    run_forward_task,
)

TASK_KEY = "fisher_kpp"
POINT_COUNTS = {"domain": 1000, "initial": 256, "periodic_pairs": 100}
X_MAX = 2 * np.pi


def _initial_np(x):
    return np.exp(-(((x[:, 0:1] - np.pi) / (np.pi / 4)) ** 2))


def _pde(x, y):
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    u_xx = dde.grad.hessian(y, x, i=0, j=0)
    return u_t - 2.0 * u_xx - 5.0 * y * (1.0 - y)


class _PairedPeriodicBC(dde.icbc.BC):
    """Explicit paired periodic condition with no geometry reclassification."""

    def __init__(self, geom, left, right, derivative_order):
        super().__init__(geom, lambda _x, _on: True, component=0)
        self.left = np.asarray(left, dtype=np.float32)
        self.right = np.asarray(right, dtype=np.float32)
        self.derivative_order = int(derivative_order)
        self.pair_count = len(self.left)

    def collocation_points(self, _X):
        return np.vstack((self.left, self.right))

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        mid = beg + self.pair_count
        if self.derivative_order == 0:
            values = outputs[:, 0:1]
        else:
            values = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        return values[beg:mid] - values[mid:end]


def _sample_sets():
    interior = np.column_stack(
        (
            np.random.uniform(0.0, X_MAX, POINT_COUNTS["domain"]),
            np.random.uniform(0.0, 1.0, POINT_COUNTS["domain"]),
        )
    )
    initial_x = np.linspace(
        0.0, X_MAX, POINT_COUNTS["initial"], endpoint=False, dtype=np.float32
    )
    initial = np.column_stack((initial_x, np.zeros_like(initial_x)))
    times = np.linspace(0.0, 1.0, POINT_COUNTS["periodic_pairs"], dtype=np.float32)
    left = np.column_stack((np.zeros_like(times), times))
    right = np.column_stack((np.full_like(times, X_MAX), times))
    return tuple((a.astype(np.float32) for a in (interior, initial, left, right)))


def _data_factory(anchors):
    geom = dde.geometry.Interval(0, X_MAX)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    sampled_interior, initial, left, right = _sample_sets()
    interior = (
        sampled_interior if anchors is None else np.asarray(anchors, dtype=np.float32)
    )
    conditions = [
        dde.icbc.PointSetBC(
            initial, _initial_np(initial).astype(np.float32), component=0
        ),
        _PairedPeriodicBC(geomtime, left, right, derivative_order=0),
        _PairedPeriodicBC(geomtime, left, right, derivative_order=1),
    ]
    return dde.data.TimePDE(
        geomtime,
        _pde,
        conditions,
        num_domain=0,
        num_boundary=0,
        num_initial=0,
        anchors=interior,
        train_distribution="uniform",
    )


def _solve_reference(nx, nt):
    x = np.linspace(0.0, X_MAX, nx, endpoint=False, dtype=np.float64)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    wave_numbers = np.fft.fftfreq(nx, d=X_MAX / nx) * 2 * np.pi
    laplace_multiplier = -(wave_numbers**2)

    def rhs(_, state):
        u_xx = np.fft.ifft(laplace_multiplier * np.fft.fft(state)).real
        return 2.0 * u_xx + 5.0 * state * (1.0 - state)

    solution = solve_ivp(
        rhs,
        (0.0, 1.0),
        _initial_np(x.reshape(-1, 1)).reshape(-1),
        t_eval=t,
        rtol=1e-08,
        atol=1e-10,
        method="RK45",
    )
    if not solution.success:
        raise RuntimeError(
            f"Fisher--KPP reference integration failed: {solution.message}"
        )
    return (x, t, solution.y.T)


@lru_cache(maxsize=1)
def _reference_values():
    x, t, values = _solve_reference(256, 100)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    X = np.column_stack((xx.ravel(), tt.ravel())).astype(np.float32)
    y = values.reshape(-1, 1).astype(np.float32)
    return (X, y)


def create_data(cfg, seed):
    return create_persisted_data(TASK_KEY, cfg, seed, _data_factory)


def prepare_job_config(cfg, seed):
    return prepare_sampling_config(TASK_KEY, cfg, seed, _data_factory)


def create_network(cfg):
    return create_standard_network(cfg)


def _reference_bundle():
    X, y = _reference_values()
    return {"X": X, "y": y}


def _extra_metrics(model, X_ref, y_ref, y_pred):
    times = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(-1, 1)
    left = np.column_stack((np.zeros_like(times), times))
    right = np.column_stack((np.full_like(times, X_MAX), times))
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

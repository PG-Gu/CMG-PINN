from __future__ import annotations
from src.paths import DATA_DIR
from dataclasses import dataclass
from typing import Callable
import deepxde as dde
import numpy as np
import torch
from src.training.low_memory_forward_common import (
    create_persisted_data,
    create_standard_network,
    prepare_sampling_config,
    run_forward_task,
)

@dataclass(frozen=True)
class TaskDefinition:
    """Task-owned callbacks.  There is intentionally no generic PDE here."""

    key: str
    label: str
    data_builder: Callable[[np.ndarray | None], object]
    reference_builder: Callable[[], dict]
    pde_operator: Callable | None
    feature_transform: Callable | None = None
    output_transform: Callable | None = None
    extra_metrics: Callable | None = None
    prediction_selector: Callable | None = None
    loss_weights_resolver: Callable | None = None


def task_create_data(task: TaskDefinition, cfg: dict, seed: int):
    return create_persisted_data(task.key, cfg, seed, task.data_builder)


def task_prepare_config(task, cfg, seed):
    return prepare_sampling_config(task.key, cfg, seed, task.data_builder)


def task_create_network(task: TaskDefinition, cfg: dict):
    return create_standard_network(
        cfg,
        output_transform=task.output_transform,
        feature_transform=task.feature_transform,
    )


def task_run(task: TaskDefinition, cfg: dict, seed: int, **kwargs) -> dict:
    return run_forward_task(
        task_key=task.key,
        cfg=cfg,
        seed=seed,
        create_data=lambda local_cfg, local_seed: task_create_data(
            task, local_cfg, local_seed
        ),
        create_network=lambda local_cfg: task_create_network(task, local_cfg),
        reference_bundle=task.reference_builder,
        pde_operator=task.pde_operator,
        extra_metrics=task.extra_metrics,
        prediction_selector=task.prediction_selector,
        loss_weights_resolver=task.loss_weights_resolver,
        **kwargs,
    )


def relative_l2_components(
    y_true: np.ndarray, y_pred: np.ndarray, names: list[str]
) -> dict:
    """Component diagnostics; primary metric remains the global norm ratio."""
    out = {}
    for index, name in enumerate(names):
        out[f"{name}_l2_relative_error"] = float(
            dde.metrics.l2_relative_error(
                y_true[:, index : index + 1], y_pred[:, index : index + 1]
            )
        )
    return out


def require_asset(name):
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class FractionalQuadratureData(dde.data.Data):
    """PyTorch differentiable fractional-Laplacian collocation data."""

    def __init__(
        self,
        centers: np.ndarray,
        *,
        alpha: float | torch.Tensor,
        dimension: int,
        radial_nodes: int,
        direction_nodes: int,
        forcing: Callable[[torch.Tensor], torch.Tensor],
        exterior_radius: float = 1.0,
        external_parameters=(),
        external_names=(),
        observation_x: np.ndarray | None = None,
        observation_y: np.ndarray | None = None,
        observation_weight: float = 1.0,
        time_derivative_index: int | None = None,
        normalization: float = 1.0,
    ):
        self.centers = np.asarray(centers, dtype=np.float32)
        self.alpha = alpha
        self.dimension = int(dimension)
        self.radial_nodes = int(radial_nodes)
        self.direction_nodes = int(direction_nodes)
        self.forcing = forcing
        self.exterior_radius = float(exterior_radius)
        self.low_memory_external_trainable_variables = list(external_parameters)
        self.low_memory_external_parameter_names = list(external_names)
        self.train_x_all = self.centers
        self.train_x_bc = np.empty((0, self.centers.shape[1]), dtype=np.float32)
        self.train_x = self.centers
        self.train_y = None
        self.test_x = self.centers
        self.test_y = None
        self.num_bcs = []
        self.low_memory_operator_layout_arrays = {
            "centers": self.centers,
            "quadrature": np.asarray(
                [self.dimension, self.direction_nodes, self.radial_nodes],
                dtype=np.int64,
            ),
        }
        self.observation_x = (
            None
            if observation_x is None
            else np.asarray(observation_x, dtype=np.float32)
        )
        self.observation_y = (
            None
            if observation_y is None
            else np.asarray(observation_y, dtype=np.float32)
        )
        self.observation_weight = float(observation_weight)
        self.time_derivative_index = time_derivative_index
        self.normalization = float(normalization)

    def train_next_batch(self, batch_size=None):
        return (self.train_x, self.train_y)

    def test(self):
        return (self.test_x, self.test_y)

    def _directions(self, device, dtype):
        n = self.direction_nodes
        index = torch.arange(n, device=device, dtype=dtype)
        if self.dimension == 1:
            return torch.tensor([[-1.0], [1.0]], device=device, dtype=dtype)
        if self.dimension == 2:
            angle = 2.0 * torch.pi * (index + 0.5) / n
            return torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
        z = 1.0 - 2.0 * (index + 0.5) / n
        angle = (
            torch.pi
            * (3.0 - torch.sqrt(torch.tensor(5.0, device=device, dtype=dtype)))
            * index
        )
        r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        return torch.stack((r * torch.cos(angle), r * torch.sin(angle), z), dim=1)

    @staticmethod
    def reference_operator(
        x: torch.Tensor,
        *,
        alpha,
        dimension: int,
        radial_nodes: int,
        direction_nodes: int,
        exact_function: Callable[[torch.Tensor], torch.Tensor],
        exterior_radius: float = 1.0,
    ) -> torch.Tensor:
        """Apply the exact same discrete exterior-zero operator to a reference."""
        probe = object.__new__(FractionalQuadratureData)
        probe.dimension, probe.direction_nodes = (int(dimension), int(direction_nodes))
        directions = probe._directions(x.device, x.dtype)
        alpha_t = (
            alpha
            if torch.is_tensor(alpha)
            else torch.tensor(alpha, device=x.device, dtype=x.dtype)
        )
        alpha_t = alpha_t.to(device=x.device, dtype=x.dtype)
        r = (
            torch.arange(radial_nodes, device=x.device, dtype=x.dtype) + 0.5
        ) / radial_nodes
        shifted = x[:, None, None, :].expand(-1, len(directions), len(r), -1).clone()
        shifted[..., :dimension] = (
            shifted[..., :dimension]
            + directions[None, :, None, :] * r[None, None, :, None]
        )
        flat = shifted.reshape(-1, x.shape[1])
        values = exact_function(flat).reshape(len(x), len(directions), len(r), 1)
        inside = (
            torch.sum(flat[:, :dimension] * flat[:, :dimension], dim=1)
            <= exterior_radius**2
        ).reshape(len(x), len(directions), len(r), 1)
        values = torch.where(inside, values, torch.zeros_like(values))
        center = exact_function(x).reshape(len(x), 1, 1, 1)
        weights = r.pow(-(1.0 + alpha_t)).reshape(1, 1, -1, 1) / radial_nodes
        return torch.mean((center - values) * weights, dim=(1, 2))

    def fractional_operator(self, inputs: torch.Tensor, model) -> torch.Tensor:
        x = inputs
        dtype, device = (x.dtype, x.device)
        alpha = (
            self.alpha
            if torch.is_tensor(self.alpha)
            else torch.tensor(self.alpha, device=device, dtype=dtype)
        )
        alpha = alpha.to(device=device, dtype=dtype)
        directions = self._directions(device, dtype)
        r = (
            torch.arange(self.radial_nodes, device=device, dtype=dtype) + 0.5
        ) / self.radial_nodes
        shifts = directions[:, None, :] * r[None, :, None]
        shifted = x[:, None, None, :].expand(-1, len(directions), len(r), -1).clone()
        shifted[..., : self.dimension] = (
            shifted[..., : self.dimension] + shifts[None, :, :, :]
        )
        flat = shifted.reshape(-1, x.shape[1])
        values = model.net(flat)[:, :1].reshape(len(x), len(directions), len(r), 1)
        inside = (
            torch.sum(flat[:, : self.dimension] * flat[:, : self.dimension], dim=1)
            <= self.exterior_radius**2
        ).reshape(len(x), len(directions), len(r), 1)
        values = torch.where(inside, values, torch.zeros_like(values))
        center = model.net(x)[:, :1].reshape(len(x), 1, 1, 1)
        weights = r.pow(-(1.0 + alpha)).reshape(1, 1, -1, 1) / self.radial_nodes
        return self.normalization * torch.mean((center - values) * weights, dim=(1, 2))

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        residual = self.fractional_operator(inputs, model) - self.forcing(inputs)
        if self.time_derivative_index is not None:
            residual = residual + dde.grad.jacobian(
                outputs, inputs, i=0, j=self.time_derivative_index
            )
        losses = [loss_fn(torch.zeros_like(residual), residual)]
        if self.observation_x is not None:
            x_obs = torch.as_tensor(
                self.observation_x, device=inputs.device, dtype=inputs.dtype
            )
            y_obs = torch.as_tensor(
                self.observation_y, device=inputs.device, dtype=inputs.dtype
            )
            observation_error = model.net(x_obs)[:, : y_obs.shape[1]] - y_obs
            losses.append(
                self.observation_weight
                * loss_fn(torch.zeros_like(observation_error), observation_error)
            )
        return losses

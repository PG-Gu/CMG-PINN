"""PirateNet for PINN experiments."""

from __future__ import annotations
import torch
import torch.nn as nn
from deepxde.nn.pytorch.nn import NN
from deepxde.nn import initializers


class _PirateBlock(nn.Module):
    """One three-layer block with uv-gate and alpha residual skip."""

    def __init__(self, hidden_dim: int, init_fn, init_zero):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        for lin in (self.lin1, self.lin2, self.lin3):
            init_fn(lin.weight)
            init_zero(lin.bias)
        self.alpha = nn.Parameter(torch.zeros(1))
        self._act = nn.Tanh()

    def forward(self, x, u, v):
        identity = x
        h = self._act(self.lin1(x))
        h = h * u + (1.0 - h) * v
        h = self._act(self.lin2(h))
        h = h * u + (1.0 - h) * v
        h = self._act(self.lin3(h))
        return self.alpha * h + (1.0 - self.alpha) * identity


class PirateNetFNN(NN):
    """PirateNet network for PINN tasks."""

    def __init__(self, layer_sizes, kernel_initializer: str = "Glorot normal"):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        n_hidden = len(layer_sizes) - 2
        if n_hidden <= 0 or n_hidden % 3 != 0:
            raise ValueError(
                f"PirateNetFNN requires a positive number of hidden layers divisible by 3, got {n_hidden} from layer_sizes={layer_sizes}."
            )
        self.hidden_dim: int = layer_sizes[1]
        self.n_blocks: int = n_hidden // 3
        input_dim: int = layer_sizes[0]
        output_dim: int = layer_sizes[-1]
        init_fn = initializers.get(kernel_initializer)
        init_zero = initializers.get("zeros")

        def _make_linear(in_f: int, out_f: int) -> nn.Linear:
            lin = nn.Linear(in_f, out_f)
            init_fn(lin.weight)
            init_zero(lin.bias)
            return lin

        self.proj = _make_linear(input_dim, self.hidden_dim)
        self.u_branch = _make_linear(input_dim, self.hidden_dim)
        self.v_branch = _make_linear(input_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                _PirateBlock(self.hidden_dim, init_fn, init_zero)
                for _ in range(self.n_blocks)
            ]
        )
        self.out_linear = _make_linear(self.hidden_dim, output_dim)
        self._act = nn.Tanh()

    def residual_parameters(self):
        """Return the per-block alpha parameters (same API as ResidualFNN)."""
        return tuple((block.alpha for block in self.blocks))

    def residual_parameter_count(self) -> int:
        return sum((p.numel() for p in self.residual_parameters()))

    def output_parameters(self):
        """Expose final readout tensors for optimizer-routing policies."""
        return tuple(self.out_linear.parameters(recurse=False))

    def forward(self, inputs):
        x_in = inputs
        if self._input_transform is not None:
            x_in = self._input_transform(x_in)
        u = self._act(self.u_branch(x_in))
        v = self._act(self.v_branch(x_in))
        x = self.proj(x_in)
        for block in self.blocks:
            x = block(x, u, v)
        out = self.out_linear(x)
        if self._output_transform is not None:
            out = self._output_transform(inputs, out)
        return out

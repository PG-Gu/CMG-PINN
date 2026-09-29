"""Fixed multi-scale Fourier-feature network for controlled PINN baselines."""

from __future__ import annotations
import json
import torch
import torch.nn as nn
from deepxde.nn import initializers
from deepxde.nn.pytorch.nn import NN

FOURIER_FEATURE_IMPLEMENTATION = "fixed_multiscale_shared_fnn_v1"
FOURIER_FEATURE_SCALES = (1.0, 10.0)


class FourierFeatureFNN(NN):
    """Two fixed Fourier branches followed by one shared three-layer tanh FNN."""

    def __init__(
        self,
        layer_sizes,
        *,
        kernel_initializer="Glorot normal",
        fourier_scales=FOURIER_FEATURE_SCALES,
        fourier_seed=None,
        fourier_features_per_branch=None,
        input_lower=None,
        input_upper=None,
    ):
        super().__init__()
        sizes = [int(value) for value in layer_sizes]
        if len(sizes) != 5:
            raise ValueError(
                f"FourierFeatureFNN requires exactly three hidden layers; got layer_sizes={sizes}."
            )
        input_dim, width, width2, width3, output_dim = sizes
        if width != width2 or width != width3 or width % 2:
            raise ValueError(
                f"FourierFeatureFNN requires one even hidden width repeated three times; got layer_sizes={sizes}."
            )
        scales = tuple((float(value) for value in fourier_scales))
        if scales != FOURIER_FEATURE_SCALES:
            raise ValueError(
                f"The controlled baseline fixes Fourier scales to {FOURIER_FEATURE_SCALES}; got {scales}."
            )
        self.layer_sizes = sizes
        self.activation_name = "fourier_features"
        self.fourier_scales = scales
        self.fourier_seed = int(
            torch.initial_seed() if fourier_seed is None else fourier_seed
        )
        default_device = torch.empty(0).device
        lower = torch.as_tensor(
            [-1.0] * input_dim if input_lower is None else input_lower,
            dtype=torch.float32,
            device=default_device,
        )
        upper = torch.as_tensor(
            [1.0] * input_dim if input_upper is None else input_upper,
            dtype=torch.float32,
            device=default_device,
        )
        if lower.shape != (input_dim,) or upper.shape != (input_dim,):
            raise ValueError(
                "Fourier input bounds must match the task input dimension."
            )
        if torch.any(upper <= lower):
            raise ValueError(
                "Each Fourier input upper bound must exceed its lower bound."
            )
        self.register_buffer("input_lower", lower)
        self.register_buffer("input_upper", upper)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.fourier_seed)
        feature_dim = int(fourier_features_per_branch or width)
        if feature_dim <= 0:
            raise ValueError("fourier_features_per_branch must be positive.")
        self.fourier_features_per_branch = feature_dim
        for index, scale in enumerate(scales):
            matrix = (
                torch.randn(
                    input_dim,
                    feature_dim,
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                ).to(default_device)
                * scale
            )
            self.register_buffer(f"fourier_matrix_{index}", matrix)
        init_fn = initializers.get(kernel_initializer)
        init_zero = initializers.get("zeros")
        self.shared_linears = nn.ModuleList(
            [
                nn.Linear(2 * feature_dim, width),
                nn.Linear(width, width),
                nn.Linear(width, width),
            ]
        )
        self.output_linear = nn.Linear(2 * width, output_dim)
        for layer in [*self.shared_linears, self.output_linear]:
            init_fn(layer.weight)
            init_zero(layer.bias)
        self.activation = nn.Tanh()

    def _normalized_coordinates(self, inputs):
        return (
            2.0 * (inputs - self.input_lower) / (self.input_upper - self.input_lower)
            - 1.0
        )

    def _encode(self, coordinates, matrix):
        phase = coordinates @ matrix
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)

    def forward(self, inputs):
        transformed = inputs
        if self._input_transform is not None:
            transformed = self._input_transform(inputs)
        if transformed.shape[-1] != self.layer_sizes[0]:
            raise RuntimeError(
                "The task feature transform changed the Fourier input dimension; provide layer_sizes and bounds for the transformed coordinates."
            )
        coordinates = self._normalized_coordinates(transformed)
        branches = []
        for index in range(len(self.fourier_scales)):
            matrix = getattr(self, f"fourier_matrix_{index}")
            hidden = self._encode(coordinates, matrix)
            for layer in self.shared_linears:
                hidden = self.activation(layer(hidden))
            branches.append(hidden)
        output = self.output_linear(torch.cat(branches, dim=-1))
        if self._output_transform is not None:
            output = self._output_transform(inputs, output)
        return output

    def muon_backbone_parameters(self):
        return tuple((layer.weight for layer in self.shared_linears))

    def output_parameters(self):
        return tuple(self.output_linear.parameters(recurse=False))

    def implementation_manifest(self) -> dict:
        return {
            "implementation": FOURIER_FEATURE_IMPLEMENTATION,
            "scales": list(self.fourier_scales),
            "seed": self.fourier_seed,
            "features_per_branch": self.fourier_features_per_branch,
            "input_lower": self.input_lower.detach().cpu().tolist(),
            "input_upper": self.input_upper.detach().cpu().tolist(),
            "formula": "concat_sigma[cos(x_norm B_sigma), sin(x_norm B_sigma)]",
            "trainable_hidden_layers": 3,
            "shared_backbone": True,
        }

    def extra_repr(self) -> str:
        return json.dumps(self.implementation_manifest(), sort_keys=True)

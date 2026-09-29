"""Three-hidden-layer MLP with the paper's activation functions."""

import torch
from torch import nn
from deepxde.nn.pytorch.nn import NN
from deepxde.nn import initializers
from src.activations.gllf_activation import GLLFActivation
from src.activations.cmg_gelu_activation import CMGGELUActivation
from src.activations.activation_baselines import (
    SigmexdActivation,
    REActActivation,
    JagtapAdaptiveTanhActivation,
)


class Sin(nn.Module):

    def forward(self, x):
        return torch.sin(x)


def activation(name, width, sharing, bounds_mode, mu_init, I_init, mode, eps, options):
    if name == "tanh":
        return nn.Tanh()
    if name == "sin":
        return Sin()
    if name == "swish":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU(approximate="none")
    if name == "sigmexd":
        return SigmexdActivation()
    if name == "react":
        return REActActivation(init=options.get("react_init"))
    if name == "jagtap_adaptive_tanh":
        return JagtapAdaptiveTanhActivation(n=options.get("jagtap_n", 5.0))
    cls = {
        "gllf": GLLFActivation,
        "cmg_gelu": CMGGELUActivation,
    }[name]
    return cls(
        num_features=width,
        sharing=sharing,
        bounds_mode=bounds_mode,
        mu_init=mu_init,
        I_init=I_init,
        mode=mode,
        eps=eps,
    )


class GLLF_FNN(NN):
    """Initialize linear layers before activation parameters, preserving RNG order."""

    def __init__(
        self,
        layer_sizes,
        activation_name="tanh",
        kernel_initializer="Glorot normal",
        sharing="neuron",
        bounds_mode="tanh_wrapper",
        norm_mode="none",
        mu_init=0.5,
        I_init=0.5,
        gllf_mode="all",
        eps=1e-06,
        activation_options=None,
        cmg_implementation="reference",
    ):
        super().__init__()
        if norm_mode != "none":
            raise ValueError("The paper configurations use norm_mode=none.")
        if cmg_implementation != "reference":
            raise ValueError(
                "The paper configurations use the reference CMG implementation."
            )
        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation_name
        init = initializers.get(kernel_initializer)
        zero = initializers.get("zeros")
        self.linears = nn.ModuleList()
        for left, right in zip(layer_sizes[:-1], layer_sizes[1:]):
            linear = nn.Linear(left, right)
            init(linear.weight)
            zero(linear.bias)
            self.linears.append(linear)
        self.activations = nn.ModuleList(
            [
                activation(
                    activation_name,
                    width,
                    sharing,
                    bounds_mode,
                    mu_init,
                    I_init,
                    gllf_mode,
                    eps,
                    activation_options or {},
                )
                for width in layer_sizes[1:-1]
            ]
        )
        self.norms = nn.ModuleList([nn.Identity() for _ in self.activations])

    def output_parameters(self):
        return tuple(self.linears[-1].parameters(recurse=False))

    def forward(self, inputs):
        x = (
            self._input_transform(inputs)
            if self._input_transform is not None
            else inputs
        )
        for linear, norm, act in zip(self.linears[:-1], self.norms, self.activations):
            x = act(norm(linear(x)))
        x = self.linears[-1](x)
        return (
            self._output_transform(inputs, x)
            if self._output_transform is not None
            else x
        )

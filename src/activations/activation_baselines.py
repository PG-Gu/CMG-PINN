"""Activation-function baselines used by the controlled PINN comparisons."""

from __future__ import annotations
from collections.abc import Iterable
import torch
from torch import nn

ACTIVATION_BASELINE_NAMES = {
    "sigmexd",
    "react",
    "jagtap_adaptive_tanh",
    "cmg_gelu",
}
FIXED_ACTIVATION_NAMES = {"sin", "swish", "gelu"}
REACT_TANH_INIT = (-2.0, 0.0, -2.0, 0.0)
JAGTAP_DEFAULT_N = 5.0


def activation_options_from_config(cfg: dict) -> dict:
    """Extract only baseline-specific options from a task configuration."""
    return {
        "react_init": tuple(cfg.get("react_init", REACT_TANH_INIT)),
        "react_parameter_sharing": cfg.get("react_parameter_sharing", "layer"),
        "jagtap_n": float(cfg.get("jagtap_n", JAGTAP_DEFAULT_N)),
    }


def is_activation_baseline(name: str | None) -> bool:
    return name in ACTIVATION_BASELINE_NAMES


def is_fixed_activation(name: str | None) -> bool:
    """Return whether *name* is a stateless fixed activation baseline."""
    return name in FIXED_ACTIVATION_NAMES


class SigmexdActivation(nn.Module):
    """Fixed Sigmexd: ``x * sigmoid(-x)``."""

    def forward(self, x):
        return x * torch.sigmoid(-x)

    def auxiliary_parameters(self) -> Iterable[nn.Parameter]:
        return ()

    def activation_state(self) -> dict:
        return {"kind": "sigmexd", "trainable": False}


class REActActivation(nn.Module):
    """Stable REAct implementation with the pre-registered tanh initialization."""

    def __init__(self, init=None):
        super().__init__()
        init = REACT_TANH_INIT if init is None else init
        if len(init) != 4:
            raise ValueError("react_init must contain exactly four values: a b c d.")
        self.a = nn.Parameter(torch.tensor(float(init[0])))
        self.b = nn.Parameter(torch.tensor(float(init[1])))
        self.c = nn.Parameter(torch.tensor(float(init[2])))
        self.d = nn.Parameter(torch.tensor(float(init[3])))

    def forward(self, x):
        u = self.a * x + self.b
        v = self.c * x + self.d
        m = torch.maximum(torch.zeros_like(x), torch.maximum(u, v))
        numerator = torch.exp(-m) - torch.exp(u - m)
        denominator = torch.exp(-m) + torch.exp(v - m)
        return numerator / denominator

    def auxiliary_parameters(self) -> Iterable[nn.Parameter]:
        return (self.a, self.b, self.c, self.d)

    def activation_state(self) -> dict:
        return {
            "kind": "react",
            "a": float(self.a.detach().cpu()),
            "b": float(self.b.detach().cpu()),
            "c": float(self.c.detach().cpu()),
            "d": float(self.d.detach().cpu()),
        }


class JagtapAdaptiveTanhActivation(nn.Module):
    """Layer-wise trainable-slope tanh from the adaptive-activation formulation."""

    def __init__(self, n: float = JAGTAP_DEFAULT_N):
        super().__init__()
        n = float(n)
        if not torch.isfinite(torch.tensor(n)) or n <= 0:
            raise ValueError("jagtap_n must be a finite positive number.")
        self.n = n
        self.a = nn.Parameter(torch.tensor(1.0 / n))

    def forward(self, x):
        return torch.tanh(self.n * self.a * x)

    def auxiliary_parameters(self) -> Iterable[nn.Parameter]:
        return (self.a,)

    def activation_state(self) -> dict:
        a = float(self.a.detach().cpu())
        return {
            "kind": "jagtap_adaptive_tanh",
            "base_activation": "tanh",
            "n": self.n,
            "a": a,
            "n_times_a": self.n * a,
            "trainable": True,
            "sharing": "layer",
        }


def module_auxiliary_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    """Return a module's protocol parameters, or an empty tuple."""
    method = getattr(module, "auxiliary_parameters", None)
    return tuple(method()) if callable(method) else ()


def activation_state(net: nn.Module) -> dict:
    """Return serializable state for all baseline activation modules in ``net``."""
    state = {}
    seen = set()
    for index, module in enumerate(net.modules()):
        method = getattr(module, "activation_state", None)
        if not callable(method) or id(module) in seen:
            continue
        seen.add(id(module))
        state[f"activation_{index}"] = method()
    return state

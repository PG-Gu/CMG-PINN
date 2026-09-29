"""CMG modulation of the tanh approximation to GELU."""

from __future__ import annotations
import math
import torch
from torch import nn
from src.activations.gllf_activation import GLLFActivation

CMG_GELU_IMPLEMENTATION_VERSION = "pinn.cmg_gelu_activation.CMGGELUActivation.v1"
CMG_GELU_TANH_CUBIC_COEFFICIENT = 0.044715
CMG_GELU_SQRT_TWO_OVER_PI = math.sqrt(2.0 / math.pi)
CMG_GELU_FORMULA = "0.5*x*(1 + CMG_mu_I(tanh(sqrt(2/pi)*(x + 0.044715*x^3))))"


class CMGGELUActivation(nn.Module):
    """Tanh-approximate GELU with its inner tanh replaced by CMG."""

    def __init__(
        self,
        *,
        num_features: int,
        sharing: str = "layer",
        bounds_mode: str = "tanh_wrapper_fast",
        mu_init: float = 0.5,
        I_init: float = 0.5,
        mode: str = "all",
        eps: float = 1e-06,
    ) -> None:
        super().__init__()
        if sharing not in {"layer", "neuron"}:
            raise ValueError("CMG-GELU sharing must be 'layer' or 'neuron'.")
        if bounds_mode != "tanh_wrapper_fast":
            raise ValueError(
                "CMG-GELU requires bounds_mode='tanh_wrapper_fast' so its CMG argument is exactly tanh(z)."
            )
        self.cmg = GLLFActivation(
            num_features=num_features,
            sharing=sharing,
            bounds_mode=bounds_mode,
            mu_init=mu_init,
            I_init=I_init,
            mode=mode,
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = CMG_GELU_SQRT_TWO_OVER_PI * (x + CMG_GELU_TANH_CUBIC_COEFFICIENT * x.pow(3))
        return 0.5 * x * (1.0 + self.cmg(z))

    def activation_state(self) -> dict:
        return {
            "kind": "cmg_gelu",
            "implementation": CMG_GELU_IMPLEMENTATION_VERSION,
            "formula": CMG_GELU_FORMULA,
            "sharing": self.cmg.sharing,
            "bounds_mode": self.cmg.bounds_mode,
            "mu": self.cmg.mu.detach().cpu().flatten().tolist(),
            "I": self.cmg.I_param.detach().cpu().flatten().tolist(),
        }

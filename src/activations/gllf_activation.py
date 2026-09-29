""""""

import math
import torch
import torch.nn as nn


def _as_tensor(value, ref):
    """Cast *value* to a tensor on the same device / dtype as *ref*."""
    if torch.is_tensor(value):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(value, device=ref.device, dtype=ref.dtype)


def _safe_logit(p):
    return torch.log(p) - torch.log1p(-p)


def _as_optional_buffer(value):
    """Convert an optional scalar/list bound into a 1D tensor buffer."""
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.get_default_dtype())
    if tensor.ndim == 0:
        return tensor.reshape(1)
    return tensor.reshape(-1)


def cm_gllf_torch(x, mu, xmin, xmax, yL, yR, I=0.5, rectify=False, eps=1e-06):
    """Vectorized CM-GLLF: exact logistic branch + one-step Newton logit branch."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    elif not torch.is_floating_point(x):
        x = x.to(torch.get_default_dtype())
    mu = _as_tensor(mu, x)
    xmin = _as_tensor(xmin, x)
    xmax = _as_tensor(xmax, x)
    yL = _as_tensor(yL, x)
    yR = _as_tensor(yR, x)
    I = _as_tensor(I, x)
    dx = xmax - xmin
    dy = yR - yL
    t = ((x - xmin) / dx).clamp(0.0, 1.0)
    tc = t.clamp(eps, 1.0 - eps)
    mu_logistic = mu.clamp_min(eps)
    mu_logit = mu.clamp_max(1.0 - eps)
    alpha_logistic = 1.0 / mu_logistic - 2.0
    z_logistic = _safe_logit(tc) + alpha_logistic * (tc - I)
    s_logistic = torch.sigmoid(z_logistic)
    s_logistic = torch.where(t <= 0.0, torch.zeros_like(s_logistic), s_logistic)
    s_logistic = torch.where(t >= 1.0, torch.ones_like(s_logistic), s_logistic)
    alpha_logit = 1.0 / (1.0 - mu_logit) - 2.0
    numerator = t + alpha_logit * I * t * (1.0 - t)
    denominator = 1.0 + alpha_logit * t * (1.0 - t)
    s_logit = (numerator / denominator.clamp_min(eps)).clamp(0.0, 1.0)
    xI = xmin + I * dx
    y_step = torch.where(x < xI, yL, torch.where(x > xI, yR, torch.maximum(yL, yR)))
    y_constant = yL + I * dy
    y_logistic = yL + dy * s_logistic
    y_logit = yL + dy * s_logit
    y = torch.where(
        mu == 1.0,
        y_constant,
        torch.where(mu == 0.0, y_step, torch.where(mu <= 0.5, y_logistic, y_logit)),
    )
    if rectify:
        y = torch.clamp_min(y, 0.0)
    return y


def _logit_eps_bounds(eps):
    eps = float(eps)
    lo = math.log(eps) - math.log1p(-eps)
    hi = math.log1p(-eps) - math.log(eps)
    return (lo, hi)


def cm_gllf_tanh_wrapper_fast(x, mu, I=0.5, eps=1e-06):
    """Fast path for tanh-wrapper CM-GLLF on fixed [-1, 1] bounds."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    elif not torch.is_floating_point(x):
        x = x.to(torch.get_default_dtype())
    mu = _as_tensor(mu, x)
    I = _as_tensor(I, x)
    lo, hi = _logit_eps_bounds(eps)
    z = (2.0 * x).clamp(min=lo, max=hi)
    t = torch.sigmoid(z)
    alpha_logistic = 1.0 / mu.clamp_min(eps) - 2.0
    y_logistic = 2.0 * torch.sigmoid(z + alpha_logistic * (t - I)) - 1.0
    mu_logit = mu.clamp_max(1.0 - eps)
    alpha_logit = 1.0 / (1.0 - mu_logit) - 2.0
    t1m = t * (1.0 - t)
    numerator = t + alpha_logit * I * t1m
    denominator = 1.0 + alpha_logit * t1m
    y_logit = 2.0 * (numerator / denominator.clamp_min(eps)).clamp(0.0, 1.0) - 1.0
    mask = (mu <= 0.5).detach()
    if bool(torch.all(mask)):
        return y_logistic
    if bool(torch.all(~mask)):
        return y_logit
    return torch.where(mask.expand_as(x), y_logistic, y_logit)


def _encode_mu(mu_val, n, mode, eps):
    """Encode *mu_val* (float or "uniform") into logit-space raw parameter."""
    if isinstance(mu_val, str) and mu_val == "uniform":
        if mode == "logistic":
            lo, hi = (0.125, 0.375)
        elif mode == "logit":
            lo, hi = (0.625, 0.875)
        else:
            lo, hi = (0.25, 0.75)
        vals = torch.empty(n).uniform_(lo, hi)
    else:
        vals = torch.full((n,), float(mu_val))
    if mode == "logistic":
        internal = (2.0 * vals).clamp(eps, 1.0 - eps)
    elif mode == "logit":
        internal = (2.0 * (vals - 0.5)).clamp(eps, 1.0 - eps)
    else:
        internal = vals.clamp(eps, 1.0 - eps)
    return torch.logit(internal)


def _encode_I(I_val, n, eps):
    """Encode *I_val* (float or "uniform") into logit-space raw parameter."""
    if isinstance(I_val, str) and I_val == "uniform":
        vals = torch.empty(n).uniform_(0.0, 1.0)
    else:
        vals = torch.full((n,), float(I_val))
    return torch.logit(vals.clamp(eps, 1.0 - eps))


def _decode_mu(mu_raw, mode):
    """Decode logit-space *mu_raw* back to mu ∈ (0,1)."""
    if mode == "logistic":
        return 0.5 * torch.sigmoid(mu_raw)
    if mode == "logit":
        return 0.5 + 0.5 * torch.sigmoid(mu_raw)
    return torch.sigmoid(mu_raw)


def _decode_I(I_raw):
    return torch.sigmoid(I_raw)


class GLLFActivation(nn.Module):
    """Trainable GLLF activation for PINN hidden layers."""

    def __init__(
        self,
        num_features: int,
        sharing: str = "neuron",
        bounds_mode: str = "tanh_wrapper",
        mu_init=0.5,
        I_init=0.5,
        mode: str = "all",
        eps: float = 1e-06,
        fixed_xmin=None,
        fixed_xmax=None,
        fixed_yL=None,
        fixed_yR=None,
    ):
        super().__init__()
        if sharing not in ("neuron", "layer"):
            raise ValueError("sharing must be 'neuron' or 'layer'")
        if bounds_mode not in (
            "tanh_wrapper",
            "tanh_wrapper_fast",
            "sin_wrapper",
            "fixed_bounds",
            "none_minmax",
            "layernorm_minmax",
            "batchnorm_minmax",
        ):
            raise ValueError(
                "bounds_mode must be one of 'tanh_wrapper', 'tanh_wrapper_fast', 'sin_wrapper', 'fixed_bounds', 'none_minmax', 'layernorm_minmax', or 'batchnorm_minmax'"
            )
        if mode not in ("all", "logistic", "logit"):
            raise ValueError("mode must be 'all', 'logistic', or 'logit'")
        self.num_features = num_features
        self.sharing = sharing
        self.bounds_mode = bounds_mode
        self.mode = mode
        self.eps = eps
        self.register_buffer("fixed_xmin", _as_optional_buffer(fixed_xmin))
        self.register_buffer("fixed_xmax", _as_optional_buffer(fixed_xmax))
        self.register_buffer("fixed_yL", _as_optional_buffer(fixed_yL))
        self.register_buffer("fixed_yR", _as_optional_buffer(fixed_yR))
        n = 1 if sharing == "layer" else num_features
        if bounds_mode == "fixed_bounds":
            for buffer_name in ("fixed_xmin", "fixed_xmax", "fixed_yL", "fixed_yR"):
                buffer = getattr(self, buffer_name)
                if buffer is None:
                    raise ValueError(
                        f"bounds_mode='fixed_bounds' requires {buffer_name} to be provided."
                    )
                if buffer.numel() not in {1, num_features}:
                    raise ValueError(
                        f"{buffer_name} must have 1 or {num_features} values for bounds_mode='fixed_bounds', found shape {tuple(buffer.shape)}."
                    )
        self.mu_raw = nn.Parameter(_encode_mu(mu_init, n, mode, eps))
        self.I_raw = nn.Parameter(_encode_I(I_init, n, eps))

    @property
    def mu(self):
        return _decode_mu(self.mu_raw, self.mode)

    @property
    def I_param(self):
        return _decode_I(self.I_raw)

    def forward(self, x):
        mu = self.mu.to(x.device, x.dtype)
        I = self.I_param.to(x.device, x.dtype)
        if x.dim() == 2:
            mu = mu.unsqueeze(0)
            I = I.unsqueeze(0)
        if self.bounds_mode == "tanh_wrapper_fast":
            return cm_gllf_tanh_wrapper_fast(x, mu=mu, I=I, eps=self.eps)
        if self.bounds_mode in {"tanh_wrapper", "sin_wrapper"}:
            if self.bounds_mode == "tanh_wrapper":
                x_inner = torch.tanh(x)
            else:
                x_inner = torch.sin(x)
            return cm_gllf_torch(
                x_inner,
                mu=mu,
                xmin=-1.0,
                xmax=1.0,
                yL=-1.0,
                yR=1.0,
                I=I,
                rectify=False,
                eps=self.eps,
            )
        if self.bounds_mode == "fixed_bounds":
            return cm_gllf_torch(
                x,
                mu=mu,
                xmin=self.fixed_xmin.to(device=x.device, dtype=x.dtype),
                xmax=self.fixed_xmax.to(device=x.device, dtype=x.dtype),
                yL=self.fixed_yL.to(device=x.device, dtype=x.dtype),
                yR=self.fixed_yR.to(device=x.device, dtype=x.dtype),
                I=I,
                rectify=False,
                eps=self.eps,
            )
        else:
            xmin = x.detach().min()
            xmax = x.detach().max()
            return cm_gllf_torch(
                x,
                mu=mu,
                xmin=xmin,
                xmax=xmax,
                yL=xmin,
                yR=xmax,
                I=I,
                rectify=False,
                eps=self.eps,
            )

    def extra_repr(self):
        return f"num_features={self.num_features}, sharing={self.sharing!r}, bounds_mode={self.bounds_mode!r}, mode={self.mode!r}"

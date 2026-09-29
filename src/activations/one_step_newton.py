import torch


def _as_tensor(value, ref):
    if torch.is_tensor(value):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(value, device=ref.device, dtype=ref.dtype)


def _safe_logit(p):
    return torch.log(p) - torch.log1p(-p)


def cm_gllf_torch(x, mu, xmin, xmax, yL, yR, I=0.5, rectify=False, eps=1e-06):
    """
    Vectorized CM-GLLF with an exact logistic branch and a one-step Newton
    approximation for the logit branch.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    mu, I : float or torch.Tensor
        Scalars or tensors broadcastable with x.
    xmin, xmax, yL, yR : float or torch.Tensor
        Scalars or tensors broadcastable with x.
    eps : float, optional
        Numerical clamp for stable logits and divisions.
    """
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
    s_logit = numerator / denominator.clamp_min(eps)
    s_logit = s_logit.clamp(0.0, 1.0)
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

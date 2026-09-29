"""Muon and auxiliary Adam parameter groups used in the paper."""

from __future__ import annotations
import torch
from src.training.muon import SingleDeviceMuonWithAuxAdam
from src.activations.activation_baselines import module_auxiliary_parameters
from src.activations.gllf_activation import GLLFActivation

MUON_MATRIX_POLICIES = {"backbone", "internal_2d"}


def _add_param(params: list[torch.nn.Parameter], ids: set[int], param) -> None:
    if (
        param is not None
        and getattr(param, "requires_grad", False)
        and (id(param) not in ids)
    ):
        params.append(param)
        ids.add(id(param))


def _output_parameter_ids(net) -> set[int]:
    """Return output-layer parameters through the small network protocol."""
    if hasattr(net, "output_parameters"):
        return {id(param) for param in net.output_parameters() if param is not None}
    linears = list(getattr(net, "linears", []))
    if linears:
        return {
            id(param)
            for param in linears[-1].parameters(recurse=False)
            if param is not None
        }
    return set()


def _backbone_muon_ids(net) -> set[int]:
    """Existing policy: only ordinary non-output backbone linear weights."""
    if hasattr(net, "muon_backbone_parameters"):
        return {
            id(param)
            for param in net.muon_backbone_parameters()
            if param is not None and param.requires_grad
        }
    linears = list(getattr(net, "linears", []))
    return {
        id(linear.weight)
        for linear in linears[:-1]
        if getattr(linear, "weight", None) is not None and linear.weight.requires_grad
    }


def _select_muon_ids(net, named_params, matrix_policy: str) -> set[int]:
    if matrix_policy not in MUON_MATRIX_POLICIES:
        raise ValueError(
            f"Unknown Muon matrix policy {matrix_policy!r}; expected one of {sorted(MUON_MATRIX_POLICIES)}."
        )
    if matrix_policy == "backbone":
        return _backbone_muon_ids(net)
    output_ids = _output_parameter_ids(net)
    return {
        id(param)
        for _, param in named_params
        if param.requires_grad and param.ndim == 2 and (id(param) not in output_ids)
    }


def _auxiliary_ids(
    net, muon_ids: set[int]
) -> tuple[list[torch.nn.Parameter], set[int]]:
    """Collect activation parameters and trainable residual coefficients."""
    aux_ids: set[int] = set()
    aux_params: list[torch.nn.Parameter] = []
    for module in net.modules():
        if isinstance(module, GLLFActivation):
            _add_param(aux_params, aux_ids, getattr(module, "mu_raw", None))
            _add_param(aux_params, aux_ids, getattr(module, "I_raw", None))
        elif isinstance(
            module, (torch.nn.LayerNorm, torch.nn.modules.batchnorm._BatchNorm)
        ):
            for param in module.parameters(recurse=False):
                _add_param(aux_params, aux_ids, param)
        else:
            for param in module_auxiliary_parameters(module):
                _add_param(aux_params, aux_ids, param)
    if hasattr(net, "residual_parameters"):
        for param in net.residual_parameters():
            _add_param(aux_params, aux_ids, param)
    filtered = [param for param in aux_params if id(param) not in muon_ids]
    return (filtered, {id(param) for param in filtered})


def build_muon_with_aux_adam(
    net,
    weight_lr: float,
    aux_lr: float,
    *,
    extra_aux_params=(),
    extra_aux_lr: float | None = None,
    matrix_policy: str = "backbone",
    require_muon: bool = True,
):
    """Build a Muon-plus-AuxAdam optimizer and its parameter-routing manifest."""
    named_params = [
        (name, param) for name, param in net.named_parameters() if param.requires_grad
    ]
    all_ids = {id(param) for _, param in named_params}
    muon_ids = _select_muon_ids(net, named_params, matrix_policy)
    unknown_muon_ids = muon_ids - all_ids
    if unknown_muon_ids:
        raise AssertionError(
            "Muon routing selected parameters outside net.named_parameters()."
        )
    if require_muon and (not muon_ids):
        raise ValueError(
            "Muon optimizer has an empty Muon group. Use matrix_policy='internal_2d' for networks such as PirateNet, or select a network with backbone matrices."
        )
    output_ids = _output_parameter_ids(net)
    if muon_ids & output_ids:
        raise AssertionError("Output-layer parameters must never enter the Muon group.")
    non_matrix_muon = [
        name
        for name, param in named_params
        if id(param) in muon_ids and param.ndim != 2
    ]
    if non_matrix_muon:
        raise AssertionError(f"Muon parameters must all be 2D: {non_matrix_muon}")
    aux_params, aux_ids = _auxiliary_ids(net, muon_ids)
    extra_ids: set[int] = set()
    extra_params: list[torch.nn.Parameter] = []
    for param in extra_aux_params:
        if id(param) in muon_ids or id(param) in aux_ids:
            continue
        _add_param(extra_params, extra_ids, param)
    ordinary_ids = all_ids - muon_ids - aux_ids - extra_ids
    muon_params = [param for _, param in named_params if id(param) in muon_ids]
    ordinary_params = [param for _, param in named_params if id(param) in ordinary_ids]
    groups = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "lr": float(weight_lr),
                "momentum": 0.95,
                "weight_decay": 0.01,
                "use_muon": True,
            }
        )
    if ordinary_params:
        groups.append(
            {
                "params": ordinary_params,
                "lr": float(weight_lr),
                "betas": (0.9, 0.95),
                "eps": 1e-08,
                "weight_decay": 0.01,
                "use_muon": False,
            }
        )
    if aux_params:
        groups.append(
            {
                "params": aux_params,
                "lr": float(aux_lr),
                "betas": (0.9, 0.95),
                "eps": 1e-08,
                "weight_decay": 0.0,
                "use_muon": False,
            }
        )
    if extra_params:
        groups.append(
            {
                "params": extra_params,
                "lr": float(weight_lr if extra_aux_lr is None else extra_aux_lr),
                "betas": (0.9, 0.95),
                "eps": 1e-08,
                "weight_decay": 0.0,
                "use_muon": False,
            }
        )
    if not groups:
        raise ValueError("Muon optimizer received no trainable parameters.")
    return SingleDeviceMuonWithAuxAdam(groups)

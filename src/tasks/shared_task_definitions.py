from __future__ import annotations
from functools import lru_cache
import deepxde as dde
import numpy as np
import torch
from src.training.shared_task_training import (
    FractionalQuadratureData,
    TaskDefinition,
    relative_l2_components,
    require_asset,
)


def _grid(*axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([item.reshape(-1) for item in mesh]).astype(np.float32)


def _safe_component_metrics(names):

    def metrics(model, X, y_true, y_pred):
        return relative_l2_components(y_true, y_pred, names)

    return metrics


@lru_cache(maxsize=8)
def _asset_payload(name: str) -> dict[str, np.ndarray]:
    """Load a verified compact asset into memory and close the NPZ handle."""
    with np.load(require_asset(name)) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _time_data(geom, pde, bcs, *, domain, boundary=0, initial=0, anchors=None):
    if anchors is None:
        return dde.data.TimePDE(
            geom,
            pde,
            bcs,
            num_domain=domain,
            num_boundary=boundary,
            num_initial=initial,
        )
    return dde.data.TimePDE(
        geom, pde, bcs, num_domain=0, num_boundary=0, num_initial=0, anchors=anchors
    )


def _laplace_disk() -> TaskDefinition:

    def exact(x):
        return x[:, 0:1] * np.cos(x[:, 1:2])

    def pde(x, y):
        return (
            x[:, :1] * dde.grad.jacobian(y, x, i=0, j=0)
            + x[:, :1] ** 2 * dde.grad.hessian(y, x, i=0, j=0)
            + dde.grad.hessian(y, x, i=1, j=1)
        )

    def data(anchors):
        geom = dde.geometry.Rectangle([0, 0], [1, 2 * np.pi])
        bc = dde.icbc.DirichletBC(
            geom,
            lambda x: np.cos(x[:, 1:2]),
            lambda x, on: on and dde.utils.isclose(x[0], 1.0),
        )
        counts = (2540, 80) if anchors is None else (0, 0)
        return dde.data.PDE(
            geom, pde, bc, num_domain=counts[0], num_boundary=counts[1], anchors=anchors
        )

    def feature(x):
        return torch.cat(
            (x[:, :1] * torch.sin(x[:, 1:2]), x[:, :1] * torch.cos(x[:, 1:2])), dim=1
        )

    def ref():
        return {
            "X": _grid(
                np.linspace(0, 1, 256, dtype=np.float32),
                np.linspace(0, 2 * np.pi, 256, endpoint=False, dtype=np.float32),
            ),
            "y": exact(
                _grid(
                    np.linspace(0, 1, 256, dtype=np.float32),
                    np.linspace(0, 2 * np.pi, 256, endpoint=False, dtype=np.float32),
                )
            ),
        }

    return TaskDefinition(
        "laplace_disk_2d_v2",
        "Laplace disk (2D)",
        data,
        ref,
        pde,
        feature_transform=feature,
    )


def _helmholtz() -> TaskDefinition:
    n, k = (2, 2 * np.pi)

    def exact(x):
        return np.sin(n * np.pi * x[:, :1]) * np.sin(n * np.pi * x[:, 1:2])

    def forcing(x):
        return (
            (2 * (n * np.pi) ** 2 - k**2)
            * torch.sin(n * np.pi * x[:, :1])
            * torch.sin(n * np.pi * x[:, 1:2])
        )

    def pde(x, y):
        return (
            -dde.grad.hessian(y, x, i=0, j=0)
            - dde.grad.hessian(y, x, i=1, j=1)
            - k**2 * y
            - forcing(x)
        )

    def data(anchors):
        geom = dde.geometry.Rectangle([0, 0], [1, 1])
        counts = (400, 0) if anchors is None else (0, 0)
        return dde.data.PDE(
            geom, pde, [], num_domain=counts[0], num_boundary=counts[1], anchors=anchors
        )

    def transform(x, y):
        return x[:, :1] * (1 - x[:, :1]) * x[:, 1:2] * (1 - x[:, 1:2]) * y

    def ref():
        X = _grid(
            np.linspace(0, 1, 101, dtype=np.float32),
            np.linspace(0, 1, 101, dtype=np.float32),
        )
        return {"X": X, "y": exact(X).astype(np.float32)}

    return TaskDefinition(
        "helmholtz_2d_v2", "Helmholtz 2D", data, ref, pde, output_transform=transform
    )


def _inverse_poisson() -> TaskDefinition:

    def exact_u(x):
        return np.sin(np.pi * x[:, :1])

    def exact_q(x):
        return -np.pi**2 * np.sin(np.pi * x[:, :1])

    def pde(x, y):
        return dde.grad.hessian(y, x, i=0, j=0) - y[:, 1:2]

    def data(anchors):
        geom = dde.geometry.Interval(0, 1)
        xobs = np.linspace(0.005, 0.995, 100, dtype=np.float32)[:, None]
        bcs = [
            dde.icbc.DirichletBC(
                geom, lambda x: np.zeros((len(x), 1)), lambda _, on: on, component=0
            ),
            dde.icbc.PointSetBC(xobs, exact_u(xobs), component=0),
        ]
        counts = (200, 2) if anchors is None else (0, 0)
        return dde.data.PDE(
            geom,
            pde,
            bcs,
            num_domain=counts[0],
            num_boundary=counts[1],
            anchors=anchors,
        )

    def ref():
        X = np.linspace(0, 1, 1000, dtype=np.float32)[:, None]
        return {"X": X, "y": exact_q(X).astype(np.float32)}

    def selector(y):
        return y[:, 1:2]

    def extra(model, X, y, p):
        return {
            "solution_l2_relative_error": float(
                dde.metrics.l2_relative_error(exact_u(X), p[:, :1])
            ),
            "q_l2_relative_error": float(dde.metrics.l2_relative_error(y, p[:, 1:2])),
        }

    def weights(_, __):
        return [1.0, 100.0, 1000.0]

    return TaskDefinition(
        "inverse_poisson_field_v2",
        "Inverse Poisson field",
        data,
        ref,
        pde,
        extra_metrics=extra,
        prediction_selector=selector,
        loss_weights_resolver=weights,
    )


def _inverse_brinkman() -> TaskDefinition:
    nu_true = K_true = 0.001
    porosity, viscosity, forcing_value = (0.4, 0.001, 1.0)

    def exact(x):
        r = np.sqrt(viscosity * porosity / (nu_true * K_true))
        return (
            forcing_value
            * K_true
            / viscosity
            * (1 - np.cosh(r * (x - 0.5)) / np.cosh(0.5 * r))
        )

    def data(anchors):
        ve = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        K = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        def pde(x, y):
            return (
                -(ve / porosity) * dde.grad.hessian(y, x, i=0, j=0)
                + viscosity * y / K
                - forcing_value
            )

        geom = dde.geometry.Interval(0, 1)
        xobs = np.linspace(0.1, 0.9, 5, dtype=np.float32)[:, None]
        bcs = [dde.icbc.PointSetBC(xobs, exact(xobs))]
        counts = (100, 0) if anchors is None else (0, 0)
        result = dde.data.PDE(
            geom,
            pde,
            bcs,
            num_domain=counts[0],
            num_boundary=counts[1],
            anchors=anchors,
        )
        result.low_memory_external_trainable_variables = [ve, K]
        result.low_memory_external_parameter_names = ["nu_e", "K"]
        result.low_memory_pde_operator = pde
        result.low_memory_metric_override = lambda model, X, y, p: {
            "nu_e_relative_error": float(
                abs(ve.detach().cpu().item() - nu_true) / nu_true
            ),
            "K_relative_error": float(abs(K.detach().cpu().item() - K_true) / K_true),
            "parameter_mean_relative_error": float(
                (
                    abs(ve.detach().cpu().item() - nu_true) / nu_true
                    + abs(K.detach().cpu().item() - K_true) / K_true
                )
                / 2
            ),
        }
        return result

    def pde_marker(x, y):
        return y

    def transform(x, y):
        return x * (1 - x) * y

    def ref():
        X = np.linspace(0, 1, 500, dtype=np.float32)[:, None]
        return {"X": X, "y": exact(X).astype(np.float32)}

    return TaskDefinition(
        "inverse_brinkman_v2",
        "Inverse Brinkman",
        data,
        ref,
        pde_marker,
        output_transform=transform,
    )


def _fractional_diffusion() -> TaskDefinition:
    alpha = 1.8

    def exact(x):
        return (x[:, :1] * (1 - x[:, :1]) * x[:, 1:2] ** 2).astype(np.float32)

    def exact_torch(x):
        return x[:, :1] * (1 - x[:, :1]) * x[:, 1:2] ** 2

    def data(anchors):
        geom = dde.geometry.GeometryXTime(
            dde.geometry.Interval(0, 1), dde.geometry.TimeDomain(0, 1)
        )
        centers = np.asarray(
            anchors if anchors is not None else geom.random_points(400),
            dtype=np.float32,
        )
        return FractionalQuadratureData(
            centers,
            alpha=alpha,
            dimension=1,
            radial_nodes=52,
            direction_nodes=2,
            forcing=lambda x: FractionalQuadratureData.reference_operator(
                x,
                alpha=alpha,
                dimension=1,
                radial_nodes=52,
                direction_nodes=2,
                exact_function=exact_torch,
            )
            + 2 * x[:, 1:2] * x[:, :1] * (1 - x[:, :1]),
            time_derivative_index=1,
        )

    def transform(x, y):
        return x[:, :1] * (1 - x[:, :1]) * x[:, 1:2] * y

    def ref():
        X = _grid(
            np.linspace(0, 1, 256, dtype=np.float32),
            np.linspace(0, 1, 101, dtype=np.float32),
        )
        return {"X": X, "y": exact(X)}

    return TaskDefinition(
        "fractional_diffusion_1d_v2",
        "Fractional diffusion 1D",
        data,
        ref,
        None,
        output_transform=transform,
    )


def _nonlinear_telegraph() -> TaskDefinition:

    def exact(x):
        return np.exp(-x[:, 1:2]) * np.sin(x[:, :1])

    def pde(x, y):
        ut = dde.grad.jacobian(y, x, i=0, j=1)
        utt = dde.grad.hessian(y, x, i=1, j=1)
        uxx = dde.grad.hessian(y, x, i=0, j=0)
        forcing = (
            -2 * torch.exp(-x[:, 1:2]) * torch.sin(x[:, :1])
            + torch.exp(-3 * x[:, 1:2]) * torch.sin(x[:, :1]) ** 3
        )
        return utt + 4 * ut + y**3 - uxx - forcing

    def data(anchors):
        geom = dde.geometry.GeometryXTime(
            dde.geometry.Interval(0, 2 * np.pi), dde.geometry.TimeDomain(0, 0.5)
        )
        x0 = np.linspace(0, 2 * np.pi, 500, dtype=np.float32)[:, None]
        initial = np.column_stack((x0[:, 0], np.zeros(len(x0), dtype=np.float32)))
        tb = np.linspace(0, 0.5, 250, dtype=np.float32)
        left = np.column_stack((np.zeros_like(tb), tb))
        right = np.column_stack((np.full_like(tb, 2 * np.pi), tb))
        bcs = [
            dde.icbc.PointSetBC(initial, np.sin(initial[:, :1]).astype(np.float32)),
            dde.icbc.PointSetOperatorBC(
                initial,
                (-np.sin(initial[:, :1])).astype(np.float32),
                lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
            ),
            dde.icbc.PointSetBC(left, np.zeros((len(left), 1), dtype=np.float32)),
            dde.icbc.PointSetBC(right, np.zeros((len(right), 1), dtype=np.float32)),
        ]
        return _time_data(
            geom, pde, bcs, domain=1000, boundary=0, initial=0, anchors=anchors
        )

    def ref():
        X = _grid(
            np.linspace(0, 2 * np.pi, 256, dtype=np.float32),
            np.linspace(0, 0.5, 101, dtype=np.float32),
        )
        return {"X": X, "y": exact(X).astype(np.float32)}

    return TaskDefinition("nonlinear_telegraph_v2", "Nonlinear telegraph", data, ref, pde)


def _gray_scott() -> TaskDefinition:
    F, k, Du, Dv = (0.04, 0.1, 1e-05, 5e-06)

    def initial(x):
        u = 1 - np.exp(-80 * ((x[:, :1] + 0.05) ** 2 + (x[:, 1:2] + 0.02) ** 2))
        v = np.exp(-80 * ((x[:, :1] - 0.05) ** 2 + (x[:, 1:2] - 0.02) ** 2))
        return np.hstack((u, v))

    def pde(x, y):
        u, v = (y[:, :1], y[:, 1:2])
        ut = dde.grad.jacobian(y, x, i=0, j=2)
        vt = dde.grad.jacobian(y, x, i=1, j=2)
        lapu = dde.grad.hessian(y, x, component=0, i=0, j=0) + dde.grad.hessian(
            y, x, component=0, i=1, j=1
        )
        lapv = dde.grad.hessian(y, x, component=1, i=0, j=0) + dde.grad.hessian(
            y, x, component=1, i=1, j=1
        )
        return [
            ut - Du * lapu + u * v * v - F * (1 - u),
            vt - Dv * lapv - u * v * v + (F + k) * v,
        ]

    def data(anchors):
        geom = dde.geometry.GeometryXTime(
            dde.geometry.Rectangle([-1, -1], [1, 1]), dde.geometry.TimeDomain(0, 20)
        )
        bcs = [
            dde.icbc.IC(
                geom, lambda x: initial(x)[:, :1], lambda _, on: on, component=0
            ),
            dde.icbc.IC(
                geom, lambda x: initial(x)[:, 1:2], lambda _, on: on, component=1
            ),
        ]
        for component in (0, 1):
            for axis in (0, 1):
                bcs.append(
                    dde.icbc.PeriodicBC(
                        geom, axis, lambda _, on: on, component=component
                    )
                )
        return _time_data(
            geom, pde, bcs, domain=2048, boundary=256, initial=512, anchors=anchors
        )

    def ref():
        payload = np.load(require_asset("gray_scott_short.npz"))
        if "X" not in payload or "y" not in payload:
            raise RuntimeError(
                "Gray--Scott asset must contain X and y reference arrays."
            )
        return {
            "X": payload["X"].astype(np.float32),
            "y": payload["y"].astype(np.float32),
            "evaluation_chunk_size": 1024,
        }

    return TaskDefinition(
        "gray_scott_2d_short_v2",
        "Gray--Scott 2D",
        data,
        ref,
        pde,
        extra_metrics=_safe_component_metrics(["u", "v"]),
    )


def _darcy() -> TaskDefinition:

    def payload():
        result = _asset_payload("darcy_sample0.npz")
        required = {"X", "y", "coefficient", "grid_x", "grid_y", "X_obs", "y_obs"}
        missing = required - set(result)
        if missing:
            raise RuntimeError(f"Darcy asset is missing arrays: {sorted(missing)}")
        return result

    def coefficient_torch(x):
        data = payload()
        coeff = torch.as_tensor(data["coefficient"], device=x.device, dtype=x.dtype)
        if coeff.ndim != 2:
            raise RuntimeError("Darcy coefficient must be a 2D grid.")
        gx = torch.as_tensor(data["grid_x"], device=x.device, dtype=x.dtype)
        gy = torch.as_tensor(data["grid_y"], device=x.device, dtype=x.dtype)
        scaled = torch.stack(
            (
                2 * (x[:, 1] - gy[0]) / (gy[-1] - gy[0]) - 1,
                2 * (x[:, 0] - gx[0]) / (gx[-1] - gx[0]) - 1,
            ),
            dim=1,
        )
        sample_grid = scaled.reshape(1, -1, 1, 2)
        image = coeff.reshape(1, 1, *coeff.shape)
        return torch.nn.functional.grid_sample(
            image,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(-1, 1)

    def pde(x, y):
        qx, qy = (y[:, 1:2], y[:, 2:3])
        a = coefficient_torch(x)
        px = dde.grad.jacobian(y, x, i=0, j=0)
        py = dde.grad.jacobian(y, x, i=0, j=1)
        div = dde.grad.jacobian(y, x, i=1, j=0) + dde.grad.jacobian(y, x, i=2, j=1)
        return [qx + a * px, qy + a * py, div - 1.0]

    def data(anchors):
        source = payload()
        geom = dde.geometry.Rectangle([0, 0], [1, 1])
        bcs = [
            dde.icbc.DirichletBC(
                geom, lambda x: np.zeros((len(x), 1)), lambda _, on: on, component=0
            ),
            dde.icbc.PointSetBC(source["X_obs"], source["y_obs"][:, :1], component=0),
        ]
        counts = (2048, 256) if anchors is None else (0, 0)
        return dde.data.PDE(
            geom,
            pde,
            bcs,
            num_domain=counts[0],
            num_boundary=counts[1],
            anchors=anchors,
        )

    def ref():
        source = payload()
        return {
            "X": source["X"].astype(np.float32),
            "y": source["y"].astype(np.float32),
            "metric_scales": source["output_scales"].astype(np.float32),
            "evaluation_chunk_size": 1024,
        }

    def transform(x, y):
        scales = torch.as_tensor(
            payload()["output_scales"], device=y.device, dtype=y.dtype
        )
        return y * scales.reshape(1, -1)

    def weights(_, __):
        return [1, 1, 1, 1, 10]

    return TaskDefinition(
        "darcy_flow_2d_v2",
        "Darcy flow 2D mixed",
        data,
        ref,
        pde,
        output_transform=transform,
        extra_metrics=_safe_component_metrics(["pressure", "qx", "qy"]),
        loss_weights_resolver=weights,
    )


def _shallow_water() -> TaskDefinition:
    g = 1.0

    def initial(x):
        r = torch if torch.is_tensor(x) else np
        r2 = x[:, :1] ** 2 + x[:, 1:2] ** 2
        h = 1 + 0.2 * r.exp(-20 * r2)
        z = h * 0
        return r.cat((h, z, z), dim=1) if torch.is_tensor(x) else np.hstack((h, z, z))

    def pde(x, y):
        h, hu, hv = (y[:, :1], y[:, 1:2], y[:, 2:3])
        safe_h = torch.clamp(h, min=0.0001)
        mass = (
            dde.grad.jacobian(y, x, i=0, j=2)
            + dde.grad.jacobian(y, x, i=1, j=0)
            + dde.grad.jacobian(y, x, i=2, j=1)
        )
        flux_x = hu * hu / safe_h + 0.5 * g * h * h
        flux_xy = hu * hv / safe_h
        flux_y = hv * hv / safe_h + 0.5 * g * h * h
        mom_x = (
            dde.grad.jacobian(y, x, i=1, j=2)
            + dde.grad.jacobian(flux_x, x, i=0, j=0)
            + dde.grad.jacobian(flux_xy, x, i=0, j=1)
        )
        mom_y = (
            dde.grad.jacobian(y, x, i=2, j=2)
            + dde.grad.jacobian(flux_xy, x, i=0, j=0)
            + dde.grad.jacobian(flux_y, x, i=0, j=1)
        )
        return [mass, mom_x, mom_y]

    def data(anchors):
        source = _asset_payload("shallow_water_smooth.npz")
        required = {"X", "y", "X_obs", "y_obs"}
        missing = required - set(source)
        if missing:
            raise RuntimeError(
                f"Shallow-water asset is missing arrays: {sorted(missing)}"
            )
        geom = dde.geometry.GeometryXTime(
            dde.geometry.Rectangle([-1, -1], [1, 1]), dde.geometry.TimeDomain(0, 0.25)
        )
        bcs = [
            dde.icbc.IC(
                geom, lambda x: initial(x)[:, :1], lambda _, on: on, component=0
            ),
            dde.icbc.IC(
                geom, lambda x: initial(x)[:, 1:2], lambda _, on: on, component=1
            ),
            dde.icbc.IC(
                geom, lambda x: initial(x)[:, 2:3], lambda _, on: on, component=2
            ),
            dde.icbc.NeumannBC(
                geom,
                lambda x: np.zeros((len(x), 1), dtype=np.float32),
                lambda _, on: on,
                component=0,
            ),
            dde.icbc.NeumannBC(
                geom,
                lambda x: np.zeros((len(x), 1), dtype=np.float32),
                lambda _, on: on,
                component=1,
            ),
            dde.icbc.NeumannBC(
                geom,
                lambda x: np.zeros((len(x), 1), dtype=np.float32),
                lambda _, on: on,
                component=2,
            ),
            dde.icbc.PointSetBC(source["X_obs"], source["y_obs"], component=[0, 1, 2]),
        ]
        return _time_data(
            geom, pde, bcs, domain=3072, boundary=256, initial=512, anchors=anchors
        )

    def ref():
        source = _asset_payload("shallow_water_smooth.npz")
        return {
            "X": source["X"].astype(np.float32),
            "y": source["y"].astype(np.float32),
            "metric_scales": source["output_scales"].astype(np.float32),
            "evaluation_chunk_size": 1024,
        }

    def transform(x, y):
        scales = torch.as_tensor([1.0, 0.2, 0.2], device=y.device, dtype=y.dtype)
        return y * scales.reshape(1, -1)

    def weights(_, __):
        return [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]

    return TaskDefinition(
        "shallow_water_2d_smooth_v2",
        "Smoothed shallow-water 2D",
        data,
        ref,
        pde,
        output_transform=transform,
        extra_metrics=_safe_component_metrics(["h", "hu", "hv"]),
        loss_weights_resolver=weights,
    )


def _taylor_green() -> TaskDefinition:
    nu = 0.01

    def exact(x):
        X, Y, t = (x[:, :1], x[:, 1:2], x[:, 2:3])
        e = np.exp(-2 * nu * t)
        u = -np.cos(X) * np.sin(Y) * e
        v = np.sin(X) * np.cos(Y) * e
        p = -0.25 * (np.cos(2 * X) + np.cos(2 * Y)) * np.exp(-4 * nu * t)
        return np.hstack((u, v, p))

    def pde(x, y):
        u, v, p = (y[:, :1], y[:, 1:2], y[:, 2:3])
        ux = dde.grad.jacobian(y, x, i=0, j=0)
        uy = dde.grad.jacobian(y, x, i=0, j=1)
        vx = dde.grad.jacobian(y, x, i=1, j=0)
        vy = dde.grad.jacobian(y, x, i=1, j=1)
        ut = dde.grad.jacobian(y, x, i=0, j=2)
        vt = dde.grad.jacobian(y, x, i=1, j=2)
        lapu = dde.grad.hessian(y, x, component=0, i=0, j=0) + dde.grad.hessian(
            y, x, component=0, i=1, j=1
        )
        lapv = dde.grad.hessian(y, x, component=1, i=0, j=0) + dde.grad.hessian(
            y, x, component=1, i=1, j=1
        )
        return [
            ut + u * ux + v * uy + dde.grad.jacobian(y, x, i=2, j=0) - nu * lapu,
            vt + u * vx + v * vy + dde.grad.jacobian(y, x, i=2, j=1) - nu * lapv,
            ux + vy,
        ]

    def data(anchors):
        geom = dde.geometry.GeometryXTime(
            dde.geometry.Rectangle([0, 0], [2 * np.pi, 2 * np.pi]),
            dde.geometry.TimeDomain(0, 1),
        )
        bcs = [
            dde.icbc.IC(geom, lambda x: exact(x)[:, :1], lambda _, on: on, component=0),
            dde.icbc.IC(
                geom, lambda x: exact(x)[:, 1:2], lambda _, on: on, component=1
            ),
            dde.icbc.IC(
                geom, lambda x: exact(x)[:, 2:3], lambda _, on: on, component=2
            ),
        ]
        for component in (0, 1, 2):
            for axis in (0, 1):
                bcs.append(
                    dde.icbc.PeriodicBC(
                        geom, axis, lambda _, on: on, component=component
                    )
                )
        return _time_data(
            geom, pde, bcs, domain=2048, boundary=256, initial=256, anchors=anchors
        )

    def ref():
        X = _grid(
            np.linspace(0, 2 * np.pi, 51, dtype=np.float32),
            np.linspace(0, 2 * np.pi, 51, dtype=np.float32),
            np.linspace(0, 1, 21, dtype=np.float32),
        )
        return {
            "X": X,
            "y": exact(X).astype(np.float32),
            "metric_scales": np.asarray([1, 1, 0.5], dtype=np.float32),
            "evaluation_chunk_size": 1024,
        }

    def transform(x, y):
        return y * torch.as_tensor(
            [1.0, 1.0, 0.5], device=y.device, dtype=y.dtype
        ).reshape(1, -1)

    return TaskDefinition(
        "taylor_green_2d_v2",
        "Taylor--Green vortex 2D",
        data,
        ref,
        pde,
        output_transform=transform,
        extra_metrics=_safe_component_metrics(["u", "v", "p"]),
    )


def _dynamic_beam() -> TaskDefinition:

    def exact(x):
        return np.sin(np.pi * x[:, :1]) * np.cos(np.pi**2 * x[:, 1:2])

    def pde(x, y):
        return dde.grad.hessian(y, x, i=1, j=1) + dde.grad.hessian(
            dde.grad.hessian(y, x, i=0, j=0), x, i=0, j=0
        )

    def data(anchors):
        geom = dde.geometry.GeometryXTime(
            dde.geometry.Interval(0, 1), dde.geometry.TimeDomain(0, 1)
        )
        x0 = np.linspace(0, 1, 256, dtype=np.float32)
        initial = np.column_stack((x0, np.zeros_like(x0)))
        bcs = [
            dde.icbc.PointSetBC(
                initial, np.sin(np.pi * initial[:, :1]).astype(np.float32)
            ),
            dde.icbc.PointSetOperatorBC(
                initial,
                np.zeros((len(initial), 1), dtype=np.float32),
                lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
            ),
            dde.icbc.DirichletBC(
                geom,
                lambda x: np.zeros((len(x), 1), dtype=np.float32),
                lambda _, on: on,
            ),
            dde.icbc.OperatorBC(
                geom, lambda x, y, _: dde.grad.hessian(y, x, i=0, j=0), lambda _, on: on
            ),
        ]
        return _time_data(
            geom, pde, bcs, domain=1024, boundary=256, initial=0, anchors=anchors
        )

    def ref():
        X = _grid(
            np.linspace(0, 1, 201, dtype=np.float32),
            np.linspace(0, 1, 101, dtype=np.float32),
        )
        return {"X": X, "y": exact(X).astype(np.float32)}

    return TaskDefinition(
        "dynamic_beam_v2", "Dynamic Euler--Bernoulli beam", data, ref, pde
    )


@lru_cache(maxsize=1)
def tasks() -> dict[str, TaskDefinition]:
    values = [
        _laplace_disk(),
        _helmholtz(),
        _inverse_poisson(),
        _inverse_brinkman(),
        _fractional_diffusion(),
        _nonlinear_telegraph(),
        _gray_scott(),
        _darcy(),
        _shallow_water(),
        _taylor_green(),
        _dynamic_beam(),
    ]
    return {task.key: task for task in values}


def get_task(key: str) -> TaskDefinition:
    return tasks()[key]

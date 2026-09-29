GROUPS = (
    ("Linear diffusive systems", ("heat_equation", "diffusion_reaction", "inverse_heat_diffusivity")),
    ("Nonlinear diffusive systems", ("burgers", "allen_cahn", "fisher_kpp", "gray_scott_2d_short_v2")),
    ("Dispersive-wave systems", ("dynamic_beam_v2", "kdv", "schrodinger")),
    ("Spatial-fractional systems", ("fractional_poisson_1d", "fractional_diffusion_1d_v2")),
    ("Additional benchmark tasks", (
        "underdamped_vibration", "nonlinear_telegraph_v2", "klein_gordon",
        "poisson_lshape", "laplace_disk_2d_v2", "helmholtz_2d_v2",
        "inverse_poisson_field_v2", "volterra_ide", "darcy_flow_2d_v2",
        "kovasznay_flow", "taylor_green_2d_v2", "shallow_water_2d_smooth_v2",
        "inverse_brinkman_v2",
    )),
)

TASKS = {
    "heat_equation": ("Heat equation", "Heat", "1D+T"),
    "diffusion_reaction": ("Diffusion--Reaction", "DiffRxn", "1D+T"),
    "inverse_heat_diffusivity": ("Inverse Heat", "InvHeat", "1D+T"),
    "burgers": ("Burgers", "Burg", "1D+T"),
    "allen_cahn": ("Allen--Cahn", "A-Cahn", "1D+T"),
    "fisher_kpp": ("Fisher--KPP", "F-KPP", "1D+T"),
    "gray_scott_2d_short_v2": ("Gray--Scott", "G-Scott", "2D+T"),
    "dynamic_beam_v2": ("Dynamic Beam", "DynBeam", "1D+T"),
    "kdv": ("KdV", "KdV", "1D+T"),
    "schrodinger": (r"Schr\"odinger", "Schr", "1D+T"),
    "fractional_poisson_1d": ("Fractional Poisson", "FracPois", "1D"),
    "fractional_diffusion_1d_v2": ("Fractional Diffusion", "FracDiff", "1D+T"),
    "underdamped_vibration": ("Underdamped vibration", "UDVib", "T-only"),
    "nonlinear_telegraph_v2": ("Nonlinear Telegraph", "NTele", "1D+T"),
    "klein_gordon": ("Klein--Gordon", "K-Gord", "1D+T"),
    "poisson_lshape": ("Poisson L-shape", "Pois-L", "2D"),
    "laplace_disk_2d_v2": ("Laplace Disk", "LapDisk", "2D"),
    "helmholtz_2d_v2": ("Helmholtz", "Helm", "2D"),
    "inverse_poisson_field_v2": ("Inverse Poisson", "InvPois", "1D"),
    "volterra_ide": ("Volterra IDE", "VoltIDE", "IDE"),
    "darcy_flow_2d_v2": ("Darcy Flow", "Darcy", "2D"),
    "kovasznay_flow": ("Kovasznay flow", "Kovasz", "2D"),
    "taylor_green_2d_v2": ("Taylor--Green", "TayGreen", "2D+T"),
    "shallow_water_2d_smooth_v2": ("Shallow Water", "ShWater", "2D+T"),
    "inverse_brinkman_v2": ("Inverse Brinkman", "InvBrink", "1D"),
}

INVERSE_TASKS = frozenset(("inverse_heat_diffusivity", "inverse_poisson_field_v2", "inverse_brinkman_v2"))
ORDER = tuple(key for _, members in GROUPS for key in members)
MEMBERSHIP = {key: index for index, (_, members) in enumerate(GROUPS) for key in members}


def validate(task_keys):
    if ([len(members) for _, members in GROUPS] != [3, 4, 3, 2, 13]
            or len(ORDER) != 25 or len(set(ORDER)) != 25
            or set(ORDER) != set(TASKS) or set(ORDER) != set(task_keys)):
        raise ValueError("Task presentation does not match the 25-task benchmark")
    if len({entry[1] for entry in TASKS.values()}) != 25 or not INVERSE_TASKS <= set(ORDER):
        raise ValueError("Task labels are not unique")

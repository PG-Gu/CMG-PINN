import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from src.analysis.task_presentation import GROUPS, MEMBERSHIP, ORDER, TASKS, validate


STYLE = (("#0072B2", "o"), ("#D55E00", "D"),
         ("#009E73", "v"), ("#7B2CBF", "P"))
GRAY_DOT = "#C8CDD2"


def canonicalize(scores):
    result = np.asarray(scores, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        anchor = int(np.argmax(np.abs(result[:, column])))
        if result[anchor, column] < 0:
            result[:, column] *= -1
    return result


def pca_embed(matrix):
    centered = matrix - matrix.mean(axis=0)
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    return canonicalize(left[:, :2] * singular[:2]), variance[:2] / variance.sum()


def medoids(tensor):
    vectors, seeds = [], []
    for task in tensor:
        distances = np.linalg.norm(task[:, None, :] - task[None, :, :], axis=2).sum(axis=1)
        candidates = np.flatnonzero(np.isclose(distances, distances.min(), rtol=1e-12, atol=1e-14))
        best = int(candidates[0])
        vectors.append(task[best])
        seeds.append(best)
    return np.asarray(vectors), seeds


def _overlap(first, second, pad=2):
    return (first[0] < second[2] + pad and first[2] > second[0] - pad
            and first[1] < second[3] + pad and first[3] > second[1] - pad)


def _place_labels(axis, figure, points, method):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = axis.get_window_extent(renderer)
    screen = axis.transData.transform(points)
    point_boxes = [(x - 7.5, y - 7.5, x + 7.5, y + 7.5) for x, y in screen]
    scored = [i for i, key in enumerate(ORDER) if MEMBERSHIP[key] < 4]
    density = {i: int(np.count_nonzero(np.linalg.norm(screen - screen[i], axis=1) < 78))
               for i in scored}
    priority = sorted(scored, key=lambda i: (-density[i], -len(TASKS[ORDER[i]][1]), i))
    offsets = [(distance * math.cos(math.radians(angle)),
                distance * math.sin(math.radians(angle)))
               for distance in (16, 26, 38, 52, 70, 90, 115, 145, 180, 220)
               for angle in range(0, 360, 20)]
    preferred = {
        ("cmg", "diffusion_reaction"): [(-60, -15), (-70, -25), (-55, 15)],
        ("cmg", "inverse_heat_diffusivity"): [(-42, 0), (-55, 18), (-75, -12)],
        ("cmg", "heat_equation"): [(42, 0), (55, 18), (75, -12)],
    }
    placed = []
    for index in priority:
        key = ORDER[index]
        name = TASKS[key][1]
        for dx, dy in preferred.get((method, key), []) + offsets:
            candidate = axis.annotate(name, xy=points[index], xytext=(dx, dy),
                                      textcoords="offset points", fontsize=17,
                                      ha="center", va="center")
            box = tuple(float(value) for value in candidate.get_window_extent(renderer).extents)
            candidate.remove()
            if (box[0] < bounds.x0 + 3 or box[1] < bounds.y0 + 3
                    or box[2] > bounds.x1 - 3 or box[3] > bounds.y1 - 3
                    or any(_overlap(box, other) for other in placed)
                    or any(_overlap(box, other) for other in point_boxes)):
                continue
            leader = math.hypot(dx, dy) > 22
            axis.annotate(name, xy=points[index], xytext=(dx, dy),
                          textcoords="offset points", fontsize=17, ha="center", va="center",
                          color=STYLE[MEMBERSHIP[key]][0], zorder=8,
                          arrowprops=({"arrowstyle": "-", "color": "#899199",
                                       "lw": .75, "shrinkA": 2, "shrinkB": 5} if leader else None),
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95, "pad": .04})
            placed.append(box)
            break
        else:
            raise RuntimeError(f"Cannot place label: {method}/{name}")
    return len(placed)


def _draw(data, method, output, dpi):
    index = {key: i for i, key in enumerate(data["task_order"])}
    selected = [index[key] for key in ORDER]
    record = data["methods"][method]
    values = np.asarray(record["raw_medoid_matrix"], dtype=float)[selected]
    points = np.asarray(record["pca"]["scores"], dtype=float)[selected]
    ratio = record["pca"]["explained_variance_ratio"]
    score = float(data["separability"]["scores"][method]["tsps_aupr"])
    if values.shape != (25, 6) or points.shape != (25, 2) or not np.isfinite(points).all():
        raise ValueError("Expected 25 six-parameter states and 25 two-dimensional points")
    figure = plt.figure(figsize=(14, 8.4), dpi=150)
    heatmap = figure.add_axes((.11, .16, .27, .72))
    colorbar = figure.add_axes((.395, .16, .012, .72))
    embedding = figure.add_axes((.505, .16, .445, .72))
    norm = TwoSlopeNorm(vmin=0, vcenter=.5, vmax=1)
    image = heatmap.imshow(values, aspect="auto", cmap="RdBu_r", norm=norm,
                           interpolation="nearest")
    figure.colorbar(ScalarMappable(norm=norm, cmap=image.cmap), cax=colorbar,
                   ticks=[0, .5, 1])
    colorbar.tick_params(labelsize=13)
    heatmap.set_yticks(range(25), [TASKS[key][1] for key in ORDER], fontsize=16.5)
    heatmap.tick_params(axis="y", length=0, pad=2)
    for tick, key in zip(heatmap.get_yticklabels(), ORDER):
        tick.set_color(STYLE[MEMBERSHIP[key]][0] if MEMBERSHIP[key] < 4 else "#626970")
    heatmap.set_xticks(range(6), [r"$\mu_1$", r"$\mu_2$", r"$\mu_3$",
                                    r"$I_1$", r"$I_2$", r"$I_3$"], fontsize=17)
    heatmap.xaxis.tick_top()
    heatmap.tick_params(axis="x", length=0, pad=3)
    heatmap.axvspan(2.45, 2.55, color="white", zorder=3)
    heatmap.axvline(2.5, color="#343A40", linewidth=1, zorder=4)
    for boundary in np.cumsum([len(members) for _, members in GROUPS])[:-1]:
        heatmap.axhline(boundary - .5, color="white", linewidth=1.5, zorder=4)
    gray = [i for i, key in enumerate(ORDER) if MEMBERSHIP[key] == 4]
    embedding.scatter(points[gray, 0], points[gray, 1], s=55, color=GRAY_DOT,
                      edgecolor="white", linewidth=.5, zorder=4)
    for group in range(4):
        members = [i for i, key in enumerate(ORDER) if MEMBERSHIP[key] == group]
        embedding.scatter(points[members, 0], points[members, 1], marker=STYLE[group][1],
                          color=STYLE[group][0], s=95, edgecolor="white", linewidth=.7, zorder=5)
    span = np.maximum(np.ptp(points, axis=0), 1e-9)
    embedding.set_xlim(float(points[:, 0].min() - .27 * span[0]),
                       float(points[:, 0].max() + .27 * span[0]))
    embedding.set_ylim(float(points[:, 1].min() - .30 * span[1]),
                       float(points[:, 1].max() + .30 * span[1]))
    embedding.grid(color="#E5E8EC", linewidth=.75, zorder=0)
    embedding.set_xlabel(f"PC1 ({100 * ratio[0]:.1f}%)", fontsize=16)
    embedding.set_ylabel(f"PC2 ({100 * ratio[1]:.1f}%)", fontsize=16)
    embedding.tick_params(labelsize=13)
    if _place_labels(embedding, figure, points, method) != 12:
        raise ValueError("Expected twelve scored task labels")
    figure.text(.245, .962, "A  Learned parameters", ha="center", va="top", fontsize=19)
    figure.text(.7275, .962, f"B  Parameter embedding (TSPS-AUPR {score:.3f})",
                ha="center", va="top", fontsize=17)
    handles = [Line2D([0], [0], marker=STYLE[i][1], linestyle="", color="none",
                      markerfacecolor=STYLE[i][0], markeredgecolor="white", markersize=9,
                      label=name) for i, (name, _) in enumerate(GROUPS[:4])]
    handles.append(Line2D([0], [0], marker="o", linestyle="", color="none",
                          markerfacecolor=GRAY_DOT, markeredgecolor="white", markersize=9,
                          label=GROUPS[4][0]))
    legend = figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .006),
                           ncol=5, frameon=False, fontsize=12, columnspacing=.35,
                           handletextpad=.2)
    figure.canvas.draw()
    box = legend.get_window_extent(figure.canvas.get_renderer())
    if box.x0 < 0 or box.x1 > figure.bbox.width:
        raise RuntimeError("Legend extends outside the figure")
    stem = "figure2-cmg-gelu" if method == "cmg_gelu" else "figure3-cmg"
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=.10,
                   metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(output / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=.10)
    plt.close(figure)


def render(data, output, dpi=190):
    validate(data["task_order"])
    if set(data["methods"]) != {"cmg", "cmg_gelu"}:
        raise ValueError("Expected both activation methods")
    if set(data["separability"]["scores"]) != {"cmg", "cmg_gelu"}:
        raise ValueError("Six-metric separability scores are required for the figures")
    output.mkdir(parents=True, exist_ok=True)
    for method in ("cmg_gelu", "cmg"):
        _draw(data, method, output, dpi)

"""Within-method parameter standardization, five-seed medoids and PCA."""

import math
import statistics
from pathlib import Path

import numpy as np
from src.paths import ROOT
from src.training.runner import read
from src.analysis.geometry_plot import medoids, pca_embed
from src.analysis.task_presentation import GROUPS, validate


def mean_sem(values):
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def select_records(root, *, methods=None):
    configuration = read(ROOT / "configs/benchmark.json")
    experiments = configuration["experiments"]
    if methods:
        experiments = [r for r in experiments if r["method"] in methods]
    found = {}
    for path in Path(root).rglob("result.json"):
        status = path.with_name("status.json")
        if not status.exists() or read(status).get("state") != "complete":
            continue
        record = read(path)
        key = (
            record["task"],
            record["method"],
            float(record["weight_lr"]),
            float(record["activation_lr"]),
            int(record["seed"]),
        )
        if key in found:
            raise ValueError(f"Duplicate run: {key}")
        if not record["budget_complete"] or not math.isfinite(record["metric_value"]):
            raise ValueError(f"Invalid result: {path}")
        found[key] = record
    selected = []
    for spec in experiments:
        pairs = spec["lr_pairs"]
        candidates = []
        for w, a in pairs:
            keys = [
                (spec["task"], spec["method"], float(w), float(a), s)
                for s in configuration["seeds"]
            ]
            missing = [key for key in keys if key not in found]
            if missing:
                raise ValueError(f"Missing required completed runs: {missing}")
            records = [found[key] for key in keys]
            if any(r["metric"] != spec["metric"] for r in records):
                raise ValueError("Metric identity differs from the task configuration")
            mean, sem = mean_sem([r["metric_value"] for r in records])
            candidates.append(
                dict(
                    task=spec["task"],
                    method=spec["method"],
                    metric=spec["metric"],
                    weight_lr=w,
                    activation_lr=a,
                    mean=mean,
                    sem=sem,
                    records=records,
                )
            )
        selected.append(
            min(
                candidates,
                key=lambda r: (r["mean"], r["weight_lr"], r["activation_lr"]),
            )
        )
    return selected


def features(record):
    if record.get("final_mu") is not None:
        mu = record["final_mu"]
        integral = record["final_I"]
        keys = sorted(mu, key=lambda k: int(k.rsplit("_", 1)[-1]))
        vector = [
            float(np.asarray(values[key]).item())
            for values in (mu, integral)
            for key in keys
        ]
    else:
        state = record["final_activation_state"]
        layers = [
            state[k]
            for k in sorted(state, key=lambda k: int(k.rsplit("_", 1)[-1]))
            if state[k]["kind"] == "cmg_gelu"
        ]
        vector = [
            float(np.asarray(layer[name]).item())
            for name in ("mu", "I")
            for layer in layers
        ]
    if len(vector) != 6 or not np.isfinite(vector).all():
        raise ValueError("Expected six finite layer-wise activation parameters")
    return vector


def build(results):
    order = read(ROOT / "configs/geometry.json")
    tasks = [
        dict(key=k, label=v) for k, v in zip(order["task_order"], order["task_labels"])
    ]
    validate([task["key"] for task in tasks])
    cells = select_records(
        results, methods=["cmg_layer_match", "cmg_gelu_match"]
    )
    lookup = {(r["task"], r["method"]): r for r in cells}
    methods = {}
    for label, method in [("cmg", "cmg_layer_match"), ("cmg_gelu", "cmg_gelu_match")]:
        raw = np.asarray(
            [
                [features(record) for record in lookup[t["key"], method]["records"]]
                for t in tasks
            ],
            dtype=np.float64,
        )
        flat = raw.reshape(-1, 6)
        center = flat.mean(axis=0)
        spread = flat.std(axis=0, ddof=0)
        if np.any(spread == 0):
            raise ValueError("Cannot standardize a constant feature")
        standardized = (raw - center) / spread
        matrix, seeds = medoids(standardized)
        scores, ratio = pca_embed(matrix)
        methods[label] = dict(
            raw_seed_tensor=raw.tolist(),
            normalization=dict(mean=center.tolist(), std_ddof0=spread.tolist()),
            medoid_seeds=seeds,
            raw_medoid_matrix=raw[np.arange(len(tasks)), seeds, :].tolist(),
            zscore_medoid_matrix=matrix.tolist(),
            pca=dict(scores=scores.tolist(), explained_variance_ratio=ratio.tolist()),
        )
    return dict(
        task_order=[t["key"] for t in tasks],
        task_labels=[t["label"] for t in tasks],
        groups={name: list(members) for name, members in GROUPS},
        methods=methods,
        separability={"scores": {}},
    )

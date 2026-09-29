"""Run independent paper configurations in fresh processes."""

from __future__ import annotations
import argparse
import concurrent.futures
import importlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
from src.paths import ROOT


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_job(job):
    """Train from initialization; reuse only an identical, completed run."""
    directory = Path(job["directory"])
    identity = {k: v for k, v in job.items() if k != "directory"}
    config_path = directory / "resolved_config.json"
    if (directory / "status.json").exists() and not config_path.exists():
        raise ValueError(f"Missing resolved configuration: {directory}")
    if config_path.exists() and read(config_path) != identity:
        raise ValueError(
            f"Configuration conflict: choose another output directory: {directory}"
        )
    if (directory / "status.json").exists() and read(directory / "status.json").get(
        "state"
    ) == "complete":
        result = read(directory / "result.json")
        if (
            result["task"] != job["task"]
            or result["method"] != job["method"]
            or result["seed"] != job["seed"]
        ):
            raise ValueError(f"Result identity mismatch: {directory}")
        if not math.isfinite(result["metric_value"]) or not result["budget_complete"]:
            raise ValueError(f"Invalid completed result: {directory}")
        if (
            result["metric"] != job["metric"]
            or result["weight_lr"] != job["config"]["weight_lr"]
            or result["activation_lr"] != job["config"]["activation_lr"]
            or result["first_order_iterations"] != job["config"]["first_order_iterations"]
            or result["lbfgs_iterations"]
            != job["config"]["lbfgs_fixed_budget_target_iters"]
        ):
            raise ValueError(f"Result settings or iteration counts differ: {directory}")
        if job["save_model"] and not (directory / "model.pt").is_file():
            raise FileNotFoundError(directory / "model.pt")
        print(f"Skipping completed {directory}", flush=True)
        return result
    write(config_path, identity)
    write(directory / "status.json", {"state": "running"})
    try:
        import deepxde as dde

        module = importlib.import_module("src.tasks." + job["task"])
        cfg = dict(job["config"])
        cfg.update(
            sampling_root=str(directory / "sampling"),
            save_selected_model_snapshot=job["save_model"],
            selected_model_snapshot_path=str(directory / "model.pt"),
        )
        if hasattr(module, "prepare_job_config"):
            cfg.update(module.prepare_job_config(cfg, job["seed"]) or {})
        counts = []
        original_train = dde.Model.train

        def observed_train(model, *args, **kwargs):
            before = int(model.train_state.step)
            result = original_train(model, *args, **kwargs)
            counts.append(int(model.train_state.step) - before)
            return result

        dde.Model.train = observed_train
        try:
            raw = module.run_single(
                cfg,
                job["seed"],
                verbose=False,
                track_reference_metrics=cfg.get("track_reference_metrics", False),
                reference_interval=cfg.get("reference_interval", 100),
                track_true_train_loss=cfg.get("track_true_train_loss", False),
                true_train_loss_interval=cfg.get("true_train_loss_interval", 100),
            )
        finally:
            dde.Model.train = original_train
        metric = float(raw[job["metric"]])
        first = sum(counts[:-1]) if len(counts) > 1 else (counts[0] if counts else 0)
        second = int(raw.get("lbfgs_budget_consumed_iters") or 0)
        if first != int(cfg["first_order_iterations"]) or second != int(
            cfg["lbfgs_fixed_budget_target_iters"]
        ):
            raise RuntimeError(
                f"Incomplete training budget: Muon={first}, L-BFGS={second}"
            )
        losses = {
            k: v for k, v in raw.items() if k.endswith("_final_loss") and v is not None
        }
        if not math.isfinite(metric) or not all(
            math.isfinite(float(v)) for v in losses.values()
        ):
            raise FloatingPointError("Non-finite final metric or loss")
        result = {
            k: raw[k]
            for k in (
                "params",
                "external_trainable_params",
                "final_mu",
                "final_I",
                "final_activation_state",
                "loss_history",
                "lbfgs_func_evals",
            )
            if k in raw
        }
        result.update(
            task=job["task"],
            method=job["method"],
            seed=job["seed"],
            metric=job["metric"],
            metric_value=metric,
            weight_lr=cfg["weight_lr"],
            activation_lr=cfg["activation_lr"],
            first_order_iterations=first,
            lbfgs_iterations=second,
            budget_complete=True,
            final_losses=losses,
        )
        write(directory / "result.json", result)
        write(directory / "status.json", {"state": "complete"})
        return result
    except BaseException as exc:
        write(directory / "status.json", {"state": "failed", "error": str(exc)})
        raise


def main(mode="train"):
    parser = argparse.ArgumentParser(
        description="Reproduce the paper using fixed published configurations."
    )
    parser.add_argument("--task", nargs="+", help="Task keys; default: all tasks")
    parser.add_argument("--method", nargs="+", help="Method keys; default: all methods")
    parser.add_argument("--seed", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--gpu",
        nargs="+",
        default=["0"],
        help="Physical GPU IDs; one process per GPU, or cpu",
    )
    parser.add_argument("--output", type=Path, default=Path("results") / mode)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_job(read(args.worker))
        return
    if len(set(args.gpu)) != len(args.gpu):
        parser.error("GPU IDs must be distinct")
    if len(set(args.seed)) != len(args.seed):
        parser.error("Seeds must be distinct")
    records = read(ROOT / "configs/benchmark.json")["experiments"]
    for field, values in (("task", args.task), ("method", args.method)):
        unknown = set(values or []) - {r[field] for r in records}
        if unknown:
            parser.error(f"Unknown {field}: {sorted(unknown)}")
    jobs = []
    for record in records:
        if args.task and record["task"] not in args.task:
            continue
        if args.method and record["method"] not in args.method:
            continue
        cfg = record["config"]
        pairs = record["lr_pairs"]
        for weight, activation in pairs:
            for seed in args.seed:
                path = (
                    args.output.resolve()
                    / record["task"]
                    / record["method"]
                    / f"wlr_{weight:g}_alr_{activation:g}"
                    / f"seed_{seed}"
                )
                jobs.append(
                    dict(
                        task=record["task"],
                        method=record["method"],
                        metric=record["metric"],
                        seed=seed,
                        save_model=args.save_model,
                        directory=str(path),
                        config=dict(
                            cfg,
                            seed=seed,
                            weight_lr=weight,
                            activation_lr=activation,
                        ),
                    )
                )
    devices = queue.Queue()
    for gpu in args.gpu:
        devices.put(gpu)

    def launch(job):
        gpu = devices.get()
        try:
            path = Path(job["directory"])
            path.mkdir(parents=True, exist_ok=True)
            request = path / "request.json"
            write(request, job)
            env = dict(
                os.environ,
                DDE_BACKEND="pytorch",
                CUDA_VISIBLE_DEVICES="" if gpu == "cpu" else gpu,
            )
            with (path / "train.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / (mode + ".py")),
                        "--worker",
                        str(request),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            print(
                f'{"OK" if result.returncode==0 else "FAILED"} {job["task"]}/{job["method"]}/seed_{job["seed"]}',
                flush=True,
            )
            return result.returncode
        finally:
            devices.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpu)) as pool:
        failures = sum(code != 0 for code in pool.map(launch, jobs))
    if failures:
        raise SystemExit(
            f"{failures} jobs failed; see train.log in their output directories."
        )

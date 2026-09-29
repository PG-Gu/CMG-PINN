from __future__ import annotations
from src.training.shared_task_training import (
    task_create_data,
    task_create_network,
    task_prepare_config,
    task_run,
)
from src.tasks.shared_task_definitions import get_task


def bind(task_key: str):
    task = get_task(task_key)

    def create_data(cfg, seed):
        return task_create_data(task, cfg, seed)

    def prepare_job_config(cfg, seed):
        return task_prepare_config(task, cfg, seed)

    def create_network(cfg):
        return task_create_network(task, cfg)

    def run_single(
        cfg,
        seed,
        verbose=True,
        track_reference_metrics=False,
        reference_interval=100,
        track_true_train_loss=False,
        true_train_loss_interval=100,
    ):
        return task_run(
            task,
            cfg,
            seed,
            verbose=verbose,
            track_reference_metrics=track_reference_metrics,
            reference_interval=reference_interval,
            track_true_train_loss=track_true_train_loss,
            true_train_loss_interval=true_train_loss_interval,
        )

    return {
        "TASK_KEY": task.key,
        "create_data": create_data,
        "prepare_job_config": prepare_job_config,
        "create_network": create_network,
        "run_single": run_single,
    }

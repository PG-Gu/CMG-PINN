import argparse
import math
import subprocess
from pathlib import Path

from src.analysis.task_presentation import GROUPS, validate
from src.paths import ROOT
from src.training.runner import read, write


METRICS = ("psi_roc", "psi_pr", "psi_mcc", "cps_aupr", "ldps_aupr", "tsps_aupr")


def matlab_path(path):
    return "'" + str(path).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--matlab", required=True)
    parser.add_argument("--concorde", type=Path, required=True)
    parser.add_argument("--psi-source", type=Path, default=ROOT / "third_party/psi")
    parser.add_argument("--tsps-source", type=Path, default=ROOT / "third_party/tsps")
    parser.add_argument("--output", type=Path, default=Path("separability"))
    args = parser.parse_args()
    for source, name in ((args.psi_source, "ProcessValidityIndices.m"),
                         (args.tsps_source, "CommunitySeparability.m")):
        if not (source / name).is_file() or not (source / "private").is_dir():
            parser.error(f"Expected {name} and private/ in {source}")
    solver = args.concorde.resolve()
    if not solver.is_file():
        parser.error(f"Concorde command not found: {solver}")
    data = read(args.geometry)
    order = data["task_order"]
    validate(order)
    tasks = [key for _, members in GROUPS[:4] for key in members]
    labels = [name for name, members in GROUPS[:4] for _ in members]
    cases = []
    for method in ("cmg_gelu", "cmg"):
        points = data["methods"][method]["pca"]["scores"]
        if len(points) != 25:
            raise ValueError("Expected 25 PCA points per method")
        cases.append(dict(method=method,
                          coordinates=[points[order.index(key)] for key in tasks],
                          group_labels=labels))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write(output / "input.json", dict(cases=cases))
    paths = (output / "input.json", output / "official_output.json",
             args.psi_source.resolve(), args.tsps_source.resolve(),
             output / "runtime", solver)
    (output / "runtime").mkdir(exist_ok=True)
    expression = "addpath(" + matlab_path(ROOT / "scripts") + "); run_official_driver(" + ",".join(
        matlab_path(path) for path in paths) + ");"
    subprocess.run([args.matlab, "-batch", expression], check=True)
    official = read(output / "official_output.json")["cases"]
    if len(official) != 2 or {entry["method"] for entry in official} != {"cmg", "cmg_gelu"}:
        raise ValueError("Expected one metric result per method")
    scores = {}
    for entry in official:
        values = {name: float(entry[name]) for name in METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite separability score")
        scores[entry["method"]] = values
    write(output / "scores.json", dict(
        task_order=order,
        coordinates={method: data["methods"][method]["pca"]["scores"] for method in scores},
        scores=scores,
    ))


if __name__ == "__main__":
    main()

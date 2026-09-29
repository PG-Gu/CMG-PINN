import argparse
from pathlib import Path
from src.analysis.geometry import build
from src.analysis.geometry_plot import render
from src.training.runner import read, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("figures"))
    p.add_argument("--separability", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    data = build(a.results)
    if a.separability:
        score = read(a.separability)
        if score["task_order"] != data["task_order"] or score["coordinates"] != {
            m: d["pca"]["scores"] for m, d in data["methods"].items()
        }:
            raise ValueError("Separability coordinates differ from the figure input")
        data["separability"]["scores"] = score["scores"]
    write(a.output / "geometry.json", data)
    if a.separability:
        render(data, a.output)


if __name__ == "__main__":
    main()

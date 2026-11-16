"""Create plots from a rolling Pi-rating optimization JSON result.

Usage:
    python scripts/visualize_pi_rating_optimization.py \
        --input results/pi_rating_rolling.json \
        --output-dir results/pi_rating_plots

The input JSON can be produced with:

    result = run_rolling_grid_search(games)
    Path("results/pi_rating_rolling.json").write_text(
        json.dumps(result, indent=2)
    )
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result.get("results"), list) or not result["results"]:
        raise ValueError("The JSON result must contain a non-empty 'results' list.")
    for key in ("best_params", "default_metrics", "zero_baseline"):
        if not isinstance(result.get(key), dict):
            raise ValueError(f"The JSON result must contain '{key}'.")
    return result


def _plot_heatmap(results: list[dict[str, Any]], output_dir: Path) -> None:
    lambdas = sorted({float(row["lr_lambda"]) for row in results})
    gammas = sorted({float(row["gamma"]) for row in results})
    values = [
        [
            next(
                float(row["mae"])
                for row in results
                if float(row["lr_lambda"]) == lr
                and float(row["gamma"]) == gamma
            )
            for gamma in gammas
        ]
        for lr in lambdas
    ]

    figure, axis = plt.subplots(figsize=(10, 7))
    image = axis.imshow(values, aspect="auto", cmap="viridis_r")
    axis.set_xticks(range(len(gammas)), [f"{value:g}" for value in gammas])
    axis.set_yticks(range(len(lambdas)), [f"{value:g}" for value in lambdas])
    axis.set_xlabel("gamma")
    axis.set_ylabel("lr_lambda")
    axis.set_title("Rolling-origin validation MAE")
    figure.colorbar(image, ax=axis, label="MAE (points)")
    figure.tight_layout()
    figure.savefig(output_dir / "pi_rating_mae_heatmap.png", dpi=150)
    plt.close(figure)


def _plot_parameter_scatter(
    results: list[dict[str, Any]], output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        [row["lr_lambda"] for row in results],
        [row["gamma"] for row in results],
        c=[row["mae"] for row in results],
        cmap="viridis_r",
        edgecolors="black",
        linewidths=0.3,
    )
    axis.set_xlabel("lr_lambda")
    axis.set_ylabel("gamma")
    axis.set_title("Pi-rating parameter search")
    figure.colorbar(scatter, ax=axis, label="Rolling-origin MAE (points)")
    figure.tight_layout()
    figure.savefig(output_dir / "pi_rating_parameter_scatter.png", dpi=150)
    plt.close(figure)


def _plot_baselines(result: dict[str, Any], output_dir: Path) -> None:
    labels = ["Best", "Default", "Zero diff"]
    values = [
        result["best_params"]["mae"],
        result["default_metrics"]["mae"],
        result["zero_baseline"]["mae"],
    ]

    figure, axis = plt.subplots(figsize=(8, 6))
    bars = axis.bar(labels, values, color=["#2ca02c", "#1f77b4", "#7f7f7f"])
    axis.set_ylabel("MAE (points)")
    axis.set_title("Rolling-origin baseline comparison")
    axis.bar_label(bars, fmt="%.2f")
    figure.tight_layout()
    figure.savefig(output_dir / "pi_rating_baselines.png", dpi=150)
    plt.close(figure)


def create_plots(input_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Create all optimization plots and return their paths."""
    result = _load_result(Path(input_path))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _plot_heatmap(result["results"], destination)
    _plot_parameter_scatter(result["results"], destination)
    _plot_baselines(result, destination)
    return [
        destination / "pi_rating_mae_heatmap.png",
        destination / "pi_rating_parameter_scatter.png",
        destination / "pi_rating_baselines.png",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for path in create_plots(args.input, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()

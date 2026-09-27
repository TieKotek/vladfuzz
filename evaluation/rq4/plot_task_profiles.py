"""Render task-level category shares from the audited RQ4 summary."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ("simlingo", "lmdrive", "bevdriver")
LABELS = ("SimLingo", "LMDrive", "BEVDriver")
CATEGORIES = ("collision", "lane_invasion", "out_of_bounds", "timeout",
              "speed_limit_exceeded", "maintain_distance_failed")
REGIONS = (("town01_t_junction_1", "T01 T-junction"),
           ("town03_crossover", "T03 Intersection"),
           ("town03_roundabout", "T03 Roundabout"),
           ("town04_highway_exit", "T04 Highway exit"),
           ("town05_crossover", "T05 Intersection"))


def render(source, output):
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tasks = [f"{region}/seed_{seed}" for region, _ in REGIONS for seed in (1, 2)]
    lookup = {(r["model"], r["scenario_seed"], r["category"]): r for r in rows}
    expected = {(m, t, c) for m in MODELS for t in tasks for c in CATEGORIES}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError("Expected one row per model, task, and category")
    for model in MODELS:
        for task in tasks:
            denominators = {int(lookup[model, task, c]["failures"]) for c in CATEGORIES}
            if len(denominators) != 1 or min(denominators) <= 0:
                raise ValueError(f"Invalid failure denominator: {model}/{task}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.6), sharey=True)
    fig.subplots_adjust(left=.185, right=.99, top=.90, bottom=.16, wspace=.08)
    for ax, model, label in zip(axes, MODELS, LABELS):
        values = np.array([[100 * int(lookup[model, task, cat]["count"]) /
                            int(lookup[model, task, cat]["failures"])
                            for cat in CATEGORIES] for task in tasks])
        if np.any((values < 0) | (values > 100)):
            raise ValueError("Category shares must lie in [0, 100]")
        ax.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        ax.set_title(label, fontsize=13, weight="bold", pad=10)
        ax.set_xticks(range(6), ["Collision", "Lane", "Departure", "Not reached",
                                 "Speed", "Distance"],
                      rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(10), [f"{name} /{seed}" for _, name in REGIONS for seed in (1, 2)])
        ax.tick_params(axis="both", length=0, pad=5)
        for row in range(10):
            for col in range(6):
                value = values[row, col]
                text = "0" if value == 0 else ("<1" if value < 1 else f"{value:.0f}")
                ax.text(col, row, text, ha="center", va="center", fontsize=10,
                        color="white" if value >= 55 else "#202020")
        for boundary in (1.5, 3.5, 5.5, 7.5):
            ax.axhline(boundary, color="white", linewidth=1.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=.04)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/rq4/summary/category_per_seed.csv"))
    parser.add_argument("--output", type=Path, default=Path("paper/fse-2027/figures/rq4-task-profiles.pdf"))
    args = parser.parse_args()
    render(args.input, args.output)

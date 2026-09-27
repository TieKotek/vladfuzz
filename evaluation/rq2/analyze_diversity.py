"""Analyze diversity of failure-inducing multimodal test cases in RQ2."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np


WEATHER_FEATURES = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "fog_density",
    "wetness",
    "sun_azimuth_angle",
    "sun_altitude_angle",
)

SCENE_FEATURES = WEATHER_FEATURES + (
    "vehicle_count",
    "walker_count",
    "puddle_count",
    "autopilot_fraction",
    "stationary_fraction",
    "nearest_actor_distance",
    "mean_actor_distance",
    "mean_forward_distance",
    "mean_abs_lateral_distance",
    "actor_distance_std",
)


def _relative_xy(ego_x: float, ego_y: float, ego_yaw: float, x: float, y: float):
    dx, dy = x - ego_x, y - ego_y
    yaw = math.radians(ego_yaw)
    return (
        math.cos(yaw) * dx + math.sin(yaw) * dy,
        -math.sin(yaw) * dx + math.cos(yaw) * dy,
    )


def _actor_summary(actors: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    actors = list(actors)
    vehicle_count = sum(actor["kind"] == "vehicle" for actor in actors)
    walker_count = sum(actor["kind"] == "walker" for actor in actors)
    if not actors:
        return {
            "vehicle_count": 0.0,
            "walker_count": 0.0,
            "autopilot_fraction": 0.0,
            "stationary_fraction": 0.0,
            "nearest_actor_distance": 0.0,
            "mean_actor_distance": 0.0,
            "mean_forward_distance": 0.0,
            "mean_abs_lateral_distance": 0.0,
            "actor_distance_std": 0.0,
        }

    distances = [math.hypot(actor["forward"], actor["lateral"]) for actor in actors]
    mean_distance = sum(distances) / len(distances)
    variance = sum((distance - mean_distance) ** 2 for distance in distances) / len(
        distances
    )
    return {
        "vehicle_count": float(vehicle_count),
        "walker_count": float(walker_count),
        "autopilot_fraction": sum(actor["motion"] == "autopilot" for actor in actors)
        / len(actors),
        "stationary_fraction": sum(actor["motion"] == "stationary" for actor in actors)
        / len(actors),
        "nearest_actor_distance": min(distances),
        "mean_actor_distance": mean_distance,
        "mean_forward_distance": sum(actor["forward"] for actor in actors)
        / len(actors),
        "mean_abs_lateral_distance": sum(abs(actor["lateral"]) for actor in actors)
        / len(actors),
        "actor_distance_std": math.sqrt(variance),
    }


def _weather_dict(values: Sequence[float]) -> Dict[str, float]:
    if len(values) != len(WEATHER_FEATURES):
        raise ValueError(
            f"expected {len(WEATHER_FEATURES)} weather values, got {len(values)}"
        )
    return dict(zip(WEATHER_FEATURES, map(float, values)))


def vladfuzz_scene_vector(
    scenario: Mapping,
    static: Mapping,
    *,
    weather_resolver: Callable[[str], Sequence[float]],
) -> Dict[str, float]:
    """Convert a VLAD-Fuzz/Instruction-CF scenario into common scene features."""
    center = static["scenario_center"]
    spawn_points = static["spawn_points"]
    ego = spawn_points[scenario["ego_car"]["spawn_point_index"]]
    ego_x = float(center["x"]) + float(ego["x"])
    ego_y = float(center["y"]) + float(ego["y"])
    ego_yaw = float(ego.get("yaw", 0.0))

    actors = []
    for npc in scenario.get("npc_vehicles", []):
        point = spawn_points[npc["spawn_point_index"]]
        x = float(center["x"]) + float(point["x"])
        y = float(center["y"]) + float(point["y"])
        forward, lateral = _relative_xy(ego_x, ego_y, ego_yaw, x, y)
        actors.append(
            {
                "kind": "vehicle",
                "motion": "autopilot",
                "forward": forward,
                "lateral": lateral,
            }
        )

    result = _weather_dict(weather_resolver(str(scenario.get("weather", "ClearNoon"))))
    result.update(_actor_summary(actors))
    result["puddle_count"] = 0.0
    return result


def drivefuzz_scene_vector(record: Mapping) -> Dict[str, float]:
    """Convert a DriveFuzz error record into the same common scene features."""
    seed = record["seed"]
    ego_x, ego_y, ego_yaw = (
        float(seed["sp_x"]),
        float(seed["sp_y"]),
        float(seed.get("yaw", 0.0)),
    )
    actors = []
    for actor in record.get("actors", []):
        forward, lateral = _relative_xy(
            ego_x,
            ego_y,
            ego_yaw,
            float(actor["sp_x"]),
            float(actor["sp_y"]),
        )
        nav_type = actor.get("nav_type")
        actors.append(
            {
                "kind": "vehicle" if actor.get("type") == 0 else "walker",
                "motion": (
                    "autopilot"
                    if nav_type == 1
                    else "stationary" if nav_type == 2 else "linear"
                ),
                "forward": forward,
                "lateral": lateral,
            }
        )

    weather = record.get("weather") or {}
    result = _weather_dict(
        [
            weather.get("cloud", 0.0),
            weather.get("rain", 0.0),
            weather.get("puddle", 0.0),
            weather.get("wind", 0.0),
            weather.get("fog", 0.0),
            weather.get("wetness", 0.0),
            weather.get("angle", 0.0),
            weather.get("altitude", 0.0),
        ]
    )
    result.update(_actor_summary(actors))
    result["puddle_count"] = float(len(record.get("puddles", [])))
    return result


def scene_matrix(vectors: Sequence[Mapping[str, float]]) -> np.ndarray:
    return np.asarray(
        [[vector[name] for name in SCENE_FEATURES] for vector in vectors], dtype=float
    )


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lower = np.nanmin(values, axis=0)
    upper = np.nanmax(values, axis=0)
    scale = np.where(upper > lower, upper - lower, 1.0)
    return (values - lower) / scale


def _bounded_pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    difference = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(difference * difference, axis=2)) / math.sqrt(values.shape[1])


def _bounded_cosine_distance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.where(norms > 0.0, norms, 1.0)
    return np.clip(1.0 - normalized @ normalized.T, 0.0, 2.0) / 2.0


def _modality_distance_matrices(
    scene_values: np.ndarray,
    instruction_embeddings: np.ndarray,
    *,
    scene_is_normalized: bool = False,
):
    scene_values = np.asarray(scene_values, dtype=float)
    instruction_embeddings = np.asarray(instruction_embeddings, dtype=float)
    if len(scene_values) != len(instruction_embeddings):
        raise ValueError("scene and instruction rows must align")
    if scene_values.ndim != 2 or scene_values.shape[1] == 0:
        raise ValueError("scene values must be a non-empty matrix")
    normalized_scene = (
        scene_values if scene_is_normalized else _minmax_normalize(scene_values)
    )
    return (
        _bounded_pairwise_euclidean(normalized_scene),
        _bounded_cosine_distance(instruction_embeddings),
    )


def joint_distance_matrix(
    scene_values: np.ndarray,
    instruction_embeddings: np.ndarray,
    *,
    scene_weight: float = 0.5,
) -> np.ndarray:
    """Return a bounded, equally composable scene--instruction distance matrix."""
    if not 0.0 <= scene_weight <= 1.0:
        raise ValueError("scene_weight must be between zero and one")
    scene_distance, instruction_distance = _modality_distance_matrices(
        scene_values,
        instruction_embeddings,
    )
    return scene_weight * scene_distance + (1.0 - scene_weight) * instruction_distance


def pairwise_diversity(
    scene_values: np.ndarray,
    instruction_embeddings: np.ndarray,
    *,
    scene_weight: float = 0.5,
    scene_is_normalized: bool = False,
) -> Dict[str, float]:
    """Measure average pairwise dispersion in scene and instruction space."""
    if not 0.0 <= scene_weight <= 1.0:
        raise ValueError("scene_weight must be between zero and one")
    if len(scene_values) < 2:
        raise ValueError("pairwise diversity requires at least two cases")
    scene_distance, instruction_distance = _modality_distance_matrices(
        scene_values,
        instruction_embeddings,
        scene_is_normalized=scene_is_normalized,
    )
    upper = np.triu_indices(len(scene_values), 1)
    scene = float(np.mean(scene_distance[upper]))
    instruction = float(np.mean(instruction_distance[upper]))
    return {
        "scene": scene,
        "instruction": instruction,
        "joint": scene_weight * scene + (1.0 - scene_weight) * instruction,
    }


@dataclass(frozen=True)
class FailureCase:
    model: str
    method: str
    task: str
    instruction: str
    scene: Mapping[str, float]
    source: Path


def _resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _load_json(path: Path) -> Mapping:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_local_failures(
    row: Mapping[str, str],
    run_dir: Path,
    repo_root: Path,
    weather_resolver: Callable[[str], Sequence[float]],
) -> List[FailureCase]:
    metadata = _load_json(run_dir / "metadata.json")
    static_path = _resolve_path(metadata["configuration"]["static_scenario"], repo_root)
    static = _load_json(static_path)
    cases = []
    for result_path in sorted((run_dir / "failures").glob("*/result.json")):
        result = _load_json(result_path)
        scenario_path = result_path.with_name("scenario_config.json")
        if not scenario_path.exists():
            raise ValueError(f"missing scenario configuration for {result_path}")
        instruction = str(
            result.get("mutated_instruction") or result.get("instruction") or ""
        ).strip()
        if not instruction:
            raise ValueError(f"missing instruction in {result_path}")
        cases.append(
            FailureCase(
                model=row["model"],
                method=row["baseline"],
                task=row["scenario_seed"],
                instruction=instruction,
                scene=vladfuzz_scene_vector(
                    _load_json(scenario_path),
                    static,
                    weather_resolver=weather_resolver,
                ),
                source=result_path,
            )
        )
    return cases


def _collect_drivefuzz_failures(
    row: Mapping[str, str],
    run_dir: Path,
    repo_root: Path,
) -> List[FailureCase]:
    metadata = _load_json(run_dir / "drivefuzz_metadata.json")
    seed_dir = _resolve_path(metadata["seed_dir"], repo_root)
    mappings = [
        json.loads(line)
        for line in (seed_dir / "mapping.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(mappings) != 1:
        raise ValueError(
            f"expected one instruction mapping in {seed_dir}, got {len(mappings)}"
        )
    instruction = str(mappings[0].get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"missing DriveFuzz instruction in {seed_dir}")
    return [
        FailureCase(
            model=row["model"],
            method=row["baseline"],
            task=row["scenario_seed"],
            instruction=instruction,
            scene=drivefuzz_scene_vector(_load_json(error_path)),
            source=error_path,
        )
        for error_path in sorted((run_dir / "errors").glob("*.json"))
    ]


def collect_failure_cases(
    selected_runs: Path,
    repo_root: Path,
    *,
    weather_resolver: Callable[[str], Sequence[float]],
) -> List[FailureCase]:
    """Load exactly the failures referenced by the frozen RQ2 run selection."""
    selected_runs, repo_root = Path(selected_runs), Path(repo_root)
    with selected_runs.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    cases = []
    for row in rows:
        run_dir = _resolve_path(row["run_dir"], repo_root)
        if row["baseline"] == "DriveFuzz":
            run_cases = _collect_drivefuzz_failures(row, run_dir, repo_root)
        elif row["baseline"] in {"VLAD-Fuzz", "Instruction-CF"}:
            run_cases = _collect_local_failures(
                row, run_dir, repo_root, weather_resolver
            )
        else:
            raise ValueError(f"unsupported RQ2 method: {row['baseline']}")
        expected = int(row["failures"])
        if len(run_cases) != expected:
            raise ValueError(
                f"failure count mismatch for {run_dir}: expected {expected}, found {len(run_cases)}"
            )
        cases.extend(run_cases)
    return cases


def analyze_task_stratified_diversity(
    cases: Sequence[FailureCase],
    embeddings: np.ndarray,
    *,
    scene_feature_names: Sequence[str] = SCENE_FEATURES,
    methods: Sequence[str] = ("VLAD-Fuzz", "DriveFuzz", "Instruction-CF"),
    models: Sequence[str] = ("simlingo", "lmdrive", "bevdriver"),
    scene_weight: float = 0.5,
    max_per_method_task: int = 50,
    bootstrap_iterations: int = 1000,
    random_seed: int = 2026,
):
    """Compare failure diversity after balancing methods within navigation tasks."""
    if len(cases) != len(embeddings):
        raise ValueError("case and embedding rows must align")
    if max_per_method_task < 2:
        raise ValueError("max_per_method_task must be at least two")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")

    embeddings = np.asarray(embeddings, dtype=float)
    rng = np.random.default_rng(random_seed)
    summary_rows, comparison_rows = [], []
    diagnostics = {
        "scene_weight": scene_weight,
        "instruction_weight": 1.0 - scene_weight,
        "max_per_method_task": max_per_method_task,
        "bootstrap_iterations": bootstrap_iterations,
        "random_seed": random_seed,
        "models": {},
    }

    for model in models:
        model_indices = np.asarray(
            [index for index, case in enumerate(cases) if case.model == model]
        )
        if not len(model_indices):
            raise ValueError(f"no failure cases for {model}")
        raw_scenes = np.asarray(
            [
                [cases[index].scene[name] for name in scene_feature_names]
                for index in model_indices
            ],
            dtype=float,
        )
        normalized_scenes = _minmax_normalize(raw_scenes)
        local_index = {
            global_index: local for local, global_index in enumerate(model_indices)
        }
        tasks = sorted({cases[index].task for index in model_indices})
        groups = {
            task: {
                method: np.asarray(
                    [
                        index
                        for index in model_indices
                        if cases[index].task == task and cases[index].method == method
                    ]
                )
                for method in methods
            }
            for task in tasks
        }
        excluded = {
            task: {method: int(len(indices)) for method, indices in task_groups.items()}
            for task, task_groups in groups.items()
            if min(len(indices) for indices in task_groups.values()) < 2
        }
        comparable = [task for task in tasks if task not in excluded]
        if not comparable:
            raise ValueError(f"no comparable tasks for {model}")
        sample_sizes = {
            task: min(
                max_per_method_task, *(len(groups[task][method]) for method in methods)
            )
            for task in comparable
        }

        draws = {
            method: {component: [] for component in ("scene", "instruction", "joint")}
            for method in methods
        }
        for _ in range(bootstrap_iterations):
            sampled_tasks = rng.choice(comparable, size=len(comparable), replace=True)
            iteration = {
                method: {
                    component: [] for component in ("scene", "instruction", "joint")
                }
                for method in methods
            }
            for task in sampled_tasks:
                sample_size = sample_sizes[str(task)]
                for method in methods:
                    chosen = rng.choice(
                        groups[str(task)][method], size=sample_size, replace=False
                    )
                    local = np.asarray([local_index[int(index)] for index in chosen])
                    metrics = pairwise_diversity(
                        normalized_scenes[local],
                        embeddings[chosen],
                        scene_weight=scene_weight,
                        scene_is_normalized=True,
                    )
                    for component, value in metrics.items():
                        iteration[method][component].append(value)
            for method in methods:
                for component in iteration[method]:
                    draws[method][component].append(
                        float(np.mean(iteration[method][component]))
                    )

        for method in methods:
            joint = np.asarray(draws[method]["joint"])
            summary_rows.append(
                {
                    "model": model,
                    "method": method,
                    "comparable_tasks": len(comparable),
                    "scene_diversity": float(np.mean(draws[method]["scene"])),
                    "instruction_diversity": float(
                        np.mean(draws[method]["instruction"])
                    ),
                    "joint_diversity": float(np.mean(joint)),
                    "joint_ci_low": float(np.percentile(joint, 2.5)),
                    "joint_ci_high": float(np.percentile(joint, 97.5)),
                }
            )

        vlad_draws = np.asarray(draws["VLAD-Fuzz"]["joint"])
        for baseline in methods:
            if baseline == "VLAD-Fuzz":
                continue
            difference = vlad_draws - np.asarray(draws[baseline]["joint"])
            comparison_rows.append(
                {
                    "model": model,
                    "comparison": f"VLAD-Fuzz - {baseline}",
                    "mean_difference": float(np.mean(difference)),
                    "ci_low": float(np.percentile(difference, 2.5)),
                    "ci_high": float(np.percentile(difference, 97.5)),
                }
            )

        diagnostics["models"][model] = {
            "comparable_tasks": comparable,
            "excluded_tasks": excluded,
            "balanced_sample_size_by_task": sample_sizes,
        }

    return summary_rows, comparison_rows, diagnostics


def carla_weather_resolver(name: str) -> Sequence[float]:
    import carla

    try:
        weather = getattr(carla.WeatherParameters, name)
    except AttributeError as exc:
        raise ValueError(f"unknown CARLA weather preset: {name}") from exc
    return [float(getattr(weather, field)) for field in WEATHER_FEATURES]


def encode_instructions(
    instructions: Sequence[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    cache_dir: Path,
    output_dir: Path,
) -> np.ndarray:
    """Encode instructions with a frozen Hugging Face sentence encoder."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    unique = sorted(set(instructions))
    digest = hashlib.sha256(
        (model_name + "\0" + "\0".join(unique)).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = output_dir / f"instruction_embeddings_{digest}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_texts = cached["texts"].tolist()
        if cached_texts == unique:
            lookup = dict(zip(cached_texts, cached["embeddings"]))
            return np.asarray([lookup[text] for text in instructions])

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir))
    model.to(device)
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            texts = unique[start : start + batch_size]
            tokens = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu().numpy())
    unique_embeddings = np.concatenate(batches, axis=0)
    np.savez_compressed(
        cache_path,
        texts=np.asarray(unique),
        embeddings=unique_embeddings,
        model=np.asarray(model_name),
    )
    lookup = dict(zip(unique, unique_embeddings))
    return np.asarray([lookup[text] for text in instructions])


def source_failure_counts(cases: Sequence[FailureCase]) -> Dict[str, Dict[str, int]]:
    counts = Counter((case.model, case.method) for case in cases)
    nested = defaultdict(dict)
    for (model, method), count in sorted(counts.items()):
        nested[model][method] = count
    return dict(nested)


def _write_records(path: Path, rows: Sequence[Mapping]):
    if not rows:
        raise ValueError(f"cannot write empty records to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_diversity_summary(
    rows: Sequence[Mapping], output_path: Path, scene_weight: float
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    methods = ("VLAD-Fuzz", "DriveFuzz", "Instruction-CF")
    models = (
        ("simlingo", "SimLingo"),
        ("lmdrive", "LMDrive"),
        ("bevdriver", "BEVDriver"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), sharey=True)
    for axis, (model, title) in zip(axes, models):
        selected = {row["method"]: row for row in rows if row["model"] == model}
        x = np.arange(len(methods))
        scene = np.asarray(
            [scene_weight * selected[method]["scene_diversity"] for method in methods]
        )
        instruction = np.asarray(
            [
                (1.0 - scene_weight) * selected[method]["instruction_diversity"]
                for method in methods
            ]
        )
        totals = scene + instruction
        lower = totals - np.asarray(
            [selected[method]["joint_ci_low"] for method in methods]
        )
        upper = (
            np.asarray([selected[method]["joint_ci_high"] for method in methods])
            - totals
        )
        axis.bar(x, scene, width=0.64, color="#4C78A8", label="Scenario contribution")
        axis.bar(
            x,
            instruction,
            width=0.64,
            bottom=scene,
            color="#F28E2B",
            label="Instruction contribution",
        )
        axis.errorbar(
            x,
            totals,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#252A30",
            elinewidth=0.8,
            capsize=2.5,
        )
        axis.set_title(title, fontsize=9, fontweight="bold")
        axis.set_xticks(x, methods, fontsize=7.2)
        axis.tick_params(axis="y", labelsize=7.5)
        axis.grid(axis="y", color="#D9DDE2", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#AAB0B7")
            axis.spines[spine].set_linewidth(0.7)
    axes[0].set_ylabel("Normalized joint diversity", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-runs",
        type=Path,
        default=Path("results/rq2/paper/selected_runs.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/rq2/diversity")
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--hf-cache", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scene-weight", type=float, default=0.5)
    parser.add_argument("--max-per-method-task", type=int, default=50)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=2026)
    args = parser.parse_args()

    repo_root = Path.cwd()
    cases = collect_failure_cases(
        args.selected_runs,
        repo_root,
        weather_resolver=carla_weather_resolver,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = encode_instructions(
        [case.instruction for case in cases],
        model_name=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.hf_cache,
        output_dir=args.output_dir,
    )
    summary, comparisons, diagnostics = analyze_task_stratified_diversity(
        cases,
        embeddings,
        scene_weight=args.scene_weight,
        max_per_method_task=args.max_per_method_task,
        bootstrap_iterations=args.bootstrap_iterations,
        random_seed=args.random_seed,
    )
    diagnostics["scene_features"] = list(SCENE_FEATURES)
    diagnostics["embedding_model"] = args.embedding_model
    diagnostics["source_failure_counts"] = source_failure_counts(cases)
    _write_records(args.output_dir / "summary.csv", summary)
    _write_records(args.output_dir / "comparisons.csv", comparisons)
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_diversity_summary(
        summary,
        args.output_dir / "rq2-diversity.pdf",
        args.scene_weight,
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "comparisons": comparisons,
                "diagnostics": diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

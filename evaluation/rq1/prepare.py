from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


COMMAND_FILES = {
    "route_prior": "command_prior.json",
    "no_prior": "command_no_prior.json",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAPPING_FIELDS = [
    "case_id",
    "pair_id",
    "scenario_name",
    "seed_name",
    "sample_slot",
    "mode",
    "command_file",
    "command_index",
    "command",
    "generator_model",
    "generation_created_at",
    "prompt_path",
    "prompt_sha256",
    "source_seed_dir",
    "source_image",
    "source_scenario_json",
    "source_sha256",
    "image_sha256",
    "scenario_sha256",
    "instruction_sha256",
    "reference_sha256",
]


@dataclass(frozen=True)
class PreparationSummary:
    case_count: int
    pair_count: int
    scenario_count: int
    seed_count: int
    output_root: Path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_command_record(path: Path) -> Tuple[List[str], Dict[str, str]]:
    parsed = _load_json(path)
    metadata: Dict[str, str] = {}
    if isinstance(parsed, dict):
        metadata = {
            "generator_model": str(parsed.get("generator_model", "")),
            "generation_created_at": str(parsed.get("created_at", "")),
            "prompt_path": str(parsed.get("prompt_path", "")),
        }
        commands = parsed.get("commands")
        if isinstance(commands, list):
            return [str(item).strip() for item in commands if str(item).strip()], metadata
        instruction = parsed.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            return [instruction.strip()], metadata
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()], metadata
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()], metadata
    return [], metadata


def _find_image(seed_dir: Path) -> Optional[Path]:
    preferred = seed_dir / "ego_camera_view.png"
    if preferred.exists():
        return preferred
    ego_images = sorted(
        path for path in seed_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and "ego" in path.stem.lower()
    )
    if ego_images:
        return ego_images[0]
    images = sorted(
        path for path in seed_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return images[0] if images else None


def _find_scenario(seed_dir: Path) -> Optional[Path]:
    dynamic = sorted(seed_dir.glob("dynamic_scenario*.json"))
    if dynamic:
        return dynamic[0]
    ignored = set(COMMAND_FILES.values()) | {"commands.json", "normal_commands.json"}
    candidates = sorted(path for path in seed_dir.glob("*.json") if path.name not in ignored)
    return candidates[0] if candidates else None


def _sample_seed_dirs(
    seeds_root: Path,
    limit: Optional[int],
    rng: random.Random,
) -> Iterable[Path]:
    for scenario_dir in sorted(path for path in seeds_root.iterdir() if path.is_dir()):
        seeds = sorted(
            path for path in scenario_dir.iterdir()
            if path.is_dir() and path.name != "seed_fail"
        )
        if limit is not None and len(seeds) > limit:
            seeds = sorted(rng.sample(seeds, limit))
        yield from seeds


def _sample_commands(commands: Sequence[str], limit: Optional[int], rng: random.Random) -> List[Tuple[int, str]]:
    indexed = list(enumerate(commands, start=1))
    if limit is not None and len(indexed) > limit:
        indexed = sorted(rng.sample(indexed, limit))
    return indexed


def _resolve_prompt_path(value: str) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.exists() and path.is_file() else None


def _read_route_reference(scenario_path: Path) -> object:
    parsed = _load_json(scenario_path)
    route_info = parsed.get("route_info", {}) if isinstance(parsed, dict) else {}
    return route_info.get("route_description", "")


def _format_reference(route_description: object) -> str:
    if isinstance(route_description, str):
        route_text = route_description
    else:
        route_text = json.dumps(route_description, ensure_ascii=False, indent=2)
    return f"Route description:\n{route_text}\n"


def _write_case(case_dir: Path, image: Path, instruction: str, reference: str) -> None:
    case_dir.mkdir(parents=True)
    shutil.copy2(image, case_dir / f"image{image.suffix.lower()}")
    (case_dir / "instruction.txt").write_text(instruction + "\n", encoding="utf-8")
    (case_dir / "reference.txt").write_text(reference, encoding="utf-8")
    (case_dir / "annotation.txt").write_text("score: \nerror_type: \nnotes: \n", encoding="utf-8")


def prepare_round(
    *,
    seeds_root: Path,
    output_root: Path,
    overwrite: bool = False,
    require_both: bool = True,
    random_seed: int = 0,
    max_seeds_per_static_scenario: Optional[int] = 5,
    instructions_per_seed: Optional[int] = 1,
) -> PreparationSummary:
    seeds_root = Path(seeds_root)
    output_root = Path(output_root)
    if not seeds_root.is_dir():
        raise FileNotFoundError(f"Seed root does not exist: {seeds_root}")
    for name, value in (
        ("max_seeds_per_static_scenario", max_seeds_per_static_scenario),
        ("instructions_per_seed", instructions_per_seed),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive.")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_root}. Use overwrite=True to replace it.")
        shutil.rmtree(output_root)

    private_dir = output_root / "private"
    package_dir = output_root / "package"
    private_dir.mkdir(parents=True)
    (package_dir / "cases").mkdir(parents=True)
    evaluator_readme = Path(__file__).with_name("evaluator_readme.md").read_text(encoding="utf-8")
    (package_dir / "README.md").write_text(evaluator_readme, encoding="utf-8")

    rng = random.Random(random_seed)
    items: List[Dict[str, object]] = []
    selected_seeds = set()
    selected_scenarios = set()
    for seed_dir in _sample_seed_dirs(seeds_root, max_seeds_per_static_scenario, rng):
        available = {mode: seed_dir / filename for mode, filename in COMMAND_FILES.items() if (seed_dir / filename).is_file()}
        if require_both and set(available) != set(COMMAND_FILES):
            continue
        if not available:
            continue
        image = _find_image(seed_dir)
        scenario = _find_scenario(seed_dir)
        if image is None or scenario is None:
            continue
        route_description = _read_route_reference(scenario)
        reference = _format_reference(route_description)
        mode_samples: Dict[str, List[Tuple[int, str, Dict[str, str], Path]]] = {}
        for mode, command_path in available.items():
            commands, metadata = _load_command_record(command_path)
            mode_samples[mode] = [
                (index, command, metadata, command_path)
                for index, command in _sample_commands(commands, instructions_per_seed, rng)
            ]
        sample_lengths = [len(mode_samples.get(mode, [])) for mode in COMMAND_FILES]
        slot_count = min(sample_lengths) if require_both else max(sample_lengths, default=0)
        for slot_index in range(slot_count):
            for mode in COMMAND_FILES:
                samples = mode_samples.get(mode, [])
                if slot_index >= len(samples):
                    continue
                command_index, command, metadata, command_path = samples[slot_index]
                prompt = _resolve_prompt_path(metadata.get("prompt_path", ""))
                items.append({
                    "pair_id": f"{seed_dir.parent.name}/{seed_dir.name}/slot_{slot_index + 1}",
                    "scenario_name": seed_dir.parent.name,
                    "seed_name": seed_dir.name,
                    "sample_slot": slot_index + 1,
                    "mode": mode,
                    "command_file": str(command_path),
                    "command_index": command_index,
                    "command": command,
                    "generator_model": metadata.get("generator_model", ""),
                    "generation_created_at": metadata.get("generation_created_at", ""),
                    "prompt_path": metadata.get("prompt_path", ""),
                    "prompt_sha256": sha256_file(prompt) if prompt else "",
                    "source_seed_dir": str(seed_dir),
                    "source_image": str(image),
                    "source_scenario_json": str(scenario),
                    "source_sha256": sha256_file(command_path),
                    "image_sha256": sha256_file(image),
                    "scenario_sha256": sha256_file(scenario),
                    "instruction_sha256": sha256_text(command + "\n"),
                    "reference_sha256": sha256_text(reference),
                    "image": image,
                    "reference": reference,
                })
        if any(mode_samples.values()):
            selected_seeds.add(str(seed_dir))
            selected_scenarios.add(seed_dir.parent.name)

    rng.shuffle(items)
    rows = []
    for number, item in enumerate(items, start=1):
        case_id = f"case_{number:06d}"
        _write_case(
            package_dir / "cases" / case_id,
            item["image"],
            str(item["command"]),
            str(item["reference"]),
        )
        row = {field: item.get(field, "") for field in MAPPING_FIELDS}
        row["case_id"] = case_id
        rows.append(row)

    with (private_dir / "mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    pair_modes: Dict[str, set] = {}
    for row in rows:
        pair_modes.setdefault(str(row["pair_id"]), set()).add(str(row["mode"]))
    pair_count = sum(modes == set(COMMAND_FILES) for modes in pair_modes.values())
    config = {
        "seeds_root": str(seeds_root),
        "random_seed": random_seed,
        "require_both": require_both,
        "max_seeds_per_static_scenario": max_seeds_per_static_scenario,
        "instructions_per_seed": instructions_per_seed,
        "command_files": COMMAND_FILES,
    }
    (private_dir / "sampling_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(rows),
        "pair_count": pair_count,
        "scenario_count": len(selected_scenarios),
        "seed_count": len(selected_seeds),
        "mapping_sha256": sha256_file(private_dir / "mapping.csv"),
        "sampling_config_sha256": sha256_file(private_dir / "sampling_config.json"),
    }
    (private_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return PreparationSummary(len(rows), pair_count, len(selected_scenarios), len(selected_seeds), output_root)

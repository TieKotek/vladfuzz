import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _load_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return None


def _find_first_file(seed_dir: Path, suffixes: Iterable[str]) -> Optional[Path]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    for path in sorted(seed_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in suffix_set:
            return path
    return None


def _find_scenario_file(seed_dir: Path) -> Optional[Path]:
    ignored = {"command_prior.json", "command_no_prior.json", "commands.json", "normal_commands.json"}
    for path in sorted(seed_dir.glob("*.json")):
        if path.name not in ignored:
            return path
    return None


def _extract_command_candidates(path: Optional[Path]) -> Optional[list]:
    if path is None or not path.exists():
        return None

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return None

    parsed = _load_json(path)
    if parsed is None:
        return raw_text

    if isinstance(parsed, str):
        return [parsed.strip()]
    if isinstance(parsed, dict):
        commands = parsed.get("commands")
        if isinstance(commands, list):
            return [str(command).strip() for command in commands if str(command).strip()]
        for key in ("instruction", "command", "text", "response"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
        return [json.dumps(parsed, ensure_ascii=False)]
    if isinstance(parsed, list):
        return [str(value).strip() for value in parsed if str(value).strip()]
    return [raw_text]


def _first_candidate(candidates: Optional[list]) -> Optional[str]:
    if not candidates:
        return None
    return candidates[0]


def build_manifest(seeds_root: Path, static_root: Path) -> Iterable[Dict]:
    for scenario_dir in sorted(seeds_root.iterdir()):
        if not scenario_dir.is_dir():
            continue

        static_scenario = static_root / f"{scenario_dir.name}.json"

        for seed_dir in sorted(scenario_dir.iterdir()):
            if not seed_dir.is_dir() or seed_dir.name == "seed_fail":
                continue

            scenario_file = _find_scenario_file(seed_dir)
            if scenario_file is None:
                continue

            scenario_data = _load_json(scenario_file)
            if scenario_data is None:
                continue

            image_file = _find_first_file(seed_dir, IMAGE_SUFFIXES)
            route_info = scenario_data.get("route_info", {})
            route_prior_candidates = _extract_command_candidates(seed_dir / "command_prior.json")
            no_prior_candidates = _extract_command_candidates(seed_dir / "command_no_prior.json")

            yield {
                "id": f"{scenario_dir.name}/{seed_dir.name}",
                "static_scenario": str(static_scenario),
                "seed_scenario": str(scenario_file),
                "image": str(image_file) if image_file else None,
                "map_name": scenario_data.get("map_name"),
                "route_length": route_info.get("route_length"),
                "route_description": route_info.get("route_description"),
                "basic_instruction": route_info.get("basic_instruction"),
                "route_prior_instruction": _first_candidate(route_prior_candidates),
                "route_prior_instruction_candidates": route_prior_candidates,
                "no_prior_instruction": _first_candidate(no_prior_candidates),
                "no_prior_instruction_candidates": no_prior_candidates,
            }


def write_jsonl(rows: Iterable[Dict], output_path: Path, overwrite: bool) -> int:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a JSONL manifest that links seed scenarios, route priors, images, and generated instructions."
    )
    parser.add_argument("--seeds-root", default="test_cases", help="Root directory containing scenario seed folders.")
    parser.add_argument("--static-root", default="static_scenarios", help="Root directory containing static scenario JSON files.")
    parser.add_argument("--output", default="configs/experiment_manifest.jsonl", help="Output JSONL manifest path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists.")
    args = parser.parse_args()

    rows = build_manifest(Path(args.seeds_root), Path(args.static_root))
    count = write_jsonl(rows, Path(args.output), args.overwrite)
    print(f"Wrote {count} manifest rows to {args.output}")


if __name__ == "__main__":
    main()

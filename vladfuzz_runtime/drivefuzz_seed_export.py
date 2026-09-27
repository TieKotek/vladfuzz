import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from vladfuzz_runtime.experiment_manifest import iter_manifest_rows, resolve_instruction


def _absolute_point(static_data: Dict, relative_point: Dict) -> Dict:
    center = static_data["scenario_center"]
    absolute = {
        "x": float(relative_point["x"]) + float(center["x"]),
        "y": float(relative_point["y"]) + float(center["y"]),
        "z": float(relative_point["z"]),
    }
    if "yaw" in relative_point:
        absolute["yaw"] = float(relative_point["yaw"])
    return absolute


def _target_yaw(dynamic_data: Dict) -> float:
    waypoints = (dynamic_data.get("route_info") or {}).get("route_waypoints") or []
    if waypoints:
        rotation = waypoints[-1].get("rotation") or {}
        if "yaw" in rotation:
            return float(rotation["yaw"])
    return 0.0


def convert_dynamic_scenario_to_drivefuzz_seed(static_data: Dict, dynamic_data: Dict) -> Dict:
    ego_index = dynamic_data["ego_car"]["spawn_point_index"]
    ego_point = static_data["spawn_points"][ego_index]
    ego_abs = _absolute_point(static_data, ego_point)
    target_abs = _absolute_point(static_data, dynamic_data["target_point"])

    return {
        "map": str(dynamic_data.get("map_name") or static_data["map_name"]).split("/")[-1],
        "sp_x": ego_abs["x"],
        "sp_y": ego_abs["y"],
        "sp_z": ego_abs["z"],
        "pitch": 0.0,
        "yaw": float(ego_abs.get("yaw", 0.0)),
        "roll": 0.0,
        "wp_x": target_abs["x"],
        "wp_y": target_abs["y"],
        "wp_z": target_abs["z"],
        "wp_yaw": _target_yaw(dynamic_data),
    }


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def export_manifest_to_drivefuzz(
    manifest_path: str,
    output_dir: str,
    *,
    instruction_source: str = "basic",
    limit: Optional[int] = None,
    rows: Optional[Iterable[Dict]] = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mapping_path = output / "mapping.jsonl"

    selected_rows = rows if rows is not None else iter_manifest_rows(manifest_path)
    count = 0
    with mapping_path.open("w", encoding="utf-8") as mapping_file:
        for row in selected_rows:
            if limit is not None and count >= limit:
                break
            static_path = row["static_scenario"]
            dynamic_path = row["seed_scenario"]
            static_data = _load_json(static_path)
            dynamic_data = _load_json(dynamic_path)
            drivefuzz_seed = convert_dynamic_scenario_to_drivefuzz_seed(static_data, dynamic_data)

            seed_name = f"scene-vlad-{count:05d}.json"
            seed_path = output / seed_name
            seed_path.write_text(json.dumps(drivefuzz_seed, indent=2), encoding="utf-8")

            mapping = {
                "seed_file": seed_name,
                "manifest_id": row.get("id"),
                "static_scenario": static_path,
                "seed_scenario": dynamic_path,
                "instruction_source": instruction_source,
                "instruction": resolve_instruction(row, instruction_source, row.get("manual_instruction")),
            }
            mapping_file.write(json.dumps(mapping, ensure_ascii=True) + "\n")
            count += 1

    return mapping_path

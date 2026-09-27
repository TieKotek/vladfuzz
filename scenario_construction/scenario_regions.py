import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ScenarioRegionSpec:
    id: str
    map_name: str
    center_x: float
    center_y: float
    extent_x: float
    extent_y: float
    scenario_type: str = "intersection"
    min_spawn_distance: float = 8.0
    min_global_distance: float = 3.0
    waypoint_sample_dist: float = 1.0
    exclude_junction: bool = False
    vehicle_filter: str = "vehicle.tesla.model3"
    highway_split_ratios: Tuple[float, float, float] = (0.5, 0.0, 0.5)


def _as_ratio(value) -> Tuple[float, float, float]:
    if value is None:
        return (0.5, 0.0, 0.5)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("highway_split_ratios must be a list of three numbers")
    return tuple(float(item) for item in value)


def _region_from_dict(data: dict, source: Path, line_number: int) -> ScenarioRegionSpec:
    required = ("id", "map_name", "center_x", "center_y", "extent_x", "extent_y")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{source}:{line_number} missing required fields: {', '.join(missing)}")

    return ScenarioRegionSpec(
        id=str(data["id"]),
        map_name=str(data["map_name"]),
        center_x=float(data["center_x"]),
        center_y=float(data["center_y"]),
        extent_x=float(data["extent_x"]),
        extent_y=float(data["extent_y"]),
        scenario_type=str(data.get("scenario_type", "intersection")),
        min_spawn_distance=float(data.get("min_spawn_distance", 8.0)),
        min_global_distance=float(data.get("min_global_distance", 3.0)),
        waypoint_sample_dist=float(data.get("waypoint_sample_dist", 1.0)),
        exclude_junction=bool(data.get("exclude_junction", False)),
        vehicle_filter=str(data.get("vehicle_filter", "vehicle.tesla.model3")),
        highway_split_ratios=_as_ratio(data.get("highway_split_ratios")),
    )


def load_region_specs(path: Path) -> List[ScenarioRegionSpec]:
    specs: List[ScenarioRegionSpec] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            specs.append(_region_from_dict(json.loads(line), path, line_number))
    return specs


def select_region_specs(
    specs: Iterable[ScenarioRegionSpec],
    scenario_id: Optional[str],
    include_all: bool,
) -> List[ScenarioRegionSpec]:
    specs = list(specs)
    if include_all:
        return specs
    if scenario_id is None:
        return specs
    selected = [spec for spec in specs if spec.id == scenario_id]
    if not selected:
        available = ", ".join(spec.id for spec in specs[:10])
        raise ValueError(f"Scenario region '{scenario_id}' not found. Available examples: {available}")
    return selected

import argparse
from pathlib import Path

import carla

from scenario_construction.scenario_constructor import ScenarioConstructor
from scenario_construction.scenario_regions import load_region_specs, select_region_specs


def diagnose_region(spec, args):
    print(f"\n=== Diagnose {spec.id} ({spec.map_name}) ===")
    constructor = ScenarioConstructor(
        host=args.host,
        port=args.port,
        map_name=spec.map_name,
        timeout=args.timeout,
        map_warmup_seconds=args.map_warmup_seconds,
        warmup_ticks=args.warmup_ticks,
    )
    bbox = {
        "min_x": spec.center_x - spec.extent_x,
        "max_x": spec.center_x + spec.extent_x,
        "min_y": spec.center_y - spec.extent_y,
        "max_y": spec.center_y + spec.extent_y,
        "center": {"x": spec.center_x, "y": spec.center_y},
        "extent": {"x": spec.extent_x, "y": spec.extent_y},
    }
    waypoints = constructor._get_waypoints_in_bbox(bbox, spec.waypoint_sample_dist)
    scenario_data = constructor._create_spawnable_lanes_data(
        waypoints,
        bbox,
        spec.min_spawn_distance,
        spec.min_global_distance,
        spec.exclude_junction,
    )
    points = [sp for lane in scenario_data["lanes"] for sp in lane["spawn_points"]]
    print(f"waypoints={len(waypoints)} candidate_points={len(points)} lanes={len(scenario_data['lanes'])}")

    blueprint = constructor.world.get_blueprint_library().find(spec.vehicle_filter)
    print(f"blueprint={blueprint.id}")

    offsets = [float(value) for value in args.z_offsets.split(",")]
    success = {offset: 0 for offset in offsets}
    samples = []

    for index, spawn_point in enumerate(points):
        base_location = carla.Location(
            x=spawn_point["x"],
            y=spawn_point["y"],
            z=spawn_point["z"],
        )
        waypoint = constructor.carla_map.get_waypoint(
            base_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if index < args.sample_count:
            samples.append(
                {
                    "index": index,
                    "spawn": (
                        round(spawn_point["x"], 2),
                        round(spawn_point["y"], 2),
                        round(spawn_point["z"], 2),
                    ),
                    "waypoint_z": None if waypoint is None else round(waypoint.transform.location.z, 2),
                    "waypoint_yaw": None if waypoint is None else round(waypoint.transform.rotation.yaw, 2),
                    "road_lane": None if waypoint is None else (waypoint.road_id, waypoint.lane_id),
                }
            )
        base_transform = waypoint.transform if waypoint else carla.Transform(
            base_location,
            carla.Rotation(yaw=spawn_point["yaw"]),
        )
        for offset in offsets:
            transform = carla.Transform(
                carla.Location(
                    x=base_transform.location.x,
                    y=base_transform.location.y,
                    z=base_transform.location.z + offset,
                ),
                base_transform.rotation,
            )
            actor = constructor.world.try_spawn_actor(blueprint, transform)
            if actor:
                success[offset] += 1
                actor.destroy()
                constructor._tick_or_wait(seconds=1.0)

    print(f"samples={samples}")
    print(f"success_by_z_offset={success}")
    constructor._cleanup_spawned_actors()


def main():
    parser = argparse.ArgumentParser(description="Diagnose CARLA spawn success for static scenario regions.")
    parser.add_argument("--regions", default="configs/scenario_regions.jsonl")
    parser.add_argument("--scenario-id")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--map-warmup-seconds", type=float, default=5.0)
    parser.add_argument("--warmup-ticks", type=int, default=3)
    parser.add_argument("--z-offsets", default="0.0,0.1,0.3,0.5,1.0,1.5,2.0")
    parser.add_argument("--sample-count", type=int, default=5)
    args = parser.parse_args()

    specs = select_region_specs(load_region_specs(Path(args.regions)), args.scenario_id, False)
    for spec in specs:
        diagnose_region(spec, args)


if __name__ == "__main__":
    main()

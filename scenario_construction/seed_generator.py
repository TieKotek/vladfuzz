import argparse
import os
import sys
import random
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario.carla_scenario import CarlaScenario
from vladfuzz_runtime.model_registry import bootstrap_model_environment


def static_scenario_output_dir(static_scenario_path: Path, output_root: Path) -> Path:
    return output_root / static_scenario_path.stem


def route_signature(route_info) -> str:
    maneuvers = route_info.get("maneuvers", [])
    compressed = []
    for maneuver in maneuvers:
        if not compressed or compressed[-1] != maneuver:
            compressed.append(maneuver)
    return ">".join(compressed)


def candidate_group_key(candidate) -> tuple:
    return (
        candidate.get("target_road_id"),
        candidate.get("target_lane_id"),
        route_signature(candidate.get("route_info", {})),
    )


def group_seed_candidates(candidates):
    groups = OrderedDict()
    for candidate in candidates:
        key = candidate_group_key(candidate)
        groups.setdefault(key, []).append(candidate)
    return groups


def select_diverse_candidates(candidates, max_seeds=None, candidates_per_group=1):
    groups = group_seed_candidates(candidates)
    selected = []
    for group_candidates in groups.values():
        for candidate in group_candidates[:candidates_per_group]:
            selected.append(candidate)
            if max_seeds is not None and len(selected) >= max_seeds:
                return selected
    return selected


def build_scenario_manager_kwargs(
    *,
    host,
    tm_seed,
    timeout,
    frame_rate,
    gpu_id,
    model,
    dynamic_filter,
):
    kwargs = {
        "host": host,
        "tm_seed": tm_seed,
        "timeout": timeout,
        "frame_rate": frame_rate,
        "gpu_id": gpu_id,
    }
    if dynamic_filter:
        spec = bootstrap_model_environment(model)
        kwargs.update({
            "config_path": str(spec.config_path),
            "hydra_config_path": str(spec.hydra_config_path) if spec.hydra_config_path else None,
            "model_name": spec.name,
        })
    return kwargs


def create_seeds(
    scenario_manager: CarlaScenario,
    static_scenario_path,
    output_dir,
    seeds_per_static_scenario,
    dynamic_filter=False,
    duration_frames=500,
    min_npc_count=0,
    max_npc_count=0,
    min_speed_diff_perc=-40.0,
    max_speed_diff_perc=10.0,
    min_target_distance=30,
    max_route_distance=100,
    overwrite=False,
    strategy="random",
    max_seeds_per_scenario=None,
    candidates_per_group=1,
):
    try:
        fail_count = 0
        os.makedirs(output_dir, exist_ok=True)
        if dynamic_filter:
            os.makedirs(os.path.join(output_dir, "seed_fail"), exist_ok=True)
        
        if not scenario_manager.load_static_scenario(static_scenario_path):
            print(f"Could not load static scenario {static_scenario_path}, skipping.")
            return

        if strategy == "diverse":
            candidates = scenario_manager.enumerate_seed_candidates(
                min_target_distance=min_target_distance,
                max_route_distance=max_route_distance,
            )
            random.shuffle(candidates)
            selected_candidates = select_diverse_candidates(
                candidates,
                max_seeds=max_seeds_per_scenario,
                candidates_per_group=candidates_per_group,
            )
            print(f"Selected {len(selected_candidates)} diverse seeds from {len(candidates)} candidates")

            for i, candidate in enumerate(selected_candidates):
                scenario_dir = os.path.join(output_dir, f"seed_{i + 1}")
                if os.path.exists(scenario_dir) and os.listdir(scenario_dir) and not overwrite:
                    print(f"Skipping existing seed directory: {scenario_dir}")
                    continue
                os.makedirs(scenario_dir, exist_ok=True)
                metadata = {
                    "strategy": "diverse",
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_group": "::".join(str(part) for part in candidate_group_key(candidate)),
                    "ego_road_id": candidate.get("ego_road_id"),
                    "ego_lane_id": candidate.get("ego_lane_id"),
                    "target_road_id": candidate.get("target_road_id"),
                    "target_lane_id": candidate.get("target_lane_id"),
                    "route_signature": route_signature(candidate.get("route_info", {})),
                    "direct_distance": candidate.get("direct_distance"),
                }
                dynamic_scenario_path = scenario_manager.generate_scenario_from_candidate(
                    candidate,
                    duration_frames=duration_frames,
                    output_dir=scenario_dir,
                    min_npc_count=min_npc_count,
                    max_npc_count=max_npc_count,
                    min_speed_diff_perc=min_speed_diff_perc,
                    max_speed_diff_perc=max_speed_diff_perc,
                    include_environment=False,
                    seed_generation_metadata=metadata,
                )
                if not scenario_manager.load_scenario(dynamic_scenario_path):
                    print(f"Error: Failed to load dynamic scenario from '{dynamic_scenario_path}'")
                    os.remove(dynamic_scenario_path)
                    continue
                print(f"✅ Diverse scenario {i+1} saved.")
                scenario_manager.capture_image(image_dir=scenario_dir)
                scenario_manager.cleanup_actors()
            return
        
        i = 0
        while i < seeds_per_static_scenario:
            scenario_dir = os.path.join(output_dir, f"seed_{i + 1}")
            if os.path.exists(scenario_dir) and os.listdir(scenario_dir) and not overwrite:
                print(f"Skipping existing seed directory: {scenario_dir}")
                i += 1
                continue
            os.makedirs(scenario_dir, exist_ok=True)

            # Generate a random dynamic scenario without NPCs
            dynamic_scenario_path = scenario_manager.generate_random_scenario(
                output_dir=scenario_dir,
                duration_frames=duration_frames,
                min_npc_count=min_npc_count,
                max_npc_count=max_npc_count,
                min_speed_diff_perc=min_speed_diff_perc,
                max_speed_diff_perc=max_speed_diff_perc,
                min_target_distance=min_target_distance,
                max_route_distance=max_route_distance,
            )
            # Load the generated dynamic scenario
            if not scenario_manager.load_scenario(dynamic_scenario_path):
                print(f"Error: Failed to load dynamic scenario from '{dynamic_scenario_path}'")
                os.remove(dynamic_scenario_path)
                continue

            if dynamic_filter:
                # Construct a natural language instruction from the route description
                language_instruction = scenario_manager.scenario_config.route_info.get('basic_instruction', None)

                # Dynamically test the scenario
                result = scenario_manager.start_scenario(
                    language_instruction=language_instruction, 
                    success_distance=5.0
                )
                
                if (not result["error"]) and result["completed"]:
                    print(f"✅ Scenario {i+1} succeeded and is saved.")
                    # The scenario is valid, take a snapshot
                    scenario_manager.capture_image(image_dir=scenario_dir)
                    i += 1
                else:
                    print(f"❌ Scenario failed validation, retrying...")
                    # Clean up the failed scenario file
                    try:
                        # copy to seed_fail folder for analysis
                        fail_scenario_path = os.path.join(output_dir, "seed_fail", f"failed_scenario_{fail_count + 1}.json")
                        os.rename(dynamic_scenario_path, fail_scenario_path)
                        fail_count += 1
                        os.remove(dynamic_scenario_path)
                        
                    except OSError as e:
                        print(f"Error removing file {dynamic_scenario_path}: {e}")
                    # The loop will continue and generate a new scenario for the same seed number.
            
            else:
                print(f"✅ Scenario {i+1} succeeded and is saved.")
                scenario_manager.capture_image(image_dir=scenario_dir)
                i += 1
            
            scenario_manager.cleanup_actors()
            
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Generate dynamic seed scenarios from static scenario files.")
    parser.add_argument("--model", choices=["lmdrive", "simlingo", "bevdriver"], default="simlingo")
    parser.add_argument("--static-root", default="static_scenarios")
    parser.add_argument("--static-scenario", action="append", dest="static_scenarios", help="Specific static scenario JSON. Can be repeated.")
    parser.add_argument("--output-root", default="seeds")
    parser.add_argument("--seeds-per-scenario", type=int, default=5)
    parser.add_argument("--dynamic-filter", action="store_true")
    parser.add_argument("--duration-frames", type=int, default=500)
    parser.add_argument("--min-npc-count", type=int, default=0)
    parser.add_argument("--max-npc-count", type=int, default=0)
    parser.add_argument("--min-speed-diff-perc", type=float, default=-40.0)
    parser.add_argument("--max-speed-diff-perc", type=float, default=10.0)
    parser.add_argument("--min-target-distance", type=float, default=30)
    parser.add_argument("--max-route-distance", type=float, default=100)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--tm-seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--frame-rate", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strategy", choices=["random", "diverse"], default="random")
    parser.add_argument("--max-seeds-per-scenario", type=int, help="Maximum selected seeds per static scenario for --strategy diverse.")
    parser.add_argument("--candidates-per-group", type=int, default=1, help="Number of candidates sampled from each route-diversity group.")
    parser.add_argument("--random-seed", type=int, help="Seed Python random choices for reproducible seed selection.")
    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)

    manager_kwargs = build_scenario_manager_kwargs(
        host=args.host,
        tm_seed=args.tm_seed,
        timeout=args.timeout,
        frame_rate=args.frame_rate,
        gpu_id=args.gpu_id,
        model=args.model,
        dynamic_filter=args.dynamic_filter,
    )
    scenario_manager = CarlaScenario(**manager_kwargs)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.static_scenarios:
        static_scenario_paths = [Path(path) for path in args.static_scenarios]
    else:
        static_scenario_paths = sorted(Path(args.static_root).glob("*.json"))

    try:
        for static_scenario_path in static_scenario_paths:
            scenario_output_dir = static_scenario_output_dir(static_scenario_path, output_root)
            scenario_output_dir.mkdir(parents=True, exist_ok=True)
            create_seeds(
                scenario_manager,
                str(static_scenario_path),
                str(scenario_output_dir),
                args.seeds_per_scenario,
                dynamic_filter=args.dynamic_filter,
                duration_frames=args.duration_frames,
                min_npc_count=args.min_npc_count,
                max_npc_count=args.max_npc_count,
                min_speed_diff_perc=args.min_speed_diff_perc,
                max_speed_diff_perc=args.max_speed_diff_perc,
                min_target_distance=args.min_target_distance,
                max_route_distance=args.max_route_distance,
                overwrite=args.overwrite,
                strategy=args.strategy,
                max_seeds_per_scenario=args.max_seeds_per_scenario,
                candidates_per_group=args.candidates_per_group,
            )
    finally:
        scenario_manager.destroy()

if __name__ == "__main__":
    main()

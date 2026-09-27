import argparse

from vladfuzz_runtime.experiment_manifest import INSTRUCTION_SOURCES, apply_manifest_selection
from vladfuzz_runtime.model_registry import bootstrap_model_environment


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single CARLA scenario with a selected VLA model.")
    parser.add_argument("--model", choices=["lmdrive", "simlingo", "bevdriver"], default="lmdrive")
    parser.add_argument("--static-scenario", help="Static scenario JSON for direct input mode")
    parser.add_argument("--dynamic-scenario", help="Dynamic scenario JSON for direct input mode")
    parser.add_argument("--instruction", help="Navigation instruction for direct input mode")
    parser.add_argument("--manifest", help="Optional JSONL experiment manifest produced by build_instruction_manifest.py")
    parser.add_argument("--manifest-id", help="Manifest row id, for example town01_t_junction_1/seed_1")
    parser.add_argument("--instruction-source", choices=INSTRUCTION_SOURCES, default="manual")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--tm-seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--frame-rate", type=int, default=20)
    parser.add_argument("--success-distance", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = bootstrap_model_environment(args.model)
    selection = apply_manifest_selection(
        manifest_path=args.manifest,
        manifest_id=args.manifest_id,
        instruction_source=args.instruction_source,
        static_scenario=args.static_scenario,
        dynamic_scenario=args.dynamic_scenario,
        instruction=args.instruction,
    )

    from scenario.carla_scenario import CarlaScenario

    scenario_manager = None
    try:
        scenario_manager = CarlaScenario(
            host=args.host,
            tm_seed=args.tm_seed,
            timeout=args.timeout,
            frame_rate=args.frame_rate,
            gpu_id=args.gpu_id,
            model_name=spec.name,
            config_path=str(spec.config_path),
            hydra_config_path=str(spec.hydra_config_path) if spec.hydra_config_path else None,
        )

        print("Experiment input:")
        print(f"  Manifest ID: {selection['manifest_id']}")
        print(f"  Static scenario: {selection['static_scenario']}")
        print(f"  Dynamic scenario: {selection['dynamic_scenario']}")
        print(f"  Instruction source: {selection['instruction_source']}")
        print(f"  Instruction: {selection['instruction']}")

        if not scenario_manager.load_static_scenario(selection["static_scenario"]):
            print(f"Could not load static scenario {selection['static_scenario']}, skipping.")
            return

        if not scenario_manager.load_scenario(selection["dynamic_scenario"]):
            print(f"Error: Failed to load dynamic scenario from '{selection['dynamic_scenario']}'")
            return

        result = scenario_manager.start_scenario(
            language_instruction=selection["instruction"],
            success_distance=args.success_distance,
        )
        print(result)
    except Exception as exc:
        print(f"❌ Execution failed: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if scenario_manager:
            scenario_manager.destroy()


if __name__ == "__main__":
    main()

import os
import json
import time
import random
import argparse
from datetime import datetime
from vladfuzz_runtime.experiment_manifest import INSTRUCTION_SOURCES, apply_manifest_selection
from vladfuzz_runtime.failure_analysis import collect_failure_analysis
from vladfuzz_runtime.run_metadata import create_run_dir
from vladfuzz_runtime.oracle import parse_oracle_checks
from vladfuzz_runtime.hourly_checkpoints import HourlyCheckpointRecorder
from vladfuzz_runtime.npc_count_range import sample_npc_count, validate_npc_count_range

class BaselineTester:
    def __init__(
        self,
        target_optimization_dir,
        vla_model="lmdrive",
        gpu_id=0,
        manifest=None,
        manifest_id=None,
        instruction_source=None,
        instruction=None,
        max_simulations=None,
        standalone=False,
        static_scenario=None,
        seed_scenario=None,
        output_root="results",
        random_seed=None,
        method="random",
        target_duration=None,
        oracle_checks=None,
        scenario_duration_frames=500,
        min_npc_count=0,
        max_npc_count=3,
    ):
        self.target_optimization_dir = target_optimization_dir
        self.vla_model = vla_model
        self.gpu_id = gpu_id
        self.max_simulations = max_simulations
        self.standalone = standalone
        self.output_root = output_root
        self.random_seed = random_seed
        self.method = method
        self.oracle_checks = oracle_checks
        self.requested_target_duration = target_duration
        self.min_npc_count, self.max_npc_count = validate_npc_count_range(min_npc_count, max_npc_count)
        self.target_metadata = self._load_target_metadata() if not self.standalone else None
        if self.standalone:
            self.config = {
                "static_scenario": static_scenario,
                "seed_scenario": seed_scenario,
                "instruction": instruction,
                "instruction_source": instruction_source,
            }
            if manifest is None and (static_scenario is None or seed_scenario is None):
                raise ValueError("Standalone baseline requires --manifest or both --static-scenario and --seed-scenario")

        config_instruction = self.config.get('instruction') if not self.standalone else instruction
        if config_instruction == "Basic instruction only (No fuzzing)":
            config_instruction = None
        resolved_instruction_source = instruction_source or self.config.get('instruction_source') or "manual"
        resolved_instruction = instruction or config_instruction
        if manifest is None and resolved_instruction_source != "manual":
            resolved_instruction_source = "manual"
        if manifest is None and resolved_instruction is None:
            scenario_for_instruction = seed_scenario or self.config['seed_scenario']
            with open(scenario_for_instruction, 'r') as f:
                seed_data = json.load(f)
            resolved_instruction = seed_data['route_info'].get('basic_instruction', "")

        self.selection = apply_manifest_selection(
            manifest_path=manifest,
            manifest_id=manifest_id,
            instruction_source=resolved_instruction_source,
            static_scenario=static_scenario or self.config.get('static_scenario'),
            dynamic_scenario=seed_scenario or self.config.get('seed_scenario'),
            instruction=resolved_instruction,
        )
        self.static_scenario = self.selection["static_scenario"]
        self.seed_scenario = self.selection["dynamic_scenario"]
        self.instruction = self.selection["instruction"]
        self.instruction_source = self.selection["instruction_source"]
        self.manifest_id = self.selection["manifest_id"]
        
        # Setup directories
        if self.standalone:
            self.base_dir = create_run_dir(
                self.output_root,
                self.method,
                self.vla_model,
                self.manifest_id,
                self.random_seed,
            )
        else:
            self.base_dir = os.path.join(self.target_optimization_dir, "baseline")
        self.failure_dir = os.path.join(self.base_dir, "failures")
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.failure_dir, exist_ok=True)
        self.checkpoint_recorder = None
        print(f"📂 Baseline output directory: {self.base_dir}")

        # Initialize Fuzzer (used for scenario generation and evaluation)
        print("Initializing LocalFuzzer for Baseline Testing...")
        from vladfuzz_workflows.local_fuzzer import LocalFuzzer

        self.fuzzer = LocalFuzzer(
            static_scenario=self.static_scenario,
            mutation_depth=0, # No mutation for baseline
            success_distance=5.0,
            vla_model=self.vla_model,
            gpu_id=self.gpu_id,
            verbose=False,
            oracle_checks=oracle_checks,
            scenario_duration_frames=scenario_duration_frames,
        )
        
        # Metrics tracking
        self.start_time = 0
        self.simulation_count = 0
        self.total_failures = 0
        if self.requested_target_duration is not None:
            self.target_duration = self.requested_target_duration
        else:
            self.target_duration = None if self.standalone else self.target_metadata['total_duration_seconds']
        
        if self.target_duration is not None:
            print(f"⏱️  Target duration from optimization: {self.target_duration:.2f} seconds")
        if self.max_simulations is not None:
            print(f"🔢 Max simulations: {self.max_simulations}")

    def _load_target_metadata(self):
        """Load metadata from the target optimization directory."""
        metadata_path = os.path.join(self.target_optimization_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            
        self.config = data.get('configuration', {})
        if not self.config:
            raise ValueError("Configuration missing in target metadata")
            
        return data

    def run(self):
        self.start_time = time.time()
        self.checkpoint_recorder = HourlyCheckpointRecorder(
            output_dir=self.base_dir,
            start_time=self.start_time,
            method=self.method,
            model=self.vla_model,
            manifest_id=self.manifest_id,
        )
        seed_scenario_path = self.seed_scenario
        
        print("🚀 Starting baseline testing loop...")
        
        while True:
            if self.target_duration is not None and (time.time() - self.start_time) >= self.target_duration:
                break
            if self.max_simulations is not None and self.simulation_count >= self.max_simulations:
                print(f"Reached max simulation budget: {self.max_simulations}")
                break
            current_duration = time.time() - self.start_time
            if self.target_duration is not None:
                remaining = self.target_duration - current_duration
                print(f"\n🔄 Simulation {self.simulation_count + 1} | Time elapsed: {current_duration:.1f}s / {self.target_duration:.1f}s | Remaining: {remaining:.1f}s")
            else:
                print(f"\n🔄 Simulation {self.simulation_count + 1} | Time elapsed: {current_duration:.1f}s")
            
            # 1. Generate Random Scenario
            npc_count = sample_npc_count(self.min_npc_count, self.max_npc_count)
            scenario_data = None
            try:
                scenario_data = self.fuzzer.scenario_manager.generate_scenario_from_seed(
                    seed_scenario_path=seed_scenario_path,
                    npc_count=npc_count,
                    min_speed_diff_perc=-50.0,
                    max_speed_diff_perc=50.0,
                    include_environment=True,
                    return_dict=True
                )
            except Exception as e:
                print(f"❌ Scenario generation failed: {e}")
                continue

            # 2. Run Evaluation using LocalFuzzer
            try:
                # Call fuzzer.evaluate directly
                # This handles: scenario loading, execution, semantic checks, and failure saving
                safety, task, detailed_log = self.fuzzer.evaluate(
                    dynamic_scenario=scenario_data,
                    language_instruction=self.instruction,
                    save_failures=True,
                    failure_dir=self.failure_dir,
                    return_detailed_log=True
                )
                
                # 3. Update Statistics
                # In baseline (depth=0), detailed_log has only 1 node (root)
                # If semantically_correct is False, it's a failure
                self.simulation_count += len(detailed_log)
                
                failures_in_eval = sum(1 for node in detailed_log if not node.get('semantically_correct', True))
                if failures_in_eval > 0:
                    self.total_failures += failures_in_eval
                    print(f"⚠️ Failure detected in this run.")
                self._update_hourly_checkpoints()
                    
            except Exception as e:
                print(f"❌ Execution exception: {e}")
                self.simulation_count += 1 
                # Note: If fuzzer.evaluate raises an exception, it might not have saved the failure.
                # However, LocalFuzzer usually handles internal exceptions and returns partial results 
                # or saves them if possible. If it crashes completely, we count it but might miss the log.
                self._update_hourly_checkpoints(extra={"last_exception": str(e)})
        
        self.save_final_metadata()
        print("\n✅ Baseline testing completed.")

    def _update_hourly_checkpoints(self, extra=None):
        if self.checkpoint_recorder is None:
            return
        self.checkpoint_recorder.update(
            now=time.time(),
            executions=self.simulation_count,
            failures=self.total_failures,
            extra=extra or {},
        )

    def save_final_metadata(self):
        end_time = time.time()
        actual_duration = end_time - self.start_time
        
        # Analyze failures
        failure_stats = collect_failure_analysis(self.failure_dir)

        metadata = {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "actual_duration_seconds": round(actual_duration, 2),
            "target_duration_seconds": self.target_duration,
            "total_simulations_executed": self.simulation_count,
            "total_failures_detected": self.total_failures,
            "failure_analysis": failure_stats,
            "target_optimization_dir": self.target_optimization_dir,
            "hourly_checkpoints": list(self.checkpoint_recorder.checkpoints) if self.checkpoint_recorder else [],
            "configuration": {
                "method": self.method,
                "static_scenario": self.config.get('static_scenario'),
                "seed_scenario": self.config.get('seed_scenario'),
                "resolved_static_scenario": self.static_scenario,
                "resolved_seed_scenario": self.seed_scenario,
                "instruction": self.instruction,
                "instruction_source": self.instruction_source,
                "manifest_id": self.manifest_id,
                "vla_model": self.vla_model,
                "max_simulations": self.max_simulations,
                "random_seed": self.random_seed,
                "oracle_checks": self.oracle_checks,
                "min_npc_count": self.min_npc_count,
                "max_npc_count": self.max_npc_count,
            }
        }
        
        metadata_path = os.path.join(self.base_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"📋 Saved baseline metadata to {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline testing against an optimization run.")
    parser.add_argument("target_dir", type=str, nargs="?", help="Path to the target optimization directory (legacy matched-baseline mode)")
    parser.add_argument("--standalone", action="store_true", help="Run a fixed-budget random baseline without a target optimization directory")
    parser.add_argument("--model", choices=["lmdrive", "simlingo", "bevdriver"], default="lmdrive")
    parser.add_argument("--manifest", help="Optional JSONL experiment manifest produced by build_instruction_manifest.py")
    parser.add_argument("--manifest-id", help="Manifest row id, for example scenario_Town01_20251013_122747/seed_1")
    parser.add_argument("--instruction-source", choices=INSTRUCTION_SOURCES)
    parser.add_argument("--instruction", help="Manual instruction override")
    parser.add_argument("--static-scenario", help="Static scenario path for standalone/manual mode")
    parser.add_argument("--seed-scenario", help="Seed dynamic scenario path for standalone/manual mode")
    parser.add_argument("--max-simulations", type=int, help="Stop baseline after this many executed simulations")
    parser.add_argument("--time-budget-seconds", type=float, help="Stop starting new simulations after this many wall-clock seconds")
    parser.add_argument("--time-budget-minutes", type=float, help="Stop starting new simulations after this many wall-clock minutes")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--random-seed", type=int, help="Random seed for baseline scenario generation")
    parser.add_argument("--method", default="random", help="Experiment method label stored in metadata")
    parser.add_argument("--output-root", default="results/rq2/runs", help="Root directory for standalone run output")
    parser.add_argument("--oracle-checks", default="default", help="Comma-separated oracle checks: collision,stuck,lane,speed,other; or all/default/none.")
    parser.add_argument("--scenario-duration-frames", type=int, default=500, help="Runtime duration in CARLA frames for each random-baseline scenario execution.")
    parser.add_argument("--min-npc-count", type=int, default=0)
    parser.add_argument("--max-npc-count", type=int, default=3)

    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)

    if not args.standalone and not args.target_dir:
        print("❌ Error: target_dir is required unless --standalone is used.")
        exit(1)
    target_duration = args.time_budget_seconds
    if args.time_budget_minutes is not None:
        target_duration = args.time_budget_minutes * 60.0

    if args.standalone and args.max_simulations is None and target_duration is None:
        print("❌ Error: --max-simulations or a wall-time budget is required in --standalone mode.")
        exit(1)

    if args.target_dir and not os.path.exists(args.target_dir):
        print(f"❌ Error: Target directory '{args.target_dir}' does not exist.")
        exit(1)

    tester = BaselineTester(
        args.target_dir,
        vla_model=args.model,
        gpu_id=args.gpu_id,
        manifest=args.manifest,
        manifest_id=args.manifest_id,
        instruction_source=args.instruction_source,
        instruction=args.instruction,
        max_simulations=args.max_simulations,
        standalone=args.standalone,
        static_scenario=args.static_scenario,
        seed_scenario=args.seed_scenario,
        output_root=args.output_root,
        random_seed=args.random_seed,
        method=args.method,
        target_duration=target_duration,
        oracle_checks=parse_oracle_checks(args.oracle_checks),
        scenario_duration_frames=args.scenario_duration_frames,
        min_npc_count=args.min_npc_count,
        max_npc_count=args.max_npc_count,
    )
    tester.run()

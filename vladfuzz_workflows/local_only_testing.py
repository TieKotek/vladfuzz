"""Budget-matched Local-only ablation for VLAD-Fuzz RQ3."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Callable, Dict, Optional

from vladfuzz_runtime.experiment_manifest import apply_manifest_selection
from vladfuzz_runtime.failure_analysis import collect_failure_analysis
from vladfuzz_runtime.hourly_checkpoints import HourlyCheckpointRecorder
from vladfuzz_runtime.infrastructure import (
    CARLA_INFRASTRUCTURE_EXIT_CODE,
    CarlaInfrastructureError,
)
from vladfuzz_runtime.mutation_operators import parse_operator_list
from vladfuzz_runtime.oracle import parse_oracle_checks
from vladfuzz_runtime.run_metadata import create_run_dir, write_metadata


def _append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class LocalOnlyTester:
    """Repeatedly fuzz language while keeping the selected dynamic scenario fixed."""

    def __init__(
        self,
        *,
        manifest: str,
        manifest_id: Optional[str],
        model: str,
        gpu_id: int,
        output_root: str,
        time_budget_seconds: float,
        random_seed: int = 0,
        mutation_depth: int = 2,
        operators=None,
        oracle_checks: Optional[Dict[str, bool]] = None,
        scenario_duration_frames: int = 500,
        success_distance: float = 5.0,
        api_provider: str = "deepseek",
        model_name: str = "deepseek-v4-flash",
        clock: Callable[[], float] = time.time,
        fuzzer_factory=None,
        checkpoint_interval_seconds: float = 3600.0,
    ):
        if time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive")
        if mutation_depth < 1:
            raise ValueError("mutation_depth must be at least one for Local-only")

        self.selection = apply_manifest_selection(
            manifest_path=manifest,
            manifest_id=manifest_id,
            instruction_source="route_prior",
            static_scenario=None,
            dynamic_scenario=None,
            instruction=None,
        )
        self.manifest_id = self.selection["manifest_id"]
        self.model = model
        self.gpu_id = gpu_id
        self.time_budget_seconds = float(time_budget_seconds)
        self.random_seed = int(random_seed)
        self.mutation_depth = int(mutation_depth)
        self.operators = list(operators) if operators is not None else parse_operator_list("all")
        self.oracle_checks = oracle_checks
        self.scenario_duration_frames = int(scenario_duration_frames)
        self.success_distance = float(success_distance)
        self.api_provider = api_provider
        self.model_name = model_name
        self.clock = clock
        self.checkpoint_interval_seconds = float(checkpoint_interval_seconds)
        # Match the Full/NSGA campaign boundary: backend initialization counts
        # toward the wall-clock budget.
        self.start_time = self.clock()

        self.base_dir = create_run_dir(
            output_root,
            "local_only",
            model,
            self.manifest_id,
            self.random_seed,
        )
        self.base_path = Path(self.base_dir)
        self.failure_dir = self.base_path / "failures"
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_path = self.base_path / "evaluation_results.jsonl"
        self._record_run_dir()

        if fuzzer_factory is None:
            from vladfuzz_workflows.local_fuzzer import LocalFuzzer
            fuzzer_factory = LocalFuzzer
        self.fuzzer = fuzzer_factory(
            static_scenario=self.selection["static_scenario"],
            mutation_depth=self.mutation_depth,
            success_distance=self.success_distance,
            api_provider=self.api_provider,
            model_name=self.model_name,
            vla_model=self.model,
            gpu_id=self.gpu_id,
            verbose=False,
            semantic_pruning=False,
            operators=self.operators,
            oracle_checks=self.oracle_checks,
            scenario_duration_frames=self.scenario_duration_frames,
        )

        self.checkpoint_recorder = HourlyCheckpointRecorder(
            output_dir=self.base_dir,
            start_time=self.start_time,
            interval_seconds=self.checkpoint_interval_seconds,
            method="local_only",
            model=self.model,
            manifest_id=self.manifest_id,
        )
        self.evaluation_count = 0
        self.simulation_count = 0
        self.total_failures = 0

    def _record_run_dir(self) -> None:
        marker = os.environ.get("VLADFUZZ_RUN_DIR_FILE")
        if not marker:
            return
        marker_path = Path(marker)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        with marker_path.open("a", encoding="utf-8") as file:
            file.write(self.base_dir + "\n")

    def _budget_exhausted(self) -> bool:
        return (self.clock() - self.start_time) >= self.time_budget_seconds

    @staticmethod
    def _failed_nodes(detailed_log) -> int:
        return sum(
            1
            for node in detailed_log
            if not node.get("valid", node.get("semantically_correct", True))
        )

    def run(self) -> None:
        while not self.simulation_count or not self._budget_exhausted():
            self.evaluation_count += 1
            safety_score, task_score, detailed_log = self.fuzzer.evaluate(
                dynamic_scenario=self.selection["dynamic_scenario"],
                language_instruction=self.selection["instruction"],
                save_failures=True,
                failure_dir=str(self.failure_dir),
                return_detailed_log=True,
                stop_requested=self._budget_exhausted,
            )
            failures = self._failed_nodes(detailed_log)
            self.simulation_count += len(detailed_log)
            self.total_failures += failures
            _append_jsonl(self.evaluation_path, {
                "evaluation_index": self.evaluation_count,
                "executions": len(detailed_log),
                "failures": failures,
                "safety_score": safety_score,
                "task_score": task_score,
                "elapsed_seconds": round(self.clock() - self.start_time, 3),
                "detailed_log": detailed_log,
            })
            self.checkpoint_recorder.update(
                now=self.clock(),
                executions=self.simulation_count,
                failures=self.total_failures,
                extra={"evaluation_count": self.evaluation_count},
            )
            if getattr(self.fuzzer, "evaluation_stopped_by_budget", False):
                break

        self._write_final_metadata()

    def _write_final_metadata(self) -> None:
        end_time = self.clock()
        write_metadata(self.base_dir, {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "actual_duration_seconds": round(end_time - self.start_time, 2),
            "target_duration_seconds": self.time_budget_seconds,
            "total_simulations_executed": self.simulation_count,
            "total_failures_detected": self.total_failures,
            "failure_analysis": collect_failure_analysis(str(self.failure_dir)),
            "hourly_checkpoints": list(self.checkpoint_recorder.checkpoints),
            "evaluation_count": self.evaluation_count,
            "configuration": {
                "method": "local_only",
                "vla_model": self.model,
                "manifest_id": self.manifest_id,
                "static_scenario": self.selection["static_scenario"],
                "seed_scenario": self.selection["dynamic_scenario"],
                "instruction_source": "route_prior",
                "instruction": self.selection["instruction"],
                "random_seed": self.random_seed,
                "mutation_depth": self.mutation_depth,
                "operators": [operator.value for operator in self.operators],
                "oracle_checks": self.oracle_checks,
                "scenario_duration_frames": self.scenario_duration_frames,
                "success_distance": self.success_distance,
                "global_scenario_search": False,
                "local_language_fuzzing": True,
                "semantic_pruning": False,
                "llm_provider": self.api_provider,
                "llm_model": self.model_name,
            },
        })

    def cleanup(self) -> None:
        if self.fuzzer is not None:
            self.fuzzer.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VLAD-Fuzz Local-only RQ3 ablation.")
    parser.add_argument("--model", choices=["lmdrive", "simlingo", "bevdriver"], default="simlingo")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-id")
    parser.add_argument("--output-root", default="results/rq3/runs/local_only")
    parser.add_argument("--time-budget-seconds", type=float)
    parser.add_argument("--time-budget-minutes", type=float, default=240.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--mutation-depth", type=int, default=2)
    parser.add_argument("--operators", default="all")
    parser.add_argument("--scenario-duration-frames", type=int, default=500)
    parser.add_argument("--success-distance", type=float, default=5.0)
    parser.add_argument("--oracle-checks", default="default")
    parser.add_argument("--llm-provider", default="deepseek", choices=["gemini", "qwen", "deepseek"])
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.random_seed is not None:
        random.seed(args.random_seed)
    budget = args.time_budget_seconds
    if budget is None:
        budget = args.time_budget_minutes * 60.0
    tester = None
    try:
        tester = LocalOnlyTester(
            manifest=args.manifest,
            manifest_id=args.manifest_id,
            model=args.model,
            gpu_id=args.gpu_id,
            output_root=args.output_root,
            time_budget_seconds=budget,
            random_seed=args.random_seed,
            mutation_depth=args.mutation_depth,
            operators=parse_operator_list(args.operators),
            oracle_checks=parse_oracle_checks(args.oracle_checks),
            scenario_duration_frames=args.scenario_duration_frames,
            success_distance=args.success_distance,
            api_provider=args.llm_provider,
            model_name=args.llm_model,
        )
        tester.run()
        return 0
    except CarlaInfrastructureError as exc:
        print(f"CARLA infrastructure failure: {exc}", file=sys.stderr)
        return CARLA_INFRASTRUCTURE_EXIT_CODE
    finally:
        if tester is not None:
            tester.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

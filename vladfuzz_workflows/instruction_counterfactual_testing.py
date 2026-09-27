"""Budget-matched instruction counterfactual testing baseline."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
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
from vladfuzz_runtime.instruction_counterfactuals import build_counterfactual_batch
from vladfuzz_runtime.npc_count_range import sample_npc_count, validate_npc_count_range
from vladfuzz_runtime.oracle import parse_oracle_checks
from vladfuzz_runtime.run_metadata import create_run_dir, write_metadata


def _stable_seed(*parts) -> int:
    material = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class InstructionCounterfactualTester:
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
        min_npc_count: int = 0,
        max_npc_count: int = 3,
        oracle_checks: Optional[Dict[str, bool]] = None,
        scenario_duration_frames: int = 500,
        success_distance: float = 5.0,
        k: int = 8,
        clock: Callable[[], float] = time.time,
        fuzzer_factory=None,
        scenario_sampler=None,
        checkpoint_interval_seconds: float = 3600.0,
    ):
        if time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive")
        if k <= 0:
            raise ValueError("k must be positive")

        self.selection = apply_manifest_selection(
            manifest_path=manifest,
            manifest_id=manifest_id,
            instruction_source="basic",
            static_scenario=None,
            dynamic_scenario=None,
            instruction=None,
        )
        self.manifest_id = self.selection["manifest_id"]
        self.source_instruction = self.selection["instruction"]
        self.model = model
        self.gpu_id = gpu_id
        self.time_budget_seconds = float(time_budget_seconds)
        self.random_seed = int(random_seed)
        self.min_npc_count, self.max_npc_count = validate_npc_count_range(
            min_npc_count,
            max_npc_count,
        )
        self.oracle_checks = oracle_checks
        self.scenario_duration_frames = int(scenario_duration_frames)
        self.success_distance = float(success_distance)
        self.k = int(k)
        self.clock = clock
        self.scenario_sampler = scenario_sampler or self._sample_scenario
        self.checkpoint_interval_seconds = checkpoint_interval_seconds

        self.base_dir = create_run_dir(
            output_root,
            "instruction_counterfactual",
            model,
            self.manifest_id,
            self.random_seed,
        )
        self.base_path = Path(self.base_dir)
        self.failure_dir = self.base_path / "failures"
        self.scenario_dir = self.base_path / "scenario_batches"
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        self.scenario_dir.mkdir(parents=True, exist_ok=True)
        self.generated_path = self.base_path / "generated_instructions.jsonl"
        self.execution_path = self.base_path / "execution_results.jsonl"
        self._record_run_dir()

        if fuzzer_factory is None:
            from vladfuzz_workflows.local_fuzzer import LocalFuzzer

            fuzzer_factory = LocalFuzzer
        self.fuzzer = fuzzer_factory(
            static_scenario=self.selection["static_scenario"],
            mutation_depth=0,
            success_distance=self.success_distance,
            vla_model=self.model,
            gpu_id=self.gpu_id,
            verbose=False,
            oracle_checks=self.oracle_checks,
            scenario_duration_frames=self.scenario_duration_frames,
        )

        # Model initialization is excluded from the method's testing budget.
        self.start_time = self.clock()
        self.checkpoint_recorder = HourlyCheckpointRecorder(
            output_dir=self.base_dir,
            start_time=self.start_time,
            interval_seconds=self.checkpoint_interval_seconds,
            method="instruction_counterfactual",
            model=self.model,
            manifest_id=self.manifest_id,
        )
        self.simulation_count = 0
        self.total_failures = 0
        self.scenario_batches_sampled = 0
        self.family_execution_counts = Counter()
        self.family_failure_counts = Counter()
        self.template_usage_counts = Counter()
        self.executed_instructions = set()

    def _record_run_dir(self) -> None:
        marker = os.environ.get("VLADFUZZ_RUN_DIR_FILE")
        if not marker:
            return
        marker_path = Path(marker)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        with marker_path.open("a", encoding="utf-8") as file:
            file.write(self.base_dir + "\n")

    def _sample_scenario(self, _tester, scenario_index: int) -> Dict:
        scenario_seed = _stable_seed(self.random_seed, self.manifest_id, scenario_index)
        previous_state = random.getstate()
        random.seed(scenario_seed)
        try:
            npc_count = sample_npc_count(self.min_npc_count, self.max_npc_count)
            return self.fuzzer.scenario_manager.generate_scenario_from_seed(
                seed_scenario_path=self.selection["dynamic_scenario"],
                npc_count=npc_count,
                min_speed_diff_perc=-50.0,
                max_speed_diff_perc=50.0,
                include_environment=True,
                return_dict=True,
            )
        finally:
            random.setstate(previous_state)

    def _budget_exhausted(self) -> bool:
        return (self.clock() - self.start_time) >= self.time_budget_seconds

    @staticmethod
    def _failed_nodes(detailed_log) -> int:
        return sum(
            1
            for node in detailed_log
            if not node.get("valid", node.get("semantically_correct", True))
        )

    def _write_generated_batch(self, scenario_index, variants) -> None:
        for variant_index, variant in enumerate(variants, start=1):
            _append_jsonl(
                self.generated_path,
                {
                    "scenario_index": scenario_index,
                    "variant_index": variant_index,
                    "family": variant.family.value,
                    "family_index": variant.family_index,
                    "template_id": variant.template_id,
                    "source_instruction": self.source_instruction,
                    "instruction": variant.instruction,
                    "maneuvers": [maneuver.value for maneuver in variant.maneuvers],
                },
            )

    def run(self) -> None:
        scenario_index = 0
        while True:
            if self.simulation_count and self._budget_exhausted():
                break

            scenario = self.scenario_sampler(self, scenario_index)
            self.scenario_batches_sampled += 1
            scenario_path = self.scenario_dir / f"scenario_{scenario_index:05d}.json"
            scenario_path.write_text(
                json.dumps(scenario, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            variants = build_counterfactual_batch(
                self.source_instruction,
                k=self.k,
                random_seed=self.random_seed,
                scenario_index=scenario_index,
            )
            self._write_generated_batch(scenario_index, variants)

            for variant_index, variant in enumerate(variants, start=1):
                if self.simulation_count and self._budget_exhausted():
                    self._write_final_metadata()
                    return

                safety_score, task_score, detailed_log = self.fuzzer.evaluate(
                    dynamic_scenario=scenario,
                    language_instruction=variant.instruction,
                    save_failures=True,
                    failure_dir=str(self.failure_dir),
                    return_detailed_log=True,
                )
                failures = self._failed_nodes(detailed_log)
                self.simulation_count += len(detailed_log)
                self.total_failures += failures
                self.family_execution_counts[variant.family.value] += len(detailed_log)
                self.family_failure_counts[variant.family.value] += failures
                self.template_usage_counts[variant.template_id] += len(detailed_log)
                self.executed_instructions.add(variant.instruction)

                _append_jsonl(
                    self.execution_path,
                    {
                        "execution_index": self.simulation_count,
                        "scenario_index": scenario_index,
                        "variant_index": variant_index,
                        "family": variant.family.value,
                        "family_index": variant.family_index,
                        "template_id": variant.template_id,
                        "instruction": variant.instruction,
                        "failed": failures > 0,
                        "failure_count": failures,
                        "safety_score": safety_score,
                        "task_score": task_score,
                        "elapsed_seconds": round(self.clock() - self.start_time, 3),
                        "detailed_log": detailed_log,
                    },
                )
                self.checkpoint_recorder.update(
                    now=self.clock(),
                    executions=self.simulation_count,
                    failures=self.total_failures,
                    extra={
                        "scenario_index": scenario_index,
                        "family": variant.family.value,
                    },
                )

            scenario_index += 1

        self._write_final_metadata()

    def _write_final_metadata(self) -> None:
        end_time = self.clock()
        metadata = {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "actual_duration_seconds": round(end_time - self.start_time, 2),
            "target_duration_seconds": self.time_budget_seconds,
            "total_simulations_executed": self.simulation_count,
            "total_failures_detected": self.total_failures,
            "failure_analysis": collect_failure_analysis(str(self.failure_dir)),
            "hourly_checkpoints": list(self.checkpoint_recorder.checkpoints),
            "scenario_batches_sampled": self.scenario_batches_sampled,
            "unique_instruction_count": len(self.executed_instructions),
            "family_execution_counts": dict(self.family_execution_counts),
            "family_failure_counts": dict(self.family_failure_counts),
            "template_usage_counts": dict(self.template_usage_counts),
            "configuration": {
                "method": "instruction_counterfactual",
                "vla_model": self.model,
                "manifest_id": self.manifest_id,
                "static_scenario": self.selection["static_scenario"],
                "seed_scenario": self.selection["dynamic_scenario"],
                "instruction_source": "basic",
                "instruction": self.source_instruction,
                "random_seed": self.random_seed,
                "k_per_family": self.k,
                "variant_families": ["paraphrase", "ambiguity", "noise"],
                "min_npc_count": self.min_npc_count,
                "max_npc_count": self.max_npc_count,
                "scenario_duration_frames": self.scenario_duration_frames,
                "success_distance": self.success_distance,
                "oracle_checks": self.oracle_checks,
            },
        }
        write_metadata(self.base_dir, metadata)

    def cleanup(self) -> None:
        if self.fuzzer is not None:
            self.fuzzer.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the budget-matched instruction counterfactual testing baseline."
    )
    parser.add_argument("--model", choices=["lmdrive", "simlingo", "bevdriver"], default="simlingo")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-id")
    parser.add_argument("--output-root", default="results/rq2/runs/instruction_counterfactual")
    parser.add_argument("--time-budget-seconds", type=float)
    parser.add_argument("--time-budget-minutes", type=float, default=240.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--min-npc-count", type=int, default=0)
    parser.add_argument("--max-npc-count", type=int, default=3)
    parser.add_argument("--scenario-duration-frames", type=int, default=500)
    parser.add_argument("--success-distance", type=float, default=5.0)
    parser.add_argument("--oracle-checks", default="default")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    budget = (
        args.time_budget_seconds
        if args.time_budget_seconds is not None
        else args.time_budget_minutes * 60.0
    )
    tester = None
    try:
        tester = InstructionCounterfactualTester(
            manifest=args.manifest,
            manifest_id=args.manifest_id,
            model=args.model,
            gpu_id=args.gpu_id,
            output_root=args.output_root,
            time_budget_seconds=budget,
            random_seed=args.random_seed,
            min_npc_count=args.min_npc_count,
            max_npc_count=args.max_npc_count,
            oracle_checks=parse_oracle_checks(args.oracle_checks),
            scenario_duration_frames=args.scenario_duration_frames,
            success_distance=args.success_distance,
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

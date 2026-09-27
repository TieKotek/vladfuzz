import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from vladfuzz_runtime.infrastructure import CarlaInfrastructureError


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeScenarioManager:
    def __init__(self):
        self.sampled = []

    def generate_scenario_from_seed(self, **kwargs):
        scenario = {"scenario_id": f"generated-{len(self.sampled)}"}
        self.sampled.append((kwargs, scenario))
        return scenario


class FakeFuzzer:
    def __init__(self, clock, seconds_per_execution=1.0, fail_every=None, infrastructure_failure=False):
        self.clock = clock
        self.seconds_per_execution = seconds_per_execution
        self.fail_every = fail_every
        self.infrastructure_failure = infrastructure_failure
        self.scenario_manager = FakeScenarioManager()
        self.calls = []
        self.cleanup_called = False

    def evaluate(self, dynamic_scenario, language_instruction, **kwargs):
        if self.infrastructure_failure:
            raise CarlaInfrastructureError("simulator unavailable")
        self.calls.append((dynamic_scenario["scenario_id"], language_instruction))
        self.clock.advance(self.seconds_per_execution)
        failed = bool(self.fail_every and len(self.calls) % self.fail_every == 0)
        log = [{
            "semantically_correct": not failed,
            "valid": not failed,
            "failure_reason": "collision" if failed else None,
            "execution_result": {"completed": not failed, "collision": failed},
        }]
        return 5.0, 0.5, log

    def cleanup(self):
        self.cleanup_called = True


class InstructionCounterfactualTestingTest(unittest.TestCase):
    def _manifest(self, root):
        path = root / "manifest.jsonl"
        path.write_text(
            json.dumps({
                "id": "town01/seed_1",
                "static_scenario": "static.json",
                "seed_scenario": "seed.json",
                "basic_instruction": "Follow current lane for a while then turn left at intersection.",
            }) + "\n",
            encoding="utf-8",
        )
        return path

    def _make_tester(
        self,
        root,
        *,
        budget,
        seconds_per_execution=1.0,
        fail_every=None,
        infrastructure_failure=False,
        checkpoint_interval=3600.0,
    ):
        from vladfuzz_workflows.instruction_counterfactual_testing import (
            InstructionCounterfactualTester,
        )

        clock = FakeClock()
        fake = FakeFuzzer(
            clock,
            seconds_per_execution=seconds_per_execution,
            fail_every=fail_every,
            infrastructure_failure=infrastructure_failure,
        )
        sampled = []

        def scenario_sampler(tester, scenario_index):
            scenario = {"scenario_id": f"scenario-{scenario_index}"}
            sampled.append(scenario)
            return scenario

        tester = InstructionCounterfactualTester(
            manifest=str(self._manifest(root)),
            manifest_id="town01/seed_1",
            model="simlingo",
            gpu_id=0,
            output_root=str(root / "results"),
            time_budget_seconds=budget,
            random_seed=7,
            min_npc_count=0,
            max_npc_count=3,
            clock=clock,
            fuzzer_factory=lambda **kwargs: fake,
            scenario_sampler=scenario_sampler,
            checkpoint_interval_seconds=checkpoint_interval,
        )
        return tester, fake, clock, sampled

    def test_reuses_scenario_for_24_variants_then_samples_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            tester, fake, _, sampled = self._make_tester(Path(tmp), budget=25.0)

            tester.run()

            self.assertEqual(len(fake.calls), 25)
            self.assertEqual([scenario_id for scenario_id, _ in fake.calls[:24]], ["scenario-0"] * 24)
            self.assertEqual(fake.calls[24][0], "scenario-1")
            self.assertEqual(len(sampled), 2)
            self.assertEqual(tester.simulation_count, 25)

    def test_stops_after_current_simulation_and_keeps_partial_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tester, fake, clock, sampled = self._make_tester(
                Path(tmp),
                budget=10.0,
                seconds_per_execution=3.0,
                fail_every=2,
                checkpoint_interval=5.0,
            )

            tester.run()

            self.assertEqual(len(fake.calls), 4)
            self.assertGreaterEqual(clock.now - tester.start_time, 10.0)
            self.assertEqual(len(sampled), 1)
            execution_rows = [
                json.loads(line)
                for line in (Path(tester.base_dir) / "execution_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            generated_rows = [
                json.loads(line)
                for line in (Path(tester.base_dir) / "generated_instructions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            metadata = json.loads((Path(tester.base_dir) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(execution_rows), 4)
            self.assertEqual(len(generated_rows), 24)
            self.assertEqual(metadata["total_simulations_executed"], 4)
            self.assertEqual(metadata["total_failures_detected"], 2)
            self.assertEqual(metadata["scenario_batches_sampled"], 1)
            self.assertEqual(sum(metadata["family_execution_counts"].values()), 4)
            self.assertEqual(len(metadata["hourly_checkpoints"]), 2)

    def test_infrastructure_failure_propagates_without_valid_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tester, fake, _, _ = self._make_tester(
                Path(tmp),
                budget=10.0,
                infrastructure_failure=True,
            )

            with self.assertRaises(CarlaInfrastructureError):
                tester.run()

            self.assertFalse((Path(tester.base_dir) / "metadata.json").exists())
            self.assertEqual(tester.simulation_count, 0)
            self.assertEqual(fake.calls, [])


    def test_single_seed_script_forwards_budget_and_basic_manifest_selection(self):
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "run_instruction_counterfactual.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "argv.json"
            fake_python = root / "fake-python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "PROJECT_ROOT": str(project_root),
                "PYTHON_BIN": str(fake_python),
                "CAPTURE": str(capture),
                "MANIFEST": str(root / "manifest.jsonl"),
                "MANIFEST_ID": "town01/seed_1",
                "TIME_BUDGET_MINUTES": "17",
                "MIN_NPC_COUNT": "0",
                "MAX_NPC_COUNT": "3",
            })

            subprocess.run(["bash", str(script)], cwd=project_root, env=env, check=True)

            argv = json.loads(capture.read_text(encoding="utf-8"))
            self.assertEqual(
                argv[:2],
                ["-m", "vladfuzz_workflows.instruction_counterfactual_testing"],
            )
            self.assertEqual(argv[argv.index("--time-budget-minutes") + 1], "17")
            self.assertEqual(argv[argv.index("--min-npc-count") + 1], "0")
            self.assertEqual(argv[argv.index("--max-npc-count") + 1], "3")
            self.assertNotIn("--instruction-source", argv)


if __name__ == "__main__":
    unittest.main()

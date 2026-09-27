import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vladfuzz_runtime.infrastructure import CarlaInfrastructureError
from vladfuzz_workflows.local_fuzzer import LocalFuzzer


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeFuzzer:
    def __init__(self, clock, seconds_per_tree=3.0, fail_every=2, infrastructure_failure=False):
        self.clock = clock
        self.seconds_per_tree = seconds_per_tree
        self.fail_every = fail_every
        self.infrastructure_failure = infrastructure_failure
        self.calls = []
        self.cleanup_called = False

    def evaluate(self, dynamic_scenario, language_instruction, **kwargs):
        if self.infrastructure_failure:
            raise CarlaInfrastructureError("simulator unavailable")
        self.calls.append((dynamic_scenario, language_instruction, kwargs))
        self.clock.advance(self.seconds_per_tree)
        failed = bool(self.fail_every and len(self.calls) % self.fail_every == 0)
        return 5.0, 0.5, [{
            "valid": not failed,
            "semantically_correct": not failed,
            "failure_reason": "collision" if failed else None,
            "execution_result": {"collision": failed, "completed": not failed},
        }]

    def cleanup(self):
        self.cleanup_called = True


class RQ3WorkflowTests(unittest.TestCase):
    def test_optimizer_ablation_flags_reflect_mutation_depth(self):
        from vladfuzz_workflows.nsga_optimization import _ablation_flags

        self.assertEqual(_ablation_flags(0), {
            "global_scenario_search": True,
            "local_language_fuzzing": False,
        })
        self.assertEqual(_ablation_flags(2), {
            "global_scenario_search": True,
            "local_language_fuzzing": True,
        })

    def _manifest(self, root):
        path = root / "manifest.jsonl"
        path.write_text(json.dumps({
            "id": "town01/seed_1",
            "static_scenario": "static.json",
            "seed_scenario": "seed.json",
            "route_prior_instruction": "Continue and turn left.",
        }) + "\n", encoding="utf-8")
        return path

    def test_depth_zero_does_not_initialize_language_mutation_engine(self):
        scenario = Mock()
        scenario.load_static_scenario.return_value = True
        spec = SimpleNamespace(name="simlingo", config_path=Path("config"), hydra_config_path=None)
        with patch("vladfuzz_workflows.local_fuzzer.bootstrap_model_environment", return_value=spec), \
             patch("vladfuzz_workflows.local_fuzzer.CarlaScenario", return_value=scenario), \
             patch("vladfuzz_workflows.local_fuzzer.MutationOperatorEngine") as engine:
            LocalFuzzer("static.json", mutation_depth=0, vla_model="simlingo")
        engine.assert_not_called()

    def test_local_only_budget_includes_backend_initialization_like_full(self):
        from vladfuzz_workflows.local_only_testing import LocalOnlyTester

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clock = FakeClock()
            fake = FakeFuzzer(clock, seconds_per_tree=3.0, fail_every=None)

            def factory(**kwargs):
                clock.advance(2.0)
                return fake

            tester = LocalOnlyTester(
                manifest=str(self._manifest(root)),
                manifest_id="town01/seed_1",
                model="simlingo",
                gpu_id=0,
                output_root=str(root / "results"),
                time_budget_seconds=10.0,
                fuzzer_factory=factory,
                clock=clock,
            )
            tester.run()
            self.assertEqual(len(fake.calls), 3)
            self.assertEqual(clock.now - tester.start_time, 11.0)

    def test_local_only_reuses_fixed_scenario_until_budget_and_writes_metadata(self):
        from vladfuzz_workflows.local_only_testing import LocalOnlyTester

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clock = FakeClock()
            fake = FakeFuzzer(clock)
            captured = {}

            def factory(**kwargs):
                captured.update(kwargs)
                return fake

            tester = LocalOnlyTester(
                manifest=str(self._manifest(root)),
                manifest_id="town01/seed_1",
                model="simlingo",
                gpu_id=0,
                output_root=str(root / "results"),
                time_budget_seconds=10.0,
                random_seed=7,
                mutation_depth=2,
                clock=clock,
                fuzzer_factory=factory,
                checkpoint_interval_seconds=5.0,
            )
            tester.run()

            self.assertEqual(len(fake.calls), 4)
            self.assertEqual({call[0] for call in fake.calls}, {"seed.json"})
            self.assertEqual({call[1] for call in fake.calls}, {"Continue and turn left."})
            self.assertTrue(all(call[2]["stop_requested"] == tester._budget_exhausted for call in fake.calls))
            self.assertFalse(captured["semantic_pruning"])
            self.assertEqual(captured["mutation_depth"], 2)
            metadata = json.loads((Path(tester.base_dir) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["configuration"]["method"], "local_only")
            self.assertFalse(metadata["configuration"]["global_scenario_search"])
            self.assertTrue(metadata["configuration"]["local_language_fuzzing"])
            self.assertEqual(metadata["total_simulations_executed"], 4)
            self.assertEqual(metadata["total_failures_detected"], 2)
            self.assertEqual(len(metadata["hourly_checkpoints"]), 2)
            rows = (Path(tester.base_dir) / "evaluation_results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 4)

    def test_local_only_propagates_infrastructure_failure_without_valid_metadata(self):
        from vladfuzz_workflows.local_only_testing import LocalOnlyTester

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clock = FakeClock()
            fake = FakeFuzzer(clock, infrastructure_failure=True)
            tester = LocalOnlyTester(
                manifest=str(self._manifest(root)),
                manifest_id="town01/seed_1",
                model="simlingo",
                gpu_id=0,
                output_root=str(root / "results"),
                time_budget_seconds=10.0,
                fuzzer_factory=lambda **kwargs: fake,
                clock=clock,
            )
            with self.assertRaises(CarlaInfrastructureError):
                tester.run()
            self.assertFalse((Path(tester.base_dir) / "metadata.json").exists())


    def test_random_scenario_local_flags_disable_evolutionary_search(self):
        from vladfuzz_workflows.nsga_optimization import _ablation_flags

        self.assertEqual(
            _ablation_flags(2, method="random_scenario_local"),
            {
                "global_scenario_search": False,
                "local_language_fuzzing": True,
                "scenario_sampling": "independent_random",
            },
        )

    def test_random_scenario_local_streams_initial_individuals_until_budget(self):
        from vladfuzz_workflows.nsga_optimization import ScenarioOptimizer, creator

        optimizer = ScenarioOptimizer.__new__(ScenarioOptimizer)
        created = []
        evaluated = []

        def make_individual():
            individual = creator.Individual({"scenario_index": len(created)})
            created.append(individual)
            return individual

        def evaluate(individual):
            evaluated.append(individual)
            return 5.0, 0.5

        optimizer.toolbox = SimpleNamespace(
            individual=make_individual,
            evaluate=evaluate,
            population=Mock(side_effect=AssertionError("bounded population must not be built")),
            select=Mock(side_effect=AssertionError("NSGA-II selection must not run")),
        )
        optimizer._budget_reached = lambda: len(evaluated) >= 3
        optimizer._budget_status = Mock(return_value="time budget reached")
        optimizer.save_evaluated_individual_log = Mock()
        optimizer.save_generation_metadata = Mock()
        optimizer.save_final_metadata = Mock()

        population = optimizer._run_initial_population_until_budget()

        self.assertEqual(len(created), 3)
        self.assertEqual(population, created)
        self.assertTrue(all(individual.fitness.valid for individual in population))
        self.assertEqual(
            [call.args[1] for call in optimizer.save_evaluated_individual_log.call_args_list],
            ["ind_0", "ind_1", "ind_2"],
        )
        optimizer.save_generation_metadata.assert_called_once()
        metadata_call = optimizer.save_generation_metadata.call_args
        self.assertEqual(metadata_call.args[0], 0)
        self.assertEqual(metadata_call.args[1], population)
        self.assertEqual(metadata_call.kwargs["generation_status"], "budget_exhausted")
        optimizer.save_final_metadata.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

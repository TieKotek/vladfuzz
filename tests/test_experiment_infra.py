import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from vladfuzz_runtime.mutation_operators import MutationOperator, MutationOperatorEngine, parse_operator_list
from vladfuzz_runtime.experiment_commands import build_experiment_commands
from vladfuzz_runtime.failure_analysis import categorize_failure_reason
from vladfuzz_runtime.run_metadata import create_run_dir
from vladfuzz_workflows.local_fuzzer import FuzzingNode, LocalFuzzer
from vladfuzz_workflows.nsga_optimization import ScenarioOptimizer, creator
from vladfuzz_workflows.ads_baseline_testing import (
    compute_av_fuzzer_risk,
    compute_drive_fuzz_risk,
)


class FailureAnalysisTests(unittest.TestCase):
    def test_categorizes_known_failure_reasons(self):
        cases = {
            "Collision detected": "Collision",
            "Scenario not completed (Target not reached)": "Timeout/Stuck",
            "target_not_reached": "Timeout/Stuck",
            "timeout": "Timeout/Stuck",
            "out_of_bounds": "Out of Bounds",
            "Max speed 48.10 > Limit 40.00 * 1.1": "Speed Limit Exceeded",
            "Min distance 4.2m < Target 8m": "Maintain Distance Failed",
            "Target vehicle at spawn 12 not found": "Target Vehicle Missing",
            "CARLA execution error: timeout": "Execution Error",
            "Unhandled exception in runner": "Unhandled Exception",
        }

        for reason, expected in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(categorize_failure_reason(reason), expected)


class OperatorParsingTests(unittest.TestCase):
    def test_parse_operator_list_accepts_comma_separated_names(self):
        operators = parse_operator_list("paraphrasing, speed_control_insertion")

        self.assertEqual(
            operators,
            [MutationOperator.PARAPHRASING, MutationOperator.SPEED_CONTROL_INSERTION],
        )

    def test_parse_operator_list_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            parse_operator_list("paraphrasing,unknown_operator")


class MutationOperatorEngineTests(unittest.TestCase):
    def test_speed_limit_range_defaults_to_moderate_pressure(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=False), patch(
            "openai.OpenAI"
        ):
            engine = MutationOperatorEngine(api_provider="deepseek")

        self.assertEqual(engine.speed_limit_range, (0.8, 1.0))

    def test_deepseek_provider_disables_thinking_mode(self):
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Drive carefully and turn left.")
                )
            ]
        )
        client = Mock()
        client.chat.completions.create.return_value = completion

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=False), patch(
            "openai.OpenAI", return_value=client
        ) as openai_cls:
            engine = MutationOperatorEngine(api_provider="deepseek")
            result = engine._call_llm_api("mutate this instruction")

        self.assertEqual(engine.model_name, "deepseek-v4-flash")
        openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )
        self.assertEqual(result, "Drive carefully and turn left.")
        client.chat.completions.create.assert_called_once()
        _, kwargs = client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})


class LocalFuzzerBudgetTests(unittest.TestCase):
    def test_task_score_uses_old_average_over_all_executed_nodes(self):
        fuzzer = LocalFuzzer.__new__(LocalFuzzer)
        fuzzer.nodes = [
            {
                "semantically_correct": True,
                "execution_result": {
                    "ettc_stats": {"avg_ettc": 8.0},
                    "path_deviation_stats": {"path_tracking_quality": 0.9},
                },
            },
            {
                "semantically_correct": False,
                "execution_result": {
                    "ettc_stats": {"avg_ettc": 5.0},
                    "path_deviation_stats": {"path_tracking_quality": 0.7},
                },
            },
            {
                "semantically_correct": True,
                "execution_result": {
                    "ettc_stats": {"avg_ettc": 6.0},
                    "path_deviation_stats": {"path_tracking_quality": 0.6},
                },
            },
        ]

        safety_score, task_score = fuzzer._calculate_fitness_metrics()

        self.assertEqual(safety_score, 5.0)
        self.assertAlmostEqual(task_score, (0.9 + 0.6) / 3)

    def test_budget_stop_prevents_expanding_more_mutation_nodes(self):
        fuzzer = LocalFuzzer.__new__(LocalFuzzer)
        fuzzer.operators = [MutationOperator.PARAPHRASING]
        fuzzer.mutation_depth = 2
        fuzzer.semantic_pruning = False
        fuzzer.nodes = []
        fuzzer.save_failures = False
        fuzzer._execute_scenario = Mock(
            return_value={
                "completed": True,
                "collision": False,
                "ettc_stats": {"ettc_values": [10.0]},
                "path_deviation_stats": {"path_tracking_quality": 0.9},
            }
        )
        fuzzer._evaluate_node_validity = Mock(return_value=(True, None))
        fuzzer._save_failed_case = Mock()
        fuzzer.mutation_engine = Mock()
        fuzzer.stop_requested = Mock(return_value=True)
        fuzzer.evaluation_stopped_by_budget = False

        root = FuzzingNode(instruction="Turn left.", parent=None)
        fuzzer._dfs(root)

        self.assertTrue(fuzzer.evaluation_stopped_by_budget)
        self.assertEqual(len(fuzzer.nodes), 1)
        fuzzer.mutation_engine.apply_mutation.assert_not_called()


class NSGAIncrementalLoggingTests(unittest.TestCase):
    def test_route_xy_points_reads_nested_waypoint_locations(self):
        optimizer = ScenarioOptimizer.__new__(ScenarioOptimizer)
        optimizer.seed_data = {
            "route_info": {
                "route_waypoints": [
                    {"location": {"x": 1.0, "y": 2.0}},
                    {"location": {"x": 3.5, "y": 4.5}},
                ]
            }
        }

        self.assertEqual(optimizer._route_xy_points(), [(1.0, 2.0), (3.5, 4.5)])

    def test_saves_evaluated_individual_immediately(self):
        with TemporaryDirectory() as tmp_dir:
            optimizer = ScenarioOptimizer.__new__(ScenarioOptimizer)
            optimizer.enable_logging = True
            optimizer.log_dir = str(Path(tmp_dir) / "logs")
            optimizer._budget_reached = Mock(return_value=False)

            individual = creator.Individual(
                {
                    "weather": "ClearNoon",
                    "npc_vehicle_count": 1,
                    "__evaluation_log__": [
                        {
                            "node_id": "Root",
                            "valid": True,
                            "semantically_correct": True,
                            "mutation_success": True,
                        }
                    ],
                }
            )
            individual.fitness.values = (1.0, 0.5)

            optimizer.save_evaluated_individual_log(0, "ind_0", individual, phase="initial")

            ind_dir = Path(tmp_dir) / "logs" / "gen_0" / "ind_0"
            self.assertTrue((ind_dir / "config.json").exists())
            self.assertTrue((ind_dir / "result.json").exists())
            self.assertTrue((Path(tmp_dir) / "logs" / "gen_0" / "metadata.json").exists())


class ExperimentCommandTests(unittest.TestCase):
    def test_builds_rq2_commands_for_all_methods_and_repeats(self):
        commands = build_experiment_commands(
            manifest="paper/experiment_manifest.jsonl",
            manifest_ids=["scenario_a/seed_1"],
            models=["simlingo", "lmdrive"],
            methods=["random", "av_fuzzer", "drive_fuzz", "instruction_counterfactual", "vlad_fuzz"],
            repeats=2,
            instruction_source="route_prior",
            simulation_budget=30,
            mutation_depth=2,
            gpu_id=0,
        )

        self.assertEqual(len(commands), 20)
        random_commands = [cmd for cmd in commands if "random" in cmd]
        self.assertTrue(any("vladfuzz_workflows.baseline_testing" in cmd for cmd in random_commands))
        self.assertTrue(any("--standalone" in cmd for cmd in random_commands))
        av_fuzzer = [cmd for cmd in commands if "--method av_fuzzer" in cmd]
        self.assertTrue(all("vladfuzz_workflows.ads_baseline_testing" in cmd for cmd in av_fuzzer))
        self.assertTrue(all("--instruction-source basic" in cmd for cmd in av_fuzzer))
        drive_fuzz = [cmd for cmd in commands if "--method drive_fuzz" in cmd]
        self.assertTrue(all("vladfuzz_workflows.ads_baseline_testing" in cmd for cmd in drive_fuzz))
        instruction_counterfactual = [cmd for cmd in commands if "--method instruction_counterfactual" in cmd]
        self.assertTrue(all("vladfuzz_workflows.local_fuzzer" in cmd for cmd in instruction_counterfactual))
        vlad_fuzz = [cmd for cmd in commands if "--method vlad_fuzz" in cmd]
        self.assertTrue(all("vladfuzz_workflows.nsga_optimization" in cmd for cmd in vlad_fuzz))

    def test_commands_use_configured_output_root(self):
        commands = build_experiment_commands(
            manifest="paper/experiment_manifest.jsonl",
            manifest_ids=["scenario_a/seed_1"],
            models=["simlingo"],
            methods=["vlad_fuzz"],
            repeats=1,
            instruction_source="route_prior",
            simulation_budget=30,
            mutation_depth=2,
            gpu_id=0,
        )

        self.assertIn("--output-root results", commands[0])


class RunMetadataTests(unittest.TestCase):
    def test_create_run_dir_places_runs_under_model_directory(self):
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(create_run_dir(tmp_dir, "full", "simlingo", "scenario_a/seed_1", 7))

            self.assertEqual(run_dir.parent.name, "simlingo")
            self.assertEqual(run_dir.parent.parent, Path(tmp_dir))
            self.assertTrue(run_dir.name.startswith("full_scenario_a__seed_1_seed7_"))
            self.assertTrue(run_dir.exists())


class ADSBaselineRiskTests(unittest.TestCase):
    def test_av_fuzzer_risk_uses_brake_margin_and_average_ettc_not_min_spike(self):
        base = {
            "completed": True,
            "collision": False,
            "ettc_stats": {
                "avg_ettc": 12.0,
                "min_ettc": 0.1,
                "danger_frames": 1,
                "detection_radius": 10.0,
                "min_distance": 8.0,
                "min_brake_margin": 3.0,
            },
            "path_deviation_stats": {"path_tracking_quality": 0.95},
        }
        same_average_different_spike = {
            **base,
            "ettc_stats": {**base["ettc_stats"], "min_ettc": 9.0},
        }
        tighter_brake_margin = {
            **base,
            "ettc_stats": {**base["ettc_stats"], "min_brake_margin": -2.0},
        }

        self.assertAlmostEqual(
            compute_av_fuzzer_risk(base),
            compute_av_fuzzer_risk(same_average_different_spike),
        )
        self.assertGreater(
            compute_av_fuzzer_risk(tighter_brake_margin),
            compute_av_fuzzer_risk(base),
        )

    def test_drive_fuzz_risk_tracks_driving_quality_degradation(self):
        high_quality = {
            "completed": True,
            "collision": False,
            "ettc_stats": {
                "avg_ettc": 12.0,
                "danger_frames": 0,
                "min_distance": 10.0,
                "min_brake_margin": 4.0,
            },
            "path_deviation_stats": {"path_tracking_quality": 0.95},
        }
        low_quality = {
            **high_quality,
            "path_deviation_stats": {"path_tracking_quality": 0.4},
        }

        self.assertGreater(compute_drive_fuzz_risk(low_quality), compute_drive_fuzz_risk(high_quality))

    def test_drive_fuzz_risk_prefers_recorded_driving_quality_deductions(self):
        low_deduction = {
            "completed": True,
            "collision": False,
            "ettc_stats": {"min_distance": 10.0, "min_brake_margin": 4.0},
            "path_deviation_stats": {"path_tracking_quality": 0.95},
            "driving_quality_stats": {"driving_quality_deductions": 1.0},
        }
        high_deduction = {
            **low_deduction,
            "driving_quality_stats": {"driving_quality_deductions": 8.0},
        }

        self.assertGreater(compute_drive_fuzz_risk(high_deduction), compute_drive_fuzz_risk(low_deduction))


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scenario_construction.generate_instructions import OUTPUT_BY_MODE, _prompt_for_mode
from scenario_construction.instruction_generation import parse_instruction_response, structured_instruction_record
from scenario_construction.scenario_regions import load_region_specs, select_region_specs
from scenario_construction.seed_generator import (
    build_scenario_manager_kwargs,
    group_seed_candidates,
    route_signature,
    select_diverse_candidates,
    static_scenario_output_dir,
)


class ScenarioRegionTests(unittest.TestCase):
    def test_load_region_specs_from_jsonl(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "regions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "town03_t_junction",
                        "map_name": "Town03",
                        "center_x": -145,
                        "center_y": -3,
                        "extent_x": 20,
                        "extent_y": 40,
                        "scenario_type": "intersection",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            specs = load_region_specs(path)

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].id, "town03_t_junction")
        self.assertEqual(specs[0].map_name, "Town03")
        self.assertEqual(specs[0].vehicle_filter, "vehicle.tesla.model3")

    def test_select_region_specs_requires_known_id(self):
        specs = load_region_specs(Path("configs/scenario_regions.jsonl"))

        with self.assertRaises(ValueError):
            select_region_specs(specs, "missing_region", include_all=False)


class SeedGeneratorPathTests(unittest.TestCase):
    def test_static_scenario_output_dir_uses_json_stem(self):
        output_dir = static_scenario_output_dir(
            Path("static_scenarios/scenario_Town03_20251014_100608.json"),
            Path("seeds"),
        )

        self.assertEqual(output_dir, Path("seeds/scenario_Town03_20251014_100608"))

    def test_generate_seeds_script_defaults_to_no_npcs(self):
        script = Path("scripts/generate_seeds.sh").read_text(encoding="utf-8")

        self.assertIn('MIN_NPC_COUNT="${MIN_NPC_COUNT:-0}"', script)
        self.assertIn('MAX_NPC_COUNT="${MAX_NPC_COUNT:-0}"', script)

    def test_seed_artifact_names_do_not_include_timestamps(self):
        scenario_source = Path("scenario/carla_scenario.py").read_text(encoding="utf-8")
        camera_source = Path("scenario/camera.py").read_text(encoding="utf-8")

        self.assertIn('f"dynamic_scenario_{map_name}.json"', scenario_source)
        self.assertIn('"ego_camera_view.png"', camera_source)
        self.assertNotIn('f"dynamic_scenario_{map_name}_{timestamp}.json"', scenario_source)
        self.assertNotIn('f"ego_camera_view_{timestamp}.png"', camera_source)

    def test_scenario_manager_kwargs_skip_model_without_dynamic_filter(self):
        kwargs = build_scenario_manager_kwargs(
            host="localhost",
            tm_seed=0,
            timeout=20,
            frame_rate=20,
            gpu_id=0,
            model="simlingo",
            dynamic_filter=False,
        )

        self.assertNotIn("model_name", kwargs)
        self.assertNotIn("config_path", kwargs)
        self.assertNotIn("hydra_config_path", kwargs)

    def test_scenario_manager_kwargs_include_model_for_dynamic_filter(self):
        kwargs = build_scenario_manager_kwargs(
            host="localhost",
            tm_seed=0,
            timeout=20,
            frame_rate=20,
            gpu_id=0,
            model="simlingo",
            dynamic_filter=True,
        )

        self.assertEqual(kwargs["model_name"], "simlingo")
        self.assertIn("config_path", kwargs)

    def test_route_signature_compresses_repeated_maneuvers(self):
        signature = route_signature(
            {
                "maneuvers": [
                    "FOLLOW_LANE",
                    "FOLLOW_LANE",
                    "LEFT",
                    "LEFT",
                    "FOLLOW_LANE",
                ]
            }
        )

        self.assertEqual(signature, "FOLLOW_LANE>LEFT>FOLLOW_LANE")

    def test_select_diverse_candidates_takes_one_from_each_group(self):
        candidates = [
            {"candidate_id": "a1", "target_road_id": 1, "target_lane_id": -1, "route_info": {"maneuvers": ["FOLLOW_LANE", "LEFT"]}},
            {"candidate_id": "a2", "target_road_id": 1, "target_lane_id": -1, "route_info": {"maneuvers": ["FOLLOW_LANE", "LEFT"]}},
            {"candidate_id": "b1", "target_road_id": 2, "target_lane_id": -1, "route_info": {"maneuvers": ["FOLLOW_LANE", "RIGHT"]}},
        ]

        groups = group_seed_candidates(candidates)
        selected = select_diverse_candidates(candidates, max_seeds=10, candidates_per_group=1)

        self.assertEqual(len(groups), 2)
        self.assertEqual([candidate["candidate_id"] for candidate in selected], ["a1", "b1"])


class InstructionGenerationTests(unittest.TestCase):
    def test_output_file_names_are_explicit_about_prior_mode(self):
        self.assertEqual(OUTPUT_BY_MODE["route_prior"], "command_prior.json")
        self.assertEqual(OUTPUT_BY_MODE["no_prior"], "command_no_prior.json")

    def test_parse_instruction_response_accepts_json_instruction(self):
        response = '{"instruction": "Turn left at the next intersection."}'

        self.assertEqual(parse_instruction_response(response), "Turn left at the next intersection.")

    def test_parse_instruction_response_accepts_commands_list(self):
        response = '{"commands": ["Turn left at the next intersection.", "Take the next left."]}'

        self.assertEqual(parse_instruction_response(response), "Turn left at the next intersection.")

    def test_structured_instruction_record_preserves_command_candidates(self):
        record = structured_instruction_record(
            raw_response='{"commands": ["Go straight.", "Continue ahead."]}',
            mode="route_prior",
            generator_model="qwen3-vl-plus",
            prompt_path=Path("prompt.txt"),
        )

        self.assertEqual(record["instruction"], "Go straight.")
        self.assertEqual(record["commands"], ["Go straight.", "Continue ahead."])
        self.assertNotIn("validity_check", record)

    def test_structured_instruction_record_preserves_raw_response(self):
        record = structured_instruction_record(
            raw_response='{"instruction": "Go straight."}',
            mode="route_prior",
            generator_model="qwen3-vl-plus",
            prompt_path=Path("prompt.txt"),
        )

        self.assertEqual(record["instruction"], "Go straight.")
        self.assertEqual(record["mode"], "route_prior")
        self.assertEqual(record["generator_model"], "qwen3-vl-plus")
        self.assertEqual(record["raw_response"], '{"instruction": "Go straight."}')

    def test_prompt_for_mode_injects_command_count(self):
        with TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "prompt.txt"
            prompt_path.write_text("Generate {num_commands} commands.", encoding="utf-8")
            scenario_path = Path(tmp_dir) / "scenario.json"
            scenario_path.write_text(
                json.dumps({"route_info": {"route_description": ["Turn left"]}}),
                encoding="utf-8",
            )

            prompt = _prompt_for_mode("route_prior", prompt_path, scenario_path, num_commands=3)

        self.assertIn("Generate 3 commands.", prompt)
        self.assertIn('"Turn left"', prompt)


if __name__ == "__main__":
    unittest.main()

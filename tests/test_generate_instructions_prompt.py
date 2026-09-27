import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scenario_construction.generate_instructions import _prompt_for_mode


class GenerateInstructionsPromptTests(unittest.TestCase):
    def test_prompt_for_mode_includes_scene_context_hint(self):
        with TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "prompt.txt"
            prompt_path.write_text("Generate {num_commands} commands.", encoding="utf-8")
            scenario_path = Path(tmp_dir) / "town06_highway_exit" / "seed_1" / "scenario.json"
            scenario_path.parent.mkdir(parents=True)
            scenario_path.write_text(
                json.dumps(
                    {
                        "map_name": "Carla/Maps/Town06",
                        "route_info": {"route_description": ["Turn right"]},
                    }
                ),
                encoding="utf-8",
            )

            route_prompt = _prompt_for_mode("route_prior", prompt_path, scenario_path, num_commands=1)
            no_prior_prompt = _prompt_for_mode("no_prior", prompt_path, scenario_path, num_commands=1)

        self.assertIn("[SCENE CONTEXT]", route_prompt)
        self.assertIn("Scenario type hint: town06_highway_exit", route_prompt)
        self.assertIn("Map: Carla/Maps/Town06", route_prompt)
        self.assertIn("Do not mention the scenario type hint or map name", route_prompt)
        self.assertIn('"Turn right"', route_prompt)
        self.assertIn("[SCENE CONTEXT]", no_prior_prompt)
        self.assertIn("Scenario type hint: town06_highway_exit", no_prior_prompt)
        self.assertIn("Map: Carla/Maps/Town06", no_prior_prompt)
        self.assertNotIn('"Turn right"', no_prior_prompt)


if __name__ == "__main__":
    unittest.main()

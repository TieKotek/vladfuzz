import json
import tempfile
import unittest
from pathlib import Path

from vladfuzz_runtime.experiment_manifest import apply_manifest_selection


class ExperimentManifestInputTests(unittest.TestCase):
    def test_direct_mode_requires_all_inputs(self):
        with self.assertRaisesRegex(ValueError, "--static-scenario"):
            apply_manifest_selection(
                manifest_path=None,
                manifest_id=None,
                instruction_source="manual",
                static_scenario=None,
                dynamic_scenario=None,
                instruction=None,
            )

    def test_manifest_mode_resolves_paths_and_instruction(self):
        row = {
            "id": "task/seed_1",
            "static_scenario": "static.json",
            "seed_scenario": "dynamic.json",
            "route_prior_instruction": "Turn left.",
        }
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            selection = apply_manifest_selection(
                manifest_path=str(manifest),
                manifest_id="task/seed_1",
                instruction_source="route_prior",
                static_scenario=None,
                dynamic_scenario=None,
                instruction=None,
            )

        self.assertEqual(selection["static_scenario"], "static.json")
        self.assertEqual(selection["dynamic_scenario"], "dynamic.json")
        self.assertEqual(selection["instruction"], "Turn left.")


if __name__ == "__main__":
    unittest.main()

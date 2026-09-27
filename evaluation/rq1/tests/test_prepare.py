import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.rq1.prepare import prepare_round


class PrepareRoundTests(unittest.TestCase):
    def make_seed(self, root: Path, scenario: str, seed: str) -> Path:
        seed_dir = root / scenario / seed
        seed_dir.mkdir(parents=True)
        (seed_dir / "bev.png").write_bytes(b"bev")
        (seed_dir / "ego_camera_view.png").write_bytes(b"ego")
        (seed_dir / "dynamic_scenario.json").write_text(
            json.dumps({"route_info": {"route_description": ["Follow lane", "Turn left"], "basic_instruction": "Follow the lane and turn left."}}),
            encoding="utf-8",
        )
        prompt_prior = root.parent / "prompt_prior.txt"
        prompt_no_prior = root.parent / "prompt_no_prior.txt"
        prompt_prior.write_text("prior prompt", encoding="utf-8")
        prompt_no_prior.write_text("image prompt", encoding="utf-8")
        for mode, prompt, commands in (
            ("route_prior", prompt_prior, ["Prior one", "Prior two"]),
            ("no_prior", prompt_no_prior, ["Image one", "Image two"]),
        ):
            (seed_dir / ("command_prior.json" if mode == "route_prior" else "command_no_prior.json")).write_text(
                json.dumps({
                    "commands": commands,
                    "mode": mode,
                    "generator_model": "test-model",
                    "prompt_path": str(prompt),
                    "created_at": "2026-01-01T00:00:00",
                }),
                encoding="utf-8",
            )
        return seed_dir

    def test_prepare_creates_one_blind_package_and_private_provenance(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeds = root / "test_cases"
            self.make_seed(seeds, "town_a", "seed_1")
            output = root / "round"

            summary = prepare_round(
                seeds_root=seeds,
                output_root=output,
                overwrite=False,
                random_seed=7,
                max_seeds_per_static_scenario=3,
                instructions_per_seed=2,
            )

            self.assertEqual(summary.case_count, 4)
            self.assertEqual(summary.pair_count, 2)
            with (output / "private" / "mapping.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["mode"] for row in rows}, {"route_prior", "no_prior"})
            self.assertEqual({row["pair_id"] for row in rows}, {"town_a/seed_1/slot_1", "town_a/seed_1/slot_2"})
            self.assertTrue(all(row["source_sha256"] for row in rows))
            self.assertTrue(all(row["prompt_sha256"] for row in rows))
            self.assertEqual({row["generator_model"] for row in rows}, {"test-model"})

            package = output / "package"
            cases = sorted(path.name for path in (package / "cases").iterdir())
            self.assertEqual(len(cases), 4)
            self.assertFalse((output / "packages").exists())
            for case_id in cases:
                case = package / "cases" / case_id
                self.assertEqual((case / "image.png").read_bytes(), b"ego")
                self.assertEqual((case / "annotation.txt").read_text(encoding="utf-8"), "score: \nerror_type: \nnotes: \n")
                reference = (case / "reference.txt").read_text(encoding="utf-8")
                self.assertIn("Route description:", reference)
                self.assertIn("Follow lane", reference)
                self.assertNotIn("Basic instruction", reference)
                self.assertNotIn("Follow the lane and turn left.", reference)
                self.assertFalse((case / "mapping.csv").exists())

            readme = (package / "README.md").read_text(encoding="utf-8").lower()
            for forbidden in ("route_prior", "no_prior", "image-only", "hypothesis", "mapping.csv"):
                self.assertNotIn(forbidden, readme)
            self.assertIn("score", readme)
            self.assertIn("wrong_maneuver", readme)
            self.assertIn("规划路线", readme)
            self.assertIn("不是", readme)
            self.assertIn("措辞模板", readme)
            manifest = json.loads((output / "private" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["case_count"], 4)
            self.assertNotIn("annotator_ids", manifest)

    def test_require_both_drops_unpaired_extra_commands(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeds = root / "test_cases"
            seed = self.make_seed(seeds, "town_a", "seed_1")
            (seed / "command_no_prior.json").write_text(
                json.dumps({"commands": ["Only one"], "mode": "no_prior"}),
                encoding="utf-8",
            )
            output = root / "round"
            summary = prepare_round(
                seeds_root=seeds,
                output_root=output,
                require_both=True,
                instructions_per_seed=None,
            )
            self.assertEqual(summary.case_count, 2)
            self.assertEqual(summary.pair_count, 1)


if __name__ == "__main__":
    unittest.main()

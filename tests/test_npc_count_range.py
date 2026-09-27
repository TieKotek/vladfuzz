import unittest
from pathlib import Path
from unittest.mock import Mock

from vladfuzz_runtime.npc_count_range import sample_npc_count, validate_npc_count_range


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NpcCountRangeTests(unittest.TestCase):
    def test_validates_and_samples_inclusive_range(self):
        self.assertEqual(validate_npc_count_range(1, 3), (1, 3))
        rng = Mock()
        rng.randint.return_value = 2

        self.assertEqual(sample_npc_count(1, 3, rng=rng), 2)
        rng.randint.assert_called_once_with(1, 3)

    def test_rejects_invalid_ranges(self):
        for minimum, maximum in [(-1, 3), (4, 3)]:
            with self.subTest(minimum=minimum, maximum=maximum):
                with self.assertRaises(ValueError):
                    validate_npc_count_range(minimum, maximum)

    def test_vladfuzz_uses_range_for_initialization_and_evolution(self):
        source = (PROJECT_ROOT / "vladfuzz_workflows/nsga_optimization.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("sample_npc_count(self.min_npc_count, self.max_npc_count)", source)
        self.assertIn("len(npc_list) < self.max_npc_count", source)
        self.assertIn("if self.max_npc_count > 0", source)
        self.assertIn("len(npc_list) > self.min_npc_count", source)

    def test_random_and_vlad_scripts_expose_same_range(self):
        for name in (
            "run_vladfuzz.sh",
            "run_random.sh",
            "run_batch_vladfuzz.sh",
            "run_batch_random.sh",
        ):
            with self.subTest(script=name):
                source = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertIn('MIN_NPC_COUNT="${MIN_NPC_COUNT:-0}"', source)
                self.assertIn('MAX_NPC_COUNT="${MAX_NPC_COUNT:-3}"', source)


if __name__ == "__main__":
    unittest.main()

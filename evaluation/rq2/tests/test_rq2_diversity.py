import unittest
from pathlib import Path

import numpy as np

from evaluation.rq2.analyze_diversity import (
    FailureCase,
    analyze_task_stratified_diversity,
    drivefuzz_scene_vector,
    joint_distance_matrix,
    pairwise_diversity,
    vladfuzz_scene_vector,
)


class RQ2DiversityTests(unittest.TestCase):
    def test_vladfuzz_scene_uses_actor_position_relative_to_ego(self):
        static = {
            "scenario_center": {"x": 100.0, "y": 200.0},
            "spawn_points": [
                {"x": 0.0, "y": 0.0, "yaw": 0.0},
                {"x": 10.0, "y": 0.0, "yaw": 0.0},
            ],
        }
        scenario = {
            "ego_car": {"spawn_point_index": 0},
            "npc_vehicles": [
                {"spawn_point_index": 1, "speed_percentage_difference": 20.0}
            ],
            "weather": "ClearNoon",
        }

        vector = vladfuzz_scene_vector(
            scenario,
            static,
            weather_resolver=lambda _: [0.0] * 8,
        )

        self.assertEqual(vector["vehicle_count"], 1.0)
        self.assertEqual(vector["walker_count"], 0.0)
        self.assertAlmostEqual(vector["mean_forward_distance"], 10.0)
        self.assertAlmostEqual(vector["mean_abs_lateral_distance"], 0.0)

    def test_drivefuzz_scene_has_the_same_spatial_semantics(self):
        record = {
            "seed": {"sp_x": 100.0, "sp_y": 200.0, "yaw": 0.0},
            "actors": [
                {
                    "type": 0,
                    "nav_type": 1,
                    "sp_x": 110.0,
                    "sp_y": 200.0,
                    "speed": None,
                }
            ],
            "puddles": [],
            "weather": {
                "cloud": 0.0,
                "rain": 0.0,
                "puddle": 0.0,
                "wind": 0.0,
                "fog": 0.0,
                "wetness": 0.0,
                "angle": 0.0,
                "altitude": 0.0,
            },
        }

        vector = drivefuzz_scene_vector(record)

        self.assertEqual(vector["vehicle_count"], 1.0)
        self.assertEqual(vector["walker_count"], 0.0)
        self.assertAlmostEqual(vector["mean_forward_distance"], 10.0)
        self.assertAlmostEqual(vector["mean_abs_lateral_distance"], 0.0)

    def test_joint_distance_gives_each_modality_equal_weight(self):
        scene = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        instruction = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])

        distance = joint_distance_matrix(scene, instruction, scene_weight=0.5)

        self.assertAlmostEqual(distance[0, 1], 0.5)
        self.assertAlmostEqual(distance[0, 2], 0.5)
        self.assertAlmostEqual(distance[0, 0], 0.0)
        np.testing.assert_allclose(distance, distance.T)

    def test_pairwise_diversity_reports_scene_instruction_and_joint_components(self):
        scenes = np.array([[0.0, 0.0], [1.0, 1.0]])
        instructions = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = pairwise_diversity(scenes, instructions, scene_weight=0.5)

        self.assertAlmostEqual(result["scene"], 1.0)
        self.assertAlmostEqual(result["instruction"], 0.5)
        self.assertAlmostEqual(result["joint"], 0.75)

    def test_analysis_balances_methods_within_each_task(self):
        methods = ("VLAD-Fuzz", "DriveFuzz", "Instruction-CF")
        cases = []
        embeddings = []
        for task in ("task-a", "task-b"):
            for method in methods:
                count = 5 if method == "VLAD-Fuzz" else 3
                for index in range(count):
                    varied = method == "VLAD-Fuzz"
                    cases.append(
                        FailureCase(
                            "simlingo",
                            method,
                            task,
                            f"{method}-{index}",
                            {"f1": float(index if varied else 0), "f2": 0.0},
                            Path(f"{task}-{method}-{index}"),
                        )
                    )
                    embeddings.append([1.0, float(index + 1)] if varied else [1.0, 0.0])

        rows, comparisons, diagnostics = analyze_task_stratified_diversity(
            cases,
            np.asarray(embeddings),
            scene_feature_names=("f1", "f2"),
            models=("simlingo",),
            max_per_method_task=3,
            bootstrap_iterations=40,
            random_seed=7,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(row["comparable_tasks"] == 2 for row in rows))
        self.assertGreater(
            next(
                row["joint_diversity"] for row in rows if row["method"] == "VLAD-Fuzz"
            ),
            next(
                row["joint_diversity"] for row in rows if row["method"] == "DriveFuzz"
            ),
        )
        self.assertEqual(diagnostics["models"]["simlingo"]["excluded_tasks"], {})

    def test_analysis_rejects_models_without_a_comparable_task(self):
        methods = ("VLAD-Fuzz", "DriveFuzz", "Instruction-CF")
        cases = []
        embeddings = []
        for method in methods:
            count = 1 if method == "Instruction-CF" else 2
            for index in range(count):
                cases.append(
                    FailureCase(
                        "simlingo",
                        method,
                        "task-a",
                        "go",
                        {"f": float(index)},
                        Path("x"),
                    )
                )
                embeddings.append([1.0, 0.0])

        with self.assertRaisesRegex(ValueError, "no comparable tasks"):
            analyze_task_stratified_diversity(
                cases,
                np.asarray(embeddings),
                scene_feature_names=("f",),
                models=("simlingo",),
                bootstrap_iterations=10,
            )


if __name__ == "__main__":
    unittest.main()

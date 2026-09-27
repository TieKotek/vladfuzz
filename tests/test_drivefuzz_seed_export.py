import unittest

from vladfuzz_runtime.drivefuzz_seed_export import convert_dynamic_scenario_to_drivefuzz_seed


class DriveFuzzSeedExportTests(unittest.TestCase):
    def test_converts_vlad_seed_to_drivefuzz_pose_seed(self):
        static_data = {
            "map_name": "Town03",
            "scenario_center": {"x": 100.0, "y": 200.0},
            "spawn_points": [
                {"x": 1.0, "y": 2.0, "z": 0.3, "yaw": 90.0},
                {"x": 20.0, "y": 30.0, "z": 0.3, "yaw": 0.0},
            ],
        }
        dynamic_data = {
            "map_name": "Town03",
            "ego_car": {"spawn_point_index": 0},
            "target_point": {"x": 50.0, "y": 60.0, "z": 0.3},
            "route_info": {
                "route_waypoints": [
                    {"rotation": {"yaw": 90.0}},
                    {"rotation": {"yaw": 180.0}},
                ]
            },
        }

        result = convert_dynamic_scenario_to_drivefuzz_seed(static_data, dynamic_data)

        self.assertEqual(result["map"], "Town03")
        self.assertEqual(result["sp_x"], 101.0)
        self.assertEqual(result["sp_y"], 202.0)
        self.assertEqual(result["yaw"], 90.0)
        self.assertEqual(result["wp_x"], 150.0)
        self.assertEqual(result["wp_y"], 260.0)
        self.assertEqual(result["wp_yaw"], 180.0)


if __name__ == "__main__":
    unittest.main()

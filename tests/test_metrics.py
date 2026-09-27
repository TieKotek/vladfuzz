import unittest
from types import SimpleNamespace

from scenario.carla_scenario import CarlaScenario
from scenario.metrics import DrivingQualityMetrics, ETTCMetrics


class FakeLocation:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5


class FakeActor:
    def __init__(self, x, y, vx, vy, actor_id=1, yaw=0.0):
        self.id = actor_id
        self.type_id = "vehicle.fake"
        self._location = FakeLocation(x=x, y=y, z=0.0)
        self._velocity = SimpleNamespace(x=vx, y=vy, z=0.0)
        self._rotation = SimpleNamespace(yaw=yaw)

    def get_transform(self):
        return SimpleNamespace(location=self._location, rotation=self._rotation)

    def get_velocity(self):
        return self._velocity

    def get_location(self):
        return self._location


class FakeControl:
    def __init__(self, steer=0.0, throttle=0.0, brake=0.0):
        self.steer = steer
        self.throttle = throttle
        self.brake = brake


class LaneOracleTests(unittest.TestCase):
    def test_allowed_lane_change_markings_do_not_trigger_lane_invasion(self):
        scenario = CarlaScenario.__new__(CarlaScenario)
        scenario.lane_invasion_detected = False
        scenario.lane_invasion_events = []

        event = SimpleNamespace(
            crossed_lane_markings=[
                SimpleNamespace(type="Broken", color="White", lane_change="Right")
            ]
        )
        scenario._on_lane_invasion(event)

        self.assertFalse(scenario.lane_invasion_detected)
        self.assertEqual(len(scenario.lane_invasion_events), 1)
        self.assertFalse(scenario.lane_invasion_events[0]["violation"])

    def test_disallowed_lane_change_markings_trigger_lane_invasion(self):
        scenario = CarlaScenario.__new__(CarlaScenario)
        scenario.lane_invasion_detected = False
        scenario.lane_invasion_events = []

        event = SimpleNamespace(
            crossed_lane_markings=[
                SimpleNamespace(type="Solid", color="White", lane_change="NONE")
            ]
        )
        scenario._on_lane_invasion(event)

        self.assertTrue(scenario.lane_invasion_detected)
        self.assertTrue(scenario.lane_invasion_events[0]["violation"])


class ETTCMetricsTests(unittest.TestCase):
    def test_reset_clears_previous_execution_values(self):
        metrics = ETTCMetrics(detection_radius=20.0)
        metrics.set_max_ttc(duration_frames=100, frame_rate=20)
        metrics.ettc_values.extend([1.0, 2.0, 5.0])

        metrics.reset()

        self.assertEqual(metrics.ettc_values, [])
        self.assertEqual(metrics.get_ettc_statistics()["avg_ettc"], 5.0)

    def test_ttc_ignores_closest_approach_without_path_overlap(self):
        metrics = ETTCMetrics(detection_radius=20.0, collision_radius=2.0)
        metrics.set_max_ttc(duration_frames=100, frame_rate=20)
        ego = FakeActor(x=0.0, y=0.0, vx=0.0, vy=0.0, actor_id=1)
        other = FakeActor(x=10.0, y=5.0, vx=-1.0, vy=0.0, actor_id=2)

        self.assertEqual(metrics._calculate_ttc_between_vehicles(ego, other), metrics.max_ttc)

    def test_ttc_reports_projected_collision(self):
        metrics = ETTCMetrics(detection_radius=20.0, collision_radius=2.0)
        metrics.set_max_ttc(duration_frames=100, frame_rate=20)
        ego = FakeActor(x=0.0, y=0.0, vx=0.0, vy=0.0, actor_id=1)
        other = FakeActor(x=10.0, y=0.0, vx=-1.0, vy=0.0, actor_id=2)

        self.assertAlmostEqual(metrics._calculate_ttc_between_vehicles(ego, other), 8.0)


class DrivingQualityMetricsTests(unittest.TestCase):
    def test_counts_hard_acceleration_and_braking_from_speed_changes(self):
        metrics = DrivingQualityMetrics(frame_rate=20)
        metrics.record_frame(FakeActor(0.0, 0.0, 0.0, 0.0), FakeControl(), timestamp=0.0)
        metrics.record_frame(FakeActor(0.0, 0.0, 3.0, 0.0), FakeControl(), timestamp=0.05)
        metrics.record_frame(FakeActor(0.0, 0.0, 0.0, 0.0), FakeControl(), timestamp=0.10)

        stats = metrics.get_statistics()

        self.assertEqual(stats["hard_acceleration_count"], 1)
        self.assertEqual(stats["hard_braking_count"], 1)
        self.assertLess(stats["driving_quality_score"], 1.0)

    def test_counts_hard_turn_and_control_oscillation(self):
        metrics = DrivingQualityMetrics(frame_rate=20)
        metrics.record_frame(FakeActor(0.0, 0.0, 1.0, 5.0), FakeControl(steer=0.6), timestamp=0.0)
        metrics.record_frame(FakeActor(0.0, 0.0, 1.0, 5.0), FakeControl(steer=-0.6), timestamp=0.05)

        stats = metrics.get_statistics()

        self.assertGreaterEqual(stats["hard_turn_count"], 1)
        self.assertEqual(stats["control_oscillation_count"], 1)

    def test_records_low_brake_margin_frames(self):
        metrics = DrivingQualityMetrics(frame_rate=20, detection_radius=20.0, collision_radius=2.0)
        ego = FakeActor(0.0, 0.0, 20.0, 0.0, actor_id=1)
        other = FakeActor(3.0, 0.0, 0.0, 0.0, actor_id=2)

        metrics.record_frame(ego, FakeControl(), spawned_actors=[ego, other], timestamp=0.0)
        stats = metrics.get_statistics()

        self.assertEqual(stats["low_brake_margin_frames"], 1)
        self.assertLess(stats["min_brake_margin"], 0.0)

    def test_records_longitudinal_lateral_speed_and_yaw_rate(self):
        metrics = DrivingQualityMetrics(frame_rate=20)

        metrics.record_frame(FakeActor(0.0, 0.0, 4.0, 3.0, actor_id=1, yaw=0.0), FakeControl(), timestamp=0.0)
        metrics.record_frame(FakeActor(0.0, 0.0, 4.0, 3.0, actor_id=1, yaw=9.0), FakeControl(), timestamp=0.05)
        stats = metrics.get_statistics()

        self.assertGreater(stats["avg_longitudinal_speed"], 0.0)
        self.assertGreater(stats["avg_lateral_speed"], 0.0)
        self.assertAlmostEqual(stats["max_abs_yaw_rate"], 180.0)

    def test_counts_oversteer_and_understeer_like_events(self):
        metrics = DrivingQualityMetrics(frame_rate=20)

        metrics.record_frame(FakeActor(0.0, 0.0, 8.0, 0.0, actor_id=1, yaw=0.0), FakeControl(steer=0.8), timestamp=0.0)
        metrics.record_frame(FakeActor(0.0, 0.0, 8.0, 5.0, actor_id=1, yaw=12.0), FakeControl(steer=0.8), timestamp=0.05)
        metrics.record_frame(FakeActor(0.0, 0.0, 7.82, 1.66, actor_id=1, yaw=12.0), FakeControl(steer=0.8), timestamp=0.10)
        stats = metrics.get_statistics()

        self.assertGreaterEqual(stats["oversteer_count"], 1)
        self.assertGreaterEqual(stats["understeer_count"], 1)


if __name__ == "__main__":
    unittest.main()

import unittest
import os
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

from vladfuzz_runtime.env_patches import bootstrap_project_python_paths

bootstrap_project_python_paths()
if os.environ.get("CARLA_PATH"):
    carla_python_api = Path(os.environ["CARLA_PATH"]) / "PythonAPI"
    if carla_python_api.exists():
        sys.path.append(str(carla_python_api))
        sys.path.append(str(carla_python_api / "carla"))

from leaderboard.autoagents.agent_wrapper import AgentWrapper
from scenario.carla_scenario import CarlaScenario


class AgentWrapperSensorStateTests(unittest.TestCase):
    def test_sensor_list_is_instance_local(self):
        wrapper_a = AgentWrapper(SimpleNamespace())
        wrapper_b = AgentWrapper(SimpleNamespace())

        wrapper_a._sensors_list.append("sensor-a")

        self.assertEqual(wrapper_a._sensors_list, ["sensor-a"])
        self.assertEqual(wrapper_b._sensors_list, [])

    def test_cleanup_continues_when_one_sensor_raises(self):
        agent = SimpleNamespace(sensor_interface=Mock())
        wrapper = AgentWrapper(agent)
        broken_sensor = Mock()
        broken_sensor.stop.side_effect = RuntimeError("sensor already stopped")
        healthy_sensor = Mock()
        wrapper._sensors_list = [broken_sensor, healthy_sensor]
        world = Mock()

        with patch(
            "leaderboard.autoagents.agent_wrapper.CarlaDataProvider.get_world",
            return_value=world,
        ):
            wrapper.cleanup()

        broken_sensor.destroy.assert_called_once()
        healthy_sensor.stop.assert_called_once()
        healthy_sensor.destroy.assert_called_once()
        agent.sensor_interface.clear_sensors.assert_called_once()
        world.tick.assert_called_once()
        self.assertEqual(wrapper._sensors_list, [])


class CarlaScenarioCleanupTests(unittest.TestCase):
    def test_per_scenario_cleanup_does_not_disable_sync_mode(self):
        scenario = CarlaScenario.__new__(CarlaScenario)
        scenario.agent_wrapper = None
        scenario.collision_sensor = None
        scenario.lane_sensor = None
        scenario.ego_camera = None
        scenario.spawned_actors = []
        scenario.ego_vehicle = None
        scenario.client = Mock()
        scenario.traffic_manager = Mock()
        scenario._print = Mock()

        settings = SimpleNamespace(synchronous_mode=True, fixed_delta_seconds=0.05)
        world = Mock()
        world.get_settings.return_value = settings
        scenario.world = world

        scenario.cleanup_actors()

        world.apply_settings.assert_not_called()
        world.wait_for_tick.assert_not_called()
        scenario.traffic_manager.set_synchronous_mode.assert_not_called()
        self.assertTrue(settings.synchronous_mode)
        self.assertEqual(settings.fixed_delta_seconds, 0.05)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
from typing import Dict, Optional, Tuple
from enum import IntEnum

from vladfuzz_runtime.agent_lifecycle import reset_agent_episode_state
from vladfuzz_runtime.env_patches import bootstrap_carla_python_paths, bootstrap_project_python_paths
from vladfuzz_runtime.model_registry import bootstrap_model_environment, resolve_agent_class

bootstrap_project_python_paths()
bootstrap_carla_python_paths()

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
try:
    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
except ImportError:
    GlobalRoutePlannerDAO = None
from leaderboard.autoagents.agent_wrapper import AgentWrapperFactory
from leaderboard.utils.route_manipulation import _get_latlon_ref, _location_to_gps
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


class RouteCommand(IntEnum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class VLABackendAdapter:
    """Leaderboard-style VLA backend adapter for external CARLA fuzzers."""

    _AGENT_CACHE: Dict[Tuple[str, str, int, str, Optional[str], int], object] = {}

    def __init__(
        self,
        *,
        model_name: str,
        instruction: str,
        host: str = "localhost",
        port: int = 2000,
        gpu_id: int = 0,
        config_path: Optional[str] = None,
        hydra_config_path: Optional[str] = None,
        reuse_backend: bool = True,
    ):
        self.model_name = model_name
        self.instruction = instruction
        self.host = host
        self.port = port
        self.gpu_id = gpu_id
        self.agent_wrapper = None
        self.agent_instance = None

        spec = bootstrap_model_environment(model_name)
        resolved_config_path = str(Path(config_path) if config_path else spec.config_path)
        resolved_hydra_config_path = hydra_config_path
        if resolved_hydra_config_path is None and spec.hydra_config_path is not None:
            resolved_hydra_config_path = str(spec.hydra_config_path)

        cache_key = (
            model_name,
            host,
            port,
            resolved_config_path,
            resolved_hydra_config_path,
            gpu_id,
        )
        if reuse_backend and cache_key in self._AGENT_CACHE:
            self.agent_instance = self._AGENT_CACHE[cache_key]
        else:
            agent_class = resolve_agent_class(model_name)
            self.agent_instance = agent_class(host, port)
            self.agent_instance.setup(resolved_config_path, resolved_hydra_config_path, gpu_id=gpu_id)
            if reuse_backend:
                self._AGENT_CACHE[cache_key] = self.agent_instance
        self._configure_instruction_mode()

    def setup(self, world, ego_vehicle, start_location, target_location) -> None:
        self.cleanup()
        client = carla.Client(self.host, self.port)
        client.set_timeout(10.0)
        CarlaDataProvider.set_client(client)
        self._set_data_provider_world(world)
        self._register_hero_actor(ego_vehicle)

        route = self._trace_route(world, start_location, target_location)
        lat_ref, lon_ref = _get_latlon_ref(world)
        world_route = []
        gps_route = []
        for waypoint, road_option in route:
            route_command = self._normalize_route_command(road_option)
            world_route.append((waypoint.transform, route_command))
            gps_route.append((_location_to_gps(lat_ref, lon_ref, waypoint.transform.location), route_command))

        reset_agent_episode_state(self.agent_instance)
        self.agent_instance.user_command = self.instruction
        self.agent_instance.dreamer_flag = False
        self.agent_instance.set_global_plan(gps_route, world_route)
        self.agent_instance.set_gps_reference(lat_ref, lon_ref)

        self.agent_wrapper = AgentWrapperFactory.get_wrapper(self.agent_instance)
        self.agent_wrapper.setup_sensors(ego_vehicle)
        world.tick()

    def _configure_instruction_mode(self) -> None:
        if self.model_name == "simlingo" and hasattr(self.agent_instance, "config"):
            self.agent_instance.config.eval_route_as = "command"

    @staticmethod
    def _trace_route(world, start_location, target_location):
        grp = VLABackendAdapter._make_global_route_planner(world.get_map(), 2.0)
        route = grp.trace_route(start_location, target_location)
        if not route:
            raise RuntimeError("VLA backend adapter could not trace a route between start and target.")
        return route

    @staticmethod
    def _normalize_route_command(road_option):
        value = getattr(road_option, "value", road_option)
        return RouteCommand(int(value))

    @staticmethod
    def _make_global_route_planner(world_map, sampling_resolution):
        try:
            return GlobalRoutePlanner(world_map, sampling_resolution)
        except TypeError:
            if GlobalRoutePlannerDAO is None:
                raise
            dao = GlobalRoutePlannerDAO(world_map, sampling_resolution)
            planner = GlobalRoutePlanner(dao)
            planner.setup()
            return planner

    @staticmethod
    def _set_data_provider_world(world) -> None:
        CarlaDataProvider._actor_velocity_map.clear()
        CarlaDataProvider._actor_location_map.clear()
        CarlaDataProvider._actor_transform_map.clear()
        CarlaDataProvider._all_actors = None
        CarlaDataProvider._carla_actor_pool = {}
        CarlaDataProvider._world = world
        CarlaDataProvider._sync_flag = world.get_settings().synchronous_mode
        CarlaDataProvider._map = world.get_map()
        CarlaDataProvider._blueprint_library = world.get_blueprint_library()
        CarlaDataProvider._grp = VLABackendAdapter._make_global_route_planner(
            CarlaDataProvider._map,
            2.0,
        )
        CarlaDataProvider.generate_spawn_points()
        CarlaDataProvider.prepare_map()

    @staticmethod
    def _register_hero_actor(ego_vehicle) -> None:
        try:
            ego_vehicle.attributes["role_name"] = "hero"
        except Exception:
            pass
        try:
            CarlaDataProvider._carla_actor_pool[ego_vehicle.id] = ego_vehicle
            CarlaDataProvider.register_actor(ego_vehicle, ego_vehicle.get_transform())
        except Exception:
            pass

    def run_step(self, timestamp=None):
        if self.agent_wrapper is None:
            raise RuntimeError("VLABackendAdapter.setup must be called before run_step.")
        if timestamp is not None:
            GameTime.on_carla_tick(timestamp)
        return self.agent_wrapper()

    def cleanup(self) -> None:
        if self.agent_wrapper is not None:
            self.agent_wrapper.cleanup()
            self.agent_wrapper = None

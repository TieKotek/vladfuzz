import carla
import json
import random
import time
import math
import os
import gc
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

from vladfuzz_runtime.env_patches import bootstrap_carla_python_paths, bootstrap_project_python_paths

bootstrap_project_python_paths()
bootstrap_carla_python_paths()

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from leaderboard.autoagents.agent_wrapper import AgentWrapperFactory
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from leaderboard.utils.route_manipulation import _location_to_gps, _get_latlon_ref

from vladfuzz_runtime.agent_lifecycle import reset_agent_episode_state
from vladfuzz_runtime.model_registry import get_model_spec, resolve_agent_class
from vladfuzz_runtime.oracle import (
    collision_actor_type,
    collision_impulse,
    should_count_collision,
)
from vladfuzz_runtime.infrastructure import CarlaInfrastructureError, is_carla_infrastructure_error

from .metrics import DrivingQualityMetrics, ETTCMetrics, PathDeviationMetrics
from .camera import EgoCamera, filter_coordinates


@dataclass
class VehicleConfig:
    blueprint: str
    spawn_point_index: int
    speed_percentage_difference: Optional[float] = None
    spawn_point_coordinates: Optional[Dict] = None
    color: Optional[str] = None


@dataclass
class ScenarioConfig:
    duration_frames: int
    scenario_area: Dict
    ego_car: VehicleConfig
    npc_vehicle_count: int
    npc_vehicles: List[VehicleConfig]
    target_point: Optional[Dict] = None
    route_info: Optional[Dict] = None
    weather: Optional[str] = None


class CarlaScenario:
    """CARLA dynamic scenario class."""

    WEATHER_PRESETS = [
        'ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'MidRainyNoon',
        'HardRainNoon', 'SoftRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset',
        'WetCloudySunset', 'MidRainSunset', 'HardRainSunset', 'SoftRainSunset',
        'ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight', 'SoftRainNight',
        'MidRainyNight', 'HardRainNight', 'DustStorm'
    ]

    NPC_VEHICLE_BLUEPRINTS = [
        'vehicle.audi.a2',
        'vehicle.mini.cooper_s',
        'vehicle.seat.leon',
        'vehicle.bmw.grandtourer',
        'vehicle.audi.tt',
        'vehicle.ford.ambulance',
        'vehicle.dodge.charger_police',
        'vehicle.mercedes.sprinter'
    ]

    SPECIAL_VEHICLES = [
        'vehicle.ford.ambulance',
        'vehicle.dodge.charger_police',
        'vehicle.mercedes.sprinter'
    ]

    VEHICLE_COLORS = [
        'Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple', 'Pink',
        'White', 'Black', 'Silver', 'Gray', 'Brown', 'Cyan', 'Magenta'
    ]

    COLOR_RGB_MAP = {
        'Red': '255,0,0',
        'Blue': '0,0,255',
        'Green': '0,255,0',
        'Yellow': '255,255,0',
        'Orange': '255,165,0',
        'Purple': '128,0,128',
        'Pink': '255,192,203',
        'White': '255,255,255',
        'Black': '0,0,0',
        'Silver': '192,192,192',
        'Gray': '128,128,128',
        'Brown': '165,42,42',
        'Cyan': '0,255,255',
        'Magenta': '255,0,255'
    }

    EGO_VEHICLE_BLUEPRINT = 'vehicle.tesla.model3'

    def __init__(
        self,
        config_path=None,
        hydra_config_path=None,
        host='localhost',
        port=2000,
        timeout=20.0,
        frame_rate=20,
        tm_seed=None,
        gpu_id=0,
        verbose=True,
        model_name=None,
        oracle_checks=None,
        speed_tolerance=0.1,
        scenario_duration_frames=None,
        tm_port=None,
    ):
        self.verbose = verbose
        self.host = host
        self.port = port
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.frame_rate = frame_rate
        self.tm_seed = tm_seed
        self.ego_vehicle = None
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)

        if tm_port is None:
            self.traffic_manager = self.client.get_trafficmanager()
        else:
            self.traffic_manager = self.client.get_trafficmanager(tm_port)
        self.tm_port = self.traffic_manager.get_port()

        if self.tm_seed is not None:
            self.traffic_manager.set_random_device_seed(self.tm_seed)
            self._print(f"✅ Traffic Manager seeded with:  {self.tm_seed}")

        self.static_scenario_data = None
        self.scenario_config = None

        self.spawned_actors = []
        self.ego_camera = None
        self.agent_wrapper = None
        self.model_spec = None
        self.agent_instance = None
        if model_name is not None:
            self.model_spec = get_model_spec(model_name)
            resolved_config_path = str(config_path or self.model_spec.config_path)
            resolved_hydra_config_path = hydra_config_path
            if resolved_hydra_config_path is None and self.model_spec.hydra_config_path is not None:
                resolved_hydra_config_path = str(self.model_spec.hydra_config_path)

            agent_class = resolve_agent_class(model_name)
            self.agent_instance = agent_class(self.host, self.port)
            self.agent_instance.setup(resolved_config_path, resolved_hydra_config_path, gpu_id=gpu_id)
        self.collision_sensor = None
        self.collision_detected = False
        self.collision_events = []
        self.lane_sensor = None
        self.lane_invasion_detected = False
        self.lane_invasion_events = []
        self.speeding_detected = False
        self.stuck_detected = False
        self.stuck_duration_frames = 0
        self.timeout_detected = False
        self.timeout_frames = None
        self.out_of_bounds_detected = False
        self.out_of_bounds_location = None
        self.other_error = None
        self.other_error_value = None
        default_oracle_checks = {
            "collision": True,
            "stuck": True,
            "lane_invasion": False,
            "speeding": False,
            "timeout": True,
            "out_of_bounds": True,
            "other": False,
        }
        if oracle_checks:
            default_oracle_checks.update(oracle_checks)
        self.oracle_checks = default_oracle_checks
        self.speed_tolerance = speed_tolerance
        self.scenario_duration_frames = scenario_duration_frames

        self.path_deviation_metrics = PathDeviationMetrics()
        self.ettc_metrics = ETTCMetrics(detection_radius=20.0)
        self.driving_quality_metrics = DrivingQualityMetrics(
            frame_rate=self.frame_rate,
            detection_radius=self.ettc_metrics.detection_radius,
            collision_radius=self.ettc_metrics.collision_radius,
        )

        self.ego_speed_history = []
        self.ego_max_speed = 0.0
        self.ego_avg_speed = 0.0

        self.visible_vehicle_ids = []
        self.npc_vehicle_info = {}
        self.actor_id_to_spawn_index = {}

        if self.model_spec is not None:
            self._print(f"CARLA scenario manager initialized with model: {self.model_spec.name}")
        else:
            self._print("CARLA scenario manager initialized without VLA model")
        self._print(f"Traffic Manager initialized on port {self.tm_port}")

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    def _setup_synchronous_mode(self):
        """Setup synchronous mode for world and traffic manager"""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / self.frame_rate
        self.world.apply_settings(settings)
        
        self.traffic_manager.set_synchronous_mode(True)
        
        self._print("✅ Synchronous mode enabled for world and traffic manager")
            
    def _ensure_correct_map(self) -> bool:
        """
        Ensure current map matches the required scenario map, automatically switch if different
        """
        if not self.static_scenario_data:
            return True
        
        current_map = self.world.get_map().name
        required_map = self.static_scenario_data['map_name']
        
        current_map_name = current_map.split('/')[-1]
        required_map_name = required_map.split('/')[-1]
        
        if current_map_name != required_map_name:
            self._print(f"Map mismatch detected: Current is {current_map}, required is {required_map}. Switching...")
            try:
                self.world = self.client.load_world(required_map_name)
                self._print("Waiting for new map to stabilize...")
                time.sleep(20.0)  # Wait longer for map to load
                self.map = self.world.get_map()                
                self._print(f"✅ Map switch successful: {self.world.get_map().name}")
                return True
            
            except Exception as e:
                self._print(f"❌ Map switch failed: {e}")
                return False
        else:
            self._print(f"✅ Map match: {current_map}")            
            return True
    
    def load_static_scenario(self, static_scenario_path: str) -> bool:
        """
        Load static scenario data
        """
        try:
            with open(static_scenario_path, 'r') as f:
                self.static_scenario_data = json.load(f)
            
            required_fields = ['map_name', 'scenario_center', 'scenario_extent', 'spawn_points', 'ego_spawn_num']
            for field in required_fields:
                if field not in self.static_scenario_data:
                    print(f"Static scenario file missing required field: {field}")
                    return False
            ego_spawn_num = self.static_scenario_data['ego_spawn_num']

            if not isinstance(ego_spawn_num, int) or ego_spawn_num <= 0 or ego_spawn_num > len(self.static_scenario_data['spawn_points']):
                return False

            self._print(f"Successfully loaded static scenario: {static_scenario_path}")
            self._print(f"  Map: {self.static_scenario_data['map_name']}")
            self._print(f"  Available spawn points: {len(self.static_scenario_data['spawn_points'])}")
            
            if not self._ensure_correct_map():
                return False
            
            # Lock traffic lights to green - this will also be called after TM reset
            self._lock_traffic_lights()
            
            return True
            
        except Exception as e:
            print(f"Failed to load static scenario: {e}")
            return False
    
    def _convert_relative_to_absolute(self, relative_point: Dict) -> Dict:
        """
        Convert relative coordinates to absolute coordinates
        """
        if not self.static_scenario_data:
            return relative_point
        
        center = self.static_scenario_data['scenario_center']
        
        result = {
            'x': relative_point['x'] + center['x'],
            'y': relative_point['y'] + center['y'],
            'z': relative_point['z']
        }
        
        # Add yaw only if it exists in the relative_point
        if 'yaw' in relative_point:
            result['yaw'] = relative_point['yaw']
        
        return result
        
    def _move_spectator_to_scenario_center(self):
        """Move spectator to above scenario center"""
        if not self.static_scenario_data:
            return
        
        center = self.static_scenario_data['scenario_center']
        extent = self.static_scenario_data['scenario_extent']
        
        height = max(extent['x'], extent['y']) * 1.2 + 50
        
        spectator_location = carla.Location(x=center['x'], y=center['y'], z=height)
        spectator_rotation = carla.Rotation(pitch=-90, yaw=0, roll=0)
        spectator_transform = carla.Transform(spectator_location, spectator_rotation)
        
        spectator = self.world.get_spectator()
        spectator.set_transform(spectator_transform)
        
    def _get_ego_vehicle_location(self) -> Optional[Dict]:
        """
        Get the current ego vehicle location relative to scenario center
        """
        if not self.spawned_actors or not self.static_scenario_data:
            return None
        
        ego_vehicle = self.ego_vehicle
        if not ego_vehicle:
            return None
        
        ego_transform = ego_vehicle.get_transform()
        center = self.static_scenario_data['scenario_center']
        
        return {
            'x': ego_transform.location.x - center['x'],
            'y': ego_transform.location.y - center['y'],
            'yaw': ego_transform.rotation.yaw
        }
    
    def _select_visible_target_point(self, ego_spawn_index: int, min_target_distance:int = 10) -> Optional[Dict]:
        """
        Select a target point visible to ego vehicle camera from available spawn points with occlusion detection
        """
        if not self.static_scenario_data:
            return None
        
        spawn_points = self.static_scenario_data['spawn_points']
        ego_spawn_num = self.static_scenario_data['ego_spawn_num']
        
        # Get candidate target points from spawn_points[ego_spawn_num:]
        candidate_points = spawn_points[ego_spawn_num:]
        
        if not candidate_points:
            self._print("❌ No candidate target points available")
            return None
        
        # Create temporary ego vehicle for camera frustum calculation
        ego_spawn_point = spawn_points[ego_spawn_index]
        ego_absolute_spawn = self._convert_relative_to_absolute(ego_spawn_point)
        ego_transform = carla.Transform(
            carla.Location(x=ego_absolute_spawn['x'], y=ego_absolute_spawn['y'], z=ego_absolute_spawn['z']),
            carla.Rotation(yaw=ego_absolute_spawn['yaw'])
        )
        
        # Calculate ego camera transform (same as EgoCamera setup)
        camera_transform = carla.Transform(
            carla.Location(
                x=ego_transform.location.x + 2.0 * math.cos(math.radians(ego_transform.rotation.yaw)),
                y=ego_transform.location.y + 2.0 * math.sin(math.radians(ego_transform.rotation.yaw)),
                z=ego_transform.location.z + 1.8
            ),
            carla.Rotation(pitch=-8, yaw=ego_transform.rotation.yaw, roll=0)
        )
        
        # Convert candidate points to absolute coordinates for frustum checking
        candidate_locations = []
        for point in candidate_points:
            abs_point = self._convert_relative_to_absolute(point)
            candidate_locations.append(carla.Location(x=abs_point['x'], y=abs_point['y'], z=abs_point['z']))
        
        # Filter visible points using camera frustum
        visible_locations = filter_coordinates(
            camera_transform,
            candidate_locations,
            fov_h=90,
            fov_v=60,
            max_distance=80
        )
        
        if not visible_locations:
            self._print("❌ No target points visible to ego camera")
            return None
        
        # Further filter by occlusion detection and distance
        candidate_locations = []
        for location in visible_locations:
            # Check distance from ego vehicle
            distance_to_ego = ego_transform.location.distance(location)
            self._print(f"Debug: Candidate point distance to ego: {distance_to_ego:.1f}m")
            if distance_to_ego >= min_target_distance:
                candidate_locations.append(location)
            else:
                self._print(f"Debug: Point too close ({distance_to_ego:.1f}m)")
        
        if not candidate_locations:
            self._print("❌ All visible target points are too close")
            # Fallback: just use the first visible location for testing
            if visible_locations:
                self._print("⚠️ Using fallback: first visible location")
                selected_location = visible_locations[0]
            else:
                return None
        else:
            # Randomly select one unoccluded point
            selected_location = random.choice(candidate_locations)
        
        # Convert selected location back to relative coordinates
        center = self.static_scenario_data['scenario_center']
        
        target_point = {
            'x': selected_location.x - center['x'],
            'y': selected_location.y - center['y'],
            'z': selected_location.z
        }
        
        self._print(f"✅ Selected target point: ({target_point['x']:.1f}, {target_point['y']:.1f}, {target_point['z']:.1f})")
        
        return target_point
    
    def _select_target_point(self, ego_spawn_index: int, 
                                                      min_target_distance: int = 10, 
                                                      max_route_distance: float = 60.0) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Select a target point visible to ego vehicle camera with route distance filtering.
        
        Args:
            ego_spawn_index: Index of ego vehicle spawn point
            min_target_distance: Minimum direct distance to target in meters
            max_route_distance: Maximum allowed route distance in meters
            
        Returns:
            Tuple[target_point, route_info]: Selected target point and its route information
            (None, None): If no suitable target point found
        """
        if not self.static_scenario_data:
            return None, None
        
        spawn_points = self.static_scenario_data['spawn_points']
        ego_spawn_num = self.static_scenario_data['ego_spawn_num']
        
        # Get candidate target points from spawn_points[ego_spawn_num:]
        candidate_points = spawn_points[ego_spawn_num:]
        
        if not candidate_points:
            self._print("❌ No candidate target points available")
            return None, None
        
        # Create ego vehicle transform for calculations
        ego_spawn_point = spawn_points[ego_spawn_index]
        ego_absolute_spawn = self._convert_relative_to_absolute(ego_spawn_point)
        ego_transform = carla.Transform(
            carla.Location(x=ego_absolute_spawn['x'], y=ego_absolute_spawn['y'], z=ego_absolute_spawn['z']),
            carla.Rotation(yaw=ego_absolute_spawn['yaw'])
        )
        ego_location = ego_transform.location
        
        # Calculate ego camera transform (same as EgoCamera setup)
        camera_transform = carla.Transform(
            carla.Location(
                x=ego_transform.location.x + 2.0 * math.cos(math.radians(ego_transform.rotation.yaw)),
                y=ego_transform.location.y + 2.0 * math.sin(math.radians(ego_transform.rotation.yaw)),
                z=ego_transform.location.z + 1.8
            ),
            carla.Rotation(pitch=-8, yaw=ego_transform.rotation.yaw, roll=0)
        )
        
        # Convert candidate points to absolute coordinates for processing
        candidate_locations = []
        for point in candidate_points:
            abs_point = self._convert_relative_to_absolute(point)
            candidate_locations.append(carla.Location(x=abs_point['x'], y=abs_point['y'], z=abs_point['z']))
        
        # Step 1: Filter visible points using camera frustum
        visible_locations = filter_coordinates(
            camera_transform,
            candidate_locations,
            fov_h=90,
            fov_v=60,
            max_distance=80
        )
        
        if not visible_locations:
            self._print("❌ No target points visible to ego camera")
            return None, None
        
        self._print(f"🔍 Found {len(visible_locations)} visible target points, filtering by route distance...")
        
        # Step 2: Filter by direct distance, route distance, and atomicity
        valid_candidates = []
        
        for location in visible_locations:
            # Check direct distance from ego vehicle
            direct_distance = ego_location.distance(location)
            
            if direct_distance < min_target_distance:
                continue
            
            # Calculate route information using the helper method
            route_info = self._calculate_route_info(ego_location, location, verbose=False)
            
            if route_info is None:
                continue
                
            route_distance = route_info['route_length']
            
            if route_distance > max_route_distance:
                continue
            
            # Check if the route is valid (atomic, not too simple, single lane changes)
            if not self._is_route_valid(route_info):
                continue
            
            self._print(f"   ✅ Valid candidate: ({location.x:.1f}, {location.y:.1f}) - Direct: {direct_distance:.1f}m, Route: {route_distance:.1f}m")
            valid_candidates.append((location, route_info))
        
        if not valid_candidates:
            self._print("❌ No target points meet all criteria (visibility + distance + route + validity)")
            return None, None
        
        # Step 3: Randomly select one valid candidate
        selected_location, selected_route_info = random.choice(valid_candidates)
        
        # Convert selected location back to relative coordinates
        center = self.static_scenario_data['scenario_center']
        target_point = {
            'x': selected_location.x - center['x'],
            'y': selected_location.y - center['y'],
            'z': selected_location.z
        }
        
        self._print(f"✅ Selected target point with route filtering:")
        self._print(f"   📍 Location: ({target_point['x']:.1f}, {target_point['y']:.1f}, {target_point['z']:.1f})")
        self._print(f"   📏 Route distance: {selected_route_info['route_length']}m")
        self._print(f"   🔄 Route maneuvers: {' → '.join(set(selected_route_info['maneuvers']))}")
        
        return target_point, selected_route_info

    def _visible_target_locations(self, ego_spawn_index: int) -> Tuple[Optional[carla.Location], List[carla.Location]]:
        if not self.static_scenario_data:
            return None, []

        spawn_points = self.static_scenario_data['spawn_points']
        ego_spawn_num = self.static_scenario_data['ego_spawn_num']
        candidate_points = spawn_points[ego_spawn_num:]
        if not candidate_points:
            return None, []

        ego_spawn_point = spawn_points[ego_spawn_index]
        ego_absolute_spawn = self._convert_relative_to_absolute(ego_spawn_point)
        ego_transform = carla.Transform(
            carla.Location(x=ego_absolute_spawn['x'], y=ego_absolute_spawn['y'], z=ego_absolute_spawn['z']),
            carla.Rotation(yaw=ego_absolute_spawn['yaw'])
        )
        camera_transform = carla.Transform(
            carla.Location(
                x=ego_transform.location.x + 2.0 * math.cos(math.radians(ego_transform.rotation.yaw)),
                y=ego_transform.location.y + 2.0 * math.sin(math.radians(ego_transform.rotation.yaw)),
                z=ego_transform.location.z + 1.8
            ),
            carla.Rotation(pitch=-8, yaw=ego_transform.rotation.yaw, roll=0)
        )
        candidate_locations = []
        for point in candidate_points:
            abs_point = self._convert_relative_to_absolute(point)
            candidate_locations.append(carla.Location(x=abs_point['x'], y=abs_point['y'], z=abs_point['z']))
        return ego_transform.location, filter_coordinates(
            camera_transform,
            candidate_locations,
            fov_h=90,
            fov_v=60,
            max_distance=80
        )

    def enumerate_seed_candidates(self,
                                  min_target_distance: int = 10,
                                  max_route_distance: float = 60.0) -> List[Dict]:
        """
        Enumerate valid ego-target route candidates for diverse seed generation.
        """
        if not self.static_scenario_data:
            raise ValueError("Please load static scenario data first")

        spawn_points = self.static_scenario_data['spawn_points']
        ego_spawn_num = self.static_scenario_data['ego_spawn_num']
        center = self.static_scenario_data['scenario_center']
        carla_map = self.world.get_map()
        candidates = []

        for ego_spawn_index in range(ego_spawn_num):
            ego_location, visible_locations = self._visible_target_locations(ego_spawn_index)
            if ego_location is None:
                continue
            ego_abs = self._convert_relative_to_absolute(spawn_points[ego_spawn_index])
            ego_wp = carla_map.get_waypoint(
                carla.Location(x=ego_abs['x'], y=ego_abs['y'], z=ego_abs['z']),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            for target_index, location in enumerate(visible_locations):
                direct_distance = ego_location.distance(location)
                if direct_distance < min_target_distance:
                    continue
                route_info = self._calculate_route_info(ego_location, location, verbose=False)
                if route_info is None:
                    continue
                if route_info['route_length'] > max_route_distance:
                    continue
                if not self._is_route_valid(route_info):
                    continue
                target_wp = carla_map.get_waypoint(
                    location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                candidates.append({
                    "candidate_id": f"ego{ego_spawn_index}_target{target_index}",
                    "ego_spawn_index": ego_spawn_index,
                    "target_point": {
                        "x": location.x - center['x'],
                        "y": location.y - center['y'],
                        "z": location.z,
                    },
                    "route_info": route_info,
                    "direct_distance": direct_distance,
                    "ego_road_id": ego_wp.road_id if ego_wp else None,
                    "ego_lane_id": ego_wp.lane_id if ego_wp else None,
                    "target_road_id": target_wp.road_id if target_wp else None,
                    "target_lane_id": target_wp.lane_id if target_wp else None,
                })

        self._print(f"Enumerated {len(candidates)} valid seed candidates")
        return candidates
       
    def _validate_blueprint(self, blueprint: str) -> bool:
        """Validate if vehicle blueprint is valid"""
        try:
            self.world.get_blueprint_library().find(blueprint)
            return True
        except Exception:
            return False
    
    def _validate_scenario_data(self, scenario_data: Dict) -> Tuple[bool, str]:
        """Validate scenario data integrity"""
        try:
            required_fields = ['duration_frames', 'ego_car', 'npc_vehicles', 'npc_vehicle_count']
            for field in required_fields:
                if field not in scenario_data:
                    return False, f"Missing required field: {field}"
            
            if not isinstance(scenario_data['duration_frames'], int) or scenario_data['duration_frames'] <= 0:
                return False, "Invalid duration_frames"
            
            if not self.static_scenario_data:
                return False, "Please load static scenario data first"
            
            available_spawn_points = len(self.static_scenario_data['spawn_points'])
            ego_spawn_num = self.static_scenario_data['ego_spawn_num']
            
            ego_validation = self._validate_vehicle_config(
                scenario_data['ego_car'], ego_spawn_num, "Ego car", is_ego=True
            )
            if not ego_validation[0]: return ego_validation
            
            if len(scenario_data['npc_vehicles']) != scenario_data['npc_vehicle_count']:
                return False, "NPC vehicle count mismatch"
            
            for i, npc_vehicle in enumerate(scenario_data['npc_vehicles']):
                npc_validation = self._validate_vehicle_config(
                    npc_vehicle, available_spawn_points, f"NPC vehicle {i+1}", is_ego=False
                )
                if not npc_validation[0]: return npc_validation
            
            if not self._check_spawn_point_conflicts(scenario_data['ego_car'], scenario_data['npc_vehicles'])[0]:
                return self._check_spawn_point_conflicts(scenario_data['ego_car'], scenario_data['npc_vehicles'])
            
            if not self._validate_all_blueprints(scenario_data['ego_car'], scenario_data['npc_vehicles'])[0]:
                return self._validate_all_blueprints(scenario_data['ego_car'], scenario_data['npc_vehicles'])
            
            return True, "Validation passed"
            
        except Exception as e:
            return False, f"Exception during validation: {str(e)}"
    
    def _validate_vehicle_config(self, vehicle_config: Dict, max_spawn_points: int,
                                vehicle_name: str, is_ego: bool = False) -> Tuple[bool, str]:
        """Validate single vehicle configuration."""
        required_fields = ['blueprint', 'spawn_point_index']
        for field in required_fields:
            if field not in vehicle_config:
                return False, f"{vehicle_name} missing required field: {field}"

        spawn_index = vehicle_config['spawn_point_index']
        if not isinstance(spawn_index, int) or not (0 <= spawn_index < max_spawn_points):
            return False, f"{vehicle_name} spawn point index invalid: {spawn_index}"

        if is_ego:
            if 'speed_percentage_difference' in vehicle_config and vehicle_config['speed_percentage_difference'] is not None:
                return False, "Ego car config should not contain 'speed_percentage_difference'"
        else:
            if 'speed_percentage_difference' not in vehicle_config:
                return False, f"NPC vehicle missing required field: speed_percentage_difference"
            speed_diff = vehicle_config['speed_percentage_difference']
            if not isinstance(speed_diff, (int, float)) or not (-50.0 <= speed_diff <= 50.0):
                return False, f"NPC vehicle speed percentage difference out of range [-50, 50]: {speed_diff}"

        return True, "Validation passed"
    
    def _check_spawn_point_conflicts(self, ego_car: Dict, npc_vehicles: List[Dict]) -> Tuple[bool, str]:
        """Check for spawn point index conflicts"""
        used_spawn_points = {ego_car['spawn_point_index']}
        
        for i, npc in enumerate(npc_vehicles):
            npc_spawn = npc['spawn_point_index']
            if npc_spawn in used_spawn_points:
                return False, f"Spawn point conflict: NPC vehicle {i+1} uses a spawn point already taken."
            used_spawn_points.add(npc_spawn)
        
        return True, "No spawn point conflicts"
    
    def _validate_all_blueprints(self, ego_car: Dict, npc_vehicles: List[Dict]) -> Tuple[bool, str]:
        """Validate all vehicle blueprints"""
        if not self._validate_blueprint(ego_car['blueprint']):
            return False, f"Ego car blueprint invalid: '{ego_car['blueprint']}'"
        
        for i, npc in enumerate(npc_vehicles):
            if not self._validate_blueprint(npc['blueprint']):
                return False, f"NPC vehicle {i+1} blueprint invalid: '{npc['blueprint']}'"
        
        return True, "All blueprints valid"

    def _is_ego_within_scenario_bounds(self):
        """
        Check if ego vehicle is within the static scenario extent boundaries.
        
        Returns:
            bool: True if ego vehicle is within bounds, False otherwise
        """
        if not self.ego_vehicle or not self.static_scenario_data:
            return True  # Default to True if no data available
        
        try:
            # Get scenario center and extent
            center = self.static_scenario_data['scenario_center']
            extent = self.static_scenario_data['scenario_extent']
            
            # Calculate boundary limits
            min_x = center['x'] - extent['x']
            max_x = center['x'] + extent['x']
            min_y = center['y'] - extent['y']
            max_y = center['y'] + extent['y']
            
            # Get ego vehicle position
            ego_location = self.ego_vehicle.get_location()
            ego_x = ego_location.x
            ego_y = ego_location.y
            
            # Check if within bounds
            within_bounds = (min_x <= ego_x <= max_x) and (min_y <= ego_y <= max_y)
            
            return within_bounds
            
        except Exception as e:
            # Return True on error to avoid false positives
            return True

    def _identify_visible_vehicles(self) -> List[int]:
        """
        Identify NPC vehicles that are visible to the ego vehicle camera at scenario start.
        Uses similar visibility detection logic as target point selection.
        
        Returns:
            List[int]: List of vehicle actor IDs that are visible to ego camera
        """
        if not self.ego_vehicle or not self.spawned_actors:
            return []
        
        # Get ego vehicle transform and camera setup
        ego_transform = self.ego_vehicle.get_transform()
        
        # Calculate ego camera transform (same as EgoCamera and target point selection)
        camera_transform = carla.Transform(
            carla.Location(
                x=ego_transform.location.x + 2.0 * math.cos(math.radians(ego_transform.rotation.yaw)),
                y=ego_transform.location.y + 2.0 * math.sin(math.radians(ego_transform.rotation.yaw)),
                z=ego_transform.location.z + 1.8
            ),
            carla.Rotation(pitch=-8, yaw=ego_transform.rotation.yaw, roll=0)
        )
        
        # Get NPC vehicle locations (exclude ego vehicle and non-vehicles)
        npc_vehicles = []
        npc_locations = []
        for actor in self.spawned_actors:
            if (actor.type_id.startswith('vehicle.') and 
                actor.id != self.ego_vehicle.id and 
                actor != self.collision_sensor):
                npc_vehicles.append(actor)
                npc_locations.append(actor.get_location())
        
        if not npc_locations:
            self._print("  📍 No NPC vehicles found for visibility analysis")
            return []
        
        # Filter visible vehicles using camera frustum (same parameters as target point selection)
        visible_locations = filter_coordinates(
            camera_transform,
            npc_locations,
            fov_h=90,
            fov_v=60,
            max_distance=80
        )
        
        # Map visible locations back to vehicle IDs
        visible_vehicle_ids = []
        for i, location in enumerate(npc_locations):
            if location in visible_locations:
                vehicle_id = npc_vehicles[i].id
                visible_vehicle_ids.append(vehicle_id)
        
        return visible_vehicle_ids

    def _is_special_vehicle(self, blueprint: str) -> bool:
        """
        Check if a vehicle blueprint is a special vehicle that should not have custom colors.
        
        Args:
            blueprint: Vehicle blueprint string
            
        Returns:
            bool: True if it's a special vehicle (ambulance, police, sprinter)
        """
        return blueprint in self.SPECIAL_VEHICLES
    
    def _get_random_vehicle_color(self) -> str:
        """
        Get a random color name for a vehicle.
        
        Returns:
            str: Random color name from VEHICLE_COLORS list
        """
        return random.choice(self.VEHICLE_COLORS)
    
    def _apply_vehicle_color(self, blueprint, color_name: str) -> bool:
        """
        Apply color to a vehicle blueprint if the color exists in the mapping.
        
        Args:
            blueprint: CARLA vehicle blueprint object
            color_name: Natural language color name (e.g., "Red", "Blue")
            
        Returns:
            bool: True if color was applied, False otherwise
        """
        if color_name and color_name in self.COLOR_RGB_MAP:
            rgb_value = self.COLOR_RGB_MAP[color_name]
            blueprint.set_attribute('color', rgb_value)
            return True
        return False

    def _on_collision(self, event):
        """Callback on collision event"""
        actor_type = collision_actor_type(event)
        if not should_count_collision(event):
            self._print(f"Ignored collision with {actor_type}")
            return

        self._print("\n\n" + "="*60)
        self._print("  COLLISION DETECTED!")
        self._print(f"  - Collided with: {actor_type}")
        self._print(f"  - Collision impulse: {event.normal_impulse}")
        self._print("="*60 + "\n")
        self.collision_detected = True
        self.collision_events.append({
            "other_actor_type": actor_type,
            "normal_impulse": collision_impulse(event),
        })

    def _on_lane_invasion(self, event):
        """Callback on lane invasion event."""
        markings = []
        violation = False
        for marking in event.crossed_lane_markings:
            lane_change = str(marking.lane_change)
            markings.append({
                "type": str(marking.type),
                "color": str(marking.color),
                "lane_change": lane_change,
            })
            if lane_change.split(".")[-1].upper() == "NONE":
                violation = True

        if violation:
            self.lane_invasion_detected = True
        self.lane_invasion_events.append({"markings": markings, "violation": violation})

    def _build_oracle_events(self) -> Dict:
        events = {
            "collision": self.collision_detected,
            "stuck": self.stuck_detected,
            "lane_invasion": self.lane_invasion_detected,
            "speeding": self.speeding_detected,
            "timeout": self.timeout_detected,
            "timeout_frames": self.timeout_frames,
            "out_of_bounds": self.out_of_bounds_detected,
            "out_of_bounds_location": self.out_of_bounds_location,
            "other": self.other_error is not None,
            "other_error": self.other_error,
            "other_error_value": self.other_error_value,
            "collision_count": len(self.collision_events),
            "collision_events": list(self.collision_events),
            "lane_invasion_count": len(self.lane_invasion_events),
            "stuck_duration_frames": self.stuck_duration_frames,
            "enabled_checks": dict(self.oracle_checks),
        }
        triggered = []
        for key in ("collision", "stuck", "lane_invasion", "speeding", "timeout", "out_of_bounds", "other"):
            if self.oracle_checks.get(key, False) and events.get(key, False):
                triggered.append(key)
        events["triggered_checks"] = triggered
        events["oracle_failure"] = bool(triggered)
        if triggered:
            reasons = []
            for key in triggered:
                if key == "other" and self.other_error:
                    reasons.append(str(self.other_error))
                else:
                    reasons.append(key)
            events["failure_reason"] = ", ".join(reasons)
        return events
    
    def _validate_weather(self, weather: str) -> Tuple[bool, str]:
        """
        Validate weather parameter.
        
        Args:
            weather: Weather preset string
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            if not isinstance(weather, str):
                return False, "Weather must be a string"
            if weather not in self.WEATHER_PRESETS:
                return False, f"Invalid weather '{weather}'. Must be one of: {self.WEATHER_PRESETS}"
            return True, "Weather validation passed"
            
        except Exception as e:
            return False, f"Exception during weather validation: {str(e)}"
    
    def _apply_environment_settings(self, weather_config: Dict) -> bool:
        """
        Apply weather configuration to CARLA world.
        
        Args:
            weather_config: Weather configuration dictionary
            
        Returns:
            bool: True if successfully applied, False otherwise
        """
        try:
            if not weather_config:
                return True  # No weather config to apply
            
            self._print("🌤️  Applying weather settings...")
            
            # Check for weather parameter
            if "weather" in weather_config:
                weather_name = weather_config["weather"]
                self._print(f"   Using weather: {weather_name}")
                
                # Map weather name to CARLA WeatherParameters attribute
                weather_presets = {
                    'ClearNoon': carla.WeatherParameters.ClearNoon,
                    'CloudyNoon': carla.WeatherParameters.CloudyNoon,
                    'WetNoon': carla.WeatherParameters.WetNoon,
                    'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
                    'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
                    'HardRainNoon': carla.WeatherParameters.HardRainNoon,
                    'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
                    'ClearSunset': carla.WeatherParameters.ClearSunset,
                    'CloudySunset': carla.WeatherParameters.CloudySunset,
                    'WetSunset': carla.WeatherParameters.WetSunset,
                    'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
                    'MidRainSunset': carla.WeatherParameters.MidRainSunset,
                    'HardRainSunset': carla.WeatherParameters.HardRainSunset,
                    'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
                    'ClearNight': carla.WeatherParameters.ClearNight,
                    'CloudyNight': carla.WeatherParameters.CloudyNight,
                    'WetNight': carla.WeatherParameters.WetNight,
                    'WetCloudyNight': carla.WeatherParameters.WetCloudyNight,
                    'SoftRainNight': carla.WeatherParameters.SoftRainNight,
                    'MidRainyNight': carla.WeatherParameters.MidRainyNight,
                    'HardRainNight': carla.WeatherParameters.HardRainNight,
                    'DustStorm': carla.WeatherParameters.DustStorm
                }
                
                if weather_name in weather_presets:
                    weather = weather_presets[weather_name]
                else:
                    self._print(f"   ⚠️  Unknown weather '{weather_name}', using ClearNoon as fallback")
                    weather = carla.WeatherParameters.ClearNoon
                
                # Apply weather to world
                self.world.set_weather(weather)
                
                self._print(f"   ☁️  Cloudiness: {weather.cloudiness:.1f}%")
                self._print(f"   🌧️  Precipitation: {weather.precipitation:.1f}%")
                self._print(f"   💨 Wind: {weather.wind_intensity:.1f}%")
                self._print(f"   ☀️  Sun altitude: {weather.sun_altitude_angle:.1f}°")
                if weather.fog_density > 0:
                    self._print(f"   🌫️  Fog density: {weather.fog_density:.1f}%")
                if weather.wetness > 0:
                    self._print(f"   💧 Wetness: {weather.wetness:.1f}%")
                
                # Auto-control vehicle lights based on sun altitude and fog
                vehicle_lights_needed = weather.sun_altitude_angle < 15.0 or weather.fog_density > 30.0
                if vehicle_lights_needed and hasattr(self, 'spawned_actors') and self.spawned_actors:
                    self._print(f"   🚗 Auto-enabling vehicle lights (low sun/fog conditions)")
                    for actor in self.spawned_actors:
                        if actor.type_id.startswith('vehicle.'):
                            try:
                                lights = (carla.VehicleLightState.LowBeam | carla.VehicleLightState.Position)
                                actor.set_light_state(carla.VehicleLightState(lights))
                            except Exception:
                                pass  # Some vehicles might not support light control
                else:
                    self._print(f"   💡 Vehicle lights remain off (daylight conditions)")
                
                # Tick world to apply changes
                self.world.tick()
                self._print("   ✅ Weather settings applied successfully")
                return True
                
            else:
                # Fallback for any other format - use default weather
                self._print("   Using default ClearNoon weather")
                self.world.set_weather(carla.WeatherParameters.ClearNoon)
                self.world.tick()
                return True
            
        except Exception as e:
            self._print(f"   ❌ Failed to apply weather settings: {e}")
            return False
    
    def _apply_default_environment(self) -> bool:
        """
        Apply CARLA default environment settings to ensure clean state.
        
        Returns:
            bool: True if successfully applied, False otherwise
        """
        try:
            self._print("   🌤️  Applying CARLA default weather settings...")
            
            # Create default weather parameters (CARLA's default values)
            default_weather = carla.WeatherParameters.ClearNoon
            
            # Apply default weather to world
            self.world.set_weather(default_weather)
            
            # Reset vehicle lights to default state (off)
            if hasattr(self, 'spawned_actors') and self.spawned_actors:
                for actor in self.spawned_actors:
                    if actor.type_id.startswith('vehicle.'):
                        try:
                            actor.set_light_state(carla.VehicleLightState.NONE)
                        except Exception:
                            pass  # Some vehicles might not support light control
            
            # Tick world to apply changes
            self.world.tick()
            
            self._print("   ☀️  Clear weather with default settings")
            self._print("   💡 All vehicle lights turned off")
            self._print("   ✅ Default environment applied successfully")
            return True
            
        except Exception as e:
            self._print(f"   ❌ Failed to apply default environment: {e}")
            return False
    
    def generate_random_scenario(self, duration_frames: int = 600,
                                output_dir: str = None,
                                min_npc_count: int = 1,
                                max_npc_count: int = 8,
                                min_speed_diff_perc: float = -50.0,
                                max_speed_diff_perc: float = 50.0,
                                min_target_distance: int = 10,
                                max_route_distance: float = 60.0,
                                include_environment: bool = False) -> str:
        """
        Generate random dynamic scenario JSON file with target point selection and route distance filtering.
        
        Args:
            duration_frames: Number of frames for scenario duration (default: 600)
            output_dir: Directory to save the scenario file (default: current directory)
            min_npc_count: Minimum number of NPC vehicles (default: 1)
            max_npc_count: Maximum number of NPC vehicles (default: 8)
            min_speed_diff_perc: Minimum speed difference percentage for NPCs (default: -50.0)
            max_speed_diff_perc: Maximum speed difference percentage for NPCs (default: 50.0)
            min_target_distance: Minimum direct distance to target in meters (default: 10)
            max_route_distance: Maximum allowed route distance in meters (default: 60.0)
            include_environment: Whether to include random environment settings (default: False)
        """
        if not self.static_scenario_data:
            raise ValueError("Please load static scenario data first")

        available_spawn_points = len(self.static_scenario_data['spawn_points'])
        ego_spawn_num = self.static_scenario_data['ego_spawn_num']
        if available_spawn_points < min_npc_count + 1:
            raise ValueError(f"Insufficient spawn points for the requested number of NPCs.")
        
        max_possible_npc = min(max_npc_count, available_spawn_points - 1)
        npc_count = random.randint(min_npc_count, max_possible_npc)
        
        target_point = None
        route_info = None
        
        while target_point is None:
            ego_spawn_index = random.randint(0, ego_spawn_num - 1)
            
            all_spawn_indices = list(i for i in range(available_spawn_points) if i != ego_spawn_index)
            random.shuffle(all_spawn_indices)
            
            used_spawn_indices = all_spawn_indices[:npc_count]
            
            # Select visible target point with route distance filtering
            target_point, route_info = self._select_target_point(
                ego_spawn_index, 
                min_target_distance=min_target_distance,
                max_route_distance=max_route_distance
            )
        if not target_point:
            self._print("Could not find any visible target points with acceptable route distance for ego camera. Retrying...")
             
        ego_car_data = {
            "blueprint": self.EGO_VEHICLE_BLUEPRINT,
            "spawn_point_index": ego_spawn_index,
        }

        # NPC vehicles data with color assignment
        npc_vehicles_data = []
        for i, npc_spawn_index in enumerate(used_spawn_indices):
            random_speed_diff = random.uniform(min_speed_diff_perc, max_speed_diff_perc)
            npc_blueprint = random.choice(self.NPC_VEHICLE_BLUEPRINTS)
            
            npc_vehicle_data = {
                "blueprint": npc_blueprint,
                "spawn_point_index": npc_spawn_index,
                "speed_percentage_difference": random_speed_diff,
            }
            
            # Add color only for non-special vehicles
            if not self._is_special_vehicle(npc_blueprint):
                npc_vehicle_data["color"] = self._get_random_vehicle_color()
            
            npc_vehicles_data.append(npc_vehicle_data)

        # Construct a natural language instruction from the route description
        language_instruction = ""
        route_description = route_info['route_description']
        # Clean up the instructions to be more natural
        cleaned_instructions = []
        for inst in route_description:
            text = inst.lower()
            if "prepare to" in text:
                # "prepare to turn left at intersection" -> "turn left at the intersection"
                cleaned_instructions.append(text.replace("prepare to ", ""))
            elif "start changing" in text:
                # "start changing right 1 lane(s)" -> "change lane to the right"
                parts = text.split(' ')
                direction = parts[2]
                cleaned_instructions.append(f"change lane to the {direction}")
            elif "follow lane" in text:
                cleaned_instructions.append(f"follow current lane for a while")
            else:
                cleaned_instructions.append(text) # fallback

        language_instruction = " then ".join(cleaned_instructions)
        # Capitalize first letter and add a period
        language_instruction = language_instruction.capitalize() + "."

        scenario_data = {
            "duration_frames": duration_frames,
            "scenario_area": {"center": self.static_scenario_data['scenario_center'], "extent": self.static_scenario_data['scenario_extent']},
            "map_name": self.static_scenario_data['map_name'],
            "ego_spawn_num": ego_spawn_num,
            "ego_car": ego_car_data,
            "npc_vehicle_count": len(npc_vehicles_data),
            "npc_vehicles": npc_vehicles_data,
            "target_point": target_point,
            "route_info": {
                "route_length": route_info['route_length'],
                "route_waypoints": route_info['waypoints'],
                "route_description": route_info['route_description'],
                "basic_instruction": language_instruction
            }
        }
        
        # Add environment configuration if requested
        if include_environment:
            # Use the new simplified weather system
            weather = random.choice(self.WEATHER_PRESETS)
            scenario_data["environment"] = {"weather": weather}
            
            self._print(f"🌤️  Generated weather configuration:")
            self._print(f"  - Weather: {weather}")
        
        map_name = self.static_scenario_data['map_name'].split('/')[-1]
        if output_dir is None:
            output_path = f"dynamic_scenario_{map_name}.json"
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"dynamic_scenario_{map_name}.json")

        def round_floats(o):
            if isinstance(o, float):
                return round(o, 2)
            if isinstance(o, dict):
                return {k: round_floats(v) for k, v in o.items()}
            if isinstance(o, list):
                return [round_floats(v) for v in o]
            return o

        with open(output_path, 'w') as f:
            json.dump(round_floats(scenario_data), f, indent=4)

        self._print(f"Successfully generated enhanced dynamic scenario: {output_path}")
        self._print(f"  - Target point: ({target_point['x']:.1f}, {target_point['y']:.1f}, {target_point['z']:.1f})")
        self._print(f"  - Route length: {route_info['route_length']} meters")
        self._print(f"  - Route maneuvers: {' → '.join(set(route_info['maneuvers']))}")
        return output_path

    def _basic_instruction_from_route_description(self, route_description: List[str]) -> str:
        cleaned_instructions = []
        for inst in route_description:
            text = inst.lower()
            if "prepare to" in text:
                cleaned_instructions.append(text.replace("prepare to ", ""))
            elif "start changing" in text:
                parts = text.split(' ')
                direction = parts[2]
                cleaned_instructions.append(f"change lane to the {direction}")
            elif "follow lane" in text:
                cleaned_instructions.append("follow current lane for a while")
            else:
                cleaned_instructions.append(text)
        return " then ".join(cleaned_instructions).capitalize() + "."

    def generate_scenario_from_candidate(self,
                                         candidate: Dict,
                                         duration_frames: int = 600,
                                         output_dir: str = None,
                                         min_npc_count: int = 1,
                                         max_npc_count: int = 8,
                                         min_speed_diff_perc: float = -50.0,
                                         max_speed_diff_perc: float = 50.0,
                                         include_environment: bool = False,
                                         seed_generation_metadata: Optional[Dict] = None) -> str:
        """
        Generate a dynamic seed scenario from a pre-enumerated ego-target candidate.
        """
        if not self.static_scenario_data:
            raise ValueError("Please load static scenario data first")

        available_spawn_points = len(self.static_scenario_data['spawn_points'])
        ego_spawn_num = self.static_scenario_data['ego_spawn_num']
        ego_spawn_index = candidate["ego_spawn_index"]
        max_possible_npc = min(max_npc_count, available_spawn_points - 1)
        if max_possible_npc < min_npc_count:
            raise ValueError(f"Insufficient spawn points for the requested number of NPCs.")
        npc_count = random.randint(min_npc_count, max_possible_npc)

        all_spawn_indices = list(i for i in range(available_spawn_points) if i != ego_spawn_index)
        random.shuffle(all_spawn_indices)
        used_spawn_indices = all_spawn_indices[:npc_count]

        npc_vehicles_data = []
        for npc_spawn_index in used_spawn_indices:
            npc_blueprint = random.choice(self.NPC_VEHICLE_BLUEPRINTS)
            npc_vehicle_data = {
                "blueprint": npc_blueprint,
                "spawn_point_index": npc_spawn_index,
                "speed_percentage_difference": random.uniform(min_speed_diff_perc, max_speed_diff_perc),
            }
            if not self._is_special_vehicle(npc_blueprint):
                npc_vehicle_data["color"] = self._get_random_vehicle_color()
            npc_vehicles_data.append(npc_vehicle_data)

        route_info = candidate["route_info"]
        language_instruction = self._basic_instruction_from_route_description(route_info['route_description'])
        scenario_data = {
            "duration_frames": duration_frames,
            "scenario_area": {"center": self.static_scenario_data['scenario_center'], "extent": self.static_scenario_data['scenario_extent']},
            "map_name": self.static_scenario_data['map_name'],
            "ego_spawn_num": ego_spawn_num,
            "ego_car": {
                "blueprint": self.EGO_VEHICLE_BLUEPRINT,
                "spawn_point_index": ego_spawn_index,
            },
            "npc_vehicle_count": len(npc_vehicles_data),
            "npc_vehicles": npc_vehicles_data,
            "target_point": candidate["target_point"],
            "route_info": {
                "route_length": route_info['route_length'],
                "route_waypoints": route_info['waypoints'],
                "route_description": route_info['route_description'],
                "basic_instruction": language_instruction,
            },
            "seed_generation": seed_generation_metadata or {},
        }
        if include_environment:
            scenario_data["environment"] = {"weather": random.choice(self.WEATHER_PRESETS)}

        map_name = self.static_scenario_data['map_name'].split('/')[-1]
        if output_dir is None:
            output_path = f"dynamic_scenario_{map_name}.json"
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"dynamic_scenario_{map_name}.json")

        def round_floats(o):
            if isinstance(o, float):
                return round(o, 2)
            if isinstance(o, dict):
                return {k: round_floats(v) for k, v in o.items()}
            if isinstance(o, list):
                return [round_floats(v) for v in o]
            return o

        with open(output_path, 'w') as f:
            json.dump(round_floats(scenario_data), f, indent=4)

        self._print(f"Successfully generated diverse dynamic scenario: {output_path}")
        return output_path
    

    def generate_scenario_from_seed(self, seed_scenario_path: str,
                                   npc_count: int,
                                   output_dir: str = None,
                                   min_speed_diff_perc: float = -50.0,
                                   max_speed_diff_perc: float = 50.0,
                                   include_environment: bool = True,
                                   return_dict: bool = False) -> Union[str, Dict]:
        """
        Generate a complete scenario from a seed scenario file by adding NPCs and weather.
        
        Args:
            seed_scenario_path: Path to the seed scenario JSON file
            npc_count: Number of NPC vehicles to add
            output_dir: Directory to save the new scenario file (default: current directory)
            min_speed_diff_perc: Minimum speed difference percentage for NPCs (default: -50.0)
            max_speed_diff_perc: Maximum speed difference percentage for NPCs (default: 50.0)
            include_environment: Whether to include random environment settings (default: True)
            return_dict: If True, return the scenario data dictionary instead of saving to file
            
        Returns:
            str: Path to the generated scenario file (if return_dict=False)
            Dict: The generated scenario data (if return_dict=True)
        """
        try:
            # Load the seed scenario
            self._print(f"Loading seed scenario: {seed_scenario_path}")
            with open(seed_scenario_path, 'r') as f:
                seed_data = json.load(f)
            
            # Validate seed scenario has required fields
            required_fields = ['ego_car', 'target_point', 'route_info', 'map_name', 'scenario_area', 'ego_spawn_num']
            for field in required_fields:
                if field not in seed_data:
                    raise ValueError(f"Seed scenario missing required field: {field}")
            
            # Check if we have enough spawn points for the requested NPCs
            if not self.static_scenario_data:
                raise ValueError("Please load static scenario data first")
            
            available_spawn_points = len(self.static_scenario_data['spawn_points'])
            ego_spawn_num = self.static_scenario_data['ego_spawn_num']
            ego_spawn_index = seed_data['ego_car']['spawn_point_index']
            
            # Calculate available spawn points for NPCs (excluding ego spawn point)
            available_npc_spawns = [i for i in range(available_spawn_points) if i != ego_spawn_index]
            
            if len(available_npc_spawns) < npc_count:
                raise ValueError(f"Not enough spawn points for {npc_count} NPCs. Available: {len(available_npc_spawns)}")
            
            # Randomly select NPC spawn points
            random.shuffle(available_npc_spawns)
            selected_npc_spawns = available_npc_spawns[:npc_count]
            
            # Generate NPC vehicles data
            npc_vehicles_data = []
            spawn_points = self.static_scenario_data['spawn_points']
            
            for i, npc_spawn_index in enumerate(selected_npc_spawns):
                random_speed_diff = random.uniform(min_speed_diff_perc, max_speed_diff_perc)
                npc_spawn_point = spawn_points[npc_spawn_index]
                
                npc_vehicle_data = {
                    "blueprint": random.choice(self.NPC_VEHICLE_BLUEPRINTS),
                    "spawn_point_index": npc_spawn_index,
                    "speed_percentage_difference": random_speed_diff,
                }
                
                # Add color if not a special vehicle
                if not self._is_special_vehicle(npc_vehicle_data["blueprint"]):
                    npc_vehicle_data["color"] = self._get_random_vehicle_color()
                    
                npc_vehicles_data.append(npc_vehicle_data)
            
            # Create the new scenario data based on seed
            new_scenario_data = {
                "duration_frames": seed_data.get('duration_frames', 600),
                "scenario_area": seed_data['scenario_area'],
                "map_name": seed_data['map_name'],
                "ego_spawn_num": seed_data['ego_spawn_num'],
                "ego_car": seed_data['ego_car'].copy(),
                "npc_vehicle_count": len(npc_vehicles_data),
                "npc_vehicles": npc_vehicles_data,
                "target_point": seed_data['target_point'],
                "route_info": seed_data['route_info']
            }
            
            # Add environment configuration if requested
            if include_environment:
                # Use the new simplified weather system
                weather = random.choice(self.WEATHER_PRESETS)
                new_scenario_data["weather"] = weather
                
                self._print(f"🌤️  Generated weather configuration:")
                self._print(f"  - Weather: {weather}")
                
            # Round floats for cleaner JSON/Dict
            def round_floats(o):
                if isinstance(o, float):
                    return round(o, 2)
                if isinstance(o, dict):
                    return {k: round_floats(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [round_floats(v) for v in o]
                return o
            
            rounded_data = round_floats(new_scenario_data)

            if return_dict:
                self._print(f"✅ Successfully generated enhanced scenario data (dictionary)")
                self._print(f"  - Base seed: {os.path.basename(seed_scenario_path)}")
                self._print(f"  - Added NPCs: {npc_count}")
                return rounded_data
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed_filename = os.path.splitext(os.path.basename(seed_scenario_path))[0]
            map_name = seed_data['map_name'].split('/')[-1]
            
            if output_dir is None:
                output_path = f"{seed_filename}_enhanced_{npc_count}npc_{timestamp}.json"
            else:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{seed_filename}_enhanced_{npc_count}npc_{timestamp}.json")
            
            # Save the enhanced scenario
            with open(output_path, 'w') as f:
                json.dump(rounded_data, f, indent=4)
            
            self._print(f"✅ Successfully generated enhanced scenario from seed: {output_path}")
            self._print(f"  - Base seed: {os.path.basename(seed_scenario_path)}")
            self._print(f"  - Added NPCs: {npc_count}")
            self._print(f"  - Environment: {'Included' if include_environment else 'Not included'}")
            self._print(f"  - Map: {map_name}")
            self._print(f"  - Target point: ({new_scenario_data['target_point']['x']:.1f}, {new_scenario_data['target_point']['y']:.1f}, {new_scenario_data['target_point']['z']:.1f})")
            self._print(f"  - Route length: {new_scenario_data['route_info']['route_length']}m")
            
            return output_path
            
        except Exception as e:
            self._print(f"❌ Failed to generate scenario from seed: {e}")
            raise
    
    def _capture_ego_image(self, output_dir="scenario_images") -> Optional[str]:
        """Capture ego camera image with target point label"""
        if not self.ego_camera:
            return None
        
        # Get target point in absolute coordinates if available
        target_point_abs = None
        if self.scenario_config and self.scenario_config.target_point:
            target_point_abs = self._convert_relative_to_absolute(self.scenario_config.target_point)
        
        # Capture image with target point marking
        image_path = self.ego_camera.capture_image(target_point=target_point_abs, output_dir=output_dir)
        return image_path
    
    def _calculate_segment_length(self, route_segment):
        """Helper function: Calculate the length of a route segment (list)"""
        segment_length = 0.0
        if len(route_segment) < 2:
            return 0.0
        for i in range(len(route_segment) - 1):
            loc1 = route_segment[i][0].transform.location
            loc2 = route_segment[i+1][0].transform.location
            segment_length += loc1.distance(loc2)
        return segment_length

    def _is_route_valid(self, route_info):
        """
        Check if a route is valid based on several criteria:
        1. Atomicity: At most one lane change block and one intersection block.
        2. Complexity: Not too simple (must contain a turn or lane change).
        3. Lane Change: Any lane change must be a single lane change.

        Args:
            route_info: Dictionary containing route information.

        Returns:
            bool: True if the route is valid, False otherwise.
        """
        if not route_info or 'maneuvers' not in route_info:
            return False

        maneuvers = route_info['maneuvers']
        route_description = route_info.get('route_description', [])

        if not maneuvers:
            return False

        # 1. Atomicity Check
        lane_change_blocks = 0
        intersection_blocks = 0
        i = 0
        while i < len(maneuvers):
            current_maneuver = maneuvers[i]
            j = i
            while j < len(maneuvers) and maneuvers[j] == current_maneuver:
                j += 1
            
            if current_maneuver in ['CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT']:
                lane_change_blocks += 1
            elif current_maneuver in ['LEFT', 'RIGHT', 'STRAIGHT']:
                intersection_blocks += 1
            i = j

        if not (lane_change_blocks <= 1 and intersection_blocks <= 1):
            return False

        # 2. Complexity Check
        interesting_maneuvers = {'LEFT', 'RIGHT', 'CHANGE_LANE_LEFT', 'CHANGE_LANE_RIGHT'}
        if not any(m in interesting_maneuvers for m in maneuvers):
            return False

        # 3. Single Lane Change Check
        for description in route_description:
            if "lane(s)" in description:
                try:
                    parts = description.split(' ')
                    if len(parts) > 2 and parts[-1] == "lane(s)":
                        num_lanes = int(parts[-2])
                        if num_lanes > 1:
                            return False
                except (ValueError, IndexError):
                    pass

        return True

    def _generate_route_description(self, route):
        """
        Generate accurate aggregated and summarized driving strategy log in English.
        Can correctly count lane changes.
        """
        if not route:
            return ["Unable to generate route!"]

        log_messages = []
        i = 0
        while i < len(route):
            current_maneuver = route[i][1]
            
            # Find the end position of current consecutive command block
            j = i
            while j < len(route) and route[j][1] == current_maneuver:
                j += 1
            
            segment = route[i:j]
            start_waypoint = segment[0][0]
            end_waypoint = segment[-1][0]  # Last waypoint of the segment
            
            if current_maneuver == RoadOption.LANEFOLLOW:
                distance = self._calculate_segment_length(segment)
                if distance > 10.0:
                    log_messages.append(f"Follow lane for {distance:.2f} meters")
            
            elif current_maneuver in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]:
                # --- Core correction logic ---
                # Get lane ID before lane change
                # If lane change is the first action in the route, can't get i-1, use first waypoint ID of segment as start
                start_lane_id = route[i - 1][0].lane_id if i > 0 else start_waypoint.lane_id
                
                # Get lane ID after lane change completion
                end_lane_id = end_waypoint.lane_id
                
                # Calculate lane ID difference
                lanes_changed = abs(end_lane_id - start_lane_id)
                
                # Only record when actual lane ID change occurs
                if lanes_changed > 0:
                    direction = "left" if current_maneuver == RoadOption.CHANGELANELEFT else "right"
                    log_messages.append(f"Start changing {direction} {lanes_changed} lane(s)")
            
            elif current_maneuver in [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]:
                action_map = { 
                    RoadOption.LEFT: "turn left", 
                    RoadOption.RIGHT: "turn right", 
                    RoadOption.STRAIGHT: "go straight" 
                }
                log_messages.append(f"Prepare to {action_map[current_maneuver]} at intersection")

            i = j
            
        return log_messages

    def _calculate_route_info(self, ego_location, target_location, verbose=True):
        """
        Calculate route information between two locations.
        
        Args:
            ego_location: carla.Location - Starting location
            target_location: carla.Location - Target location
            verbose: bool - Whether to print detailed info
            
        Returns:
            dict: Dictionary containing:
                - 'route_length': Total route distance in meters (float)
                - 'maneuvers': List of human-readable maneuver strings (List[str])
                - 'waypoints': List of waypoint coordinates (List[Dict])
                - 'route_description': List of human-readable driving instructions (List[str])
            None: If error occurs or route cannot be found
        """
        try:
            if verbose:
                self._print(f"🗺️  Planning route...")
                self._print(f"   Start location: ({ego_location.x:.2f}, {ego_location.y:.2f}, {ego_location.z:.2f})")
                self._print(f"   Target location: ({target_location.x:.2f}, {target_location.y:.2f}, {target_location.z:.2f})")
            
            # Create route planner and trace route
            carla_map = self.world.get_map()
            grp = GlobalRoutePlanner(carla_map, 2.0)
            route = grp.trace_route(ego_location, target_location)
            
            if not route:
                if verbose:
                    self._print("❌ Could not find route between locations")
                return None
            
            # Calculate route length by summing distances between consecutive waypoints
            route_length = 0.0
            for i in range(len(route) - 1):
                current_waypoint = route[i][0]
                next_waypoint = route[i + 1][0]
                distance = current_waypoint.transform.location.distance(next_waypoint.transform.location)
                route_length += distance
            
            # Process maneuvers - convert RoadOption enums to strings
            maneuver_mapping = {
                RoadOption.VOID: "VOID",
                RoadOption.LEFT: "LEFT", 
                RoadOption.RIGHT: "RIGHT",
                RoadOption.STRAIGHT: "STRAIGHT",
                RoadOption.LANEFOLLOW: "FOLLOW_LANE",
                RoadOption.CHANGELANELEFT: "CHANGE_LANE_LEFT",
                RoadOption.CHANGELANERIGHT: "CHANGE_LANE_RIGHT"
            }
            
            processed_maneuvers = []
            waypoints_data = []
            
            for i, (waypoint, maneuver) in enumerate(route):
                maneuver_str = maneuver_mapping.get(maneuver, f"UNKNOWN_{maneuver}")
                processed_maneuvers.append(maneuver_str)
                
                # Store waypoint information
                waypoint_info = {
                    'index': i,
                    'location': {
                        'x': round(waypoint.transform.location.x, 2),
                        'y': round(waypoint.transform.location.y, 2),
                        'z': round(waypoint.transform.location.z, 2)
                    },
                    'rotation': {
                        'yaw': round(waypoint.transform.rotation.yaw, 2),
                        'pitch': round(waypoint.transform.rotation.pitch, 2),
                        'roll': round(waypoint.transform.rotation.roll, 2)
                    },
                    'maneuver': maneuver_str
                }
                waypoints_data.append(waypoint_info)
            
            # Generate route description using the new method
            route_description = self._generate_route_description(route)
            
            result = {
                'route_length': round(route_length, 2),
                'maneuvers': processed_maneuvers,
                'waypoints': waypoints_data,
                'route_description': route_description
            }
            
            if verbose:
                self._print(f"✅ Route analysis complete:")
                self._print(f"   📏 Total route length: {result['route_length']} meters")
                self._print(f"   🔄 Maneuver sequence: {' → '.join(set(result['maneuvers']))}")
                self._print(f"   📍 Total waypoints: {len(waypoints_data)}")
                self._print(f"   📋 Route description: {len(route_description)} instructions")
                for desc in route_description:
                    self._print(f"      • {desc}")
            
            return result
            
        except Exception as e:
            if verbose:
                self._print(f"❌ Error calculating route info: {e}")
                import traceback
                traceback.print_exc()
            return None

    def get_maneuver(self, ego_location=None, target_location=None):
        """
        Get maneuver information for the route from ego vehicle to target point.
        
        Args:
            ego_location: carla.Location (optional) - Override ego location
            target_location: carla.Location (optional) - Override target location
        
        Returns:
            dict: Dictionary containing:
                - 'route_length': Total route distance in meters (float)
                - 'maneuvers': List of human-readable maneuver strings (List[str])
                - 'waypoints': List of waypoint coordinates (List[Dict])
            None: If error occurs or prerequisites not met
        """
        try:
            # Use provided locations or get from current scenario state
            if ego_location is None or target_location is None:
                if not self.scenario_config:
                    self._print("Error: Please load scenario configuration first")
                    return None
                
                if not self.spawned_actors:
                    self._print("Error: No vehicles spawned yet")
                    return None
                
                if not self.scenario_config.target_point:
                    self._print("Error: No target point defined in scenario")
                    return None
                
                # Get ego vehicle information
                ego_vehicle = self.ego_vehicle                
                ego_transform = ego_vehicle.get_transform()
                ego_location = ego_transform.location
                target_point_abs = self._convert_relative_to_absolute(self.scenario_config.target_point)
                target_location = carla.Location(x=target_point_abs['x'], y=target_point_abs['y'], z=target_point_abs['z'])
            
            return self._calculate_route_info(ego_location, target_location, verbose=True)
            
        except Exception as e:
            self._print(f"❌ Error getting maneuver info: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_scenario(self, scenario_source: Union[str, Dict]) -> bool:
        """
        Load dynamic scenario from JSON file path or Dictionary and perform validation.
        """
        try:
            if isinstance(scenario_source, str):
                self._print(f"Loading dynamic scenario: {scenario_source}")
                with open(scenario_source, 'r') as f:
                    scenario_data = json.load(f)
            elif isinstance(scenario_source, dict):
                # self._print(f"Loading dynamic scenario from dictionary")
                scenario_data = scenario_source
            else:
                self._print(f"❌ Error: Invalid scenario source type: {type(scenario_source)}")
                return False
            
            validation_result = self._validate_scenario_data(scenario_data)
            if not validation_result[0]:
                self._print(f"❌ Scenario data validation failed: {validation_result[1]}")
                return False
            self._print("✓ Scenario data validation passed")

            # Extract only the core fields for VehicleConfig (backward compatibility)
            ego_car_core = {
                'blueprint': scenario_data['ego_car']['blueprint'],
                'spawn_point_index': scenario_data['ego_car']['spawn_point_index'],
                'speed_percentage_difference': scenario_data['ego_car'].get('speed_percentage_difference'),
                'spawn_point_coordinates': scenario_data['ego_car'].get('spawn_point_coordinates')
            }
            ego_config = VehicleConfig(**ego_car_core)
            
            npc_configs = []
            for npc in scenario_data['npc_vehicles']:
                npc_core = {
                    'blueprint': npc['blueprint'],
                    'spawn_point_index': npc['spawn_point_index'],
                    'speed_percentage_difference': npc['speed_percentage_difference'],
                    'spawn_point_coordinates': npc.get('spawn_point_coordinates'),
                    'color': npc.get('color')  # Include color parameter
                }
                npc_configs.append(VehicleConfig(**npc_core))
            
            # Load target point if available
            target_point = scenario_data.get('target_point')
            route_info = scenario_data.get('route_info')
            
            # Load weather configuration if available
            weather = scenario_data.get('weather')
            if weather:
                weather_validation = self._validate_weather(weather)
                if not weather_validation[0]:
                    self._print(f"❌ Weather validation failed: {weather_validation[1]}")
                    return False
                self._print("✓ Weather validation passed")
            
            duration_frames = scenario_data['duration_frames']
            if self.scenario_duration_frames is not None:
                duration_frames = int(self.scenario_duration_frames)

            self.scenario_config = ScenarioConfig(
                duration_frames=duration_frames,
                scenario_area=scenario_data['scenario_area'],
                ego_car=ego_config,
                npc_vehicle_count=scenario_data['npc_vehicle_count'],
                npc_vehicles=npc_configs,
                target_point=target_point,
                route_info=route_info,
                weather=weather
            )
            
            source_name = scenario_source if isinstance(scenario_source, str) else "Dictionary Source"
            self._print(f"\n🎉 Successfully loaded dynamic scenario: {source_name}")
            self._print(f"  - Duration: {self.scenario_config.duration_frames} frames")
            self._print(f"  - Total vehicles: {1 + len(self.scenario_config.npc_vehicles)} (1 Ego + {len(self.scenario_config.npc_vehicles)} NPC)")
            
            if target_point:
                self._print(f"  - Target point: ({target_point['x']:.1f}, {target_point['y']:.1f}, {target_point['z']:.1f})")

            if route_info:
                self._print(f"  - Route length: {route_info.get('route_length', 'N/A')}m")
            
            for i, npc in enumerate(self.scenario_config.npc_vehicles):
                self._print(f"    - NPC {i+1}: spawn {npc.spawn_point_index} (Speed Diff: {npc.speed_percentage_difference:.1f}%)")

            return True

        except Exception as e:
            self._print(f"❌ Failed to load dynamic scenario: {e}")
            return False
    
    def spawn_vehicles(self, tm_enable=True) -> bool:
        """
        Spawn all vehicles and hand them over to the Traffic Manager.
        """
        if not self.scenario_config or not self.static_scenario_data:
            self._print("Error: Scenario configuration not loaded.")
            return False

        spawn_points = self.static_scenario_data['spawn_points']
        blueprint_library = self.world.get_blueprint_library()
        successfully_spawned_actors = []

        try:
            # --- 1. Spawn Ego vehicle ---
            self._print("\n--- Spawning Ego Vehicle ---")
            ego_config = self.scenario_config.ego_car
            ego_spawn_point = spawn_points[ego_config.spawn_point_index]
            ego_absolute_spawn = self._convert_relative_to_absolute(ego_spawn_point)
            ego_transform = carla.Transform(
                carla.Location(x=ego_absolute_spawn['x'], y=ego_absolute_spawn['y'], z=ego_absolute_spawn['z']),
                carla.Rotation(yaw=ego_absolute_spawn['yaw'])
            )
            ego_bp = blueprint_library.find(self.EGO_VEHICLE_BLUEPRINT)
            ego_bp.set_attribute('color', '255,0,0')

            ego_vehicle = self.world.try_spawn_actor(ego_bp, ego_transform)
            if ego_vehicle is None:
                self._print("❌ FATAL: Failed to spawn the ego vehicle. Aborting.")
                return False
            
            successfully_spawned_actors.append(ego_vehicle)
            self._print(f"✓ Ego vehicle successfully spawned.")

            self.ego_vehicle = ego_vehicle

            # --- 2. Spawn NPC vehicles ---
            self._print("\n--- Spawning NPC Vehicles ---")
            for i, npc_config in enumerate(self.scenario_config.npc_vehicles):
                npc_spawn_point = spawn_points[npc_config.spawn_point_index]
                npc_absolute_spawn = self._convert_relative_to_absolute(npc_spawn_point)
                npc_transform = carla.Transform(
                    carla.Location(x=npc_absolute_spawn['x'], y=npc_absolute_spawn['y'], z=npc_absolute_spawn['z']),
                    carla.Rotation(yaw=npc_absolute_spawn['yaw'])
                )

                npc_bp = blueprint_library.find(npc_config.blueprint)
                
                # Apply color if specified and not a special vehicle
                color_applied = False
                if hasattr(npc_config, 'color') and npc_config.color and not self._is_special_vehicle(npc_config.blueprint):
                    color_applied = self._apply_vehicle_color(npc_bp, npc_config.color)
                
                npc_vehicle = self.world.try_spawn_actor(npc_bp, npc_transform)

                if npc_vehicle is None:
                    self._print(f"⚠️ WARNING: Could not spawn NPC {i + 1} ({npc_config.blueprint}). Skipping.")
                    continue

                successfully_spawned_actors.append(npc_vehicle)
                self.actor_id_to_spawn_index[npc_vehicle.id] = npc_config.spawn_point_index
                
                # Enhanced logging with color information
                color_info = f" (Color: {npc_config.color})" if color_applied else ""
                special_info = " [Special Vehicle - No Color]" if self._is_special_vehicle(npc_config.blueprint) else ""
                self._print(f"✓ NPC {i + 1} ({npc_config.blueprint}){color_info}{special_info} successfully spawned.")
                
                if tm_enable:
                    npc_vehicle.set_autopilot(True, self.tm_port)
                    self.traffic_manager.vehicle_percentage_speed_difference(npc_vehicle, npc_config.speed_percentage_difference)
                    self._print(f"✓ NPC {i + 1} ({npc_config.blueprint}) handed to TM. Speed diff: {npc_config.speed_percentage_difference:.1f}%")
            
            self.spawned_actors = successfully_spawned_actors
            self._print(f"\nAll possible vehicles spawned. Total: {len(self.spawned_actors)}.")
            return True

        except Exception as e:
            self._print(f"An unexpected error occurred during vehicle spawning: {e}")
            import traceback
            traceback.print_exc()
            return False 

    def capture_image(self, image_dir="scenario_images", freeze_the_scenario=False) -> bool:
        """
        Captures scenario image from the ego camera.
        """
        if not self.scenario_config:
            self._print("Error: Please load scenario configuration first")
            return False

        self._print("\n1. Setting spectator position...")
        self._move_spectator_to_scenario_center()

        self._print("\n2. Spawning vehicles...")
        if not self.spawn_vehicles(tm_enable=False):
            self.cleanup_actors()
            return False

        # --- Attach ego camera ---
        self._print("\n3. Attaching Ego Camera...")
        self.ego_camera = EgoCamera(self.world, self.ego_vehicle)

        ego_image_path = None
        if self.ego_camera:
            self._print("\n4. Capturing ego camera image...")
            ego_image_path = self._capture_ego_image(output_dir=image_dir)

        if ego_image_path:
            self._print(f"  ✅ Ego camera image captured: {ego_image_path}")
        else:
            self.cleanup_actors()
            return False
        
        if freeze_the_scenario:
            self._print("\n" + "="*60)
            self._print("Scenario preparation complete! Vehicles are FROZEN.")
            self._print("Press [Enter] to exit...")
            self._print("="*60)

            try:
                input()
            except KeyboardInterrupt:
                self._print("\nUser cancelled scenario scenario")

        self.cleanup_actors()
        return True

    def _get_execution_result(self, error=False, error_message=None, crashed=False, completed=False, actual_frames_executed=0, min_distance=None, ettc_stats=None, path_deviation_stats=None, ego_speed_stats=None, npc_vehicles=None, driving_quality_stats=None, oracle_events=None):
        """
        Get execution result with comprehensive monitoring statistics.
        
        Args:
            error: bool - Whether an error occurred during execution
            error_message: str - Detailed error message if error occurred
            crashed: bool - Whether a collision was detected
            completed: bool - Whether the scenario completed successfully
            actual_frames_executed: int - Number of frames actually executed
            min_distance: float - Minimum distance to target point (optional)
            ettc_stats: dict - ETTC statistics (optional)
            path_deviation_stats: dict - Path deviation statistics (optional)
            ego_speed_stats: dict - Ego vehicle speed statistics (optional)
            npc_vehicles: dict - Visible vehicle distances statistics (optional)
            driving_quality_stats: dict - Driving quality statistics (optional)
            oracle_events: dict - Unified test oracle events and enabled checks (optional)
            
        Returns:
            dict: Execution result containing scenario outcome and comprehensive monitoring data
        """
        result = {
            'error': error,
            'collision': crashed,
            'completed': completed,
            'actual_frames_executed': actual_frames_executed
        }
        
        if error and error_message:
            result['error_message'] = str(error_message)
        elif error:
            result['error_message'] = "Unknown error occurred during scenario execution"

        if min_distance is not None:
            result['min_distance'] = round(min_distance, 2)

        if ettc_stats:
            result['ettc_stats'] = ettc_stats

        if path_deviation_stats:
            result['path_deviation_stats'] = path_deviation_stats
            
        if ego_speed_stats:
            result['ego_speed_stats'] = ego_speed_stats

        if driving_quality_stats:
            result['driving_quality_stats'] = driving_quality_stats

        if oracle_events:
            result['oracle_events'] = oracle_events
            result['oracle_failure'] = oracle_events.get('oracle_failure', False)
            if oracle_events.get('failure_reason'):
                result['oracle_failure_reason'] = oracle_events['failure_reason']
            
        if npc_vehicles:
            result['npc_vehicles'] = npc_vehicles
            
        return result

    def _lock_traffic_lights(self):
        """
        Lock all traffic lights to green state and freeze them.
        This ensures consistent traffic flow regardless of Traffic Manager resets.
        """
        try:
            self._print("🚦 Setting traffic lights to green and freezing them...")
            traffic_light_actors = self.world.get_actors().filter('traffic.traffic_light*')
            
            if not traffic_light_actors:
                self._print("   ⚠️  No traffic lights found in the world")
                return
            
            for traffic_light in traffic_light_actors:
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.set_green_time(1000.0)
                traffic_light.freeze(True)
            
            self.world.tick()
            self._print(f"   ✅ {len(traffic_light_actors)} traffic lights locked to green")
            
        except Exception as e:
            self._print(f"   ❌ Failed to lock traffic lights: {e}")

    def _reset_traffic_manager(self):
        """
        Reset and reseed the Traffic Manager to ensure consistent NPC behavior across scenario runs.
        This addresses the issue where NPC vehicles have inconsistent path selection between runs.
        """
        self._print("\n🔄 Resetting Traffic Manager...")
        
        try:
            # Get a fresh traffic manager instance
            self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
            
            # Reset the random device seed to ensure deterministic behavior
            if self.tm_seed is not None:
                self.traffic_manager.set_random_device_seed(self.tm_seed)
                self._print(f"   ✅ Traffic Manager reseeded with: {self.tm_seed}")
            else:
                self._print("   ⚠️  No seed provided, Traffic Manager will use default randomization")
            
            # Ensure synchronous mode is maintained
            self.traffic_manager.set_synchronous_mode(True)
            
            self._print("   ✅ Traffic Manager reset completed")
            
        except Exception as e:
            self._print(f"   ❌ Failed to reset Traffic Manager: {e}")
            raise

    def start_scenario(self, language_instruction, success_distance=3.0) -> Dict[str, bool]:
        """
        Prepares and starts the scenario execution.
        Returns:
            Dict: A dictionary containing scenario execution results defined in _get_execution_result().
        """
        if self.agent_instance is None:
            raise ValueError("No VLA model is loaded. Initialize CarlaScenario with model_name to execute scenarios.")

        # Initialize stats variables early to ensure safety in finally block
        ettc_stats = None
        path_deviation_stats = None
        driving_quality_stats = None
        ego_speed_stats = None
        npc_vehicles = None
        completed = False
        scenario_failed = False
        error_message = None
        infrastructure_error = None
        actual_frames_executed = 0
        min_distance = float('inf')

        if not self.scenario_config:
            self._print("Error: Please load scenario configuration first")
            return self._get_execution_result(error=True)
            
        # Ensure synchronous mode is set up before starting scenario logic
        # This is now the canonical place to enable it
        self._setup_synchronous_mode()

        self._print("\n1. Resetting and reseeding Traffic Manager...")
        self._reset_traffic_manager()

        self._print("\n1.1. Re-locking traffic lights after TM reset...")
        self._lock_traffic_lights()

        self._print("\n2. Setting spectator position...")
        self._move_spectator_to_scenario_center()

        self._print("\n3. Spawning vehicles...")
        if not self.spawn_vehicles(tm_enable=True):
            self.cleanup_actors()
            return self._get_execution_result(error=True)

        # --- Apply Environment Settings ---
        self._print("\n4. Applying environment settings...")
        if self.scenario_config.weather:
            # Apply the weather configuration directly
            weather_config = {"weather": self.scenario_config.weather}
            if not self._apply_environment_settings(weather_config):
                self._print("⚠️  Warning: Failed to apply weather settings, falling back to default weather")
                self._apply_default_environment()
        else:
            self._print("   No weather configuration found, applying CARLA default weather...")
            self._apply_default_environment()

        # --- Setup Collision Sensor ---
        self._print("\n5. Setting up collision sensor...")
        self.collision_detected = False
        self.collision_events = []
        self.lane_invasion_detected = False
        self.lane_invasion_events = []
        self.speeding_detected = False
        self.stuck_detected = False
        self.stuck_duration_frames = 0
        self.timeout_detected = False
        self.timeout_frames = None
        self.out_of_bounds_detected = False
        self.out_of_bounds_location = None
        self.other_error = None
        self.other_error_value = None
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(self._on_collision)
        self.spawned_actors.append(self.collision_sensor)
        self._print("  ✅ Collision sensor attached")

        self._print("\n5.1. Setting up lane invasion sensor...")
        lane_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.lane_sensor.listen(self._on_lane_invasion)
        self.spawned_actors.append(self.lane_sensor)
        self._print("  ✅ Lane invasion sensor attached")

        if self.scenario_config.route_info and 'route_waypoints' in self.scenario_config.route_info:
            
            # Inverse mapping from maneuver string to RoadOption enum
            maneuver_mapping = {
                "VOID": RoadOption.VOID,
                "LEFT": RoadOption.LEFT,
                "RIGHT": RoadOption.RIGHT,
                "STRAIGHT": RoadOption.STRAIGHT,
                "FOLLOW_LANE": RoadOption.LANEFOLLOW,
                "CHANGE_LANE_LEFT": RoadOption.CHANGELANELEFT,
                "CHANGE_LANE_RIGHT": RoadOption.CHANGELANERIGHT
            }
            
            reconstructed_route = []
            waypoints_data = self.scenario_config.route_info['route_waypoints']
            
            for waypoint_info in waypoints_data:
                loc_data = waypoint_info['location']
                location = carla.Location(x=loc_data['x'], y=loc_data['y'], z=loc_data['z'])
                
                # Get the waypoint from the map
                waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
                
                if waypoint is None:
                    self._print(f"   ⚠️  Warning: Could not find waypoint for location: {location}")
                    continue
                
                maneuver_str = waypoint_info['maneuver']
                road_option = maneuver_mapping.get(maneuver_str, RoadOption.VOID)
                
                reconstructed_route.append((waypoint, road_option))
            
            if reconstructed_route:
                world_coord_route = reconstructed_route
                self._print(f"   ✅ Route reconstructed successfully with {len(world_coord_route)} waypoints.")
            else:
                self._print("   ❌ Route reconstruction failed.")

        reset_agent_episode_state(self.agent_instance)
        self.agent_instance.user_command = language_instruction
        self.agent_instance.dreamer_flag = False
        route = []
        gps_route = []

        # Convert the route to GPS format, as expected by the agent
        lat_ref, lon_ref = _get_latlon_ref(self.world)

        for waypoint, connection in world_coord_route:
            route.append((waypoint.transform, connection))
            gps_coord = _location_to_gps(lat_ref, lon_ref, waypoint.transform.location)
            gps_route.append((gps_coord, connection))

        self.agent_instance.set_global_plan(gps_route, route)
        self.agent_instance.set_gps_reference(lat_ref, lon_ref)
        
        # 6. Setup Sensors
        self.agent_wrapper = AgentWrapperFactory.get_wrapper(self.agent_instance)
        self.agent_wrapper.setup_sensors(self.ego_vehicle)
        
        self.world.tick() # tick to register sensors
        
        self._print("\n🚀 Starting scenario execution...")
        
        # Initialize ETTC tracking
        self.ettc_metrics.reset()
        self.ettc_metrics.set_max_ttc(self.scenario_config.duration_frames, self.frame_rate)
        
        # Initialize path deviation tracking
        self.path_deviation_metrics.reset()
        if self.scenario_config.route_info and 'route_waypoints' in self.scenario_config.route_info:
            self.path_deviation_metrics.set_target_path(self.scenario_config.route_info['route_waypoints'])
            self._print(f"  📍 Path deviation metrics initialized with {len(self.scenario_config.route_info['route_waypoints'])} target waypoints")

        # Initialize driving quality tracking
        self.driving_quality_metrics.reset()
        
        # Initialize speed and vehicle distance tracking
        self._print("\n7. Initializing monitoring variables...")
        self.ego_speed_history = []
        self.ego_max_speed = 0.0
        self.visible_vehicle_ids = self._identify_visible_vehicles()
        self.vehicle_min_distances = {}
        self.npc_vehicle_info = {}  # Reset NPC info to prevent accumulation
        
        # Initialize detailed NPC vehicle information for visible vehicles
        if self.visible_vehicle_ids:
            ego_location = self.ego_vehicle.get_location()
            ego_transform = self.ego_vehicle.get_transform()
            ego_yaw = ego_transform.rotation.yaw
            
            for actor in self.spawned_actors:
                if actor.id in self.visible_vehicle_ids:
                    # Get stable spawn index ID
                    spawn_index = self.actor_id_to_spawn_index.get(actor.id)
                    if spawn_index is None:
                        continue # Should not happen for NPCs
                        
                    initial_distance = ego_location.distance(actor.get_location())
                    
                    # Calculate initial yaw difference between ego and NPC vehicle
                    npc_transform = actor.get_transform()
                    npc_yaw = npc_transform.rotation.yaw
                    
                    # Calculate the absolute difference in yaw angles
                    yaw_diff = abs(ego_yaw - npc_yaw)
                    # Normalize to [0, 180] range (handle angle wrapping)
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    
                    # Get vehicle color and name
                    vehicle_color = None
                    vehicle_name = self._get_vehicle_name(actor.type_id)
                    
                    # Find color from scenario config
                    for npc_config in self.scenario_config.npc_vehicles:
                        if hasattr(npc_config, 'color') and npc_config.color and actor.type_id == npc_config.blueprint:
                            vehicle_color = npc_config.color
                            break
                    
                    # Use spawn_index (as string) as the stable key
                    self.npc_vehicle_info[str(spawn_index)] = {
                        'blueprint': actor.type_id,
                        'spawn_index': spawn_index, # Store explicitly too
                        'actor_id': actor.id,       # Store transient ID for reference
                        'name': vehicle_name,
                        'color': vehicle_color,
                        'initial_distance': round(initial_distance, 2),
                        'initial_yaw_diff': round(yaw_diff, 2),
                        'min_distance': initial_distance
                    }
                    
                    color_info = f" ({vehicle_color})" if vehicle_color else ""
                    self._print(f"  📏 Vehicle SpawnIdx:{spawn_index} (ID:{actor.id}, {vehicle_name}{color_info}) - Initial distance: {initial_distance:.2f}m")
        
        try:
            # Get target point in absolute coordinates
            target_location = None
            target_location = None
            if self.scenario_config.target_point:
                target_point_abs = self._convert_relative_to_absolute(self.scenario_config.target_point)
                target_location = carla.Location(x=target_point_abs['x'], y=target_point_abs['y'], z=target_point_abs['z'])

            print(f"Language instruction: {language_instruction}")
            
            for frame_count in range(self.scenario_config.duration_frames):
                self.world.tick()
                timestamp = self.world.get_snapshot().timestamp
                GameTime.on_carla_tick(timestamp)
                
                actual_frames_executed = frame_count + 1  # Track actual frames executed
                
                # Move spectator to follow vehicle
                spectator = self.world.get_spectator()
                transform = self.ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(transform.location - 10 * transform.get_forward_vector() + carla.Location(z=5), transform.rotation))
                
                # The agent's __call__ method reads from the sensor interface
                control = self.agent_wrapper()
                self.ego_vehicle.apply_control(control)
                
                current_speed = self.ego_vehicle.get_velocity()
                speed_magnitude = math.sqrt(current_speed.x**2 + current_speed.y**2 + current_speed.z**2)
                speed_kmh = 3.6 * speed_magnitude

                # Do not terminate solely because the ego vehicle is waiting at
                # low speed. Timeout handles non-completion without penalizing
                # legitimate waits such as red lights.
                self.stuck_duration_frames = 0

                speed_limit = self.ego_vehicle.get_speed_limit()
                if speed_limit and speed_kmh > speed_limit * (1.0 + self.speed_tolerance):
                    self.speeding_detected = True

                # Track speed statistics
                self.ego_speed_history.append(speed_kmh)
                self.ego_max_speed = max(self.ego_max_speed, speed_kmh)

                # Track distances to visible vehicles
                ego_location = self.ego_vehicle.get_location()
                for actor in self.spawned_actors:
                    if actor.id in self.visible_vehicle_ids:
                        spawn_index = self.actor_id_to_spawn_index.get(actor.id)
                        if spawn_index is not None:
                            str_idx = str(spawn_index)
                            if str_idx in self.npc_vehicle_info:
                                distance = ego_location.distance(actor.get_location())
                                self.npc_vehicle_info[str_idx]['min_distance'] = min(self.npc_vehicle_info[str_idx]['min_distance'], distance)

                # Calculate ETTC for current frame
                frame_ettc = self.ettc_metrics.calculate_frame_ettc(self.ego_vehicle, self.spawned_actors)

                # Calculate path deviation for current frame
                frame_deviation = self.path_deviation_metrics.calculate_frame_deviation(
                    self.ego_vehicle, timestamp.elapsed_seconds
                )

                # Calculate driving quality telemetry for current frame
                self.driving_quality_metrics.record_frame(
                    self.ego_vehicle,
                    control,
                    spawned_actors=self.spawned_actors,
                    timestamp=timestamp.elapsed_seconds,
                )

                # Early exit checks (in priority order)
                # 1. Collision detected
                if self.collision_detected:
                    self._print("\n\n Scenario terminated due to collision.")
                    break

                # 2. Check if ego vehicle is out of scenario bounds
                if not self._is_ego_within_scenario_bounds():
                    ego_location = self.ego_vehicle.get_location()
                    self._print(f"\n\n Scenario terminated: Ego vehicle drove out of scenario bounds.")
                    self._print(f"   Ego position: ({ego_location.x:.2f}, {ego_location.y:.2f}, {ego_location.z:.2f})")
                    self._print(f"   Scenario bounds: Center({self.static_scenario_data['scenario_center']['x']:.2f}, {self.static_scenario_data['scenario_center']['y']:.2f}), Extent({self.static_scenario_data['scenario_extent']['x']:.2f}x{self.static_scenario_data['scenario_extent']['y']:.2f})")
                    self.out_of_bounds_detected = True
                    self.out_of_bounds_location = {
                        "x": round(ego_location.x, 2),
                        "y": round(ego_location.y, 2),
                        "z": round(ego_location.z, 2),
                    }
                    break

                # 3. Check target reached
                if target_location:
                    ego_location = self.ego_vehicle.get_location()
                    distance_to_target = ego_location.distance(target_location)
                    
                    if distance_to_target < min_distance:
                        min_distance = distance_to_target
                        
                    # Display progress with ETTC information
                    if frame_ettc < self.ettc_metrics.max_ttc:
                        print(f"Step {frame_count}/{self.scenario_config.duration_frames} - Speed: {speed_kmh:.2f} km/h - Dist: {distance_to_target:.2f}m - ETTC: {frame_ettc:.2f}s", end='\r')
                    else:
                        print(f"Step {frame_count}/{self.scenario_config.duration_frames} - Speed: {speed_kmh:.2f} km/h - Dist: {distance_to_target:.2f}m - ETTC: Safe", end='\r')
                    
                    if distance_to_target < success_distance:
                        self._print(f"\n\n Success! Ego vehicle reached the target area (distance < {success_distance}m).")
                        self._print(" Scenario execution completed successfully!")
                        completed = True
                        break
                else:
                    raise ValueError("No target point defined in scenario configuration.")
            
            # Calculate ETTC statistics
            ettc_stats = self.ettc_metrics.get_ettc_statistics()
            self._print(f"\n📊 ETTC Statistics:")
            self._print(f"   Minimum ETTC: {ettc_stats['min_ettc']}s")
            self._print(f"   Average ETTC: {ettc_stats['avg_ettc']}s")
            self._print(f"   Frames with finite TTC: {ettc_stats['danger_frames']}")
            
            # Calculate path deviation statistics
            path_deviation_stats = None
            if len(self.path_deviation_metrics.frame_deviations) > 0:
                path_deviation_stats = self.path_deviation_metrics.get_comprehensive_metrics(self.ego_vehicle)
                
                if self.collision_detected:
                    path_deviation_stats['path_tracking_quality'] = 0.0
                
                self._print(f"\n📍 Path Deviation Statistics:")
                self._print(f"   Average lateral deviation: {path_deviation_stats['avg_lateral_deviation']:.3f}m")
                self._print(f"   Maximum deviation: {path_deviation_stats['max_deviation']:.3f}m")
                self._print(f"   Minimum deviation: {path_deviation_stats['min_deviation']:.3f}m")
                self._print(f"   Deviation standard deviation: {path_deviation_stats['std_deviation']:.3f}m")
                self._print(f"   Path completion ratio: {path_deviation_stats['path_completion_ratio']:.1%}")
                self._print(f"   Deviation severity: {path_deviation_stats['deviation_severity']:.3f}")
                self._print(f"   Path tracking quality: {path_deviation_stats['path_tracking_quality']:.3f}")
                self._print(f"   Cumulative deviation area: {path_deviation_stats['cumulative_deviation_area']:.3f}m²")

            # Calculate driving quality statistics
            driving_quality_stats = self.driving_quality_metrics.get_statistics()
            self._print(f"\n🕹️ Driving Quality Statistics:")
            self._print(f"   Hard acceleration frames: {driving_quality_stats['hard_acceleration_count']}")
            self._print(f"   Hard braking frames: {driving_quality_stats['hard_braking_count']}")
            self._print(f"   Hard turn frames: {driving_quality_stats['hard_turn_count']}")
            self._print(f"   Oversteer-like frames: {driving_quality_stats['oversteer_count']}")
            self._print(f"   Understeer-like frames: {driving_quality_stats['understeer_count']}")
            self._print(f"   Control oscillations: {driving_quality_stats['control_oscillation_count']}")
            self._print(f"   Max yaw rate: {driving_quality_stats['max_yaw_rate']:.2f} deg/s")
            self._print(f"   Driving quality score: {driving_quality_stats['driving_quality_score']:.3f}")
            
            # Calculate ego speed statistics
            ego_speed_stats = None
            if len(self.ego_speed_history) > 0:
                self.ego_avg_speed = sum(self.ego_speed_history) / len(self.ego_speed_history)
                ego_speed_stats = {
                    'average_speed': round(self.ego_avg_speed, 2),
                    'max_speed': round(self.ego_max_speed, 2),
                    'speed_history': [round(speed, 2) for speed in self.ego_speed_history]
                }
                self._print(f"\n🚗 Ego Speed Statistics:")
                self._print(f"   Average speed: {ego_speed_stats['average_speed']:.2f} km/h")
                self._print(f"   Maximum speed: {ego_speed_stats['max_speed']:.2f} km/h")
                self._print(f"   Total speed samples: {len(self.ego_speed_history)}")

            # Process NPC vehicle information with detailed statistics
            npc_vehicles = self.npc_vehicle_info  # Already formatted correctly with stable keys
            
            if npc_vehicles:
                self._print(f"\n🚙 NPC Vehicle Statistics:")
                for spawn_idx, data in npc_vehicles.items():
                    color_info = f" ({data['color']})" if data.get('color') else ""
                    self._print(f"   Vehicle SpawnIdx:{spawn_idx} ({data['name']}{color_info}):")
                    self._print(f"     Blueprint: {data['blueprint']}")
                    self._print(f"     Initial distance: {data['initial_distance']:.2f}m")
                    self._print(f"     Initial yaw difference: {data['initial_yaw_diff']:.1f}°")
                    self._print(f"     Minimum distance: {data['min_distance']:.2f}m")

            if (not completed and not self.collision_detected and not self.stuck_detected
                    and not self.out_of_bounds_detected and not self.timeout_detected):
                self.timeout_detected = True
                self.timeout_frames = actual_frames_executed

            if self.timeout_detected:
                self._print(f"\n\n Scenario timed out. min distance to target: {min_distance:.2f} m")
            elif completed:
                self._print(f"\n\n Scenario execution completed! min distance to target: {min_distance:.2f} m")

        except KeyboardInterrupt:
            self._print("\n\n⚠️  User interrupted scenario execution")
            scenario_failed = True
            error_message = "User interrupted execution (KeyboardInterrupt)"
            self.other_error = "execution_error"
            self.other_error_value = error_message
        except Exception as e:
            if is_carla_infrastructure_error(e):
                infrastructure_error = CarlaInfrastructureError(str(e))
            else:
                self._print(f"\n\n❌ Scenario execution failed: {e}")
                scenario_failed = True
                error_message = str(e)
                self.other_error = "execution_error"
                self.other_error_value = error_message
        finally:
            if infrastructure_error is not None:
                # The batch supervisor will terminate the degraded CARLA process;
                # additional cleanup RPCs would only delay recovery.
                raise infrastructure_error
            oracle_events = self._build_oracle_events()
            self.cleanup_actors()
            return self._get_execution_result(error=scenario_failed, error_message=error_message, crashed=self.collision_detected, 
                                              completed=completed, ettc_stats=ettc_stats, 
                                              path_deviation_stats=path_deviation_stats,
                                              driving_quality_stats=driving_quality_stats,
                                              ego_speed_stats=ego_speed_stats,
                                              npc_vehicles=npc_vehicles,
                                              oracle_events=oracle_events,
                                              actual_frames_executed=actual_frames_executed, min_distance=min_distance)
        
    def cleanup_actors(self):
        """
        Clean up only the spawned actors and camera for the current scenario.
        """
        self._print("\n🧹 Starting scenario actor cleanup...")
        
        # Clean up agent wrapper sensors
        if self.agent_wrapper:
            try:
                self.agent_wrapper.cleanup()
                self.agent_wrapper = None
                self._print("✅ Agent wrapper sensors destroyed")
            except Exception as e:
                self._print(f"Warning: Error destroying agent wrapper sensors: {e}")

        # Clean up collision sensor
        if self.collision_sensor:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
                self.collision_sensor = None
                self._print("✅ Collision sensor destroyed")
            except Exception as e:
                self._print(f"Warning: Error destroying collision sensor: {e}")

        if self.lane_sensor:
            try:
                self.lane_sensor.stop()
                self.lane_sensor.destroy()
                self.lane_sensor = None
                self._print("✅ Lane invasion sensor destroyed")
            except Exception as e:
                self._print(f"Warning: Error destroying lane invasion sensor: {e}")
        
        # Clean up ego camera
        if self.ego_camera:
            try:
                self.ego_camera.destroy()
                self.ego_camera = None
                self._print("✅ Ego camera destroyed")
            except Exception as e:
                self._print(f"Warning: Error destroying ego camera: {e}")

        # Clean up spawned actors
        self.ego_vehicle = None
        if self.client and self.spawned_actors:
            actor_ids = [actor.id for actor in self.spawned_actors if actor and actor.is_alive]
            if actor_ids:
                self.client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in actor_ids])
                self._print(f"✅ {len(actor_ids)} vehicles destroyed")
                
                # CRITICAL: Tick the world once to ensure server processes the destruction
                # This prevents "ghost" actors and segmentation faults in subsequent runs
                try:
                    if not self.world.get_settings().synchronous_mode:
                        self.world.wait_for_tick(seconds=0.5)
                    else:
                        self.world.tick()
                except Exception as e:
                    self._print(f"Warning: Failed to tick world after destruction: {e}")
        
        self.spawned_actors.clear()
        
        gc.collect()
        torch.cuda.empty_cache()
        self._print("✅ Scenario actor cleanup completed")

    def _get_vehicle_name(self, blueprint_id: str) -> str:
        """
        Convert CARLA blueprint ID to human-friendly vehicle name.
        
        Args:
            blueprint_id: CARLA blueprint ID (e.g., 'vehicle.audi.a2')
            
        Returns:
            str: Human-friendly name (e.g., 'Audi A2')
        """
        # Define mapping of blueprint IDs to vehicle names
        blueprint_to_name = {
            'vehicle.audi.a2': 'Audi A2',
            'vehicle.mini.cooper_s': 'Mini Cooper S',
            'vehicle.seat.leon': 'Seat Leon',
            'vehicle.bmw.grandtourer': 'BMW Gran Tourer',
            'vehicle.audi.tt': 'Audi TT',
            'vehicle.ford.ambulance': 'Ford Ambulance',
            'vehicle.dodge.charger_police': 'Dodge Charger Police',
            'vehicle.mercedes.sprinter': 'Mercedes Sprinter',
            'vehicle.tesla.model3': 'Tesla Model 3',
            # Add more mappings as needed
        }
        
        # Return vehicle name if found, otherwise format blueprint ID
        if blueprint_id in blueprint_to_name:
            return blueprint_to_name[blueprint_id]
        else:
            # Fallback: convert blueprint ID to readable format
            # e.g., 'vehicle.brand.model' -> 'Brand Model'
            parts = blueprint_id.split('.')
            if len(parts) >= 3:
                brand = parts[1].capitalize()
                model = parts[2].replace('_', ' ').title()
                return f"{brand} {model}"
            else:
                return blueprint_id

    def destroy(self):
        """
        Destroy all actors and disconnect from the simulator by disabling sync mode.
        This instance should not be used after calling this method.
        """
        self.cleanup_actors()

        self._print("\n🧹 Starting final resource cleanup...")
        try:
            settings = self.world.get_settings()
            if settings.synchronous_mode:
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                self.traffic_manager.set_synchronous_mode(False)
                self._print("✅ Synchronous mode disabled")
        except Exception as e:
            self._print(f"Warning: Error resetting synchronous mode: {e}")

        self._print("✅ Final resource cleanup completed")

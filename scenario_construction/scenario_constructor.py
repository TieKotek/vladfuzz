import carla
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional
import json
import os
import sys
import queue
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_construction.scenario_regions import load_region_specs, select_region_specs

class ScenarioConstructor:
    def __init__(
        self,
        host='127.0.0.1',
        port=2000,
        map_name: Optional[str] = None,
        timeout: float = 30.0,
        map_warmup_seconds: float = 5.0,
        warmup_ticks: int = 5,
    ):
        """
        Initialize the CARLA client and connect to the world.
        """
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.timeout = timeout
        self.map_warmup_seconds = map_warmup_seconds
        self.warmup_ticks = warmup_ticks
        self.spawned_actors = []
        
        if map_name:
            self.world = self._load_world(map_name)
        else:
            self.world = self.client.get_world()
            self._prepare_world_after_switch()
            
        self.carla_map = self.world.get_map()
        print(f"Using map: {self.carla_map.name}")

    def _load_world(self, map_name: str):
        required_map_name = map_name.split("/")[-1]
        current_world = self.client.get_world()
        current_map_name = current_world.get_map().name.split("/")[-1]

        if current_map_name == required_map_name:
            print(f"Map already loaded: {current_world.get_map().name}")
            self.world = current_world
            self._prepare_world_after_switch()
            return self.world

        print(f"Loading map: {required_map_name}")
        world = self.client.load_world(required_map_name)
        self.world = world
        self._prepare_world_after_switch()

        actual_map_name = self.world.get_map().name.split("/")[-1]
        if actual_map_name != required_map_name:
            raise RuntimeError(f"Map switch failed: expected {required_map_name}, got {self.world.get_map().name}")
        return self.world

    def _prepare_world_after_switch(self):
        self._force_async_mode()
        if self.map_warmup_seconds > 0:
            print(f"Waiting {self.map_warmup_seconds:.1f}s for map to stabilize...")
            time.sleep(self.map_warmup_seconds)
        for _ in range(max(1, self.warmup_ticks)):
            self._tick_or_wait(seconds=min(5.0, self.timeout))

    def _force_async_mode(self):
        try:
            settings = self.world.get_settings()
            if settings.synchronous_mode:
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                print("Reset CARLA world to asynchronous mode for scenario construction.")
        except Exception as e:
            print(f"Warning: failed to reset CARLA synchronous mode: {e}")

        try:
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(False)
        except Exception as e:
            print(f"Warning: failed to reset Traffic Manager synchronous mode: {e}")

    def _tick_or_wait(self, seconds: float = 5.0):
        settings = self.world.get_settings()
        if settings.synchronous_mode:
            return self.world.tick()
        return self.world.wait_for_tick(seconds=seconds)
    
    def construct_scenario(self, center_x: float, center_y: float, 
                             extent_x: float, extent_y: float,
                             min_spawn_distance: float = 10.0,
                             min_global_distance: float = 2.0,
                             output_dir: str = "scenario_output",
                             waypoint_sample_dist: float = 1.0,
                             exclude_junction=False,
                             vehicle_filter='vehicle.tesla.model3',
                             scenario_type='intersection',
                             scenario_id: Optional[str] = None,
                             highway_split_ratios: Tuple[float, float, float] = (0.5, 0, 0.5)):
        """
        Constructs a scenario description file and various visualizations for a specified area.
        Now includes automated spawn verification, classification (spawn/target), and filtering.

        Args:
            center_x (float): X coordinate of the scenario area's center.
            center_y (float): Y coordinate of the scenario area's center.
            extent_x (float): Half-width of the scenario area in the X-direction.
            extent_y (float): Half-height of the scenario area in the Y-direction.
            min_spawn_distance (float): The minimum longitudinal distance between points on the SAME lane.
            min_global_distance (float): The minimum distance between ANY two spawn points globally.
            output_dir (str): Directory to save the output files (JSON and PNG).
            waypoint_sample_dist (float): Sampling distance (in meters) for collecting waypoints.
            exclude_junction (bool): Switch for junction exclusion.
            vehicle_filter (str): Vehicle blueprint filter for spawn verification.
            scenario_type (str): Type of scenario (e.g., 'intersection', 'highway').
            highway_split_ratios (Tuple[float, float, float]): Ratio of (spawn, blank, target) areas for highway scenarios.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        bbox = {
            'min_x': center_x - extent_x, 'max_x': center_x + extent_x,
            'min_y': center_y - extent_y, 'max_y': center_y + extent_y,
            'center': {'x': center_x, 'y': center_y},
            'extent': {'x': extent_x, 'y': extent_y}
        }
        
        print(f"Constructing scenario for map '{self.carla_map.name}'...")
        print(f"  Area Center: ({center_x:.2f}, {center_y:.2f})")
        print(f"  Min Spawn Distance (per lane): {min_spawn_distance}m")
        print(f"  Min Global Distance (any two points): {min_global_distance}m")
        
        # Step 1: Capture initial CARLA BEV image
        output_stem = scenario_id or self.carla_map.name.split('/')[-1]
        carla_bev_path = self._capture_carla_bev_image(bbox, output_dir, output_stem)
        
        # Step 2: Get waypoints and create initial spawn points
        waypoints_in_area = self._get_waypoints_in_bbox(bbox, waypoint_sample_dist)
        print(f"Found {len(waypoints_in_area)} waypoints in the area.")
        
        initial_scenario_data = self._create_spawnable_lanes_data(
            waypoints_in_area, bbox, min_spawn_distance, min_global_distance, exclude_junction
        )
        
        # Step 3: Verify spawn points by spawning vehicles
        print("\n--- Starting Spawn Verification ---")
        verified_spawn_points = self._verify_spawn_points(initial_scenario_data, vehicle_filter)
        
        # Step 3.5: Classify verified points into spawn and target sets
        print("\n--- Classifying Verified Points ---")
        classified_points = self._classify_and_split_points(
            center_x, center_y,
            verified_spawn_points, waypoints_in_area, highway_split_ratios, scenario_type
        )

        # Step 4: Create final scenario data with classified points
        final_scenario_data = self._create_final_scenario_data(initial_scenario_data, classified_points, bbox)
        
        # Step 5: Generate matplotlib BEV with classified spawn points
        mpl_bev_path = self._generate_mpl_bev_image(waypoints_in_area, final_scenario_data, bbox, output_dir, output_stem)
        
        # Step 6: Save final scenario JSON
        scenario_json_path = self._save_scenario_json(final_scenario_data, output_dir, output_stem)
        
        # Step 7: Cleanup spawned vehicles
        self._cleanup_spawned_actors()
        
        print("\n" + "="*50)
        print("Scenario construction complete!")
        print(f"  -> Final Scenario JSON: {scenario_json_path}")
        print(f"  -> CARLA Rendered BEV Image: {carla_bev_path}")
        print(f"  -> Matplotlib BEV Image: {mpl_bev_path}")
        print("="*50)
        
        return {
            'scenario_json_path': scenario_json_path,
            'carla_bev_image_path': carla_bev_path,
            'mpl_bev_path': mpl_bev_path,
        }

    def _create_spawnable_lanes_data(self, waypoints: List[carla.Waypoint], bbox: Dict, 
                                     min_lane_dist: float, min_global_dist: float, exclude_junction: bool) -> Dict:
        # (This method requires no changes)
        lanes_raw = {}
        for wp in waypoints:
            if wp.lane_type != carla.LaneType.Driving:
                continue
            lane_key = f"road_{wp.road_id}_lane_{wp.lane_id}"
            if lane_key not in lanes_raw:
                lanes_raw[lane_key] = []
            lanes_raw[lane_key].append(wp)

        all_confirmed_spawn_points = []
        spawnable_lanes_list = []
        
        for lane_id, lane_wps in lanes_raw.items():
            sorted_wps = sorted(lane_wps, key=lambda w: w.s)
            if not sorted_wps:
                continue

            lane_spawn_points = []
            last_spawn_s = -float('inf')
            
            for wp in sorted_wps:
                if exclude_junction and wp.is_junction:
                    continue
                if wp.s < last_spawn_s + min_lane_dist:
                    continue
                
                current_point = carla.Location(wp.transform.location.x, wp.transform.location.y, wp.transform.location.z)
                
                too_close_to_existing = False
                for existing_point in all_confirmed_spawn_points:
                    if current_point.distance(existing_point) < min_global_dist:
                        too_close_to_existing = True
                        break
                
                if not too_close_to_existing:
                    spawn_point_data = {
                        'x': wp.transform.location.x, 'y': wp.transform.location.y, 'z': wp.transform.location.z,
                        'yaw': wp.transform.rotation.yaw,
                    }
                    lane_spawn_points.append(spawn_point_data)
                    all_confirmed_spawn_points.append(current_point)
                    last_spawn_s = wp.s
            
            if lane_spawn_points:
                spawnable_lanes_list.append({
                    'lane_id': lane_id,
                    'spawn_points': lane_spawn_points,
                    'spawn_point_count': len(lane_spawn_points)
                })

        total_spawn_points = len(all_confirmed_spawn_points)
        print(f"Generated {total_spawn_points} initial spawn points (distance & junction constraints applied).")
        return {
            'map_name': self.carla_map.name,
            'scenario_center': bbox['center'],
            'scenario_extent': bbox['extent'],
            'lanes': spawnable_lanes_list
        }

    def _verify_spawn_points(self, scenario_data: Dict, vehicle_filter: str) -> List[Dict]:
        blueprint_library = self.world.get_blueprint_library()
        car_bp = self._resolve_spawn_verification_blueprint(blueprint_library, vehicle_filter)

        verified_spawn_points = []
        spawn_count = 0
        total_points = sum(lane['spawn_point_count'] for lane in scenario_data['lanes'])
        if total_points == 0:
            print("No verifiable spawn points found.")
            return []
            
        print(f"Verifying {total_points} spawn points by spawning vehicles...")

        for lane in scenario_data['lanes']:
            for sp in lane['spawn_points']:
                vehicle, transform = self._try_spawn_verification_vehicle(sp, car_bp)
                if vehicle is not None and transform is not None:
                    self.spawned_actors.append(vehicle)
                    spawn_count += 1
                    verified_point = sp.copy()
                    verified_point['x'] = transform.location.x
                    verified_point['y'] = transform.location.y
                    verified_point['z'] = transform.location.z
                    verified_point['yaw'] = transform.rotation.yaw
                    verified_spawn_points.append(verified_point)
        
        print(f"Successfully verified {spawn_count} / {total_points} spawn points ({spawn_count/total_points*100:.1f}% success rate).")
        return verified_spawn_points

    def _resolve_spawn_verification_blueprint(self, blueprint_library, vehicle_filter: str):
        try:
            car_bp = blueprint_library.find(vehicle_filter)
        except IndexError:
            raise RuntimeError(f"Vehicle blueprint '{vehicle_filter}' not found.")
        print(f"Using '{car_bp.id}' for spawn verification.")
        return car_bp

    def _try_spawn_verification_vehicle(self, spawn_point: Dict, blueprint):
        base_location = carla.Location(x=spawn_point['x'], y=spawn_point['y'], z=spawn_point['z'])
        waypoint = self.carla_map.get_waypoint(
            base_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        base_transform = waypoint.transform if waypoint is not None else carla.Transform(
            base_location,
            carla.Rotation(yaw=spawn_point['yaw']),
        )

        for z_offset in (0.5, 1.0, 1.5):
            transform = carla.Transform(
                carla.Location(
                    x=base_transform.location.x,
                    y=base_transform.location.y,
                    z=base_transform.location.z + z_offset,
                ),
                base_transform.rotation,
            )
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                return vehicle, transform
        return None, None

    def _classify_and_split_points(self, center_x, center_y, verified_points: List[Dict], all_waypoints: List[carla.Waypoint],
                                     highway_split_ratios: Tuple[float, float, float], scenario_type: str) -> Dict:
        """
        Automatically classifies the area and splits points into spawn and target sets.
        For highway scenarios, it uses a custom ratio to create spawn, blank, and target zones.

        Args:
            verified_points (List[Dict]): The list of spawn points that have been successfully verified.
            all_waypoints (List[carla.Waypoint]): All waypoints in the bounding box, used for classification.
            highway_split_ratios (Tuple[float, float, float]): The ratio of (spawn, blank, target) areas.
            scenario_type (str): The type of scenario (e.g., 'intersection', 'highway').
        """
        # 1. Automatically identify scenario type
        junction_wps = [wp for wp in all_waypoints if wp.is_junction]
        if not all_waypoints: return {'spawn_points': [], 'target_points': []} # Safety check
        
        junction_ratio = len(junction_wps) / len(all_waypoints)
        # This threshold can be adjusted as needed
        print(f"Detected Scenario Type: {scenario_type} (Junction Waypoint Ratio: {junction_ratio:.2f})")

        spawn_points = []
        target_points = []

        if not verified_points:
            return {'spawn_points': [], 'target_points': []}

        if scenario_type == 'intersection':
            # Method A: Handling intersections
            if not junction_wps:
                print("Warning: Scene identified as intersection but no junction waypoints found. Treating all points as spawn points.")
                return {'spawn_points': verified_points, 'target_points': []}
            
            # Calculate Junction center
            # junc = junction_wps[0].get_junction()
            # junc_center = junc.bounding_box.location
            
            junc_center = carla.Location(center_x, center_y, 0) # Approximate center
            
            for sp in verified_points:
                loc = carla.Location(sp['x'], sp['y'], sp['z'])
                wp = self.carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                future_wp_list = wp.next(1.5) # Look 1.5 meters ahead
                
                if not future_wp_list: continue # End of the lane
                future_wp = future_wp_list[0]
                
                dist_current = wp.transform.location.distance(junc_center)
                dist_future = future_wp.transform.location.distance(junc_center)

                # If the future point is closer to the center, it's an entry point (spawn)
                if dist_future < dist_current - 0.1: # Subtract a small tolerance for floating point errors
                    spawn_points.append(sp)
                else:
                    target_points.append(sp)
        else: # scenario_type == 'highway'
            print(f"Applying highway splitting with ratio (Spawn:Blank:Target): {highway_split_ratios}")
            total_ratio = sum(highway_split_ratios)
            if total_ratio <= 0:
                raise ValueError("The sum of highway_split_ratios must be positive.")

            # Calculate dominant axis (same as before)
            yaws_rad = [np.deg2rad(p['yaw']) for p in verified_points]
            vectors = [np.array([np.cos(yaw), np.sin(yaw)]) for yaw in yaws_rad]
            dominant_axis = np.mean(vectors, axis=0)
            dominant_axis /= np.linalg.norm(dominant_axis) # Normalize

            # Calculate progress of each point along the dominant axis and sort
            points_with_progress = []
            for sp in verified_points:
                progress = np.dot([sp['x'], sp['y']], dominant_axis)
                points_with_progress.append((sp, progress))
            
            points_with_progress.sort(key=lambda item: item[1])
            
            # Calculate split indices based on the provided ratios
            num_points = len(points_with_progress)
            spawn_ratio, blank_ratio, _ = highway_split_ratios
            
            # Index after which spawn points end
            split_index_1 = int(num_points * (spawn_ratio / total_ratio))
            # Index after which the blank area ends (and target points begin)
            split_index_2 = int(num_points * ((spawn_ratio + blank_ratio) / total_ratio))

            spawn_points = [item[0] for item in points_with_progress[:split_index_1]]
            # The middle section is the blank/buffer area and is intentionally ignored
            target_points = [item[0] for item in points_with_progress[split_index_2:]]

        print(f"Classification complete: {len(spawn_points)} spawn points, {len(target_points)} target points.")
        return {'spawn_points': spawn_points, 'target_points': target_points}

    def _create_final_scenario_data(self, initial_data: Dict, classified_points: Dict, bbox: Dict) -> Dict:
        """
        Creates the final scenario data structure with classified points.
        """
        spawn_points = classified_points.get('spawn_points', [])
        target_points = classified_points.get('target_points', [])
        
        for point in spawn_points:
            point["x"] -= bbox["center"]["x"]
            point["y"] -= bbox["center"]["y"]
            
        for point in target_points:
            point["x"] -= bbox["center"]["x"]
            point["y"] -= bbox["center"]["y"]   

        ego_spawn_num = len(spawn_points)
        
        return {
            'map_name': initial_data['map_name'],
            'scenario_center': bbox['center'],
            'scenario_extent': bbox['extent'],
            'ego_spawn_num': ego_spawn_num,
            'spawn_points': spawn_points + target_points,
        }

    def _cleanup_spawned_actors(self):
        # (This method requires no changes)
        if self.spawned_actors:
            print(f"Cleaning up {len(self.spawned_actors)} spawned vehicles...")
            self.client.apply_batch([carla.command.DestroyActor(actor) for actor in self.spawned_actors])
            self.spawned_actors = []
            try:
                self._tick_or_wait(seconds=1.0)
            except Exception as e:
                print(f"Warning: failed to tick world after cleanup: {e}")
            print("Vehicle cleanup complete.")

    def _capture_carla_bev_image(self, bbox: Dict, output_dir: str, output_stem: str) -> Optional[str]:
        # (This method requires no changes)
        print("Capturing BEV image from the CARLA simulator...")
        sensor = None
        image_path = None
        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            
            aspect_ratio = bbox['extent']['y'] / bbox['extent']['x'] if bbox['extent']['x'] != 0 else 1
            image_width = 1024
            image_height = int(image_width * aspect_ratio)
            camera_bp.set_attribute('image_size_x', str(image_width))
            camera_bp.set_attribute('image_size_y', str(image_height))
            camera_bp.set_attribute('fov', '90')

            # Increase height for a wider view
            camera_z = max(bbox['extent']['x'], bbox['extent']['y']) * 0.8 + 25 
            camera_transform = carla.Transform(
                carla.Location(x=bbox['center']['x'], y=bbox['center']['y'], z=camera_z),
                carla.Rotation(pitch=-90, yaw=-90)
            )
            
            sensor = self.world.spawn_actor(camera_bp, camera_transform)
            
            image_queue = queue.Queue()
            sensor.listen(image_queue.put)
            
            for _ in range(max(1, self.warmup_ticks)):
                self._tick_or_wait(seconds=min(5.0, self.timeout))
            
            image = None
            for _ in range(max(1, self.warmup_ticks)):
                try:
                    image = image_queue.get(timeout=2.0)
                    break
                except queue.Empty:
                    self._tick_or_wait(seconds=min(5.0, self.timeout))

            if image is None:
                print("Error: Timed out waiting for CARLA BEV image.")
                return None

            filename = f"{output_stem}_carla_bev.png"
            image_path = os.path.join(output_dir, filename)
            image.save_to_disk(image_path)
            print(f"CARLA BEV image saved to: {image_path}")

        except Exception as e:
            print(f"An error occurred while capturing the CARLA BEV image: {e}")
            image_path = None
        finally:
            if sensor:
                sensor.stop()
                sensor.destroy()
                try:
                    self._tick_or_wait(seconds=1.0)
                except Exception:
                    pass
        
        return image_path

    def _generate_mpl_bev_image(self, waypoints: List[carla.Waypoint], scenario_data: Dict, bbox: Dict, output_dir: str, output_stem: str) -> str:
        """
        Generates a Matplotlib BEV image that displays lane centerlines and classified spawn/target points.
        """
        fig, ax = plt.subplots(figsize=(16, 16))

        padding = 10
        ax.set_xlim(-bbox['extent']['x'] - padding, bbox['extent']['x'] + padding)
        ax.set_ylim(-bbox['extent']['y'] - padding, bbox['extent']['y'] + padding)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f"Matplotlib BEV (Spawn/Target Points)\nMap: {self.carla_map.name.split('/')[-1]}", fontsize=16)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')

        # 1. Plot lane centerlines from all waypoints
        lanes_wps = {}
        for wp in waypoints:
            lane_key = f"{wp.road_id}_{wp.lane_id}"
            if lane_key not in lanes_wps:
                lanes_wps[lane_key] = []
            lanes_wps[lane_key].append(wp)

        for _, lane_waypoints in lanes_wps.items():
            sorted_wps = sorted(lane_waypoints, key=lambda w: w.s)
            x = [wp.transform.location.x - bbox["center"]["x"] for wp in sorted_wps]
            y = [wp.transform.location.y - bbox["center"]["y"] for wp in sorted_wps]
            ax.plot(x, y, color='darkgray', linewidth=1, alpha=0.7, zorder=1)
        
        # 2. Plot Spawn Points (Entry) and Target Points (Exit)
        all_spawn_points = scenario_data.get('spawn_points', [])
        spawn_points = all_spawn_points[:scenario_data['ego_spawn_num']]
        target_points = all_spawn_points[scenario_data['ego_spawn_num']:]
        
        # Plot Spawn points
        if spawn_points:
            sp_x = [p['x'] for p in spawn_points]
            sp_y = [p['y'] for p in spawn_points]
            ax.scatter(sp_x, sp_y, marker='o', s=50, c='green', edgecolors='black', linewidth=0.5, zorder=5, label='Spawn Points (Entry)')
            for sp in spawn_points:
                yaw_rad = np.deg2rad(sp['yaw'])
                ax.arrow(sp['x'], sp['y'], 2.5 * np.cos(yaw_rad), 2.5 * np.sin(yaw_rad),
                         head_width=0.7, head_length=0.8, fc='darkgreen', ec='darkgreen', zorder=6, alpha=0.9)
        
        # Plot Target points
        if target_points:
            tp_x = [p['x'] for p in target_points]
            tp_y = [p['y'] for p in target_points]
            ax.scatter(tp_x, tp_y, marker='s', s=50, c='blue', edgecolors='black', linewidth=0.5, zorder=5, label='Target Points (Exit)')
            for tp in target_points:
                yaw_rad = np.deg2rad(tp['yaw'])
                ax.arrow(tp['x'], tp['y'], 2.5 * np.cos(yaw_rad), 2.5 * np.sin(yaw_rad),
                         head_width=0.7, head_length=0.8, fc='darkblue', ec='darkblue', zorder=6, alpha=0.9)

        print(f"Plotted {len(lanes_wps)} lane centerlines, "
              f"{len(spawn_points)} spawn points, and {len(target_points)} target points on the Matplotlib BEV.")

        # 3. Create legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))

        plt.grid(True, linestyle=':', color='gray', alpha=0.5)
        plt.tight_layout()

        filename = f"{output_stem}_mpl_bev.png"
        image_path = os.path.join(output_dir, filename)
        plt.savefig(image_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return image_path

    def _save_scenario_json(self, scenario_data: Dict, output_dir: str, output_stem: str) -> str:
        # (This method requires no changes)
        filename = f"{output_stem}.json"
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(scenario_data, f, indent=4)
        return file_path

    def _get_waypoints_in_bbox(self, bbox: Dict, distance: float) -> List[carla.Waypoint]:
        # (This method requires no changes)
        all_waypoints = self.carla_map.generate_waypoints(distance)
        return [wp for wp in all_waypoints if self._is_point_in_bbox(wp.transform.location, bbox)]
    
    def _is_point_in_bbox(self, location: carla.Location, bbox: Dict) -> bool:
        return (bbox['min_x'] <= location.x <= bbox['max_x'] and 
                bbox['min_y'] <= location.y <= bbox['max_y'])

def main():
    parser = argparse.ArgumentParser(description="Construct static CARLA scenario regions from a JSONL region config.")
    parser.add_argument("--regions", default="configs/scenario_regions.jsonl", help="JSONL file containing scenario region specs.")
    parser.add_argument("--scenario-id", help="Construct only one region id. Defaults to all regions.")
    parser.add_argument("--all", action="store_true", help="Construct all regions in --regions. Kept for backward compatibility; this is now the default.")
    parser.add_argument("--output-dir", default="static_scenarios", help="Directory for generated static scenario JSON files.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0, help="CARLA client timeout in seconds.")
    parser.add_argument("--map-warmup-seconds", type=float, default=5.0, help="Seconds to wait after switching maps.")
    parser.add_argument("--warmup-ticks", type=int, default=5, help="Number of world ticks/snapshots to wait after switching maps.")
    args = parser.parse_args()

    specs = select_region_specs(
        load_region_specs(Path(args.regions)),
        scenario_id=args.scenario_id,
        include_all=args.all,
    )

    for spec in specs:
        constructor = None
        try:
            constructor = ScenarioConstructor(
                host=args.host,
                port=args.port,
                map_name=spec.map_name,
                timeout=args.timeout,
                map_warmup_seconds=args.map_warmup_seconds,
                warmup_ticks=args.warmup_ticks,
            )
            output_dir = Path(args.output_dir)
            result = constructor.construct_scenario(
                center_x=spec.center_x,
                center_y=spec.center_y,
                extent_x=spec.extent_x,
                extent_y=spec.extent_y,
                min_spawn_distance=spec.min_spawn_distance,
                min_global_distance=spec.min_global_distance,
                output_dir=str(output_dir),
                waypoint_sample_dist=spec.waypoint_sample_dist,
                exclude_junction=spec.exclude_junction,
                vehicle_filter=spec.vehicle_filter,
                scenario_type=spec.scenario_type,
                scenario_id=spec.id,
                highway_split_ratios=spec.highway_split_ratios,
            )

            print("\n--- Execution Summary ---")
            print(f"Region ID: {spec.id}")
            print(f"Final Verified Scenario JSON: {result.get('scenario_json_path')}")
            print(f"CARLA BEV Image: {result.get('carla_bev_image_path')}")
            print(f"Matplotlib BEV Image: {result.get('mpl_bev_path')}")
        except Exception as e:
            print(f"\nAn error occurred during execution for region {spec.id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if constructor:
                constructor._cleanup_spawned_actors()
    return

if __name__ == "__main__":
    main()

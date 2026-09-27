"""
Metrics module for CARLA scenario evaluation.
Provides classes and functions to calculate path tracking performance and collision metrics.
"""

import math
from typing import List, Tuple, Dict, Optional


class PathDeviationMetrics:
    """
    Path deviation metrics calculation class.
    Measures the gap between target path and actual driving path.
    """
    
    def __init__(self):
        self.target_path = []  # Target path points list [(x, y, z), ...]
        self.actual_path = []  # Actual path points list [(x, y, z, timestamp), ...]
        self.frame_deviations = []  # Frame-by-frame deviation distances
        
    def set_target_path(self, waypoints_data: List[Dict]) -> None:
        """
        Set target path from waypoint data.
        
        Args:
            waypoints_data: Path waypoint data in format [{'location': {'x': x, 'y': y, 'z': z}}, ...]
        """
        self.target_path = []
        for waypoint in waypoints_data:
            loc = waypoint['location']
            self.target_path.append((loc['x'], loc['y'], loc['z']))
        print(f"✅ Target path set with {len(self.target_path)} waypoints")
    
    def record_actual_position(self, ego_vehicle, timestamp: float) -> None:
        """
        Record actual vehicle position.
        
        Args:
            ego_vehicle: carla.Vehicle - Ego vehicle
            timestamp: float - Timestamp
        """
        if ego_vehicle:
            location = ego_vehicle.get_location()
            self.actual_path.append((location.x, location.y, location.z, timestamp))
    
    def _point_to_line_segment_distance(self, point: Tuple[float, float, float], 
                                       line_start: Tuple[float, float, float], 
                                       line_end: Tuple[float, float, float]) -> float:
        """
        Calculate shortest distance from point to line segment.
        
        Args:
            point: (x, y, z) - Query point
            line_start: (x, y, z) - Line segment start point
            line_end: (x, y, z) - Line segment end point
            
        Returns:
            float: Shortest distance
        """
        px, py, pz = point
        ax, ay, az = line_start
        bx, by, bz = line_end
        
        # Calculate line segment vector
        ab_x, ab_y, ab_z = bx - ax, by - ay, bz - az
        
        # Calculate vector from line start to point
        ap_x, ap_y, ap_z = px - ax, py - ay, pz - az
        
        # Calculate line segment length squared
        ab_length_sq = ab_x**2 + ab_y**2 + ab_z**2
        
        if ab_length_sq == 0:
            # If line segment length is 0, return distance to start point
            return math.sqrt(ap_x**2 + ap_y**2 + ap_z**2)
        
        # Calculate projection parameter t
        t = max(0, min(1, (ap_x * ab_x + ap_y * ab_y + ap_z * ab_z) / ab_length_sq))
        
        # Calculate projection point
        proj_x = ax + t * ab_x
        proj_y = ay + t * ab_y
        proj_z = az + t * ab_z
        
        # Return distance from point to projection point
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2 + (pz - proj_z)**2)
    
    def _find_closest_path_segment(self, actual_point: Tuple[float, float, float]) -> Tuple[float, int]:
        """
        Find the closest target path segment to actual position point.
        
        Args:
            actual_point: (x, y, z) - Actual position point
            
        Returns:
            tuple: (shortest distance, path segment index)
        """
        if len(self.target_path) < 2:
            return float('inf'), -1
        
        min_distance = float('inf')
        closest_segment_idx = -1
        
        for i in range(len(self.target_path) - 1):
            distance = self._point_to_line_segment_distance(
                actual_point, self.target_path[i], self.target_path[i + 1]
            )
            if distance < min_distance:
                min_distance = distance
                closest_segment_idx = i
        
        return min_distance, closest_segment_idx
    
    def calculate_frame_deviation(self, ego_vehicle, timestamp: float) -> float:
        """
        Calculate current frame path deviation.
        
        Args:
            ego_vehicle: carla.Vehicle - Ego vehicle
            timestamp: float - Timestamp
            
        Returns:
            float: Current frame deviation distance
        """
        if not ego_vehicle or len(self.target_path) < 2:
            return 0.0
        
        # Record actual position
        self.record_actual_position(ego_vehicle, timestamp)
        
        # Get current position
        location = ego_vehicle.get_location()
        actual_point = (location.x, location.y, location.z)
        
        # Calculate shortest distance to target path
        deviation, _ = self._find_closest_path_segment(actual_point)
        
        # Record deviation distance
        self.frame_deviations.append(deviation)
        
        return deviation
    
    def calculate_path_completion_ratio(self, ego_vehicle) -> float:
        """
        Calculate path completion ratio.
        
        Args:
            ego_vehicle: carla.Vehicle - Ego vehicle
            
        Returns:
            float: Completion ratio (0.0 - 1.0)
        """
        if not ego_vehicle or len(self.target_path) < 2:
            return 0.0
        
        location = ego_vehicle.get_location()
        ego_pos = (location.x, location.y, location.z)
        
        # Find closest path segment
        _, closest_segment_idx = self._find_closest_path_segment(ego_pos)
        
        if closest_segment_idx == -1:
            return 0.0
        
        # Calculate completion = closest path segment index / total path segments
        completion = closest_segment_idx / (len(self.target_path) - 1)
        return min(1.0, max(0.0, completion))
    
    def get_deviation_statistics(self) -> Dict[str, float]:
        """
        Get path deviation statistics.
        
        Returns:
            dict: Dictionary containing various deviation metrics
        """
        if not self.frame_deviations:
            return {
                'avg_lateral_deviation': 0.0,
                'max_deviation': 0.0,
                'min_deviation': 0.0,
                'std_deviation': 0.0,
                'cumulative_deviation_area': 0.0
            }
        
        deviations = self.frame_deviations
        
        # Basic statistics
        avg_deviation = sum(deviations) / len(deviations)
        max_deviation = max(deviations)
        min_deviation = min(deviations)
        
        # Standard deviation
        variance = sum((d - avg_deviation) ** 2 for d in deviations) / len(deviations)
        std_deviation = math.sqrt(variance)
        
        # Cumulative deviation area (simplified as integral of deviation distances)
        cumulative_area = sum(deviations)
        
        return {
            'avg_lateral_deviation': round(avg_deviation, 3),
            'max_deviation': round(max_deviation, 3),
            'min_deviation': round(min_deviation, 3),
            'std_deviation': round(std_deviation, 3),
            'cumulative_deviation_area': round(cumulative_area, 3)
        }
    
    def get_comprehensive_metrics(self, ego_vehicle=None) -> Dict[str, float]:
        """
        Get comprehensive path deviation metrics.
        
        Args:
            ego_vehicle: carla.Vehicle - Ego vehicle (for completion calculation)
            
        Returns:
            dict: Comprehensive metrics
        """
        deviation_stats = self.get_deviation_statistics()
        
        # Calculate completion ratio
        completion_ratio = 0.0
        if ego_vehicle:
            completion_ratio = self.calculate_path_completion_ratio(ego_vehicle)
        
        # Comprehensive metric calculation
        # Deviation severity = average deviation + weighted max deviation
        deviation_severity = (
            deviation_stats['avg_lateral_deviation'] * 0.6 + 
            deviation_stats['max_deviation'] * 0.4
        )
        
        # Path tracking quality = 1 / (1 + avg lateral deviation), range (0, 1]
        tracking_quality = 1.0 / (1.0 + deviation_stats['avg_lateral_deviation'])
        
        return {
            **deviation_stats,
            'path_completion_ratio': round(completion_ratio, 3),
            'deviation_severity': round(deviation_severity, 3),
            'path_tracking_quality': round(tracking_quality, 3),
            'actual_path_length': len(self.actual_path),
            'target_path_length': len(self.target_path)
        }
    
    def reset(self) -> None:
        """Reset all recorded data."""
        self.actual_path.clear()
        self.frame_deviations.clear()
        print("✅ Path deviation metrics reset")


class ETTCMetrics:
    """
    Estimate Time to Collision (ettc) metrics calculation class.
    Calculates collision risk metrics between ego vehicle and nearby vehicles.
    """
    
    def __init__(self, detection_radius: float = 10.0, collision_radius: float = 4.6):
        """
        Initialize ettc metrics calculator.
        
        Args:
            detection_radius: Detection radius in meters for nearby vehicles
        """
        self.detection_radius = detection_radius
        self.collision_radius = collision_radius
        self.ettc_values = []  # Store ettc values for each frame
        self.distance_values = []
        self.brake_margin_values = []
        self.max_ttc = None  # Maximum TTC value (total simulation duration)

    @staticmethod
    def _brake_distance(speed_mps: float) -> float:
        """AV-Fuzzer-style empirical braking distance in meters."""
        return max(0.0, 0.0467 * speed_mps ** 2 + 0.4116 * speed_mps - 1.9913 + 0.5)

    @staticmethod
    def _speed_mps(velocity) -> float:
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    
    def set_max_ttc(self, duration_frames: int, frame_rate: float) -> None:
        """
        Set maximum TTC value based on scenario duration.
        
        Args:
            duration_frames: Total scenario duration in frames
            frame_rate: Simulation frame rate
        """
        self.max_ttc = duration_frames / frame_rate
        print(f"✅ ETTC tracking initialized - Max TTC: {self.max_ttc:.1f}s, Detection radius: {self.detection_radius}m")
    
    def _calculate_ttc_between_vehicles(self, ego_vehicle, other_vehicle) -> float:
        """
        Calculate Time to Collision (TTC) between two vehicles using relative motion.
        
        Args:
            ego_vehicle: carla.Vehicle - Ego vehicle
            other_vehicle: carla.Vehicle - Other vehicle
            
        Returns:
            float: TTC in seconds, or self.max_ttc if no collision predicted
        """
        try:
            # Get vehicle positions and velocities
            ego_transform = ego_vehicle.get_transform()
            other_transform = other_vehicle.get_transform()
            ego_velocity = ego_vehicle.get_velocity()
            other_velocity = other_vehicle.get_velocity()
            
            # Calculate relative position vector (other - ego)
            delta_p_x = other_transform.location.x - ego_transform.location.x
            delta_p_y = other_transform.location.y - ego_transform.location.y
            
            # Calculate relative velocity vector (other - ego)
            delta_v_x = other_velocity.x - ego_velocity.x
            delta_v_y = other_velocity.y - ego_velocity.y
            
            # Calculate dot product: Δp · Δv
            dot_product = delta_p_x * delta_v_x + delta_p_y * delta_v_y
            
            # Calculate velocity magnitude squared: |Δv|²
            velocity_squared = delta_v_x * delta_v_x + delta_v_y * delta_v_y
            
            # Check edge cases
            if velocity_squared < 1e-6:  # Vehicles are relatively stationary
                return self.max_ttc
            
            if dot_product >= 0:  # Vehicles are moving away from each other
                return self.max_ttc
            
            # Solve ||Δp + tΔv|| <= collision_radius. This avoids reporting a
            # low TTC for vehicles that are merely moving toward a closest
            # approach point but whose projected paths do not overlap.
            position_squared = delta_p_x * delta_p_x + delta_p_y * delta_p_y
            radius_squared = self.collision_radius * self.collision_radius
            c = position_squared - radius_squared
            if c <= 0:
                return 0.0

            discriminant = dot_product * dot_product - velocity_squared * c
            if discriminant < 0:
                return self.max_ttc

            ttc = (-dot_product - math.sqrt(discriminant)) / velocity_squared
            
            # Return valid TTC or max_ttc for invalid results
            return ttc if ttc > 0 else self.max_ttc
            
        except Exception:
            # Return max_ttc on any calculation error
            return self.max_ttc
    
    def _get_vehicles_in_radius(self, ego_vehicle, spawned_actors) -> List:
        """
        Get all vehicles within detection radius of the ego vehicle.
        
        Args:
            ego_vehicle: carla.Vehicle - Ego vehicle
            spawned_actors: List of spawned actors in the scenario
            
        Returns:
            List[carla.Vehicle]: List of vehicles within radius (excluding ego vehicle)
        """
        try:
            ego_location = ego_vehicle.get_location()
            nearby_vehicles = []
            
            for actor in spawned_actors:
                if (actor.type_id.startswith('vehicle.') and 
                    actor.id != ego_vehicle.id and 
                    hasattr(actor, 'get_location')):
                    
                    distance = ego_location.distance(actor.get_location())
                    if distance <= self.detection_radius:
                        nearby_vehicles.append(actor)
            
            return nearby_vehicles
            
        except Exception:
            return []
    
    def calculate_frame_ettc(self, ego_vehicle, spawned_actors) -> float:
        """
        Calculate Estimate Time to Collision (ETTC) for current frame.
        
        Args:
            ego_vehicle: carla.Vehicle - Ego vehicle
            spawned_actors: List of spawned actors in the scenario
            
        Returns:
            float: ETTC value in seconds for this frame
        """
        if not ego_vehicle or not spawned_actors or not self.max_ttc:
            return self.max_ttc if self.max_ttc else float('inf')
        
        try:
            # Get vehicles within detection radius
            nearby_vehicles = self._get_vehicles_in_radius(ego_vehicle, spawned_actors)
            
            if not nearby_vehicles:
                ettc_value = self.max_ttc
            else:
                # Calculate TTC with each nearby vehicle
                min_ttc = self.max_ttc
                min_distance = float('inf')
                min_brake_margin = float('inf')
                ego_location = ego_vehicle.get_location()
                ego_speed = self._speed_mps(ego_vehicle.get_velocity())
                brake_distance = self._brake_distance(ego_speed)
                for vehicle in nearby_vehicles:
                    ttc = self._calculate_ttc_between_vehicles(ego_vehicle, vehicle)
                    min_ttc = min(min_ttc, ttc)
                    distance = ego_location.distance(vehicle.get_location())
                    min_distance = min(min_distance, distance)
                    min_brake_margin = min(min_brake_margin, distance - self.collision_radius - brake_distance)
                
                ettc_value = min_ttc
                if min_distance < float('inf'):
                    self.distance_values.append(min_distance)
                if min_brake_margin < float('inf'):
                    self.brake_margin_values.append(min_brake_margin)
            
            # Record ETTC value
            self.ettc_values.append(ettc_value)
            return ettc_value
            
        except Exception:
            fallback_value = self.max_ttc if self.max_ttc else float('inf')
            self.ettc_values.append(fallback_value)
            return fallback_value
    
    def get_ettc_statistics(self) -> Dict[str, float]:
        """
        Get ETTC statistics.
        
        Returns:
            dict: Dictionary containing ETTC metrics
        """
        if not self.ettc_values:
            return {
                'min_ettc': self.max_ttc if self.max_ttc else float('inf'),
                'avg_ettc': self.max_ttc if self.max_ttc else float('inf'),
                'danger_frames': 0,
                'detection_radius': self.detection_radius,
                'collision_radius': self.collision_radius,
                'min_distance': float('inf'),
                'min_brake_margin': float('inf'),
            }
        
        valid_ettc_values = [v for v in self.ettc_values if v < self.max_ttc] if self.max_ttc else []
        
        return {
            'min_ettc': round(min(self.ettc_values), 2),
            'avg_ettc': round(sum(self.ettc_values) / len(self.ettc_values), 2),
            'danger_frames': len(valid_ettc_values),  # Frames with finite TTC
            'detection_radius': self.detection_radius,
            'collision_radius': self.collision_radius,
            'min_distance': round(min(self.distance_values), 2) if self.distance_values else float('inf'),
            'min_brake_margin': round(min(self.brake_margin_values), 2) if self.brake_margin_values else float('inf'),
        }
    
    def reset(self) -> None:
        """Reset all recorded ettc data."""
        self.ettc_values.clear()
        self.distance_values.clear()
        self.brake_margin_values.clear()
        print("✅ ETTC metrics reset")


class DrivingQualityMetrics:
    """
    Lightweight DriveFuzz-style driving quality monitor.

    The metric records reproducible telemetry-derived deductions without
    depending on scipy/skfuzzy, so it can be used by all VLAD-Fuzz workflows.
    """

    HARD_ACCELERATION_KMHPS = 21.2
    HARD_BRAKING_KMHPS = -21.2
    STEERING_DEGREES = 70.0
    HARD_TURN_MIN_STEER_DEG = 20.0
    HARD_TURN_RATIO = 0.18
    OSCILLATION_STEER_THRESHOLD = 0.2
    OVERSTEER_YAW_RATE_DEGPS = 90.0
    OVERSTEER_LATERAL_ACCEL_KMHPS = 200.0
    UNDERSTEER_MIN_STEER_DEG = 35.0
    UNDERSTEER_MAX_YAW_RATE_DEGPS = 30.0

    def __init__(self, frame_rate: float = 20.0, detection_radius: float = 20.0, collision_radius: float = 4.6):
        self.frame_rate = frame_rate
        self.detection_radius = detection_radius
        self.collision_radius = collision_radius
        self.reset()

    @staticmethod
    def _speed_kmh(velocity) -> float:
        return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    @staticmethod
    def _brake_distance(speed_mps: float) -> float:
        return ETTCMetrics._brake_distance(speed_mps)

    @staticmethod
    def _yaw_delta(current: float, previous: float) -> float:
        delta = current - previous
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        return delta

    @staticmethod
    def _signed_nonzero(value: float, threshold: float) -> int:
        if value > threshold:
            return 1
        if value < -threshold:
            return -1
        return 0

    def _delta_time(self, timestamp: Optional[float]) -> float:
        if timestamp is None or self._last_timestamp is None:
            return 1.0 / self.frame_rate
        return max(1e-6, float(timestamp) - self._last_timestamp)

    def _record_near_vehicle_margin(self, ego_vehicle, spawned_actors) -> None:
        if not ego_vehicle or not spawned_actors:
            return
        try:
            ego_location = ego_vehicle.get_location()
            ego_velocity = ego_vehicle.get_velocity()
            ego_speed_mps = math.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2)
            brake_distance = self._brake_distance(ego_speed_mps)
            for actor in spawned_actors:
                if actor.id == ego_vehicle.id:
                    continue
                if not getattr(actor, "type_id", "").startswith("vehicle."):
                    continue
                if not hasattr(actor, "get_location"):
                    continue
                distance = ego_location.distance(actor.get_location())
                if distance > self.detection_radius:
                    continue
                brake_margin = distance - self.collision_radius - brake_distance
                self.min_distance = min(self.min_distance, distance)
                self.min_brake_margin = min(self.min_brake_margin, brake_margin)
                if brake_margin < 0.0:
                    self.low_brake_margin_frames += 1
        except Exception:
            return

    def record_frame(self, ego_vehicle, control=None, spawned_actors=None, timestamp: Optional[float] = None) -> None:
        if not ego_vehicle:
            return

        try:
            velocity = ego_vehicle.get_velocity()
            transform = ego_vehicle.get_transform()
            speed_kmh = self._speed_kmh(velocity)
            longitudinal_speed_kmh, lateral_speed_kmh = self._body_frame_speeds_kmh(velocity, transform)
            dt = self._delta_time(timestamp)

            if self._last_speed_kmh is not None:
                acceleration_kmhps = (speed_kmh - self._last_speed_kmh) / dt
                self.acceleration_values.append(acceleration_kmhps)
                if acceleration_kmhps >= self.HARD_ACCELERATION_KMHPS:
                    self.hard_acceleration_count += 1
                if acceleration_kmhps <= self.HARD_BRAKING_KMHPS:
                    self.hard_braking_count += 1

            steer = float(getattr(control, "steer", 0.0)) if control is not None else 0.0
            self.steer_values.append(steer)
            steer_angle_deg = abs(steer) * self.STEERING_DEGREES
            if steer_angle_deg > self.HARD_TURN_MIN_STEER_DEG:
                hard_turn_ratio = abs(lateral_speed_kmh) / steer_angle_deg
                if hard_turn_ratio > self.HARD_TURN_RATIO:
                    self.hard_turn_count += 1

            steer_sign = self._signed_nonzero(steer, self.OSCILLATION_STEER_THRESHOLD)
            if self._last_steer_sign and steer_sign and steer_sign != self._last_steer_sign:
                self.control_oscillation_count += 1
            if steer_sign:
                self._last_steer_sign = steer_sign

            yaw = float(transform.rotation.yaw)
            yaw_rate = 0.0
            if self._last_yaw is not None:
                yaw_rate = abs(self._yaw_delta(yaw, self._last_yaw)) / dt
                self.max_abs_yaw_rate = max(self.max_abs_yaw_rate, yaw_rate)

            if self._last_lateral_speed_kmh is not None:
                lateral_acceleration_kmhps = abs(lateral_speed_kmh - self._last_lateral_speed_kmh) / dt
                if (
                    steer_angle_deg >= self.HARD_TURN_MIN_STEER_DEG
                    and yaw_rate >= self.OVERSTEER_YAW_RATE_DEGPS
                    and lateral_acceleration_kmhps >= self.OVERSTEER_LATERAL_ACCEL_KMHPS
                ):
                    self.oversteer_count += 1
                if (
                    abs(longitudinal_speed_kmh) > 18.0
                    and steer_angle_deg >= self.UNDERSTEER_MIN_STEER_DEG
                    and yaw_rate <= self.UNDERSTEER_MAX_YAW_RATE_DEGPS
                    and abs(lateral_speed_kmh) <= 2.0
                ):
                    self.understeer_count += 1

            self.speed_values.append(speed_kmh)
            self.longitudinal_speed_values.append(longitudinal_speed_kmh)
            self.lateral_speed_values.append(lateral_speed_kmh)
            self.yaw_rate_values.append(yaw_rate)
            self._record_near_vehicle_margin(ego_vehicle, spawned_actors)
            self._last_speed_kmh = speed_kmh
            self._last_lateral_speed_kmh = lateral_speed_kmh
            self._last_yaw = yaw
            self._last_timestamp = timestamp
        except Exception:
            return

    @staticmethod
    def _body_frame_speeds_kmh(velocity, transform) -> Tuple[float, float]:
        yaw_rad = math.radians(float(transform.rotation.yaw))
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        right_x = -math.sin(yaw_rad)
        right_y = math.cos(yaw_rad)
        longitudinal_speed_mps = velocity.x * forward_x + velocity.y * forward_y
        lateral_speed_mps = velocity.x * right_x + velocity.y * right_y
        return 3.6 * longitudinal_speed_mps, 3.6 * lateral_speed_mps

    @staticmethod
    def _average(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def get_statistics(self) -> Dict[str, float]:
        distance_deduction = 0.0
        if self.min_distance < float("inf"):
            distance_deduction = 0.0 if int(self.min_distance) > 100 else 1.0 / max(1.0, int(self.min_distance))

        brake_margin_deduction = max(0.0, -self.min_brake_margin) if self.min_brake_margin < float("inf") else 0.0
        deductions = (
            self.hard_acceleration_count
            + self.hard_braking_count
            + self.hard_turn_count
            + self.control_oscillation_count
            + self.oversteer_count
            + self.understeer_count
            + self.low_brake_margin_frames
            + distance_deduction
            + brake_margin_deduction
        )
        score = 1.0 / (1.0 + deductions)

        return {
            "hard_acceleration_count": self.hard_acceleration_count,
            "hard_braking_count": self.hard_braking_count,
            "hard_turn_count": self.hard_turn_count,
            "oversteer_count": self.oversteer_count,
            "understeer_count": self.understeer_count,
            "control_oscillation_count": self.control_oscillation_count,
            "low_brake_margin_frames": self.low_brake_margin_frames,
            "avg_longitudinal_speed": round(self._average(self.longitudinal_speed_values), 2),
            "avg_lateral_speed": round(self._average([abs(value) for value in self.lateral_speed_values]), 2),
            "max_lateral_speed": round(max([abs(value) for value in self.lateral_speed_values], default=0.0), 2),
            "max_yaw_rate": round(max(self.yaw_rate_values, default=0.0), 2),
            "min_distance": round(self.min_distance, 2) if self.min_distance < float("inf") else float("inf"),
            "min_brake_margin": round(self.min_brake_margin, 2) if self.min_brake_margin < float("inf") else float("inf"),
            "max_abs_yaw_rate": round(self.max_abs_yaw_rate, 2),
            "driving_quality_deductions": round(deductions, 3),
            "driving_quality_score": round(score, 3),
            "speed_sample_count": len(self.speed_values),
        }

    def reset(self) -> None:
        self.speed_values = []
        self.longitudinal_speed_values = []
        self.lateral_speed_values = []
        self.yaw_rate_values = []
        self.acceleration_values = []
        self.steer_values = []
        self.hard_acceleration_count = 0
        self.hard_braking_count = 0
        self.hard_turn_count = 0
        self.oversteer_count = 0
        self.understeer_count = 0
        self.control_oscillation_count = 0
        self.low_brake_margin_frames = 0
        self.min_distance = float("inf")
        self.min_brake_margin = float("inf")
        self.max_abs_yaw_rate = 0.0
        self._last_speed_kmh = None
        self._last_lateral_speed_kmh = None
        self._last_yaw = None
        self._last_timestamp = None
        self._last_steer_sign = 0
        print("✅ Driving quality metrics reset")

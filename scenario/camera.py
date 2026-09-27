"""
Camera utilities module for CARLA scenario visualization.
Provides camera frustum calculation and ego vehicle camera functionality.
"""

import carla
import cv2
import math
import numpy as np
import os
import time
from typing import List, Optional, Tuple

class CameraFrustum:
    """Camera frustum calculation for visibility testing"""
    
    def __init__(self, camera_transform, fov_horizontal=90, fov_vertical=60, near_plane=0.1, far_plane=100):
        self.camera_transform = camera_transform
        self.fov_h = np.radians(fov_horizontal)
        self.fov_v = np.radians(fov_vertical)
        self.near = near_plane
        self.far = far_plane
        
        # Calculate camera coordinate system
        self.forward = self.camera_transform.get_forward_vector()
        self.right = self.camera_transform.get_right_vector()
        self.up = self.camera_transform.get_up_vector()
        
        # Calculate frustum planes
        self._calculate_frustum_planes()
    
    def _calculate_frustum_planes(self):
        """Calculate the 6 frustum planes (left, right, top, bottom, near, far)"""
        pos = self.camera_transform.location
        
        # Near and far planes
        self.near_plane_normal = self.forward
        self.near_plane_point = pos + self.forward * self.near
        
        self.far_plane_normal = carla.Vector3D(-self.forward.x, -self.forward.y, -self.forward.z)
        self.far_plane_point = pos + self.forward * self.far
        
        # Calculate frustum angles
        half_fov_h = self.fov_h / 2
        half_fov_v = self.fov_v / 2
        
        # Left plane
        left_normal_x = np.cos(half_fov_h) * self.forward.x + np.sin(half_fov_h) * self.right.x
        left_normal_y = np.cos(half_fov_h) * self.forward.y + np.sin(half_fov_h) * self.right.y
        left_normal_z = np.cos(half_fov_h) * self.forward.z + np.sin(half_fov_h) * self.right.z
        self.left_plane_normal = carla.Vector3D(left_normal_x, left_normal_y, left_normal_z)
        self.left_plane_point = pos
        
        # Right plane
        right_normal_x = np.cos(half_fov_h) * self.forward.x - np.sin(half_fov_h) * self.right.x
        right_normal_y = np.cos(half_fov_h) * self.forward.y - np.sin(half_fov_h) * self.right.y
        right_normal_z = np.cos(half_fov_h) * self.forward.z - np.sin(half_fov_h) * self.right.z
        self.right_plane_normal = carla.Vector3D(right_normal_x, right_normal_y, right_normal_z)
        self.right_plane_point = pos
        
        # Top plane
        top_normal_x = np.cos(half_fov_v) * self.forward.x - np.sin(half_fov_v) * self.up.x
        top_normal_y = np.cos(half_fov_v) * self.forward.y - np.sin(half_fov_v) * self.up.y
        top_normal_z = np.cos(half_fov_v) * self.forward.z - np.sin(half_fov_v) * self.up.z
        self.top_plane_normal = carla.Vector3D(top_normal_x, top_normal_y, top_normal_z)
        self.top_plane_point = pos
        
        # Bottom plane
        bottom_normal_x = np.cos(half_fov_v) * self.forward.x + np.sin(half_fov_v) * self.up.x
        bottom_normal_y = np.cos(half_fov_v) * self.forward.y + np.sin(half_fov_v) * self.up.y
        bottom_normal_z = np.cos(half_fov_v) * self.forward.z + np.sin(half_fov_v) * self.up.z
        self.bottom_plane_normal = carla.Vector3D(bottom_normal_x, bottom_normal_y, bottom_normal_z)
        self.bottom_plane_point = pos
        
        self.planes = [
            (self.left_plane_normal, self.left_plane_point),
            (self.right_plane_normal, self.right_plane_point),
            (self.top_plane_normal, self.top_plane_point),
            (self.bottom_plane_normal, self.bottom_plane_point),
            (self.near_plane_normal, self.near_plane_point),
            (self.far_plane_normal, self.far_plane_point)
        ]
    
    def point_in_frustum(self, point):
        """Check if point is within frustum"""
        for normal, plane_point in self.planes:
            # Calculate distance from point to plane
            to_point = carla.Vector3D(
                point.x - plane_point.x,
                point.y - plane_point.y,
                point.z - plane_point.z
            )
            if self._vector_dot(normal, to_point) < 0:
                return False
        return True
    
    def _vector_dot(self, v1, v2):
        """Calculate vector dot product"""
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def filter_coordinates(camera_transform, coordinates, fov_h=90, fov_v=60, max_distance=100):
    """
    Filter coordinates using precise frustum checking.
    
    Args:
        camera_transform: carla.Transform - Camera transform
        coordinates: List of carla.Location - Coordinates to filter
        fov_h: float - Horizontal field of view
        fov_v: float - Vertical field of view
        max_distance: float - Maximum visibility distance
        
    Returns:
        List[carla.Location]: Visible coordinates
    """
    frustum = CameraFrustum(camera_transform, fov_h, fov_v, 0.1, max_distance)
    visible_coordinates = []
    
    for coord in coordinates:
        if frustum.point_in_frustum(coord):
            visible_coordinates.append(coord)
    
    return visible_coordinates


class EgoCamera:
    """Ego vehicle camera for image capture"""
    
    def __init__(self, world, ego_vehicle, image_width=1920, image_height=1080):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.image_width = image_width
        self.image_height = image_height
        self.image_captured = False
        self.image_data = None
        self.camera = None
        self.fov = 90  # Field of view in degrees
        
        self._setup_camera()
    
    def _setup_camera(self):
        """Setup camera sensor on ego vehicle"""
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_width))
        camera_bp.set_attribute('image_size_y', str(self.image_height))
        camera_bp.set_attribute('fov', '90')
        
        # Camera position: front of vehicle, slightly elevated (simulating autonomous driving sensor)
        camera_transform = carla.Transform(
            carla.Location(x=1.0, y=0.0, z=1.8),  # 1m forward, 1.8m high
            carla.Rotation(pitch=-8, yaw=0, roll=0)  # Slightly downward pitch
        )
        
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        self.camera.listen(self._on_image_received)
        
        print(f"✅ Ego camera attached: {self.image_width}x{self.image_height}, FOV=90°")
    
    def _on_image_received(self, image):
        """Image reception callback"""
        self.image_data = image
        self.image_captured = True
    
    def project_world_to_image(self, world_point):
        """
        Project 3D world coordinates to 2D image coordinates using simplified method.
        
        Args:
            world_point: carla.Location - World point to project
            
        Returns:
            tuple: (x, y) image coordinates or None if projection fails
        """
        camera_transform = self.get_camera_transform()
        if not camera_transform:
            return None
        
        # Get camera vectors (CARLA provides these directly)
        camera_location = camera_transform.location
        forward = camera_transform.get_forward_vector()
        right = camera_transform.get_right_vector()
        up = camera_transform.get_up_vector()
        
        # Vector from camera to world point
        world_to_camera = carla.Vector3D(
            world_point.x - camera_location.x,
            world_point.y - camera_location.y,
            world_point.z - camera_location.z
        )
        
        # Project to camera coordinate system
        # In camera coordinates: X=right, Y=up, Z=forward (depth)
        cam_x = world_to_camera.x * right.x + world_to_camera.y * right.y + world_to_camera.z * right.z
        cam_y = world_to_camera.x * up.x + world_to_camera.y * up.y + world_to_camera.z * up.z
        cam_z = world_to_camera.x * forward.x + world_to_camera.y * forward.y + world_to_camera.z * forward.z
        
        # Debug intermediate values
        print(f"   📍 Camera vectors - Forward: ({forward.x:.3f}, {forward.y:.3f}, {forward.z:.3f})")
        print(f"   📍 World to camera vector: ({world_to_camera.x:.3f}, {world_to_camera.y:.3f}, {world_to_camera.z:.3f})")
        print(f"   📍 Camera coordinates: x={cam_x:.3f}, y={cam_y:.3f}, z={cam_z:.3f}")
        
        # Check if point is behind camera
        if cam_z <= 0:
            print(f"   ❌ Point is behind camera (depth={cam_z:.3f})")
            return None
        
        # Calculate focal length from FOV
        fov_radians = math.radians(self.fov)
        focal_length = self.image_width / (2.0 * math.tan(fov_radians / 2.0))
        
        # Project to image coordinates
        # Standard pinhole camera model
        pixel_x = (focal_length * cam_x / cam_z) + (self.image_width / 2.0)
        pixel_y = -(focal_length * cam_y / cam_z) + (self.image_height / 2.0)  # Flip Y axis
        
        print(f"   📍 Focal length: {focal_length:.2f}")
        print(f"   📍 Projected coordinates: ({pixel_x:.2f}, {pixel_y:.2f})")
        
        # Check if point is within image bounds
        if 0 <= pixel_x < self.image_width and 0 <= pixel_y < self.image_height:
            return (int(pixel_x), int(pixel_y))
        else:
            print(f"   ❌ Point outside image bounds: ({pixel_x:.2f}, {pixel_y:.2f})")
            return None
    
    def add_target_point_marker(self, image_array, image_coords, marker_size=15):
        """Add green target point marker to image"""
        x, y = image_coords
        
        # Ensure coordinates are integers
        x, y = int(x), int(y)
        
        # Make a copy to avoid modifying the original if needed
        result_image = image_array.copy()
        
        # Draw green circle marker (BGR format: Green = 0,255,0)
        cv2.circle(result_image, (x, y), marker_size, (0, 255, 0), -1)  # Filled green circle
        cv2.circle(result_image, (x, y), marker_size + 2, (0, 0, 0), 2)  # Black outline
        
        # Add text label
        label = "TARGET"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        # Get text size for positioning
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Position text above the marker
        text_x = x - text_width // 2
        text_y = y - marker_size - 10
        
        # Ensure text is within image bounds
        text_x = max(0, min(text_x, result_image.shape[1] - text_width))
        text_y = max(text_height, text_y)
        
        # Draw text with black outline and green fill
        cv2.putText(result_image, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(result_image, label, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
        
        return result_image
    
    def capture_image(self, output_dir="scenario_images", filename=None, target_point=None):
        """Capture and save image with optional target point marking"""
        if filename is None:
            filename = "ego_camera_view.png"
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Wait for image capture
        self.image_captured = False
        max_tick_times = 10
        tick_count = 0
        while not self.image_captured and tick_count < max_tick_times:
            self.world.tick()
            tick_count += 1
            time.sleep(0.5)
        
        if self.image_captured and self.image_data:
            filepath = os.path.join(output_dir, filename)
            
            # If target point is provided, add marker to image
            if target_point:
                # Convert target point to world location
                target_location = carla.Location(x=target_point['x'], y=target_point['y'], z=target_point['z'])
                
                # Debug information
                camera_transform = self.get_camera_transform()
                if camera_transform:
                    print(f"🔍 Debug Info:")
                    print(f"   Camera Location: ({camera_transform.location.x:.2f}, {camera_transform.location.y:.2f}, {camera_transform.location.z:.2f})")
                    print(f"   Camera Rotation: (pitch={camera_transform.rotation.pitch:.2f}°, yaw={camera_transform.rotation.yaw:.2f}°, roll={camera_transform.rotation.roll:.2f}°)")
                    print(f"   Target Location: ({target_location.x:.2f}, {target_location.y:.2f}, {target_location.z:.2f})")
                    
                    # Calculate distance
                    distance = math.sqrt(
                        (target_location.x - camera_transform.location.x)**2 +
                        (target_location.y - camera_transform.location.y)**2 +
                        (target_location.z - camera_transform.location.z)**2
                    )
                    print(f"   Distance to target: {distance:.2f}m")
                
                # Project target point to image coordinates
                image_coords = self.project_world_to_image(target_location)
                
                if image_coords:
                    print(f"   ✅ Projection successful: {image_coords}")
                    
                    # Convert CARLA image to numpy array
                    image_array = np.frombuffer(self.image_data.raw_data, dtype=np.uint8)
                    image_array = image_array.reshape((self.image_height, self.image_width, 4))  # BGRA format
                    image_array = image_array[:, :, :3]  # Remove alpha channel, keep BGR
                    
                    # Ensure the array is contiguous and in the correct format for OpenCV
                    image_array = np.ascontiguousarray(image_array, dtype=np.uint8)
                    
                    # Add target point marker
                    image_array = self.add_target_point_marker(image_array, image_coords)
                    
                    # Save modified image
                    cv2.imwrite(filepath, image_array)
                    print(f"✅ Ego camera image saved with target point marker: {filepath}")
                    print(f"   Target point projected to image coordinates: {image_coords}")
                else:
                    print(f"   ❌ Projection failed - target point not visible")
                    # Target point not visible, save original image
                    self.image_data.save_to_disk(filepath)
                    print(f"✅ Ego camera image saved (target point not visible): {filepath}")
            else:
                # No target point, save original image
                self.image_data.save_to_disk(filepath)
                print(f"✅ Ego camera image saved: {filepath}")
            
            return filepath
        else:
            print("❌ Ego camera image capture failed!")
            return None
    
    def get_camera_transform(self):
        """Get current camera world transform"""
        if self.camera:
            return self.camera.get_transform()
        return None
    
    def destroy(self):
        """Destroy camera sensor"""
        if self.camera and self.camera.is_alive:
            self.camera.destroy()
            self.camera = None

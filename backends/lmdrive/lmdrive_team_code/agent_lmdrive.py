import os
import json
import random
import datetime
import pathlib
import time
import imp
from collections import deque
import math

import cv2
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms

from leaderboard.autoagents import autonomous_agent
from backends.lmdrive.lmdrive_team_code.planner import RoutePlanner
from backends.lmdrive.lmdrive_team_code.pid_controller import PIDController
from backends.lmdrive.timm.models import create_model
from backends.lmdrive.lmdrive_lavis.common.registry import registry
import backends.lmdrive.lmdrive_lavis.models

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def rotate_lidar(lidar, angle):
    radian = np.deg2rad(angle)
    return lidar @ [
        [ np.cos(radian), np.sin(radian), 0, 0],
        [-np.sin(radian), np.cos(radian), 0, 0],
        [0,0,1,0],
        [0,0,0,1]
    ]

def lidar_to_raw_features(lidar):
    def preprocess(lidar_xyzr, lidar_painted=None):
        idx = (lidar_xyzr[:,0] > -1.2)&(lidar_xyzr[:,0] < 1.2)&(lidar_xyzr[:,1]>-1.2)&(lidar_xyzr[:,1]<1.2)
        idx = np.argwhere(idx)
        if lidar_painted is None:
            return np.delete(lidar_xyzr, idx, axis=0)
        else:
            return np.delete(lidar_xyzr, idx, axis=0), np.delete(lidar_painted, idx, axis=0)

    lidar_xyzr = preprocess(lidar)
    idxs = np.arange(len(lidar_xyzr))
    np.random.shuffle(idxs)
    lidar_xyzr = lidar_xyzr[idxs]
    lidar = np.zeros((40000, 4), dtype=np.float32)
    num_points = min(40000, len(lidar_xyzr))
    lidar[:num_points,:4] = lidar_xyzr[:num_points]
    lidar[np.isinf(lidar)] = 0
    lidar[np.isnan(lidar)] = 0
    lidar = rotate_lidar(lidar, -90).astype(np.float32)
    return lidar, num_points

def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    class Resize2FixedSize:
        def __init__(self, size):
            self.size = size
        def __call__(self, pil_img):
            pil_img = pil_img.resize(self.size)
            return pil_img

    if need_scale:
        if input_size == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            tfl.append(Resize2FixedSize((input_size, input_size)))
    
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))
    return transforms.Compose(tfl)

class LMDriveAgent(autonomous_agent.AutonomousAgent):
    def setup(self, config_path, hydra_config_path=None, gpu_id=0):
        conf_file = os.path.join(config_path, "lmdriver_config.py")
        if not os.path.exists(conf_file):
             conf_file = os.path.join(os.path.dirname(__file__), "lmdriver_config.py")
        
        self.config = imp.load_source("MainModel", conf_file).GlobalConfig()
        self.device = torch.device(f'cuda:{gpu_id}')
        
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.initialized = False
        
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)
        
        self.visual_feature_buffer = []
        self.turn_controller = PIDController(K_P=self.config.turn_KP, K_I=self.config.turn_KI, K_D=self.config.turn_KD, n=self.config.turn_n)
        self.speed_controller = PIDController(K_P=self.config.speed_KP, K_I=self.config.speed_KI, K_D=self.config.speed_KD, n=self.config.speed_n)
        
        model_cls = registry.get_model_class('vicuna_drive')
        print(f'Building LMDrive model on {self.device}...')
        self.net = model_cls(preception_model=self.config.preception_model,
                          preception_model_ckpt=self.config.preception_model_ckpt,
                          llm_model=self.config.llm_model,
                          max_txt_len=64,
                          use_notice_prompt=self.config.agent_use_notice)
        
        print(f'Loading weights from {self.config.lmdrive_ckpt}...')
        self.net.load_state_dict(torch.load(self.config.lmdrive_ckpt)["model"], strict=False)
        self.net.to(self.device)
        self.net.eval()
        
        self.softmax = torch.nn.Softmax(dim=1)
        self.prev_lidar = None
        self.prev_control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self.user_command = None 
        self.curr_instruction = None
        self.sample_rate = self.config.sample_rate * 2
        
        self.stuck_detector = 0
        self.force_move = 0

    def set_gps_reference(self, lat_ref, lon_ref):
        self.lat_ref = lat_ref
        self.lon_ref = lon_ref

    def _init(self):
        self._route_planner = RoutePlanner(5, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def reset_episode_state(self):
        """Reset per-scenario state while keeping model weights loaded."""
        self.step = -1
        self.initialized = False
        self.prev_lidar = None
        self.prev_control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self.user_command = None
        self.curr_instruction = None
        self.visual_feature_buffer = []
        self.stuck_detector = 0
        self.force_move = 0
        self._route_planner = None

        self.turn_controller = PIDController(
            K_P=self.config.turn_KP,
            K_I=self.config.turn_KI,
            K_D=self.config.turn_KD,
            n=self.config.turn_n,
        )
        self.speed_controller = PIDController(
            K_P=self.config.speed_KP,
            K_I=self.config.speed_KI,
            K_D=self.config.speed_KD,
            n=self.config.speed_n,
        )

    def sensors(self):
        return [
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "width": 1200, "height": 900, "fov": 100, "id": "rgb_front"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": -60.0, "width": 400, "height": 300, "fov": 100, "id": "rgb_left"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 60.0, "width": 400, "height": 300, "fov": 100, "id": "rgb_right"},
            {"type": "sensor.camera.rgb", "x": -1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 180.0, "width": 400, "height": 300, "fov": 100, "id": "rgb_rear"},
            {"type": "sensor.lidar.ray_cast", "x": 1.3, "y": 0.0, "z": 2.5, "roll": 0.0, "pitch": 0.0, "yaw": -90.0, "id": "lidar"},
            {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.05, "id": "imu"},
            {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.01, "id": "gps"},
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):
        rgb_front = cv2.cvtColor(input_data["rgb_front"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear = cv2.cvtColor(input_data["rgb_rear"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if math.isnan(compass):
            compass = 0.0

        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        
        lidar_data = input_data['lidar'][1][..., :4]
        if self.prev_lidar is not None:
            lidar_full = np.concatenate([lidar_data, self.prev_lidar])
        else:
            lidar_full = lidar_data
        self.prev_lidar = lidar_data
        lidar_processed, num_points = lidar_to_raw_features(lidar_full)

        next_wp, next_cmd = self._route_planner.run_step(gps)
        
        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local_target = R.T.dot(np.array([next_wp[0] - gps[0], next_wp[1] - gps[1]]))

        return {
            "rgb_front": rgb_front, "rgb_left": rgb_left, "rgb_right": rgb_right, "rgb_rear": rgb_rear,
            "gps": gps, "speed": speed, "compass": compass, "lidar": lidar_processed, "num_points": num_points,
            "target_point": local_target, "next_command": next_cmd.value
        }

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        self.step += 1
        tick_data = self.tick(input_data)

        # 1. Warm-up phase: Give initial momentum for the first 20 frames
        if self.step < 20:
            control = carla.VehicleControl(steer=0.0, throttle=0.5, brake=0.0)
            self.prev_control = control
            return control

        # 2. Control Smoothing (Strictly align with step % 2)
        if self.step % 2 != 0 and self.step > 4:
            return self.prev_control

        # 3. Instruction Handling & Buffer Management
        if self.user_command is None:
            raise ValueError("LMDriveAgent requires a user_command (instruction) to be set.")

        # Reset buffer if instruction changes or buffer limit reached (Official logic)
        if self.curr_instruction is not None and self.curr_instruction != self.user_command:
            self.visual_feature_buffer = []
        if len(self.visual_feature_buffer) > 400:
            self.visual_feature_buffer = []
            
        self.curr_instruction = self.user_command

        # 4. Prepare model inputs
        model_input = {
            "rgb_front": self.rgb_front_transform(Image.fromarray(tick_data["rgb_front"])).unsqueeze(0).to(self.device),
            "rgb_left": self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"])).unsqueeze(0).to(self.device),
            "rgb_right": self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"])).unsqueeze(0).to(self.device),
            "rgb_rear": self.rgb_right_transform(Image.fromarray(tick_data["rgb_rear"])).unsqueeze(0).to(self.device),
            "rgb_center": self.rgb_center_transform(Image.fromarray(cv2.resize(tick_data["rgb_front"], (800, 600)))).unsqueeze(0).to(self.device),
            "target_point": torch.tensor(tick_data["target_point"]).to(self.device).view(1, 2).float(),
            "lidar": torch.from_numpy(tick_data["lidar"]).float().to(self.device).unsqueeze(0),
            "num_points": torch.tensor([tick_data["num_points"]]).to(self.device).unsqueeze(0),
            "velocity": torch.tensor([tick_data["speed"]]).to(self.device).view(1, 1).float(),
            "text_input": [self.curr_instruction]
        }

        image_embeds = self.net.visual_encoder(model_input)
        self.visual_feature_buffer.append(image_embeds)
        
        result_embeds = self.visual_feature_buffer[::self.sample_rate]
        if (len(self.visual_feature_buffer) - 1) % self.sample_rate != 0:
            result_embeds.append(self.visual_feature_buffer[-1])
        batched_embeds = torch.stack(result_embeds, 1)
        
        model_input['valid_frames'] = [batched_embeds.size(1)]
        
        # Inference
        with torch.cuda.amp.autocast(enabled=True):
            waypoints, is_end = self.net(model_input, inference_mode=True, image_embeds=batched_embeds)

        # Output parsing
        waypoints = waypoints[-1].view(5, 2)
        end_prob = self.softmax(is_end)[-1][1]
        
        if end_prob > 0.75:
            self.visual_feature_buffer = []

        # 5. PID Control Logic
        steer, throttle, brake = self.control_pid_logic(waypoints, tick_data["speed"])
        
        # --- Official Logic: Thresholding & Smoothing ---
        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0
        # -------------------------------------------------

        # --- Force Move (Stuck Recovery) Logic ---
        if tick_data["speed"] < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0
        
        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration
        
        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = 0.0
            self.force_move -= 1
        # ------------------------------------------
        
        control = carla.VehicleControl(steer=float(steer)*0.8, throttle=float(throttle), brake=float(brake))
        self.prev_control = control
        return control

    def control_pid_logic(self, waypoints, speed):
        waypoints = waypoints.data.cpu().numpy()
        waypoints[:, 1] *= -1 
        
        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / (desired_speed + 1e-6)) > self.config.brake_ratio
        
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        
        if speed < 0.01:
            angle = 0.0
        
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        
        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        
        # Official mutual exclusion logic
        if brake:
            throttle = 0.0
            brake = 1.0
        else:
            brake = 0.0
            
        return steer, throttle, brake

    def destroy(self):
        if hasattr(self, 'net'):
            del self.net
        self.visual_feature_buffer = []
        torch.cuda.empty_cache()

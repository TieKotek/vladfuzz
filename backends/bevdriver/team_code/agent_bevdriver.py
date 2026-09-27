import math
import os
import imp

import carla
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from leaderboard.autoagents import autonomous_agent
from backends.bevdriver.modeling.drive import Blip2VicunaDrive
from backends.bevdriver.team_code.pid_controller import PIDController
from backends.bevdriver.team_code.planner import RoutePlanner
from backends.bevdriver.team_code.utils import lidar_to_histogram_features, transform_2d_points

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_entry_point():
    return "BEVDriverAgent"


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        return pil_img.resize(self.size)


def create_carla_rgb_transform(input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
        input_size_num = input_size[-1]
    else:
        img_size = input_size
        input_size_num = input_size

    transforms_list = []
    if need_scale:
        if input_size_num == 112:
            transforms_list.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            transforms_list.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            transforms_list.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            transforms_list.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError(f"Unsupported crop size: {input_size_num}")

    transforms_list.append(transforms.CenterCrop(img_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))
    return transforms.Compose(transforms_list)


def rotate_lidar(lidar, angle):
    radian = np.deg2rad(angle)
    return lidar @ [
        [np.cos(radian), np.sin(radian), 0, 0],
        [-np.sin(radian), np.cos(radian), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def lidar_to_raw_features(lidar):
    mask = (lidar[:, 0] > -1.2) & (lidar[:, 0] < 1.2) & (lidar[:, 1] > -1.2) & (lidar[:, 1] < 1.2)
    lidar_xyzr = np.delete(lidar, np.argwhere(mask), axis=0)
    indices = np.arange(len(lidar_xyzr))
    np.random.shuffle(indices)
    lidar_xyzr = lidar_xyzr[indices]
    dense = np.zeros((40000, 4), dtype=np.float32)
    num_points = min(40000, len(lidar_xyzr))
    dense[:num_points, :4] = lidar_xyzr[:num_points]
    dense[np.isinf(dense)] = 0
    dense[np.isnan(dense)] = 0
    dense = rotate_lidar(dense, -90).astype(np.float32)
    return dense, num_points


def normalize_bevdriver_checkpoint(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("visual_encoder."):
            normalized["bev_encoder." + key[len("visual_encoder."):]] = value
        else:
            normalized[key] = value
    return normalized


def validate_bevdriver_load(incompatible_keys):
    allowed_missing_prefixes = (
        "bev_encoder.",
        "llm_model.base_model.model.model.",
    )
    allowed_missing_exact = {
        "llm_model.base_model.model.lm_head.weight",
    }
    allowed_unexpected = {
        "scene_description_head.weight",
    }

    bad_missing = [
        key for key in incompatible_keys.missing_keys
        if key not in allowed_missing_exact and not key.startswith(allowed_missing_prefixes)
    ]
    bad_unexpected = [
        key for key in incompatible_keys.unexpected_keys
        if key not in allowed_unexpected
    ]

    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "BEVDriver checkpoint mismatch. "
            f"Unexpected missing keys: {bad_missing[:20]}; "
            f"unexpected checkpoint keys: {bad_unexpected[:20]}"
        )

    print(
        "BEVDriver checkpoint load validated: "
        f"missing={len(incompatible_keys.missing_keys)} "
        f"(expected base LLaMA + encoder weights), "
        f"unexpected={len(incompatible_keys.unexpected_keys)}"
    )


class BEVDriverAgent(autonomous_agent.AutonomousAgent):
    def setup(self, config_path, hydra_config_path=None, gpu_id=0):
        conf_file = os.path.join(config_path, "bevdriver_config.py")
        if not os.path.exists(conf_file):
            conf_file = os.path.join(os.path.dirname(__file__), "bevdriver_config.py")

        self.config = imp.load_source("BEVDriverConfig", conf_file).GlobalConfig()
        self.device = torch.device(f"cuda:{gpu_id}")
        self.track = autonomous_agent.Track.SENSORS

        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        print(f"Building BEVDriver model on {self.device}...")
        self.net = Blip2VicunaDrive(
            encoder_model=self.config.encoder_model,
            encoder_model_ckpt=self.config.encoder_model_ckpt,
            llm_model=self.config.llm_model,
            bert_model=self.config.bert_model,
            max_txt_len=64,
        )
        print(f"Loading weights from {self.config.bevdriver_ckpt}...")
        checkpoint = torch.load(self.config.bevdriver_ckpt, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        state_dict = normalize_bevdriver_checkpoint(state_dict)
        incompatible_keys = self.net.load_state_dict(state_dict, strict=False)
        validate_bevdriver_load(incompatible_keys)
        self.net.to(self.device)
        self.net.eval()

        self.softmax = torch.nn.Softmax(dim=1)
        self.sample_rate = self.config.sample_rate * 2
        self.reset_episode_state()

    def set_gps_reference(self, lat_ref, lon_ref):
        self.lat_ref = lat_ref
        self.lon_ref = lon_ref

    def _init(self):
        self._route_planner = RoutePlanner(5, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def reset_episode_state(self):
        self.step = -1
        self.initialized = False
        self.prev_lidar_hist = None
        self.prev_raw_lidar = None
        self.prev_control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self.visual_feature_buffer = []
        self.user_command = None
        self.curr_instruction = None
        self._route_planner = None
        self.stuck_detector = 0
        self.force_move = 0
        self.turn_controller = PIDController(K_P=self.config.turn_KP, K_I=self.config.turn_KI, K_D=self.config.turn_KD, n=self.config.turn_n)
        self.speed_controller = PIDController(K_P=self.config.speed_KP, K_I=self.config.speed_KI, K_D=self.config.speed_KD, n=self.config.speed_n)

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

    def _get_position(self, gps):
        return (gps - self._route_planner.mean) * self._route_planner.scale

    def _command_to_measurements(self, command_value, speed):
        one_hot = np.zeros(6, dtype=np.float32)
        index = int(command_value) - 1
        if index < 0 or index >= 6:
            index = 3
        one_hot[index] = 1.0
        return np.concatenate([one_hot, np.array([speed], dtype=np.float32)])

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

        pos = self._get_position(gps)

        raw_lidar = input_data["lidar"][1][..., :4]
        if self.prev_raw_lidar is not None:
            raw_lidar_full = np.concatenate([raw_lidar, self.prev_raw_lidar])
        else:
            raw_lidar_full = raw_lidar
        self.prev_raw_lidar = raw_lidar
        _, num_points = lidar_to_raw_features(raw_lidar_full)

        lidar_points = input_data["lidar"][1][:, :3].copy()
        lidar_points[:, 1] *= -1
        full_lidar = transform_2d_points(
            lidar_points,
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
        )
        lidar_hist = lidar_to_histogram_features(full_lidar, crop=224)
        if self.step % 2 == 0 or self.step < 4 or self.prev_lidar_hist is None:
            self.prev_lidar_hist = lidar_hist

        next_wp, next_cmd = self._route_planner.run_step(pos)
        theta = compass + np.pi / 2
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local_target = rotation.T.dot(np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]]))

        return {
            "rgb_front": rgb_front,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "rgb_rear": rgb_rear,
            "speed": speed,
            "lidar": self.prev_lidar_hist,
            "num_points": num_points,
            "target_point": local_target.astype(np.float32),
            "measurements": self._command_to_measurements(next_cmd.value, speed),
        }

    def update_and_collect(self, image_embeds):
        self.visual_feature_buffer.append(image_embeds)
        result = self.visual_feature_buffer[::self.sample_rate]
        if (len(self.visual_feature_buffer) - 1) % self.sample_rate != 0:
            result.append(self.visual_feature_buffer[-1])
        return torch.stack(result, 1)

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1
        tick_data = self.tick(input_data)

        if self.step < 20:
            control = carla.VehicleControl(steer=0.0, throttle=0.5, brake=0.0)
            self.prev_control = control
            return control

        if self.step % 2 != 0 and self.step > 4:
            return self.prev_control

        if self.user_command is None:
            raise ValueError("BEVDriverAgent requires a user_command (instruction) to be set.")

        if self.curr_instruction is not None and self.curr_instruction != self.user_command:
            self.visual_feature_buffer = []
        if len(self.visual_feature_buffer) > 400:
            self.visual_feature_buffer = []

        self.curr_instruction = self.user_command

        model_input = {
            "rgb": self.rgb_front_transform(Image.fromarray(tick_data["rgb_front"])).unsqueeze(0).to(self.device).float(),
            "rgb_left": self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"])).unsqueeze(0).to(self.device).float(),
            "rgb_right": self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"])).unsqueeze(0).to(self.device).float(),
            "rgb_rear": self.rgb_right_transform(Image.fromarray(tick_data["rgb_rear"])).unsqueeze(0).to(self.device).float(),
            "rgb_center": self.rgb_center_transform(Image.fromarray(cv2.resize(tick_data["rgb_front"], (800, 600)))).unsqueeze(0).to(self.device).float(),
            "lidar": torch.from_numpy(tick_data["lidar"]).float().unsqueeze(0).to(self.device),
            "target_point": torch.from_numpy(tick_data["target_point"]).view(1, 2).float().to(self.device),
            "measurements": torch.from_numpy(tick_data["measurements"]).view(1, -1).float().to(self.device),
        }

        image_embeds = self.net.visual_encoder(model_input)
        image_embeds = self.update_and_collect(image_embeds)
        model_input["valid_frames"] = [image_embeds.size(1)]
        model_input["text_input"] = [self.curr_instruction]

        with torch.cuda.amp.autocast(enabled=True):
            waypoints, is_end = self.net(model_input, inference_mode=True, image_embeds=image_embeds)

        waypoints = waypoints[-1].view(5, 2)
        end_prob = self.softmax(is_end)[-1][1]
        steer, throttle, brake, _ = self.control_pid(waypoints, tick_data["speed"])

        if end_prob > 0.75:
            self.visual_feature_buffer = []

        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

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

        control = carla.VehicleControl()
        control.steer = float(steer) * 0.8
        control.throttle = float(throttle)
        control.brake = float(brake)
        self.prev_control = control
        return control

    def control_pid(self, waypoints, velocity):
        waypoints = waypoints.data.cpu().numpy()
        waypoints[:, 1] *= -1
        speed = velocity
        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / max(desired_speed, 1e-4)) > self.config.brake_ratio

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if speed < 0.01:
            angle = np.array(0.0)
        steer = np.clip(self.turn_controller.step(angle), -1.0, 1.0)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = np.clip(self.speed_controller.step(delta), 0.0, self.config.max_throttle)

        if brake:
            throttle = 0.0
            brake = 1.0
        else:
            brake = 0.0

        metadata = {
            "speed": float(speed),
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
        }
        return steer, throttle, brake, metadata

    def destroy(self):
        del self.net

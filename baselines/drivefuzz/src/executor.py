#!/usr/bin/env python3

# Python packages
import glob
import os
import sys
import argparse
import json
from subprocess import Popen, PIPE
import signal
import numpy as np
import random
import time
import math
import traceback
import atexit
import faulthandler

try:
    import docker
except ModuleNotFoundError:
    docker = None

import config
import constants as c
from fuzz_utils import quaternion_from_euler, get_carla_transform

project_root = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

config.set_carla_api_path()
try:
    import carla
except ModuleNotFoundError as e:
    print("Carla module not found. Make sure you have built Carla.")
    proj_root = config.get_proj_root()
    print("Try `cd {}/carla && make PythonAPI' if not.".format(proj_root))
    exit(-1)

from vladfuzz_runtime.cuda_diagnostics import CudaMemoryDiagnostics
from vladfuzz_runtime.oracle import collision_actor_type, collision_impulse, should_count_collision
# VLAD-Fuzz baseline integration: classify CARLA RPC failures consistently
# with the main framework so invalid attempts are restarted, not scored.
from vladfuzz_runtime.infrastructure import (
    CARLA_INFRASTRUCTURE_EXIT_CODE,
    is_carla_infrastructure_error,
)

import DummyWorld

import pygame

client = None
tm = None
list_spawn_points = None
active_actor_ids = set()
CUDA_OOM_EXIT_CODE = 86
SENSOR_FAILURE_EXIT_CODE = 87


def _is_cuda_oom(exc):
    return (
        exc.__class__.__name__ == "OutOfMemoryError"
        or "CUDA out of memory" in str(exc)
    )


def _is_sensor_data_failure(exc):
    """Recognize Leaderboard sensor starvation without importing backend code here."""
    return (
        exc.__class__.__name__ == "SensorReceivedNoData"
        or "A sensor took too long to send their data" in str(exc)
    )


def _clear_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _register_active_actor(actor):
    if actor is not None and isinstance(actor, carla.Actor):
        active_actor_ids.add(actor.id)
    return actor


def _forget_active_actor(actor):
    if actor is not None and isinstance(actor, carla.Actor):
        active_actor_ids.discard(actor.id)


def _cleanup_active_actors():
    global active_actor_ids
    if not active_actor_ids or client is None:
        return
    try:
        world = client.get_world()
        commands = []
        for actor_id in list(active_actor_ids):
            actor = world.get_actor(actor_id)
            if actor is not None:
                try:
                    # VLAD-Fuzz baseline adapter: CARLA may warn when stop()
                    # is called on a sensor that has already unsubscribed.
                    # This only affects cleanup logging, so preserve the
                    # original destroy path and stop sensors only when the API
                    # reports an active listener.
                    if actor.type_id.startswith("sensor.") and getattr(actor, "is_listening", False):
                        actor.stop()
                except Exception:
                    pass
                commands.append(carla.command.DestroyActor(actor))
        if commands:
            client.apply_batch_sync(commands, False)
        active_actor_ids.clear()
    except Exception as exc:
        print("[-] Failed to clean active DriveFuzz actors:", exc)


atexit.register(_cleanup_active_actors)


def _ensure_project_root(conf):
    if conf.project_root and conf.project_root not in sys.path:
        sys.path.insert(0, conf.project_root)


def _load_vla_instruction(conf):
    row = _load_vla_mapping_row(conf)
    if row:
        return row.get("instruction") or conf.vla_instruction
    return conf.vla_instruction


def _load_vla_mapping_row(conf):
    if not conf.vla_instruction_map:
        return None

    with open(conf.vla_instruction_map, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("seed_file") == conf.cur_scenario:
                return row

    return None


def _resolve_project_path(conf, path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    root = conf.project_root or globals().get("project_root")
    return os.path.join(root, path)


def _load_vla_scenario_bounds(conf):
    row = _load_vla_mapping_row(conf)
    if not row:
        return None

    static_path = _resolve_project_path(conf, row.get("static_scenario"))
    if not static_path or not os.path.exists(static_path):
        return None

    try:
        with open(static_path, "r", encoding="utf-8") as file:
            static_data = json.load(file)
        center = static_data.get("scenario_center") or {}
        extent = static_data.get("scenario_extent") or {}
        return {
            "center": {
                "x": float(center["x"]),
                "y": float(center["y"]),
            },
            "extent": {
                "x": float(extent["x"]),
                "y": float(extent["y"]),
            },
            "static_scenario": static_path,
        }
    except Exception as exc:
        print("[-] Failed to load static scenario bounds:", exc)
        return None


def _is_location_within_bounds(location, bounds):
    if not bounds:
        return True
    center = bounds["center"]
    extent = bounds["extent"]
    return (
        center["x"] - extent["x"] <= location.x <= center["x"] + extent["x"]
        and center["y"] - extent["y"] <= location.y <= center["y"] + extent["y"]
    )


def _choose_player_blueprint(blueprint_library):
    blueprints = blueprint_library.filter("vehicle.tesla.model3")
    if not blueprints:
        raise RuntimeError("CARLA blueprint vehicle.tesla.model3 is not available")
    return blueprints[0]


def _on_collision(event, state):
    # print("COLLISION:", event)

    if event.frame > state.first_frame_id + state.num_frames:
        # ignore collision happened AFTER simulation ends
        # (can happen because of sluggish garbage collection of Carla)
        return

    if not should_count_collision(event):
        return

    state.crashed = True
    state.collision_event = event
    state.collision_actor_type = collision_actor_type(event)
    state.collision_impulse = collision_impulse(event)


def _on_invasion(event, state):
    # lane_types = set(x.type for x in event.crossed_lane_markings)
    # text = ['%r' % str(x).split()[-1] for x in lane_types]
    # self.hud.notification('Crossed line %s' % ' and '.join(text))

    if event.frame > state.first_frame_id + state.num_frames:
        return

    crossed_lanes = event.crossed_lane_markings
    for crossed_lane in crossed_lanes:
        if crossed_lane.lane_change == carla.LaneChange.NONE:
            # print("LANE INVASION:", event)
            state.laneinvaded = True
            state.laneinvasion_event.append(event)

    # print(crossed_lane.color, crossed_lane.lane_change, crossed_lane.type)
    # print(type(crossed_lane.color), type(crossed_lane.lane_change),
            # type(crossed_lane.type))


def _on_front_camera_capture(image):
    image.save_to_disk(f"/tmp/fuzzerdata/front-{image.frame}.jpg")


def _on_top_camera_capture(image):
    image.save_to_disk(f"/tmp/fuzzerdata/top-{image.frame}.jpg")

# def _on_view_image(self, image):
    # """
    # Callback when receiving a camera image
    # """
    # global _surface
    # array = np.frombuffer(image.data, dtype=np.dtype("uint8"))
    # array = np.reshape(array, (image.height, image.width, 4))
    # array = array[:, :, :3]
    # array = array[:, :, ::-1]
    # _surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


def set_camera(conf, player, spectator):
    if conf.view == c.BIRDSEYE:
        cam_over_player(player, spectator)
    elif conf.view == c.ONROOF:
        cam_chase_player(player, spectator)
    else: # fallthru default
        cam_chase_player(player, spectator)


def cam_chase_player(player, spectator):
    location = player.get_location()
    rotation = player.get_transform().rotation
    fwd_vec = rotation.get_forward_vector()

    # chase from behind
    constant = 4
    location.x -= constant * fwd_vec.x
    location.y -= constant * fwd_vec.y
    # and above
    location.z += 3
    rotation.pitch -= 5
    spectator.set_transform(
        carla.Transform(location, rotation)
    )


def cam_over_player(player, spectator):
    location = player.get_location()
    location.z += 100
    # rotation = player.get_transform().rotation
    rotation = carla.Rotation() # fix rotation for better sim performance
    rotation.pitch -= 90
    spectator.set_transform(
        carla.Transform(location, rotation)
    )


def is_player_on_puddle(player_loc, actor_frictions):
    for friction in actor_frictions:
        len_x = float(friction.attributes["extent_x"])
        len_y = float(friction.attributes["extent_y"])
        loc_x = friction.get_location().x
        loc_y = friction.get_location().y
        p1 = loc_x - len_x / 100
        p2 = loc_x + len_x / 100
        p3 = loc_y - len_y / 100
        p4 = loc_y + len_y / 100
        p_x = player_loc.x
        p_y = player_loc.y
        if p1 <= p_x and p_x <= p2 and p3 <= p_y and p_y <= p4:
            return True
        else:
            return False


def connect(conf):
    global client
    global tm

    client = carla.Client(conf.sim_host, conf.sim_port)
    client.set_timeout(10.0)
    try:
        client.get_server_version()
    except Exception as e:
        print("[-] Error: Check client connection.")
        sys.exit(-1)
    if conf.debug:
        print("Connected to:", client)

    tm = client.get_trafficmanager(conf.sim_tm_port)
    tm.set_synchronous_mode(True)
    if conf.debug:
        print("Traffic Manager Server:", tm)

    return (client, tm)


def switch_map(conf, town):
    """
    Switch map in the simulator and retrieve legitimate waypoints (a list of
    carla.Transform objects) in advance.
    """
    global client
    global list_spawn_points

    assert (client is not None)

    try:
        client.set_timeout(20) # Handle sluggish loading bug
        world = client.get_world()
        current_town = world.get_map().name.split("/")[-1]
        requested_town = str(town).split("/")[-1]

        if current_town != requested_town:
            if conf.debug:
                print("[*] Switching town to {} (slow)".format(town))

            try:
                settings = world.get_settings()
                if settings.synchronous_mode:
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = None
                    world.apply_settings(settings)
            except Exception as e:
                print("[-] Warning: failed to disable sync mode before map switch:", e)

            try:
                tm.set_synchronous_mode(False)
            except Exception as e:
                print("[-] Warning: failed to disable TM sync before map switch:", e)

            world = client.load_world(str(town)) # e.g., "/Game/Carla/Maps/Town01"
            time.sleep(5.0)
            if conf.debug:
                print("[+] Switched")
        elif conf.debug:
            print("[*] Reusing already loaded town {}".format(current_town))

        client.set_timeout(10.0)
        town_map = world.get_map()
        list_spawn_points = town_map.get_spawn_points()

    except Exception as e:
        print("[-] Error:", e)
        sys.exit(-1)


def lock_traffic_lights_green(world):
    traffic_lights = world.get_actors().filter("traffic.traffic_light*")
    for traffic_light in traffic_lights:
        traffic_light.set_state(carla.TrafficLightState.Green)
        traffic_light.set_green_time(1000.0)
        traffic_light.freeze(True)
    if traffic_lights:
        world.tick()


def simulate(conf, state, town, sp, wp, weather_dict, frictions_list, actors_list):
    # simulate() is always called by TestScenario instance,
    # so we won't need to switch map unless a new instance is created.
    # switch_map(conf, client, town)

    global client
    global tm

    # always reuse the existing client instance
    assert(client is not None)
    assert(tm is not None)

    # (client, tm) = connect(self.conf)
    # tm = client.get_trafficmanager(tm.get_port())
    # print("TM_CLIENT:", tm, tm.get_port())
    if hasattr(tm, "reset_traffic_lights"):
        tm.reset_traffic_lights() # XXX: might need this later
    retval = 0
    world = None
    sensors = []
    actor_vehicles = []
    actor_walkers = []
    actor_controllers = []
    actor_frictions = []
    vla_adapter = None
    player = None
    cuda_diag = CudaMemoryDiagnostics(getattr(conf, "cuda_diag_dir", None))

    try:
        # print("before world setting", time.time())
        client.set_timeout(10.0)
        world = client.get_world()
        if conf.debug:
            print("[debug] world:", world)

        if getattr(conf, "lock_traffic_lights_green", False):
            lock_traffic_lights_green(world)
        cuda_diag.snapshot(
            "simulation_start",
            campaign=getattr(state, "campaign_cnt", None),
            cycle=getattr(state, "cycle_cnt", None),
            mutation=getattr(state, "mutation", None),
            town=str(town),
        )

        town_map = world.get_map()
        if conf.debug:
            print("[debug] map:", town_map)

        scenario_bounds = _load_vla_scenario_bounds(conf)

        blueprint_library = world.get_blueprint_library()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / c.FRAME_RATE # FPS
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        frame_id = world.tick()
        init_frame_id = frame_id
        frame_0 = frame_id
        start_time = time.time()
        clock = pygame.time.Clock()

        # set weather
        weather = world.get_weather()
        weather.cloudiness = weather_dict["cloud"]
        weather.precipitation = weather_dict["rain"]
        weather.precipitation_deposits = weather_dict["puddle"]
        weather.wetness = weather_dict["wetness"]
        weather.wind_intensity = weather_dict["wind"]
        weather.fog_density = weather_dict["fog"]
        weather.sun_azimuth_angle = weather_dict["angle"]
        weather.sun_altitude_angle = weather_dict["altitude"]
        world.set_weather(weather)

        ros_pid = 0

        world.tick() # sync once with simulator

        # spawn player
        # how DriveFuzz spawns a player vehicle depends on
        # the autonomous driving agent
        player_bp = _choose_player_blueprint(blueprint_library)
        if player_bp.has_attribute("role_name"):
            player_bp.set_attribute("role_name", "hero")

        goal_loc = wp.location
        goal_rot = wp.rotation

        # mark goal position
        if conf.debug:
            world.debug.draw_box(
                box=carla.BoundingBox(
                    goal_loc,
                    carla.Vector3D(0.2, 0.2, 1.0)
                ),
                rotation=goal_rot,
                life_time=0,
                thickness=1.0,
                color=carla.Color(r=0,g=255,b=0)
            )

        if conf.agent_type == c.BASIC:
            from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

            player = _register_active_actor(world.try_spawn_actor(player_bp, sp))
            if player is None:
                print("[-] Failed spawning player")
                state.spawn_failed = True
                state.spawn_failed_object = 0 # player
                retval = -1
                return # trap to finally

            world.tick() # sync once with simulator
            player.set_simulate_physics(True)

            agent = BasicAgent(player)
            agent.set_destination((wp.location.x, wp.location.y, wp.location.z))
            print("[+] spawned BasicAgent")

        elif conf.agent_type == c.BEHAVIOR:
            from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error

            player = _register_active_actor(world.try_spawn_actor(player_bp, sp))
            if player is None:
                print("[-] Failed spawning player")
                state.spawn_failed = True
                state.spawn_failed_object = 0 # player
                retval = -1
                return # trap to finally

            world.tick() # sync once with simulator
            player.set_simulate_physics(True)

            agent = BehaviorAgent(
                player,
                ignore_traffic_light=True,
                behavior="cautious"
            )
            agent.set_destination(
                start_location=sp.location,
                end_location=wp.location,
                clean=True
            )

            # BehaviorAgent requires a World object to be supplied
            # but internally only dereferences the player object.
            # We'll simply create a DummyWorld instance to avoid the hassle.
            dummy_world = DummyWorld.DummyWorld(world, player)

            print("[+] spawned cautious BehaviorAgent")

        elif conf.agent_type == c.VLA:
            _ensure_project_root(conf)
            from vladfuzz_runtime.vla_backend_adapter import VLABackendAdapter

            player = _register_active_actor(world.try_spawn_actor(player_bp, sp))
            if player is None:
                print("[-] Failed spawning player")
                state.spawn_failed = True
                state.spawn_failed_object = 0
                retval = -1
                return

            world.tick()
            player.set_simulate_physics(True)

            vla_adapter = VLABackendAdapter(
                model_name=conf.vla_model,
                instruction=_load_vla_instruction(conf),
                host=conf.sim_host,
                port=conf.sim_port,
                gpu_id=conf.vla_gpu_id,
                reuse_backend=conf.reuse_vla_backend,
            )
            vla_adapter.setup(world, player, sp.location, wp.location)
            print("[+] spawned VLA backend {}".format(conf.vla_model))

        elif conf.agent_type == c.AUTOWARE:
            loc = sp.location
            rot = sp.rotation
            if conf.function == "collision":
                goal_ox = 0.0
                goal_oy = 0.0
                goal_oz = 1.0
                goal_ow = 0.0
            elif conf.function == "traction":
                goal_ox = 0.0
                goal_oy = 0.0
                goal_oz = -0.96
                goal_ow = 0.26
            elif conf.function == "eval-us":
                goal_ox = 0.0
                goal_oy = 0.0
                goal_oz = -0.01
                goal_ow = 0.9998
            elif conf.function == "eval-os":
                goal_ox = 0.0
                goal_oy = 0.0
                goal_oz = 0.679
                goal_ow = 0.733
            else:
                goal_quaternion = quaternion_from_euler(0.0, 0.0, goal_rot.yaw)
                goal_ox = goal_quaternion[0]
                goal_oy = goal_quaternion[1]
                goal_oz = goal_quaternion[2]
                goal_ow = goal_quaternion[3]
            sp_str = "{},{},{},{},{},{}".format(loc.x, loc.y, loc.z, rot.roll,
                    rot.pitch, rot.yaw * -1)
            goal_str = "{},{},{},{},{},{},{}".format(goal_loc.x, goal_loc.y,
                    goal_loc.z, goal_ox, goal_oy, goal_oz, goal_ow)

            if docker is None:
                raise RuntimeError("The docker Python package is required for DriveFuzz Autoware target.")
            docker_client = docker.from_env()
            proj_root = config.get_proj_root()
            xauth = os.path.join(os.getenv("HOME"), ".Xauthority")
            username = os.getenv("USER")
            vol_dict = {
                "{}/carla-autoware/autoware-contents".format(proj_root): {
                    "bind": "/home/autoware/autoware-contents",
                    "mode": "ro"
                    },
                "/tmp/.X11-unix": {
                    "bind": "/tmp/.X11-unix",
                    "mode": "rw"
                    },
                f"/home/{username}/.Xauthority": {
                    "bind": xauth,
                    "mode": "rw"
                    },
                "/tmp/fuzzerdata": {
                    "bind": "/tmp/fuzzerdata",
                    "mode": "rw"
                    }
                }
            env_dict = {
                "DISPLAY": os.getenv("DISPLAY"),
                "XAUTHORITY":xauth
                }

            autoware_cla = "{} \'{}\'".format(town_map.name, sp_str)
            print(autoware_cla)
            state.autoware_cmd = autoware_cla

            autoware_container = None
            killed = False
            while autoware_container is None:
                try:
                    autoware_container = docker_client.containers.run(
                            "carla-autoware:improved",
                            command=autoware_cla,
                            detach=True,
                            auto_remove=True,
                            name="autoware-{}".format(os.getenv("USER")),
                            volumes=vol_dict,
                            privileged=True,
                            network_mode="host",
                            runtime="nvidia",
                            environment=env_dict,
                        )
                except docker.errors.APIError as e:
                    print("[-] Could not launch docker:", e)
                    if "Conflict" in str(e):
                        os.system("docker rm -f autoware-{}".format(
                            os.getenv("USER")))
                        killed = True
                    time.sleep(1)
                except:
                    # https://github.com/docker/for-mac/issues/4957
                    print("[-] Fatal error. Check dmesg")
                    exit(-1)

            while True:
                running_container_list = docker_client.containers.list()
                if autoware_container in running_container_list:
                    break
                print("[*] Waiting for Autoware container to be launched")
                time.sleep(1)

            # wait for autoware bridge to spawn player vehicle
            autoware_agent_found = False
            i = 0
            while True:
                print("[*] Waiting for Autoware agent " + "."*i + "\r", end="")
                vehicles = world.get_actors().filter("*vehicle.*")
                for vehicle in vehicles:
                    if vehicle.attributes["role_name"] == "ego_vehicle":
                        autoware_agent_found = True
                        player = vehicle
                        print("\n    [*] found [{}] at {}".format(player.id,
                            player.get_location()))
                        break
                if autoware_agent_found:
                    break
                if i > 60:
                    print("\n something is wrong")
                    exit(-1)
                i += 1
                time.sleep(0.5)

            world.tick() # sync with simulator
            player.set_transform(sp)
            while True:
                world.tick() # spin until the player is moved to the sp
                if player.get_location().distance(sp.location) < 1:
                    break

        # print("after spawning ego_vehicle", time.time())

        # print("before spawning actors", time.time())

        # Attach collision detector
        collision_bp = blueprint_library.find('sensor.other.collision')
        sensor_collision = _register_active_actor(world.spawn_actor(collision_bp, carla.Transform(),
                attach_to=player))
        sensor_collision.listen(lambda event: _on_collision(event, state))
        sensors.append(sensor_collision)

        # Attach lane invasion sensor
        lanesensor_bp = blueprint_library.find("sensor.other.lane_invasion")
        sensor_lane = _register_active_actor(world.spawn_actor(lanesensor_bp, carla.Transform(),
                attach_to=player))
        sensor_lane.listen(lambda event: _on_invasion(event, state))
        sensors.append(sensor_lane)

        if conf.agent_type == c.BASIC or conf.agent_type == c.BEHAVIOR:
            # Attach RGB camera (front)
            rgb_camera_bp = blueprint_library.find("sensor.camera.rgb")

            rgb_camera_bp.set_attribute("image_size_x", "800")
            rgb_camera_bp.set_attribute("image_size_y", "600")
            rgb_camera_bp.set_attribute("fov", "105")

            # position relative to the parent actor (player)
            camera_tf = carla.Transform(carla.Location(z=1.8))

            # time in seconds between sensor captures - should sync w/ fps?
            # rgb_camera_bp.set_attribute("sensor_tick", "1.0")

            camera_front = _register_active_actor(world.spawn_actor(
                    rgb_camera_bp,
                    camera_tf,
                    attach_to=player,
                    attachment_type=carla.AttachmentType.Rigid
            ))

            camera_front.listen(lambda image: _on_front_camera_capture(image))

            sensors.append(camera_front)

            camera_tf = carla.Transform(
                carla.Location(z=50.0),
                carla.Rotation(pitch=-90.0)
            )
            camera_top = _register_active_actor(world.spawn_actor(
                    rgb_camera_bp,
                    camera_tf,
                    attach_to=player,
                    attachment_type=carla.AttachmentType.Rigid
            ))

            camera_top.listen(lambda image: _on_top_camera_capture(image))
            sensors.append(camera_top)

        world.tick() # sync with simulator

        # get vehicle's maximum steering angle
        physics_control = player.get_physics_control()
        max_steer_angle = 0
        for wheel in physics_control.wheels:
            if wheel.max_steer_angle > max_steer_angle:
                max_steer_angle = wheel.max_steer_angle

        spectator = world.get_spectator() if conf.follow_spectator else None
        if spectator is not None:
            set_camera(conf, player, spectator)

        # spawn friction triggers
        friction_bp = blueprint_library.find('static.trigger.friction')
        for friction in frictions_list:
            friction_bp.set_attribute('friction', str(friction["level"]))
            friction_bp.set_attribute('extent_x', str(friction["size"][0]))
            friction_bp.set_attribute('extent_y', str(friction["size"][1]))
            friction_bp.set_attribute('extent_z', str(friction["size"][2]))

            friction_sp_transform = get_carla_transform(
                friction["spawn_point"]
            )
            friction_size_loc = carla.Location(
                friction["size"][0],
                friction["size"][1],
                friction["size"][2]
            )

            friction_trigger = _register_active_actor(world.try_spawn_actor(
                                friction_bp, friction_sp_transform))

            if friction_trigger is None:
                print("[-] Failed spawning lvl {} puddle at ({}, {})".format(
                    friction["level"],
                    friction_sp_transform.location.x,
                    friction_Sp_transform.location.y)
                    )

                state.spawn_failed = True
                state.spawn_failed_object = friction
                retval = -1
                return
            actor_frictions.append(friction_trigger) # to destroy later

            # Optional for visualizing trigger (for debugging)
            if conf.debug:
                world.debug.draw_box(
                    box=carla.BoundingBox(
                        friction_sp_transform.location,
                        friction_size_loc * 1e-2
                    ),
                    rotation=friction_sp_transform.rotation,
                    life_time=0,
                    thickness=friction["level"] * 1, # the stronger the thicker
                    color=carla.Color(r=0,g=0,b=255)
                )
            print("[+] New puddle [%d] @(%.2f, %.2f) lvl %.2f"  %(
                    friction_trigger.id,
                    friction_sp_transform.location.x,
                    friction_sp_transform.location.y,
                    friction["level"])
                )

        # spawn actors
        vehicle_bp = blueprint_library.find("vehicle.bmw.grandtourer")
        vehicle_bp.set_attribute("color", "255,0,0")
        walker_bp = blueprint_library.find("walker.pedestrian.0001") # 0001~0014
        walker_controller_bp = blueprint_library.find('controller.ai.walker')

        for actor in actors_list:
            actor_sp = get_carla_transform(actor["spawn_point"])
            if actor["type"] == c.VEHICLE: # vehicle
                actor_vehicle = _register_active_actor(world.try_spawn_actor(vehicle_bp, actor_sp))
                actor_nav = c.NAVTYPE_NAMES[actor["nav_type"]]
                if actor_vehicle is None:
                    actor_str = c.ACTOR_NAMES[actor["type"]]
                    print("[-] Failed spawning {} {} at ({}, {})".format(
                        actor_nav, actor_str, actor_sp.location.x,
                        actor_sp.location.y)
                        )

                    state.spawn_failed = True
                    state.spawn_failed_object = actor
                    retval = -1
                    return # trap to finally

                actor_vehicles.append(actor_vehicle)
                print("[+] New %s vehicle [%d] @(%.2f, %.2f) yaw %.2f" %(
                        actor_nav,
                        actor_vehicle.id,
                        actor_sp.location.x,
                        actor_sp.location.y,
                        actor_sp.rotation.yaw)
                    )

            elif actor["type"] == c.WALKER: # walker
                actor_walker = _register_active_actor(world.try_spawn_actor(walker_bp, actor_sp))
                actor_nav = c.NAVTYPE_NAMES[actor["nav_type"]]
                if actor_walker is None:
                    actor_str = c.ACTOR_NAMES[actor["type"]]
                    print("[-] Failed spawning {} {} at ({}, {})".format(
                        actor_nav, actor_str, actor_sp.location.x,
                        actor_sp.location.y)
                        )

                    state.spawn_failed = True
                    state.spawn_failed_object = actor
                    retval = -1
                    return # trap to finally

                actor_walkers.append(actor_walker)
                print("[+] New %s walker [%d] @(%.2f, %.2f) yaw %.2f" %(
                        actor_nav,
                        actor_walker.id,
                        actor_sp.location.x,
                        actor_sp.location.y,
                        actor_sp.rotation.yaw)
                    )
        # print("after spawning actors", time.time())

        if conf.agent_type == c.AUTOWARE:
            # print("before launching autoware", time.time())
            num_vehicle_topics = len(actor_vehicles)
            num_walker_topics = 0
            if len(actor_walkers) > 0:
                num_walker_topics = 2
            # clock = pygame.time.Clock()
            i = 0
            while True:
                print("[*] Waiting for Autoware nodes " + "." * i + "\r", end="")
                proc1 = Popen(["rostopic", "list"], stdout=PIPE)
                proc2 = Popen(["wc", "-l"], stdin=proc1.stdout, stdout=PIPE)
                proc1.stdout.close()
                output = proc2.communicate()[0]

                num_topics = c.WAIT_AUTOWARE_NUM_TOPICS + num_vehicle_topics + num_walker_topics
                if int(output) >= num_topics:
                    # FIXME: hardcoding the num of topics :/
                    # on top of that, each vehicle adds one topic, and any walker
                    # contribute to two pedestrian topics.
                    print("")
                    break
                i += 1
                if i == 60:
                    print("    [-] something went wrong while launching Autoware.")
                    raise KeyboardInterrupt
                time.sleep(0.5)

            world.tick()

            # exec a detached process that monitors the output of Autoware's
            # decision-maker state, with which we can get an idea of when Autoware
            # thinks it has reached the goal
            proc_state = Popen(["rostopic echo /decision_maker/state"],
                    shell=True, stdout=PIPE, stderr=PIPE)

            # set_camera(conf, player, spectator)

            # Wait for Autoware (esp, for Town04)
            while True:
                output_state = proc_state.stdout.readline()
                if b"---" in output_state:
                    output_state = proc_state.stdout.readline()
                if b"VehicleReady" in output_state:
                    break
                time.sleep(0.5)

            pub_topic = "/move_base_simple/goal"
            msg_type = "geometry_msgs/PoseStamped"
            goal_hdr = "header: {stamp: now, frame_id: \'map\'}"
            goal_pose = "pose: {position: {x: %.6f, y: %.6f, z: 0}, orientation: {x: %.6f, y: %.6f, z: %.6f, w: %.6f}}" %(goal_loc.x, (-1) * float(goal_loc.y), goal_ox, goal_oy, goal_oz, goal_ow)
            goal_msg = "'{" + goal_hdr + ", " + goal_pose + "}'"
            pub_cmd = "rostopic pub --once {} {} {} > /dev/null".format(pub_topic, msg_type, goal_msg)
            os.system(pub_cmd)
            if conf.debug:
                print(goal_msg)
            print("[carla] Goal published")
            time.sleep(1) # give some time (Autoware initialization is slow)
            state.autoware_goal = pub_cmd

            world.tick()
            # print("after launching autoware", time.time())

        # print("real simulation begins", time.time())
        # handle actor missions after Autoware's goal is published
        cnt_v = 0
        cnt_w = 0
        for actor in actors_list:
            actor_sp = get_carla_transform(actor["spawn_point"])
            actor_dp = get_carla_transform(actor["dest_point"])
            if actor["type"] == c.VEHICLE: # vehicle
                actor_vehicle = actor_vehicles[cnt_v]
                cnt_v += 1
                if actor["nav_type"] == c.LINEAR:
                    forward_vec = actor_sp.rotation.get_forward_vector()
                    actor_vehicle.set_target_velocity(forward_vec * actor["speed"])

                elif actor["nav_type"] == c.MANEUVER:
                    # set initial speed 0
                    # trajectory control happens in the control loop below
                    forward_vec = actor_sp.rotation.get_forward_vector()
                    actor_vehicle.set_target_velocity(forward_vec * 0)

                elif actor["nav_type"] == c.AUTOPILOT:
                    actor_vehicle.set_autopilot(True, tm.get_port())

                actor_vehicle.set_simulate_physics(True)
            elif actor["type"] == c.WALKER: # walker
                actor_walker = actor_walkers[cnt_w]
                cnt_w += 1
                if actor["nav_type"] == c.LINEAR:
                    forward_vec = actor_sp.rotation.get_forward_vector()
                    controller_walker = carla.WalkerControl()
                    controller_walker.direction = forward_vec
                    controller_walker.speed = actor["speed"]
                    actor_walker.apply_control(controller_walker)

                elif actor["nav_type"] == c.AUTOPILOT:
                    controller_walker = _register_active_actor(world.spawn_actor(
                            walker_controller_bp,
                            actor_sp,
                            actor_walker))

                    world.tick() # without this, walker vanishes
                    controller_walker.start()
                    controller_walker.set_max_speed(float(actor["speed"]))
                    controller_walker.go_to_location(actor_dp.location)

                elif actor["nav_type"] == c.IMMOBILE:
                    controller_walker = None

                if controller_walker: # can be None if immobile walker
                    actor_controllers.append(controller_walker)

        elapsed_time = 0
        start_time = time.time()

        yaw = sp.rotation.yaw

        player_loc = player.get_transform().location
        init_x = player_loc.x
        init_y = player_loc.y

        # SIMULATION LOOP FOR AUTOWARE and BasicAgent
        signal.signal(signal.SIGINT, signal.default_int_handler)
        faulthandler.enable(all_threads=True)

        try:
            # actual monitoring of the driving simulation begins here
            snapshot0 = world.get_snapshot()
            first_frame_id = snapshot0.frame
            first_sim_time = snapshot0.timestamp.elapsed_seconds

            last_frame_id = first_frame_id

            state.first_frame_id = first_frame_id
            state.sim_start_time = snapshot0.timestamp.platform_timestamp
            state.num_frames = 0
            state.elapsed_time = 0
            s_started = False
            s_stopped_frames = 0

            if conf.debug:
                print("[*] START DRIVING: {} {}".format(first_frame_id,
                    first_sim_time))

            while True:
                # Use sampling frequency of FPS*2 for precision!
                clock.tick(c.FRAME_RATE * 2)

                # Carla agents are running in synchronous mode,
                # so we need to send ticks. Not needed for Autoware
                if conf.agent_type == c.BASIC or conf.agent_type == c.BEHAVIOR or conf.agent_type == c.VLA:
                    world.tick()

                snapshot = world.get_snapshot()
                cur_frame_id = snapshot.frame
                cur_sim_time = snapshot.timestamp.elapsed_seconds

                if cur_frame_id <= last_frame_id:
                    # skip if we got the same frame data as last
                    continue

                last_frame_id = cur_frame_id # update last
                state.num_frames = cur_frame_id - first_frame_id
                state.elapsed_time = cur_sim_time - first_sim_time

                player_transform = player.get_transform()
                player_loc = player_transform.location
                player_rot = player_transform.rotation

                if spectator is not None:
                    set_camera(conf, player, spectator)

                # Get speed
                vel = player.get_velocity()
                speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                speed_limit = player.get_speed_limit()

                try:
                    last_speed_limit = state.speed_lim[-1]
                except:
                    last_speed_limit = 0

                if speed_limit != last_speed_limit:
                    frame_speed_lim_changed = cur_frame_id

                state.speed.append(speed)
                state.speed_lim.append(speed_limit)

                print("(%.2f,%.2f)>(%.2f,%.2f)>(%.2f,%.2f) %.2f m left, %.2f/%d km/h   \r" %(
                    sp.location.x, sp.location.y, player_loc.x,
                    player_loc.y, goal_loc.x, goal_loc.y,
                    player_loc.distance(goal_loc),
                    speed, speed_limit), end="")

                if player.is_at_traffic_light():
                    traffic_light = player.get_traffic_light()
                    if traffic_light.get_state() == carla.TrafficLightState.Red:
                        # within red light triggerbox
                        if state.on_red:
                            state.on_red_speed.append(speed)
                        else:
                            state.on_red = True
                            state.on_red_speed = list()
                else:
                    # not at traffic light
                    if state.on_red:
                        # out of red light triggerbox
                        state.on_red = False
                        stopped_at_red = False
                        for i, ors in enumerate(state.on_red_speed):
                            if ors < 0.1:
                                stopped_at_red = True

                        if not stopped_at_red:
                            state.red_violation = True

                # world.debug.draw_point(
                        # player_loc + carla.Location(z=10),
                        # size=0.1,
                        # life_time=0.1,
                        # color=carla.Color(255, 0, 0)
                    # )
                # set_camera(player, spectator)

                if conf.agent_type == c.BASIC:
                    # for carla agents, we should apply controls ourselves
                    # XXX: check and resolve BehaviorAgent's run_step issue of
                    # not being able to get adjacent waypoints
                    control = agent.run_step(debug=True)
                    player.apply_control(control)

                elif conf.agent_type == c.BEHAVIOR:
                    agent.update_information(dummy_world)
                    agent.get_local_planner().set_speed(speed_limit)

                    control = agent.run_step(debug=True)
                    player.apply_control(control)

                elif conf.agent_type == c.AUTOWARE:
                    # autoware does it on its own. we just retrieve the
                    # control for state computation
                    control = player.get_control()

                elif conf.agent_type == c.VLA:
                    control = vla_adapter.run_step(snapshot.timestamp)
                    player.apply_control(control)

                state.cont_throttle.append(control.throttle)
                state.cont_brake.append(control.brake)
                state.cont_steer.append(control.steer)
                steer_angle = control.steer * max_steer_angle
                state.steer_angle_list.append(steer_angle)

                current_yaw = player_rot.yaw
                state.yaw_list.append(current_yaw)

                yaw_diff = current_yaw - yaw
                # Yaw range is -180 ~ 180. When vehicle's yaw is oscillating
                # b/w -179 and 179, yaw_diff can be messed up even if the
                # diff is very small. Assuming that it's unrealistic that
                # a vehicle turns more than 180 degrees in under 1/20 seconds,
                # just round the diff around 360.
                if yaw_diff > 180:
                    yaw_diff = 360 - yaw_diff
                elif yaw_diff < -180:
                    yaw_diff = 360 + yaw_diff

                yaw_rate = yaw_diff * c.FRAME_RATE
                state.yaw_rate_list.append(yaw_rate)
                yaw = current_yaw

                # uncomment below to follow along the player
                # set_camera(conf, player, spectator)

                for v in actor_vehicles:
                    try:
                        actor_location = v.get_location()
                    except RuntimeError:
                        continue
                    dist = player_loc.distance(actor_location)
                    if dist < state.min_dist:
                        state.min_dist = dist

                for w in actor_walkers:
                    try:
                        actor_location = w.get_location()
                    except RuntimeError:
                        continue
                    dist = player_loc.distance(actor_location)
                    if dist < state.min_dist:
                        state.min_dist = dist

                # Get the lateral speed
                player_right_vec = player_rot.get_right_vector()

                # [Note]
                # Lateral velocity is a scalar projection of velocity vector.
                # A: velocity vector.
                # B: right vector. B is a unit vector, thus |B| = 1
                # lat_speed = |A| * cos(theta)
                # As dot_product(A, B) = |A| * |B| * cos(theta),
                # lat_speed = dot_product(A, B) / |B|
                # Given that |B| is always 1,
                # we get lat_speed = dot_product(A, B), which is equivalent to
                # lat_speed = vel.x * right_vel.x + vel.y * right_vel.y

                lat_speed = abs(vel.x * player_right_vec.x + vel.y * player_right_vec.y)
                lat_speed *= 3.6 # m/s to km/h
                state.lat_speed_list.append(lat_speed)

                player_fwd_vec = player_rot.get_forward_vector()
                lon_speed = abs(vel.x * player_fwd_vec.x + vel.y * player_fwd_vec.y)
                lon_speed *= 3.6
                state.lon_speed_list.append(lon_speed)

                # Handle actor maneuvers
                if conf.strategy == c.TRAJECTORY:
                    actor = actors_list[0]
                    maneuvers = actor["maneuvers"]

                    maneuver_id = int(state.num_frames / c.FRAMES_PER_TIMESTEP)
                    if maneuver_id < 5:
                        maneuver = maneuvers[maneuver_id]

                        if maneuver[2] == 0:
                            # print(f"\nPerforming maneuver #{maneuver_id} at frame {state.num_frames}")
                            # mark as done
                            maneuver[2] = state.num_frames

                            # retrieve the actual actor vehicle object
                            # there is only one actor in Trajectory mode
                            actor_vehicle = actor_vehicles[0]

                            # perform the action
                            actor_direction = maneuver[0]
                            actor_speed = maneuver[1]

                            forward_vec = get_carla_transform(
                                actor["spawn_point"]).rotation.get_forward_vector()

                            if actor_direction == 0: # forward

                                actor_vehicle.set_target_velocity(
                                    forward_vec * actor_speed
                                )

                        elif maneuver[2] > 0 and abs(maneuver[2] - state.num_frames) < 40:
                            # continuously apply lateral force to the vehicle
                            # for 40 frames (2 secs)
                            actor_direction = maneuver[0]
                            apex_degree = maneuver[1]

                            """
                            Model smooth lane changing through varying thetas
                            (theta)
                            45           * *
                            30       * *     * *
                            15     *             * *
                            0  * *                   *
                               0 5 10 15 20 25 30 35 40 (t = # frame)
                            """

                            theta_max = apex_degree
                            force_constant = 5 # should weigh by actor_speed?

                            t = abs(maneuver[2] - state.num_frames)
                            if t < 20:
                                theta = t * (theta_max / 20)
                            else:
                                theta = t * -1 * (theta_max / 20) + 2 * theta_max

                            if actor_direction != 0: # skip if fwd
                                if actor_direction == -1: # switch to left lane
                                    theta *= -1 # turn cc-wise
                                elif actor_direction == 1: # switch to right lane
                                    pass # turn c-wise

                                theta_rad = math.radians(theta)
                                sin = math.sin(theta_rad)
                                cos = math.cos(theta_rad)

                                x0 = forward_vec.x
                                y0 = forward_vec.y

                                x1 = cos * x0 - sin * y0
                                y1 = sin * x0 + cos * y0

                                dir_vec = carla.Vector3D(x=x1, y=y1, z=0.0)
                                actor_vehicle.set_target_velocity(
                                    dir_vec * force_constant
                                )

                # Check Autoware-defined destination
                # VehicleReady\nDriving\nMoving\nLaneArea\nCruise\nStraight\nDrive\nGo\n
                # VehicleReady\nWaitOrder\nStopping\nWaitDriveReady\n

                # Check destination
                dist_to_goal = player_loc.distance(goal_loc)

                if conf.agent_type == c.AUTOWARE:
                    if not conf.function.startswith("eval"):
                        output_state = proc_state.stdout.readline()
                        if b"---" in output_state:
                            output_state = proc_state.stdout.readline()
                        if b"Go" in output_state:
                            s_started = True
                        elif b"nWaitDriveReady" in output_state and s_started:
                            print("\n[*] (Autoware) Reached the destination")
                            print("      dist to goal:", dist_to_goal)

                            if dist_to_goal > conf.goal_distance_threshold and state.num_frames > 300:
                                state.other_error = "goal"
                                state.other_error_val = dist_to_goal

                            retval = 0
                            break
                elif conf.agent_type == c.BASIC:
                    if hasattr(agent, "done") and agent.done():
                        print("\n[*] (BasicAgent) Reached the destination")

                        if dist_to_goal > conf.goal_distance_threshold and state.num_frames > 300:
                            state.other_error = "goal"
                            state.other_error_val = dist_to_goal

                        break

                elif conf.agent_type == c.BEHAVIOR:
                    lp = agent.get_local_planner()
                    if len(lp.waypoints_queue) == 0:
                        print("\n[*] (BehaviorAgent) Reached the destination")

                        if dist_to_goal > conf.goal_distance_threshold and state.num_frames > 300:
                            state.other_error = "goal"
                            state.other_error_val = dist_to_goal

                        break

                elif conf.agent_type == c.VLA:
                    if dist_to_goal < conf.goal_distance_threshold:
                        print("\n[*] (VLA) Reached the destination")
                        break

                if dist_to_goal < conf.goal_distance_threshold:
                    print("\n[*] (Carla heuristic) Reached the destination")
                    retval = 0
                    break

                # Check speeding
                if conf.check_dict["speed"]:
                    # allow T seconds to slow down if speed limit suddenly
                    # decreases
                    T = 3 # 0 for strict checking
                    if (speed > speed_limit and
                        cur_frame_id > frame_speed_lim_changed + T * c.FRAME_RATE):
                        print("\n[*] Speed violation: {} km/h on a {} km/h road".format(
                            speed, speed_limit))
                        state.speeding = True
                        retval = 1
                        break

                # Check whether the ego vehicle left the exported VLAD-Fuzz
                # static scenario region. This prevents wrong-route executions
                # from consuming the full timeout budget.
                if conf.check_dict["out_of_bounds"] and not _is_location_within_bounds(player_loc, scenario_bounds):
                    print("\n[*] Out-of-bounds detected: ({:.2f}, {:.2f}, {:.2f})".format(
                        player_loc.x, player_loc.y, player_loc.z))
                    state.out_of_bounds = True
                    state.out_of_bounds_location = {
                        "x": round(player_loc.x, 2),
                        "y": round(player_loc.y, 2),
                        "z": round(player_loc.z, 2),
                    }
                    state.out_of_bounds_bounds = scenario_bounds
                    retval = 1
                    break

                # Check crash
                if conf.check_dict["crash"]:
                    if state.crashed:
                        print("\n[*] Collision detected against {}: {:.2f}".format(
                            state.collision_actor_type, state.elapsed_time))
                        retval = 1
                        break

                # Check lane violation. VLAD-Fuzz VLA integration records
                # this event but evaluates it after the scenario, matching the
                # shared CarlaScenario oracle. Preserve upstream early stopping
                # for DriveFuzz's original non-VLA targets.
                if conf.check_dict["lane"] and conf.agent_type != c.VLA:
                    if state.laneinvaded:
                        print("\n[*] Lane invasion detected: %.2f" %(
                            state.elapsed_time))
                        retval = 1
                        break

                # Check traffic light violation
                if conf.check_dict["red"]:
                    if state.red_violation:
                        print("\n[*] Red light violation detected: %.2f" %(
                            state.elapsed_time))
                        retval = 1
                        break

                # Check inactivity
                if speed < 1: # km/h
                    state.stuck_duration += 1
                else:
                    state.stuck_duration = 0

                if conf.check_dict["stuck"]:
                    if state.stuck_duration > (conf.timeout * c.FRAME_RATE):
                        state.stuck = True
                        print("\n[*] Stuck for too long: %d" %(state.stuck_duration))
                        retval = 1
                        break

                if conf.check_dict["other"]:
                    # VLAD-Fuzz VLA integration: the frame cap is authoritative
                    # when supplied; the upstream simulated-time cap remains a
                    # fallback for original DriveFuzz targets.
                    frame_limit_reached = (
                        conf.max_simulation_frames is not None
                        and state.num_frames >= conf.max_simulation_frames
                    )
                    seconds_limit_reached = (
                        conf.max_simulation_frames is None
                        and state.num_frames > int(conf.max_simulation_seconds * c.FRAME_RATE)
                    )
                    if frame_limit_reached or seconds_limit_reached:
                        print("\n[*] Simulation taking too long")
                        state.other_error = "timeout"
                        state.other_error_val = state.num_frames
                        retval = 1
                        break
                    if state.other_error:
                        print("\n[*] Other error: %d" %(state.signal))
                        retval = 1
                        break

        except KeyboardInterrupt:
            print("quitting")
            retval = 128

        # jump to finally
        return

    except Exception as e:
        # update states
        # state.num_frames = frame_id - frame_0
        # state.elapsed_time = time.time() - start_time

        print("[-] Runtime error:")
        traceback.print_exc()
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # print("   (line #{0}) {1}".format(exc_tb.tb_lineno, exc_type))

        if _is_sensor_data_failure(e):
            # VLAD-Fuzz baseline integration: sensor starvation indicates a
            # degraded CARLA instance, not a behavioral failure. The batch
            # supervisor archives this attempt and restarts the simulator.
            print("[-] VLA sensor infrastructure failure; restarting CARLA and this seed.")
            retval = SENSOR_FAILURE_EXIT_CODE
        elif is_carla_infrastructure_error(e):
            # VLAD-Fuzz baseline integration: an unavailable simulator invalidates
            # the attempt and must immediately return control to the supervisor.
            print("[-] CARLA RPC infrastructure failure; restarting CARLA and this seed.")
            retval = CARLA_INFRASTRUCTURE_EXIT_CODE
        elif _is_cuda_oom(e):
            print("[-] CUDA OOM detected; aborting this DriveFuzz process for a clean restart.")
            cuda_diag.record_oom(
                e,
                campaign=getattr(state, "campaign_cnt", None),
                cycle=getattr(state, "cycle_cnt", None),
                mutation=getattr(state, "mutation", None),
                town=str(town),
            )
            _clear_cuda_cache()
            retval = CUDA_OOM_EXIT_CODE
        else:
            retval = 1

    finally:
        # Finalize simulation
        # rospy.signal_shutdown("fin")

        if retval in (SENSOR_FAILURE_EXIT_CODE, CARLA_INFRASTRUCTURE_EXIT_CODE):
            # Avoid further RPCs against an unresponsive rendering pipeline.
            # The batch supervisor owns CARLA teardown after this process exits.
            active_actor_ids.clear()
            return retval

        if conf.agent_type == c.BASIC or conf.agent_type == c.BEHAVIOR:
            # assemble images into an mp4 container
            # remove jpg files
            print("Saving front camera video", end=" ")

            vid_filename = "/tmp/fuzzerdata/front.mp4"
            if os.path.exists(vid_filename):
                os.remove(vid_filename)

            cmd_cat = "cat /tmp/fuzzerdata/front-*.jpg"
            cmd_ffmpeg = " ".join([
                "ffmpeg",
                "-f image2pipe",
                f"-r {c.FRAME_RATE}",
                "-vcodec mjpeg",
                "-i -",
                "-vcodec libx264",
                vid_filename
            ])

            cmd = f"{cmd_cat} | {cmd_ffmpeg} {c.DEVNULL}"
            os.system(cmd)
            print("(done)")

            cmd = "rm -f /tmp/fuzzerdata/front-*.jpg"
            os.system(cmd)

            print("Saving top camera video", end=" ")

            vid_filename = "/tmp/fuzzerdata/top.mp4"
            if os.path.exists(vid_filename):
                os.remove(vid_filename)

            cmd_cat = "cat /tmp/fuzzerdata/top-*.jpg"
            cmd_ffmpeg = " ".join([
                "ffmpeg",
                "-f image2pipe",
                f"-r {c.FRAME_RATE}",
                "-vcodec mjpeg",
                "-i -",
                "-vcodec libx264",
                vid_filename
            ])

            cmd = f"{cmd_cat} | {cmd_ffmpeg} {c.DEVNULL}"
            os.system(cmd)
            print("(done)")

            cmd = "rm -f /tmp/fuzzerdata/top-*.jpg"
            os.system(cmd)

        elif conf.agent_type == c.AUTOWARE:
            os.system("rosnode kill /recorder_video_front")
            os.system("rosnode kill /recorder_video_rear")
            os.system("rosnode kill /recorder_bag")
            while os.path.exists("/tmp/fuzzerdata/bagfile.lz4.bag.active"):
                print("waiting for rosbag to dump data")
                time.sleep(1)

            # [for LCOV-based coverage experiment]
            # print("checking docker")
            # os.system("docker ps")
            # print("done")

            # # # coverage exp
            # username = os.getenv("USER")
            # docker_prefix = f"docker exec autoware-{username} "

            # print("killing autoware")
            # kill_docker_cmd = docker_prefix + "killall python"
            # os.system(kill_docker_cmd)
            # time.sleep(30)
            # print("killed")

            # print("checking docker (2)")
            # os.system("docker ps")
            # print("done")

            # print("running lcov")
            # lcov_cmd = docker_prefix + "lcov --no-external --capture --directory ./ --output-file /tmp/fuzzerdata/autoware_{}.info".format(
                    # start_time)
            # os.system(lcov_cmd)
            # print("done")

            # print("merging lcov result with base")
            # lcov_cmd2 = docker_prefix + "lcov --add-tracefile /tmp/fuzzerdata/autoware_base.info --add-tracefile /tmp/fuzzerdata/autoware_{}.info --output-file /tmp/fuzzerdata/autoware_{}_total.info".format(start_time, start_time)
            # os.system(lcov_cmd2)
            # print("done")

            # print("generating html report")
            # genhtml_cmd = docker_prefix + "genhtml /tmp/fuzzerdata/autoware_{}_total.info --output-directory /tmp/fuzzerdata/autoware_{}_total/".format(start_time, start_time)
            # os.system(genhtml_cmd)
            # print("done")
            # LCOV-based coverage experiment fin

            try:
                autoware_container.kill()
            except docker.errors.APIError as e:
                print("[-] Couldn't kill Autoware container:", e)
            except UnboundLocalError:
                print("[-] Autoware container was not launched")
            except:
                print("[-] Autoware container was not killed for an unknown reason")
                print("    Trying manually")
                os.system("docker rm -f autoware-{}".format(os.getenv("USER")))
                # still doesn't fix docker hanging..

        elif conf.agent_type == c.VLA and vla_adapter is not None:
            vla_adapter.cleanup()
            cuda_diag.snapshot(
                "after_vla_cleanup",
                campaign=getattr(state, "campaign_cnt", None),
                cycle=getattr(state, "cycle_cnt", None),
                mutation=getattr(state, "mutation", None),
                retval=retval,
            )

        # if retval == 0:
            # # update states
            # state.num_frames = frame_id - frame_0
            # state.elapsed_time = time.time() - start_time

        for s in sensors:
            try:
                if getattr(s, "is_listening", False):
                    s.stop()
            except Exception:
                pass

        for controller in actor_controllers:
            try:
                controller.stop()
            except Exception:
                pass

        for v in actor_vehicles:
            try:
                v.set_autopilot(False)
            except Exception:
                pass

        destroy_commands = []
        for controller in actor_controllers:
            if isinstance(controller, carla.Actor):
                destroy_commands.append(carla.command.DestroyActor(controller))
        for v in actor_vehicles:
            destroy_commands.append(carla.command.DestroyActor(v))
        for w in actor_walkers:
            destroy_commands.append(carla.command.DestroyActor(w))
        for f in actor_frictions:
            destroy_commands.append(carla.command.DestroyActor(f))
        for s in sensors:
            destroy_commands.append(carla.command.DestroyActor(s))
        if player is not None:
            if conf.debug:
                print("[debug] player's last loc:", player.get_location())
            destroy_commands.append(carla.command.DestroyActor(player))

        if destroy_commands:
            try:
                client.apply_batch_sync(destroy_commands, False)
            except Exception as e:
                print("[-] Failed to synchronously destroy actors:", e)
                try:
                    client.apply_batch(destroy_commands)
                except Exception as fallback_error:
                    print("[-] Failed to submit asynchronous destroy batch:", fallback_error)

            # VLAD-Fuzz baseline adapter: these actors have already been
            # submitted for destruction by DriveFuzz's normal cleanup path.
            # Remove them from the emergency cleanup registry so atexit/finally
            # cleanup does not stop or destroy sensors a second time.
            for actor in actor_controllers + actor_vehicles + actor_walkers + actor_frictions + sensors:
                _forget_active_actor(actor)
            if player is not None:
                _forget_active_actor(player)

            if world is not None:
                try:
                    if world.get_settings().synchronous_mode:
                        world.tick()
                    else:
                        world.wait_for_tick(seconds=1.0)
                except Exception as e:
                    print("[-] Failed to tick after actor destruction:", e)

        _cleanup_active_actors()

        # Stop sync mode to prevent the simulator from being blocked
        # waiting for another tick.
        if world is not None:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
            except Exception as e:
                print("[-] Failed to disable world synchronous mode:", e)

        try:
            tm.set_synchronous_mode(False)
        except Exception as e:
            print("[-] Failed to disable Traffic Manager synchronous mode:", e)

        # Don't reload and exit if user requests so
        if retval == 128:
            return retval

        else:
            if getattr(conf, "reload_world_after_simulation", False):
                if conf.debug:
                    print("[debug] reload")
                client.reload_world()

                if conf.debug:
                    print('[debug] done.')

            return retval


def set_args(argparser):
    """
    [arguments]
    * Weather # XXX: really has to be float?
    - float cloudiness: 0 (clear), 100 (cloudy)
    - float precipitation: 0 (none), 100 (heaviest rain)
    - float puddles: 0 (no puddle), 100 (completely covered with puddle)
    - float wind_intensity: 0 (no wind), 100 (strongest wind)
    - float fog_density: 0 (no fog), 100 (densest fog)
    # - float fog_distance: 0 (starts in the beginning), +inf (does not start)
    - float wetness: 0 (completely dry), 100 (completely wet)
    - float sun_azimuth_angle: 0 ~ 360
    - float sun_altitude_angle: -90 (midnight) ~ +90 (noon)

    * Friction triggerbox (optional w/ var nargs)
    - float level: 0.0 (zero friction), 1.0 (full friction)
    - carla.Location extent: x, y, z (size of triggerbox in cm)
    - carla.Transform transform: carla.Location(x, y, z) (where to spawn)

    * Dynamic Actors (also optional w/ variable nargs)
    - int actor_type: 0 (vehicle), 1 (walker), 2 (RESERVED: other)
    - carla.Transform spawn_point: where actor is spawned
    - if actor_type == 0:
      - carla.Vector3D velocity: vehicle's velocity (might apply to walker too)
        - set_target_velocity(velocity)
    - elif actor_type == 1:
      - carla.Vector3D direction: walker's direction
      - float speed: walker's speed (m/s)

    * Sensors (TBD)

    * Static Actors (traffic signals and lights)
      * NOTE: Dynamic spawning of traffic signals and lights is not possible.
        When the simulation starts, stop, yields and traffic light are
        automatically generated using the information in the OpenDRIVE file.
      * Need to update carla to 0.9.9, which manages traffic lights through
        OpenDRIVE (.xodr) files.
    """
    # add weather args
    argparser.add_argument(
        "--cloud",
        default=0,
        type=float)
    argparser.add_argument(
        "--rain",
        default=0,
        type=float)
    argparser.add_argument(
        "--puddle",
        default=0,
        type=float)
    argparser.add_argument(
        "--wind",
        default=0,
        type=float)
    argparser.add_argument(
        "--fog",
        default=0,
        type=float)
    argparser.add_argument(
        "--wetness",
        default=0,
        type=float)
    argparser.add_argument(
        "--angle",
        default=0,
        type=float)
    argparser.add_argument(
        "--altitude",
        default=0,
        type=float)

    # Town map
    argparser.add_argument(
        "--town",
        default="Town01",
        type=str
        )

    # spawn point: x, y, z, pitch, yaw, roll
    argparser.add_argument(
        "--spawn",
        nargs="*",
        type=float
    )

    # destination: x, y, z
    argparser.add_argument(
        "--dest",
        nargs="*",
        type=float
    )

    # BasicAgent's target speed
    argparser.add_argument(
        "--speed",
        default=60,
        type=float)

    argparser.add_argument(
        "--view",
        default=config.ONROOF,
        type=int)

    # Friction triggerbox
    argparser.add_argument(
        "--friction",
        action="append",
        nargs="*",
        type=float
    )
    # Can set multiple triggerboxes.
    # Each --friction arg should be followed by
    # level (0-1.0), size of triggerbox (x, y, z in cm),
    # and location (x, y, z).

    # Dynamic actors
    argparser.add_argument(
        "--actor",
        action="append",
        nargs="*",
        type=float
    )

    argparser.add_argument(
        "--debug",
        action="store_true"
    )

    argparser.add_argument(
        "--no-speed-check",
        action="store_true"
    )
    argparser.add_argument(
        "--no-lane-check",
        action="store_true"
    )
    argparser.add_argument(
        "--no-crash-check",
        action="store_true"
    )
    argparser.add_argument(
        "--no-stuck-check",
        action="store_true"
    )


if __name__ == '__main__':
    conf = config.Config()
    argparser = argparse.ArgumentParser()
    argparser = set_args()

    args = argparser.parse_args()
    if args.debug:
        conf.debug = True
    else:
        conf.debug = False

    if args.cloud < 0 or args.cloud > 100:
        print("[-] Error in arg cloud: {}".format(c.MSG_BAD_WEATHER_ARG))
        sys.exit(-1)
    if args.rain < 0 or args.rain > 100:
        print("[-] Error in arg rain: {}".format(c.MSG_BAD_WEATHER_ARG))
        sys.exit(-1)
    if args.puddle < 0 or args.puddle > 100:
        print("[-] Error in arg puddle: {}".format(c.MSG_BAD_WEATHER_ARG))
        sys.exit(-1)
    if args.wind < 0 or args.wind > 100:
        print("[-] Error in arg wind: {}".format(c.MSG_BAD_WEATHER_ARG))
        sys.exit(-1)
    if args.fog < 0 or args.fog > 100:
        print("[-] Error in arg fog: {}".format(c.MSG_BAD_WEATHER_ARG))
        sys.exit(-1)
    if args.wetness < 0 or args.wetness > 100:
        print("[-] Error in arg wetness: {}".format(c.MSG_BAD_WEATHER_ARG))
        sys.exit(-1)
    if args.angle < 0 or args.angle > 360:
        print("[-] Error in arg angle: {}".format(c.MSG_BAD_SUN_ARG))
        sys.exit(-1)
    if args.altitude < -90 or args.altitude > 90:
        print("[-] Error in arg altitude: {}".format(c.MSG_BAD_SUN_ARG))
        sys.exit(-1)

    weather_dict = {
        "cloud": args.cloud,
        "rain": args.rain,
        "puddle": args.puddle,
        "wind": args.wind,
        "fog": args.fog,
        "wetness": args.wetness,
        "angle": args.angle,
        "altitude": args.altitude
    }

    frictions_list = []
    if args.friction:
        frictions = args.friction
        for friction in frictions:
            if not friction:
                print("[-] Error in arg: {}".format(c.MSG_EMPTY_FRICTION_ARG))
                sys.exit(-1)

            if len(friction) != 7:
                print("[-] Error in arg: {}".format(c.MSG_BAD_FRICTION_ARG))
                sys.exit(-1)

            if friction[0] < 0 or friction[0] > 1:
                print("[-] Error in arg: {}".format(c.MSG_BAD_FRICTION_LEVEL))
                sys.exit(-1)

            friction_level = friction[0]
            friction_box_size = carla.Location(
                x=friction[1], y=friction[2], z=friction[3]
            )
            friction_spawn_point = carla.Transform(
                carla.Location(x=friction[4], y=friction[5], z=friction[6]),
                carla.Rotation()
            )

            frictions_list.append({
                    "level": friction_level,
                    "size": friction_box_size,
                    "spawn_point": friction_spawn_point
                }
            )

    # parse actors
    actors_list = []
    if args.actor:
        actors = args.actor
        for actor in actors:
            if not actor:
                print("[-] Error in arg: {}".format(c.MSG_EMPTY_ACTOR_ARG))
                sys.exit(-1)

            if int(actor[0]) == 0 or int(actor[0]) == 1:
                if len(actor) != 12:
                    print("[-] Error in arg: {}".format(c.MSG_BAD_ACTOR_ATTR))
                    sys.exit(-1)

                # actor_type: vehicle or walker
                actor_type = actor[0]
                nav_type = actor[1]
                actor_spawn_point = carla.Transform(
                    carla.Location(x=actor[2], y=actor[3], z=actor[4]),
                    carla.Rotation(pitch=actor[5], yaw=actor[6], roll=actor[7])
                )
                actor_dest_point = carla.Location(
                    x=actor[8], y=actor[9], z=actor[10]
                )

                actor_speed = actor[11]

                actors_list.append({
                        "type": actor_type,
                        "nav_type": nav_type,
                        "spawn_point": actor_spawn_point,
                        "dest_point": actor_dest_point,
                        "speed": actor_speed
                    }
                )

            elif int(actor[0]) == 2: # placeholder for other actors
                pass

            else:
                print("[-] Error in arg: {}".format(c.MSG_BAD_ACTOR_TYPE))
                sys.exit(-1)

    if args.spawn:
        if len(args.spawn) != 6:
            print("[-] Error in arg spawn: {}".format(c.MSG_BAD_SPAWN_ARG))
            sys.exit(-1)

        sp = carla.Transform(
                carla.Location(args.spawn[0], args.spawn[1], args.spawn[2]),
                carla.Rotation(args.spawn[3], args.spawn[4], args.spawn[5]),
            )
    else: # default for Town01
        sp = carla.Transform(
                carla.Location(334.83, 217.1, 1.32),
                carla.Rotation(0.0, 90, 0.0)
            )

    if args.dest:
        if len(args.spawn) != 6:
            print("[-] Error in arg dest: {}".format(c.MSG_BAD_DEST_ARG))
            sys.exit(-1)

        wp = carla.Location(args.dest[0], args.dest[1], args.dest[2])
    else: # default for Town01
        wp = carla.Location(335.49, 298.81, 1.32)

    if args.speed:
        config.TARGET_SPEED = args.speed

    if args.view:
        if int(args.view) == config.ONROOF:
            config.VIEW = config.ONROOF
        if int(args.view) == config.BIRDSEYE:
            config.VIEW = config.BIRDSEYE

    if args.town:
        town = args.town
    else:
        town = "Town01"

    if args.no_speed_check:
        config.CHECKS["speed"] = False
    if args.no_crash_check:
        config.CHECKS["crash"] = False
    if args.no_lane_check:
        config.CHECKS["lane"] = False
    if args.no_stuck_check:
        config.CHECKS["stuck"] = False

    (client, tm) = connect("localhost", 2000, 8000)

    ret = simulate(client, town, tm, sp, wp,
            weather_dict, frictions_list, actors_list)
    print("simulation returned:", ret)

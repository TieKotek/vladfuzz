#!/usr/bin/env python3

import os
import sys
import time
import random
import argparse
import json
from collections import deque
import subprocess as sp
import math
import datetime
import signal
from copy import deepcopy
import traceback
import shutil

import config
import constants as c
import states
import executor
from vladfuzz_runtime.hourly_checkpoints import HourlyCheckpointRecorder

config.set_carla_api_path()
try:
    import carla
except ModuleNotFoundError as e:
    print("[-] Carla module not found. Make sure you have built Carla.")
    proj_root = config.get_proj_root()
    print("    Try `cd {}/carla && make PythonAPI' if not.".format(proj_root))
    exit(-1)

import fuzz_utils


def handler(signum, frame):
    raise Exception("HANG")


def init(conf, args):
    conf.cur_time = time.time()
    if args.determ_seed:
        conf.determ_seed = args.determ_seed
    else:
        conf.determ_seed = conf.cur_time
    random.seed(conf.determ_seed)
    print("[info] determ seed set to:", conf.determ_seed)

    conf.out_dir = args.out_dir
    try:
        os.mkdir(conf.out_dir)
    except Exception as e:
        estr = f"Output directory {conf.out_dir} already exists. Remove with "\
                "caution; it might contain data from previous runs."
        print(estr)
        sys.exit(-1)

    conf.seed_dir = args.seed_dir
    if not os.path.exists(conf.seed_dir):
        os.mkdir(conf.seed_dir)
    else:
        print(f"Using seed dir {conf.seed_dir}")

    if args.verbose:
        conf.debug = True
    else:
        conf.debug = False

    conf.set_paths()

    with open(conf.meta_file, "w") as f:
        f.write(" ".join(sys.argv) + "\n")
        f.write("start: " + str(int(conf.cur_time)) + "\n")

    try:
        os.mkdir(conf.queue_dir)
        os.mkdir(conf.error_dir)
        os.mkdir(conf.cov_dir)
        os.mkdir(conf.rosbag_dir)
        os.mkdir(conf.cam_dir)
        os.mkdir(conf.score_dir)
    except Exception as e:
        print(e)
        sys.exit(-1)

    conf.sim_host = args.sim_host
    conf.sim_port = args.sim_port

    conf.max_cycles = args.max_cycles
    conf.max_mutations = args.max_mutations

    conf.timeout = args.timeout

    conf.function = args.function

    if args.target.lower() == "basic":
        conf.agent_type = c.BASIC
    elif args.target.lower() == "behavior":
        conf.agent_type = c.BEHAVIOR
    elif args.target.lower() == "autoware":
        conf.agent_type = c.AUTOWARE
    elif args.target.lower() == "vla":
        conf.agent_type = c.VLA
    else:
        print("[-] Unknown target: {}".format(args.target))
        sys.exit(-1)

    conf.project_root = args.project_root
    conf.vla_model = args.vla_model
    conf.vla_gpu_id = args.vla_gpu_id
    conf.vla_instruction = args.vla_instruction
    conf.vla_instruction_map = args.vla_instruction_map
    conf.reuse_vla_backend = not args.no_reuse_vla_backend
    conf.continue_with_random_seed = args.continue_with_random_seed
    conf.repeat_seeds_until_budget = args.repeat_seeds_until_budget
    conf.time_budget_seconds = args.time_budget_seconds
    if args.time_budget_minutes is not None:
        conf.time_budget_seconds = args.time_budget_minutes * 60.0
    conf.reload_world_after_simulation = args.reload_world_after_simulation
    conf.max_simulation_seconds = args.max_simulation_seconds
    # VLAD-Fuzz VLA integration: keep upstream time fallback while allowing
    # the shared 500-frame evaluation window to be selected by our launcher.
    conf.max_simulation_frames = args.max_simulation_frames
    conf.simulation_alarm_seconds = args.simulation_alarm_seconds
    if conf.simulation_alarm_seconds is None and conf.agent_type != c.VLA:
        conf.simulation_alarm_seconds = 10 * 60
    conf.goal_distance_threshold = args.goal_distance_threshold
    conf.lock_traffic_lights_green = args.lock_traffic_lights_green
    conf.allow_walkers = not args.no_walkers
    conf.cuda_diag_dir = args.cuda_diag_dir

    conf.town = args.town
    conf.follow_spectator = args.follow_spectator
    conf.view = args.view

    if conf.agent_type == c.VLA and not args.enable_speed_check:
        conf.check_dict["speed"] = False
    if args.enable_red_check:
        conf.check_dict["red"] = True
    if args.no_speed_check:
        conf.check_dict["speed"] = False
    if args.no_crash_check:
        conf.check_dict["crash"] = False
    if args.no_lane_check:
        conf.check_dict["lane"] = False
    if args.no_stuck_check:
        conf.check_dict["stuck"] = False
    if args.no_red_check:
        conf.check_dict["red"] = False
    if args.no_out_of_bounds_check:
        conf.check_dict["out_of_bounds"] = False
    if args.no_other_check:
        conf.check_dict["other"] = False

    if args.strategy == "all":
        conf.strategy = c.ALL
    elif args.strategy == "congestion":
        conf.strategy = c.CONGESTION
    elif args.strategy == "entropy":
        conf.strategy = c.ENTROPY
    elif args.strategy == "instability":
        conf.strategy = c.INSTABILITY
    elif args.strategy == "trajectory":
        conf.strategy = c.TRAJECTORY
    else:
        print("[-] Please specify a strategy")
        exit(-1)


def mutate_weather(test_scenario):
    test_scenario.weather["cloud"] = random.randint(0, 100)
    test_scenario.weather["rain"] = random.randint(0, 100)
    test_scenario.weather["puddle"] = random.randint(0, 100)
    test_scenario.weather["wind"] = random.randint(0, 100)
    test_scenario.weather["fog"] = random.randint(0, 100)
    test_scenario.weather["wetness"] = random.randint(0, 100)
    test_scenario.weather["angle"] = random.randint(0, 360)
    test_scenario.weather["altitude"] = random.randint(-90, 90)


def mutate_weather_fixed(test_scenario):
    test_scenario.weather["cloud"] = 0
    test_scenario.weather["rain"] = 0
    test_scenario.weather["puddle"] = 0
    test_scenario.weather["wind"] = 0
    test_scenario.weather["fog"] = 0
    test_scenario.weather["wetness"] = 0
    test_scenario.weather["angle"] = 0
    test_scenario.weather["altitude"] = 60


def get_spawn_point_index_range():
    if not executor.list_spawn_points:
        raise RuntimeError("DriveFuzz spawn points are not initialized for the current town.")
    return len(executor.list_spawn_points) - 1


def _location_to_actor_point(location, yaw=0.0):
    return (
        (location.x, location.y, location.z),
        (0.0, 0.0, yaw),
    )


def sample_walker_navigation_pair(world, ego_location, attempts=2000, relaxed=False):
    """Sample a pedestrian-controller-safe start/destination pair.

    Vehicle spawn points are not guaranteed to lie on CARLA's pedestrian
    navigation mesh. Passing those points to WalkerAIController.go_to_location()
    can crash CARLA 0.9.15 in native walker routing code, so autopilot walkers
    must use simulator-provided navigation locations instead. The relaxed mode
    keeps the walker start near the ego vehicle but accepts farther destinations,
    preserving pedestrian mutations when dense local navmesh pairs are scarce.
    """
    min_dist = c.MIN_DIST_FROM_PLAYER
    max_dist = c.MAX_DIST_FROM_PLAYER
    min_route_dist = max(3.0, c.MIN_DIST_FROM_PLAYER)
    max_dest_ego_dist = max_dist * (5.0 if relaxed else 2.0)
    dest_attempts = 100 if relaxed else 50

    for _ in range(attempts):
        start = world.get_random_location_from_navigation()
        if start is None:
            continue
        ego_dist = start.distance(ego_location)
        if ego_dist < min_dist or ego_dist > max_dist:
            continue

        for _ in range(dest_attempts):
            dest = world.get_random_location_from_navigation()
            if dest is None:
                continue
            if start.distance(dest) < min_route_dist:
                continue
            if dest.distance(ego_location) > max_dest_ego_dist:
                continue
            return start, dest

    return None, None


def sample_linear_walker_navigation_point(world, ego_location, attempts=2000):
    """Sample a navmesh-safe linear walker start and direction target."""
    min_dist = c.MIN_DIST_FROM_PLAYER
    max_dist = c.MAX_DIST_FROM_PLAYER
    min_route_dist = max(3.0, c.MIN_DIST_FROM_PLAYER)

    for _ in range(attempts):
        start = world.get_random_location_from_navigation()
        if start is None:
            continue
        ego_dist = start.distance(ego_location)
        if ego_dist < min_dist or ego_dist > max_dist:
            continue

        dest = world.get_random_location_from_navigation()
        if dest is None or start.distance(dest) < min_route_dist:
            continue
        return start, dest

    return None, None


def choose_actor_profile(conf):
    actor_list = c.ACTOR_LIST if getattr(conf, "allow_walkers", True) else [c.VEHICLE]

    if conf.function == "general":
        if conf.strategy == c.ALL:
            actor_type = random.choice(actor_list)
            nav_type = random.randint(0, len(c.NAVTYPE_LIST) - 1)

        elif conf.strategy == c.CONGESTION:
            actor_type = random.choice(actor_list)
            nav_type = c.AUTOPILOT

        elif conf.strategy == c.ENTROPY:
            actor_type = random.choice(actor_list)
            nav_type = c.LINEAR

        elif conf.strategy == c.INSTABILITY:
            actor_type = c.NULL
            nav_type = c.NULL

        elif conf.strategy == c.TRAJECTORY:
            actor_type = c.VEHICLE
            nav_type = c.MANEUVER

    elif conf.function == "collision":
        actor_type = c.VEHICLE
        nav_type = c.LINEAR

    elif conf.function == "traction":
        actor_type = c.NULL
        nav_type = c.NULL

    else:
        actor_type = c.NULL
        nav_type = c.NULL

    return actor_type, nav_type


def set_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-o", "--out-dir", default="output", type=str,
            help="Directory to save fuzzing logs")
    argparser.add_argument("-s", "--seed-dir", default="seed", type=str,
            help="Seed directory")
    argparser.add_argument("-c", "--max-cycles", default=10, type=int,
            help="Maximum number of loops")
    argparser.add_argument("-m", "--max-mutations", default=8, type=int,
            help="Size of the mutated population per cycle")
    argparser.add_argument("-d", "--determ-seed", type=float,
            help="Set seed num for deterministic mutation (e.g., for replaying)")
    argparser.add_argument("-v", "--verbose", action="store_true",
            default=False, help="enable debug mode")
    argparser.add_argument("-u", "--sim-host", default="localhost", type=str,
            help="Hostname of Carla simulation server")
    argparser.add_argument("-p", "--sim-port", default=2000, type=int,
            help="RPC port of Carla simulation server")
    argparser.add_argument("-t", "--target", default="Autoware", type=str,
            help="Target autonomous driving system (basic/behavior/autoware/vla)")
    argparser.add_argument("--project-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
            type=str, help="VLAD-Fuzz project root, required for --target vla")
    argparser.add_argument("--vla-model", default="simlingo", type=str,
            help="VLAD-Fuzz model backend for --target vla")
    argparser.add_argument("--vla-gpu-id", default=0, type=int,
            help="GPU id passed to the VLAD-Fuzz VLA backend")
    argparser.add_argument("--vla-instruction", default="Follow the planned route to the destination.", type=str,
            help="Fallback natural-language instruction for --target vla")
    argparser.add_argument("--vla-instruction-map", type=str,
            help="JSONL mapping from DriveFuzz seed file to natural-language instruction")
    argparser.add_argument("--no-reuse-vla-backend", action="store_true",
            help="Reload the VLA model for every simulation instead of reusing one backend instance per process.")
    argparser.add_argument("-f", "--function", default="general", type=str,
            choices=["general", "collision", "traction", "eval-os", "eval-us",
                "figure", "sens1", "sens2", "lat", "rear"],
            help="Functionality to test (general / collision / traction)")
    argparser.add_argument("--strategy", default="all", type=str,
            help="Input mutation strategy (all / congestion / entropy / instability / trajectory)")
    argparser.add_argument("--town", default=None, type=int,
            help="Test on a specific town (e.g., '--town 3' forces Town03)")
    argparser.add_argument("--timeout", default="20", type=int,
            help="Seconds to timeout if vehicle is not moving")
    argparser.add_argument("--continue-with-random-seed", action="store_true",
            help="Continue with randomly generated seeds after the input seed queue is exhausted")
    argparser.add_argument("--repeat-seeds-until-budget", action="store_true",
            help="Restart the input seed queue until the wall-time budget is exhausted")
    argparser.add_argument("--time-budget-seconds", type=float,
            help="Stop starting new simulations after this many wall-clock seconds")
    argparser.add_argument("--time-budget-minutes", type=float,
            help="Stop starting new simulations after this many wall-clock minutes")
    argparser.add_argument("--reload-world-after-simulation", action="store_true",
            help="Reload the CARLA world after each simulation. Disabled by default because repeated reloads can destabilize long VLA campaigns.")
    argparser.add_argument("--max-simulation-seconds", default=600.0, type=float,
            help="Fallback maximum simulated seconds for one scenario before reporting a timeout.")
    # VLAD-Fuzz VLA integration: optional frame-based cap for oracle alignment.
    argparser.add_argument("--max-simulation-frames", type=int, default=None,
            help="Optional maximum simulated frames before reporting a timeout.")
    argparser.add_argument("--simulation-alarm-seconds", type=int, default=None,
            help="Optional wall-clock alarm for one simulation. Disabled by default for --target vla to avoid interrupting torch/CUDA inference.")
    argparser.add_argument("--goal-distance-threshold", default=5.0, type=float,
            help="Distance in meters at which the ego vehicle is considered to have reached the destination.")
    argparser.add_argument("--lock-traffic-lights-green", action="store_true",
            help="Set all traffic lights to green and freeze them for this run.")
    argparser.add_argument("--no-walkers", action="store_true",
            help="Disable pedestrian/walker mutations. Useful for VLA experiments whose scenario model does not support pedestrians.")
    argparser.add_argument("--cuda-diag-dir", type=str,
            help="Directory for CUDA memory diagnostics JSON/JSONL files.")
    argparser.add_argument("--enable-speed-check", action="store_true",
            help="Enable speed-limit violations for VLA targets. Disabled by default for --target vla")
    argparser.add_argument("--enable-red-check", action="store_true",
            help="Enable red-light violations.")
    argparser.add_argument("--follow-spectator", action="store_true",
            help="Move the CARLA spectator every frame to follow the ego vehicle")
    argparser.add_argument("--view", default=c.BIRDSEYE, type=int, choices=[c.ONROOF, c.BIRDSEYE],
            help="Spectator view when --follow-spectator is enabled: 0=chase/on-roof, 1=bird's-eye")
    argparser.add_argument("--no-speed-check", action="store_true")
    argparser.add_argument("--no-lane-check", action="store_true")
    argparser.add_argument("--no-crash-check", action="store_true")
    argparser.add_argument("--no-stuck-check", action="store_true")
    argparser.add_argument("--no-red-check", action="store_true")
    argparser.add_argument("--no-out-of-bounds-check", action="store_true",
            help="Disable static-scenario out-of-bounds checks for VLAD-Fuzz exported seeds.")
    argparser.add_argument("--no-other-check", action="store_true")

    return argparser


def main():
    conf = config.Config()
    argparser = set_args()
    args = argparser.parse_args()

    init(conf, args)
    conf.fuzzing_start_time = time.time()
    checkpoint_recorder = HourlyCheckpointRecorder(
        output_dir=conf.out_dir,
        start_time=conf.fuzzing_start_time,
        method="drivefuzz",
        model=conf.vla_model,
        manifest_id=os.path.basename(os.path.abspath(conf.seed_dir)),
    )
    executed_scenarios = 0
    detected_failures = 0

    cov_dict = {}

    initial_seed_queue = conf.enqueue_seed_scenarios()
    queue = deque(initial_seed_queue)

    scene_id = 0
    campaign_cnt = 0

    signal.signal(signal.SIGALRM, handler)

    while True:
        if (conf.time_budget_seconds is not None and
                time.time() - conf.fuzzing_start_time >= conf.time_budget_seconds):
            print("[+] Wall-time budget exhausted. Stop.")
            break

        cycle_cnt = 0

        # STEP 0: Restart Carla simulator at the beginning of each cycle
        # (Carla hangs after a while due to a memory leak)
        # UPDATE: can't do this due to a bug in Carla. TimeoutException will
        # be raised when we stop the container.
        # print("[*] Stopping existing Carla simulator")
        # os.system(f"docker rm -f carla-{os.getenv('USER')}")
        # for i in range(10):
            # print(".", end="", flush=True)
            # time.sleep(1)
        # print()

        # print("[*] Starting Carla simulator")
        # os.system("../run_carla.sh")
        # for i in range(30):
            # print(".", end="", flush=True)
            # time.sleep(1)
        # print()

        # STEP 1: SEED
        # Pop a seed from the seed queue. If seed queue is empty, ramdonly
        # generate a seed, enqueue it, and come back here.

        # print("before seed gen", time.time())
        try:
            scenario = queue.popleft()
            conf.cur_scenario = scenario
            scene_id += 1
            campaign_cnt += 1

            print("\n\033[1m\033[92m" + "=" * 10 + " NEW CAMPAIGN " + "=" * 10 + "\033[0m\n")

        except IndexError:
            if (conf.repeat_seeds_until_budget and initial_seed_queue and
                    conf.time_budget_seconds is not None):
                print("[+] Seed queue is empty. Restart input seeds until wall-time budget is exhausted.")
                queue = deque(initial_seed_queue)
                continue

            if not conf.continue_with_random_seed:
                print("[+] Seed queue is empty. Stop.")
                break

            print("[-] Seed queue is empty. Continue with random seed.")

            if conf.town is not None:
                town_map = "Town0{}".format(conf.town)
            else:
                town_map = "Town0{}".format(random.randint(1, 2))
            (client, tm) = executor.connect(conf)
            client.set_timeout(20)
            client.load_world(town_map)
            world = client.get_world()
            town = world.get_map()
            spawn_points = town.get_spawn_points()
            sp = random.choice(spawn_points)
            sp_x = sp.location.x
            sp_y = sp.location.y
            sp_z = sp.location.z
            pitch = sp.rotation.pitch
            yaw = sp.rotation.yaw
            roll = sp.rotation.roll

            wp = random.choice(spawn_points)
            wp_x = wp.location.x
            wp_y = wp.location.y
            wp_z = wp.location.z
            wp_yaw = wp.rotation.yaw

            # restrict destinition to be within 200 meters
            while math.sqrt((sp_x - wp_x) ** 2 + (sp_y - wp_y) ** 2) > 100:
                wp = random.choice(spawn_points)
                wp_x = wp.location.x
                wp_y = wp.location.y
                wp_z = wp.location.z
                wp_yaw = wp.rotation.yaw

            seed_dict = {
                    "map": town_map,
                    "sp_x": sp_x,
                    "sp_y": sp_y,
                    "sp_z": sp_z,
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                    "wp_x": wp_x,
                    "wp_y": wp_y,
                    "wp_z": wp_z,
                    "wp_yaw": wp_yaw
                    }
            scene_name = "scene-created{}.json".format(scene_id)
            with open(os.path.join(conf.seed_dir, scene_name), "w") as fp:
                json.dump(seed_dict, fp)
            queue.append(scene_name)

            continue

        if conf.debug:
            print("[*] USING SEED FILE:", scenario)

        # STEP 2: TEST CASE INITIALIZATION
        # Create and initialize a TestScenario instance based on the metadata
        # read from the popped seed file.

        test_scenario = fuzz_utils.TestScenario(conf, base=scenario)
        successor_scenario = None

        # STEP 3: SCENE MUTATION
        # While test scenario has quota left, mutate scene by randomly
        # generating walker/vehicle/puddle.

        while cycle_cnt < conf.max_cycles:
            cycle_cnt += 1

            if successor_scenario is not None:
                test_scenario = successor_scenario

            mutated_scenario_list = list() # mutated TestScenario objects
            score_list = list() # driving scores of each mutated scenario

            round_cnt = 0
            test_scenario_m = None
            town = test_scenario.town
            (min_x, max_x, min_y, max_y) = fuzz_utils.get_valid_xy_range(town)

            while round_cnt < conf.max_mutations: # mutation rounds
                if (conf.time_budget_seconds is not None and
                        time.time() - conf.fuzzing_start_time >= conf.time_budget_seconds):
                    print("[+] Wall-time budget exhausted. Stop starting new mutations.")
                    break

                round_cnt += 1

                print("\n\033[1m\033[92mCampaign #{} Cycle #{}/{} Mutation #{}/{}".format(
                    campaign_cnt, cycle_cnt, conf.max_cycles, round_cnt,
                    conf.max_mutations), "\033[0m", datetime.datetime.now())

                test_scenario_m = deepcopy(test_scenario)

                # STEP 3-1: ACTOR PROFILE GENERATION

                actor_type, nav_type = choose_actor_profile(conf)

                ret = True
                while ret:
                    if actor_type == c.NULL:
                        break

                    elif actor_type == c.VEHICLE:
                        # spawn vehicle at a random location
                        # run test while mutating weather
                        if nav_type == c.LINEAR:
                            x = random.randint(min_x, max_x)
                            y = random.randint(min_y, max_y)
                            z = 1.5
                            pitch = 0
                            yaw = random.randint(0, 360)
                            roll = 0
                            speed = random.uniform(0, c.VEHICLE_MAX_SPEED)

                            loc = (x, y, z) # carla.Location(x, y, z)
                            rot = (roll, pitch, yaw) # carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

                            ret = test_scenario_m.add_actor(actor_type,
                                    nav_type, loc, rot, speed, None, None)

                        elif nav_type == c.IMMOBILE:
                            x = random.randint(min_x, max_x)
                            y = random.randint(min_y, max_y)
                            z = 1.5
                            pitch = 0
                            yaw = random.randint(0, 360)
                            roll = 0
                            speed = 0

                            loc = (x, y, z) # carla.Location(x, y, z)
                            rot = (roll, pitch, yaw) # carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

                            ret = test_scenario_m.add_actor(actor_type,
                                    nav_type, loc, rot, speed, None, None)

                        elif nav_type == c.AUTOPILOT:
                            max_spawn_idx = get_spawn_point_index_range()
                            sp_idx = random.randint(0, max_spawn_idx)
                            dp_idx = random.randint(0, max_spawn_idx)
                            ret = test_scenario_m.add_actor(actor_type,
                                    nav_type, None, None, None, sp_idx,
                                    dp_idx)

                        elif nav_type == c.MANEUVER:
                            if len(test_scenario_m.actors) == 0:
                                # add an actor
                                x = test_scenario_m.sp.location.x + random.randint(-50, 50) / 10.0

                                y = test_scenario_m.sp.location.y + random.randint(-50, 50) / 10.0
                                z = 1.5

                                roll = test_scenario_m.sp.rotation.roll
                                pitch = test_scenario_m.sp.rotation.pitch
                                yaw = test_scenario_m.sp.rotation.yaw

                                speed = 0

                                loc = (x, y, z) # carla.Location(x, y, z)
                                rot = (roll, pitch, yaw) # carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

                                ret = test_scenario_m.add_actor(actor_type, nav_type, loc, rot,
                                        speed, None, None)

                            else:
                                # mutate maneuvers if we have an actor
                                i = random.randint(0, 4)
                                direction = random.randint(-1, 1)

                                if direction == 0:
                                    speed = random.randint(0, 10) # m/s
                                    test_scenario_m.actors[0]["maneuvers"][i] = [direction, speed, 0]
                                else:
                                    degree = random.randint(30, 60) # deg
                                    test_scenario_m.actors[0]["maneuvers"][i] = [direction, degree, 0]

                                # reset executed frame id
                                for i in range(5):
                                    test_scenario_m.actors[0]["maneuvers"][i][2] = 0

                                ret = 0

                            print("Maneuvers:", test_scenario_m.actors[0]["maneuvers"])

                    elif actor_type == c.WALKER:
                        if nav_type == c.LINEAR:
                            x = random.randint(min_x, max_x)
                            y = random.randint(min_y, max_y)
                            z = 1.5
                            pitch = 0
                            yaw = random.randint(0, 360)
                            roll = 0
                            speed = random.uniform(0, c.WALKER_MAX_SPEED)

                            loc = (x, y, z) # carla.Location(x, y, z)
                            rot = (roll, pitch, yaw) # carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

                            ret = test_scenario_m.add_actor(actor_type, nav_type, loc, rot,
                                    speed, None, None)

                        elif nav_type == c.IMMOBILE:
                            x = random.randint(min_x, max_x)
                            y = random.randint(min_y, max_y)
                            z = 1.5
                            pitch = 0
                            yaw = random.randint(0, 360)
                            roll = 0
                            speed = 0

                            loc = (x, y, z) # carla.Location(x, y, z)
                            rot = (roll, pitch ,yaw) # carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)

                            ret = test_scenario_m.add_actor(actor_type, nav_type, loc, rot,
                                    speed, None, None)

                        elif nav_type == c.AUTOPILOT:
                            world = executor.client.get_world()
                            ego_location = test_scenario_m.get_seed_sp_transform(
                                    test_scenario_m.seed_data).location
                            start, dest = sample_walker_navigation_pair(world, ego_location)
                            fallback_mode = None
                            if start is None or dest is None:
                                start, dest = sample_walker_navigation_pair(
                                        world, ego_location, relaxed=True)
                                if start is not None and dest is not None:
                                    fallback_mode = "relaxed_autopilot"

                            if start is not None and dest is not None:
                                yaw = random.randint(0, 360)
                                speed = random.randint(0, 10)
                                loc = _location_to_actor_point(start, yaw=yaw)[0]
                                rot = _location_to_actor_point(start, yaw=yaw)[1]
                                dest_point = _location_to_actor_point(dest)

                                ret = test_scenario_m.add_actor(actor_type, nav_type, loc,
                                        rot, speed, None, dest_point)
                                # VLAD-Fuzz baseline adapter: keep DriveFuzz's
                                # original resampling behavior, but avoid
                                # printing walker-nav fallback details unless
                                # debug logging is explicitly enabled.
                                if fallback_mode and conf.debug:
                                    print("[walker] fallback to relaxed autopilot walker")
                            else:
                                start, dest = sample_linear_walker_navigation_point(world, ego_location)
                                if start is not None and dest is not None:
                                    yaw = math.degrees(math.atan2(dest.y - start.y, dest.x - start.x))
                                    speed = random.uniform(0, c.WALKER_MAX_SPEED)
                                    loc = _location_to_actor_point(start, yaw=yaw)[0]
                                    rot = _location_to_actor_point(start, yaw=yaw)[1]
                                    ret = test_scenario_m.add_actor(c.WALKER, c.LINEAR,
                                            loc, rot, speed, None, None)
                                    if ret == 0:
                                        nav_type = c.LINEAR
                                    if conf.debug:
                                        print("[walker] fallback to linear walker")
                                else:
                                    if conf.debug:
                                        print("[-] No valid walker navigation point near ego; using autopilot vehicle instead")
                                    max_spawn_idx = get_spawn_point_index_range()
                                    sp_idx = random.randint(0, max_spawn_idx)
                                    dp_idx = random.randint(0, max_spawn_idx)
                                    ret = test_scenario_m.add_actor(c.VEHICLE, c.AUTOPILOT,
                                            None, None, None, sp_idx, dp_idx)
                                    if ret == 0:
                                        actor_type = c.VEHICLE

                if conf.debug:
                    if actor_type != c.NULL:
                        print("[debug] successfully added {} {}".format(
                            c.NAVTYPE_NAMES[nav_type], c.ACTOR_NAMES[actor_type]))


                # STEP 3-2: PUDDLE PROFILE GENERATION
                if conf.function == "traction":
                    prob_puddle = 0 # always add a puddle
                elif conf.function == "collision":
                    prob_puddle = 100 # never add a puddle
                elif conf.strategy == c.INSTABILITY:
                    prob_puddle = 0
                else:
                    prob_puddle = random.randint(0, 100)


                if (prob_puddle < c.PROB_PUDDLE):
                    ret = True
                    while ret:
                        # slippery puddles only
                        level = random.randint(0, 200) / 100
                        x = random.randint(min_x, max_x)
                        y = random.randint(min_y, max_y)
                        z = 0

                        xy_size = c.PUDDLE_MAX_SIZE
                        z_size = 1000 # doesn't affect anything

                        loc = (x, y, z) # carla.Location(x, y, z)
                        size = (xy_size, xy_size, z_size) # carla.Location(xy_size, xy_size, z_size)
                        ret = test_scenario_m.add_puddle(level, loc, size)

                    if conf.debug:
                        print("successfully added a puddle")

                # print("after seed gen and mutation", time.time())
                # STEP 3-3: EXECUTE SIMULATION
                ret = None

                state = states.State()
                state.campaign_cnt = campaign_cnt
                state.cycle_cnt = cycle_cnt
                state.mutation = round_cnt

                mutate_weather(test_scenario_m) # mutate_weather_fixed(test)

                if conf.simulation_alarm_seconds is not None:
                    signal.alarm(conf.simulation_alarm_seconds)
                try:
                    ret = test_scenario_m.run_test(state)

                except Exception as e:
                    if e.args and e.args[0] == "HANG":
                        print("[-] simulation hanging. abort.")
                        ret = -1
                    else:
                        print("[-] run_test error:")
                        traceback.print_exc()

                finally:
                    if conf.simulation_alarm_seconds is not None:
                        signal.alarm(0)

                if ret in (executor.CUDA_OOM_EXIT_CODE, executor.SENSOR_FAILURE_EXIT_CODE, executor.CARLA_INFRASTRUCTURE_EXIT_CODE):
                    failure_name = {
                        executor.CUDA_OOM_EXIT_CODE: "CUDA OOM",
                        executor.SENSOR_FAILURE_EXIT_CODE: "Sensor",
                        executor.CARLA_INFRASTRUCTURE_EXIT_CODE: "CARLA RPC",
                    }[ret]
                    print("[-] {} infrastructure failure; exiting for batch-level restart".format(failure_name))
                    exit(ret)

                if getattr(test_scenario_m, "log_filename", None):
                    executed_scenarios += 1
                    if ret == 1:
                        detected_failures += 1
                    checkpoint_recorder.update(
                        now=time.time(),
                        executions=executed_scenarios,
                        failures=detected_failures,
                        extra={
                            "campaign": campaign_cnt,
                            "cycle": cycle_cnt,
                            "mutation": round_cnt,
                        },
                    )

                # t3 = time.time()

                # STEP 3-4: DECIDE WHAT TO DO BASED ON THE RESULT OF EXECUTION
                # TODO: map return codes with failures, e.g., spawn
                if ret is None:
                    # failure
                    pass

                elif ret == -1:
                    print("Spawn / simulation failure - don't add round cnt")
                    round_cnt -= 1

                elif ret == 1:
                    print("fuzzer - found an error")
                    # found an error - move on to next one in queue
                    # test.quota = 0
                    break

                elif ret == 128:
                    print("Exit by user request")
                    exit(0)

                else:
                    if ret == -1:
                        print("[-] Fatal error occurred during test")
                        exit(-1)

                if hasattr(test_scenario_m, "log_filename"):
                    mutated_scenario_list.append(test_scenario_m)
                    score_list.append(test_scenario_m.driving_quality_score)
                ### mutation loop ends

            if test_scenario_m is not None and test_scenario_m.found_error:
                # error detected. start a new cycle with a new seed
                successor_scenario = test_scenario_m
                break

            if len(score_list) == 0:
                print("[-] no successful mutations in this cycle")
                break

            idx = score_list.index(min(score_list))
            successor_scenario = mutated_scenario_list[idx]

            # VLAD-Fuzz baseline adapter: these queue diagnostics are present
            # in upstream DriveFuzz, but they are very noisy in long VLA
            # campaigns. Gate them behind debug without changing successor
            # selection or queue behavior.
            if conf.debug:
                print(score_list)
                print(mutated_scenario_list)
                print("successor:", vars(successor_scenario))

            shutil.copyfile(
                os.path.join(conf.queue_dir, successor_scenario.log_filename),
                os.path.join(conf.cov_dir, successor_scenario.log_filename)
            )

            print("="*10 + " END OF ALL CYCLES " + "="*10)

    end_time = time.time()
    metadata = {
        "method": "drivefuzz",
        "model": conf.vla_model,
        "seed_dir": conf.seed_dir,
        "out_dir": conf.out_dir,
        "start_time": datetime.datetime.fromtimestamp(conf.fuzzing_start_time).isoformat(),
        "end_time": datetime.datetime.fromtimestamp(end_time).isoformat(),
        "duration_seconds": round(end_time - conf.fuzzing_start_time, 2),
        "time_budget_seconds": conf.time_budget_seconds,
        "executed_scenarios": executed_scenarios,
        "failures": detected_failures,
        "failure_rate": round(detected_failures / executed_scenarios, 6) if executed_scenarios else 0.0,
        "hourly_checkpoints": checkpoint_recorder.checkpoints,
    }
    with open(os.path.join(conf.out_dir, "drivefuzz_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()

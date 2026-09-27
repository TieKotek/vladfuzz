import os, sys, glob, json
import constants as c


def get_proj_root():
    config_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(config_path)
    proj_root = os.path.dirname(src_dir)

    return proj_root


def set_carla_api_path():
    try:
        import carla  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    proj_root = get_proj_root()

    dist_path = os.path.join(proj_root, "carla/PythonAPI/carla/dist")
    glob_path = os.path.join(dist_path, "carla-*%d.%d-%s.egg" % (
        sys.version_info.major,
        sys.version_info.minor,
        "win-amd64" if os.name == "nt" else "linux-x86_64"
    ))

    try:
        api_path = glob.glob(glob_path)[0]
    except IndexError:
        project_root = os.environ.get(
            "PROJECT_ROOT",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        )
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        try:
            from vladfuzz_runtime.env_patches import bootstrap_carla_python_paths, bootstrap_project_python_paths
            bootstrap_project_python_paths()
            bootstrap_carla_python_paths()
            import carla  # noqa: F401
            return
        except ModuleNotFoundError:
            print("Couldn't set Carla API path.")
            exit(-1)

    if api_path not in sys.path:
        sys.path.append(api_path)
        print(f"API: {api_path}")


class Config:
    """
    A class defining fuzzing configuration and helper methods.
    An instance of this class should be created by the main module (fuzzer.py)
    and then be shared across other modules as a context handler.
    """

    def __init__(self):
        self.debug = False

        # simulator config
        self.sim_host = "localhost"
        self.sim_port = 2000
        self.sim_tm_port = 8000

        # Fuzzer config
        self.max_cycles = 0
        self.max_mutation = 0
        self.num_dry_runs = 1
        self.num_param_mutations = 1
        # self.initial_quota = 10

        # Fuzzing metadata
        self.cur_time = None
        self.determ_seed = None
        self.out_dir = None
        self.seed_dir = None
        self.continue_with_random_seed = False
        self.repeat_seeds_until_budget = False
        self.time_budget_seconds = None
        self.fuzzing_start_time = None
        self.reload_world_after_simulation = False
        self.max_simulation_seconds = 600.0
        # VLAD-Fuzz VLA integration: optional frame cap for oracle-aligned comparisons.
        self.max_simulation_frames = None
        self.simulation_alarm_seconds = None
        self.goal_distance_threshold = 5.0
        self.lock_traffic_lights_green = False
        self.allow_walkers = True
        self.cuda_diag_dir = None

        # Target config
        self.agent_type = c.AUTOWARE # c.AUTOWARE
        self.project_root = None
        self.vla_model = None
        self.vla_gpu_id = 0
        self.vla_instruction = "Follow the planned route to the destination."
        self.vla_instruction_map = None
        self.reuse_vla_backend = True

        # Enable/disable Various Checks
        self.check_dict = {
            "speed": True,
            "lane": False,
            "crash": True,
            "stuck": True,
            "red": False,
            "out_of_bounds": True,
            "other": True,
        }

        # Functional testing
        self.function = "general"

        # Sim-debug settings
        self.view = c.BIRDSEYE
        self.follow_spectator = False

    def set_paths(self):
        self.queue_dir = os.path.join(self.out_dir, "queue")
        self.error_dir = os.path.join(self.out_dir, "errors")
        self.cov_dir = os.path.join(self.out_dir, "cov")
        self.meta_file = os.path.join(self.out_dir, "meta")
        self.cam_dir = os.path.join(self.out_dir, "camera")
        self.rosbag_dir = os.path.join(self.out_dir, "rosbags")
        self.score_dir = os.path.join(self.out_dir, "scores")

    def enqueue_seed_scenarios(self):
        try:
            seed_scenarios = os.listdir(self.seed_dir)
        except:
            print("[-] Error - cannot find seed directory ({})".format(self.seed_dir))
            sys.exit(-1)

        mapping_path = os.path.join(self.seed_dir, "mapping.jsonl")
        if os.path.exists(mapping_path):
            queue = []
            with open(mapping_path, "r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    seed_file = json.loads(line).get("seed_file")
                    if seed_file and os.path.exists(os.path.join(self.seed_dir, seed_file)):
                        queue.append(seed_file)
            return queue

        queue = [seed for seed in seed_scenarios if not seed.startswith(".")
                and seed.endswith(".json")]

        return queue

import argparse
import os
import json
import random
import copy
import time
import math
from datetime import datetime
from deap import base, creator, tools, algorithms
from vladfuzz_workflows.local_fuzzer import LocalFuzzer
from scenario.carla_scenario import CarlaScenario
from vladfuzz_runtime.experiment_manifest import INSTRUCTION_SOURCES, apply_manifest_selection
from vladfuzz_runtime.failure_analysis import collect_failure_analysis
from vladfuzz_runtime.model_registry import bootstrap_model_environment
from vladfuzz_runtime.run_metadata import create_run_dir
from vladfuzz_runtime.mutation_operators import parse_operator_list
from vladfuzz_runtime.oracle import parse_oracle_checks
from vladfuzz_runtime.hourly_checkpoints import HourlyCheckpointRecorder
from vladfuzz_runtime.npc_count_range import sample_npc_count, validate_npc_count_range
from vladfuzz_runtime.infrastructure import (
    CARLA_INFRASTRUCTURE_EXIT_CODE,
    CarlaInfrastructureError,
    is_carla_infrastructure_error,
)


def _ablation_flags(mutation_depth, method=None):
    if method == "random_scenario_local":
        return {
            "global_scenario_search": False,
            "local_language_fuzzing": mutation_depth > 0,
            "scenario_sampling": "independent_random",
        }
    return {
        "global_scenario_search": True,
        "local_language_fuzzing": mutation_depth > 0,
    }


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", dict, fitness=creator.FitnessMin)

class ScenarioOptimizer:
    def __init__(
        self,
        static_scenario_path,
        seed_scenario_path,
        mutation_depth,
        success_distance,
        pop_size,
        ngen,
        cxpb,
        mutpb,
        instruction=None,
        enable_logging=True,
        vla_model="lmdrive",
        gpu_id=0,
        manifest_id=None,
        instruction_source="manual",
        max_simulations=None,
        time_budget_seconds=None,
        random_seed=None,
        method="full",
        output_root=None,
        operators=None,
        oracle_checks=None,
        scenario_duration_frames=500,
        min_npc_count=0,
        max_npc_count=3,
        llm_provider="deepseek",
        llm_model="deepseek-v4-flash",
        initial_population_until_budget=False,
    ):
        self.static_scenario_path = static_scenario_path
        self.seed_scenario_path = seed_scenario_path
        self.mutation_depth = mutation_depth
        self.success_distance = success_distance
        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.instruction = instruction
        self.enable_logging = enable_logging
        self.vla_model = vla_model
        self.gpu_id = gpu_id
        self.manifest_id = manifest_id
        self.instruction_source = instruction_source
        self.max_simulations = max_simulations
        self.time_budget_seconds = time_budget_seconds
        self.random_seed = random_seed
        self.method = method
        self.output_root = output_root
        self.operators = operators
        self.oracle_checks = oracle_checks
        self.scenario_duration_frames = scenario_duration_frames
        self.min_npc_count, self.max_npc_count = validate_npc_count_range(min_npc_count, max_npc_count)
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.initial_population_until_budget = bool(initial_population_until_budget)
        if (
            self.initial_population_until_budget
            and self.max_simulations is None
            and self.time_budget_seconds is None
        ):
            raise ValueError(
                "initial_population_until_budget requires a time or simulation budget"
            )
        bootstrap_model_environment(self.vla_model)
        
        # Metadata tracking
        self.start_time = time.time()
        self.simulation_count = 0
        self.total_failures = 0
        self.evaluation_cache = {}
        self.cache_hits = 0
        
        # Setup directory structure
        if self.enable_logging:
            self.base_dir = create_run_dir(
                self.output_root or "results",
                self.method,
                self.vla_model,
                self.manifest_id,
                self.random_seed,
            )
            self.log_dir = os.path.join(self.base_dir, "logs")
            self.failure_dir = os.path.join(self.base_dir, "failures")
            
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.failure_dir, exist_ok=True)
            self.checkpoint_recorder = HourlyCheckpointRecorder(
                output_dir=self.base_dir,
                start_time=self.start_time,
                method=self.method,
                model=self.vla_model,
                manifest_id=self.manifest_id,
            )
            run_dir_file = os.environ.get("VLADFUZZ_RUN_DIR_FILE")
            if run_dir_file:
                marker = os.path.abspath(run_dir_file)
                os.makedirs(os.path.dirname(marker), exist_ok=True)
                with open(marker, "a", encoding="utf-8") as file:
                    file.write(os.path.abspath(self.base_dir) + "\n")
            print(f"📂 Optimization output directory: {self.base_dir}")
        else:
            self.log_dir = None
            self.failure_dir = None
            self.checkpoint_recorder = None

        # Initialize Fuzzer Instance specific to this optimizer
        print("Initializing LocalFuzzer for Optimization...")
        self.fuzzer = LocalFuzzer(
            static_scenario=self.static_scenario_path,
            mutation_depth=self.mutation_depth,
            success_distance=self.success_distance,
            vla_model=self.vla_model,
            gpu_id=self.gpu_id,
            verbose=False,
            semantic_pruning=False,
            operators=self.operators,
            oracle_checks=self.oracle_checks,
            scenario_duration_frames=self.scenario_duration_frames,
            api_provider=self.llm_provider,
            model_name=self.llm_model,
        )

        # Load seed data for initialization reference
        with open(self.seed_scenario_path, 'r') as f:
            self.seed_data = json.load(f)
            
        # Load static data for reference (available spawns etc)
        with open(self.static_scenario_path, 'r') as f:
            self.static_data = json.load(f)
        
        self.ego_spawn_index = self.seed_data['ego_car']['spawn_point_index']
        self.available_spawns = [
            i for i in range(len(self.static_data['spawn_points']))
            if i != self.ego_spawn_index
        ]
        self.route_biased_spawns = self._compute_route_biased_spawns()
        
        self.toolbox = base.Toolbox()
        self.setup_toolbox()

    def _simulation_budget_reached(self):
        return self.max_simulations is not None and self.simulation_count >= self.max_simulations

    def _time_budget_reached(self):
        return self.time_budget_seconds is not None and (time.time() - self.start_time) >= self.time_budget_seconds

    def _budget_reached(self):
        return self._simulation_budget_reached() or self._time_budget_reached()

    def _budget_status(self):
        parts = []
        if self.max_simulations is not None:
            parts.append(f"simulations {self.simulation_count}/{self.max_simulations}")
        if self.time_budget_seconds is not None:
            parts.append(f"time {time.time() - self.start_time:.1f}/{self.time_budget_seconds:.1f}s")
        return ", ".join(parts) if parts else "unbounded"
        
    def _spawn_xy(self, spawn_index):
        spawn = self.static_data['spawn_points'][spawn_index]
        return float(spawn['x']), float(spawn['y'])

    def _route_xy_points(self):
        route_info = self.seed_data.get('route_info', {})
        waypoints = route_info.get('route_waypoints') or route_info.get('waypoints') or []
        points = []
        for waypoint in waypoints:
            if isinstance(waypoint, dict) and 'x' in waypoint and 'y' in waypoint:
                points.append((float(waypoint['x']), float(waypoint['y'])))
            elif isinstance(waypoint, dict) and isinstance(waypoint.get('location'), dict):
                location = waypoint['location']
                if 'x' in location and 'y' in location:
                    points.append((float(location['x']), float(location['y'])))
        return points

    @staticmethod
    def _point_to_segment_distance(px, py, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def _distance_to_route(self, spawn_index):
        route_points = self._route_xy_points()
        px, py = self._spawn_xy(spawn_index)
        if len(route_points) < 2:
            ego_x, ego_y = self._spawn_xy(self.ego_spawn_index)
            return math.hypot(px - ego_x, py - ego_y)
        return min(
            self._point_to_segment_distance(px, py, ax, ay, bx, by)
            for (ax, ay), (bx, by) in zip(route_points, route_points[1:])
        )

    def _compute_route_biased_spawns(self, corridor_width=25.0, fallback_count=12):
        scored = [(self._distance_to_route(index), index) for index in self.available_spawns]
        close = [index for distance, index in scored if distance <= corridor_width]
        if close:
            return close
        return [index for _, index in sorted(scored)[:fallback_count]]

    def _choose_spawn(self, used_spawns, prefer_route=True):
        available = [s for s in self.available_spawns if s not in used_spawns]
        if not available:
            return None
        if prefer_route and self.route_biased_spawns and random.random() < 0.75:
            biased = [s for s in self.route_biased_spawns if s in available]
            if biased:
                return random.choice(biased)
        return random.choice(available)

    def _bias_npcs_towards_route(self, individual, probability=0.7):
        npc_list = individual.get('npc_vehicles', [])
        used_spawns = {individual['ego_car']['spawn_point_index']}
        for npc in npc_list:
            used_spawns.add(npc['spawn_point_index'])

        for npc in npc_list:
            if random.random() >= probability:
                continue
            current_spawn = npc['spawn_point_index']
            used_without_current = used_spawns - {current_spawn}
            biased_spawn = self._choose_spawn(used_without_current, prefer_route=True)
            if biased_spawn is not None:
                npc['spawn_point_index'] = biased_spawn
                npc['spawn_point_coordinates'] = self.static_data['spawn_points'][biased_spawn]
                used_spawns = used_without_current | {biased_spawn}
        individual['npc_vehicles'] = npc_list
        individual['npc_vehicle_count'] = len(npc_list)
        return individual

    def _scenario_hash(self, individual):
        clean = copy.deepcopy(dict(individual))
        clean.pop('__evaluation_log__', None)
        clean.pop('__evaluation_stopped_by_budget__', None)
        for npc in clean.get('npc_vehicles', []):
            npc.pop('spawn_point_coordinates', None)
        return json.dumps(clean, sort_keys=True, separators=(',', ':'))

    def random_npc(self, used_spawns, prefer_route=True):
        """Generate a random NPC configuration."""
        spawn_index = self._choose_spawn(used_spawns, prefer_route=prefer_route)
        if spawn_index is None:
            return None

        blueprint = random.choice(CarlaScenario.NPC_VEHICLE_BLUEPRINTS)
        
        # Check if special vehicle (no color)
        is_special = blueprint in CarlaScenario.SPECIAL_VEHICLES
        color = random.choice(CarlaScenario.VEHICLE_COLORS) if not is_special else None
        
        return {
            "blueprint": blueprint,
            "spawn_point_index": spawn_index,
            "speed_percentage_difference": random.uniform(-50.0, 50.0),
            "color": color
        }

    def init_individual(self):
        """Create a random individual based on the seed using CarlaScenario's generator."""
        
        npc_count = sample_npc_count(self.min_npc_count, self.max_npc_count)
        
        try:
            # Use the scenario manager's generation logic directly
            # This ensures all constraints (spawn points, etc.) are handled correctly
            individual = self.fuzzer.scenario_manager.generate_scenario_from_seed(
                seed_scenario_path=self.seed_scenario_path,
                npc_count=npc_count,
                min_speed_diff_perc=-50.0,
                max_speed_diff_perc=50.0,
                include_environment=True,
                return_dict=True  # Crucial: Get dict back, don't write file
            )
            individual = self._bias_npcs_towards_route(individual)
            return creator.Individual(individual)
        except Exception as e:
            raise RuntimeError(f"Failed to generate an individual within NPC range [{self.min_npc_count}, {self.max_npc_count}]") from e

    def evaluate(self, individual):
        """Evaluate the individual using LocalFuzzer directly with dictionary."""
        try:
            scenario_key = self._scenario_hash(individual)
            if scenario_key in self.evaluation_cache:
                self.cache_hits += 1
                cached = self.evaluation_cache[scenario_key]
                individual['__evaluation_log__'] = copy.deepcopy(cached['detailed_log'])
                individual['__evaluation_stopped_by_budget__'] = False
                return cached['fitness']

            if self.instruction:  
                instruction = self.instruction
            else:
                instruction = individual['route_info'].get('basic_instruction', "")
            
            # Pass the dictionary directly! No temp files needed.
            # Also enable failure logging and detailed return
            failure_path = self.failure_dir if self.failure_dir else "optimization_failures"
            
            safety, task, detailed_log = self.fuzzer.evaluate(
                individual, 
                instruction,
                save_failures=self.enable_logging,
                failure_dir=failure_path,
                return_detailed_log=True,
                stop_requested=self._time_budget_reached,
            )
            
            # Increment total simulation count by the number of nodes executed in this fuzzing tree
            self.simulation_count += len(detailed_log)
            
            # Count failures in this evaluation
            failures_in_eval = sum(1 for node in detailed_log if not node.get('valid', node.get('semantically_correct', True)))
            self.total_failures += failures_in_eval
            
            # Store detailed log in individual for later logging
            individual['__evaluation_log__'] = detailed_log
            individual['__evaluation_stopped_by_budget__'] = self.fuzzer.evaluation_stopped_by_budget
            if self.checkpoint_recorder:
                self.checkpoint_recorder.update(
                    now=time.time(),
                    executions=self.simulation_count,
                    failures=self.total_failures,
                    extra={
                        "evaluation_cache_size": len(self.evaluation_cache),
                        "evaluation_cache_hits": self.cache_hits,
                    },
                )
            if not self.fuzzer.evaluation_stopped_by_budget:
                self.evaluation_cache[scenario_key] = {
                    'fitness': (safety, task),
                    'detailed_log': copy.deepcopy(detailed_log),
                }
            
            return safety, task
            
        except Exception as e:
            if is_carla_infrastructure_error(e):
                raise CarlaInfrastructureError(str(e)) from e
            print(f"Evaluation failed: {e}")
            individual.pop('__evaluation_log__', None)
            individual.pop('__evaluation_stopped_by_budget__', None)
            return 1000.0, 1000.0 # Penalize non-infrastructure evaluation failure

    def mutate(self, individual):
        """Mutate an individual."""
        # 1. Mutate Weather (30% chance)
        if random.random() < 0.3:
            individual['weather'] = random.choice(CarlaScenario.WEATHER_PRESETS)
        
        # 2. Mutate NPCs
        npc_list = individual['npc_vehicles']
        
        if not npc_list:
            if self.max_npc_count > 0:
                new_npc = self.random_npc(set(), prefer_route=True)
                if new_npc:
                    npc_list.append(new_npc)
        else:
            mutation_type = random.choice(['modify', 'add', 'remove'])
            
            if mutation_type == 'modify':
                # Modify attributes of a random NPC
                idx = random.randrange(len(npc_list))
                npc = npc_list[idx]
                
                sub_mut = random.choice(['speed', 'color', 'position'])
                
                if sub_mut == 'speed':
                    npc['speed_percentage_difference'] = random.uniform(-50.0, 50.0)
                elif sub_mut == 'color':
                    is_special = npc['blueprint'] in CarlaScenario.SPECIAL_VEHICLES
                    npc['color'] = random.choice(CarlaScenario.VEHICLE_COLORS) if not is_special else None
                elif sub_mut == 'position':
                    # Move to a new spawn point
                    current_spawns = {n['spawn_point_index'] for n in npc_list}
                    available = [s for s in self.available_spawns if s not in current_spawns]
                    new_spawn = self._choose_spawn(current_spawns, prefer_route=True)
                    if new_spawn is not None:
                        npc['spawn_point_index'] = new_spawn
                        npc['spawn_point_coordinates'] = self.static_data['spawn_points'][new_spawn]
                        
            elif mutation_type == 'add':
                if len(npc_list) < self.max_npc_count:
                    current_spawns = {n['spawn_point_index'] for n in npc_list}
                    new_npc = self.random_npc(current_spawns, prefer_route=True)
                    if new_npc:
                        npc_list.append(new_npc)
                        
            elif mutation_type == 'remove':
                if len(npc_list) > self.min_npc_count:
                    npc_list.pop(random.randrange(len(npc_list)))
        
        individual['npc_vehicles'] = npc_list
        individual['npc_vehicle_count'] = len(npc_list)
        
        return individual,

    def mate(self, ind1, ind2):
        """Crossover between two individuals."""
        # 1. Swap Weather (50% chance)
        if random.random() < 0.5:
            ind1['weather'], ind2['weather'] = ind2['weather'], ind1['weather']
        
        # 2. NPC Crossover
        # Strategy: Mix NPCs from both parents
        # Combine all NPCs
        pool = ind1['npc_vehicles'] + ind2['npc_vehicles']
        random.shuffle(pool)
        
        # Split back into two lists, respecting max/min counts
        # Constraint: No duplicate spawn points in a single individual
        
        def fill_individual(target_size):
            new_list = []
            used_spawns = set()
            for npc in pool:
                if len(new_list) >= target_size:
                    break
                if npc['spawn_point_index'] not in used_spawns:
                    # Clone to avoid shared reference issues
                    new_list.append(copy.deepcopy(npc))
                    used_spawns.add(npc['spawn_point_index'])
            return new_list

        # Determine sizes
        size1 = len(ind1['npc_vehicles'])
        size2 = len(ind2['npc_vehicles'])
        
        ind1['npc_vehicles'] = fill_individual(size1)
        ind1['npc_vehicle_count'] = len(ind1['npc_vehicles'])
        
        ind2['npc_vehicles'] = fill_individual(size2)
        ind2['npc_vehicle_count'] = len(ind2['npc_vehicles'])
        
        return ind1, ind2

    def setup_toolbox(self):
        self.toolbox.register("individual", self.init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", self.mate)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selNSGA2)

    @staticmethod
    def _round_floats(o):
        if isinstance(o, float):
            return round(o, 2)
        if isinstance(o, dict):
            return {k: ScenarioOptimizer._round_floats(v) for k, v in o.items()}
        if isinstance(o, list):
            return [ScenarioOptimizer._round_floats(v) for v in o]
        return o

    @staticmethod
    def _json_serializer(obj):
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if hasattr(obj, 'value'):
            return obj.value
        return str(obj)

    def _clean_individual_for_logging(self, ind):
        ind_dict = dict(ind)
        eval_log = ind_dict.pop('__evaluation_log__', None)
        evaluation_stopped_by_budget = ind_dict.pop('__evaluation_stopped_by_budget__', False)
        return self._round_floats(ind_dict), eval_log, evaluation_stopped_by_budget

    def _individual_summary(
        self,
        ind_id,
        ind,
        ind_dict,
        eval_log,
        evaluation_stopped_by_budget,
        phase=None,
        selected=False,
    ):
        sim_count = len(eval_log) if eval_log else 0
        summary = {
            "id": ind_id,
            "dir": str(ind_id),
            "fitness": ind.fitness.values if ind.fitness.valid else None,
            "npc_count": ind_dict.get('npc_vehicle_count'),
            "weather": ind_dict.get('weather'),
            "simulation_count": sim_count,
            "evaluation_stopped_by_budget": evaluation_stopped_by_budget,
            "selected": selected,
        }
        if phase is not None:
            summary["phase"] = phase
        return summary

    def save_evaluated_individual_log(self, gen, ind_id, ind, phase):
        """Persist an evaluated individual immediately after its fitness is available."""
        if not self.enable_logging:
            return

        gen_dir = os.path.join(self.log_dir, f"gen_{gen}")
        ind_dir = os.path.join(gen_dir, str(ind_id))
        os.makedirs(ind_dir, exist_ok=True)

        ind_dict, eval_log, evaluation_stopped_by_budget = self._clean_individual_for_logging(ind)

        try:
            with open(os.path.join(ind_dir, "config.json"), 'w') as f:
                json.dump(ind_dict, f, indent=4)
            if eval_log:
                with open(os.path.join(ind_dir, "result.json"), 'w') as f:
                    json.dump(eval_log, f, indent=4, default=self._json_serializer)
        except Exception as e:
            print(f"❌ Failed to save evaluated individual {ind_id}: {e}")
            return

        metadata_path = os.path.join(gen_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {"generation": gen, "individuals_summary": []}
        else:
            metadata = {"generation": gen, "individuals_summary": []}

        summary = self._individual_summary(
            ind_id,
            ind,
            ind_dict,
            eval_log,
            evaluation_stopped_by_budget,
            phase=phase,
            selected=False,
        )
        metadata["timestamp"] = datetime.now().isoformat()
        metadata["budget_exhausted_at_save"] = self._budget_reached()
        metadata["individuals_summary"] = [
            item for item in metadata.get("individuals_summary", [])
            if item.get("id") != ind_id
        ]
        metadata["individuals_summary"].append(summary)
        metadata["evaluated_count"] = len(metadata["individuals_summary"])

        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"📋 Saved evaluated individual {ind_id} for gen {gen} to {ind_dir}")
        except Exception as e:
            print(f"❌ Failed to save evaluated metadata for generation {gen}: {e}")

    def save_generation_metadata(
        self,
        gen,
        pop,
        selected_individual_ids,
        generation_status="complete",
        evaluated_offspring_count=None,
        planned_offspring_count=None,
    ):
        """
        Update generation metadata after selection. Individual config/result files
        are written incrementally by save_evaluated_individual_log().
        """
        if not self.enable_logging:
            return
        
        gen_dir = os.path.join(self.log_dir, f"gen_{gen}")
        os.makedirs(gen_dir, exist_ok=True)

        existing_by_id = {}
        metadata_path = os.path.join(gen_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
                existing_by_id = {
                    item.get("id"): item
                    for item in existing_metadata.get("individuals_summary", [])
                    if item.get("id") is not None
                }
            except Exception:
                existing_by_id = {}

        selected_set = set(selected_individual_ids)
        selected_from_current_generation = 0
        gen_failures = 0
        gen_api_failures = []
        for ind_id in sorted(existing_by_id, key=str):
            summary_entry = existing_by_id[ind_id]
            summary_entry["selected"] = ind_id in selected_set
            if summary_entry["selected"]:
                selected_from_current_generation += 1
            result_path = os.path.join(gen_dir, str(ind_id), "result.json")
            if os.path.exists(result_path):
                try:
                    with open(result_path, 'r') as f:
                        eval_log = json.load(f)
                    gen_failures += sum(1 for node in eval_log if not node.get('valid', node.get('semantically_correct', True)))
                    for node in eval_log:
                        if not node.get('mutation_success', True):
                            gen_api_failures.append({
                                "individual_id": ind_id,
                                "node_depth": node.get('depth'),
                                "operator": node.get('applied_operator'),
                                "error": node.get('error_message')
                            })
                except Exception as e:
                    print(f"❌ Failed to summarize individual {ind_id} result log: {e}")

        metadata = {
            "generation": gen,
            "timestamp": datetime.now().isoformat(),
            "population_size": len(pop),
            "generation_status": generation_status,
            "budget_exhausted_at_save": self._budget_reached(),
            "evaluated_offspring_count": evaluated_offspring_count,
            "planned_offspring_count": planned_offspring_count,
            "selected_individual_ids": selected_individual_ids,
            "selected_from_current_generation": selected_from_current_generation,
            "selected_from_previous_generations": max(0, len(pop) - selected_from_current_generation),
            "evaluated_count": len(existing_by_id),
            "individuals_summary": [existing_by_id[ind_id] for ind_id in sorted(existing_by_id, key=str)],
        }
            
        metadata["failures_detected"] = gen_failures
        metadata["api_failures"] = gen_api_failures
            
        # 4. Save metadata file
        metadata_path = os.path.join(gen_dir, "metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"📋 Saved generation {gen} logs to {gen_dir}")
        except Exception as e:
            print(f"❌ Failed to save generation {gen} metadata: {e}")

    def save_final_metadata(self):
        """Save overall optimization metadata."""
        if not self.enable_logging:
            return
            
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Analyze failures
        failure_stats = collect_failure_analysis(self.failure_dir) if self.failure_dir else {}
        
        metadata = {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "total_duration_seconds": round(duration, 2),
            "total_simulations_executed": self.simulation_count,
            "total_failures_detected": self.total_failures,
            "failure_analysis": failure_stats,
            "configuration": {
                "pop_size": self.pop_size,
                "ngen": self.ngen,
                "cxpb": self.cxpb,
                "mutpb": self.mutpb,
                "static_scenario": self.static_scenario_path,
                "seed_scenario": self.seed_scenario_path,
                "vla_model": self.vla_model,
                "manifest_id": self.manifest_id,
                "instruction_source": self.instruction_source,
                "instruction": self.instruction,
                "max_simulations": self.max_simulations,
                "time_budget_seconds": self.time_budget_seconds,
                "budget_exhausted": self._budget_reached(),
                "method": self.method,
                "random_seed": self.random_seed,
                "operators": [operator.value for operator in self.operators] if self.operators else None,
                "oracle_checks": self.oracle_checks,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "mutation_depth": self.mutation_depth,
                "initial_population_until_budget": self.initial_population_until_budget,
                **_ablation_flags(self.mutation_depth, method=self.method),
                "min_npc_count": self.min_npc_count,
                "max_npc_count": self.max_npc_count,
                "fitness_policy": "worst_case_min_avg_ettc_and_average_path_tracking_quality",
                "route_biased_spawn_count": len(self.route_biased_spawns),
                "evaluation_cache_size": len(self.evaluation_cache),
                "evaluation_cache_hits": self.cache_hits,
            }
        }
        if self.checkpoint_recorder:
            metadata["hourly_checkpoints"] = list(self.checkpoint_recorder.checkpoints)
        
        metadata_path = os.path.join(self.base_dir, "metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"📋 Saved final optimization metadata to {metadata_path}")
        except Exception as e:
            print(f"❌ Failed to save final metadata: {e}")

    def _run_initial_population_until_budget(self):
        """Evaluate independently sampled initial individuals until the budget expires."""
        population = []
        print("Evaluating an unbounded initial population until the budget is exhausted...")

        while not self._budget_reached():
            individual = self.toolbox.individual()
            individual.fitness.values = self.toolbox.evaluate(individual)
            individual_id = f"ind_{len(population)}"
            population.append(individual)
            self.save_evaluated_individual_log(
                0,
                individual_id,
                individual,
                phase="initial_random",
            )

        self.save_generation_metadata(
            0,
            population,
            selected_individual_ids=[],
            generation_status="budget_exhausted",
            evaluated_offspring_count=len(population),
            planned_offspring_count=None,
        )
        self.save_final_metadata()
        print(
            "Initial-population baseline stopped after "
            f"{len(population)} scenarios: {self._budget_status()}"
        )
        return population

    def run(self):
        if self.initial_population_until_budget:
            return self._run_initial_population_until_budget()

        pop = self.toolbox.population(n=self.pop_size)
        
        # Evaluate initial population
        print(f"Evaluating initial population ({self.pop_size} individuals)...")
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        evaluated_initial_ids = []
        for i, ind in enumerate(invalid_ind):
            if self._budget_reached():
                print(f"Budget reached during initial population: {self._budget_status()}")
                break
            fit = self.toolbox.evaluate(ind)
            ind.fitness.values = fit
            ind_id = f"ind_{i}"
            evaluated_initial_ids.append(ind_id)
            self.save_evaluated_individual_log(0, ind_id, ind, phase="initial")
            if self._budget_reached():
                print(f"Budget reached after evaluating initial individual: {self._budget_status()}")
                break
            
        # Assign crowding distance for NSGA-II selection
        pop = [ind for ind in pop if ind.fitness.valid]
        if not pop:
            print("No initial individuals were evaluated before budget exhaustion.")
            self.save_final_metadata()
            return []
        pop = self.toolbox.select(pop, len(pop))
        
        # Log initial population (Generation 0)
        self.save_generation_metadata(0, pop, selected_individual_ids=evaluated_initial_ids[:len(pop)])
        print("Initial population evaluated.")
        if self._budget_reached():
            print("Stopping after initial population because budget is exhausted.")
            self.save_final_metadata()
            return pop

        for gen in range(1, self.ngen + 1):
            if self._budget_reached():
                print(f"Stopping before generation {gen} because budget is exhausted: {self._budget_status()}")
                break

            print(f"\n--- Generation {gen}/{self.ngen} ---")
            
            # Select the next generation individuals (Tournmanet/NSGA2 selection)
            # For NSGA-II, typically we select parents, mate/mutate to create offspring, then merge and select best N
            selection_size = len(pop) - (len(pop) % 4)
            if selection_size < 4:
                print("Population too small for tournament-DCD selection; stopping.")
                break
            offspring = tools.selTournamentDCD(pop, selection_size)
            offspring = [copy.deepcopy(ind) for ind in offspring]
            
            # Apply crossover and mutation
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(ind1, ind2)
                    del ind1.fitness.values
                    del ind2.fitness.values

            for ind in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(ind)
                    del ind.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            print(f"Evaluating {len(invalid_ind)} offspring...")
            evaluated_offspring = []
            evaluated_offspring_ids = []
            for i, ind in enumerate(invalid_ind):
                if self._budget_reached():
                    print(f"Budget reached: {self._budget_status()}")
                    break
                fit = self.toolbox.evaluate(ind)
                ind.fitness.values = fit
                evaluated_offspring.append(ind)
                ind_id = f"ind_{i}"
                evaluated_offspring_ids.append(ind_id)
                self.save_evaluated_individual_log(gen, ind_id, ind, phase="evaluated")
                if self._budget_reached():
                    print(f"Budget reached after evaluating offspring: {self._budget_status()}")
                    break

            # Select the next generation population
            valid_offspring = [ind for ind in offspring if ind.fitness.valid]
            if not valid_offspring and self._budget_reached():
                print("No evaluated offspring remain after budget exhaustion; keeping previous population.")
                self.save_generation_metadata(
                    gen,
                    pop,
                    selected_individual_ids=[],
                    generation_status="partial_budget_exhausted",
                    evaluated_offspring_count=len(evaluated_offspring),
                    planned_offspring_count=len(invalid_ind),
                )
                break

            previous_pop = pop
            selected_pool = previous_pop + valid_offspring
            pop = self.toolbox.select(selected_pool, self.pop_size)
            selected_individual_ids = []
            for selected in pop:
                for idx, offspring_ind in enumerate(evaluated_offspring):
                    if selected is offspring_ind:
                        selected_individual_ids.append(evaluated_offspring_ids[idx])
                        break
            
            # Statistics
            fits = [ind.fitness.values for ind in pop]
            
            print(f"Generation {gen} Statistics:")
            for i, fit in enumerate(fits):
                print(f"  Ind {i}: Safety={fit[0]:.2f}, Task={fit[1]:.2f}")
            
            # Log current generation
            generation_status = "complete"
            if self._budget_reached() and (
                len(evaluated_offspring) < len(invalid_ind)
                or any(ind.get('__evaluation_stopped_by_budget__', False) for ind in evaluated_offspring)
            ):
                generation_status = "partial_budget_exhausted"
            self.save_generation_metadata(
                gen,
                pop,
                selected_individual_ids=selected_individual_ids,
                generation_status=generation_status,
                evaluated_offspring_count=len(evaluated_offspring),
                planned_offspring_count=len(invalid_ind),
            )
            if self._budget_reached():
                print(f"Stopping after generation {gen} because budget is exhausted: {self._budget_status()}")
                break
            
        # Save final metadata at the end of run
        self.save_final_metadata()

        return pop

def main():
    parser = argparse.ArgumentParser(description="Run NSGA-II scenario optimization with a selected VLA model.")
    parser.add_argument("--model", choices=["lmdrive", "simlingo", "bevdriver"], default="lmdrive")
    parser.add_argument("--static-scenario", help="Static scenario JSON for direct input mode")
    parser.add_argument("--seed-scenario", help="Seed dynamic scenario JSON for direct input mode")
    parser.add_argument("--instruction", help="Navigation instruction for direct input mode")
    parser.add_argument("--manifest", help="Optional JSONL experiment manifest produced by build_instruction_manifest.py")
    parser.add_argument("--manifest-id", help="Manifest row id, for example town01_t_junction_1/seed_1")
    parser.add_argument("--instruction-source", choices=INSTRUCTION_SOURCES, default="manual")
    parser.add_argument("--mutation-depth", type=int, default=2)
    parser.add_argument("--success-distance", type=float, default=5.0)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--ngen", type=int, default=3)
    parser.add_argument("--cxpb", type=float, default=0.7)
    parser.add_argument("--mutpb", type=float, default=0.3)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-simulations", type=int, help="Stop after this many executed scenario simulations, after the initial population is evaluated")
    parser.add_argument("--time-budget-seconds", type=float, help="Stop starting new evaluations after this many wall-clock seconds")
    parser.add_argument("--time-budget-minutes", type=float, help="Stop starting new evaluations after this many wall-clock minutes")
    parser.add_argument("--random-seed", type=int, help="Random seed for scenario search operators")
    parser.add_argument("--method", default="full", help="Experiment method label stored in metadata")
    parser.add_argument("--output-root", default="results/rq2/runs", help="Root directory for experiment run output")
    parser.add_argument("--operators", default="all", help="Comma-separated local mutation operators, or 'all'")
    parser.add_argument("--oracle-checks", default="default", help="Comma-separated oracle checks: collision,stuck,lane,speeding,timeout,out_of_bounds,other; or all/default/none.")
    parser.add_argument("--scenario-duration-frames", type=int, default=500, help="Runtime duration in CARLA frames for each VLAD-Fuzz scenario execution.")
    parser.add_argument("--min-npc-count", type=int, default=0)
    parser.add_argument("--max-npc-count", type=int, default=3)
    parser.add_argument("--llm-provider", default="deepseek", choices=["gemini", "qwen", "deepseek"])
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument(
        "--initial-population-until-budget",
        action="store_true",
        help=(
            "Repeatedly sample and evaluate independent initial individuals until "
            "the budget expires, without NSGA-II selection or offspring generation."
        ),
    )
    parser.add_argument("--disable-logging", action="store_true")
    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)

    selection = apply_manifest_selection(
        manifest_path=args.manifest,
        manifest_id=args.manifest_id,
        instruction_source=args.instruction_source,
        static_scenario=args.static_scenario,
        dynamic_scenario=args.seed_scenario,
        instruction=args.instruction,
    )

    print("Experiment input:")
    print(f"  Manifest ID: {selection['manifest_id']}")
    print(f"  Static scenario: {selection['static_scenario']}")
    print(f"  Seed scenario: {selection['dynamic_scenario']}")
    print(f"  Instruction source: {selection['instruction_source']}")
    print(f"  Instruction: {selection['instruction']}")
    operators = parse_operator_list(args.operators)
    oracle_checks = parse_oracle_checks(args.oracle_checks)
    print(f"  Operators: {', '.join(operator.value for operator in operators)}")
    print(f"  Oracle checks: {', '.join(name for name, enabled in oracle_checks.items() if enabled) or 'none'}")
    print(f"  NPC count range: [{args.min_npc_count}, {args.max_npc_count}]")
    time_budget_seconds = args.time_budget_seconds
    if args.time_budget_minutes is not None:
        time_budget_seconds = args.time_budget_minutes * 60.0

    optimizer = ScenarioOptimizer(
        static_scenario_path=selection["static_scenario"],
        seed_scenario_path=selection["dynamic_scenario"],
        mutation_depth=args.mutation_depth,
        success_distance=args.success_distance,
        pop_size=args.pop_size,
        ngen=args.ngen,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        instruction=selection["instruction"],
        enable_logging=not args.disable_logging,
        vla_model=args.model,
        gpu_id=args.gpu_id,
        manifest_id=selection["manifest_id"],
        instruction_source=selection["instruction_source"],
        max_simulations=args.max_simulations,
        time_budget_seconds=time_budget_seconds,
        random_seed=args.random_seed,
        method=args.method,
        output_root=args.output_root,
        operators=operators,
        oracle_checks=oracle_checks,
        scenario_duration_frames=args.scenario_duration_frames,
        min_npc_count=args.min_npc_count,
        max_npc_count=args.max_npc_count,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        initial_population_until_budget=args.initial_population_until_budget,
    )

    final_pop = optimizer.run()

    print("\n=== Optimization Complete ===")
    if args.initial_population_until_budget:
        print(f"Evaluated random scenarios: {len(final_pop)}")
    else:
        print("Pareto Front (Final Population):")
        for i, ind in enumerate(final_pop):
            print(f"Individual {i}:")
            print(f"  Fitness: {ind.fitness.values}")
            print(f"  Weather: {ind.get('weather')}")
            print(f"  NPC Count: {ind.get('npc_vehicle_count')}")
            print(f"  NPCs: {[n['blueprint'] for n in ind.get('npc_vehicles', [])]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CarlaInfrastructureError as exc:
        print(f"CARLA infrastructure failure: {exc}")
        raise SystemExit(CARLA_INFRASTRUCTURE_EXIT_CODE)

import argparse
import copy
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vladfuzz_workflows.local_fuzzer import LocalFuzzer
from scenario.carla_scenario import CarlaScenario
from vladfuzz_runtime.experiment_manifest import INSTRUCTION_SOURCES, apply_manifest_selection
from vladfuzz_runtime.failure_analysis import collect_failure_analysis
from vladfuzz_runtime.run_metadata import create_run_dir


ADS_BASELINE_METHODS = ("av_fuzzer", "drive_fuzz")


def _finite_float(value, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _failure_bonus(result: Dict) -> float:
    risk = 0.0
    if result.get("collision", False):
        risk += 100.0
    if not result.get("completed", True):
        risk += 20.0
    if result.get("error"):
        risk += 10.0
    return risk


def compute_av_fuzzer_risk(result: Dict) -> float:
    """AV-Fuzzer-style fitness: prioritize collisions and near-collision proximity."""
    ettc_stats = result.get("ettc_stats") or {}
    detection_radius = _finite_float(ettc_stats.get("detection_radius"), 10.0)
    avg_ettc = _finite_float(ettc_stats.get("avg_ettc"), 15.0)
    min_distance = _finite_float(ettc_stats.get("min_distance"), detection_radius)
    min_brake_margin = _finite_float(ettc_stats.get("min_brake_margin"), detection_radius)

    proximity_risk = max(0.0, detection_radius - min_distance)
    brake_risk = max(0.0, -min_brake_margin) + 0.25 * max(0.0, detection_radius - min_brake_margin)
    ettc_risk = max(0.0, 15.0 - avg_ettc)

    return _failure_bonus(result) + proximity_risk + brake_risk + ettc_risk


def compute_drive_fuzz_risk(result: Dict) -> float:
    """DriveFuzz-style fitness: prioritize degraded driving quality and unsafe margins."""
    driving_quality_stats = result.get("driving_quality_stats") or {}
    if "driving_quality_deductions" in driving_quality_stats:
        return _failure_bonus(result) + _finite_float(
            driving_quality_stats.get("driving_quality_deductions"),
            0.0,
        )

    ettc_stats = result.get("ettc_stats") or {}
    path_stats = result.get("path_deviation_stats") or {}
    detection_radius = _finite_float(ettc_stats.get("detection_radius"), 10.0)
    min_distance = _finite_float(ettc_stats.get("min_distance"), detection_radius)
    min_brake_margin = _finite_float(ettc_stats.get("min_brake_margin"), detection_radius)
    danger_frames = _finite_float(ettc_stats.get("danger_frames"), 0.0)
    path_quality = _finite_float(path_stats.get("path_tracking_quality"), 1.0)

    quality_risk = 20.0 * max(0.0, 1.0 - path_quality)
    proximity_risk = 1.0 / max(0.1, min_distance)
    brake_risk = max(0.0, -min_brake_margin)
    danger_risk = 0.05 * danger_frames

    return _failure_bonus(result) + quality_risk + proximity_risk + brake_risk + danger_risk


class ADSBaselineRunner:
    """Re-implement ADS scenario fuzzing baselines on VLAD-Fuzz's execution stack."""

    def __init__(
        self,
        *,
        method: str,
        static_scenario: str,
        seed_scenario: str,
        instruction: str,
        model: str,
        gpu_id: int,
        manifest_id: Optional[str],
        instruction_source: str,
        max_simulations: int,
        random_seed: Optional[int],
        output_root: str,
        success_distance: float,
        pop_size: int,
        max_gens: int,
        max_cycles: int,
        max_mutations: int,
        cxpb: float,
        mutpb: float,
    ):
        if method not in ADS_BASELINE_METHODS:
            raise ValueError(f"Unsupported ADS baseline method: {method}")
        if max_simulations < 1:
            raise ValueError("--max-simulations must be positive.")

        self.method = method
        self.static_scenario = static_scenario
        self.seed_scenario = seed_scenario
        self.instruction = instruction
        self.model = model
        self.gpu_id = gpu_id
        self.manifest_id = manifest_id
        self.instruction_source = instruction_source
        self.max_simulations = max_simulations
        self.random_seed = random_seed
        self.success_distance = success_distance
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.max_cycles = max_cycles
        self.max_mutations = max_mutations
        self.cxpb = cxpb
        self.mutpb = mutpb

        self.base_dir = create_run_dir(output_root, method, model, manifest_id, random_seed)
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.failure_dir = os.path.join(self.base_dir, "failures")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.failure_dir, exist_ok=True)

        with open(seed_scenario, "r", encoding="utf-8") as file:
            self.seed_data = json.load(file)
        with open(static_scenario, "r", encoding="utf-8") as file:
            self.static_data = json.load(file)

        self.ego_spawn_index = self.seed_data["ego_car"]["spawn_point_index"]
        self.available_spawns = [
            index
            for index in range(len(self.static_data["spawn_points"]))
            if index != self.ego_spawn_index
        ]
        self.route_biased_spawns = self._compute_route_biased_spawns()

        self.fuzzer = LocalFuzzer(
            static_scenario=static_scenario,
            mutation_depth=0,
            success_distance=success_distance,
            vla_model=model,
            gpu_id=gpu_id,
            verbose=False,
        )

        self.start_time = 0.0
        self.simulation_count = 0
        self.total_failures = 0
        self.evaluations: List[Dict] = []
        self.evaluation_cache: Dict[str, Dict] = {}
        self.cache_hits = 0

    def _simulation_budget_reached(self) -> bool:
        return self.simulation_count >= self.max_simulations

    def _spawn_xy(self, spawn_index: int) -> Tuple[float, float]:
        spawn = self.static_data["spawn_points"][spawn_index]
        return float(spawn["x"]), float(spawn["y"])

    def _route_xy_points(self) -> List[Tuple[float, float]]:
        route_info = self.seed_data.get("route_info", {})
        waypoints = route_info.get("route_waypoints") or route_info.get("waypoints") or []
        points = []
        for waypoint in waypoints:
            if isinstance(waypoint, dict) and "x" in waypoint and "y" in waypoint:
                points.append((float(waypoint["x"]), float(waypoint["y"])))
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

    def _distance_to_route(self, spawn_index: int) -> float:
        route_points = self._route_xy_points()
        px, py = self._spawn_xy(spawn_index)
        if len(route_points) < 2:
            ego_x, ego_y = self._spawn_xy(self.ego_spawn_index)
            return math.hypot(px - ego_x, py - ego_y)
        return min(
            self._point_to_segment_distance(px, py, ax, ay, bx, by)
            for (ax, ay), (bx, by) in zip(route_points, route_points[1:])
        )

    def _compute_route_biased_spawns(self, corridor_width: float = 25.0, fallback_count: int = 12) -> List[int]:
        scored = [(self._distance_to_route(index), index) for index in self.available_spawns]
        close = [index for distance, index in scored if distance <= corridor_width]
        if close:
            return close
        return [index for _, index in sorted(scored)[:fallback_count]]

    def _choose_spawn(self, used_spawns, prefer_route: bool = True) -> Optional[int]:
        available = [spawn for spawn in self.available_spawns if spawn not in used_spawns]
        if not available:
            return None
        if prefer_route and random.random() < 0.8:
            biased = [spawn for spawn in self.route_biased_spawns if spawn in available]
            if biased:
                return random.choice(biased)
        return random.choice(available)

    @staticmethod
    def _scenario_hash(scenario: Dict) -> str:
        clean = copy.deepcopy(scenario)
        clean.pop("__evaluation_log__", None)
        for npc in clean.get("npc_vehicles", []):
            npc.pop("spawn_point_coordinates", None)
        return json.dumps(clean, sort_keys=True, separators=(",", ":"))

    def _random_npc(self, used_spawns, prefer_route: bool = True) -> Optional[Dict]:
        spawn_index = self._choose_spawn(used_spawns, prefer_route=prefer_route)
        if spawn_index is None:
            return None
        blueprint = random.choice(CarlaScenario.NPC_VEHICLE_BLUEPRINTS)
        is_special = blueprint in CarlaScenario.SPECIAL_VEHICLES
        return {
            "blueprint": blueprint,
            "spawn_point_index": spawn_index,
            "spawn_point_coordinates": self.static_data["spawn_points"][spawn_index],
            "speed_percentage_difference": random.uniform(-50.0, 50.0),
            "color": random.choice(CarlaScenario.VEHICLE_COLORS) if not is_special else None,
        }

    def _initial_scenario(self) -> Dict:
        scenario = copy.deepcopy(self.seed_data)
        scenario["weather"] = random.choice(CarlaScenario.WEATHER_PRESETS)
        used_spawns = {self.ego_spawn_index}
        npc_count = random.randint(2, 6)
        npc_vehicles = []
        for _ in range(npc_count):
            npc = self._random_npc(used_spawns, prefer_route=True)
            if npc is None:
                break
            npc_vehicles.append(npc)
            used_spawns.add(npc["spawn_point_index"])
        scenario["npc_vehicles"] = npc_vehicles
        scenario["npc_vehicle_count"] = len(npc_vehicles)
        return scenario

    def _mutate_scenario(self, scenario: Dict, local: bool = False) -> Dict:
        mutated = copy.deepcopy(scenario)
        npc_list = mutated.get("npc_vehicles", [])
        if random.random() < (0.2 if local else 0.35):
            mutated["weather"] = random.choice(CarlaScenario.WEATHER_PRESETS)

        mutation_type = random.choice(["speed", "position", "vehicle", "add", "remove"])
        if not npc_list:
            mutation_type = "add"

        if mutation_type == "speed" and npc_list:
            npc = random.choice(npc_list)
            if local:
                npc["speed_percentage_difference"] = max(
                    -50.0,
                    min(50.0, npc["speed_percentage_difference"] + random.uniform(-15.0, 15.0)),
                )
            else:
                npc["speed_percentage_difference"] = random.uniform(-50.0, 50.0)
        elif mutation_type == "position" and npc_list:
            npc = random.choice(npc_list)
            used_spawns = {mutated["ego_car"]["spawn_point_index"]}
            used_spawns.update(other["spawn_point_index"] for other in npc_list if other is not npc)
            new_spawn = self._choose_spawn(used_spawns, prefer_route=True)
            if new_spawn is not None:
                npc["spawn_point_index"] = new_spawn
                npc["spawn_point_coordinates"] = self.static_data["spawn_points"][new_spawn]
        elif mutation_type == "vehicle" and npc_list:
            npc = random.choice(npc_list)
            blueprint = random.choice(CarlaScenario.NPC_VEHICLE_BLUEPRINTS)
            npc["blueprint"] = blueprint
            npc["color"] = (
                random.choice(CarlaScenario.VEHICLE_COLORS)
                if blueprint not in CarlaScenario.SPECIAL_VEHICLES
                else None
            )
        elif mutation_type == "add" and len(npc_list) < 8:
            used_spawns = {mutated["ego_car"]["spawn_point_index"]}
            used_spawns.update(npc["spawn_point_index"] for npc in npc_list)
            npc = self._random_npc(used_spawns, prefer_route=True)
            if npc is not None:
                npc_list.append(npc)
        elif mutation_type == "remove" and len(npc_list) > 1:
            npc_list.pop(random.randrange(len(npc_list)))

        mutated["npc_vehicles"] = npc_list
        mutated["npc_vehicle_count"] = len(npc_list)
        return mutated

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        if random.random() < 0.5:
            child1["weather"], child2["weather"] = child2.get("weather"), child1.get("weather")

        npcs1 = child1.get("npc_vehicles", [])
        npcs2 = child2.get("npc_vehicles", [])
        if npcs1 and npcs2:
            cut1 = random.randint(0, len(npcs1))
            cut2 = random.randint(0, len(npcs2))
            child1["npc_vehicles"] = self._deduplicate_npcs(npcs1[:cut1] + npcs2[cut2:], child1)
            child2["npc_vehicles"] = self._deduplicate_npcs(npcs2[:cut2] + npcs1[cut1:], child2)
            child1["npc_vehicle_count"] = len(child1["npc_vehicles"])
            child2["npc_vehicle_count"] = len(child2["npc_vehicles"])
        return child1, child2

    def _deduplicate_npcs(self, npc_list: List[Dict], scenario: Dict) -> List[Dict]:
        used = {scenario["ego_car"]["spawn_point_index"]}
        deduped = []
        for npc in npc_list:
            if npc["spawn_point_index"] in used:
                continue
            deduped.append(copy.deepcopy(npc))
            used.add(npc["spawn_point_index"])
            if len(deduped) >= 8:
                break
        if not deduped:
            npc = self._random_npc(used, prefer_route=True)
            if npc is not None:
                deduped.append(npc)
        return deduped

    def _risk_from_log(self, detailed_log: List[Dict]) -> Tuple[float, int]:
        best_risk = float("-inf")
        failures = 0
        for node in detailed_log:
            result = node.get("execution_result") or {}
            semantically_correct = node.get("semantically_correct", True)
            if not semantically_correct:
                failures += 1

            if self.method == "av_fuzzer":
                risk = compute_av_fuzzer_risk(result)
            elif self.method == "drive_fuzz":
                risk = compute_drive_fuzz_risk(result)
            else:
                raise ValueError(f"Unsupported ADS baseline method: {self.method}")
            best_risk = max(best_risk, risk)
        return best_risk, failures

    def _evaluate(self, scenario: Dict) -> Dict:
        scenario_key = self._scenario_hash(scenario)
        if scenario_key in self.evaluation_cache:
            self.cache_hits += 1
            return copy.deepcopy(self.evaluation_cache[scenario_key])

        safety, task, detailed_log = self.fuzzer.evaluate(
            scenario,
            self.instruction,
            save_failures=True,
            failure_dir=self.failure_dir,
            return_detailed_log=True,
        )
        self.simulation_count += len(detailed_log)
        risk, failures = self._risk_from_log(detailed_log)
        self.total_failures += failures

        record = {
            "eval_id": len(self.evaluations),
            "risk": risk,
            "fitness": risk,
            "safety_score": safety,
            "task_score": task,
            "failures": failures,
            "simulation_count": len(detailed_log),
            "scenario": copy.deepcopy(scenario),
            "detailed_log": detailed_log,
        }
        self.evaluation_cache[scenario_key] = copy.deepcopy(record)
        self.evaluations.append(record)

        eval_dir = Path(self.log_dir) / f"eval_{record['eval_id']:05d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
        (eval_dir / "result.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return record

    def _tournament_select(self, population: List[Dict], k: int = 3) -> Dict:
        candidates = random.sample(population, min(k, len(population)))
        return max(candidates, key=lambda item: item["fitness"])

    def run_av_fuzzer(self) -> None:
        population = []
        for _ in range(self.pop_size):
            if self._simulation_budget_reached():
                break
            scenario = self._initial_scenario()
            result = self._evaluate(scenario)
            population.append({"scenario": scenario, "fitness": result["fitness"]})

        for _ in range(self.max_gens):
            if self._simulation_budget_reached() or not population:
                break
            population = sorted(population, key=lambda item: item["fitness"], reverse=True)
            next_population = [copy.deepcopy(population[0])]
            while len(next_population) < self.pop_size and not self._simulation_budget_reached():
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                child1, child2 = (
                    self._crossover(parent1["scenario"], parent2["scenario"])
                    if random.random() < self.cxpb
                    else (copy.deepcopy(parent1["scenario"]), copy.deepcopy(parent2["scenario"]))
                )
                for child in (child1, child2):
                    if random.random() < self.mutpb:
                        child = self._mutate_scenario(child, local=True)
                    result = self._evaluate(child)
                    next_population.append({"scenario": child, "fitness": result["fitness"]})
                    if len(next_population) >= self.pop_size or self._simulation_budget_reached():
                        break
            population = next_population

    def run_drive_fuzz(self) -> None:
        current = self._initial_scenario()
        current_result = self._evaluate(current)
        current_fitness = current_result["fitness"]

        for _ in range(self.max_cycles):
            if self._simulation_budget_reached():
                break
            candidates = []
            for _ in range(self.max_mutations):
                if self._simulation_budget_reached():
                    break
                candidate = self._mutate_scenario(current, local=True)
                result = self._evaluate(candidate)
                candidates.append({"scenario": candidate, "fitness": result["fitness"]})
            if not candidates:
                break
            successor = max(candidates, key=lambda item: item["fitness"])
            if successor["fitness"] >= current_fitness:
                current = successor["scenario"]
                current_fitness = successor["fitness"]
            else:
                current = self._mutate_scenario(current, local=False)
                current_result = self._evaluate(current)
                current_fitness = current_result["fitness"]

    def run(self) -> None:
        self.start_time = time.time()
        if self.method == "av_fuzzer":
            self.run_av_fuzzer()
        elif self.method == "drive_fuzz":
            self.run_drive_fuzz()
        self.save_metadata()

    def save_metadata(self) -> None:
        end_time = time.time()
        failure_stats = collect_failure_analysis(self.failure_dir)
        metadata = {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "actual_duration_seconds": round(end_time - self.start_time, 2),
            "total_simulations_executed": self.simulation_count,
            "total_failures_detected": self.total_failures,
            "failure_analysis": failure_stats,
            "configuration": {
                "method": self.method,
                "baseline_type": "reimplemented_ads_scenario_fuzzer",
                "static_scenario": self.static_scenario,
                "seed_scenario": self.seed_scenario,
                "manifest_id": self.manifest_id,
                "instruction_source": self.instruction_source,
                "instruction": self.instruction,
                "vla_model": self.model,
                "max_simulations": self.max_simulations,
                "random_seed": self.random_seed,
                "pop_size": self.pop_size,
                "max_gens": self.max_gens,
                "max_cycles": self.max_cycles,
                "max_mutations": self.max_mutations,
                "cxpb": self.cxpb,
                "mutpb": self.mutpb,
                "route_biased_spawn_count": len(self.route_biased_spawns),
                "evaluation_cache_size": len(self.evaluation_cache),
                "evaluation_cache_hits": self.cache_hits,
                "fixed_instruction_policy": "basic_instruction",
            },
        }
        metadata_path = os.path.join(self.base_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print(f"Saved ADS baseline metadata to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ADS scenario-fuzzing baselines in the VLAD-Fuzz framework.")
    parser.add_argument("--method", choices=ADS_BASELINE_METHODS, required=True)
    parser.add_argument("--model", choices=["lmdrive", "simlingo", "bevdriver"], default="simlingo")
    parser.add_argument("--manifest", help="JSONL experiment manifest produced by build_instruction_manifest.py")
    parser.add_argument("--manifest-id", help="Manifest row id")
    parser.add_argument("--instruction-source", choices=INSTRUCTION_SOURCES, default="basic")
    parser.add_argument("--instruction", help="Manual instruction override")
    parser.add_argument("--static-scenario", help="Static scenario JSON for direct input mode")
    parser.add_argument("--seed-scenario", help="Seed dynamic scenario JSON for direct input mode")
    parser.add_argument("--max-simulations", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--success-distance", type=float, default=5.0)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--output-root", default="results/rq2/runs")
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--max-gens", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--max-mutations", type=int, default=6)
    parser.add_argument("--cxpb", type=float, default=0.2)
    parser.add_argument("--mutpb", type=float, default=0.3)
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

    runner = ADSBaselineRunner(
        method=args.method,
        static_scenario=selection["static_scenario"],
        seed_scenario=selection["dynamic_scenario"],
        instruction=selection["instruction"],
        model=args.model,
        gpu_id=args.gpu_id,
        manifest_id=selection["manifest_id"],
        instruction_source=selection["instruction_source"],
        max_simulations=args.max_simulations,
        random_seed=args.random_seed,
        output_root=args.output_root,
        success_distance=args.success_distance,
        pop_size=args.pop_size,
        max_gens=args.max_gens,
        max_cycles=args.max_cycles,
        max_mutations=args.max_mutations,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
    )
    runner.run()


if __name__ == "__main__":
    main()

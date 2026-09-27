"""
Simplified Local Natural Language Instruction Fuzzing Module for VLA Autonomous Driving Systems

This module provides a simplified tree-based DFS approach for fuzzing natural language
instruction processing in autonomous driving systems.
"""

import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from scenario.carla_scenario import CarlaScenario
from vladfuzz_runtime.carla_connection import resolve_carla_connection
from vladfuzz_runtime.model_registry import bootstrap_model_environment
from vladfuzz_runtime.experiment_manifest import INSTRUCTION_SOURCES, apply_manifest_selection
from vladfuzz_runtime.mutation_operators import (
    MutationOperator, 
    MutationResult, 
    MutationOperatorEngine,
    get_all_operators,
    parse_operator_list,
)
from vladfuzz_runtime.infrastructure import CarlaInfrastructureError, is_carla_infrastructure_error
from vladfuzz_runtime.run_metadata import create_run_dir, write_metadata
from vladfuzz_runtime.failure_analysis import collect_failure_analysis
from vladfuzz_runtime.oracle import parse_oracle_checks
import json
import random
import copy
from typing import Callable, List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import os
import uuid
import time
from datetime import datetime

@dataclass
class FuzzingNode:
    """Node in the fuzzing tree representing a mutation state"""
    instruction: str
    parent: Optional['FuzzingNode']
    children: List['FuzzingNode'] = field(default_factory=list)
    applied_operator: Optional[MutationOperator] = None
    operator_params: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    mutation_metadata: Optional[Dict[str, Any]] = None
    depth: int = 0
    node_id: str = ""  # Will be set based on path
    mutation_success: bool = True
    error_message: Optional[str] = None
    used_operators: List[MutationOperator] = field(default_factory=list)
    # Speed control related fields
    has_speed_mutation_in_path: bool = False
    speed_reference_for_path: Optional[float] = None  # baseline speed for FASTER/SLOWER, target limit for ABSOLUTE_LIMIT
    speed_control_type_for_path: Optional[str] = None
    # Maintain distance related fields
    has_distance_mutation_in_path: bool = False
    target_spawn_index_for_path: Optional[int] = None # Changed from vehicle_id to spawn_index
    maintain_distance_for_path: Optional[float] = None
    
    # Failure reason fields. The semantic_* name is kept for backward
    # compatibility with older result files.
    semantic_failure_reason: Optional[str] = None
    failure_reason: Optional[str] = None
    
    def __post_init__(self):
        """Generate node_id based on path after initialization"""
        if self.parent is None:
            self.node_id = "Root"
        else:
            operator_name = self.applied_operator.value if self.applied_operator else "Unknown"
            self.node_id = f"{self.parent.node_id}-{operator_name}"

class LocalFuzzer:
    """
    Simplified local fuzzer for natural language driving instructions with tree-based DFS mutation.
    """
    
    def __init__(
        self,
        static_scenario,
        mutation_depth,
        success_distance=5.0,
        api_provider="deepseek",
        model_name: str = "deepseek-v4-flash",
        vla_model: str = "lmdrive",
        gpu_id=0,
        verbose: bool = True,
        semantic_pruning: bool = True,
        operators: Optional[List[MutationOperator]] = None,
        oracle_checks: Optional[Dict[str, bool]] = None,
        scenario_duration_frames: Optional[int] = 500,
    ):
        """
        Initialize the local fuzzer with CARLA scenario manager and mutation operator engine.
        
        Args:
            static_scenario: Path to static scenario file
            mutation_depth: Maximum depth of mutation tree
            success_distance: Distance threshold for success
            api_provider: The API provider to use ("gemini", "qwen", or "deepseek")
            model_name: Model name to use
            gpu_id: GPU ID to load the model on
            verbose: Whether to enable verbose output for the scenario manager
            semantic_pruning: Whether to stop expanding a branch after a failed node outcome
        """
        spec = bootstrap_model_environment(vla_model)
        carla_connection = resolve_carla_connection()
        self.vla_model = spec.name
        self.scenario_manager = CarlaScenario(
            host=carla_connection.host,
            port=carla_connection.rpc_port,
            tm_port=carla_connection.tm_port,
            tm_seed=0,
            config_path=str(spec.config_path),
            hydra_config_path=str(spec.hydra_config_path) if spec.hydra_config_path else None,
            model_name=spec.name,
            gpu_id=gpu_id,
            verbose=verbose,
            oracle_checks=oracle_checks,
            scenario_duration_frames=scenario_duration_frames,
        )
        self.mutation_depth = mutation_depth
        self.root_node = None
        self.nodes = []
        self.success_distance = success_distance 
        self.semantic_pruning = semantic_pruning
        self.operators = get_all_operators() if operators is None else list(operators)
        self.stop_requested: Optional[Callable[[], bool]] = None
        self.evaluation_stopped_by_budget = False
        
        if not self.scenario_manager.load_static_scenario(static_scenario):
            raise ValueError(f"Could not load static scenario {static_scenario}")
        
        # Global-only evaluation executes only the root instruction and must not
        # depend on an external language-model API.
        self.mutation_engine = None
        if self.mutation_depth > 0 and self.operators:
            self.mutation_engine = MutationOperatorEngine(
                api_provider=api_provider,
                model_name=model_name,
            )
            print("Initialized LocalFuzzer with mutation engine")
        else:
            print("Initialized LocalFuzzer without language mutation engine")

    def _budget_stop_requested(self) -> bool:
        if self.stop_requested is None:
            return False
        if self.stop_requested():
            self.evaluation_stopped_by_budget = True
            return True
        return False
    
    def _execute_scenario(self, node: FuzzingNode) -> Dict[str, Any]:
        """
        Execute the scenario with the given instruction and return results.
        
        Args:
            node: The fuzzing node containing the instruction
            
        Returns:
            Dictionary containing execution results
        """
        try:
            print(f"Executing scenario for node {node.node_id} with instruction: '{node.instruction}'")
            result = self.scenario_manager.start_scenario(node.instruction, self.success_distance)
            print(f"Scenario execution completed for node {node.node_id}")
            return result
        except Exception as e:
            if is_carla_infrastructure_error(e):
                raise CarlaInfrastructureError(str(e)) from e
            print(f"Scenario execution failed for node {node.node_id}: {str(e)}")
            return {
                'error': True,
                'collision': False,
                'completed': False,
                'actual_frames_executed': 0,
                'error_message': str(e)
            }

    @staticmethod
    def _normalize_failure_reason(reason: Optional[str]) -> Optional[str]:
        if not reason:
            return reason
        lowered = reason.lower()
        if lowered.startswith("execution_error"):
            return reason
        if "max speed" in lowered and "> limit" in lowered:
            return "speed_limit_exceeded"
        if "avg speed" in lowered and "<= baseline" in lowered:
            return "speed_too_slow"
        if "avg speed" in lowered and ">= baseline" in lowered:
            return "speed_too_fast"
        if "min distance" in lowered and "< target" in lowered:
            return "maintain_distance_failed"
        if "target vehicle" in lowered and "not found" in lowered:
            return "target_vehicle_missing"
        if "missing speed" in lowered or "speed data missing" in lowered:
            return "semantic_constraint_violation"
        if "missing distance" in lowered or "min distance data missing" in lowered:
            return "semantic_constraint_violation"
        if "no execution result" in lowered:
            return "unknown_error"
        if "oracle failure" in lowered:
            return "unknown_error"
        return reason

    def _evaluate_node_validity(self, node: FuzzingNode) -> Tuple[bool, Optional[str]]:
        """
        Check whether the current node remains valid for further fuzzing.
        
        Args:
            node: The fuzzing node to check
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, failure_reason)
        """
        # First check if task is completed
        if not node.execution_result:
             return False, "unknown_error"

        if node.execution_result.get('error', False):
            error_message = node.execution_result.get('error_message')
            if error_message:
                return False, f"execution_error: {error_message}"
            return False, "execution_error"
             
        if node.execution_result.get('collision', False):
            return False, "collision"

        if node.execution_result.get('oracle_failure', False):
            return False, node.execution_result.get('oracle_failure_reason', "unknown_error")
            
        if not node.execution_result.get('completed', False):
            return False, "target_not_reached"
        
        # Check speed mutation conditions if exists
        if node.has_speed_mutation_in_path:
            speed_tolerence = 0.1
            speed_reference = node.speed_reference_for_path
            speed_type = node.speed_control_type_for_path
            
            if speed_reference is None or speed_type is None:
                return False, "Missing speed reference or type"
            
            ego_stats = node.execution_result.get('ego_speed_stats', {})
            current_speed = ego_stats.get('average_speed')
            max_speed = ego_stats.get('max_speed')
            
            if speed_type == "FASTER":
                if current_speed is None: return False, "semantic_constraint_violation"
                print(f"Speed check - Current: {current_speed:.2f}, Baseline: {speed_reference:.2f}, Type: {speed_type}")
                if current_speed <= speed_reference:
                    return False, f"Avg speed {current_speed:.2f} <= Baseline {speed_reference:.2f} (Faster required)"
            
            elif speed_type == "SLOWER":
                if current_speed is None: return False, "semantic_constraint_violation"
                print(f"Speed check - Current: {current_speed:.2f}, Baseline: {speed_reference:.2f}, Type: {speed_type}")
                if current_speed >= speed_reference:
                    return False, f"Avg speed {current_speed:.2f} >= Baseline {speed_reference:.2f} (Slower required)"
            
            elif speed_type == "ABSOLUTE_LIMIT":
                if max_speed is None: return False, "semantic_constraint_violation"
                print(f"Speed limit check - Max: {max_speed:.2f}, Limit: {speed_reference}")
                if max_speed > speed_reference * (1.0 + speed_tolerence):
                    return False, f"Max speed {max_speed:.2f} > Limit {speed_reference:.2f} * {1.0 + speed_tolerence}"
        
        # Check distance maintenance conditions if exists
        if node.has_distance_mutation_in_path:
            target_spawn_index = node.target_spawn_index_for_path
            maintain_distance = node.maintain_distance_for_path
            
            if target_spawn_index is None or maintain_distance is None:
                return False, "semantic_constraint_violation"
            
            # Check if the target vehicle is still in NPC vehicles and get its minimum distance
            npc_vehicles = node.execution_result.get('npc_vehicles', {})
            target_key = str(target_spawn_index)
            
            if target_key not in npc_vehicles:
                print(f"Distance check - Target vehicle with spawn index {target_spawn_index} not found in execution results")
                return False, f"Target vehicle at spawn {target_spawn_index} not found"
            
            target_vehicle_data = npc_vehicles[target_key]
            min_distance = target_vehicle_data.get('min_distance')
            if min_distance is None:
                return False, "semantic_constraint_violation"
            
            # Define tolerance for distance maintenance (±2m tolerance)
            distance_tolerance = 1.0  # meters - configurable parameter
            distance_lower_bound = maintain_distance - distance_tolerance
            
            print(f"Distance check - Target: {maintain_distance:.2f}m, Actual min: {min_distance:.2f}m, "
                  f"Tolerance: {distance_lower_bound:.2f}m")
            
            # Check if minimum distance is within acceptable range
            if not (distance_lower_bound <= min_distance):
                return False, f"Min distance {min_distance:.2f}m < Target {maintain_distance:.2f}m (Tolerance {distance_lower_bound:.2f}m)"
        
        return True, None

    def _dfs(self, current_node: FuzzingNode):
        """
        Perform depth-first search to build the mutation tree and execute scenarios.
        
        Args:
            current_node: Current node in the DFS
        """
        # Execute scenario for current node
        current_node.execution_result = self._execute_scenario(current_node)
        
        # Stability: Sleep briefly to let CARLA/GPU settle after execution and cleanup
        # This helps prevent segmentation faults caused by rapid context switching
        time.sleep(1.0)
        
        # Check whether the current node remains valid for further fuzzing.
        is_valid, raw_failure_reason = self._evaluate_node_validity(current_node)
        failure_reason = self._normalize_failure_reason(raw_failure_reason)
        current_node.failure_reason = failure_reason
        current_node.semantic_failure_reason = failure_reason
        node_outcome = "passed" if is_valid else "failed"
        print(f"Node {current_node.node_id} outcome: {node_outcome} ({failure_reason})")
        
        # Save failure case if the node failed but executed without system error.
        if not is_valid:
            if current_node.execution_result and not current_node.execution_result.get('error', False):
                if hasattr(self, 'save_failures') and self.save_failures:
                    self._save_failed_case(current_node)

        # Add to nodes list for later processing
        self.nodes.append({
            'node_id': current_node.node_id,
            'instruction': current_node.instruction,
            'applied_operator': current_node.applied_operator.value if current_node.applied_operator else None,
            'execution_result': current_node.execution_result,
            'mutation_metadata': current_node.mutation_metadata,
            'depth': current_node.depth,
            'node_outcome': node_outcome,
            'valid': is_valid,
            'failure_reason': failure_reason,
            'pruned': self.semantic_pruning and not is_valid,
            'semantically_correct': is_valid,
            'semantic_failure_reason': failure_reason,
            'semantic_pruned': self.semantic_pruning and not is_valid
        })

        if self._budget_stop_requested():
            print(f"Stopping mutation expansion after node {current_node.node_id} because budget is exhausted.")
            return

        if self.semantic_pruning and not is_valid:
            print(f"Pruning branch at node {current_node.node_id} due to failure: {failure_reason}")
            return
        
        # If we've reached maximum depth, stop recursion
        if current_node.depth >= self.mutation_depth:
            return
        
        # Get available operators (all operators minus those already used in the path from root)
        available_operators = [op for op in self.operators if op not in current_node.used_operators]
        
        if not available_operators:
            print(f"No more operators available for node {current_node.node_id} at depth {current_node.depth}")
            return
        
        # Apply each available operator to create child nodes
        for operator in available_operators:
            if self._budget_stop_requested():
                print(f"Stopping before applying more mutation operators at node {current_node.node_id} because budget is exhausted.")
                return

            print(f"Applying operator {operator.value} to node {current_node.node_id}")
            
            # For speed control, get baseline speed from current node's execution result
            avg_speed_ref = None
            max_speed_ref = None
            if (operator == MutationOperator.SPEED_CONTROL_INSERTION and 
                current_node.execution_result and 
                current_node.execution_result.get('ego_speed_stats')):
                avg_speed_ref = current_node.execution_result['ego_speed_stats'].get('average_speed')
                max_speed_ref = current_node.execution_result['ego_speed_stats'].get('max_speed')
                print(f"Using baseline average speed {avg_speed_ref:.2f} km/h and max speed {max_speed_ref:.2f} km/h for speed control mutation")
            elif operator == MutationOperator.SPEED_CONTROL_INSERTION:
                print("Speed statistics not available for speed control mutation, skipping")
                continue
            
            # For maintain distance, check if suitable vehicles are available
            target_vehicle = None
            maintain_distance = None
            if (operator == MutationOperator.MAINTAIN_DISTANCE and 
                current_node.execution_result and 
                current_node.execution_result.get('npc_vehicles')):
                
                # Check for vehicles within 20m distance and 60 degrees yaw difference
                distance_threshold = 20.0  # meters
                yaw_threshold = 60.0  # degrees
                suitable_vehicles = []
                
                for vehicle_id, vehicle_data in current_node.execution_result['npc_vehicles'].items():
                    initial_distance = vehicle_data.get('initial_distance', float('inf'))
                    initial_yaw_diff = vehicle_data.get('initial_yaw_diff', float('inf'))
                    
                    # Only consider vehicles within distance and yaw angle thresholds
                    if initial_distance < distance_threshold and initial_yaw_diff < yaw_threshold:
                        vehicle_info = {
                            'id': vehicle_id,
                            'name': vehicle_data.get('name', 'vehicle'),
                            'color': vehicle_data.get('color'),
                            'initial_distance': initial_distance,
                            'initial_yaw_diff': initial_yaw_diff
                        }
                        suitable_vehicles.append(vehicle_info)
                
                if suitable_vehicles:
                    # Randomly select one suitable vehicle
                    target_vehicle = random.choice(suitable_vehicles)
                    # Generate random maintain distance (5-15m）
                    maintain_distance = random.randint(5, 15)
                    maintain_distance = min(maintain_distance, int(target_vehicle['initial_distance'])) # Ensure it's less than initial distance
                    print(f"Selected vehicle {target_vehicle['id']} at {target_vehicle['initial_distance']:.2f}m, "
                          f"yaw diff: {target_vehicle['initial_yaw_diff']:.1f}°, target distance: {maintain_distance:.2f}m")
                else:
                    print("No suitable vehicles within distance and yaw angle thresholds for maintain distance mutation, skipping")
                    continue
            elif operator == MutationOperator.MAINTAIN_DISTANCE:
                print("No NPC vehicles available or within range for maintain distance mutation, skipping")
                continue
            
            # Apply mutation using the mutation engine
            if operator == MutationOperator.SPEED_CONTROL_INSERTION:
                mutation_result = self.mutation_engine.insert_speed_control(
                    current_node.instruction, 
                    baseline_avg_speed=avg_speed_ref,
                    baseline_max_speed=max_speed_ref
                )
            elif operator == MutationOperator.MAINTAIN_DISTANCE:
                mutation_result = self.mutation_engine.insert_maintain_distance(
                    current_node.instruction,
                    target_vehicle=target_vehicle,
                    maintain_distance=maintain_distance
                )
            else:
                mutation_result = self.mutation_engine.apply_mutation(current_node.instruction, operator)
            
            # Determine speed control path information for child node
            child_has_speed_mutation = current_node.has_speed_mutation_in_path
            child_speed_reference = current_node.speed_reference_for_path
            child_speed_type = current_node.speed_control_type_for_path
            
            # Determine distance control path information for child node
            child_has_distance_mutation = current_node.has_distance_mutation_in_path
            child_target_spawn_index = current_node.target_spawn_index_for_path
            child_maintain_distance = current_node.maintain_distance_for_path
            
            # If this is a speed control insertion, establish the reference for the path
            if operator == MutationOperator.SPEED_CONTROL_INSERTION:
                child_has_speed_mutation = True
                child_speed_type = mutation_result.metadata.get('speed_control_type') if mutation_result.metadata else None
                
                if child_speed_type == "ABSOLUTE_LIMIT":
                    # For ABSOLUTE_LIMIT, use target_speed_limit as reference
                    child_speed_reference = mutation_result.metadata.get('target_speed_limit') if mutation_result.metadata else None
                    print(f"Establishing speed reference {child_speed_reference} km/h for path with type {child_speed_type}")
                else:
                    # For FASTER/SLOWER, use baseline speed as reference (which was avg_speed_ref)
                    child_speed_reference = avg_speed_ref  
                    print(f"Establishing speed reference {child_speed_reference:.2f} km/h for path with type {child_speed_type}")
            
            # If this is a maintain distance insertion, establish the reference for the path
            if operator == MutationOperator.MAINTAIN_DISTANCE:
                child_has_distance_mutation = True
                child_target_spawn_index = int(mutation_result.metadata.get('target_vehicle_id')) if mutation_result.metadata and mutation_result.metadata.get('target_vehicle_id') is not None else None
                child_maintain_distance = mutation_result.metadata.get('maintain_distance') if mutation_result.metadata else None
                print(f"Establishing distance reference {child_maintain_distance:.2f}m for vehicle {child_target_spawn_index}")
            
            # Create child node with mutation metadata
            child_node = FuzzingNode(
                instruction=mutation_result.mutated_text,
                parent=current_node,
                applied_operator=operator,
                operator_params={
                    "original_instruction": current_node.instruction, 
                    "target_vehicle": target_vehicle,
                    "maintain_distance": maintain_distance
                },
                mutation_metadata=mutation_result.metadata,
                depth=current_node.depth + 1,
                mutation_success=mutation_result.success,
                error_message=mutation_result.error_message if not mutation_result.success else None,
                used_operators=current_node.used_operators + [operator],  # Add operator to used list
                # Speed control path tracking
                has_speed_mutation_in_path=child_has_speed_mutation,
                speed_reference_for_path=child_speed_reference,
                speed_control_type_for_path=child_speed_type,
                # Distance control path tracking
                has_distance_mutation_in_path=child_has_distance_mutation,
                target_spawn_index_for_path=child_target_spawn_index,
                maintain_distance_for_path=child_maintain_distance
            )
            
            # Add child to current node
            current_node.children.append(child_node)
            
            if mutation_result.success:
                print(f"Successfully created child node {child_node.node_id} with operator {operator.value}")
                # Log speed control specific information
                if operator == MutationOperator.SPEED_CONTROL_INSERTION and mutation_result.metadata:
                    speed_type = mutation_result.metadata.get('speed_control_type')
                    target_speed = mutation_result.metadata.get('target_speed_limit')
                    if target_speed:
                        print(f"Speed control type: {speed_type}, target: {target_speed} km/h")
                    else:
                        print(f"Speed control type: {speed_type}")
                
                # Log maintain distance specific information
                if operator == MutationOperator.MAINTAIN_DISTANCE and mutation_result.metadata:
                    target_vehicle_id = mutation_result.metadata.get('target_vehicle_id')
                    target_distance = mutation_result.metadata.get('maintain_distance')
                    target_vehicle_name = mutation_result.metadata.get('target_vehicle_name')
                    target_vehicle_color = mutation_result.metadata.get('target_vehicle_color')
                    vehicle_desc = f"{target_vehicle_color} {target_vehicle_name}" if target_vehicle_color else target_vehicle_name
                    print(f"Maintain distance: {target_distance:.2f}m from {vehicle_desc} (ID: {target_vehicle_id})")
                
                # Recursively explore this branch
                self._dfs(child_node)
                if self.evaluation_stopped_by_budget:
                    return
            else:
                print(f"Mutation failed for operator {operator.value}: {mutation_result.error_message}")

    def _calculate_fitness_metrics(self) -> Tuple[float, float]:
        """
        Calculate fitness metrics from all nodes in the fuzzing tree.
        
        Returns:
            Tuple[float, float]: (safety_score, task_score)
                - safety_score: worst-case per-execution average ETTC across all nodes
                - task_score: average path tracking quality over the full executed mutation tree
        """
        all_ettc_values = []
        tracking_quality_scores = []
        executed_node_count = 0
        
        for node_data in self.nodes:
            execution_result = node_data.get('execution_result', {})
            executed_node_count += 1
            
            # Use the per-execution average ETTC to avoid overreacting to
            # single-frame TTC spikes, then aggregate across instruction
            # variants with a worst-case minimum.
            ettc_stats = execution_result.get('ettc_stats')
            
            if ettc_stats and 'avg_ettc' in ettc_stats:
                ettc_value = ettc_stats['avg_ettc']
                all_ettc_values.append(ettc_value)
            else:
                # If statistics are missing (e.g. early crash/error), assume safe (max ETTC) or max danger (min ETTC)
                if execution_result.get('collision', False):
                    all_ettc_values.append(0.0) # Collision is max danger (min ETTC)
                else:
                    # For other errors/early exits, if no ETTC measured, assume max safe ETTC
                    # This value comes from CarlaScenario.ettc_metrics.max_ttc
                    # Access via scenario_manager as it holds the ettc_metrics instance
                    if hasattr(self.scenario_manager, 'ettc_metrics') and hasattr(self.scenario_manager.ettc_metrics, 'max_ttc'):
                        all_ettc_values.append(self.scenario_manager.ettc_metrics.max_ttc)
                    else:
                        all_ettc_values.append(15.0) # Fallback if ettc_metrics not properly initialized
                # Do not raise ValueError, just log warning if needed
                print("Warning: Missing ETTC stats in execution result, using default.")
            
            # Lower path_tracking_quality means worse task following. We follow
            # the original local-fuzz objective: valid nodes contribute their
            # trajectory quality to the numerator, while all executed nodes
            # remain in the denominator. This keeps the metric smooth but still
            # penalizes failed language variants.
            if node_data.get('semantically_correct', False):
                path_deviation_stats = execution_result.get('path_deviation_stats', {})
                path_tracking_quality = path_deviation_stats.get('path_tracking_quality')
                if path_tracking_quality is not None:
                    tracking_quality_scores.append(path_tracking_quality)
                else:
                    raise ValueError("Missing path tracking quality or completion ratio in deviation stats")
        
        if all_ettc_values:
            safety_score = min(all_ettc_values)
        else:
            raise ValueError("No ETTC values found from executed nodes")
        
        if tracking_quality_scores:
            task_score = sum(tracking_quality_scores) / executed_node_count
        else:
            task_score = 0.0
        
        return safety_score, task_score
    
    def evaluate(
        self,
        dynamic_scenario: Union[str, Dict],
        language_instruction: str,
        save_failures: bool = True,
        failure_dir: str = "failed_cases",
        return_detailed_log: bool = False,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> Union[Tuple[float, float], Tuple[float, float, List[Dict[str, Any]]]]:
        """
        Evaluate the language instruction by building a complete mutation tree and calculating fitness metrics.
        
        Args:
            dynamic_scenario: Path to dynamic scenario file OR Dictionary containing scenario data
            language_instruction: Original language instruction
            save_failures: Whether to save semantically incorrect cases to disk
            failure_dir: Directory to save failure cases
            return_detailed_log: If True, return list of all node execution details along with scores
            stop_requested: Optional callback checked after each executed node to
                stop expanding the mutation tree under a wall-clock budget.
            
        Returns:
            If return_detailed_log is False:
                Tuple[float, float]: (safety_score, task_score)
            If return_detailed_log is True:
                Tuple[float, float, List[Dict]]: (safety_score, task_score, detailed_results)
        """
        # Store scenario data and configuration for failure logging
        self.save_failures = save_failures
        self.failure_dir = failure_dir
        self.stop_requested = stop_requested
        self.evaluation_stopped_by_budget = False
        
        if isinstance(dynamic_scenario, str):
            try:
                with open(dynamic_scenario, 'r') as f:
                    self.current_scenario_data = json.load(f)
            except Exception as e:
                 raise ValueError(f"Failed to load scenario file for logging: {e}")
        else: # dynamic_scenario is a dict
            # Make a deep copy to ensure no internal __evaluation_log__ from previous runs persists
            self.current_scenario_data = copy.deepcopy(dynamic_scenario)
            # Remove any existing __evaluation_log__ that might be present from a previous evaluation of this exact individual object
            if '__evaluation_log__' in self.current_scenario_data:
                del self.current_scenario_data['__evaluation_log__']

        if save_failures:
            os.makedirs(failure_dir, exist_ok=True)

        if not self.scenario_manager.load_scenario(dynamic_scenario):
            scenario_info = dynamic_scenario if isinstance(dynamic_scenario, str) else "dictionary data"
            raise ValueError(f"Error: Failed to load dynamic scenario from '{scenario_info}'")
        
        print(f"Starting fuzzing evaluation with original instruction: '{language_instruction}'")
        
        # Reset nodes list
        self.nodes = []
        
        # Create root node
        self.root_node = FuzzingNode(
            instruction=language_instruction,
            parent=None,
            depth=0,
            used_operators=[]  # Root starts with no used operators
        )
        
        # Start DFS from root
        self._dfs(self.root_node)
        
        print(f"Fuzzing evaluation completed. Generated {len(self.nodes)} nodes in tree.")
        
        # Calculate fitness metrics
        safety_score, task_score = self._calculate_fitness_metrics()
        
        print(f"Fitness metrics calculated:")
        print(f"  - Safety Score: {safety_score:.3f} seconds")
        print(f"  - Task Score: {task_score:.3f}")
        
        if return_detailed_log:
            return safety_score, task_score, self.nodes
        else:
            return safety_score, task_score

    def _save_failed_case(self, node: FuzzingNode):
        """Save semantically incorrect case to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            
            # Create a dedicated directory for this failure case
            case_dir_name = f"failure_{timestamp}_{unique_id}"
            case_dir_path = os.path.join(self.failure_dir, case_dir_name)
            os.makedirs(case_dir_path, exist_ok=True)
            
            # 1. Save the scenario configuration (for reproduction)
            scenario_config_path = os.path.join(case_dir_path, "scenario_config.json")
            
            # Helper to handle non-serializable types
            def json_serializer(obj):
                if isinstance(obj, (datetime,)):
                    return obj.isoformat()
                return str(obj)

            with open(scenario_config_path, 'w') as f:
                json.dump(self.current_scenario_data, f, indent=4, default=json_serializer)
            
            # 2. Save the failure details and execution result
            failure_data = {
                "timestamp": timestamp,
                "mutated_instruction": node.instruction,
                "applied_operator": node.applied_operator.value if node.applied_operator else "ROOT",
                "operator_params": node.operator_params,
                "execution_result": node.execution_result,
                "node_depth": node.depth,
                "node_outcome": "failed",
                "valid": False,
                "failure_reason": node.failure_reason,
                "pruned": self.semantic_pruning,
                "semantic_check_failure": True,
                "semantic_failure_reason": node.semantic_failure_reason
            }
            
            result_path = os.path.join(case_dir_path, "result.json")
            with open(result_path, 'w') as f:
                json.dump(failure_data, f, indent=4, default=json_serializer)
                
            print(f"⚠️ Saved failure case to directory: {case_dir_path}")
            
        except Exception as e:
            print(f"❌ Failed to save failure case: {e}")

    def cleanup(self):
        """Clean up resources."""
        if self.scenario_manager:
            self.scenario_manager.destroy()
        print("LocalFuzzer cleanup completed")

def main():
    parser = argparse.ArgumentParser(description="Run local instruction fuzzing against a selected VLA model.")
    parser.add_argument("--model", choices=["lmdrive", "simlingo", "bevdriver"], default="lmdrive")
    parser.add_argument("--static-scenario", help="Static scenario JSON for direct input mode")
    parser.add_argument("--dynamic-scenario", help="Dynamic scenario JSON for direct input mode")
    parser.add_argument("--instruction", help="Navigation instruction for direct input mode")
    parser.add_argument("--manifest", help="Optional JSONL experiment manifest produced by build_instruction_manifest.py")
    parser.add_argument("--manifest-id", help="Manifest row id, for example town01_t_junction_1/seed_1")
    parser.add_argument("--instruction-source", choices=INSTRUCTION_SOURCES, default="manual")
    parser.add_argument("--mutation-depth", type=int, default=2)
    parser.add_argument("--success-distance", type=float, default=5.0)
    parser.add_argument("--llm-provider", default="deepseek", choices=["gemini", "qwen", "deepseek"])
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable-semantic-pruning", action="store_true", help="Continue mutating branches after failed node outcomes.")
    parser.add_argument("--operators", default="all", help="Comma-separated mutation operators, or 'all'.")
    parser.add_argument("--oracle-checks", default="default", help="Comma-separated oracle checks: collision,stuck,lane,speeding,timeout,out_of_bounds,other; or all/default/none.")
    parser.add_argument("--random-seed", type=int, help="Random seed for local mutation choices.")
    parser.add_argument("--method", default="language_only", help="Experiment method label stored in metadata.")
    parser.add_argument("--output-root", default="results/rq2/runs", help="Root directory for standalone run metadata.")
    parser.add_argument("--output-dir", help="Exact output directory for standalone run metadata.")
    args = parser.parse_args()

    fuzzer = None
    try:
        if args.random_seed is not None:
            random.seed(args.random_seed)

        selection = apply_manifest_selection(
            manifest_path=args.manifest,
            manifest_id=args.manifest_id,
            instruction_source=args.instruction_source,
            static_scenario=args.static_scenario,
            dynamic_scenario=args.dynamic_scenario,
            instruction=args.instruction,
        )

        print("Experiment input:")
        print(f"  Manifest ID: {selection['manifest_id']}")
        print(f"  Static scenario: {selection['static_scenario']}")
        print(f"  Dynamic scenario: {selection['dynamic_scenario']}")
        print(f"  Instruction source: {selection['instruction_source']}")
        print(f"  Instruction: {selection['instruction']}")
        operators = parse_operator_list(args.operators)
        oracle_checks = parse_oracle_checks(args.oracle_checks)
        operator_names = [operator.value for operator in operators]
        print(f"  Operators: {', '.join(operator_names)}")
        print(f"  Oracle checks: {', '.join(name for name, enabled in oracle_checks.items() if enabled) or 'none'}")

        fuzzer = LocalFuzzer(
            selection["static_scenario"],
            mutation_depth=args.mutation_depth,
            success_distance=args.success_distance,
            api_provider=args.llm_provider,
            model_name=args.llm_model,
            vla_model=args.model,
            gpu_id=args.gpu_id,
            verbose=args.verbose,
            semantic_pruning=not args.disable_semantic_pruning,
            operators=operators,
            oracle_checks=oracle_checks,
        )

        output_dir = args.output_dir or create_run_dir(
            args.output_root,
            args.method,
            args.model,
            selection["manifest_id"],
            args.random_seed,
        )
        failure_dir = os.path.join(output_dir, "failures")

        start_time = time.time()
        safety_score, task_score, detailed_log = fuzzer.evaluate(
            selection["dynamic_scenario"],
            selection["instruction"],
            save_failures=True,
            failure_dir=failure_dir,
            return_detailed_log=True,
        )
        end_time = time.time()

        metadata = {
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "total_duration_seconds": round(end_time - start_time, 2),
            "total_simulations_executed": len(detailed_log),
            "total_failures_detected": sum(1 for node in detailed_log if not node.get("valid", node.get("semantically_correct", True))),
            "failure_analysis": collect_failure_analysis(failure_dir),
            "fitness": {
                "safety_score": safety_score,
                "task_score": task_score,
            },
            "configuration": {
                "method": args.method,
                "static_scenario": selection["static_scenario"],
                "seed_scenario": selection["dynamic_scenario"],
                "vla_model": args.model,
                "manifest_id": selection["manifest_id"],
                "instruction_source": selection["instruction_source"],
                "instruction": selection["instruction"],
                "mutation_depth": args.mutation_depth,
                "operators": operator_names,
                "oracle_checks": oracle_checks,
                "semantic_pruning": not args.disable_semantic_pruning,
                "random_seed": args.random_seed,
                "success_distance": args.success_distance,
                "llm_provider": args.llm_provider,
                "llm_model": args.llm_model,
            },
        }
        write_metadata(output_dir, metadata)
        with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as file:
            json.dump(detailed_log, file, indent=4, ensure_ascii=False, default=str)

        print("\nFinal Fitness Metrics:")
        print(f"  - Safety Score (minimize for higher danger): {safety_score:.3f}")
        print(f"  - Task Score (minimize for worse performance): {task_score:.3f}")
        print(f"  - Output Dir: {output_dir}")

        print("\nDetailed Node Results:")
        for i, node_data in enumerate(fuzzer.nodes):
            print(f"Node {i+1}: {node_data['node_id']}")
            print(f"  Instruction: {node_data['instruction']}")
            print(f"  Operator: {node_data['applied_operator']}")
            print(f"  Depth: {node_data['depth']}")
            print(f"  Outcome: {node_data.get('node_outcome', 'passed' if node_data.get('semantically_correct') else 'failed')}")
            if node_data.get("failure_reason"):
                print(f"  Failure Reason: {node_data['failure_reason']}")
            if node_data['execution_result']:
                result = node_data['execution_result']
                print(f"  Completed: {result.get('completed', False)}")
                print(f"  Collision: {result.get('collision', False)}")
                if result.get('ego_speed_stats'):
                    print(f"  Avg Speed: {result['ego_speed_stats'].get('average_speed', 'N/A')} km/h")
                if result.get('ettc_stats'):
                    print(f"  Min ETTC: {result['ettc_stats'].get('min_ettc', 'N/A')}s")
                if result.get('path_deviation_stats'):
                    print(f"  Path Tracking Quality: {result['path_deviation_stats'].get('path_tracking_quality', 'N/A')}")
                    print(f"  Path Completion Ratio: {result['path_deviation_stats'].get('path_completion_ratio', 'N/A')}")
            print()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to set the GEMINI_API_KEY or DASHSCOPE_API_KEY environment variable and verify scenario files exist")
    finally:
        if fuzzer:
            fuzzer.cleanup()


if __name__ == "__main__":
    main()

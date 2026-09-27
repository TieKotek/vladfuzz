"""
Mutation Operators Module for Natural Language Instruction Fuzzing

This module contains all mutation operators used for generating semantically-preserved
variants of natural language driving instructions for testing VLA autonomous driving systems.
"""

import random
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import os
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class MutationOperator(Enum):
    """Enumeration of available mutation operators"""
    IRRELEVANT_CONVERSATION = "irrelevant_conversation"
    PARAPHRASING = "paraphrasing"
    SPEED_CONTROL_INSERTION = "speed_control_insertion"
    MAINTAIN_DISTANCE = "maintain_distance"

@dataclass
class MutationResult:
    """Result of a single mutation operation"""
    original_text: str
    mutated_text: str
    operator: MutationOperator
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MutationOperatorEngine:
    """
    Engine for applying mutation operators to natural language instructions.
    
    This class handles all mutation operations including API calls to Gemini, Qwen, or DeepSeek
    and provides a unified interface for applying different types of mutations.
    """
    
    def __init__(self, model_name: Optional[str] = None, 
                 speed_limit_range: tuple = (0.8, 1.0), api_provider: str = "deepseek"):
        """
        Initialize the mutation operator engine.
        
        Args:
            model_name: Model name to use. Defaults: 'gemini-2.5-flash-lite' for gemini,
                'qwen3-max' for qwen, and 'deepseek-v4-flash' for deepseek
            speed_limit_range: Tuple of (min, max) speed mutation percentages for absolute speed limits
            api_provider: The API provider to use ("gemini", "qwen", or "deepseek")
        """
        self.api_provider = api_provider.lower()
        self.speed_limit_range = speed_limit_range
        from openai import OpenAI
        
        if self.api_provider == "gemini":
            self.model_name = model_name or "gemini-2.5-flash-lite"
            self.api_key = os.getenv('GEMINI_API_KEY')
            if not self.api_key:
                raise ValueError("Google API key must be provided either as parameter or GEMINI_API_KEY environment variable")
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            
        elif self.api_provider == "qwen":
            self.model_name = model_name or "qwen3-max"
            self.api_key = os.getenv('DASHSCOPE_API_KEY')
            if not self.api_key:
                raise ValueError("Dashscope API key must be provided either as parameter or DASHSCOPE_API_KEY environment variable")
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        elif self.api_provider == "deepseek":
            self.model_name = model_name or "deepseek-v4-flash"
            self.api_key = os.getenv('DEEPSEEK_API_KEY')
            if not self.api_key:
                raise ValueError("DeepSeek API key must be provided as DEEPSEEK_API_KEY environment variable")

            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")
        
        logger.info(f"Initialized MutationOperatorEngine with provider: {self.api_provider}, model: {self.model_name}")
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Make a call to the LLM API with error handling and retries.
        """
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                request_kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    "stream": False,
                }
                if self.api_provider == "deepseek":
                    # DeepSeek V4 defaults to thinking mode; disable it for deterministic
                    # instruction mutation and to avoid reasoning_content handling.
                    request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

                completion = self.client.chat.completions.create(**request_kwargs)
                return completion.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"{self.api_provider.capitalize()} API call failed after {max_retries} attempts: {str(e)}")
                    raise Exception(f"{self.api_provider.capitalize()} API call failed: {e}")
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"{self.api_provider.capitalize()} API call failed (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s... Error: {e}")
                time.sleep(delay)

    def _call_llm(self, prompt: str) -> str:
        """
        Dispatch call to the appropriate LLM provider.
        """
        return self._call_llm_api(prompt)

    def insert_irrelevant_conversation(self, instruction: str) -> MutationResult:
        """
        Insert irrelevant daily conversation before/after the driving instruction.
        Tests the model's ability to extract meaningful information from noisy input.
        
        Args:
            instruction: Original driving instruction
            
        Returns:
            MutationResult containing the mutated instruction
        """
        prompt = f"""
You are tasked with inserting irrelevant daily conversation around a driving instruction to test an autonomous driving system's ability to extract meaningful information.

Original driving instruction: "{instruction}"

Insert casual, unrelated conversation before or after the driving instruction. The conversation should be natural but completely unrelated to driving.

Requirements:
- Keep the original driving instruction exactly the same
- Add 1-2 sentences of irrelevant conversation before or after
- Make the conversation sound natural and human-like
- Ensure the driving instruction is still clearly identifiable

Return only the modified instruction with irrelevant conversation added. Do not include any explanations or additional text.

Example format: "How was your meeting today? Turn left at the next intersection."
"""
        
        try:
            mutated_text = self._call_llm(prompt)
            return MutationResult(
                original_text=instruction,
                mutated_text=mutated_text,
                operator=MutationOperator.IRRELEVANT_CONVERSATION,
                success=True
            )
        except Exception as e:
            return MutationResult(
                original_text=instruction,
                mutated_text=instruction,
                operator=MutationOperator.IRRELEVANT_CONVERSATION,
                success=False,
                error_message=str(e)
            )

    def paraphrase_instruction(self, instruction: str) -> MutationResult:
        """
        Paraphrase the instruction while preserving semantics.
        
        Args:
            instruction: Original instruction
            
        Returns:
            MutationResult containing the paraphrased instruction
        """
        prompt = f"""
You are tasked with paraphrasing the following text while preserving the exact same semantic meaning.

Original text: "{instruction}"

Paraphrase the text using different wording while maintaining identical semantic meaning. All information and details must be preserved without any loss or addition.

Critical requirements:
- Preserve ALL semantic information exactly - no loss, no addition, no modification
- Maintain the same level of detail and specificity
- Keep the same logical structure and meaning
- Only change the wording and phrasing
- The paraphrased text should be similar in length to the original

Return only the paraphrased text. Do not include explanations or additional text.
"""
        
        try:
            mutated_text = self._call_llm(prompt)
            return MutationResult(
                original_text=instruction,
                mutated_text=mutated_text,
                operator=MutationOperator.PARAPHRASING,
                success=True
            )
        except Exception as e:
            return MutationResult(
                original_text=instruction,
                mutated_text=instruction,
                operator=MutationOperator.PARAPHRASING,
                success=False,
                error_message=str(e)
            )

    def insert_speed_control(self, instruction: str, baseline_avg_speed: float, baseline_max_speed: float) -> MutationResult:
        """
        Insert speed control commands into the instruction with three types of control.
        
        Args:
            instruction: Original driving instruction
            baseline_avg_speed: Average speed from previous execution (for relative control)
            baseline_max_speed: Maximum speed from previous execution (for absolute limit control)
            
        Returns:
            MutationResult containing the instruction with speed control and metadata
        """
        # Randomly select speed control type
        # control_types = ["FASTER", "SLOWER", "ABSOLUTE_LIMIT"]
        # chosen_type = random.choice(control_types)
        
        chosen_type = "ABSOLUTE_LIMIT"
        
        metadata = {
            "speed_control_type": chosen_type,
            "baseline_avg_speed": baseline_avg_speed,
            "baseline_max_speed": baseline_max_speed
        }
        
        chosen_command = "" # Initialize
        # Generate the speed command based on type
        if chosen_type == "FASTER":
            chosen_command = "FASTER"
        elif chosen_type == "SLOWER":
            chosen_command = "SLOWER"
        else:  # ABSOLUTE_LIMIT
            speed_limit = int(baseline_max_speed * random.uniform(self.speed_limit_range[0], self.speed_limit_range[1])) 
            metadata["target_speed_limit"] = speed_limit
            chosen_command = f"LIMIT_{speed_limit}"
        
        prompt = f"""
You are tasked with adding a speed control instruction to a driving command.

Original driving instruction: "{instruction}"
Speed control command: "{chosen_command}"

Your task is to integrate the speed control naturally into the driving instruction. Parse the command as follows:
- "FASTER": Add instructions for driving at a faster overall pace, like "drive fast and", "drive quickly and", "go at a fast pace and", "drive with more speed and", etc.
- "SLOWER": Add instructions for driving at a slower overall pace, like "drive slowly and", "drive at a slow pace and", "go slowly and", "drive gently and", "take it easy and", etc.
- "LIMIT_XX": Add speed limit instructions like "don't exceed XX km/h and", "keep under XX km/h and", "stay below XX km/h and", "limit speed to XX km/h and", etc.

Requirements:
- Maintain the original driving action exactly
- Integrate the speed command smoothly and naturally before the main action
- Ensure the resulting instruction sounds like something a human would say
- Keep the instruction clear and actionable
- Focus on overall driving pace rather than momentary acceleration/deceleration actions
- Choose natural, varied expressions for the speed control

Examples:
- "Turn left" + "FASTER" → "Drive fast and turn left"
- "Change lanes" + "SLOWER" → "Drive slowly and change lanes"
- "Go straight" + "LIMIT_45" → "Don't exceed 45 km/h and go straight"

Return only the modified instruction. Do not include explanations or additional text.
"""
        
        try:
            mutated_text = self._call_llm(prompt)
            
            logger.info(f"Applied speed control: {chosen_type} ({chosen_command})")
            if baseline_avg_speed:
                logger.info(f"Baseline average speed: {baseline_avg_speed:.2f} km/h")
            if baseline_max_speed:
                logger.info(f"Baseline maximum speed: {baseline_max_speed:.2f} km/h")
            
            return MutationResult(
                original_text=instruction,
                mutated_text=mutated_text,
                operator=MutationOperator.SPEED_CONTROL_INSERTION,
                success=True,
                metadata=metadata
            )
        except Exception as e:
            return MutationResult(
                original_text=instruction,
                mutated_text=instruction,
                operator=MutationOperator.SPEED_CONTROL_INSERTION,
                success=False,
                error_message=str(e),
                metadata=metadata
            )

    def insert_maintain_distance(self, instruction: str, target_vehicle: Dict[str, Any], maintain_distance: float) -> MutationResult:
        """
        Insert maintain distance commands into the instruction.
        
        Args:
            instruction: Original driving instruction
            target_vehicle: Dictionary containing target vehicle information
            maintain_distance: Distance to maintain from the target vehicle in meters
            
        Returns:
            MutationResult containing the instruction with distance maintenance and metadata
        """
        # Extract vehicle information
        vehicle_name = target_vehicle.get('name', 'vehicle')
        vehicle_color = target_vehicle.get('color')
        
        # Create vehicle description
        if vehicle_color:
            vehicle_description = f"{vehicle_color.lower()} {vehicle_name.lower()}"
        else:
            vehicle_description = vehicle_name.lower()
        
        metadata = {
            "target_vehicle_id": target_vehicle.get('id'),
            "target_vehicle_name": vehicle_name,
            "target_vehicle_color": vehicle_color,
            "maintain_distance": maintain_distance,
            "initial_distance": target_vehicle.get('initial_distance')
        }
        
        prompt = f"""
You are tasked with adding a distance maintenance instruction to a driving command.

Original driving instruction: "{instruction}"
Target vehicle: {vehicle_description}
Distance to maintain: {maintain_distance} meters

Your task is to integrate the distance maintenance naturally into the driving instruction. The system should maintain the specified distance from the target vehicle while executing the original instruction.

Requirements:
- Maintain the original driving action exactly
- Integrate the distance maintenance smoothly and naturally
- Ensure the resulting instruction sounds like something a human would say
- Keep the instruction clear and actionable
- Use natural expressions for distance maintenance

Examples of natural distance maintenance phrases:
- "keep {maintain_distance}m from the {vehicle_description} and"
- "maintain {maintain_distance} meters distance from the {vehicle_description} while"
- "stay {maintain_distance}m behind the {vehicle_description} and"
- "keep a {maintain_distance}-meter gap from the {vehicle_description} and"
- "maintain safe distance of {maintain_distance}m from the {vehicle_description} and"

Examples:
- "Turn left" + "maintain 8m from red car" → "Keep 8 meters from the red car and turn left"
- "Change lanes" + "maintain 10m from blue truck" → "Maintain 10 meters distance from the blue truck while changing lanes"
- "Go straight" + "maintain 7m from white sedan" → "Stay 7m behind the white sedan and go straight"

Return only the modified instruction. Do not include explanations or additional text.
"""
        
        try:
            mutated_text = self._call_llm(prompt)
            
            logger.info(f"Applied maintain distance: {maintain_distance}m from {vehicle_description}")
            logger.info(f"Target vehicle initial distance: {target_vehicle.get('initial_distance', 'N/A')}m")
            
            return MutationResult(
                original_text=instruction,
                mutated_text=mutated_text,
                operator=MutationOperator.MAINTAIN_DISTANCE,
                success=True,
                metadata=metadata
            )
        except Exception as e:
            return MutationResult(
                original_text=instruction,
                mutated_text=instruction,
                operator=MutationOperator.MAINTAIN_DISTANCE,
                success=False,
                error_message=str(e),
                metadata=metadata
            )

    def apply_mutation(self, instruction: str, operator: MutationOperator) -> MutationResult:
        """
        Apply a specific mutation operator to an instruction.
        
        Args:
            instruction: The instruction to mutate
            operator: The mutation operator to apply
            
        Returns:
            MutationResult containing the mutation result
        """
        if operator == MutationOperator.IRRELEVANT_CONVERSATION:
            return self.insert_irrelevant_conversation(instruction)
        elif operator == MutationOperator.PARAPHRASING:
            return self.paraphrase_instruction(instruction)
        else:
            raise ValueError(f"Unknown operator: {operator}")

def get_all_operators() -> List[MutationOperator]:
    """
    Get a list of all available mutation operators.
    
    Returns:
        List of all MutationOperator enum values
    """
    return list(MutationOperator)


def parse_operator_list(operator_names: Optional[str]) -> List[MutationOperator]:
    """
    Parse a comma-separated operator list from CLI input.

    Args:
        operator_names: Comma-separated operator names, or None/"all" for all operators.

    Returns:
        List of mutation operators in the requested order.
    """
    if operator_names is None or not operator_names.strip() or operator_names.strip().lower() == "all":
        return get_all_operators()

    valid_names = {operator.value: operator for operator in MutationOperator}
    operators = []
    for raw_name in operator_names.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in valid_names:
            expected = ", ".join(sorted(valid_names))
            raise ValueError(f"Unsupported mutation operator '{name}'. Expected one of: {expected}")
        operators.append(valid_names[name])

    if not operators:
        raise ValueError("At least one mutation operator must be selected")
    return operators

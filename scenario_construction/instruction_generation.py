import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _strip_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _extract_commands(parsed) -> List[str]:
    if isinstance(parsed, str):
        command = parsed.strip()
        return [command] if command else []

    if isinstance(parsed, list):
        return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]

    if isinstance(parsed, dict):
        commands = parsed.get("commands")
        if isinstance(commands, list):
            extracted = [item.strip() for item in commands if isinstance(item, str) and item.strip()]
            if extracted:
                return extracted

        for key in ("instruction", "command", "text", "response"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return [value.strip()]

    return []


def parse_instruction_response(raw_response: str) -> str:
    text = _strip_json_fence(raw_response.strip())
    if not text:
        raise ValueError("Instruction response is empty")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text

    commands = _extract_commands(parsed)
    if commands:
        return commands[0]

    raise ValueError("Could not extract an instruction from response")


def parse_command_candidates(raw_response: str) -> List[str]:
    text = _strip_json_fence(raw_response.strip())
    if not text:
        raise ValueError("Instruction response is empty")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]

    commands = _extract_commands(parsed)
    if not commands:
        raise ValueError("Could not extract command candidates from response")
    return commands


def structured_instruction_record(
    *,
    raw_response: str,
    mode: str,
    generator_model: str,
    prompt_path: Path,
) -> Dict:
    return {
        "instruction": parse_instruction_response(raw_response),
        "commands": parse_command_candidates(raw_response),
        "mode": mode,
        "generator_model": generator_model,
        "prompt_path": str(prompt_path),
        "created_at": datetime.now().isoformat(),
        "raw_response": raw_response,
    }

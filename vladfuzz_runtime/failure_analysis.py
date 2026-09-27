from pathlib import Path
from typing import Dict, Optional
import json


FAILURE_CATEGORIES = (
    "Collision",
    "Timeout/Stuck",
    "Out of Bounds",
    "Lane Invasion",
    "Other Oracle Failure",
    "Speed Limit Exceeded",
    "Maintain Distance Failed",
    "Speed Too Slow",
    "Speed Too Fast",
    "Execution Error",
    "Unhandled Exception",
    "Target Vehicle Missing",
)


def categorize_failure_reason(reason: Optional[str]) -> str:
    if not reason:
        return "Unknown"
    if "Max speed" in reason and "> Limit" in reason:
        return "Speed Limit Exceeded"
    if "Avg speed" in reason and "<= Baseline" in reason:
        return "Speed Too Slow"
    if "Avg speed" in reason and ">= Baseline" in reason:
        return "Speed Too Fast"
    if "Min distance" in reason and "< Target" in reason:
        return "Maintain Distance Failed"
    if "Collision detected" in reason:
        return "Collision"
    if "collision" in reason:
        return "Collision"
    if "lane_invasion" in reason or "lane invasion" in reason:
        return "Lane Invasion"
    if "execution_error" in reason or "CARLA execution error" in reason:
        return "Execution Error"
    if "timeout" in reason or "target_not_reached" in reason or "Scenario not completed" in reason:
        return "Timeout/Stuck"
    if "stuck" in reason:
        return "Timeout/Stuck"
    if "out_of_bounds" in reason or "out of bounds" in reason:
        return "Out of Bounds"
    if "other" in reason:
        return "Other Oracle Failure"
    if "Unhandled exception" in reason:
        return "Unhandled Exception"
    if "Target vehicle" in reason and "not found" in reason:
        return "Target Vehicle Missing"
    return reason


def collect_failure_analysis(failure_dir: str) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    root = Path(failure_dir)
    if not root.exists():
        return stats

    for result_path in sorted(root.glob("*/result.json")):
        try:
            with result_path.open("r", encoding="utf-8") as file:
                result = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        category = categorize_failure_reason(result.get("semantic_failure_reason"))
        stats[category] = stats.get(category, 0) + 1
    return stats

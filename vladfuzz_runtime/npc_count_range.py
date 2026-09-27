import random
from typing import Optional, Tuple


def validate_npc_count_range(minimum: int, maximum: int) -> Tuple[int, int]:
    if minimum < 0:
        raise ValueError("Minimum NPC count must be non-negative")
    if maximum < minimum:
        raise ValueError("Maximum NPC count must be greater than or equal to the minimum")
    return minimum, maximum


def sample_npc_count(minimum: int, maximum: int, rng: Optional[object] = None) -> int:
    minimum, maximum = validate_npc_count_range(minimum, maximum)
    return (rng or random).randint(minimum, maximum)

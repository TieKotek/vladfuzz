"""Deterministic instruction counterfactual generation primitives."""

from dataclasses import dataclass
from enum import Enum
import hashlib
import random
import re
from typing import Dict, List, Sequence, Tuple


class Maneuver(str, Enum):
    FOLLOW_LANE = "follow_lane"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    CHANGE_LANE_LEFT = "change_lane_left"
    CHANGE_LANE_RIGHT = "change_lane_right"
    GO_STRAIGHT = "go_straight"


@dataclass(frozen=True)
class CanonicalInstruction:
    source: str
    maneuvers: Tuple[Maneuver, ...]


class InstructionFamily(str, Enum):
    PARAPHRASE = "paraphrase"
    AMBIGUITY = "ambiguity"
    NOISE = "noise"


@dataclass(frozen=True)
class CounterfactualVariant:
    family: InstructionFamily
    family_index: int
    template_id: str
    instruction: str
    maneuvers: Tuple[Maneuver, ...]


CLAUSE_TO_MANEUVER = {
    "follow current lane for a while": Maneuver.FOLLOW_LANE,
    "turn left at intersection": Maneuver.TURN_LEFT,
    "turn right at intersection": Maneuver.TURN_RIGHT,
    "change lane to the left": Maneuver.CHANGE_LANE_LEFT,
    "change lane to the right": Maneuver.CHANGE_LANE_RIGHT,
    "go straight at intersection": Maneuver.GO_STRAIGHT,
}


def parse_basic_instruction(text: str) -> CanonicalInstruction:
    source = text
    normalized = text.strip().lower()
    normalized = re.sub(r"[.!?]+$", "", normalized).strip()
    clauses = re.split(r"\s+then\s+", normalized) if normalized else []

    maneuvers = []
    if not clauses or any(not clause.strip() for clause in clauses):
        raise ValueError("Unsupported basic_instruction clause: <empty>")

    for clause in clauses:
        clause = clause.strip()
        maneuver = CLAUSE_TO_MANEUVER.get(clause)
        if maneuver is None:
            raise ValueError(f"Unsupported basic_instruction clause: {clause or '<empty>'}")
        maneuvers.append(maneuver)

    return CanonicalInstruction(source=source, maneuvers=tuple(maneuvers))


_PARAPHRASE_CLAUSES: Dict[Maneuver, Sequence[str]] = {
    Maneuver.FOLLOW_LANE: (
        "Continue in the current lane for a while",
        "Keep following the current lane for a while",
        "Stay in the current lane and continue for a while",
        "Proceed along the current lane for a while",
        "Keep to the current lane for a while",
        "Drive ahead in the current lane for a while",
        "Continue ahead in this lane for a while",
        "Remain in the current lane for a while",
        "Carry on in the current lane for a while",
        "Maintain the current lane for a while",
        "Follow this lane for a while",
        "Continue driving in the current lane for a while",
    ),
    Maneuver.TURN_LEFT: (
        "Turn left at the next intersection",
        "Make a left turn at the next intersection",
        "Take a left turn at the next intersection",
        "At the next intersection, turn left",
        "Proceed to the next intersection and turn left",
        "Continue until the next intersection, then turn left",
        "Make the next intersection a left turn",
        "Take the left turn at the upcoming intersection",
        "When you reach the next intersection, turn left",
        "Head left at the next intersection",
        "At the upcoming intersection, make a left turn",
        "Follow the road to the next intersection and turn left",
    ),
    Maneuver.TURN_RIGHT: (
        "Turn right at the next intersection",
        "Make a right turn at the next intersection",
        "Take a right turn at the next intersection",
        "At the next intersection, turn right",
        "Proceed to the next intersection and turn right",
        "Continue until the next intersection, then turn right",
        "Make the next intersection a right turn",
        "Take the right turn at the upcoming intersection",
        "When you reach the next intersection, turn right",
        "Head right at the next intersection",
        "At the upcoming intersection, make a right turn",
        "Follow the road to the next intersection and turn right",
    ),
    Maneuver.CHANGE_LANE_LEFT: (
        "Change to the lane on the left",
        "Move into the lane to the left",
        "Shift into the left lane",
        "Change lanes to the left",
        "Move one lane to the left",
        "Merge into the lane on the left",
        "Take the lane immediately to the left",
        "Make a lane change to the left",
        "Move over to the left lane",
        "Transition into the lane to your left",
        "Change into the adjacent lane on the left",
        "Shift one lane to the left",
    ),
    Maneuver.CHANGE_LANE_RIGHT: (
        "Change to the lane on the right",
        "Move into the lane to the right",
        "Shift into the right lane",
        "Change lanes to the right",
        "Move one lane to the right",
        "Merge into the lane on the right",
        "Take the lane immediately to the right",
        "Make a lane change to the right",
        "Move over to the right lane",
        "Transition into the lane to your right",
        "Change into the adjacent lane on the right",
        "Shift one lane to the right",
    ),
    Maneuver.GO_STRAIGHT: (
        "Go straight through the next intersection",
        "Continue straight at the next intersection",
        "Proceed straight through the next intersection",
        "Drive straight across the next intersection",
        "At the next intersection, continue straight",
        "Keep straight through the upcoming intersection",
        "Head straight at the next intersection",
        "Cross the next intersection going straight",
        "Maintain a straight course through the next intersection",
        "Take the straight path at the next intersection",
        "Continue directly ahead through the upcoming intersection",
        "Proceed directly ahead through the next intersection",
    ),
}

_AMBIGUITY_CLAUSES: Dict[Maneuver, Sequence[str]] = {
    Maneuver.FOLLOW_LANE: (
        "Continue in the current lane", "Keep following the current lane",
        "Stay in the current lane", "Proceed along the current lane",
        "Keep to the current lane", "Drive ahead in the current lane",
        "Continue ahead in this lane", "Remain in the current lane",
        "Carry on in the current lane", "Maintain the current lane",
        "Follow this lane", "Continue driving in the current lane",
    ),
    Maneuver.TURN_LEFT: (
        "Turn left", "Make a left turn", "Take a left turn", "Turn to the left",
        "Proceed with a left turn", "Continue by turning left",
        "Take the left turn", "Turn toward the left", "Make the turn to the left",
        "Follow the left turn", "Bear left by turning", "Carry out a left turn",
    ),
    Maneuver.TURN_RIGHT: (
        "Turn right", "Make a right turn", "Take a right turn", "Turn to the right",
        "Proceed with a right turn", "Continue by turning right",
        "Take the right turn", "Turn toward the right", "Make the turn to the right",
        "Follow the right turn", "Bear right by turning", "Carry out a right turn",
    ),
    Maneuver.CHANGE_LANE_LEFT: (
        "Change to a lane on the left", "Move into a lane to the left",
        "Shift into the left lane", "Change lanes to the left",
        "Move one lane to the left", "Merge into a lane on the left",
        "Take a lane immediately to the left", "Make a lane change to the left",
        "Move over to the left lane", "Transition into a lane to your left",
        "Change into an adjacent lane on the left", "Shift one lane to the left",
    ),
    Maneuver.CHANGE_LANE_RIGHT: (
        "Change to a lane on the right", "Move into a lane to the right",
        "Shift into the right lane", "Change lanes to the right",
        "Move one lane to the right", "Merge into a lane on the right",
        "Take a lane immediately to the right", "Make a lane change to the right",
        "Move over to the right lane", "Transition into a lane to your right",
        "Change into an adjacent lane on the right", "Shift one lane to the right",
    ),
    Maneuver.GO_STRAIGHT: (
        "Go straight", "Continue straight", "Proceed straight", "Drive straight",
        "Keep going straight", "Head straight", "Maintain a straight course",
        "Take the straight path", "Continue directly ahead", "Proceed directly ahead",
        "Carry on straight", "Keep straight",
    ),
}

_CONNECTORS = (
    ", then ", "; then ", ", and then ", ". Then ", "; after that, ",
    ", after which ", ". Afterward, ", ", before you ", "; next, ",
    ". From there, ", ", and afterward ", "; subsequently, ",
)


def _render_clause_sequence(
    maneuvers: Sequence[Maneuver],
    bank: Dict[Maneuver, Sequence[str]],
    index: int,
) -> str:
    clauses = [bank[maneuver][index] for maneuver in maneuvers]
    clauses[1:] = [clause[0].lower() + clause[1:] for clause in clauses[1:]]
    text = _CONNECTORS[index].join(clauses)
    return text[0].upper() + text[1:] + "."


def _noise_candidates(source: str) -> List[Tuple[str, str]]:
    base = source.strip().rstrip(".!?")
    first, separator, rest = base.partition(" ")
    comma_variant = base.replace(" at intersection", ", at intersection", 1)
    if comma_variant == base:
        comma_variant = base.replace(" then ", ", then ", 1)
    and_then = base.replace(" then ", " and then ", 1)
    if and_then == base:
        and_then = base.replace(" at intersection", " at the intersection", 1)
    semicolon = base.replace(" then ", "; then ", 1)
    if semicolon == base:
        semicolon = base.replace(" at intersection", " at intersection;", 1)
    intersecton = base.replace("intersection", "intersecton", 1)
    if intersecton == base:
        intersecton = base.replace("while", "whlie", 1)
    interseciton = base.replace("intersection", "interseciton", 1)
    if interseciton == base:
        interseciton = base.replace("while", "whiel", 1)
    whlie = base.replace("while", "whlie", 1)
    if whlie == base:
        whlie = base.replace("intersection", "intersetion", 1)
    return [
        ("noise-lowercase", base.lower() + "."),
        ("noise-uppercase", base.upper() + "."),
        ("noise-exclamation", base + "!"),
        ("noise-no-terminal", base),
        ("noise-double-space", first + "  " + rest + "." if separator else base + " ."),
        ("noise-comma", comma_variant + "."),
        ("noise-and-then", and_then + "."),
        ("noise-semicolon", semicolon + "."),
        ("noise-intersecton", intersecton + "."),
        ("noise-interseciton", interseciton + "."),
        ("noise-whlie", whlie + "."),
        ("noise-title", base.title() + "."),
    ]


_ACTION_PATTERNS = (
    (
        Maneuver.CHANGE_LANE_LEFT,
        re.compile(
            r"\b(?:change|move|shift|merge|take|transition|make)\b[^.;,]{0,45}"
            r"\blane(?:s)?\b[^.;,]{0,35}\bleft\b|\bleft\s+lane\b",
            re.I,
        ),
    ),
    (
        Maneuver.CHANGE_LANE_RIGHT,
        re.compile(
            r"\b(?:change|move|shift|merge|take|transition|make)\b[^.;,]{0,45}"
            r"\blane(?:s)?\b[^.;,]{0,35}\bright\b|\bright\s+lane\b",
            re.I,
        ),
    ),
    (
        Maneuver.TURN_LEFT,
        re.compile(
            r"\b(?:turn|take|make|go|head|bear|carry|follow)\b[^.;,]{0,35}\bleft\b"
            r"|\bleft\b[^.;,]{0,20}\bturn(?:ing)?\b",
            re.I,
        ),
    ),
    (
        Maneuver.TURN_RIGHT,
        re.compile(
            r"\b(?:turn|take|make|go|head|bear|carry|follow)\b[^.;,]{0,35}\bright\b"
            r"|\bright\b[^.;,]{0,20}\bturn(?:ing)?\b",
            re.I,
        ),
    ),
    (
        Maneuver.GO_STRAIGHT,
        re.compile(
            r"\b(?:go|continue|proceed|drive|keep|head|cross|maintain|take|carry)\b"
            r"[^.;,]{0,40}\b(?:straight|directly ahead)\b"
            r"|\bstraight\b[^.;,]{0,20}\b(?:path|course)\b",
            re.I,
        ),
    ),
    (
        Maneuver.FOLLOW_LANE,
        re.compile(
            r"\b(?:follow|continue|keep|stay|proceed|drive|remain|carry|maintain)\b"
            r"[^.;,]{0,50}\b(?:current|this)\s+lane\b",
            re.I,
        ),
    ),
)


def classify_rendered_instruction(text: str) -> Tuple[Maneuver, ...]:
    matches = []
    for maneuver, pattern in _ACTION_PATTERNS:
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), maneuver))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    selected = []
    occupied_until = -1
    for start, end, maneuver in matches:
        if start < occupied_until:
            continue
        selected.append(maneuver)
        occupied_until = end
    if not selected:
        raise ValueError(f"Cannot classify rendered instruction: {text}")
    return tuple(selected)


def _stable_rng(random_seed: int, scenario_index: int, source: str) -> random.Random:
    material = f"{random_seed}\0{scenario_index}\0{source}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _validated_candidates(
    family: InstructionFamily,
    canonical: CanonicalInstruction,
) -> List[Tuple[str, str]]:
    if family is InstructionFamily.NOISE:
        candidates = _noise_candidates(canonical.source)
    else:
        bank = (
            _PARAPHRASE_CLAUSES
            if family is InstructionFamily.PARAPHRASE
            else _AMBIGUITY_CLAUSES
        )
        candidates = [
            (
                f"{family.value}-{index + 1:02d}",
                _render_clause_sequence(canonical.maneuvers, bank, index),
            )
            for index in range(12)
        ]

    source_normalized = canonical.source.strip().casefold()
    valid = []
    seen = set()
    for template_id, instruction in candidates:
        key = instruction.strip().casefold()
        if key == source_normalized or key in seen:
            continue
        try:
            signature = classify_rendered_instruction(instruction)
        except ValueError:
            continue
        if signature != canonical.maneuvers:
            continue
        seen.add(key)
        valid.append((template_id, instruction))
    return valid


def build_counterfactual_batch(
    source: str,
    *,
    k: int = 8,
    random_seed: int = 0,
    scenario_index: int = 0,
) -> List[CounterfactualVariant]:
    if k <= 0:
        raise ValueError("k must be positive")
    canonical = parse_basic_instruction(source)
    rng = _stable_rng(random_seed, scenario_index, canonical.source)
    chosen: Dict[InstructionFamily, List[CounterfactualVariant]] = {}
    used_instructions = set()

    for family in InstructionFamily:
        candidates = _validated_candidates(family, canonical)
        rng.shuffle(candidates)
        family_variants = []
        for template_id, instruction in candidates:
            key = instruction.strip().casefold()
            if key in used_instructions:
                continue
            family_variants.append(
                CounterfactualVariant(
                    family=family,
                    family_index=len(family_variants) + 1,
                    template_id=template_id,
                    instruction=instruction,
                    maneuvers=canonical.maneuvers,
                )
            )
            used_instructions.add(key)
            if len(family_variants) == k:
                break
        if len(family_variants) < k:
            raise ValueError(
                f"{family.value} cannot supply {k} unique variants for: {source}"
            )
        chosen[family] = family_variants

    return [
        chosen[family][index]
        for index in range(k)
        for family in InstructionFamily
    ]

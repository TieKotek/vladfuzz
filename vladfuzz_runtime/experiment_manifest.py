import json
from pathlib import Path
from typing import Dict, Iterable, Optional


INSTRUCTION_SOURCES = ("manual", "basic", "route_prior", "no_prior")


def iter_manifest_rows(manifest_path: str) -> Iterable[Dict]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row must be an object in {manifest_path}:{line_number}")
            yield row


def load_manifest_row(manifest_path: str, manifest_id: Optional[str] = None) -> Dict:
    rows = list(iter_manifest_rows(manifest_path))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    if manifest_id is None:
        return rows[0]

    for row in rows:
        if row.get("id") == manifest_id:
            return row

    available = ", ".join(str(row.get("id")) for row in rows[:10])
    suffix = "..." if len(rows) > 10 else ""
    raise ValueError(f"Manifest id '{manifest_id}' not found in {manifest_path}. Available examples: {available}{suffix}")


def resolve_instruction(row: Dict, source: str, manual_instruction: Optional[str] = None) -> str:
    if source not in INSTRUCTION_SOURCES:
        expected = ", ".join(INSTRUCTION_SOURCES)
        raise ValueError(f"Unsupported instruction source '{source}'. Expected one of: {expected}")

    if source == "manual":
        if manual_instruction:
            return manual_instruction
        raise ValueError("--instruction is required when --instruction-source manual is used")

    field_name = {
        "basic": "basic_instruction",
        "route_prior": "route_prior_instruction",
        "no_prior": "no_prior_instruction",
    }[source]

    instruction = row.get(field_name)
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()

    raise ValueError(f"Manifest row '{row.get('id')}' does not contain a usable {field_name}")


def apply_manifest_selection(
    manifest_path: Optional[str],
    manifest_id: Optional[str],
    instruction_source: str,
    static_scenario: Optional[str],
    dynamic_scenario: Optional[str],
    instruction: Optional[str],
) -> Dict:
    if manifest_path is None:
        if instruction_source != "manual":
            raise ValueError("--manifest is required when --instruction-source is not manual")
        missing = []
        if not static_scenario:
            missing.append("--static-scenario")
        if not dynamic_scenario:
            missing.append("--dynamic-scenario")
        if not instruction:
            missing.append("--instruction")
        if missing:
            raise ValueError(
                "Direct input mode requires " + ", ".join(missing)
            )
        return {
            "manifest_row": None,
            "manifest_id": None,
            "static_scenario": static_scenario,
            "dynamic_scenario": dynamic_scenario,
            "instruction": resolve_instruction({}, "manual", instruction),
            "instruction_source": instruction_source,
        }

    row = load_manifest_row(manifest_path, manifest_id)
    resolved_instruction = resolve_instruction(row, instruction_source, instruction)
    resolved_static_scenario = row.get("static_scenario") or static_scenario
    resolved_dynamic_scenario = row.get("seed_scenario") or dynamic_scenario
    missing = []
    if not resolved_static_scenario:
        missing.append("static_scenario")
    if not resolved_dynamic_scenario:
        missing.append("seed_scenario")
    if missing:
        raise ValueError(
            f"Manifest row '{row.get('id')}' is missing: {', '.join(missing)}"
        )

    return {
        "manifest_row": row,
        "manifest_id": row.get("id"),
        "static_scenario": resolved_static_scenario,
        "dynamic_scenario": resolved_dynamic_scenario,
        "instruction": resolved_instruction,
        "instruction_source": instruction_source,
    }

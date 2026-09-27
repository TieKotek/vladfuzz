import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_construction.instruction_generation import structured_instruction_record


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
OUTPUT_BY_MODE = {
    "route_prior": "command_prior.json",
    "no_prior": "command_no_prior.json",
}


def image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _find_first_image(seed_dir: Path) -> Optional[Path]:
    for path in sorted(seed_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def _find_scenario_json(seed_dir: Path) -> Optional[Path]:
    ignored = set(OUTPUT_BY_MODE.values())
    for path in sorted(seed_dir.glob("*.json")):
        if path.name not in ignored:
            return path
    return None


def iter_seed_dirs(seeds_root: Path) -> Iterable[Path]:
    for scenario_dir in sorted(seeds_root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        for seed_dir in sorted(scenario_dir.iterdir()):
            if seed_dir.is_dir() and seed_dir.name != "seed_fail":
                yield seed_dir


def render_prompt_template(prompt_text: str, *, num_commands: int) -> str:
    return prompt_text.replace("{num_commands}", str(num_commands))


def _scenario_type_hint(scenario_path: Path) -> str:
    seed_dir = scenario_path.parent
    if seed_dir.name.startswith("seed_") and seed_dir.parent != seed_dir:
        return seed_dir.parent.name
    return seed_dir.name


def _scene_context_block(scenario_path: Path, scenario_file: dict) -> str:
    map_name = scenario_file.get("map_name", "unknown")
    return (
        "\n\n[SCENE CONTEXT]\n"
        f"Scenario type hint: {_scenario_type_hint(scenario_path)}\n"
        f"Map: {map_name}\n"
        "Use this scene context as a weak hint for road type. "
        "Do not mention the scenario type hint or map name in the output."
    )


def _prompt_for_mode(mode: str, prompt_path: Path, scenario_path: Path, *, num_commands: int) -> str:
    prompt_text = render_prompt_template(prompt_path.read_text(encoding="utf-8"), num_commands=num_commands)
    with scenario_path.open("r", encoding="utf-8") as json_file:
        scenario_file = json.load(json_file)

    prompt_text += _scene_context_block(scenario_path, scenario_file)
    if mode != "route_prior":
        return prompt_text

    route_description = json.dumps(
        scenario_file["route_info"]["route_description"],
        ensure_ascii=False,
        indent=4,
    )
    return prompt_text + "'''\n" + route_description + "\n'''"


def generate_instruction(image_path: Path, prompt_text: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    base64_image = image_to_base64(image_path)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    completion = client.chat.completions.create(model=model, messages=messages, stream=False)
    return completion.choices[0].message.content


def _modes_from_arg(mode: str):
    if mode == "both":
        return ["route_prior", "no_prior"]
    return [mode]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VLM driving instructions for seed scenarios.")
    parser.add_argument("--seeds-root", default="seeds")
    parser.add_argument("--mode", choices=["route_prior", "no_prior", "both"], default="both")
    parser.add_argument("--route-prior-prompt", default=str(SCRIPT_DIR / "prompt_prior.txt"))
    parser.add_argument("--no-prior-prompt", default=str(SCRIPT_DIR / "prompt_no_prior.txt"))
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--num-commands", type=int, default=5, help="Number of command variants to request per seed.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_commands <= 0:
        raise ValueError("--num-commands must be positive")

    prompt_by_mode = {
        "route_prior": Path(args.route_prior_prompt),
        "no_prior": Path(args.no_prior_prompt),
    }
    count = 0
    for seed_dir in iter_seed_dirs(Path(args.seeds_root)):
        if args.limit is not None and count >= args.limit:
            break

        image_path = _find_first_image(seed_dir)
        scenario_path = _find_scenario_json(seed_dir)
        if image_path is None or scenario_path is None:
            print(f"Skipping {seed_dir}: missing image or scenario JSON")
            continue

        for mode in _modes_from_arg(args.mode):
            output_path = seed_dir / OUTPUT_BY_MODE[mode]
            if output_path.exists() and not args.overwrite:
                print(f"Skipping existing output: {output_path}")
                continue

            prompt_path = prompt_by_mode[mode]
            prompt_text = _prompt_for_mode(mode, prompt_path, scenario_path, num_commands=args.num_commands)
            print(f"Processing {mode} instruction for seed: {seed_dir}")
            raw_response = generate_instruction(image_path, prompt_text, args.model)
            record = structured_instruction_record(
                raw_response=raw_response,
                mode=mode,
                generator_model=args.model,
                prompt_path=prompt_path,
            )
            output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {output_path}")
        count += 1

    print(f"Processed {count} seed directories")


if __name__ == "__main__":
    main()

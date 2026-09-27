import base64
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_construction.generate_instructions import render_prompt_template


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_command(image_path: Path, route_description: str, prompt_path: Path, model: str, num_commands: int) -> str:
    from openai import OpenAI

    with prompt_path.open("r", encoding="utf-8") as file:
        prompt_text = render_prompt_template(file.read(), num_commands=num_commands)

    prompt_text += "'''\n" + route_description + "\n'''"

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        }
    ]

    base64_image = image_to_base64(image_path)
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt_text},
            ],
        }
    )

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )
    return completion.choices[0].message.content


def _find_first_image(seed_dir: Path) -> Optional[Path]:
    for path in sorted(seed_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return path
    return None


def _find_scenario_json(seed_dir: Path) -> Optional[Path]:
    ignored = {"command_prior.json", "command_no_prior.json", "commands.json", "normal_commands.json"}
    for path in sorted(seed_dir.glob("*.json")):
        if path.name not in ignored:
            return path
    return None


def _iter_seed_dirs(seeds_root: Path) -> Iterable[Path]:
    for scenario_dir in sorted(seeds_root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        for seed_dir in sorted(scenario_dir.iterdir()):
            if seed_dir.is_dir() and seed_dir.name != "seed_fail":
                yield seed_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate route-prior VLM driving instructions for seed scenarios.")
    parser.add_argument("--seeds-root", default="seeds")
    parser.add_argument("--prompt", default=str(SCRIPT_DIR / "prompt_prior.txt"))
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--output-name", default="command_prior.json")
    parser.add_argument("--num-commands", type=int, default=5, help="Number of command variants to request per seed.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_commands <= 0:
        raise ValueError("--num-commands must be positive")

    prompt_path = Path(args.prompt)
    count = 0
    for seed_dir in _iter_seed_dirs(Path(args.seeds_root)):
        if args.limit is not None and count >= args.limit:
            break

        output_path = seed_dir / args.output_name
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing output: {output_path}")
            continue

        image_path = _find_first_image(seed_dir)
        scenario_path = _find_scenario_json(seed_dir)
        if image_path is None or scenario_path is None:
            print(f"Skipping {seed_dir}: missing image or scenario JSON")
            continue

        with scenario_path.open("r", encoding="utf-8") as json_file:
            scenario_file = json.load(json_file)
        route_description = json.dumps(
            scenario_file["route_info"]["route_description"],
            ensure_ascii=False,
            indent=4,
        )

        print(f"Processing seed: {seed_dir}")
        response = generate_command(image_path, route_description, prompt_path, args.model, args.num_commands)
        output_path.write_text(response, encoding="utf-8")
        print(f"Wrote {output_path}")
        count += 1

    print(f"Generated route-prior commands for {count} seeds")


if __name__ == "__main__":
    main()

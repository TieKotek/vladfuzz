import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vladfuzz_runtime.experiment_commands import RQ_METHODS, build_experiment_commands
from vladfuzz_runtime.experiment_manifest import INSTRUCTION_SOURCES, iter_manifest_rows


def _select_rows(manifest_path: str, manifest_ids: List[str], limit: int) -> List[Dict]:
    rows = list(iter_manifest_rows(manifest_path))
    if manifest_ids:
        wanted = set(manifest_ids)
        rows = [row for row in rows if row.get("id") in wanted]
        missing = sorted(wanted - {row.get("id") for row in rows})
        if missing:
            raise ValueError(f"Manifest ids not found: {', '.join(missing)}")
    if limit is not None:
        rows = rows[:limit]
    return rows


def _base_command(args, row: Dict) -> List[str]:
    command = [
        "python3",
        "-m",
        args.mode_script,
        "--model",
        args.model,
        "--manifest",
        args.manifest,
        "--manifest-id",
        row["id"],
        "--instruction-source",
        args.instruction_source,
    ]

    if args.mode == "run":
        command.extend([
            "--gpu-id",
            str(args.gpu_id),
            "--success-distance",
            str(args.success_distance),
        ])
    elif args.mode == "fuzz":
        command.extend([
            "--gpu-id",
            str(args.gpu_id),
            "--mutation-depth",
            str(args.mutation_depth),
            "--success-distance",
            str(args.success_distance),
            "--llm-provider",
            args.llm_provider,
            "--llm-model",
            args.llm_model,
        ])
    elif args.mode == "nsga":
        command.extend([
            "--gpu-id",
            str(args.gpu_id),
            "--mutation-depth",
            str(args.mutation_depth),
            "--success-distance",
            str(args.success_distance),
            "--pop-size",
            str(args.pop_size),
            "--ngen",
            str(args.ngen),
            "--cxpb",
            str(args.cxpb),
            "--mutpb",
            str(args.mutpb),
        ])
        if args.max_simulations is not None:
            command.extend(["--max-simulations", str(args.max_simulations)])
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    return command


def _write_commands(commands: Iterable[List[str]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for command in commands:
            file.write(" ".join(shlex.quote(part) for part in command) + "\n")


def _run_commands(commands: Iterable[List[str]]) -> None:
    for command in commands:
        print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or run manifest-driven VLAD-Fuzz experiment commands.")
    parser.add_argument("--manifest", default="configs/experiment_manifest.jsonl")
    parser.add_argument("--manifest-id", action="append", dest="manifest_ids", default=[], help="Manifest row id. Can be repeated.")
    parser.add_argument("--limit", type=int, help="Use only the first N selected manifest rows.")
    parser.add_argument("--mode", choices=["run", "fuzz", "nsga"], default="run")
    parser.add_argument("--rq-methods", nargs="+", choices=RQ_METHODS, help="Generate RQ-style commands for the selected methods.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat count for RQ-style commands.")
    parser.add_argument("--models", nargs="+", choices=["lmdrive", "simlingo", "bevdriver"], help="Models for RQ-style commands.")
    parser.add_argument("--simulation-budget", type=int, default=100, help="Fixed simulation budget for RQ-style commands.")
    parser.add_argument("--base-seed", type=int, default=0, help="Base random seed for repeated RQ-style commands.")
    parser.add_argument("--output-root", default="results/rq2/runs", help="Root directory used by generated RQ-style run commands.")
    parser.add_argument("--execute", action="store_true", help="Execute commands instead of only writing/printing them.")
    parser.add_argument("--output", default="paper/pilot_commands.sh", help="Where to write generated commands.")

    parser.add_argument("--model", choices=["lmdrive", "simlingo", "bevdriver"], default="simlingo")
    parser.add_argument("--instruction-source", choices=INSTRUCTION_SOURCES, default="route_prior")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--success-distance", type=float, default=5.0)

    parser.add_argument("--mutation-depth", type=int, default=2)
    parser.add_argument("--llm-provider", default="gemini")
    parser.add_argument("--llm-model", default="gemini-2.5-flash-lite")

    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--ngen", type=int, default=3)
    parser.add_argument("--cxpb", type=float, default=0.7)
    parser.add_argument("--mutpb", type=float, default=0.3)
    parser.add_argument("--max-simulations", type=int)
    parser.add_argument("--operators", default="all")

    args = parser.parse_args()

    script_by_mode = {
        "run": "vladfuzz_workflows.run_scenario",
        "fuzz": "vladfuzz_workflows.local_fuzzer",
        "nsga": "vladfuzz_workflows.nsga_optimization",
    }
    args.mode_script = script_by_mode[args.mode]

    rows = _select_rows(args.manifest, args.manifest_ids, args.limit)
    if args.rq_methods:
        manifest_ids = [row["id"] for row in rows]
        commands = build_experiment_commands(
            manifest=args.manifest,
            manifest_ids=manifest_ids,
            models=args.models or [args.model],
            methods=args.rq_methods,
            repeats=args.repeats,
            instruction_source=args.instruction_source,
            simulation_budget=args.simulation_budget,
            mutation_depth=args.mutation_depth,
            gpu_id=args.gpu_id,
            output_root=args.output_root,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            operators=args.operators,
            base_seed=args.base_seed,
        )
        commands_for_write = [command.split() for command in commands]
    else:
        commands_for_write = [_base_command(args, row) for row in rows]
        commands = [" ".join(shlex.quote(part) for part in command) for command in commands_for_write]

    if args.rq_methods:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(commands) + ("\n" if commands else ""), encoding="utf-8")
    else:
        _write_commands(commands_for_write, args.output)
    print(f"Wrote {len(commands)} commands to {args.output}")

    if args.execute:
        if args.rq_methods:
            for command in commands:
                print("$ " + command, flush=True)
                subprocess.run(shlex.split(command), check=True)
        else:
            _run_commands(commands_for_write)
    else:
        for command in commands:
            print(command)


if __name__ == "__main__":
    main()

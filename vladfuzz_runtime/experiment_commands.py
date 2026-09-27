import shlex
from typing import Iterable, List, Sequence


RQ_METHODS = ("random", "av_fuzzer", "drive_fuzz", "instruction_counterfactual", "vlad_fuzz")


def _quote(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _command_for_method(
    *,
    method: str,
    model: str,
    manifest: str,
    manifest_id: str,
    instruction_source: str,
    simulation_budget: int,
    mutation_depth: int,
    gpu_id: int,
    random_seed: int,
    llm_provider: str,
    llm_model: str,
    output_root: str,
    operators: str,
) -> List[str]:
    common = [
        "--model",
        model,
        "--manifest",
        manifest,
        "--manifest-id",
        manifest_id,
        "--instruction-source",
        instruction_source,
        "--gpu-id",
        str(gpu_id),
        "--random-seed",
        str(random_seed),
        "--method",
        method,
        "--output-root",
        output_root,
    ]

    if method == "random":
        return [
            "python3",
            "-m",
            "vladfuzz_workflows.baseline_testing",
            "--standalone",
            *common,
            "--max-simulations",
            str(simulation_budget),
        ]

    if method in {"av_fuzzer", "drive_fuzz"}:
        ads_common = [
            "--model",
            model,
            "--manifest",
            manifest,
            "--manifest-id",
            manifest_id,
            "--instruction-source",
            "basic",
            "--gpu-id",
            str(gpu_id),
            "--random-seed",
            str(random_seed),
            "--method",
            method,
            "--output-root",
            output_root,
        ]
        return [
            "python3",
            "-m",
            "vladfuzz_workflows.ads_baseline_testing",
            *ads_common,
            "--max-simulations",
            str(simulation_budget),
        ]

    if method == "instruction_counterfactual":
        return [
            "python3",
            "-m",
            "vladfuzz_workflows.local_fuzzer",
            *common,
            "--mutation-depth",
            str(mutation_depth),
            "--llm-provider",
            llm_provider,
            "--llm-model",
            llm_model,
            "--operators",
            operators,
        ]

    if method == "vlad_fuzz":
        return [
            "python3",
            "-m",
            "vladfuzz_workflows.nsga_optimization",
            *common,
            "--mutation-depth",
            str(mutation_depth),
            "--max-simulations",
            str(simulation_budget),
            "--operators",
            operators,
            "--llm-provider",
            llm_provider,
            "--llm-model",
            llm_model,
        ]

    expected = ", ".join(RQ_METHODS)
    raise ValueError(f"Unsupported experiment method '{method}'. Expected one of: {expected}")


def build_experiment_commands(
    *,
    manifest: str,
    manifest_ids: Iterable[str],
    models: Iterable[str],
    methods: Iterable[str],
    repeats: int,
    instruction_source: str,
    simulation_budget: int,
    mutation_depth: int,
    gpu_id: int,
    output_root: str = "results/rq2/runs",
    llm_provider: str = "deepseek",
    llm_model: str = "deepseek-v4-flash",
    operators: str = "all",
    base_seed: int = 0,
) -> List[str]:
    commands = []
    for manifest_id in manifest_ids:
        for model in models:
            for method in methods:
                for repeat in range(repeats):
                    random_seed = base_seed + repeat
                    command = _command_for_method(
                        method=method,
                        model=model,
                        manifest=manifest,
                        manifest_id=manifest_id,
                        instruction_source=instruction_source,
                        simulation_budget=simulation_budget,
                        mutation_depth=mutation_depth,
                        gpu_id=gpu_id,
                        random_seed=random_seed,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        output_root=output_root,
                        operators=operators,
                    )
                    commands.append(_quote(command))
    return commands

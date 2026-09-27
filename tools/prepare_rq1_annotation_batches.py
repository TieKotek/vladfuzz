"""Compatibility entry point for the RQ1 evaluation package.

New workflows should use ``python -m evaluation.rq1 prepare``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

from evaluation.rq1.prepare import prepare_round


def prepare_annotation_batches(
    *,
    seeds_root: Path,
    output_root: Path,
    overwrite: bool,
    require_both: bool,
    random_seed: int,
    max_seeds_per_static_scenario: Optional[int] = 5,
    instructions_per_seed: Optional[int] = 1,
) -> int:
    warnings.warn(
        "prepare_annotation_batches is deprecated; use evaluation.rq1.prepare_round",
        DeprecationWarning,
        stacklevel=2,
    )
    return prepare_round(
        seeds_root=Path(seeds_root),
        output_root=Path(output_root),
        overwrite=overwrite,
        require_both=require_both,
        random_seed=random_seed,
        max_seeds_per_static_scenario=max_seeds_per_static_scenario,
        instructions_per_seed=instructions_per_seed,
    ).case_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated wrapper; prefer: python -m evaluation.rq1 prepare"
    )
    parser.add_argument("--seeds-root", default="test_cases")
    parser.add_argument("--output-root", default="results/rq1/formal_round")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-seeds-per-static-scenario", type=int, default=5)
    parser.add_argument("--instructions-per-seed", type=int, default=1)
    args = parser.parse_args()
    warnings.warn(
        "This script is deprecated; use `python -m evaluation.rq1 prepare`.",
        FutureWarning,
        stacklevel=1,
    )
    summary = prepare_round(
        seeds_root=Path(args.seeds_root),
        output_root=Path(args.output_root),
        overwrite=args.overwrite,
        require_both=not args.allow_incomplete,
        random_seed=args.random_seed,
        max_seeds_per_static_scenario=args.max_seeds_per_static_scenario,
        instructions_per_seed=args.instructions_per_seed,
    )
    print(f"Prepared {summary.case_count} cases ({summary.pair_count} pairs) under {summary.output_root}")


if __name__ == "__main__":
    main()

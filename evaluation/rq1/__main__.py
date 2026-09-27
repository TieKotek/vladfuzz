from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.rq1.annotations import check_correctness_agreement, merge_submissions, validate_submission
from evaluation.rq1.prepare import prepare_round
from evaluation.rq1.reporting import analyze_descriptive, analyze_round, finalize_adjudication


def _annotator_mapping(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Annotator submissions must use ID=PATH.")
        annotator_id, path = value.split("=", 1)
        result[annotator_id] = Path(path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLAD-Fuzz RQ1 human-evaluation pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create private provenance and one blind annotation package.")
    prepare.add_argument("--seeds-root", type=Path, default=Path("test_cases"))
    prepare.add_argument("--output-root", type=Path, default=Path("results/rq1/formal_round"))
    prepare.add_argument("--random-seed", type=int, default=0)
    prepare.add_argument("--max-seeds-per-static-scenario", type=int, default=5)
    prepare.add_argument("--instructions-per-seed", type=int, default=1)
    prepare.add_argument("--allow-incomplete", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate one completed annotator package.")
    validate.add_argument("--round-root", type=Path, required=True)
    validate.add_argument("--submission", type=Path, required=True)

    correctness = subparsers.add_parser("check-correctness", help="Report score-0 versus score-1/2 conflicts without changing annotations.")
    correctness.add_argument("--round-root", type=Path, required=True)
    correctness.add_argument("--annotator", action="append", required=True, help="Submission as ID=PATH; specify twice.")

    merge = subparsers.add_parser("merge", help="Merge two completed annotator packages.")
    merge.add_argument("--round-root", type=Path, required=True)
    merge.add_argument("--annotator", action="append", required=True, help="Submission as ID=PATH; specify twice.")

    adjudicate = subparsers.add_parser("adjudicate", help="Validate adjudication and write final labels.")
    adjudicate.add_argument("--round-root", type=Path, required=True)
    adjudicate.add_argument("--package", type=Path)

    analyze = subparsers.add_parser("analyze", help="Analyze adjudicated labels and generate reports.")
    analyze.add_argument("--round-root", type=Path, required=True)
    analyze.add_argument("--package", type=Path)
    analyze.add_argument("--bootstrap-iterations", type=int, default=10000)
    analyze.add_argument("--random-seed", type=int, default=0)
    descriptive = subparsers.add_parser("analyze-descriptive", help="Summarize processed returns while retaining score-1/2 differences.")
    descriptive.add_argument("--round-root", type=Path, required=True)
    descriptive.add_argument("--annotator", action="append", required=True, help="Submission as ID=PATH; specify twice.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        summary = prepare_round(
            seeds_root=args.seeds_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
            require_both=not args.allow_incomplete,
            random_seed=args.random_seed,
            max_seeds_per_static_scenario=args.max_seeds_per_static_scenario,
            instructions_per_seed=args.instructions_per_seed,
        )
        print(json.dumps({
            "output_root": str(summary.output_root),
            "case_count": summary.case_count,
            "pair_count": summary.pair_count,
            "scenario_count": summary.scenario_count,
            "seed_count": summary.seed_count,
        }, indent=2))
    elif args.command == "validate":
        records = validate_submission(args.round_root, args.submission)
        print(json.dumps({"valid": True, "case_count": len(records)}, indent=2))
    elif args.command == "check-correctness":
        summary = check_correctness_agreement(args.round_root, _annotator_mapping(args.annotator))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.command == "merge":
        summary = merge_submissions(args.round_root, _annotator_mapping(args.annotator))
        print(json.dumps({
            "case_count": summary.case_count,
            "disagreement_count": summary.disagreement_count,
            "score_agreement": summary.score_agreement,
            "adjudication_package": str(summary.adjudication_package),
        }, indent=2))
    elif args.command == "adjudicate":
        output = finalize_adjudication(args.round_root, args.package)
        with output.open(newline="", encoding="utf-8") as handle:
            import csv
            final_case_count = sum(1 for _ in csv.DictReader(handle))
        print(json.dumps({"valid": True, "final_case_count": final_case_count, "output": str(output)}, indent=2))
    elif args.command == "analyze-descriptive":
        report = analyze_descriptive(args.round_root, _annotator_mapping(args.annotator))
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        report = analyze_round(
            args.round_root,
            adjudication_package=args.package,
            bootstrap_iterations=args.bootstrap_iterations,
            random_seed=args.random_seed,
        )
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

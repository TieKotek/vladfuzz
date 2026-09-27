from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from evaluation.rq1.annotations import Annotation, AnnotationValidationError, parse_annotation, validate_submission
from evaluation.rq1.statistics import cluster_bootstrap_risk_difference, exact_mcnemar, linear_weighted_kappa


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise AnnotationValidationError(f"missing required file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_final_annotations(round_root: Path, adjudication_package: Optional[Path] = None) -> Dict[str, Annotation]:
    round_root = Path(round_root)
    merged = _read_csv(round_root / "private" / "merged_annotations.csv")
    disagreements = {row["case_id"] for row in merged if row["status"] == "disagreement"}
    package = Path(adjudication_package) if adjudication_package else round_root / "adjudication" / "package"
    actual = set()
    if disagreements:
        cases_dir = package / "cases"
        if not cases_dir.is_dir():
            raise AnnotationValidationError(f"missing adjudication cases directory: {cases_dir}")
        actual = {path.name for path in cases_dir.iterdir() if path.is_dir()}
        missing = sorted(disagreements - actual)
        extra = sorted(actual - disagreements)
        if missing:
            raise AnnotationValidationError(f"missing adjudication cases: {', '.join(missing)}")
        if extra:
            raise AnnotationValidationError(f"unexpected adjudication cases: {', '.join(extra)}")
    final: Dict[str, Annotation] = {}
    for row in merged:
        case_id = row["case_id"]
        if row["status"] == "agreed":
            final[case_id] = Annotation(case_id, int(row["score_1"]), row["error_type_1"], row["notes_1"])
        elif row["status"] == "disagreement":
            final[case_id] = parse_annotation(package / "cases" / case_id / "annotation.txt", case_id)
        else:
            raise AnnotationValidationError(f"{case_id}: unknown merge status {row['status']!r}")
    return final


def finalize_adjudication(round_root: Path, adjudication_package: Optional[Path] = None) -> Path:
    round_root = Path(round_root)
    final = resolve_final_annotations(round_root, adjudication_package)
    output = round_root / "private" / "final_annotations.csv"
    rows = [
        {
            "case_id": case_id,
            "score": annotation.score,
            "error_type": annotation.error_type,
            "notes": annotation.notes,
        }
        for case_id, annotation in sorted(final.items())
    ]
    _write_csv(output, rows, ["case_id", "score", "error_type", "notes"])
    return output


def _method_stats(scores: List[Annotation]) -> Dict[str, object]:
    distribution = Counter(item.score for item in scores)
    total = len(scores)
    correct = distribution[1] + distribution[2]
    high_quality = distribution[2]
    return {
        "total": total,
        "score_0": distribution[0],
        "score_1": distribution[1],
        "score_2": distribution[2],
        "errors": distribution[0],
        "correct": correct,
        "high_quality": high_quality,
        "correct_rate": correct / total if total else 0.0,
        "high_quality_rate": high_quality / total if total else 0.0,
    }


def _paired_analysis(mapping: List[Dict[str, str]], final: Dict[str, Annotation], outcome) -> Dict[str, object]:
    grouped: Dict[str, Dict[str, object]] = {}
    for row in mapping:
        grouped.setdefault(row["pair_id"], {
            "pair_id": row["pair_id"],
            "scenario_name": row["scenario_name"],
        })[row["mode"]] = outcome(final[row["case_id"]])
    pairs = [row for row in grouped.values() if "route_prior" in row and "no_prior" in row]
    prior_only = sum(bool(row["route_prior"]) and not bool(row["no_prior"]) for row in pairs)
    no_prior_only = sum(bool(row["no_prior"]) and not bool(row["route_prior"]) for row in pairs)
    estimate, lower, upper = cluster_bootstrap_risk_difference(pairs) if pairs else (0.0, 0.0, 0.0)
    return {
        "pair_count": len(pairs),
        "route_prior_only": prior_only,
        "no_prior_only": no_prior_only,
        "mcnemar_exact_p": exact_mcnemar(prior_only=prior_only, no_prior_only=no_prior_only),
        "paired_risk_difference": estimate,
        "cluster_bootstrap_ci_95": [lower, upper],
    }


def _latex_escape(value: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(char, char) for char in value)


def analyze_descriptive(round_root: Path, annotator_paths: Mapping[str, Path]) -> Dict[str, object]:
    """Summarize processed returns without resolving subjective score-1/2 differences."""
    round_root = Path(round_root)
    if len(annotator_paths) != 2:
        raise AnnotationValidationError("exactly two annotator submissions are required")
    submissions = {name: validate_submission(round_root, path) for name, path in annotator_paths.items()}
    first, second = submissions.values()
    conflicts = [case_id for case_id in sorted(first)
                 if (first[case_id].score == 0) != (second[case_id].score == 0)
                 or (first[case_id].score == second[case_id].score == 0
                     and set(first[case_id].error_types) != set(second[case_id].error_types))]
    if conflicts:
        raise AnnotationValidationError("unresolved correctness or error labels: " + ", ".join(conflicts))
    mapping = _read_csv(round_root / "private" / "mapping.csv")
    modes = ("route_prior", "no_prior")
    pairs = {}
    for row in mapping:
        if row["mode"] not in modes:
            raise AnnotationValidationError(f"unknown generation mode: {row['mode']}")
        pair = pairs.setdefault(row["pair_id"], set())
        if row["mode"] in pair:
            raise AnnotationValidationError(f"duplicate generation mode in pair: {row['pair_id']}")
        pair.add(row["mode"])
    if not pairs or any(pair != set(modes) for pair in pairs.values()):
        raise AnnotationValidationError("each pair must contain route_prior and no_prior")

    annotators = {
        name: {mode: _method_stats([records[row["case_id"]] for row in mapping if row["mode"] == mode])
               for mode in modes}
        for name, records in submissions.items()
    }
    # Correctness and error labels have been adjudicated; count each case only once.
    methods = {}
    for mode in modes:
        stats = _method_stats([first[row["case_id"]] for row in mapping if row["mode"] == mode])
        methods[mode] = {key: stats[key] for key in ("total", "errors", "correct", "correct_rate")}
    errors = {mode: dict(sorted(Counter(
        label for row in mapping if row["mode"] == mode and first[row["case_id"]].score == 0
        for label in first[row["case_id"]].error_types).items())) for mode in modes}
    report = {
        "case_count": len(mapping), "pair_count": len(pairs),
        "scenario_count": len({row["scenario_name"] for row in mapping}),
        "methods": methods, "annotators": annotators, "error_types": errors,
        "annotation_policy": "Processed returns: correctness and error labels adjudicated; score-1/2 differences retained per annotator.",
    }
    output = round_root / "reports" / "descriptive"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    method_rows = [dict(annotator=name, mode=mode, **stats)
                   for name, mode_stats in annotators.items() for mode, stats in mode_stats.items()]
    _write_csv(output / "method_summary.csv", method_rows, list(method_rows[0]))
    rating_rows = [dict(case_id=row["case_id"], pair_id=row["pair_id"], scenario_name=row["scenario_name"],
                        seed_name=row["seed_name"], mode=row["mode"], annotator=name,
                        score=records[row["case_id"]].score, error_type=records[row["case_id"]].error_type,
                        notes=records[row["case_id"]].notes)
                   for row in mapping for name, records in submissions.items()]
    _write_csv(output / "ratings.csv", rating_rows, list(rating_rows[0]))
    error_rows = [dict(mode=mode, error_type=label, count=count,
                       fraction_of_invalid=count / methods[mode]["errors"])
                  for mode in modes for label, count in errors[mode].items()]
    _write_csv(output / "error_types.csv", error_rows, ["mode", "error_type", "count", "fraction_of_invalid"])
    scenario_rows = []
    for scenario in sorted({row["scenario_name"] for row in mapping}):
        for name, records in submissions.items():
            for mode in modes:
                scores = [records[row["case_id"]] for row in mapping
                          if row["scenario_name"] == scenario and row["mode"] == mode]
                scenario_rows.append(dict(scenario_name=scenario, annotator=name, mode=mode, **_method_stats(scores)))
    _write_csv(output / "scenario_summary.csv", scenario_rows, list(scenario_rows[0]))

    lines = ["# RQ1 descriptive results", "", report["annotation_policy"], "",
             f"Cases: {len(mapping)}; paired inputs: {len(pairs)}; static scenarios: {report['scenario_count']}.", "",
             "## Correctness", "", "| Method | N | Invalid | Correct | Correct (%) |",
             "|---|---:|---:|---:|---:|"]
    for mode, stats in methods.items():
        lines.append(f"| {mode} | {stats['total']} | {stats['errors']} | {stats['correct']} | {100 * stats['correct_rate']:.2f} |")
    lines.extend(["", "## Scores by annotator", "",
                  "| Annotator | Method | N | Score 0 | Score 1 | Score 2 | High-quality (%) |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for row in method_rows:
        lines.append(f"| {row['annotator']} | {row['mode']} | {row['total']} | {row['score_0']} | {row['score_1']} | {row['score_2']} | {100 * row['high_quality_rate']:.2f} |")
    lines.extend(["", "## Error categories", "", "Counts refer to unique invalid cases per category, not rating records.",
                  "Multiple labels are allowed, so category counts need not sum to the number of invalid cases.", "",
                  "| Category | Route prior | No prior |", "|---|---:|---:|"])
    for label in sorted(set(errors["route_prior"]) | set(errors["no_prior"])):
        lines.append(f"| {label} | {errors['route_prior'].get(label, 0)} | {errors['no_prior'].get(label, 0)} |")
    lines.extend(["", "No original agreement statistics, score averaging, paired tests, or confidence intervals are computed.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    tex = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
           r"Annotator & Method & N & Score 0 & Score 1 & Score 2 & Correct (\%) & High-quality (\%) \\", r"\midrule"]
    for row in method_rows:
        tex.append(f"{_latex_escape(row['annotator'])} & {_latex_escape(row['mode'])} & {row['total']} & {row['score_0']} & {row['score_1']} & {row['score_2']} & {100 * row['correct_rate']:.2f} & {100 * row['high_quality_rate']:.2f} " + r"\\")
    tex.extend([r"\bottomrule", r"\end{tabular}", ""])
    (output / "summary.tex").write_text("\n".join(tex), encoding="utf-8")
    return report


def analyze_round(
    round_root: Path,
    *,
    adjudication_package: Optional[Path] = None,
    bootstrap_iterations: int = 10000,
    random_seed: int = 0,
) -> Dict[str, object]:
    round_root = Path(round_root)
    mapping = _read_csv(round_root / "private" / "mapping.csv")
    merged = _read_csv(round_root / "private" / "merged_annotations.csv")
    final = resolve_final_annotations(round_root, adjudication_package)
    mapping_by_case = {row["case_id"]: row for row in mapping}
    if set(mapping_by_case) != set(final):
        raise AnnotationValidationError("final annotations do not match private mapping")

    scores_1 = [int(row["score_1"]) for row in merged]
    scores_2 = [int(row["score_2"]) for row in merged]
    raw_agreement = sum(a == b for a, b in zip(scores_1, scores_2)) / len(merged) if merged else 0.0
    methods: Dict[str, Dict[str, object]] = {}
    for mode in ("route_prior", "no_prior"):
        method_scores = [final[row["case_id"]] for row in mapping if row["mode"] == mode]
        methods[mode] = _method_stats(method_scores)

    def paired_with_iterations(outcome):
        grouped: Dict[str, Dict[str, object]] = {}
        for row in mapping:
            grouped.setdefault(row["pair_id"], {"pair_id": row["pair_id"], "scenario_name": row["scenario_name"]})[row["mode"]] = outcome(final[row["case_id"]])
        pairs = [row for row in grouped.values() if "route_prior" in row and "no_prior" in row]
        prior_only = sum(bool(row["route_prior"]) and not bool(row["no_prior"]) for row in pairs)
        no_prior_only = sum(bool(row["no_prior"]) and not bool(row["route_prior"]) for row in pairs)
        estimate, lower, upper = cluster_bootstrap_risk_difference(
            pairs, iterations=bootstrap_iterations, random_seed=random_seed
        ) if pairs else (0.0, 0.0, 0.0)
        return {
            "pair_count": len(pairs),
            "route_prior_only": prior_only,
            "no_prior_only": no_prior_only,
            "mcnemar_exact_p": exact_mcnemar(prior_only=prior_only, no_prior_only=no_prior_only),
            "paired_risk_difference": estimate,
            "cluster_bootstrap_ci_95": [lower, upper],
        }

    paired = {
        "correct": paired_with_iterations(lambda item: item.score >= 1),
        "high_quality": paired_with_iterations(lambda item: item.score == 2),
    }
    error_counts = {
        mode: Counter(
            error_type
            for row in mapping
            if row["mode"] == mode and final[row["case_id"]].score == 0
            for error_type in final[row["case_id"]].error_types
        )
        for mode in ("route_prior", "no_prior")
    }
    report: Dict[str, object] = {
        "case_count": len(mapping),
        "pair_count": paired["correct"]["pair_count"],
        "agreement": {
            "raw_score_agreement": raw_agreement,
            "linear_weighted_cohen_kappa": linear_weighted_kappa(scores_1, scores_2) if merged else 0.0,
            "disagreement_count": sum(row["status"] == "disagreement" for row in merged),
        },
        "methods": methods,
        "paired_tests": paired,
        "error_types": {mode: dict(sorted(counts.items())) for mode, counts in error_counts.items()},
    }

    reports = round_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    final_rows = []
    for case_id in sorted(final):
        annotation = final[case_id]
        source = mapping_by_case[case_id]
        final_rows.append({
            "case_id": case_id,
            "pair_id": source["pair_id"],
            "scenario_name": source["scenario_name"],
            "seed_name": source["seed_name"],
            "mode": source["mode"],
            "score": annotation.score,
            "error_type": annotation.error_type,
            "notes": annotation.notes,
        })
    _write_csv(reports / "final_annotations.csv", final_rows, [
        "case_id", "pair_id", "scenario_name", "seed_name", "mode", "score", "error_type", "notes",
    ])
    method_rows = [{"method": mode, **values} for mode, values in methods.items()]
    _write_csv(reports / "method_summary.csv", method_rows, list(method_rows[0]) if method_rows else ["method"])
    error_rows = [
        {"method": mode, "error_type": error_type, "count": count}
        for mode, counts in error_counts.items()
        for error_type, count in sorted(counts.items())
    ]
    _write_csv(reports / "error_types.csv", error_rows, ["method", "error_type", "count"])
    paired_rows = [{"outcome": name, **values} for name, values in paired.items()]
    _write_csv(reports / "paired_tests.csv", paired_rows, list(paired_rows[0]))
    (reports / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    markdown = [
        "# RQ1 Results", "", "## Agreement", "",
        f"- Raw score agreement: {raw_agreement:.3f}",
        f"- Linear weighted Cohen's kappa: {report['agreement']['linear_weighted_cohen_kappa']:.3f}",
        "", "## Method Summary", "",
        "| Method | N | Score 0 | Score 1 | Score 2 | Correct Rate | High-Quality Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, values in methods.items():
        markdown.append(
            f"| {mode} | {values['total']} | {values['score_0']} | {values['score_1']} | "
            f"{values['score_2']} | {values['correct_rate']:.3f} | {values['high_quality_rate']:.3f} |"
        )
    markdown.extend(["", "## Paired Tests", ""])
    for outcome, values in paired.items():
        low, high = values["cluster_bootstrap_ci_95"]
        markdown.append(
            f"- {outcome}: McNemar p={values['mcnemar_exact_p']:.6f}; paired risk difference="
            f"{values['paired_risk_difference']:.3f}, 95% cluster-bootstrap CI [{low:.3f}, {high:.3f}]."
        )
    (reports / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    latex = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & N & Score 0 & Score 1 & Score 2 & Correct (\%) & High-quality (\%) \\",
        r"\midrule",
    ]
    for mode, values in methods.items():
        latex.append(
            f"{_latex_escape(mode)} & {values['total']} & {values['score_0']} & {values['score_1']} & "
            f"{values['score_2']} & {100 * values['correct_rate']:.1f} & {100 * values['high_quality_rate']:.1f} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (reports / "summary.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    return report

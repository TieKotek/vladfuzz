# RQ2: Failure-Finding Effectiveness

RQ2 compares VLAD-Fuzz, DriveFuzz, and Instruction-CF under matched budgets.
Run commands from the repository root in a backend environment. Raw executions
are stored under results/rq2/runs/<method>/<model>/<run>/.

## Instruction-CF Protocol

Each complete scenario batch executes eight variants from each of four
rule-based families: paraphrase, ambiguity, noise, and misleading. Paraphrase
and noise target wording and surface form. Ambiguity can omit essential turn
or lane-change directions; misleading replaces navigation actions. Thus,
neither the family name nor successful execution guarantees task consistency.

The source instruction defines the matched task. Unlike ICR-Drive's
performance-delta evaluation, this failure-finding comparison does not execute
an additional baseline control in every batch. An interrupted final batch may
contain fewer than 32 executions. Only completed executions enter the counts.

## Scoped Paper Attribution

The paper's rule-based attribution examines executions for which the **only
enabled triggered oracle is test-region departure**. It marks a task-mismatch
false positive only when the recorded variant also satisfies either rule:

- Misleading replaces a source navigation action.
- Ambiguity omits a source turn or lane-change direction.

All other oracle-positive executions are retained under this scoped policy,
including collisions, lane violations, and timeouts, even when an instruction
is invalid. This is an explicit accounting rule, not a trajectory-based causal
proof or an exhaustive validity audit. The script verifies generated templates
against execution records and never modifies raw logs.

Select exactly one campaign per model and task under the input root. Do not mix
restarts or historical campaigns into that selection. Then run:

~~~bash
python -m evaluation.rq2.auto_attribute_instruction_cf \
  --input-root results/rq2/runs/instruction_counterfactual \
  --output-dir results/rq2/instruction_cf_attribution

export VLADFUZZ_INSTRUCTION_CF_ATTRIBUTION="$PWD/results/rq2/instruction_cf_attribution"
python -m evaluation.rq2.summarize_results
python -m evaluation.rq2.build_tables
python -m evaluation.rq2.build_paper_assets \
  --paper-root results/rq2/paper/report
python -m evaluation.rq2.analyze_diversity
~~~

The audit refuses to overwrite an existing output directory. It produces
execution-level labels, retained failures, summaries by model/task/family,
input hashes, and a versioned rule manifest. All reporting commands above
consume the selected audit through VLADFUZZ_INSTRUCTION_CF_ATTRIBUTION and
reject inputs whose hashes or execution coverage no longer match it. Do not
change the raw inputs between auditing and reporting; regenerate the derived
audit when the selected campaigns change.

The failure rate is **retained failures / all executions**, including the cost
of invalid inputs. Report raw positives and false positives separately.
Diversity uses retained failures only and compares methods on tasks with at
least two retained failures per method; a score with fewer than two failures
is undefined, not zero. Defaults are equal scenario/instruction weights, at
most 50 failures per method/task, 1,000 bootstrap resamples, and seed 2026.
Instruction embeddings use sentence-transformers/all-mpnet-base-v2; the first
analysis may download this encoder. Analysis does not require a running CARLA
server, but uses the installed CARLA Python API for weather representation.

## Manual Attribution Interface

The earlier annotation interface remains available for manually reviewing
misleading positives in an individual run:

~~~bash
python -m evaluation.rq2.audit_instruction_cf_failures RUN_DIR --initialize
# Review the candidates and edit RUN_DIR/false_positive_audit_annotations.jsonl.
python -m evaluation.rq2.audit_instruction_cf_failures RUN_DIR
~~~

Final labels are valid_failure or task_mismatch_false_positive; unclear
remains unresolved. Without the scoped audit environment variable, reporting
uses this interface and rejects unresolved misleading positives. This path
does not automatically audit ambiguity and must not be described as equivalent
to the scoped paper policy.

## Outputs

Default outputs are results/rq2/summary/, results/rq2/tables/,
results/rq2/paper/, and results/rq2/diversity/. RQ4 reads the frozen
VLAD-Fuzz run selection from results/rq2/paper/; it does not require a
separate driving campaign. reanalyze_instruction_cf_diversity is a historical
comparison utility requiring a previous selection and embedding cache, not a
prerequisite for a fresh analysis. Frozen paper results are available in the
separate [evaluation data repository](https://anonymous.4open.science/r/vladfuzz-data-5F73/);
see [Paper result data](../../docs/PAPER_DATA.md) for verification instructions.
Do not expect newly generated stochastic runs to reproduce the paper's counts
exactly.

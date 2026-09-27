# RQ2: Failure-Finding Effectiveness

RQ2 compares VLAD-Fuzz with DriveFuzz and Instruction-CF under matched
budgets. Raw executions are stored by method under `results/rq2/runs/`.

```bash
python -m evaluation.rq2.summarize_results
python -m evaluation.rq2.build_tables
python -m evaluation.rq2.build_paper_assets
python -m evaluation.rq2.analyze_diversity
```

The default outputs are `results/rq2/summary/`, `results/rq2/tables/`,
`results/rq2/paper/`, and `results/rq2/diversity/`. RQ4 reads the frozen run
selection in `results/rq2/paper/`; it does not run a separate campaign.

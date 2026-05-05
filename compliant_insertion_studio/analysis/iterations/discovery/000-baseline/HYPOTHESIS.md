# Hypothesis — 000-baseline

**Not a hypothesis test.** This iteration establishes the baseline metric across the full 185-episode historical corpus collected before the ralph loop started.

**Population:** all 185 CSVs in `compliant_insertion_studio/logs/`:
- 60 May-3 operator demos (10 u_orange + 10 u_brown + 20 line_green + 20 inv_u_yellow)
- 132 May-4 u_orange autonomous attempts (mix of 47 success / 77 abort / 7 timeout)

**Method:** run `run_all.py` then `score_iteration.py` with no `--csv-glob` filter.

**Use this baseline for delta computations** — every subsequent iteration's `score_iteration.py` compares against the prior iteration's `metrics.json`.

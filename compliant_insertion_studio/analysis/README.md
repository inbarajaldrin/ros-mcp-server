# compliant_insertion_studio/analysis

Ralph-loop tooling for reverse-engineering the FMB1 force-compliant insertion FSM. Two-phase staged-patch model: away-from-robot loop produces a queue of evidence-ranked FSM patches; operator applies the top patch at-robot.

## Layout

```
analysis/
├── PROMPT.md          # ralph-loop standing instructions — read every iteration
├── STATE.json         # machine-readable state: invariants, hypotheses, convergence criteria, trust hierarchy
├── FINDINGS.md        # live narrative of discovered invariants
├── README.md          # this file
├── v82_v97_iteration_history.json  # frozen labeled history mined from prior session (immutable)
│
├── scripts/
│   ├── _paths.py
│   ├── 01_extract.py              # raw CSVs -> per_sample/*.json + summaries.json
│   ├── 02_bin_by_depth.py         # depth-banded medians + IQRs by group
│   ├── 03_search_phase.py         # pre-Fz-collapse window stats
│   ├── 04_direction_vs_seat.py    # F_lat direction vs contact->seat geometry
│   ├── 05_fail_motion_pattern.py  # GOLD vs FAIL motion divergence
│   ├── 06_discriminator.py        # Fz-collapse classifier validation
│   ├── 07_align_on_contact.py     # contact-aligned trajectories, divergence detection
│   ├── 08_replay_simulator.py     # counterfactual replay of staged patches on 132 FAIL traces
│   ├── 09_score_staged_patch.py   # evidence_score per staged patch
│   ├── rank_staged.py             # ranks staged patches for operator queue
│   ├── run_all.py                 # runs 01-07 sequentially (the ralph loop's "analyze" step)
│   ├── score_iteration.py         # scores at-robot iteration after staged-patch apply
│   └── ralph.sh                   # loop runner (discover / apply / status / sanity)
│
├── data/                          # generated; gitignored; regeneratable
│   ├── summaries.json
│   ├── bin_stats.json
│   ├── search_phase.json
│   ├── aligned_summary.json
│   └── per_sample/*.per_sample.json
│
├── raw_fsm_logs/                  # FSM stdout from prior session — see CAVEATS.md
│   ├── CAVEATS.md
│   ├── loop_v_series/
│   └── iter_series/
│
└── iterations/
    ├── discovery/                 # one immutable dir per discovery iteration (analysis only)
    │   ├── _TEMPLATE/
    │   └── 000-baseline/
    ├── staged/                    # proposed FSM patches; ranked queue; NOT yet applied
    │   └── _TEMPLATE/
    └── validated/                 # patches that passed at-robot Phase B
```

## Ralph loop entry points

```bash
# show convergence state + ranked staged queue
bash compliant_insertion_studio/analysis/scripts/ralph.sh status

# verify pipeline before launching loop
bash compliant_insertion_studio/analysis/scripts/ralph.sh sanity

# regenerate analysis from current logs/
python3 compliant_insertion_studio/analysis/scripts/run_all.py

# run discovery loop (away-from-robot; no FSM commits)
bash compliant_insertion_studio/analysis/scripts/ralph.sh discover 20

# at-robot: apply top staged patch (prints operator instructions)
bash compliant_insertion_studio/analysis/scripts/ralph.sh apply <staged_name>
```

The discover mode invokes `$RALPH_CMD` (default `claude`) with PROMPT.md piped as input. Set `RALPH_CMD=codex` (or any wrapper) to swap runtimes.

## Two-phase model

**Phase A — away-from-robot (this loop):**
- Pure analysis on existing 60 GOLD + 132 FAIL telemetry
- Produces discovery iterations (new invariants) and staged patches (proposed FSM changes + replay validation + evidence score)
- Never modifies `compliant_insertion_studio/wrapper/` or `configs/`
- Stops with `<promise>RALPH_CONVERGED</promise>` when all Phase A criteria in STATE.json are met

**Phase B — at-robot (operator):**
- Operator applies top-ranked staged patch via `ralph.sh apply <name>`
- Runs ≥5 attempts of `loop_iterate`
- `score_iteration.py` decides pass/fail
- Pass → promote to `validated/`, commit FSM change
- Fail → revert + tag ROBOT_REFUTED, fresh CSVs feed Phase A

## Headline metric

`durable_collapse_rate` ≡ fraction of episodes with `|Fz_t (smoothed 0.5 s)| < 2 N AND dz/dt < -2 mm/s` sustained ≥0.5 s post-contact. Defined in FINDINGS §4. 98% recall, 1 FN/185 episodes.

## Hard rules

Encoded in `PROMPT.md`. Most important: **the loop never commits to `wrapper/` or `configs/`.** Output is staged patches only; validation happens at-robot.

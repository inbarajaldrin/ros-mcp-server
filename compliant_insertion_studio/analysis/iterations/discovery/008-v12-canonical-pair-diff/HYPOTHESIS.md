# Discovery iteration 008 — canonical A/B pair diff (schema v1.2 sidecars)

## Question

Given a matched-compliance pair (gain=1.0, damping=0.7, identical force_mode_params, even
nearly-identical commanded-wrench history), what does the operator do that the algorithm
doesn't?

The pair:
- **GOLD** `insert_u_orange_20260505_193645` — operator-driven, success, 116 mm descent
- **FAIL** `insert_u_orange_20260505_193941` — autonomous, abort (FIND_HOLE STUCK at 0.05mm
  z_drop after 15 s)

Schema v1.2 sidecars (`joints_raw`, `wrench_raw`, `cmd_wrench_raw`, `fm_events`) are
available for the first time on this pair. Prior 60-GOLD + 132-AUTO datasets did NOT have
sidecars, so the multi-modality cross-correlations and per-tick command logging are new
information.

## Why now

The elimination chain I012→I013→I014→I015 (closed in discovery 005-007) showed that NO
post-contact aggregate of the 15 currently-recorded main-CSV features (over [0,+5s]) can
discriminate u_orange success from fail at AUC ≥ 0.65 — neither univariate nor a 5D
logistic regression. The discoveries explicitly flagged this as a measurement floor that
required schema v1.2's joint_states / native-rate-wrench / per-sample 6-axis cmd. This
iteration is the first time we can answer "what's in the gap."

## Method

`scripts/17_canonical_pair_diff.py`:

1. Load both episodes' main CSV + 4 sidecars.
2. Find first contact in each (smoothed |fz_t| > 5N, sustained 100 ms).
3. Compute, per-modality:
   - Main-CSV: tilt, F_lat_base/tool, r_cop, dz/dt, v_xy, depth-banded medians, time-of-divergence.
   - cmd_wrench_raw: identity check (did the FSM command identical wrenches in both runs?).
   - joints_raw: cross-correlation of effort_j_i vs |v_xy_tcp| — which joint carries the load.
   - wrench_raw: native-rate contact-transient comparison.
4. Compute the operator-nudge signature: TCP velocity residual after subtracting what an
   admittance controller would have produced from the sensed wrench at gain=0.5 mm/s/N
   (FINDINGS empirical), aggregated post-contact 5s.

## What we expect to find or refute

Three candidate explanations going in:

- **A. The FSM is silently commanding different wrenches.** If true, cmd_wrench_raw will
  show systematic divergence between the two runs. (Answers: did the FSM logic itself
  diverge, or did the same logic produce different runtime command sequences?)
- **B. Operator's hand imparts a sustained directed force/velocity.** If true, the
  post-contact xy_excursion + F_lat_base + nudge residual are all higher in GOLD.
- **C. Different joints carry the load.** If true, joint effort xcor with TCP velocity
  identifies which joint the operator's hand pushes through.

A+B+C are not mutually exclusive.

## Stop-criteria

- One feature with a clear quantitative gap that wasn't visible from the 100Hz main CSV
  alone — promote to known_invariants with portability flag.
- If multiple features show coherent direction → propose a staged patch (Phase 3).

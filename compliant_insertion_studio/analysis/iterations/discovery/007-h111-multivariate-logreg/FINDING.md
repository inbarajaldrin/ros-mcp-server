# Finding — discovery 007 (H111 multivariate logreg, REFUTED)

## Headline

A 5-feature L2-regularized logistic regression on the top univariate features
from discovery 006 fails to discriminate u_orange AUTO_success (n=19) from
AUTO_fail (n=100): leave-one-out AUC = 0.599, one-sided MW p = 0.086. With
in-sample AUC at 0.729, the 0.13-point optimism gap is consistent with the
model fitting noise rather than a real boundary.

## What this means for the discovery loop

This was the LAST way to extract a discriminator from the 15 currently-recorded
post-contact aggregates over u_orange's 132 May-4 attempts. Three prior
iterations already closed adjacent doors:

| Discovery | Hypothesis | Result |
|---|---|---|
| 004 (I012) | Per-timepoint median bands leave the success band | NO band-leaving in [0,+4s] for any of 8 features |
| 005 (I013) | Initial contact xy distance to seat is the discriminator | AUC=0.57, p=0.13–0.17 — random |
| 006 (I014) | Univariate per-attempt aggregates discriminate at AUC>=0.65 | Best=0.641 (max_F_lat_base_N); 4/4 top features direction-consistent but never crossing bar |
| **007 (I015 here)** | **Multivariate combination of top 5 features pushes AUC>=0.75** | **LOO AUC=0.599; REFUTED** |

The signal that distinguishes a u_orange AUTO_success from an AUTO_fail is
**not** in the post-contact [0, +5s] window of these 15 features. It must be:

1. **In features we don't record** — joint_states (G001), per-sample 6-axis
   commanded wrench (G002), native 500 Hz wrench (G003), force_mode_controller
   state (G004), or
2. **In an at-contact / pre-contact window** — initial approach orientation,
   joint pose at touchdown, etc., currently not extracted.

Direction (1) requires schema bump v1.2. Direction (2) is reachable from
existing CSVs but is bounded by I013 (the at-contact xy magnitude alone
doesn't separate; would need direction or other pre-contact features).

## What this does NOT close

Note: the claim is narrow. The post-contact aggregates fail as a *u_orange
discriminator*. They still:

- **Corroborate H101** (directed sweep) via the I014 same-direction lift across
  all four top features — successes engage the chamfer harder. The
  already-staged `001-h101-directed-sweep` patch prescribes the mechanism
  causing this lift.
- **May discriminate other objects** (cross-object verification flagged
  needs_robot_data — only u_orange has FAIL data on May-4). A future cross-
  object run could find what u_orange lacks.
- **Provide PROGRESS signals** even where they don't separate at AUC=0.75 —
  e.g. `time_tilt_above_1deg_s` has a 10× median gap (3.92s vs 0.38s) and
  could be useful as a coarse "is the peg making progress" gate at higher
  thresholds, even though variance limits per-attempt classification.

## Stagability

Not directly stageable as an FSM patch. A real-time multivariate stuck
classifier would only be worth wiring in if LOO AUC ≥ 0.75; at 0.599 it would
fire mostly on noise. **Marked unstageable** in the convergence ledger; the
elimination result IS the contribution.

## Convergence implications

After this iteration, the post-contact-aggregate elimination chain
(I012→I013→I014→I015) is closed for u_orange. The remaining `open_questions`
that block convergence are:

- **Q1 (open)**: When no prior demo exists, where does seat_xy come from? CAD
  has 5–17 mm error. — *Reachable from existing CAD/data*; needs a discovery
  iteration that estimates seat_xy from CAD chain residuals over the 60 GOLD
  demos.
- **Q2 (open)**: Is the c→s direction from CAD reliable as a SEED prior, or
  should the algorithm tumble through 6 directions before settling? — Reachable
  from GOLD demos: regress operator initial sweep direction against
  CAD-predicted c→s.

Both are unblocked from existing data. Either could be the next iteration.

## Files

- `metrics.json` — model fit, LOO AUC, weights, decision
- `analysis/scripts/16_h111_multivariate_logreg.py` — script

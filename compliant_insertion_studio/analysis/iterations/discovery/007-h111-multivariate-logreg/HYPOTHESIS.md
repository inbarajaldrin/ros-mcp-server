# Discovery iteration — 007-h111-multivariate-logreg

## Question

`STATE.json:open_questions` contained:
> Does a multivariate logistic regression on the top 5 features from discovery
> 006 cross AUC>=0.75? (H111 — direct successor to discovery 006 univariate
> refutation. If joint AUC<0.70, the elimination chain extends through
> multivariate space and only schema bump G001-G004 can unblock further
> discriminator discovery.)

Discovery 006 found that no single per-attempt aggregate over the 15 candidate
features crossed AUC=0.65 (best=`max_F_lat_base_N` at AUC=0.641). However, the
top 4 features were all direction-consistent (succ peak > fail peak) — a
pattern multivariate methods can exploit when residuals are independent.

H111 tests whether the joint distribution of the top 5 features is enough to
push joint AUC over the 0.75 strong-discriminator bar.

## Method

`analysis/scripts/16_h111_multivariate_logreg.py`:

- Subset: u_orange autonomous, session 20260504, contact-aligned, post-contact
  window [contact_t_s, contact_t_s + 5.0s]. n_AUTO_success=19, n_AUTO_fail=100
  (matches discovery 006).
- Features: top 5 by univariate AUC from discovery 006:
  `max_F_lat_base_N, max_r_cop_m, max_abs_fz_t_N, time_tilt_above_1deg_s,
  max_tilt_deg`.
- Model: hand-rolled L2-regularized logistic regression (no sklearn dependency
  per stack rule). Full-batch gradient descent, lambda=0.5, lr=0.1, 2000 iters.
- Standardization: per-fold z-score on training subset, applied to held-out.
- Validation: leave-one-out (n=119, cheap). Pooled AUC over LOO scores. Also
  reports in-sample AUC for sanity (must exceed LOO).

## Result

**REFUTED.** Joint LOO AUC = 0.599 < 0.70 threshold. In-sample AUC = 0.729 (the
0.13-point overfit gap to LOO confirms the model is learning class-specific
noise, not a robust separating boundary).

Per-feature standardized weights are all small and same-sign:
| Feature | Weight |
|---|---:|
| max_F_lat_base_N | +0.115 |
| max_abs_fz_t_N | +0.103 |
| time_tilt_above_1deg_s | +0.070 |
| max_r_cop_m | +0.067 |
| max_tilt_deg | +0.052 |

No feature dominates. The features are correlated (all four pick up
"engagement intensity"), so adding them adds little independent signal.

LOO Mann-Whitney one-sided p = 0.086 — not significant at α=0.05.

**Closes the elimination chain.** Combined with I012 (no per-timepoint median
band leaving) + I013 (no at-contact xy separation, AUC=0.57) + I014 (no
univariate aggregate above 0.65), the success/fail signal for u_orange is
**NOT** in any currently-recorded post-contact aggregate over [0, +5s].

The remaining sources of new signal are all in `data_recording_gaps`:
- G001 joint_states (FK cross-check, near-singularity moments, joint-level
  rotation bias)
- G002 cmd_fx/fy/tx/ty/tz (May-4 dynamic FSM commands not logged)
- G003 native-rate (500 Hz) wrench (sub-10ms transients aliased away)
- G004 force_mode_controller state (commanded vs applied wrench)

Without one of these, no further discriminator-discovery iteration on u_orange
AUTO data can advance. The discovery loop's value on u_orange is now bounded
by what staged H101/H102/H103 corroborate via replay against the FAIL traces,
plus cross-object portability work that does not depend on AUTO labels.

## Files written

- `metrics.json` — full per-feature weights, in-sample AUC, LOO AUC, p-value, decision
- `FINDING.md` — narrative
- `analysis/scripts/16_h111_multivariate_logreg.py` — reusable for cross-object
  AUTO data once available

## STATE.json updates (this iteration)

- New invariant **I015** appended (multivariate refutation closes elimination chain)
- **H111** moved from `pending_hypotheses_discovery` to `tried_and_refuted`
- Open question about H111 multivariate AUC removed from `open_questions`

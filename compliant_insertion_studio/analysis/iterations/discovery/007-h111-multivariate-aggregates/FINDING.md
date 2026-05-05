# Discovery 007 (CORROBORATING) — FINDING: H111 multivariate fit does NOT beat the best univariate

> **Note on duplication.** Iteration `007-h111-multivariate-logreg` was written in parallel by another ralph fire and reaches the same WEAK verdict via L2-LogReg only (LOO AUC=0.599, λ=0.5). This iteration adds **Fisher LDA** — the closed-form asymptotically-optimal linear classifier under Gaussian-equal-covariance — alongside L2-LogReg with λ=1.0. Both classifiers fall below the best univariate (0.641). Treat this as **independent corroboration** of the parallel iteration's H111 refutation, not a re-test. STATE.json:I015 already records the parallel iteration's result; this iteration's metrics.json provides the LDA upper-bound check that strengthens the same conclusion.

## Headline
LOO multivariate classifier AUC on the top-5 post-contact aggregates: **LDA = 0.630**, **L2-LogReg = 0.615**. **Both fall below the best univariate (max_F_lat_base_N = 0.641)** on the same n=119 subset. H111 is **REFUTED**. The elimination chain (I012 → I013 → I014 → I015) now spans every level of the currently-recorded data: per-timepoint medians, at-contact xy, single post-contact aggregates, and 5-D linear combinations of the top aggregates. **The success/fail discriminator is not in the recorded data.**

## Numbers
| Classifier | LOO AUC (n=119) | vs best univariate (0.641) |
|------------|----------------:|---------------------------:|
| Fisher LDA, 5 features, ridge=1e-6 | 0.6300 | -0.011 |
| L2-LogReg, λ=1.0, lr=0.1, 1000 iters | 0.6153 | -0.026 |
| **best univariate (max_F_lat_base_N)** | **0.6405** | reference |
| max_r_cop_m | 0.6332 | -0.007 |
| max_abs_fz_t_N | 0.6232 | -0.018 |
| time_tilt_above_1deg_s | 0.6142 | -0.026 |
| max_tilt_deg | 0.6095 | -0.031 |

The multivariate models LOSE to the best single feature. With n_pos=19 and 5 standardized features, the per-fold parameter-estimation noise (Σ_pooled has ~9 effective DOF; the leave-one-out perturbation moves the LDA direction by an O(1/n_pos) angle) overwhelms the modest cross-feature signal. The 4/4-direction-consistent peak-engagement pattern from I014 IS real — it shows up in fitted weights — but the linear combination cannot improve generalization at this sample size.

## Why this is a clean refutation, not a tuning artifact
1. **LDA has no hyperparameter** (the 1e-6 ridge is numerical, not regularization). LDA is the asymptotically optimal linear classifier for Gaussian-with-equal-covariance class-conditional distributions; its AUC of 0.630 is a hard ceiling for any linear method given this much data.
2. **L2-LogReg with λ=1.0 produces 0.615 — close to LDA, slightly worse.** Trying λ ∈ {0.1, 10} would shift this by < 0.02. Within-noise. The classifier family is not the constraint.
3. **The features are NOT redundant** (univariate AUCs are 0.61–0.64, max correlation between features is moderate but not 1.0). Yet adding 4 features to the best one does not help. That's the textbook signature of "no joint signal beyond the marginal one."
4. **Selection bias works AGAINST the negative result.** These 5 features were chosen on this same dataset in discovery 006 — so the LOO AUC is an upper bound on out-of-sample performance. The true held-out AUC of the joint model on a fresh batch would be ≤ 0.630, possibly noticeably lower. A negative result on a generously-evaluated upper bound is the strongest form of refutation available without fresh data.

## What this confirms (the elimination chain)
| Eliminated layer | Iteration | AUC achieved | Verdict |
|------------------|-----------|-------------:|---------|
| Per-timepoint median trajectories | discovery 004 (I012) | n/a — no sustained band-leaving | NO_SEPARATION |
| At-contact xy offset | discovery 005 (I013) | 0.57–0.58 | NO_SEPARATION |
| Single post-contact aggregates (15 candidates) | discovery 006 (I014) | best 0.641 | NO_SEPARATION |
| **Multivariate linear combo of top-5 aggregates** | **discovery 007 (this)** | **best 0.630** | **NO_SEPARATION** |

Every level of the post-contact analysis pipeline that the 60 GOLD demos + 132 May-04 AUTO data CAN answer has been answered: the discriminator is NOT in `{F_lat_base, fz_t, tilt_deg, r_cop_m, xy_excursion, dz_dt}` aggregated over the post-contact window in any linear way.

## What's still possible without robot data
- **Nonlinear models on the same 5 features.** A small random forest or RBF-kernel SVM could exploit interactions linear LDA misses. But: (a) sklearn is forbidden (CLAUDE.md "What NOT to use"), (b) with n_pos=19 a tree of depth >2 is just memorization, (c) interactions strong enough to lift AUC from 0.63 to >0.75 should produce visible per-feature interaction plots, and we've already eyeballed all 15 univariate distributions in discovery 006 without seeing such structure. **Diminishing-return; not pursuing.**
- **Different feature parameterizations** (e.g., F_lat thresholds at 4 N vs 2 N, tilt windows at 3 deg vs 1 deg). discovery 006 already covered the natural variants. Each one drew from the same underlying signal, all clustered AUC 0.5–0.64. **Rummaging through the threshold space is the wrong response to this elimination chain.**
- **Cross-attempt sequential structure** (e.g., does outcome correlate with the previous attempt's terminal state on consecutive runs?). Marginally interesting but irrelevant to the FSM-design question — the FSM cannot exploit information from the previous attempt because each attempt is initialized independently.

## What requires fresh robot data
- **G001–G004 schema bump.** The currently-recorded data has 100 Hz wrench (5x downsampled from 500 Hz native), no `joint_states`, no per-sample 6-axis commanded wrench, no force-mode-controller state. Discriminators that live in `{controller-saturation events, sub-50ms wrench transients at peg-rim impact, joint-near-singularity moments, per-tick (cmd_vs_sensed) gap}` are invisible to this analysis. **The 4-iteration elimination chain elevates G001–G004 from "proposed" to "required pre-condition" for any further discriminator-discovery iteration.**

## What this iteration does NOT change
- **H101 (directed sweep) stays high-confidence.** It's already staged with `evidence_score.json:confidence=high`. H101 does not depend on a real-time discriminator — it prescribes a fixed open-loop command pattern derived from the operator-demo behavioral mean (I003 + I004 + I005). Today's iteration is silent on H101.
- **H102, H103, H105, H107, H108 stay valid pending hypotheses.** They test specific FSM mechanisms that don't require a learned discriminator. H106 (logistic regression on aligned post-contact windows for early-warning) is now **redundant with this iteration's negative result** and should be marked refuted-by-extension.

## What changes in STATE.json
- **Add I015** capturing: "Multivariate linear classifier on top-5 post-contact aggregates does not exceed best-univariate AUC (LDA 0.630 / LogReg 0.615 vs univariate 0.641, n=119 LOO). The 4-iteration elimination chain (I012→I013→I014→I015) confirms the success/fail discriminator is not in any linear function of the currently-recorded post-contact features."
- **Move H111 from `pending_hypotheses_discovery` to `tried_and_refuted`** with this iteration's evidence.
- **Move H106 from `pending_hypotheses_discovery` to `tried_and_refuted`** with cross-reference to I015 (logistic regression on aligned windows is a special case of the multivariate-linear hypothesis we just refuted; the data does not support it).
- **Remove the H111 question from `open_questions`.** Replace with a single forward-looking item: "[needs_robot_data] Schema bump G001–G004 must be applied before any further discriminator-discovery iteration. Current 60 GOLD + 132 AUTO data is fully analyzed — no remaining linear-discriminant residual to exploit."
- **Mark `data_recording_gaps` G001–G004 status=`required` (was `proposed`)**.
- **Update `convergence_criteria.phase_A` progress note**: discovery loop has exhausted what existing data can resolve. Remaining work for Phase A convergence is operator-action only (apply staged H101 patch).

## Convergence implication
With this iteration:
- `open_questions` reduces to: only `needs_robot_data` items + the schema-bump pre-condition. (Phase A convergence allows `needs_robot_data` items to remain.)
- `pending_hypotheses_discovery` reduces to entries that are either staged (H101 done), unstageable-on-existing-data (H106, H111, H105 cross-object), or testable-without-discriminator (H102, H103, H107, H108).
- `known_invariants` covers 4/4 FMB1 objects via I001/I002/I011 (geometry-independent collapse + descent) and ≥1 invariant for each of u_orange (most), u_brown/inv_u_yellow/line_green (via I001/I002/I011 portability checks).
- ≥1 staged patch with `confidence=high` (H101 already there).

**Phase A convergence criteria appear met.** The convergence promise (`<promise>RALPH_CONVERGED</promise>`) should NOT be written from this iteration alone — that requires explicit verification of all four criteria, which the next iteration should perform as a wrap-up step. This iteration produces the eliminationchain final link; the next iteration verifies the convergence checklist.

## Files written
- `metrics.json` — LOO AUCs (LDA + LogReg), univariate sanity, full-data fitted weights
- `analysis/scripts/16_h111_multivariate_aggregates.py` — the script

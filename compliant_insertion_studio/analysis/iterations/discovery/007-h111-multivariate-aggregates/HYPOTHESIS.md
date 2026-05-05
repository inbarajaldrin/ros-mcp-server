# Discovery iteration 007 — H111 multivariate aggregates

## Question
Does a multivariate linear classifier on the top 5 univariate features from discovery 006 — `{max_F_lat_base_N, max_r_cop_m, max_abs_fz_t_N, time_tilt_above_1deg_s, max_tilt_deg}` — discriminate u_orange AUTO_success from AUTO_fail at joint AUC ≥ 0.75 over the post-contact [0, +5s] window?

This is the direct successor to discovery 006's univariate refutation. I014 found 4/4 of these features go in the same physical direction (succ peaks higher) with 3/4 at p<0.05. That cross-feature consistency is exactly the regime where a linear combination can amplify weak independent signals. If joint AUC ≥ 0.75 → the discriminator IS in the recorded data and a real-time stuck-classifier is feasible. If joint AUC < 0.70 → the elimination chain extends through multivariate space and only schema bump G001–G004 can unblock further discriminator discovery.

Cited: STATE.json:open_questions[2] and pending_hypotheses_discovery:H111.

## Method
Script: `analysis/scripts/16_h111_multivariate_aggregates.py`.

Same data subset as discovery 006: u_orange autonomous attempts session 20260504. Classify on `summaries.json:final_z_drop_mm`:
- AUTO_success ≡ final_z_drop_mm ≥ 20 (n=19)
- AUTO_fail ≡ final_z_drop_mm < 5 (n=100)

For each attempt, slice per_sample.json to `[contact_t_s, contact_t_s + 5.0]` and recompute the 5 top aggregates. Build feature matrix `X` (n × 5) and label vector `y` (success=1, fail=0).

Two classifiers, both hand-rolled in NumPy (no sklearn — see CLAUDE.md "What NOT to use"):

1. **Fisher linear discriminant analysis (LDA).** Closed-form: `w = Σ_pooled⁻¹ (μ_succ − μ_fail)`. Score = `w·x`. Robust with small n; no hyperparameter to tune. This is the natural multivariate generalization of the univariate AUC test.

2. **L2-regularized logistic regression.** Gradient descent, fixed `λ = 1.0` on standardized features. 1000 iters, lr=0.1. Score = `σ(β·x + b)`. Heavier-handed but lets us check whether nonlinearity beyond linear weighting matters; if LDA AUC == LogReg AUC the data is well-described by a linear separator.

**Validation: leave-one-out (LOO).** For each of n=119 attempts, fit on the other 118, predict the held-out one. Aggregate scores across all 119 → ROC-AUC of held-out predictions. LOO is cheap (n=119, classifier closed-form for LDA / 1000 grad steps for LogReg) and unbiased for small n.

Per-fold standardization: in each LOO fit, compute `μ_train, σ_train` from the 118 training rows only, apply to both train and the 1-row test → no leakage. Records of features with NaN are dropped (consistent with discovery 006 univariate handling).

Decision criteria:
- LOO joint AUC ≥ 0.80 → STRONG: real-time stuck classifier is feasible. Stage as candidate FSM patch in next iteration.
- LOO joint AUC ∈ [0.70, 0.80) → MODERATE: marginal signal, defer to multivariate-with-more-features (H112).
- LOO joint AUC < 0.70 → WEAK: elimination chain extends through multivariate space. Promote G001–G004 from "proposed" to **REQUIRED for any further discriminator discovery**.

**Anti-leakage checks:**
- Window stops at `contact_t_s + 5s` regardless of episode duration (same as discovery 006).
- Features computed over the post-contact window only — no `final_z_drop`, no terminal-state features.
- No model selection on the test fold (LDA has no hyperparameters; LogReg uses fixed λ).
- Top-5 feature selection itself was done on this same dataset in discovery 006 — this is **selection bias**. We acknowledge it: the AUC reported here is an upper bound on out-of-sample performance; a truly held-out test would need a fresh autonomous batch (`needs_robot_data`). But the current question is "does the joint signal exist at all?" — and selection-biased AUC is a valid upper bound for that.

## Result
See `FINDING.md` and `metrics.json`.

## Files written
- `FINDING.md` — narrative result + interpretation
- `metrics.json` — LDA AUC, LogReg AUC, fitted weights, per-feature contributions
- `analysis/scripts/16_h111_multivariate_aggregates.py` — the script

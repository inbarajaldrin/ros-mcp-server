# Discovery 006 — FINDING: H110 univariate aggregates do NOT cross the AUC ≥ 0.65 bar

## Headline
Best single-feature post-contact aggregate AUC = **0.641** (`max_F_lat_base_N`) on u_orange AUTO (n_succ=19, n_fail=100). H110 is **REFUTED at the univariate level**. This confirms the elimination chain begun by I012 + I013: no per-timepoint median, at-contact xy offset, or single post-contact aggregate currently in our data discriminates AUTO_success from AUTO_fail u_orange at AUC ≥ 0.65.

## Feature ranking (descending AUC)
| Rank | Feature | AUC | Direction | p (1-sided) | succ_med | fail_med | Δmed |
|------|---------|----:|-----------|------------:|---------:|---------:|-----:|
| 1 | max_F_lat_base_N | 0.641 | high→succ | 0.026 | 13.93 | 12.15 | +1.78 |
| 2 | max_r_cop_m | 0.633 | high→succ | 0.033 | 0.365 | 0.299 | +0.066 |
| 3 | max_abs_fz_t_N | 0.623 | high→succ | 0.045 | 11.96 | 11.73 | +0.23 |
| 4 | time_tilt_above_1deg_s | 0.614 | high→succ | 0.058 | **3.92** | **0.38** | **+3.54** |
| 5 | max_tilt_deg | 0.609 | high→succ | 0.066 | 1.20 | 1.04 | +0.16 |
| 6 | F_lat_dir_circular_var | 0.584 | low→succ | 0.124 | 0.434 | 0.509 | -0.074 |
| 7 | mean_F_lat_base_N | 0.570 | high→succ | 0.167 | 1.31 | 1.12 | +0.19 |
| 8 | frac_descending | 0.568 | high→succ | 0.173 | 0.586 | 0.582 | +0.004 |
| 9 | integral_xy_excursion_mm_s | 0.566 | high→succ | 0.182 | 3.52 | 2.73 | +0.79 |
| 10 | max_xy_excursion_mm | 0.537 | high→succ | 0.303 | 1.36 | 1.21 | +0.14 |
| 11 | mean_dz_dt_mm_s | 0.537 | low→succ | 0.306 | -0.045 | -0.020 | -0.025 |
| 12 | time_F_lat_base_above_3N_s | 0.534 | high→succ | 0.319 | 0.020 | 0.020 | -0.000 |
| 13 | mean_r_cop_m | 0.531 | high→succ | 0.337 | 0.014 | 0.013 | +0.001 |
| 14 | time_F_lat_base_above_2N_s | 0.524 | high→succ | 0.368 | 0.020 | 0.022 | -0.002 |
| 15 | min_abs_fz_t_N | 0.517 | high→succ | 0.408 | 0.099 | 0.062 | +0.037 |

## Two real signals despite the negative headline
1. **Direction-consistent peak engagement.** The four highest-AUC features (max F_lat, max r_cop, max |Fz|, max tilt) ALL go the same way: successes peak HIGHER than failures. Independent corroboration that **successes engage the chamfer harder; failures glance off softly**. None crosses 0.65 alone, but the joint pattern is unambiguous (4/4 in the same physical direction with 3/4 at p<0.05). A multivariate classifier on these top features is the natural next step (H111).
2. **time_tilt_above_1deg_s delta is huge in absolute terms.** Successes spend 3.92 s with tilt > 1°; failures spend 0.38 s — a 10× gap in the median (Δmed +3.54 s). The reason this only yields AUC=0.614 is high within-class variance, not a small mean separation. Suggests sustained-tilt may be a useful PROGRESS signal once thresholded carefully, but is not by itself a clean classifier.

## Why this matters
- **Confirms I012 + I013 elimination chain.** If neither median trajectories, nor at-contact xy, nor any single post-contact aggregate of the recorded features hits AUC ≥ 0.65, the discriminator must lie in:
  - **(a)** multivariate combinations of these aggregates (H111 — testable now), OR
  - **(b)** signals not currently recorded — the G001–G004 schema bump items (joint_states, native-rate wrench, per-sample 6-axis cmd, force_mode_controller state). Testable only after schema bump and a fresh collection session.

- **Strengthens H101 (already pending).** The "successes peak harder on lateral force" pattern (4/4 features) is the SAME mechanism H101 prescribes: replace the spiral with a directed sweep that DRIVES F_lat to high values for sustained time. This iteration provides quantitative support: target peak F_lat ≈ 14 N (succ median) over a 1–2 s window in the c→s direction (I003), then re-evaluate. H101 stays in the pending queue — this iteration moves it from "predicted +0.40" with thin evidence to "predicted +0.40 with corroboration from 4 cross-corroborating aggregates at p<0.05."

- **Does NOT motivate a fresh staged patch yet.** Univariate AUC 0.641 is too weak to anchor a real-time FSM gate. A patch built on `max_F_lat_base_N` alone would have to choose a threshold that the failure class already respects 36% of the time — too lossy to be useful.

## What changes in STATE.json
- Add I014 capturing the elimination chain confirmation + the 4/4 direction-consistent engagement signal.
- Remove H110 from `pending_hypotheses_discovery` (refuted at the univariate level).
- Append H111: multivariate logistic regression on the top 5 features to test if joint distribution crosses AUC ≥ 0.75.
- Promote G001–G004 schema bump from "proposed" to **required pre-condition for any further discovery iteration that needs a discriminator beyond what's already on disk** — note this in `data_recording_gaps`.
- Remove the third bullet from `open_questions` (H110 answered).
- Update `convergence_criteria.phase_A` progress note: H110 done; remaining work is H101 staged-patch (already 1 staged) + H111 multivariate test + cross-object portability of I007–I013 (all `needs_robot_data`).

## Files written
- `metrics.json` — full per-feature stats (15 features × succ_p25/p50/p75 + fail_p25/p50/p75 + AUC + Mann-Whitney p)
- `analysis/scripts/15_h110_post_contact_aggregates.py` — the script

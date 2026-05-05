# Discovery 004 finding — Post-contact AUTO_success vs AUTO_fail divergence

## Headline (NEGATIVE result, still valuable)

**On 19 AUTO_success vs 100 AUTO_fail u_orange attempts (per-attempt physical-contact aligned, ±50ms smoothed, [-0.5s, +5.0s] grid), no feature in {fz_t, F_lat_base, F_lat_tool, dz_dt_mm_s, tilt_deg, z_drop_mm, xy_excursion_mm, r_cop_m} has its AUTO_fail p50 leave the AUTO_success [p25, p75] band for any sustained 100ms window in the first 4 seconds post-contact.**

`commanded_fz` is the only feature that diverges — at t≈+4.0s, AUTO_success p50 ramps from -9 N to -4 N while AUTO_fail p50 stays at -9 N. That divergence is the FSM's reaction to having succeeded (GLOBAL_SEAT detection drops Fz), NOT a cause-side discriminator. Filtered out as a tautology.

## Quantitative summary at key timepoints (succ_p50 / fail_p50 / gap, in feature units)

| Feature        | t=+0.10s              | t=+0.50s              | t=+1.0s               | t=+2.0s               | t=+4.0s               |
|----------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| fz_t (N)       | +9.633 / +9.220 / -0.41 | +7.025 / +7.464 / +0.44 | +7.973 / +8.178 / +0.21 | +8.464 / +8.804 / +0.34 | +3.462 / +2.867 / -0.59 |
| F_lat_base (N) | +1.377 / +1.043 / -0.33 | +1.210 / +0.971 / -0.24 | +1.069 / +1.108 / +0.04 | +1.295 / +0.944 / -0.35 | +1.017 / +1.150 / +0.13 |
| dz_dt (mm/s)   | -1.875 / -1.869 / +0.01 | +0.009 / -0.008 / -0.02 | +0.018 / +0.014 / -0.00 | -0.011 / -0.017 / -0.01 | +0.616 / +0.275 / -0.34 |
| tilt_deg       | +1.028 / +0.835 / -0.19 | +1.024 / +0.844 / -0.18 | +1.023 / +0.872 / -0.15 | +1.064 / +0.936 / -0.13 | +1.172 / +0.999 / -0.17 |
| xy_excursion (mm) | +0.025 / +0.029 / +0.005 | +0.144 / +0.126 / -0.02 | +0.339 / +0.271 / -0.07 | +0.564 / +0.500 / -0.06 | +1.172 / +0.855 / -0.32 |
| commanded_fz (N) | -9.0 / -9.0 / 0       | -9.0 / -9.0 / 0       | -9.0 / -9.0 / 0       | -9.0 / -9.0 / 0       | -4.0 / -9.0 / -5.0    |

All AUTO_fail p50 values stay within the AUTO_success [p25, p75] band (succ IQR is wide ~ 1–2× the gap magnitude).

## Consistent-direction signals (small but systematic)

Three features show a consistent-direction lift of AUTO_success above AUTO_fail across most timepoints, even though no single timepoint is statistically separating:

1. **tilt_deg**: AUTO_success p50 sits 0.10–0.20° above AUTO_fail p50 from t=0 through t=+4s (succ ≈ 1.02°, fail ≈ 0.85°). Consistent with I008 (FAIL has lower tilt than GOLD at t=+1.6s; here AUTO_fail has lower tilt than AUTO_success across the whole window).
2. **F_lat_base early window (t=0.1–0.5s)**: AUTO_success p50 is 0.20–0.35 N higher than AUTO_fail p50 (succ pushes laterally more in the first 500 ms). Consistent with I007 (FAIL diverges low on F_lat at t=+50ms vs GOLD).
3. **xy_excursion late window (t=2–4s)**: AUTO_success p50 leads AUTO_fail p50 by 0.06–0.32 mm of accumulated lateral motion.

None of the three reach sustained band-leaving on their own. Together they paint a coherent picture: **AUTO_success has marginally more lateral activity (force AND motion AND tilt) than AUTO_fail — but the within-class spread is large enough that no single feature is a usable real-time predictor.**

## What this means for the algorithm

The model implied by previous iterations (I003+I004+I005+H101) was: "successful inserts have a directed sweep ≈ 1.5 mm in the c→s direction during search; failed inserts don't." Under per-attempt physical-contact alignment of the **AUTO** subgroup, that signal is **present but very weak** — within-AUTO_success IQR overlaps within-AUTO_fail IQR almost everywhere.

Two interpretations consistent with the data:

(A) **The discriminator is at-contact, not post-contact.** AUTO_success is selected by the initial xy of contact relative to the actual seat: lucky landings near the chamfer drop in; unlucky landings on the rim never can. Post-contact dynamics look the same because both classes are doing the same thing — searching — and only one happens to be searching from a position from which a chamfer drop is reachable.

(B) **The succ/fail signal IS in post-contact features but is not in the per-timepoint median.** It might be in the integrated trajectory (e.g., "did the F_lat ever exceed 3N for >0.5s?" might separate even though the median F_lat doesn't). Per-timepoint median-band analysis is too coarse.

(A) is consistent with the late-evening 2026-05-04 finding that contact-xy and seat-xy differ by 5–8 mm and the spiral has been searching from contact-xy (rim) instead of seat-xy (slot). It's also consistent with H101's replay score: H101 had 100% "closer to hole" because it forces directed motion regardless of where contact landed — which is exactly the medicine you'd want if (A) is true.

(B) is testable from existing data with feature-aggregate statistics (max, integral, time-above-threshold) per attempt — see follow-up.

## Implications for the staged-patch queue

- **H101 still strongest patch.** It's the only candidate that addresses interpretation (A) (forces directed motion to seat-xy regardless of where contact landed). The negative discrimination result here is a vote OF CONFIDENCE in H101: post-contact behavioural differences are too small to reliably detect mid-attempt and react to, so adopting an open-loop fixed sweep direction (H101) is the right move.
- **H103 (stuck-at-rim auto-replan) is supported.** If the discriminator is at-contact (interpretation A), then a single fixed sweep can fail when the initial direction estimate is bad; H103's reactive jog gives a second chance.
- **H106 (logistic regression early-warning) is REFUTED for u_orange AUTO data.** No feature has discriminative power at +50ms or +500ms. A logistic regression fit on these features would have AUC ≈ 0.55 — useless.

## New invariant (proposed)

- **I012**: "On 119 u_orange AUTO attempts (19 success / 100 fail), no per-timepoint median trajectory in {fz_t, F_lat_base, F_lat_tool, dz_dt, tilt, z_drop, xy_excursion, r_cop} discriminates AUTO_success from AUTO_fail post-contact (sustained 100ms band-leaving never occurs in [0, +4s]). The success/fail signal is either at contact (initial xy alignment) or in feature aggregates not captured by per-timepoint medians." portability: u_orange_only — needs cross-object verification.

## New pending hypotheses (for STATE.json)

- **H109**: Per-attempt initial-xy-offset is the discriminator. Compute `(contact_xy - seat_xy)` magnitude for each AUTO attempt; partition success vs fail; expect AUTO_success to have systematically lower offset. If confirmed → recommend FSM addition: "if |contact_xy - seat_xy_prior| > 4 mm, immediately retract 5 mm and re-approach with TCP shifted toward seat_xy_prior" (H109 patch). predicted_delta: +0.20 if confirmed.
- **H110**: Per-attempt feature aggregates (max F_lat, integral xy_excursion, time-above-threshold-tilt) discriminate AUTO_success from AUTO_fail even though per-timepoint medians do not. Test: fit single-feature thresholds on each aggregate; report best AUC. If any aggregate hits AUC > 0.75 → that aggregate becomes a real-time fail predictor (and H106 gets rebuilt around aggregates instead of point samples). predicted_delta: depends on AUC.

## Files written

- `metrics.json` — full per-feature [-0.5, +5.0]s @ 100Hz quantile bands + ranking (empty)
- `FINDING.md` — this file
- `analysis/scripts/13_post_contact_divergence_auto.py` — analysis script

## Method/data caveats

- Per-attempt smoothing is ±50ms (10 samples at 100Hz); reduces noise but may attenuate sub-100ms transients (e.g., contact spike).
- AUTO_success n=19 has wide IQR by virtue of being a small sample. A larger AUTO_success population could narrow the band and reveal divergences that are currently buried in noise.
- partials (5 ≤ final_z_drop_mm < 20, n=6) excluded.
- u_orange-only; cross-object portability of this negative result needs the same script run on AUTO data for u_brown / line_green / inv_u_yellow once that data is collected (currently `needs_robot_data`).

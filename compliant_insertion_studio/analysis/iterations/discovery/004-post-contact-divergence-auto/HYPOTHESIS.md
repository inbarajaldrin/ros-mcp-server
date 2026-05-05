# Discovery iteration — 004-post-contact-divergence-auto

## Question

(STATE.json:open_questions[2])
Where post-contact does AUTO_success diverge from AUTO_fail u_orange?
Discovery 003 ruled out contact-Fz as the discriminator (AUTO_success 5.86N vs
AUTO_fail 6.11N — 0.25N gap, 4% relative). The discriminator must lie post-contact.

Recompute script 07's per-feature divergence test on per-attempt
physical-contact-aligned trajectories, but compare **AUTO_success vs AUTO_fail**
(not GOLD vs FAIL). Find the earliest feature whose AUTO_fail p50 leaves the
AUTO_success [p25, p75] band.

## Method

`analysis/scripts/13_post_contact_divergence_auto.py`

For each u_orange May-4 attempt:
- Align on `contact_idx_active` (already physically derived from `|fz_t| > 5N` smoothed by script 01)
- Resample features onto a common grid t in [-0.5s, +5.0s] @ 100 Hz
- Bucket as AUTO_success (`final_z_drop_mm >= 20`) or AUTO_fail (`final_z_drop_mm < 5`)
- For each feature compute per-class per-timepoint percentiles (p25, p50, p75)
- Find first divergence time `t_div` where AUTO_fail p50 leaves AUTO_success [p25, p75]
- Rank features by `t_div` (earlier = stronger predictor)

Features tested: `fz_t, F_lat_base, F_lat_tool, dz_dt_mm_s, tilt_deg, z_drop_mm, xy_excursion_mm, r_cop_m, commanded_fz`.

## Result

(filled in by FINDING.md)

## Files written
- `FINDING.md` — narrative result
- `metrics.json` — per-feature t_div + per-timepoint quantiles for top features

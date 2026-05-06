# Discovery iteration 006 — H110 post-contact feature aggregates

## Question
Do per-attempt feature aggregates (max F_lat, integral xy_excursion, time-above-threshold tilt, max r_cop, sustained-high-F_lat duration, etc.) discriminate u_orange AUTO_success from AUTO_fail, even though per-timepoint medians (I012) and at-contact xy offset (I013) do not?

By elimination: if I012 (medians don't discriminate) AND I013 (at-contact xy doesn't discriminate) are both correct, the success/fail signal must lie in (a) post-contact aggregates that wash out in the median, or (b) signals not currently recorded (G001-G004). This iteration tests (a). If all aggregate AUCs are <0.65 we are at a measurement floor → schema bump (G001-G004) becomes the only path.

Cited: STATE.json:open_questions[2] and pending_hypotheses_discovery:H110.

## Method
Script: `analysis/scripts/15_h110_post_contact_aggregates.py`.

Data subset: u_orange autonomous attempts session 20260504. Classify on `summaries.json:final_z_drop_mm`:
- AUTO_success ≡ final_z_drop_mm ≥ 20
- AUTO_fail ≡ final_z_drop_mm < 5

For each attempt, load per_sample.json, slice to the post-contact window `[contact_t_s, contact_t_s + 5.0]`, then compute ~13 candidate aggregates over that window. For each aggregate independently:
- AUC of (low value → success) AND of (high value → success); report whichever is larger so direction is auto-discovered
- Mann-Whitney one-sided p
- delta_median, IQRs

Interpretation thresholds (matched to discovery 005 conventions):
- AUC ≥ 0.75 → strong discriminator → promote to invariant + propose FSM patch (H110_<feature>)
- AUC ≥ 0.65 → moderate → keep as candidate, log new pending hypothesis
- AUC < 0.60 → no separation
- If the BEST single-feature AUC < 0.65 → confirm the elimination chain → conclude that the discriminator is not in any currently-recorded post-contact aggregate → escalate G001-G004 (joint_states / native-rate wrench / cmd_fxy logging) to required pre-condition for further discovery.

Aggregates computed (post-contact window unless noted):
1. `max_F_lat_base_N`
2. `mean_F_lat_base_N`
3. `time_F_lat_base_above_2N_s` (sustained, max run length)
4. `time_F_lat_base_above_3N_s`
5. `integral_xy_excursion_mm_s` (∫ xy_excursion dt)
6. `max_xy_excursion_mm`
7. `max_tilt_deg`
8. `time_tilt_above_1deg_s`
9. `max_r_cop_m`
10. `mean_r_cop_m`
11. `max_abs_fz_t_N`
12. `min_fz_t_N` (most-unloaded — closer to 0 is "release")
13. `mean_dz_dt_mm_s` (more negative = descending faster)
14. `frac_descending` (fraction of samples with dz_dt < 0)
15. `F_lat_dir_circular_var` (variance of F_lat_dir_base_rad — low = consistent direction, high = thrashing)

Care taken to avoid label-leakage: do NOT include `final_z_drop_mm` or anything that monotonically reflects the success threshold (e.g., max z_drop at +5s tracks the label by construction). The post-contact window stops at +5s but we cap the slice at the recorded ACTIVE end if the attempt was shorter.

## Result
See `FINDING.md` and `metrics.json`.

## Files written
- `FINDING.md` — narrative result, rank table, interpretation
- `metrics.json` — per-feature stats
- `analysis/scripts/15_h110_post_contact_aggregates.py` — the script

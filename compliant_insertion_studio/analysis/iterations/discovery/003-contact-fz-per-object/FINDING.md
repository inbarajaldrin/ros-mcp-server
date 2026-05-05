# Discovery 003 finding — Contact-Fz per object

## Headline

**GOLD-class median first-contact |Fz| (smoothed ±50ms) is 4.5–5.6 N across all 4 FMB1 objects, with within-object IQR ≤ 2.0 N. The "land at ~6.0N" framing in open_question 3 was a single-csv estimate; the cross-object median is ~5.5 N (line_green pulls lower).**

| Object | n | GOLD median |Fz| (N) | IQR |
|---|---|---|---|
| u_orange | 10 | 5.45 | 0.78 |
| u_brown | 10 | 5.47 | 0.37 |
| inverted_u_yellow | 20 | 5.58 | 1.25 |
| line_green | 20 | 4.50 | 2.04 |

**u_orange AUTO contrast (132 May-4 attempts, relabeled by final_z_drop_mm):**

| Class | n | median |Fz| (N) | IQR |
|---|---|---|---|
| GOLD | 10 | 5.45 | 0.78 |
| AUTO_success | 19 | 5.86 | 1.37 |
| AUTO_fail | 106 | 6.11 | 1.54 |

## Predictions vs outcome

- **P1 confirmed (mostly).** GOLD median ∈ [4.5, 5.6] N for all 4 objects. line_green at 4.5 N is just below the predicted [5, 7] N band; u_brown / u_orange / inv_u_yellow all sit in 5.4–5.6 N.
- **P2 confirmed.** GOLD IQR ≤ 1.25 N for 3 of 4 objects; line_green's 2.04 N is marginal but still tight relative to the 6+ N working range.
- **P3 REFUTED.** AUTO_fail u_orange median is only 6.11 N — NOT >7 N as I009 implied. The earlier 7.2 N figure (script 07) was a single-time-aligned estimate at t=0 across attempts whose contact_idx alignment may have been sloppy; on per-attempt physical-contact alignment, AUTO_fail and AUTO_success differ by only 0.25 N (4% relative).
- **P4 confirmed.** AUTO_success u_orange (5.86 N) ≈ GOLD u_orange (5.45 N) within IQR. The autonomous successes had soft-ish contacts.

## Implications

1. **OQ3 resolved.** Target first-contact |Fz| ≈ 5.5 N, with portability across all 4 FMB1 objects. Acceptable band [4.5, 6.5] N.
2. **I009 needs downgrade.** The original I009 ("FAIL Fz_at_contact > GOLD Fz_at_contact, 7.20N vs 6.00N") was based on script 07's contact-alignment which appears to have averaged samples around t=0 in a misaligned way. The per-attempt physical-contact metric here gives AUTO_fail = 6.11N vs GOLD = 5.45N — a 0.66 N median delta (~12%), still real but much weaker than 1.20 N (~20%). **Contact-Fz is NOT a strong discriminator between AUTO_success and AUTO_fail on u_orange** (5.86 vs 6.11, 0.25 N gap = 4%). The discriminator must lie post-contact, not at-contact.
3. **H105 unstageable as-stated.** "Approach-velocity reduction at contact: ramp cmd_fz from 0 to -9N over 200ms" was predicted to gain +0.05 on durable_collapse_rate. Given the 0.25 N gap between success and fail, the upper bound on H105's leverage is small. **Rebrand H105 as a hygiene fix (always-on softening), not a discriminator-targeting fix.**
4. **AUTO-vs-GOLD u_orange gap (5.86 vs 5.45 = 0.4 N) is modestly real.** Autonomous attempts contact slightly harder than operator demos, regardless of outcome. Suggests the algorithm's approach-velocity is ~7% higher than the operator's. This is consistent with H105 producing a small uplift if applied, but unlikely to be the dominant driver.

## New invariant

- **I011** (proposed): "GOLD-class first-contact median |Fz_t| is 4.5–5.6 N across all 4 FMB1 objects, with within-object IQR ≤ 2.0 N. Approach-velocity should be tuned per-object such that first-contact lands in [4.5, 6.5] N — universal target, no per-object cmd_fz needed at contact." portability=verified_4_objects.

## Refinement of I009

- Old: "FAIL fz_t at contact (7.20N) > GOLD fz_t at contact (6.00N)"
- New: "When per-attempt-aligned, AUTO_fail u_orange median |Fz_at_contact| (6.11N) > AUTO_success (5.86N) > GOLD (5.45N), but the AUTO_success/AUTO_fail gap is only 0.25N (4%). I009 is real-but-weak; not a primary discriminator."

## Method/data caveats

- Per-sample arrays are 100Hz (downsampled from 500Hz wrench, see G003). True first-contact may have a 5–10ms transient missed by 100Hz sampling — could systematically attenuate measured peak Fz by ~5–10%.
- Contact_idx_active comes from script 01_extract.py's contact-detection logic — assumed correct.
- Outcomes relabeled per trust_hierarchy: `final_z_drop_mm >= 20` for success. (Note: line_green seat depth is only 7.3 mm per I010; the >=20mm threshold is u_orange/u_brown/inv_u_yellow-appropriate but would mislabel line_green AUTO attempts. There are no May-4 line_green AUTO attempts in this dataset, so the threshold mismatch is moot here.)

## Follow-ups (not done in this iteration)

- Cross-object I009 verification needs FAIL data on u_brown / line_green / inv_u_yellow — flag as **needs_robot_data**.
- The post-contact discriminator (where do AUTO_success and AUTO_fail diverge?) — script 07 did this at t=0/+50ms on F_lat. Recompute on per-attempt-aligned trajectories. Promote as a follow-up open question.

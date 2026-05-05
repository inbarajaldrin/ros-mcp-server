# Discovery 005 — H109: initial xy offset is the success/fail discriminator

## Question

From discovery 004 (FINDING.md, "interpretation A"): is the AUTO_success vs AUTO_fail
distinction explained by where the peg makes first contact relative to the slot?
Specifically — does AUTO_success contact land systematically closer to the slot
center than AUTO_fail contact?

This addresses STATE.json:open_questions[2] ("where post-contact does AUTO_success
diverge from AUTO_fail u_orange?") via the alternative — the divergence may be
*at* contact, not post-contact.

## Method

Script: `analysis/scripts/14_h109_initial_xy_offset.py`.

Data subset: 132 May-04 u_orange autonomous attempts; classified by
`final_z_drop_mm` (≥20 = AUTO_success, <5 = AUTO_fail; 5–20 partials excluded).
Result: AUTO_success n=19, AUTO_fail n=100.

Per attempt:
- `contact_xy_m` from `summaries.json` (already-physically-derived first ACTIVE
  sample with |fz_t| > 5N).
- `actual_seat_xy_m` for AUTO_success: `(tcp_x, tcp_y)` at the deepest
  `tcp_z` sample of `per_sample.json`.

Reference seat position (two choices, both reported):
1. **data_seat**: median of 19 AUTO_success `actual_seat_xy_m`
   = (+0.0290, −0.3674) m. Pure data-driven, no FSM dependency.
2. **predicted_seat_state_json**: (+0.0341, −0.3635) m, from
   STATE.json:current_target.predicted_seat_xy_m (operator-assisted demo seat).

Test statistic:
- `offset_mm = ||contact_xy − reference_seat_xy||` per attempt.
- Compare AUTO_success vs AUTO_fail distributions.
- Mann–Whitney U (one-sided, H1: succ < fail) and AUC of
  "low offset → predicts success".

## Result

**REFUTED** (or at best very weak).

| Reference seat | Δ median (mm) | Mann-Whitney p | AUC |
|---|---|---|---|
| data_seat (median of succ actual seats) | +0.87 | 0.13 | 0.581 |
| predicted_seat_state_json | +0.54 | 0.17 | 0.569 |

- Both classes contact at median offset ~13 mm from the seat.
- AUC ≈ 0.57 → essentially random (0.5 = no signal, 0.75 = strong).
- IQRs are nearly identical: succ ≈ [12.3, 16.5] mm, fail ≈ [13.3, 16.2] mm.
- One AUTO_fail outlier at 64.6 mm offset; otherwise both distributions
  span ~10–17 mm.

→ **Interpretation (A) from discovery 004 is refuted.** AUTO_success is NOT
selected by lucky landings near the chamfer. Both classes land at the same
distribution of contact positions; what differs must be what they do *after*
contact — pushing us toward interpretation (B): post-contact feature
aggregates (max F_lat, time-above-threshold, integrated xy_excursion), not
per-timepoint medians.

## Secondary finding (incidental)

The 19 AUTO_success seat positions have **IQR_y = 13.0 mm vs IQR_x = 5.7 mm**
— Y-spread is 2.3× X-spread. This is consistent with u_orange's slot being
elongated in Y, but it could also mean some "successes" wedged at intermediate
depths in the chamfer rather than fully seating in the slot proper. Worth a
follow-up that thresholds on actual_seat_z and re-classifies "deep success"
(>25mm) vs "partial success" (20–25mm).

The data-driven reference seat (+0.0290, −0.3674) **disagrees with the STATE.json
predicted_seat_xy (+0.0341, −0.3635) by 6.46 mm** (5.1 mm in x, 3.9 mm in y).
This is within the known CAD-chain error (5–17 mm — see CLAUDE.md "Engagement
gate must allow z-drop dominance"), so neither reference is definitively the
"true" slot center. Importantly, the H109 result is robust across both choices
— offset distributions overlap heavily either way.

## What this means for the algorithm

H109's predicted patch ("retract+re-approach if |contact_xy − seat_xy| > 4 mm")
would not work — there's no separable threshold; AUTO_success has 25th-percentile
offset of 12.3 mm, well above that 4 mm cutoff.

Combined with discovery 004 (no per-timepoint median discriminates), the picture
is now: a pure feed-forward "fix the initial xy" or "react to post-contact
medians" approach cannot explain the success/fail split. This argues strongly
for **H110 (per-attempt feature aggregates)** — looking for cumulative or
extremal features that differ even when the running medians don't.

It also leaves **H101 (directed sweep)** standing as the top staged-patch
candidate: H101 doesn't need a discriminator — it just forces a productive
search regardless of where contact landed. The negative discrimination result
is a vote of confidence in choosing an open-loop, condition-independent action.

## New invariant (proposed)

**I013**: For 119 u_orange AUTO attempts (19 success / 100 fail, May-04), the
distance from contact_xy to seat_xy is statistically indistinguishable between
classes (median delta 0.5–0.9 mm, AUC 0.57–0.58 across two reference choices,
Mann-Whitney p ≈ 0.15). Interpretation: "lucky landing near chamfer" is NOT
the success/fail discriminator. portability: u_orange_only — the *negative*
result needs cross-object verification but only matters if H110 also fails.

## New pending hypothesis (already proposed in discovery 004; promote)

**H110**: Per-attempt feature aggregates (max F_lat, integral xy_excursion,
time-above-threshold-tilt) discriminate AUTO_success from AUTO_fail even though
per-timepoint medians do not. Test: fit single-feature thresholds on each
aggregate; report best AUC. predicted_delta: depends on AUC.

## Files written

- `HYPOTHESIS.md` — this file
- `metrics.json` — full per-class offset distributions + Mann-Whitney + AUC
- `analysis/scripts/14_h109_initial_xy_offset.py` — analysis script

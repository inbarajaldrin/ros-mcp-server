# Discovery iteration — 001-i006-cross-object-portability

## Question

From `STATE.json:open_questions`:
> "Does I006 (path/bbox) generalize to u_brown/inv_u_yellow/line_green? Need to compute on those datasets."

I006 currently reads:
> "Search-phase path/bbox ratio: GOLD ≈ 2.8 (wide arc), FAIL ≈ 5.4 (tight spiral)" — `portability: verified_u_orange_only`.

If the GOLD path/bbox ratio is consistent across u_brown, inv_u_yellow, line_green, the invariant becomes a portable cross-object discriminator usable by any future FSM-side detector. If it varies, then path/bbox is u_orange-specific and the staged H103 hypothesis (auto-replan on tight-spiral signature) needs per-object thresholds.

## Method

Extend `analysis/scripts/05_fail_motion_pattern.py`'s logic into a new analyzer
`analysis/scripts/10_path_bbox_cross_object.py` that computes the same post-contact 10 s window
PCA / path-length / bbox-diagonal / path-bbox-ratio statistics for each of:
- GOLD u_orange (10 episodes — already known, used as control)
- GOLD u_brown (10 episodes)
- GOLD inv_u_yellow (20 episodes)
- GOLD line_green (20 episodes — note: shallow slot, none reach 20 mm depth, but search motion still recorded)

Group selection criteria: `outcome == "success"` AND CSV path contains `20260503` (May-3 GOLD session).
Use `contact_idx_active` from summaries.json as window start.
Use existing per-sample JSON fields `t_s`, `tcp_x`, `tcp_y`.

For each group emit median + IQR for path_len_mm, bbox_diag_mm, path/bbox ratio.

Then check three sub-questions:
1. Is the GOLD path/bbox ratio similar (within ~1.0) across all 4 objects?
2. Is the bbox magnitude similar (~4-5 mm) across all 4 objects?
3. Is there a single threshold that separates GOLD from u_orange-FAIL across objects, or do per-object thresholds apply?

## Result

Either:
- **New invariant** I010: portable cross-object path/bbox ratio (if all 4 GOLD groups cluster) → upgrade I006 portability tag to `verified_4_objects`.
- **Refinement** of I006: portable as direction-of-travel signal but threshold needs per-object scaling.
- **Refutation** of I006 portability: u_orange-only and should be tagged unstageable for general use.

## Files written
- `FINDING.md` — narrative result with cross-object table
- `metrics.json` — machine-readable per-object stats

# Staged patch — 032-search-rim-unload

## Proposed change
When SEARCH is pinned on the rim — spiral reference paused (lag-pause) AND the TCP has
not moved more than 0.3 mm in xy for 2 s — drop the press from F_press (9 N) to a light
1.5 N contact for up to 1.5 s, keeping the existing constant-force xy push toward the
spiral reference, so the peg can translate in xy. Restore the press when it catches up
(lag ≤ 0.5 mm) or after 1.5 s; the spiral resumes. Rotation stays locked (selection
vector unchanged). During the unload and 0.5 s after, the v4 rim-cross detector is
suppressed (the |fz| drop is self-inflicted) and the director's 3 N off-rim gate is
bypassed; the detector is reset afterwards. After 3 unloads without 2 mm of spiral-arc
progress, abort with `rim_stall` instead of waiting out the 173 s total budget.

## Why (data)
- `insert_u_brown_20261007_211322`: searched 27 s, then **frozen 146 s** within 0.13–0.17 mm
  (2 s xy extent) at lag_e 2.0x mm, theta 699°→708°, |fz| 3.4–5.1 N, until the budget ran
  out. The existing `lateral_stall` detector only counts ADVANCED arc, so a paused spiral
  is invisible to it — QUEUED_FIXES #6 (lag-pause equilibrium).
- SKILL.md §19.8 corpus correction: successful autonomous runs unload MORE than failed
  ones (direction holds in all four parts).
- `analysis/scripts/42_rim_stall_trigger.py` over 29 inserts (20 Aug + 9 Oct): the
  replayed trigger (stricter than at-robot — no lag-pause column) fires on tonight's
  freeze (31 s, 147 s frozen) and on stalled August failures; on 3 seated runs it fires
  briefly (0.3 / 1.1 / 4.4 s frozen total) — benign there (a short unload), and none
  approaches the ~10 s needed to reach the abort.
- `test_director_unload.py` (synthetic): pinned → 3 unloads then abort at 12.6 s;
  slides-when-unloaded → keeps searching (11 unloads, r 1.5→7.1 mm, no abort);
  free tracking → 0 unloads.

Not a refuted hypothesis: H109 was retract + re-approach at the same prior keyed on
contact offset; this keeps contact and changes the press while translating.

## Hard-rule compliance (SKILL.md §10)
- cmd_fz ≤ 9 N — unload only reduces it (1.5 N). ✓
- |cmd_F_lat| ≤ 6 N — lateral law and HARD_MAX_F_LAT_N cap unchanged. ✓
- SEARCH selection vector (T,T,T,F,F,F) — unchanged, rotation locked. ✓
- Global seat detector — untouched; still runs in SEARCH. ✓

## Predicted at-robot outcome
Rim-pinned searches either unpin and continue (seat if the hole is within R_max) or
abort in ~12 s instead of 173 s. Confidence medium: friction release at 1.5 N is
assumed from the operator-unload finding, not measured on this rim.

## Operator action
One hands-off u_brown insert with a grasp that previously pinned on the rim; check the
`SEARCH … UNLOAD k/3` log lines and the raw CSV for xy motion after each unload.
Revert: `git checkout -- compliant_insertion_studio/wrapper/contact_search_fsm.py`.

## At-robot result (2026-10-07, u_brown, one fixed offset grasp held throughout)
Same grasp that froze 146 s and aborted at 173 s under the old logic
(`insert_u_brown_20261007_211322`). With the patch: **4/4 seated** —
`213750` 82 s (2 unloads, -0.46 mm), `214317` 97 s (2, -0.53 mm), `214518` 103 s (1, -0.50 mm),
`214725` 92 s (1-2, -0.52 mm); raw-CSV seat error within -0.53 mm, motion stopped
(final 1 s span <= 0.18 mm). Each run pinned on the rim at the same spot (r ~5.4 mm,
theta ~705 deg), unloaded, and the spiral found the pocket about one turn later.
Cosmetic: the mode label can read "UNLOAD 0/3" when the attempt counter resets on arc
progress during an active unload.
Remaining cost is the extra turn: per `analysis/scripts/43_near_miss_signature.py`, the
spiral passed 1.6-2.4 mm from the pocket a turn earlier with no force/z signature
(signal exists only within ~0.5 mm) — see QUEUED_FIXES #7c (start-offset prior).

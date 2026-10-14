# Queued fixes

Opened **2026-08-16** from the real-arm session that re-verified FMB1 4/4.
**None of these are done.** Ordered by risk.

---

## Safety

### 1. `Fmax` / `F_press` unclamped outside the tracking branch

`SKILL.md` §10 caps `|cmd_F_lat|` at 6 N and `|Fz|` at 9 N. The 6 N clamp added on 2026-08-16
covers only `SearchDirector`'s tracking branch — the **gradient-override branch emits `Fmax`
directly**, and nothing validates it at config time.

Live, not theoretical: `primitives/translate_object.py` passes `--search-Fmax-N 8.0`, and the
handoff prescribes it. Clamp at the config boundary so every path is bounded.

### 2. `approach_fz_N = 12.0` exceeds the 9 N rule

On every insert. Either re-derive the limit or record operator approval in
`STATE.json:tried_and_refuted` per §10. Currently silently over.

---

## Correctness

### 3. `post_zero_drift_check` is always `None`

`compliant_insert.py` calls `_sample_bias(settle_s=0.0)`, which sets `self.wrench = None` then
computes `deadline = now + 0.0` — the spin loop never runs, so it returns `None` unconditionally.
Fix: `settle_s=0.05`.

Until then the third F/T calibration layer described in the docs does not exist. `--bias-abort-n`
catches an acutely bad zero; slow drift (sensor warming over a 70 s episode) goes undetected.

### 4. v4 is Fz-only with the z-check as a downstream veto

It fires on `|fz|` collapse; INSERT_DESCENT rejects it ~1 s later if the peg hasn't descended.
Measured on u_brown: repeated false fires at the same rim site, each costing an excursion.

Proposed shape (from the 2026-08-16 GPT review): a `HOLE_CANDIDATE` state that freezes θ, zeroes
lateral force, keeps Z compliant, and confirms on actual Z evidence before committing — with a
robust *local* Z baseline plus noise-derived significance, **not** a fixed `dz ≥ 0.3 mm` gate,
since background dip already reaches 0.39 mm. Bound it with a spatial refractory of one
detection-basin diameter after a rejected candidate.

### 5. Spiral pitch (2 mm) is coarser than the sensing radius (0.5–1.0 mm)

Force contrast vs distance from the true hole, contact samples only, rep3 (seated):

| dist | mean \|fz\| |
|---|---|
| 0.0 mm | 1.79 N |
| 0.5 mm | 3.77 N |
| 1.0 mm | 4.67 N |
| 2–7 mm | 4.6–5.6 N (background ~5.1, scatter ±0.5) |

The hole is a 65% force drop at 0 mm, 26% at 0.5 mm, and noise-level by 1.0 mm. With 2 mm pitch
the peg can pass 1.3 mm away and register nothing, so capture is probabilistic on a *fixed*
grasp — measured 26 s / 33 s / 54 s / never across four identical runs.

Literature agrees pitch should be ≤ the insertion clearance;
[Kang et al., IEEE RA-L 7 (2022) 6661](https://ieeexplore.ieee.org/document/9780009/) instead
shape the trajectory from an uncertainty distribution (90% hole-search success, 100% with
recovery, at 0.1 mm clearance).

> **Do not mutate `SearchDirector.pitch` in place.** Radius is derived as
> `r0 + (pitch/2π)·θ`, so reassigning pitch at θ=661° jumps the reference radius
> 5.17 → 3.34 mm and yanks the peg off the hole at the exact moment a detector would fire.
> Use a two-pass phase-shifted sweep (outward at 2 mm, inward through the gaps) or a
> separately-anchored local path.

Cost of going finer: arc scales as 1/pitch, so 1 mm doubles the sweep to ~138 s ideal.

### 6. Lag-pause conflates along-path lag with cross-track error

The peg sits pinned at exactly `lag_pause_thresh` (2.00 mm) for entire runs. That is the expected
equilibrium when the reference runs 5 mm/s against a 1.4 mm/s peg — not evidence of tracking.
Note `v_s` rate-matching is **already implemented** in the soft re-search path (`v_s = 0.001`) and
still stalls, so that is not the fix.

Replace with projection-paced progress: project TCP onto the planned path, maintain a monotonic
`s_peg`, place the target a small lookahead ahead, and track cross-track error separately since
that is what actually governs coverage.

### 7. Multiple v4 fires overwrite one `v4_predicate_fire` dict

Meta records only the last, so multi-false-fire runs can't be reconstructed post hoc.

### 7b. ~~INSERT_DESCENT seat check is symmetric; a seat *below* prediction is rejected~~ — DONE 2026-10-07 (one-sided band −20 mm…+5 mm, `seat_accept_below_m`)

Added **2026-10-07** (`insert_u_brown_20261007_193849`). The v4 fire was genuine, but the
z-check `|tcp_z − predicted_seat| > 5 mm` rejected it and resumed a soft search that pressed
on a seated part until stopped by hand at ~120 s (total SEARCH budget is 173 s here).

| | CAD | observed | offset |
|---|---|---|---|
| first contact (rim) | 230.07 mm | 219.3 mm | −10.8 |
| rest after descent | 199.07 mm | 188.0 mm | −11.1 |
| descent contact→rest | 31.0 mm | 31.3 mm | +0.3 |

Everything shifted by the grasp (the regrasp held the part 10.9 mm off — root cause was a
single 11 mm z blip in the step-2 grasp-point sample, fixed in `move_to_grasp` with a median
of recent samples). A false v4 (rim snag) leaves the peg *above* the seat; ending *below* it
after descending the full slot depth is a seat. Proposed: make the rejection one-sided, or
re-anchor the predicted seat by `surface_z − predicted_rim` at first contact. FSM change —
stage it per the insertion skill before any robot test.

**Second occurrence and cause (2026-10-07, u_orange 215309):** same −11.1 mm deeper seat,
then SEARCH on a seated part → rim-unload couldn't move it → `rim_stall` abort → replay
cleanup `move_home` without safe height pulled the part out (cleanup fixed: safe height
first). Cause of the depth: the grasp-point z of a part on its side flips between two
values ~10 mm apart (−0.032 / −0.042) with the viewpoint's fold interpretation; regrasp
flange z 0.1870 vs normal 0.1971 (u_orange) and 0.1862 (u_brown) explains both seats to
within 0.7 mm. Neither the latest sample nor the hover median picks the right branch
reliably — open calibration/estimation question, deliberately not chased.

### 7c. Start the SEARCH spiral at a learned per-(object, fold) offset — staged for later

Added **2026-10-07**. Where the hole actually was vs the spiral centre (predicted seat),
seated runs only (Aug 16-17 + Oct 7):

| object | fold eq (xyzw) | n | mean (dx, dy) | std |
|---|---|---|---|---|
| u_brown | (0, .71, -.71, 0) | 8 | (+1.8, -2.4) mm | (0.6, 1.7) mm |
| u_brown | (-.71, 0, 0, .71) | 4 | 0 for touchdown seats; one bad-grasp outlier (+2.6, +6.7) | — |
| u_orange | both | 7 | 0.1 mm (always seats on touchdown) | — |

For u_brown's first fold the hole sits consistently ~3 mm toward (+x, -y); the spiral
sweeps the wrong side first and finds it a full turn later. Proposal: derive a
per-(object, fold) start-offset table from seated CSVs with a script, and centre the
spiral at predicted + offset when that (object, fold) has enough consistent history.
Strictly per fold (SKILL.md §11 — priors mirror between fold equivalents). Force can't
sense "close" beyond ~0.5-1 mm (#5), so where to start is the lever, not when to switch.

---

## Blocked — needs new data, not a code change

### 0. `line_green` rotates at grasp and seats wedged; nothing available detects it

**This is not a tuning problem and should not be attacked as one.** Operator assessment
2026-08-16: `line_green` is strongly inclined to rotate in the jaws at grasp. It then seats
proud/wedged rather than flat, and the only recovery is a full re-seat. Solving it requires
**more force data and camera-pose work** — the signals the decision currently rests on are not
accurate enough to build any predicate on.

Why no threshold fixes it — measured across two passes on the same part:

| | depth_err | pos_dev | ori_dev | truth |
|---|---|---|---|---|
| pass 1 | −1.65 mm | 0.8 mm | 0.74° | **genuine seat** |
| pass 2 | −2.32 mm | 0.5 mm | 0.45° | **wedged** |

- `depth_err` separates them by only **0.67 mm**, so any threshold sits knife-edge at ~−2.0 mm.
- `pos_dev` and `ori_dev` are **inverted** — the wedge scores *better* on both. They are blind
  to this failure.
- The torque check that would catch tilt was deliberately disabled (`prismatic_peg_insertion.py`
  ~L96): it "never fires even when the part is correctly seated". Its `avg_Tx`/`avg_Ty` are
  printed as `ignored` in the success line.

So both available signals have failed on this part. The gap is upstream — grasp stability and
pose accuracy — not the acceptance criterion.

Downstream consequence: a wedged `line_green` sits proud and blocks `inverted_u_yellow`. On the
2026-08-16 pass 2, yellow only seated after the operator pushed `line_green` home by hand. Any
report of "4/4" must state whether an operator touched the parts.

`EXIT_GEOMETRIC_Z_TOL_BELOW_M = 0.0080` (8 mm) is undeniably permissive, but tightening it
without better signals converts silent false accepts into unreliable false rejects. Leave it
until there is data to place it.

### 0b. `prismatic_peg_insertion` can hang after reporting success

Observed 2026-08-16: printed `[SUCCESS] Insertion complete on attempt 3` and never returned;
the replay stalled until killed. A primitive that succeeds and does not exit blocks the whole
assembly with no error to act on — the operator has to notice.

---

## Robustness / process

### 8. `move_to_grasp` gaps

- No camera-detection retry (pre-existing; item 2 on the older open list).
- Step-2 y target sits **~11 mm off the camera object centre** for `line_green`, despite a
  `center_point` grasp with a ~0.0001 m offset. Consistent every attempt, so it predates
  2026-08-16 — but for a long thin bar that is a large fraction of the graspable width.
- `compute_duration` floors at 2.5 s, so lowering `cart_vel` for this move does nothing. Worth
  knowing before trying to slow it.

### 9. `translate_object`'s 3.0 s zero settle is only exercised on the replay path

Verified once on 2026-08-16. `run_assembly_step` uses the wrapper's own 5 s default and never
touches it.

### 10. Unify the two insertion paths

Long-standing. `prismatic_peg_insertion.py` still lives under `primitives/_real_mode_stash/`;
eventual home should be `primitives/inserts/`.

### 11. `main` is sim-only

Composite vs flat grasp ids, plus `--grasp-candidate` refusing real mode. Either finish the
candidate-native migration on hardware (verify the top-down gate and W-dynamic offset, lift the
sim-only guard, remap the assembly JSONs) or port this branch's fixes forward.

### 12. Constant ~2 mm −y offset — camera mount (hand-eye) estimate (queued 2026-10-14)

With the pad-centre grasp (commit 386e31e) the inserts seat, but every part lands ~2.3 mm in
world −y of its CAD-predicted seat, whatever the part's yaw: u_brown (0.0, −2.3, −0.6) mm,
u_orange (+0.8, −2.3, +0.8) mm (seat TCP − prediction, raw CSV, 2026-10-14). The operator
observed the same constant −y on the table. A yaw-independent, world-fixed offset points at the
camera mount translation (`config/robot_config.yaml` hand-eye, 2026-10-13), not at per-part
stickers or grasp points. Fix later — the assembly works; the compliant search absorbs it.
Start: re-solve the hand-eye translation with base1 as a fixture (seat TCP vs prediction over
several parts/yaws), and re-check `DEFAULT_BASE_POSITION` / `PER_OBJECT_BASE_OFFSET_M`, which
were fitted with the old shallow grasp and may already absorb part of it.
Loop data (seat TCP − prediction, y mm, 2026-10-14 13:01–14:05): u_brown −2.3, +5.0, 0.0, −0.0;
u_orange −2.3, +0.1, −4.7, ~0. Not a clean constant: a bias plus a few mm of run-to-run scatter.

### 13. line_green tilt: nudge it along its axis (queued 2026-10-14; method verified on the robot)

Diagnosis (depth profile, `seat_check.py` now records it): a tilted line_green (pitch ~3.5°, one
end ~5 mm high, centre at seat height) has that end resting on base1's TOP past the end of its
pocket (pocket floor 10 mm, bar bottom 11.5 mm on the U bottoms, base top 17.5 mm in the base
frame; the 140 mm bar has almost no axial play). The seat check is user-confirmed as the tell.
Fix that worked (operator's method, 15:55): gripper CLOSED, approach from safe height, lower
beside the raised end with the pad centre at the end face (base z 37 mm, tips 22 mm), push along
the bar toward the slot in ≤5 mm steps, guarded at **5 N** (base1 is glued to the table) — it
dropped in flat (0.44°) with the push stopping at 5.1 N.
Did NOT work: top-down reseat, tilted-wrist regrasp (+ prismatic reinsert, or levelled in the air
and lowered), pressing the high end down, sliding it while gripped.
To do: automate this nudge in `replay_real_assembly.py:seat_check_and_reseat` (direction from the
measured high end; re-check depth after each step), replacing the top-down reseat. Root cause to
chase separately: why the bar lands axially off (insert pose error along base y, see #12).

---

## Investigated and closed

- **`replay_real_assembly.py` exit code** — not a bug. It returns 1 on failure correctly. An
  earlier report of "exits 0" was an artifact of `python3 …; echo "EXIT=$?"` making the *wrapper*
  exit 0.
- **Base fixture drift** — the camera/`DEFAULT_BASE_POSITION` disagreement of 11.6 mm is the
  documented CAD-prior error, not a moved fixture. Searching at the camera-derived centre swept a
  full 8 mm radius and found nothing; the base is fixed and correct.
- **Rate-matching `v_s`** — already implemented in the soft path; still stalls.
- **Operator press duty cycle as a cross-part target** — refuted. GOLD unload duty ranges
  6.1%–35.9% across the four parts. Direction holds (successful autonomous runs unload more than
  failed ones); the level is part-specific.

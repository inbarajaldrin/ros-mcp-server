# Autonomous Insertion Methodology — Follow-up to COLLECTION_METHODOLOGY.md

> Companion to `COLLECTION_METHODOLOGY.md`. That document covered the GUIDED-mode operator-demo collection pipeline that produced the 60+ telemetry records used to derive the v4 Found Hole detector + the SEARCH director. This document picks up where that left off: **how to go from "we have demos" to "the robot inserts autonomously without operator hands."**
>
> Validated 2026-05-07 on UR5e + OnRobot RG2 + workspace camera, FMB1 assembly. **All four FMB1 parts insert end-to-end** via `replay_real_assembly.py` (tag `real-world-verified-2026-05-07`):
> - u_brown / u_orange via the autonomous SEARCH path described in this document (`compliant_insert` wrapper).
> - line_green / inverted_u_yellow via a **second insert path** — `prismatic_peg_insertion` (under `primitives/_real_mode_stash/`) routed through `translate_object.py`'s line_green/yellow branch. This path uses XY-compliant settling + Rx/Ry compliance + geometric depth-based exit (autonomous SEARCH spiral does not work for these wide-grip parts where the gripper jaws rest on the base rim — see §9).
>
> Unification of the two paths is queued as future work.
>
> **Which tree (2026-08-16):** the real path runs only from branch `ur5e-fmb1-demo` (or a worktree
> at the tag above), with the grasp publisher started from that same tree. On `main` the grasp
> publisher emits composite marker ids (101…303) while the assembly JSONs carry flat ids, so every
> pick fails with `"Grasp point 1 not found"`. See `docs/HANDOFF_NEXT_AGENT.md`.
>
> **Re-verified 2026-08-16:** one genuine autonomous 4/4 pass. A second pass was **not** 4/4 —
> `line_green` was false-accepted at a wedge and `inverted_u_yellow` seated only after the
> operator pushed line_green in by hand. The prismatic path's seat-acceptance criterion is not
> solved; see §9.4.
>
> **Rotation stays LOCKED in SEARCH** — `selection_vector = (T,T,T,F,F,F)`, see §2. `SKILL.md`
> §10's "all-True" rule is wrong here and following it broke five consecutive real-arm runs.
>
> Pair this doc with the `insertion-control-law-derivation` skill, especially §13–§19, before changing code.

---

## TL;DR — what changed between collection and autonomous

Collection mode (operator-driven, Z-locked, hand drags peg):
```
APPROACH → Contact → GUIDED (operator drags) → SIGUSR1 → INSERT_DESCENT → DONE
```

Autonomous mode (robot-only, Z-compliant, spiral search):
```
APPROACH → Contact → SEARCH (spiral PD director) → v4 fires → INSERT_DESCENT → DONE
                                  ↘ global seat detector → DONE   (peg dropped straight through)
```

Five non-obvious facts that determined whether the autonomous version actually works:

1. **One-time per-fixture base calibration** is required and lives in `DEFAULT_BASE_POSITION`, NOT in the FSM. Derive it from a SINGLE centered-grasp demo per (object, base) — averaging across grasps in any other configuration produces grasp-contaminated bias.
2. **Per-grasp offset is a runtime unknown.** Spiral search exists to discover it, not to bake it in. Hardcoded biases work for one grasp configuration and break the next.
3. **Constant-force tracking, not PD.** Position-error PD gives sub-stiction commanded force at small errors → instant stall. Constant Fmax in unit(error) direction always engages stiction-breaking force.
4. **r_launch > 0.** When DEFAULT_BASE_POSITION is correctly calibrated AND grasp is centered, peg lands AT spiral center. Spiral starts at radius 0 → no traversal direction → stall. r_launch = 1.5mm guarantees a starting direction.
5. **Z-locked operator demos and Z-compliant autonomous SEARCH produce different |fz| signatures.** v4 thresholds derived from Z-locked operator demos may not apply directly to Z-compliant autonomous mode — particularly for multi-prong parts. Lower `F_press` to mimic operator drag pressure if signal is too saturated.

---

## 1. Calibrating DEFAULT_BASE_POSITION from a single centered-grasp demo

**Goal**: make CAD-predicted seat xy match the physical hole xy to <2mm.

**Procedure** (~5 min on robot):
1. Operator manually grasps the part with a CENTERED grasp (gripper symmetric on the part body, no off-center bias).
2. Run a single GUIDED demo with `--variations A_pos_x_10mm --reps 1`.
3. Operator drags peg to actual hole; SIGUSR1 records `hole_observed_operator.xy_m`.
4. Compute bias:
   ```
   CAD_with_offset_applied  = meta.cad_prediction.predicted_tcp_at_seat.xyz_m[0:2]
   CAD_default_base         = CAD_with_offset_applied − applied_offset (e.g. -10mm in X)
   bias                     = hole_observed_operator.xy_m − CAD_default_base
   ```
5. Update `DEFAULT_BASE_POSITION` in `primitives/shared/config.py` by `+bias`. CAD now predicts the actual hole.

**Anti-pattern**: averaging biases from multiple GUIDED demos with different grasps. The operator's drag-to-perceived-hole records EE-centered-over-hole, not peg-tip-over-hole, so off-center grasps shift the recorded `hole_observed_operator` by the grasp offset. Average across off-center grasps gives a contaminated bias that's wrong for any single autonomous run.

**Cross-check**: after updating DEFAULT_BASE_POSITION, predict u_orange and u_brown's hole xy via cad_lookup. They should differ ONLY in Y (per `fmb_assembly1.json`'s position field — same X for both parts in base1). If they differ in X by >1mm, the calibration is contaminated.

---

## 2. The autonomous SEARCH director (Archimedean spiral PD with constant-force tracking)

Source: GPT-5 analysis (job `gpt-temaxw-74fd8c`, 2026-05-06), backed by Chhatpar/Branicky 2001 + Jasim/Plapper/Voos 2014.

**State**: SEARCH (added between APPROACH and INSERT_DESCENT). Replaces operator-drag GUIDED in autonomous mode. Implementation: `compliant_insertion_studio/wrapper/contact_search_fsm.py:SearchDirector`.

**Path generator** (Archimedean spiral, parametrized in time):
```
center_xy   = predicted_tcp_xy        # CAD-corrected; no FSM-side bias
r(θ)        = r0 + (pitch / 2π) · θ
θ̇           = v_s / max(r, r0)         # constant tangential speed
p_ref(t)    = center + r(θ) · (cos θ, sin θ)
```

**Control law** (constant-force tracking, sign-flipped):
```
e        = p_ref − p_tcp
if |e| > ε:
    F_xy = −Fmax · unit(e) − Kd · v_tcp
else:
    F_xy = 0
F_z      = −F_press
selection_vector = (T, T, T, F, F, F)     # X/Y/Z compliant, rotation LOCKED — not optional
```

**The rotation lock is load-bearing.** With rotation compliant, lateral force applies a **moment
about the grasp point**: the part pivots in the jaws while the gripper translates, so TCP
displacement stops being peg displacement and every swept-area / coverage figure computed from TCP
becomes fiction. `SKILL.md` §10 states `selection_vector` "must remain all-True" as a hard rule —
that is **wrong for SEARCH**, and an agent who followed it broke the insert for five consecutive
real-arm runs on 2026-08-16. Every 2026-05-07 run that seated commanded `(1,1,1,0,0,0)` during
SEARCH; check the `cmd_wrench_raw` sidecars before trusting any written rule to the contrary.

The sign-flip is empirical: peg moves opposite-direction of commanded F_lat in our `base_link↔base` 180° setup. Single-axis behavior validated by `verify_baselink_motion.py`; multi-axis SEARCH inverted in practice. **Measure on your robot** before assuming sign convention.

**Lag-pause** (peg-paced spiral expansion):
```
if peg-to-ref distance ≤ lag_pause_thresh (2mm):
    advance theta
    spiral_arc_path_len += v_s · dt
else:
    hold theta; let peg catch up
```

**Gradient-following override** (active control when `|fz|` drops toward v4):
```
d_fz_dt = (|fz|_now − |fz|_200ms_ago) / 0.2

if d_fz_dt < −3 N/s AND |fz| < 6N AND |v_peg| > 0.5 mm/s:
    # Peg moving from rim into chamfer — keep going in same direction.
    F_xy = −Fmax · unit(v_peg)
```

This is the user's "active control kicks in when value going down" rule. Without it, the spiral pulls peg off the chamfer edge before v4's 0.3s sustain elapses.

**Stall detector**:
```
if (peg progress in 1s window) / (spiral arc grown in 1s window) < 0.15:
    ABORT lateral_stall
```

**Termination**: v4 detector fires (state-independent), or spiral exhausts at R_max (8mm), or stall, or 25s timeout.

### 2.1 Default parameters that worked

| Param | Value | Notes |
|---|---|---|
| `r0_m` | 0.0015 (1.5mm) | Avoids peg-at-center stall |
| `pitch_m` | 0.002 (2mm) | Chamfer-capture, not clearance |
| `v_s_m_s` | 0.005 (5mm/s) | Operator drag speed |
| `R_max_m` | 0.008 (8mm) | Per-grasp + cal-residual disk |
| `Fmax_N` | 5.0 → **8.0 tuned** | Above stiction, below 30N abort. `translate_object` passes 8.0; 5.0 is measurably worse on u_brown. |
| `F_press_N` | 5.0–7.0 → **9.0 tuned** | Originally 5 for multi-prong, 7-9 for single peg. `translate_object` now passes 9.0. |
| `lag_pause_thresh_m` | 0.002 | Peg-to-ref tolerance |
| `stall_progress_ratio` | 0.15 | tcp progress vs spiral arc |
| `stall_window_s` | 1.0 | Detection window |
| `near_miss_fz_thresh_N` | 4.0 | Below this, gradient may activate |

---

## 3. Detection (v4 Found Hole predicate)

Defined in `analysis/CONTROL_LAW.md`. Unchanged from collection-mode validation. Runs in parallel during SEARCH:

```
ON_RIM   ⇔ |fz_smoothed| > rim_high (4N)
OFF_RIM  ⇔ |fz_smoothed| < rim_low  (3N)
FIRE when:
  - state has been OFF_RIM continuously for ≥ off_sustain_s (0.3s), AND
  - state was ON_RIM at any point in the previous recent_window_s (2.5s)
```

Tool-frame, direction-invariant by construction. Validated 10/10 across 3 directions (analysis/CONTROL_LAW.md).

---

## 4. Global seat detector — must run in SEARCH/APPROACH

Verified case 2026-05-06: u_brown peg fell straight through during APPROACH. fz briefly bumped >3 entering SEARCH. tcp_z was already 1.7mm below predicted seat with motion stopped. Without the seat detector running in SEARCH/APPROACH, the FSM aborted on lateral_stall.

**Rule**: state-independent seat detector runs every tick in `(APPROACH, FIND_HOLE, ENTRY_SETTLE, WEDGE_RECOVERY, INSERT, SEARCH, INSERT_DESCENT)`:
```
if |tcp_z − predicted_tcp_z| < at_target_z_tol_m (5mm)
   AND |dz/dt| < insert_motion_thresh_m_s (0.5 mm/s)
   AND tilt_deg < insert_tilt_abort_deg (5°)
   sustained 1s:
       transition to DONE
```

---

## 5. Per-object configuration (gripper_width, grasp_id) from `fmb1_assembly.json`

Source of truth: `ablations/eval_resources/fmb1_assembly.json` → `assembly_order[i].gripper_width_mm` and `.grasp_id`.

API: `primitives.shared.config.get_gripper_width_mm(object_name, grasp_id, default_mm=35)` and `get_grasp_id_for_assembly(object_name)`.

Wired into:
- `run_assembly_step.py` — `--grasp-width None` triggers auto-resolve
- `regrasp_held_object.py` — same
- `loop_autonomous_insert.sh` — auto-resolves grasp_id from object name; doesn't pass hardcoded width

---

## 6. The `loop_autonomous_insert.sh` test harness

Three test modes:

| Flag combo | Purpose |
|---|---|
| `N --no-randomize` | Repeatability at zero offset. Tests stability of the search across iterations with the same configuration. |
| `N --regrasp` (recommended) | Each iteration releases + camera-grasps + runs autonomous insert. Tests robustness to natural per-grasp variation (~1-3 mm). |
| `N` (default = randomize) | Synthetic ±7mm `--base-offset-xy` perturbations + 5 cardinal/diagonal directions. Tests larger-than-natural offset robustness. |

Use `--object u_brown` etc. to switch parts. The script auto-resolves grasp_id and width.

Robust autonomous insertion validation = `--regrasp` mode passing N=3+ across multiple parts.

---

## 7. Iteration discipline (re-read the SKILL.md TL;DR before changing code)

The autonomous SEARCH director was reached in 8 iterations from "no autonomous insertion at all" to "u_orange / u_brown solid." Every iteration that worked obeyed the SKILL.md TL;DR:

- **Hypothesis-driven**: each change addresses a specific empirical failure mode in the prior iteration's data.
- **Structural over tuned**: we changed control law structure (PD → constant force, position-only → gradient-aware), not just parameter values.
- **Falsifiable predictions**: each iteration declared an expected outcome before launching. Mismatch → diagnose → next iteration.
- **No invented thresholds**: every threshold (3N, 4N, 0.3s, 2.5s, 1.5mm, 8mm) was derived from operator-demo data quartile analysis or explicit physics.

`analysis/AUTONOMOUS_RUN_LOG.md` is the live record of those iterations and is the format for future iteration cycles. **Append to it; don't overwrite.**

---

## 8. Anti-patterns (committed and recovered from in this session)

| Anti-pattern | What happened | Lesson |
|---|---|---|
| Hardcoding "bias" in FSM from un-centered demos | Confounded base-cal-error with grasp offset; "worked" for u_orange but broke for u_brown | One-time fixture cal in DEFAULT_BASE_POSITION; per-grasp absorbed by spiral |
| Trusting `-F_lat` / `-r_cop` direction in autonomous mode | Friction positive-feedback loop: -F_lat reinforces commanded direction → self-confirming | Use blind spiral + v4 detection. Direction signals only valid with externally-known-correct drag (operator) |
| Position-error PD with default Kp=350 N/m | 1.3N at typical errors → instant stall | Constant-force tracking |
| Spiral starting at θ=0 (always +X heading) | Peg can drift in arbitrary wrong direction first | Spiral CENTER at predicted_seat (CAD), not at peg landing |
| Excluding SEARCH/APPROACH from seat detector | Peg-already-seated cases aborted on lateral_stall | State-independent seat detector |
| Using v4 thresholds derived from Z-locked operator demos for Z-compliant autonomous | Multi-prong parts saturate `|fz|` even at chamfer | Lower F_press to match operator drag; future: drop-relative-to-baseline detection |
| Freezing spiral on chamfer-edge dip | Peg sits in marginal spot, doesn't fall in | Active gradient-following: continue motion in dip-direction |
| **Unlocking rotation in SEARCH** (2026-08-16) | An agent read `SKILL.md` §10's "all-True" rule literally and made all 6 DOFs compliant. Lateral force then applied a moment about the grasp point: the part pivoted in the jaws while the gripper translated, TCP displacement stopped tracking peg displacement, and the swept-area figures computed from TCP were fiction — one session concluded "the hole is not within 6 mm" about a hole 3.38 mm away. Five consecutive failed real-arm runs. | Rotation stays LOCKED: `(T,T,T,F,F,F)`. Verify against the `cmd_wrench_raw` sidecars of runs that actually seated, not against a written rule. |
| **Sampling a force reference on a still-settling arm** (2026-08-16) | The trajectory controller reports "complete" at commanded position, not at rest. A zero taken there is confidently wrong and nothing downstream can tell: 0.0 s settle → 15.6 N post-zero bias → force mode drove the TCP 116 mm upward. | Settle ≥ 1.5 s **before** the zero (5.36 s measured clean). Never shorten a settle/step-back window for speed. |
| **Single-sample contact thresholds** (2026-08-16) | The wrench carries ~60 ms impulses reaching 40–60 N with nothing touched (`+0.18, +19.32, +56.98, +20.56, +0.36 N` at 50 Hz), tripping a 40 N gate. Peak varies run to run, so re-zeroing or slowing the move changes the number without fixing it. | Require the threshold to hold over a sustained window, the way v4 uses `off_sustain_s`. |

---

## 9. Open problems (next agent's queue)

### 9.1 Multi-prong / multi-contact parts (inverted_u_yellow)

`|fz|` saturates around 7N during autonomous SEARCH (vs 3N during operator GUIDED). v4 collapse signal doesn't appear because some prong is always on rim. F_press lowering to 5N helps but doesn't fundamentally solve it.

Possible directions:
- **Relative `|fz|` drop detection**: replace absolute v4 thresholds with `|fz| < 0.5 × recent_median`. Catches dips even when baseline is high.
- **Peg-z descent dominance**: detect peg sliding down chamfer slope via `tcp_z` decreasing, not `|fz|` collapse.
- **Hybrid Z control**: lock Z during search like operator demos; switch to compliant Z only on rim-cross detection.

### 9.2 Two-stage insertion

When a part must clear through an alignment phase before the final slot. The "underlying object" the spiral pushes against might itself move with the spiral. Operator-flagged but not yet implemented. Likely needs a separate FSM state with constraints on lateral force / spiral coverage.

### 9.3 Sign convention root cause *(numbering continues; §9.4–§9.6 added 2026-08-16)*

The empirical `F_xy = -Fmax * unit(e)` sign-flip works but the underlying convention conflict between `verify_baselink_motion`'s single-axis result and SEARCH's multi-axis behavior is unresolved. A clean fix requires a controlled multi-axis test in force-mode.

### 9.4 Prismatic seat acceptance is too permissive — it false-accepted a wedge

Opened 2026-08-16. `EXIT_GEOMETRIC_Z_TOL_BELOW_M = 8 mm`, but the real separation between a seat
and a wedge is more than an order of magnitude tighter:

| Case | depth_err | pos_dev | ori_dev |
|---|---|---|---|
| good `line_green` seat | **−1.65 mm** | 0.8 mm | 0.74° |
| accepted wedge | **−2.32 mm** | 0.5 mm | 0.45° |

**0.67 mm apart on depth, and `pos_dev` / `ori_dev` do not discriminate at all — the wedge scored
*better* on both.** So the two secondary gates cannot be used to rescue the depth gate. This needs
tuning against more seats; until it is done, do not treat a `[SUCCESS]` from the prismatic path as
a seat without checking depth against the known-good figure for that part.

### 9.5 `line_green` rotates at grasp

Known, unsolved. When it happens the part seats proud or wedged, and the only recovery is a full
re-seat. **Camera pose and force reads are both unreliable for detecting this state**, so there is
no autonomous detector today — this is what forced the manual intervention in the second
2026-08-16 pass. Any fix probably needs a signal neither of those two channels provides.

### 9.6 `prismatic_peg_insertion` hangs after success

Observed printing `[SUCCESS] Insertion complete on attempt 3` and then not exiting. Blocks
unattended chaining of the assembly.

---

## 10. Files relevant to this work

| Path | Role |
|---|---|
| `compliant_insertion_studio/wrapper/contact_search_fsm.py` | FSM + SearchDirector + FoundHoleDetector |
| `compliant_insertion_studio/wrapper/compliant_insert.py` | Wrapper, CLI flags, meta persistence |
| `compliant_insertion_studio/scripts/run_assembly_step.py` | Full pipeline (pick→rotate→place→regrasp→rotate→insert) |
| `compliant_insertion_studio/scripts/regrasp_held_object.py` | Camera-driven release+regrasp from held state |
| `compliant_insertion_studio/scripts/loop_autonomous_insert.sh` | Multi-iteration test harness |
| `primitives/shared/config.py` | DEFAULT_BASE_POSITION + grasp lookup helpers |
| `analysis/CONTROL_LAW.md` | v4 Found Hole detector spec |
| `analysis/SEARCH_CONTROL_LAW.md` | SEARCH director spec |
| `analysis/AUTONOMOUS_RUN_LOG.md` | Per-iteration test log (append-only) |
| `ablations/eval_resources/fmb1_assembly.json` | Per-(object, grasp_id) gripper width + assembly order |

---

## 11. The minimum-viable port to a new (object, base)

1. Confirm part is in `fmb1_assembly.json` with correct `grasp_id` and `gripper_width_mm`. If not, add it.
2. Verify `DEFAULT_BASE_POSITION` is calibrated for this base (one-time per fixture; redo only if the fixture is physically moved). **A residual disagreement of ~11.6 mm between `DEFAULT_BASE_POSITION` and the camera is expected and is the documented CAD-prior error** — the chain is empirically 5–17 mm off. It is not evidence that the fixture moved; base1 is fixed and correct. Do not re-derive the calibration on the strength of that residual, and do not fold it into the FSM as a bias — absorbing it is what the spiral is for (see §8, anti-pattern 1).
3. Run a centered-grasp GUIDED demo on this (object, grasp_id) to validate hole_observed matches CAD prediction within ~2mm.
4. Run `loop_autonomous_insert.sh 1 --object <name> --no-randomize --regrasp`. If success → run with N=3 for stability.
5. If failure: check `|fz|` profile during SEARCH. If saturated >5N throughout, lower F_press. If peg never moves, raise Fmax. If spiral exhausts without v4, the per-slot calibration error exceeds R_max — use centered-grasp demo to derive base correction.
6. Iterate per `analysis/AUTONOMOUS_RUN_LOG.md` discipline: hypothesis → declare expectation → run → diagnose mismatch → structural fix.

The loop ends when `--regrasp` mode passes 3/3 with predictable timing.

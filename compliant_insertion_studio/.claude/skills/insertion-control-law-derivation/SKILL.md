---
name: insertion-control-law-derivation
description: Authoritative methodology for deriving force-compliant peg-in-hole insertion control laws for the compliant_insertion_studio project from raw operator-demo + autonomous-fail telemetry. Use this skill WHENEVER touching compliant_insertion_studio's contact_search_fsm.py / _find_hole_wrench / wrapper/compliant_insert.py, tuning configs/defaults.yaml's fsm/find_hole/correction sections, analyzing insert_u_orange_*.csv (or any FMB1 object's insert telemetry), iterating to make the autonomous insert primitive succeed, debugging FIND_HOLE STUCK / APPROACH timeout / ENTRY_SETTLE failures, comparing GOLD operator demos against FAIL autonomous runs, building staged patches under analysis/iterations/staged/, or asked to "make the autonomous insert work" / "fix the spiral" / "tune the insertion algorithm" / "why won't the peg go in" / "improve the control law". This skill encodes hard-won lessons from a 10-iteration debugging session (May 2026) where the wrong methodology (parameter sweeps, narrow z_drop metric, treating CAD prior as ground truth, ignoring orientation feedback) cost ~16 prior session iterations + 10 in-session iterations before the correct approach was identified. Read it BEFORE proposing ANY FSM code change or parameter tweak — the right approach is data-derived structural changes informed by 3-way pose comparison and orientation feedback, not intuition-based value tuning.
---

# Insertion Control-Law Derivation — Methodology

This skill is the canonical guide for working on the compliant_insertion_studio's insertion primitive. It encodes **hard rules** that were violated repeatedly across 26+ failed iterations before being identified. Read every section before proposing any code change.

## TL;DR — the five things people miss

1. **The target is SOFT, not hard.** This primitive exists BECAUSE perception has grasp offset. The CAD-derived `target_TCP` is an estimate (empirically off by 11+ mm). Force feedback IS the primary guide.
2. **Grasp uncertainty is xy-only**, not orientational. So TCP orientation deviation during ACTIVE has physical meaning — it is the **primary position-error feedback signal**, not noise.
3. **z_drop is too narrow.** Compare full 6-DOF TCP pose + object pose + joint torques + native-rate wrench across **GOLD ↔ FAIL ↔ PROJECTED** (3-way diff), not just `drop_so_far`.
4. **Trust only raw data.** Never use FSM stdout claims, signals.json segmentation, or any threshold whose derivation can't be re-run from raw CSVs as ground truth.
5. **Iterate on structure, not values.** Parameter sweeps without structural insight are how 16 prior iterations failed. Each new iteration needs a data-derived structural reason.

---

## 1. Why this primitive exists (mental model)

The peg sits in the gripper after `rotate_object` snaps it to canonical face-down. Visual perception that produced the grasp had **perception offset** — typically 5–17 mm in xy. So:

- The CAD chain `predict_tcp_at_seat(base_world, object, grasp_id, ...)` produces a `target_TCP` that is **an estimate, not ground truth**. Empirical: in the May-5 GOLD run (`insert_u_orange_20260505_193645`), CAD-predicted seat was at `(+25.28, -349.98, +200.80) mm` but the operator's peg actually seated at `(+28.55, -360.62, +200.11) mm` — **11.14 mm 3D offset**.
- Even prior-demo-derived `hole_xy_prior` goes stale across runs (peg presentation varies per `rotate_object` IK selection; rim contact geometry shifts; we observed 4–6 mm per-run drift between two GOLD demos of the same object).
- The whole point of force-compliant insertion is to **feel its way** to the actual slot using force/torque feedback. Treating any prior as a hard waypoint contradicts the primitive's purpose.

**Implication for control law:** the prior is a starting estimate that must be **refined online** using sensed feedback (orientation, force, joint state). A control law that drives blindly to a stored xy will work only when the prior happens to match — most of the time it won't.

## 2. What's uncertain vs what's known

| Quantity | Uncertain? | Source |
|---|---|---|
| Peg orientation rel. to gripper | **Known** | rotate_object snaps to canonical face-down; mechanical grasp pose well-defined |
| Peg position rel. to gripper | **Uncertain (5–17 mm xy)** | Visual perception had grasp offset |
| Slot xy in world | Uncertain | CAD prior + camera-base calibration both have error |
| Slot z (depth) | Mostly known | Per-object CAD depth is reliable; `predict_tcp_at_seat.z` is accurate within ~1 mm |
| Peg orientation in world during contact | **Physically determined by contact geometry** | Treat as feedback signal, not noise |

**Critical inference:** because grasp uncertainty is xy-only, **TCP orientation deviation observed during ACTIVE is contact feedback**, not perception noise. Use it.

## 3. TCP orientation as the primary position-error feedback signal

The single biggest missed insight in the May-5 session. The control law must close a feedback loop on TCP orientation:

- **Tilt rises during FIND_HOLE** → peg is wedged at a rim edge → current TCP xy is wrong.
- **Tilt direction** → indicates which edge is pinning the peg. If TCP top tilts in +X direction, the peg's bottom is held back by a rim edge in **-X**. To free the peg, move TCP toward **+X** so peg slides past that edge.
- **Tilt relaxation** (sustained drop after a peak) → chamfer engaged, peg slipping into slot. Use this as the trigger to update the seat-xy estimate to the current xy, AND as a stronger transition signal than `tcp_z < surface_z - find_hole_drop_thresh_m` (which never fires for tight-clearance pegs).
- **Tilt tolerance band** must be derived from operator demos for each object, not guessed. Empirical for u_orange: GOLD shows ~0.4° at contact growing to ~3° peak during chamfer engagement, then RELAXING back to ~1.8° as peg seats. Tilt > 4–5° = position error.

**Anti-pattern (committed in iters 1–9):** allowing TCP to tilt freely (selection_vector all-True is correct) but only **logging** tilt as diagnostic. The orientation is the steering signal; close the loop on it.

## 4. The trust hierarchy — what to use as ground truth

**FULLY TRUSTED** (raw, computed by the device, not the wrapper):
- Main CSV columns: `tcp_xyz`, `tcp_quat`, `fx..fz`, `tx..tz`, `gripper_width`, `commanded_fz`
- v1.2 sidecar contents (see Section 5)
- `meta.json:force_mode_params` (the actual controller settings sent)
- `meta.json:assist_level`, `meta.json:outcome_reason` ONLY when manually tagged with the controller mode

**DERIVED-TRUSTED** (computed by analysis scripts from raw):
- Phase boundaries (contact / engagement / seated) from physics-based detectors in `analysis/scripts/01_extract.py`
- Smoothed F_lat, dz/dt, tilt, r_cop computed by analysis scripts

**NEVER TRUSTED**:
- FSM stdout strings: `"engagement_confirmed"`, `"STUCK"`, `"GLOBAL SEAT"`, `"seat_detected"`, etc. — these are based on guessed thresholds; the same thresholds caused 16+ prior session iterations to be misread.
- The FSM-reported `outcome` flag (`success` / `abort` / `timeout`) — derived from those same buggy predicates.
- `signals.json` segmentation methods M1–M5 — they use baseline-derived thresholds that are **circular** when treated as labels (you'd be using one model's output to evaluate another).
- Default values in `configs/defaults.yaml` whose derivation can't be re-run from raw data — treat them as priors, not facts.

**Hard rule:** every threshold, label, or event used to drive the control law must be **post-analysis** of raw CSV columns by a Python script. Annotation comes AFTER the data, not before. If a value's origin is "intuition" or "the prior session set it", treat it as suspect until re-derived.

## 5. Required telemetry — schema v1.2 sidecars

The wrapper's CSV writer (post-2026-05-05 schema bump) emits five files per episode:

| File | Topic | Rate | Purpose |
|---|---|---|---|
| `insert_<obj>_<ts>.csv` | fused 100 Hz | downsampled | dashboard-friendly main stream |
| `insert_<obj>_<ts>.joints_raw.csv` | `/joint_states` | native ~250 Hz | per-joint pos/vel/eff — operator's nudge signature lives in joint torques |
| `insert_<obj>_<ts>.wrench_raw.csv` | `/force_torque_sensor_broadcaster/wrench` | native 500 Hz | sub-10 ms contact transients aliased in main CSV |
| `insert_<obj>_<ts>.cmd_wrench_raw.csv` | every `SetForceMode` call | event-based | per-event 6-axis cmd: distinguishes algo intent from sensed disturbance |
| `insert_<obj>_<ts>.fm_events.csv` | `/force_mode_controller/transition_event` | rare | controller lifecycle |

**Key fact about the dataset:**
- The 60 May-3 GOLD operator demos are **schema v1.1** — they DON'T have these sidecars.
- Only the four May-5 episodes have the v1.2 sidecars. Use the canonical pair:
  - GOLD: `insert_u_orange_20260505_193645` (operator success, gain=1.0 damp=0.7)
  - FAIL: `insert_u_orange_20260505_193941` (autonomous abort, same compliance)

When deriving control laws, do the analysis on the v1.2 pair first; cross-check on v1.1 demos for the columns that exist (TCP pose, downsampled wrench).

## 6. The 3-way pose comparison (mandatory for any iteration)

Before proposing any control-law change, compare three trajectories at every timepoint:

1. **GOLD** — operator-driven success TCP+object pose (e.g. `insert_u_orange_20260505_193645`)
2. **FAIL** — best/closest autonomous fail with the same compliance settings (e.g. `insert_u_orange_20260505_193941`, or the closest in-progress iteration)
3. **PROJECTED** — CAD-derived `predicted_tcp_at_seat` from `meta.json:cad_prediction.predicted_tcp_at_seat`

For each timepoint, compute:
- `tcp_xyz` distance to PROJECTED, broken into xy + z
- `tcp_quat` → tilt magnitude (from canonical face-down) + yaw
- `obj_pose` via `tcp_to_object_transform` (rotation-only chain) + per-row `obj_qx..qw` columns
- Joint torques `j0..j5_eff` from joints_raw
- Native-rate wrench at the moment of contact transition (sub-100 ms structure)

The reference script: `analysis/scripts/20_three_way_pose_diff.py`. If it doesn't fit your specific question, write a new script and add it to `analysis/scripts/`.

**The GOLD/FAIL diff** tells you what the operator does that the algorithm doesn't.
**The GOLD/PROJECTED diff** tells you the CAD prior's accuracy on this run (and how much your control law must compensate).
**The FAIL/PROJECTED diff** tells you whether the algorithm is steering toward the wrong target or whether targeting is fine and execution is wrong.

## 7. The right iteration loop (data-derived structural changes)

Each iteration must follow this loop. Skipping any step is how the May-5 session burned ~9 iterations before the correct approach was identified.

```
1. Run a cross-run analysis script on raw data (3-way diff or specialized analyzer)
2. Identify ONE specific feature where GOLD and FAIL diverge
3. Hypothesize a STRUCTURAL change motivated by that finding
   (new state, new feedback signal, new control mode — NOT a magic-number tweak)
4. Make the smallest possible code change that implements the structural insight
5. Test on robot ONE attempt, hands-off
6. Run the cross-run analysis script again — what's still different?
7. Loop
```

**What "structural" means:**
- Adding a new feedback signal to the control law (e.g. tilt-direction steering)
- Changing the state-transition predicate (e.g. obj_tilt-relaxation as alternative to z_drop)
- Adding/removing a state in the FSM
- Changing the control variable (e.g. distance-to-seat vs Fz_t)
- Changing the controller mode (force vs admittance)

**What is NOT structural (don't iterate on these alone):**
- `find_hole_directed_F_N`, `find_hole_directed_dwell_s`, `damping_find_hole`, `gain_find_hole`, `spiral_F_max_N`
- Any single number whose change you can't justify with a specific data observation
- "Try 0.7 instead of 0.5 because maybe it's better"

Parameter values exist to express structural designs — derive them from operator demos when possible, otherwise leave defaults and change structure.

## 8. Compliance settings confound comparisons

The May-3 controller settings (`gain=1.0 damp=0.7`) are the project's canonical compliance. They work for both:

- **Operator demos**: operator must overpower the controller's resistance (~5 N net hand force at peak). The 53 May-3 assisted demos all worked at this setting.
- **Autonomous attempts**: the controller is stiff enough to track the commanded `cmd_fz=-9 N` reliably during APPROACH.

Looser settings (e.g. `gain=1.5 damp=0.025`) feel less resistive in operator-driven mode but **break autonomous APPROACH** (controller too sluggish to push peg into rim hard enough to register contact within 30 s).

**Rule:** when comparing autonomous-vs-operator runs to derive control laws, hold compliance constant at `gain=1.0 damp=0.7`. Otherwise the diff confounds two variables.

## 9. Anti-patterns committed in the May-5 session — DO NOT REPEAT

| # | Anti-pattern | Why it's wrong | Right approach |
|---|---|---|---|
| 1 | Treating `hole_xy_prior` as a hard target | Stale; off by 4–6 mm per run. | Treat as soft initial estimate; refine online via tilt feedback. |
| 2 | Using `z_drop_so_far` as the sole control metric | Misses orientation, force-pattern, object-pose signatures. | Use 6-DOF + obj_pose + tilt evolution. |
| 3 | Iterating on `F_max` / `dwell_s` / `damping` values | Parameter sweep without structural reason. | Find structural divergence in data first. |
| 4 | Comparing runs at different gain/damping | Two variables confounded. | Hold compliance constant. |
| 5 | Treating FSM "STUCK" verdict as ground truth | Verdict comes from same buggy predicate. | Look at raw CSV for actual peg motion. |
| 6 | Letting tilt go unused as a feedback signal | The signal is sitting right there in the quat columns. | Close the feedback loop on tilt magnitude + direction. |
| 7 | Allowing rotating force direction (R<0.88) | GOLD has R=0.88 (fixed direction); operator does NOT rotate during search. **See correction below — 0.88 does not reproduce.** | Fixed direction toward best seat estimate; refine estimate online via tilt. |

> **CORRECTION 2026-08-16 (anti-pattern 7).** Re-measured across the full corpus (79 GOLD
> demos, all four parts), straightness `R = |net displacement| / path length` over the
> contact→SIGUSR1 window is **0.150 / 0.361 / 0.475 / 0.597** (yellow / green / brown /
> orange) — not 0.88. Operators are *directionally persistent*, which is not the same as a
> fixed direction, and 0.88 may have come from a different window or metric. The comparative
> claim still holds and is strong: GOLD is **2.6–5.1× straighter than the autonomous spiral**
> in every part (AUTO_OK: 0.058 / 0.139 / 0.093 / 0.148). Do not build a controller that
> assumes a near-straight operator path.
| 8 | Using `F=0` perfect-zero dwell windows | GOLD's `T_lat` relaxation came from operator-introduced motion, not perfect zero. | Sustained sub-tolerance push during chamfer engagement window. |
| 9 | Using FSM stdout / signals.json labels as truth | Circular — labels derived from same model under test. | Recompute labels from raw CSV with a script. |
| 10 | Using CAD `predicted_tcp_at_seat` as ground truth | Empirically 11+ mm off. | Treat as estimate; refine via contact feedback. |

## 10. Hard rules — encoded in `analysis/PROMPT.md` and enforced by review

These are NOT to be violated even with strong empirical motivation, without an explicit operator approval and a recorded reason in `STATE.json:tried_and_refuted`:

- `cmd_fz` magnitude ≤ 9 N. Operator's `|Fz_t|` peaks at ~17.9 N (operator's hand contributed ~10 N), but autonomous can't replicate this within the safety envelope. Compensate via geometry (tilt-aware steering, online seat estimate refinement, chamfer engagement detection via orientation derivative) — not by raising `cmd_fz`.
- `|cmd_F_lat|` ≤ 6 N.
- `selection_vector` must remain all-True (force-controlled in all 6 DOFs) — **EXCEPT IN SEARCH.**

  > **CORRECTION 2026-08-16 — read this before "fixing" the code to match the rule above.**
  > SEARCH commands `(True, True, True, False, False, False)`: **rotation LOCKED**. This is
  > required, not an oversight. An agent read the all-True rule as stated, unlocked rotation at
  > both SEARCH command sites, and broke the insert for five consecutive real-arm runs.
  >
  > Mechanism (observed physically by the operator): with rotation compliant, lateral force
  > applies a **moment about the grasp point**. The part pivots in the jaws while the gripper
  > translates, so the peg tip barely moves. TCP displacement stops being peg displacement, and
  > every swept-area / coverage figure computed from TCP becomes fiction — which is exactly how
  > that session concluded "the hole is not within 6 mm" about a hole 3.38 mm away.
  >
  > Evidence: every 2026-05-07 run that actually seated commanded `(1,1,1,0,0,0)` during SEARCH
  > (11–24 issues per run; see the `insert_u_brown_20260507_*.cmd_wrench_raw.csv` sidecars).
  > Reverting to locked seated the part on the next run.
  >
  > The all-True rule still holds where it was derived — the XY-locked *position-tracked* case
  > refuted at v91 is a different thing from rotational compliance during a lateral sweep.
  > **Corollary:** the "Tilt-relax detector — tilt < 0.01° throughout under full 6-DOF
  > compliance" entry in the project anti-pattern list is a *consequence of this clamp*, not a
  > measurement of contact physics. Unlocking rotation on 2026-08-16 did make tilt responsive
  > (excursion 0.004° → 0.25°, peak 0.59°), but still far below the ~3° GOLD shows at chamfer
  > engagement — and it did not change the outcome. Do not cite the tilt figure as grounds for
  > unlocking, and do not treat §3/§13's tilt-steering recommendation as available in SEARCH.
- Direction of corrective force = TOWARD seat (or per Section 3, toward the tilt direction when tilt is over tolerance). NEVER counter-residual (tested at v77/v78 — destabilizing Z-rotation cross-coupling).
- All primitive subprocesses use module mode: `python3 -m primitives.X`, never script path.
- The state-independent global seat detector (added v87) must remain.
- Do not re-test entries marked REFUTED in `STATE.json:tried_and_refuted` or `v82_v97_iteration_history.json`.

## 11. Reference paths for this project

| What | Path |
|---|---|
| Raw episode telemetry | `compliant_insertion_studio/logs/insert_*.{csv, joints_raw.csv, wrench_raw.csv, cmd_wrench_raw.csv, fm_events.csv, meta.json}` |
| Canonical v1.2 GOLD operator | `insert_u_orange_20260505_193645` |
| Canonical v1.2 FAIL autonomous (matched compliance) | `insert_u_orange_20260505_193941` |
| Closest iter-6 autonomous (peg reached err=(0.7,-0.4) mm) | `insert_u_orange_20260505_203331` |
| Analysis scripts | `compliant_insertion_studio/analysis/scripts/01_extract.py` … `20_three_way_pose_diff.py` |
| Wrapper FSM source | `compliant_insertion_studio/wrapper/contact_search_fsm.py` |
| Wrapper main entry | `compliant_insertion_studio/wrapper/compliant_insert.py` |
| Default config | `compliant_insertion_studio/configs/defaults.yaml` |
| Loop runner | `compliant_insertion_studio/scripts/loop_iterate.py` (operator-paced) |
| CAD chain | `compliant_insertion_studio/wrapper/cad_lookup.py` (`predict_tcp_at_seat`) |
| Iteration history (frozen) | `compliant_insertion_studio/analysis/v82_v97_iteration_history.json` |
| Discovery + staged tracking | `compliant_insertion_studio/analysis/iterations/{discovery,staged,validated}/` |
| Trust hierarchy + invariants | `compliant_insertion_studio/analysis/STATE.json` |
| Findings narrative | `compliant_insertion_studio/analysis/FINDINGS.md` |
| Loop spec for ralph-style automation | `compliant_insertion_studio/analysis/PROMPT.md` |
| FSM stdout from prior session (NOT trusted as labels) | `compliant_insertion_studio/analysis/raw_fsm_logs/` (see CAVEATS.md) |

## 12. Maintenance — clear stale references when applying this skill

When this skill triggers and you do new work, also do an audit pass on the adjacent docs to clear stale references that contradict the methodology here:

1. **`compliant_insertion_studio/analysis/FINDINGS.md`** — early sections (§1–§9) used `z_drop`-only metrics and treated `hole_xy_prior` as hard. Newer sections (§10–§12) reflect the v1.2 schema + 3-way comparison. If you cite an old finding, verify it still holds under the orientation-feedback model in this skill; flag any contradiction with `<!-- STALE_PER_SKILL_v2: ... -->` for later cleanup.
2. **`compliant_insertion_studio/analysis/STATE.json:known_invariants`** — review I001–I016 against the orientation-feedback insight (Section 3 here). I003/I004 (operator-search direction + displacement) are still correct. I006 (path/bbox ratio) is u_orange-only and was already refuted on other objects. Some I007–I009 alignments may need re-checking against the GOLD/FAIL canonical pair using v1.2 sidecars, not the v1.1-only alignment.
3. **`compliant_insertion_studio/analysis/PROMPT.md`** — add an "auto-trigger this skill" line at the top of the Standing Instructions, so the ralph loop's discovery iterations always read this skill before proposing FSM changes.
4. **`compliant_insertion_studio/configs/defaults.yaml`** — any param under the `directed` block whose value isn't justified by data or by the most-recent staged patch should be flagged with a comment pointing to this skill.

## 13. The fastest path to a working autonomous insert (recommended next iteration)

If you're picking up this work cold, the highest-leverage structural change available — **never tested as of skill creation date 2026-05-05 21:00 PT**:

**Tilt-direction steering with online seat refinement.**

Pseudocode for `_find_hole_directed_wrench` in `wrapper/contact_search_fsm.py`:

```python
# Compute tilt + tilt direction every tick
tilt_deg = compute_tilt_from_quat(tcp_quat)
tilt_dir_unit = compute_tilt_axis_unit(tcp_quat)  # already in self._tilt_err_world

tilt_tolerance_deg = self.find_hole_tilt_tolerance_deg  # default ~3°, derive per-object from GOLD demos

if tilt_deg > tilt_tolerance_deg:
    # OVERRIDE seat-prior steering: the tilt direction IS the position-error signal
    # Move TCP in tilt direction to slide peg past the pinning rim edge
    Fx = -F_max * tilt_dir_unit[0]
    Fy = -F_max * tilt_dir_unit[1]
    zone = "TILT_OVERRIDE"
    # Track this as a candidate for the new seat estimate
    self._tilt_correction_active_t = t
else:
    # Normal steering toward best seat estimate
    Fx, Fy = directed_push_toward_seat(seat_xy_estimate, F_max)
    zone = "NORMAL_PUSH"

# Tilt-relaxation event = chamfer engaged → latch new seat estimate
if self._tilt_was_above_tolerance and tilt_deg < tilt_tolerance_deg - 0.5:
    # Sustained drop confirms relaxation; peg slipped into slot at this xy
    self._seat_xy_refined = (tcp_xy[0], tcp_xy[1])
    # Trigger ENTRY_SETTLE despite low z_drop
    return signal_entry_settle()
```

Combine with: log `tilt_deg` in `cmd_wrench_raw.csv` so post-hoc analysis can correlate tilt with successful chamfer engagement events. Derive `find_hole_tilt_tolerance_deg` per-object from GOLD demos using a new analysis script.

This was the explicit recommendation at the end of the May-5 session before pause. Validate by running ONE attempt at-robot, then 3-way diff against GOLD.

---

## Quick checklist before any code change to insertion control law

- [ ] Did I run the 3-way pose comparison script on the relevant episodes?
- [ ] Have I identified the SPECIFIC feature where GOLD and FAIL diverge?
- [ ] Is my proposed change STRUCTURAL (new feedback signal / new state) rather than a parameter sweep?
- [ ] Am I closing a feedback loop on TCP orientation?
- [ ] Are compliance settings (gain/damping) held constant between the runs I'm comparing?
- [ ] Does my change avoid the 10 anti-patterns in Section 9?
- [ ] Does my change comply with all hard rules in Section 10?
- [ ] Will I write the change as a staged patch under `analysis/iterations/staged/<NNN>-<slug>/` with replay validation, before applying at-robot?
- [ ] Did I check `STATE.json:tried_and_refuted` and `v82_v97_iteration_history.json` to verify I'm not re-testing a refuted hypothesis?

If any answer is "no", stop and address it before changing code.

---

## 11. Fold-symmetry mirror in fixed priors — case study (2026-05-05)

**Hard rule:** never pass a `hole_xy_prior` extracted from one GOLD demo into a different run. It can be the **mirror image** of where the actual seat is for the new run's `held_quat`, and the algorithm will steer the peg AWAY from the slot, not toward it.

### What happened

Stage A run (`insert_u_orange_20260505_224205`) used `--hole-xy-prior 0.0341 -0.3635` — extracted from a May-4 GOLD demo. Outcome: TCP steered from `(-24, -352)mm` toward `(+1, -361)mm`, which was *away* from the actual slot. The peg never reached chamfer.

Operator confirmed visually that the slot was in **-X** from final TCP, contradicting the override which pointed to **+X**. Empirical verification:

- `predict_tcp_at_seat` for THIS run's `held_quat=(-0.0296, 0.7065, -0.7065, 0.0296)` returns **(-25.51, -351.50)mm** with `fold_pos_err=2.14mm` ✓
- Same function called with the 180°-rotated fold equivalent returns **(+25.51, -351.50)mm** with `fold_pos_err=26.50mm` (rejected fold)
- May-3 successful operator-seated cluster: median **(-26.84, -358.45)mm** ← matches fold-A
- May-4 successful operator-seated cluster: median **(+26.27, -361.61)mm** ← matches fold-B
- These two clusters are **mirror-image across X≈0**, the X-center of base1

### Why it happened

`rotate_object` picks one of two fold-equivalent EE orientations to hold the peg in canonical face-down. The two equivalents are 180° rotated around the peg's symmetry axis. `predict_tcp_at_seat` correctly returns a TCP target consistent with the picked equivalent. But a `hole_xy_prior` *extracted from a different demo* may correspond to the OTHER equivalent — in which case it is the mirror of the correct target.

The wrapper accepts the override as the FSM's steering target without checking which fold equivalent it's consistent with. So the override silently overrules the (correct) CAD-chain prediction with a (mirrored) wrong one.

### Hard rules from this case

1. **Never store a fixed `hole_xy_prior` for an object across runs.** Run-to-run, the correct prior depends on `held_quat` which depends on `rotate_object`'s IK choice, which can be either fold equivalent.
2. **`predict_tcp_at_seat` IS the correct chain.** It accounts for `held_quat`. Use its live output for THIS run's quat. Never cheat with a hard-coded XY.
3. **A "GOLD-empirical seat XY" is meaningful only paired with the held_quat from that demo.** If you must compare against GOLD, also store the GOLD demo's `held_quat` and only use the GOLD seat when the new run's `held_quat` matches (within fold-equivalence) the GOLD demo's.
4. **Before iterating on the FSM, every run must pass a sign sanity check:** is `predicted_tcp_xy − tcp_xy_at_HOVER_end` ≈ 0? If not, the algorithm is starting at a different fold than its target — diagnose that BEFORE adjusting any parameter.

### Methodology rule (per-iteration verification)

After every autonomous run, before proposing any change:

1. Run 3-way comparison: `actual_TCP_path` vs `predicted_TCP_at_seat` (this run's held_quat) vs `GOLD_TCP_path` (matched fold equivalent, if available).
2. Verify `predicted_TCP_at_seat` ≈ `tcp_xy_at_HOVER_end`. If they disagree by >5mm, hover and FSM disagree on fold equivalent — STOP and diagnose.
3. Verify the **direction** of TCP motion during FIND_HOLE matches the direction from `tcp_at_contact` toward `predicted_TCP_at_seat`. If TCP moved AWAY from the prediction, the steering vector is sign-inverted somewhere.
4. Only after the prediction-vs-actual sign agrees, proceed to analyze force/torque and propose structural changes.

This sign sanity check would have caught the Stage A bug in 5 minutes instead of after a full 30s wrapper run.

---

## 12. Mental model correction: TCP-at-canonical = peg-at-canonical (2026-05-05 PM)

**Hard rule:** Never reason from the CAD chain's `fold_symmetry_used.angle_error_deg` (or any object-orientation quaternion derived through `R_grasp = canonical_EE.T @ R_object_seed`) as if it were a real physical offset. **It isn't.**

### Why
The objects we work with are **prismatic** (rotationally symmetric around the peg axis). Camera-based pose detection of prismatic parts has unreliable yaw — multiple physical orientations look identical to the camera. The whole reason `rotate_object` drives TCP to a fixed canonical face-down EE pose (rather than trusting the seed orientation) is to bypass this camera ambiguity:

> **TCP at canonical face-down ⟹ peg at canonical orientation.** Period. The prismatic geometry guarantees this — you can rotate the peg around its own axis any amount and it still fits the slot the same way.

The seed `current_object_orientation` passed into `rotate_object` is camera-detected and propagates noise into `R_grasp`. The *predicted* held-object orientation (`TCP × R_grasp`) downstream inherits this noise. So a `fold_angle_err` of 4–5° in the CAD chain meta does NOT mean the peg is physically held 4–5° off canonical — it means the camera's yaw read at grasp time was off by that much.

### What this means for control-law analysis

- The peg orientation is whatever TCP × canonical-grasp puts it at. With TCP commanded to canonical face-down by `rotate_object`, the peg is at canonical regardless of what the held_quat / fold residual numbers say.
- Tilt observed during ACTIVE (`_tilt_deg` from EE Z-axis vs world -Z) IS real-world TCP orientation feedback. Subtracting any "baseline" derived from held_quat math is wrong — the held_quat math is fictional.
- Differences in `fold_angle_err` between GOLD and FAIL runs are NOT a discriminator of insertion success. Earlier in this session I treated a 0.3° vs 4.8° gap as causally significant; it's not. Both runs had the peg physically at canonical.

### When to actually use the seed quat

Only as input to `rotate_object`'s computation of which fold-equivalent EE cardinal to pick. The seed determines which orientation "neighborhood" we're in (peg pointing this way vs. that way). The cardinal-snap that follows discards the within-cardinal noise — that's by design, not a bug.

### Anti-pattern from this session

I diagnosed a 4.8° "object misalignment" as the root cause of Stage B's failure and proposed modifying `rotate_object` to "compensate." This would have actively introduced misalignment by rotating TCP off canonical based on camera yaw noise. The user (rightly) caught this before code was changed. The mental-model fix in this section prevents the pattern from recurring.

### Where to look instead for Stage B-class failures

When peg orientation is ruled out (TCP confirmed canonical), failures during FIND_HOLE / ENTRY_SETTLE are about XY position and Fz dynamics:
- CAD prediction has known mm-scale XY error (~10mm typical for u_orange)
- Peg lands on rim near slot but not in it
- Lateral push without intermittent Fz release means peg slides across rim without falling into chamfer
- Tilt rises on real rim contact; tilt-direction push is geometrically reasonable but doesn't generate the z-drop opportunity needed for engagement

---

## 13. GUIDED-mode data collection — operator labels Found Hole (2026-05-06)

The methodology that finally produced clean GOLD-equivalent data. **Use this for any new object's data collection.**

### Mechanism

Wrapper has a `--guided-mode` flag. With it set, the FSM routes APPROACH-Contact → new `GUIDED` state instead of FIND_HOLE. In GUIDED:
- `selection_vector = (T,T,F,F,F,F)` — XY compliant, Z LOCKED, rotation LOCKED
- `wrench = zero` — pure compliance, operator drags EE laterally
- Robot is "gimbal-stabilized" — peg height + orientation locked while operator moves it across the rim

Operator drags peg above the slot, presses Enter in the collection script. SIGUSR1 fires → wrapper's FSM `mark_hole()` captures `tcp_xy` as `hole_observed_operator` in meta. FSM transitions to `INSERT_DESCENT` (autonomous Z descent at the marked xy). The global seat detector fires when `|tcp_z - predicted_tcp_z| < 5mm` sustained → DONE.

### Why this works

- **Each demo IS a labeled Found-Hole sample.** Operator's SIGUSR1 timestamp is the ground-truth "hole reached" moment.
- **Same trajectory captures rim sliding, hole engagement, and seat descent** in one continuous F/T + tilt + cmd_wrench stream.
- **No per-demo annotation effort.** Drop the part on the rim, drag, press Enter, robot finishes.

### Required preconditions

- `launch_robot.sh real` was used (auto-applies `set_payload`)
- F/T post-zero residual subtracted from contact-detection (already in wrapper)
- APPROACH grace period 1.0s (suppresses force-mode startup transient)
- Rotation-locked selection vector throughout (eliminates yaw-drift-via-grasp-lever)
- Per-demo `physical_check` verifier in collection script (catches phantom seats)

All of the above are settled — don't re-derive.

### What to vary in collection

Vary STARTING xy via `--base-offset-xy DX DY` (script: `compliant_insertion_studio/scripts/collect_regime_data.py`). The point is **direction-invariance validation**, not direction coverage. 3 directions × 3 reps is sufficient if the marker is direction-invariant by physics (it should be — see §14).

---

## 14. Marker (direction-invariant) vs Director (direction-dependent)

**Don't conflate these. Repeat: don't conflate these.**

### Marker = local sensor signature → fires when peg is at a specific contact regime

- **Contact**: fz crossed threshold (any direction)
- **Found Hole**: tilt-relax + F_lat collapse + dz/dt onset (any direction)
- **At Target**: |tcp_z - predicted_tcp_z| < tol + motion stopped + tilt low (any direction)

These are **direction-invariant by construction**. The signal is local to where peg currently is — doesn't depend on where peg came from or which slot side it approaches from. **Computed in TOOL frame, not world frame.**

### Director = "which way to move now" → world-frame vector

- **In GUIDED mode**: operator's hand IS the director (via the drag).
- **In autonomous mode**: director = vector from current xy toward `predicted_tcp_xy` (CAD chain) refined by `base_calibration_*.yaml`.
- Has a definite direction; depends on world geometry; NOT something you derive from operator data.

### Hazard pattern (caught 2026-05-06)

A subagent computing alignment between operator drag direction and `-r_cop` in **world frame** observed bimodal sign across variations and concluded "anomaly across variations — need more data." Wrong diagnosis. The correct fix:

1. Recompute the alignment in **tool frame** — direction-invariant by construction.
2. If still bimodal, the predicate has a frame conversion bug — fix structurally, not by collecting more data.
3. The 3+ directions in the GUIDED dataset are the *test* of invariance, not a coverage gap.

### When to actually collect more data

If the marker predicate, computed in tool frame, fails to fire at the SIGUSR1 timestamp for some variations but succeeds for others. That points to a real direction-dependent effect needing investigation. Otherwise, **3 directions × 3 reps is enough** to test direction-invariance.

---

## 15. Frame discipline — tool frame for sensor features

When deriving any predicate from F/T or torque data, default to **tool frame** (`tool0_controller`). Convert to base/world only for visualization or for matching against world-frame TCP positions.

Why:
- Sensor frame IS tool frame natively
- Local contact geometry (rim, chamfer, slot wall) is most cleanly described in tool frame
- World-frame conversions introduce TCP-orientation-dependent sign flips that confound direction-invariance testing
- Any feature that should be direction-invariant (any local-contact predicate) is invariant in tool frame by construction

CSV's `wrench_frame_id` column will say `tool0_controller` for the raw wrench sidecar (`<basename>.wrench_raw.csv`). Use that, not the base-frame transformed values in the main CSV.

---

## 16. F/T bias and force-mode transient discipline (2026-05-06)

Two corrections that were missing and caused phantom-contact bugs:

### Post-zero F/T bias subtraction

- Wrapper samples post-zero residual bias in PRE/ZERO phase, stores on `ep.post_zero_bias_baselink`
- Subtract from each Fx/Fy/Fz/Tx/Ty/Tz before passing to FSM
- Without this: residual bias > contact_threshold_N causes phantom contact at hover
- CSV records RAW (uncorrected) wrench. For post-hoc analysis: `corrected_fz = csv.fz - meta.post_zero_bias.Fz`

### APPROACH grace period

- 1.0s after ACTIVE start, contact detection ignored
- Lets force_mode_controller startup transient settle (raw fz can oscillate ±5N for ~0.5s)
- Real APPROACH descent takes ~10s; sacrificing 1s is safe
- Configurable via `approach_grace_period_s`

Combined with the 3N threshold + 0.1s smoothing + bias subtraction, contact detection is now robust to:
- Sensor noise during force-mode startup
- Residual F/T zero bias up to ~2.5N
- Brief operator-induced perturbations during the STEP-BACK gate

### 16.3 Settle the ARM before taking any force reference (2026-08-16)

**Hard rule: never sample a force zero or baseline while the arm is still moving.** The
trajectory controller reports "complete" when the commanded position is reached, not when the
mechanical ring-down has finished. A reference taken in that window is confidently wrong and
nothing downstream can detect it — the failure mode is always a *plausible number* that
something then acts on, never an error.

Found in four independent places on 2026-08-16. Every one waited *after* the sample; none
waited *before* it:

| site | what it waited for | what was missing | symptom |
|---|---|---|---|
| `compliant_insert` ZERO phase | step-back gate, set to 0.0s | arm settle before `zero_ftsensor` | post-zero bias **15.6 N**, force_mode drove TCP **116 mm UP** |
| `translate_object` insert | same, `--auto-step-back-seconds 0.0` | same | same path |
| `move_to_grasp` baseline | `sleep(0.1)` | arm settle before sampling | phantom **59.6 N** "contact" |
| `move_to_grasp` sensor zero | `sleep(0.5)` **after** zeroing | arm settle **before** zeroing | phantom **56.5 N** "contact" |

Calibration measured this session: **0.0 s settle → 15.6 N residual; 5.36 s settle → 0.11 N.**
Use ≥1.5 s after a Cartesian move; anything between 0 and 5.36 s is uncharacterised.

Guard rather than trust: `--bias-abort-n` (default 5 N) now **aborts** rather than subtracting
a residual that large, because a bias that big is a botched zero, not sensor bias. Good runs
sit at 0.07–0.54 N, so the margin is ~10×.

### 16.4 Contact predicates need a sustain window, not a threshold crossing

The wrench stream carries brief impulses that are indistinguishable from contact on a single
sample. Measured at 50 Hz during a `move_to_grasp` descent, nothing touched:

```
+0.18, +19.32, +56.98, +20.56, +0.36 N     ~60 ms, symmetric, from and back to a quiet baseline
```

That tripped a 40 N single-sample threshold. Peak amplitude varies 40–60 N run to run, so
re-zeroing, re-baselining and slowing the descent all move the number without fixing anything —
three separate "fixes" were attributed to it before the raw trace was recorded.

**Record the trace before theorising about a force number.** Real contact holds an elevated
force; an impulse does not. `move_to_grasp` now requires `z_force_sustain_s = 0.15`, mirroring
how the v4 predicate uses `off_sustain_s`. Any new force predicate needs the same treatment.

---

## 17. At-Target marker uses absolute predicted_tcp_z (multi-contact safe)

Old design: At-Target = `surface_z - tcp_z >= 25mm`. Resets on every Contact event because surface_z re-latches. **Fails for multi-contact insertions** (peg encountering features along the way).

New design: At-Target = `|tcp_z - predicted_tcp_z_at_seat| < 5mm` + motion stopped + tilt low, sustained 1s.
- `predicted_tcp_z_at_seat` is the absolute CAD-derived target z — invariant across the trajectory
- Multi-contact loops can re-latch surface_z without affecting At-Target
- Empirical validation 2026-05-06: peg seated 1.6mm BELOW CAD prediction (well within 5mm tolerance)

Plumbed: wrapper passes `predicted_tcp_z_at_seat` (already computed for hover) to FSM via `predicted_tcp_z` constructor arg. FSM stores as `self.predicted_tcp_z`. Global seat detector prefers absolute path; falls back to relative z_drop only when CAD prediction is unavailable.

---

## 18. Methodology: the 4-marker model (final)

The user's proposed FSM consolidation, validated through the GUIDED collection:

```
States (3):  Inserting | Aligning | At Target
Markers (4):
  1. Contact     — Inserting → Aligning      (peg-bottom touches a surface)
  2. Found Hole  — Aligning → Inserting      (peg cleared rim, descending into slot)
  3. Contact    — Inserting → Aligning      (re-entry; multi-contact loop)
  4. At Target   — Inserting → DONE          (peg fully seated)
```

3 of 4 are autonomous predicates (Contact #1, Re-Contact #3, At Target #4). Found Hole #2 is the operator-labeled marker; the autonomous predicate is derived from cross-demo analysis of GUIDED data.

**For autonomous insertion on any new object/base:**
1. Collect GUIDED demos (varied starting xy, 3+ directions × 3 reps)
2. Aggregate Found Hole signature across demos in TOOL frame (per §14, §15)
3. Validate predicate fires at operator's SIGUSR1 timestamp ±300ms
4. Replace GUIDED-mode SIGUSR1 path with autonomous predicate in FIND_HOLE state
5. Run base calibration from `hole_observed_operator` measurements (per `BASE_CALIBRATION_FROM_HOLE_OBSERVATIONS.md`)
6. Validate end-to-end autonomous insertion without operator intervention

That's the complete pipeline. Each step is the test of the previous.

---

## 19. Autonomous SEARCH director — what worked (2026-05-06 session)

The 2026-05-06 session reached fully autonomous insertion on u_orange (6/6 success including 3 with fresh regrasps) and u_brown (multiple successes), with two distinct hard-won architectural lessons. inverted_u_yellow exposed the limits and pointed the way forward.

### 19.1 The "bias" was two confounded things

What we initially called a single "calibration bias" was actually:

1. **`b_obj` — fixture/base calibration error**: stable per-base-position. Belongs in `DEFAULT_BASE_POSITION` (`primitives/shared/config.py`). Derived from a single GUIDED demo with a centered grasp. Empirically (-3.66, -5.81) mm for u_orange but contaminated by grasp; the centered-grasp u_brown demo gave the true value (+1.52, -3.78) mm.
2. **`g` — per-grasp offset**: random, ±1-3mm, unknown a priori. Belongs in the SEARCH director's local exploration radius. **CANNOT be hardcoded** because every regrasp is different.

The original FSM `bias_x = -0.00366, bias_y = -0.00581` baked u_orange's `b_obj + g_avg` into the spiral center. That made `u_orange` work by coincidence (offset peg from spiral center provided traversal direction) but generalized poorly.

**Correct architecture**:
```
DEFAULT_BASE_POSITION = calibrated value (one-time)   ← b_obj
Spiral center = predicted_tcp_xy from CAD              ← no FSM-side bias
r_launch = 1.5mm                                        ← absorbs g
Spiral R_max = 8mm                                      ← absorbs g + b_obj_residual
```

### 19.2 Friction positive-feedback ruled out -F_lat / -r_cop as autonomous direction signals

In operator demos, `align(-F_lat_sensed, →hole)` and `align(-r_cop_base, →hole)` were +0.6 to +0.8 in late-search Q4. **This was an artifact**: operator's drag direction = friction direction = -F_lat direction, AND operator chose to drag toward the hole. So -F_lat aligns with hole because operator chose hole, not because -F_lat is autonomously useful.

Iter 1+2 with `-r_cop` / `-F_lat` autonomous direction control: peg drifted in self-confirming direction, never crossed rim. Confirmed by GPT analysis.

**The autonomous direction signal must come from somewhere else.** Specifically:
- Spiral search (blind-but-bounded) for the local-grasp-offset disk (Chhatpar/Branicky 2001, Jasim 2014)
- v4 detector (|fz| state transition) for the rim-cross event

### 19.3 Constant-force replaces PD position tracking

Default PD gain Kp=350 N/m gave ~1.3N commanded force at typical 0.5-4 mm position errors — below stiction → instant stall. **Constant-force tracking** (always Fmax in unit(error) direction) breaks stiction reliably.

```
if |e| > 1e-6:
    F_xy = -Fmax * unit(e) - Kd * v_tcp     # sign-flipped per empirical convention
```

The sign-flip (`F_xy = -Fmax * unit(e)` instead of `+Fmax * unit(e)`) is an empirical observation: peg moves opposite of commanded F_lat direction in our base_link↔base 180° setup. `verify_baselink_motion.py` validated single-axis behavior; multi-axis under SEARCH inverted in practice. Not derived from first principles — **measured and applied as a constant**.

### 19.4 Lag-pause: theta only advances when peg tracks ref

Without lag-pause, spiral ref orbits at v_s while peg lags by 1-3mm under stiction. Stall detector trips because spiral arc grows faster than tcp progress. With lag-pause:

```
if peg-to-ref distance ≤ 2mm: advance theta
else: hold theta until peg catches up
```

This is self-paced spiral. Spiral arc-length tracking for stall detection ALSO gates on `spiral_advanced` so a paused spiral doesn't trip the stall detector.

### 19.5 r_launch = 1.5mm prevents peg-at-center stall

When DEFAULT_BASE_POSITION is correctly calibrated AND grasp is approximately centered, peg lands ON the spiral center. Initial position error is 0. PD or constant-force command magnitude is undefined / tiny → no traversal → stall.

`r_launch = 1.5mm` makes the spiral start 1.5mm offset, guaranteeing peg has direction to traverse even at zero grasp offset. Verified: u_brown autonomous succeeded in 1.11s when r_launch was active vs 0.5mm prior config that stalled instantly.

### 19.6 Gradient-following: active control kicks in when `|fz|` is dropping

User insight: when `|fz|` is dropping, that direction is toward chamfer. Continuing the spiral (rotating to next angle) PULLS peg back to rim. Active override needed.

```
d_fz_dt = ( |fz|_now - |fz|_200ms_ago ) / 0.2
if d_fz_dt < -3 N/s AND |fz| < 6:
    # peg moving from rim into chamfer
    F_xy = -Fmax * unit(v_peg)        # continue current peg motion direction
else:
    # normal spiral PD
```

This deviates from the spiral path when sensors say "hole is in current direction." The peg is pushed deeper into the chamfer rather than orbiting away.

### 19.7 Global seat detector must run in SEARCH and APPROACH

Verified case 2026-05-06: u_brown peg fell straight through during APPROACH, briefly bumped fz>3 entering SEARCH, then stalled. tcp_z was 1.7mm below predicted_seat with motion stopped throughout. The detector was excluded from SEARCH/APPROACH state list, so the wrapper missed an obvious success.

**Fix**: include APPROACH and SEARCH in the seat-detector state set. Peg-already-seated is detected anywhere.

### 19.8 Multi-prong parts need lower F_press

`inverted_u_yellow` failed where u_orange/u_brown succeeded. Operator-mode GUIDED has Z **LOCKED** (selection_vector `T,T,F,F,F,F`); autonomous SEARCH has Z **COMPLIANT** with commanded Fz=-9N. Result:
- Operator's `|fz|` during drag: median 3.14N, p5=0.20N, 50% time <3N
  > **CORRECTION 2026-08-16:** these figures are specific to this object/window and do NOT
  > generalise. Across the corpus (contact→SIGUSR1, tool frame), GOLD median `|fz|` is
  > **3.30–8.89 N** depending on part, and the fraction below 3 N ranges **7.7%–45.3%**.
  > The *self-normalising* version (fraction below 0.5× that episode's own median) is
  > 6.1% / 35.9% / 24.5% / 15.6% — a 29.8-point spread, so there is no single operator
  > "unload duty cycle" to copy and it cannot be used as a cross-part target. What does hold
  > in all four parts: successful autonomous runs unload MORE than failed ones. Direction is
  > real; the level is part-specific.
- Autonomous's `|fz|` during search: median 7.26N, p5=6.06N, 0% time <3N

For multi-prong parts (multiple contact points), Z-compliant push-down keeps `|fz|` saturated regardless of whether one prong is over a chamfer. The v4 collapse signal that worked for single-peg parts doesn't appear.

Mitigation: lower `F_press` from 9N → 5N to mimic operator's drag pressure. Doesn't fundamentally fix multi-prong but improves signal quality. Further work needed for parts where v4 doesn't fire even with reduced press.

### 19.9 Per-object configuration (gripper_width, grasp_id) from `fmb1_assembly.json`

Hardcoded `--grasp-width 35` failed for `inverted_u_yellow` (needs 56.7mm). Now `run_assembly_step.py` and `regrasp_held_object.py` auto-resolve per `(object_name, grasp_id)` via `primitives.shared.config.get_gripper_width_mm()`. The loop script auto-resolves grasp_id via `get_grasp_id_for_assembly()`. Pass `--grasp-width N` or `--grasp-id N` explicitly to override.

### 19.10 What's still open

1. **Multi-prong parts** (inverted_u_yellow): need a fundamentally different rim-cross detection that doesn't rely on `|fz|` collapse. Candidates: relative `|fz|` drop vs recent median, peg-z descent, |F_lat| pattern change.
2. **Two-stage insertion** (operator-flagged): when one part has to clear an intermediate alignment phase before the final slot. Spiral can't apply pressure on the underlying object (it'd move). Needs a separate FSM state.
3. **Per-slot calibration bias**: the centered-grasp demo for each part gives a per-slot bias. We chose NOT to bake these in (would conflate again with grasp), but a future architecture might use them as an a priori for r_launch direction.

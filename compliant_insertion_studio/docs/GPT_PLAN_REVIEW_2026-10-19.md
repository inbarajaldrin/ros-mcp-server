# GPT (Codex, 2026-10-19) review of the FMB1 hands-off game plan

Brief: the game plan + INSERT_GEOMETRY.md + the day's data, asked for a prioritized plan, what we miss, three single-part experiments, and the planner / execute-mate / vision boundary. Answer verbatim below.

Prioritized plan

1. Make the measurements trustworthy. Log timestamped camera frames, detection
   sets, base and held-part poses, commanded and actual TCP poses, gripper width,
   wrench, and every contact transition. Recompute all seat errors in the base
   frame. Correct the metric name: N·s is lateral force integrated over time,
   not mechanical work; report both that quantity and ∫F·dx.
   Measurement: replay reproduces each of the 127 insert classifications and
   the reported metrics; every trial has a trace from first contact to seat.
   Risk: old logs may lack enough data for full replay.
   Effort: 8–12 hours.

2. Make the moving base a reliable reference before changing the primitive.
   Store learned seat offsets in base coordinates. Read the base close to each
   mate, reject changes in detected marker set that cause pose jumps, and
   re-read after contact or retreat to detect a base that moved during the
   insert. Do not average the two 8 mm-apart detection states.
   Measurement: deliberately move and turn the base between single-part
   trials; the same physical seat error stays stable in base coordinates, and
   a moved base is detected before the next action.
   Risk: contact may move the loose base during an insertion, leaving no valid
   static reference for that attempt.
   Effort: 16–24 hours. OpenCV’s board-pose guidance supports using several
   marker corners and surviving partial views; it does not establish that
   your two agreeing yaw estimates are physically correct.
   [OpenCV board detection](https://docs.opencv.org/4.7.0/db/da9/tutorial_aruco_board_detection.html)

3. Identify the source of the seat bias with an independent reference.
   Check camera-to-tool calibration, marker-to-CAD transforms, seated mesh
   geometry, and physical pocket locations. Survey or gently touch a known
   base feature and compare it with the camera estimate. Repeat before and
   after a controlled base rotation. The consistent −y residual suggests a
   systematic transform or seat-model error; growth after the knock especially
   implicates stale robot-frame offsets or an incorrect base transform. It
   does not prove camera bias or finger slip. A 2° yaw error produces about
   2.4 mm at a 70 mm lever arm.
   Measurement: the residual is assigned to a specific transform or physical
   contact effect; its prediction holds after a second base move.
   Risk: touching a feature can itself shift the loose base.
   Effort: 16–24 hours.

4. Validate and then use the new held-part pose read. Measure the part
   relative to the gripper once after each pick or regrasp, resolve fold
   symmetry using the intended mate, and chain that measured transform from
   the arm thereafter. Correct xy as well as yaw. Compare the prediction with
   a second safe-height view after a long move to test for slip.
   Measurement: held-pose correction reduces the first-contact residual and
   force-time integral on paired trials; the second view agrees within the
   error budget required by the mate. For yellow’s ±0.5 mm pocket clearance,
   that budget must be substantially below 0.5 mm.
   Risk: the tool camera can measure relative grasp error, but shared camera
   calibration errors can also make the base and held-part readings agree
   falsely. An independent contact or metrology check remains necessary.
   Effort: 12–20 hours.

5. Add a corridor-wedge exit, then tune entry. Detect a wedge from depth
   progress stopping while force persists, lateral force rising, or commanded
   motion diverging from observed progress. Do not wait for TCP tilt: the pads
   hide the part’s tilt. Stop downward motion, unload, retract above the
   corridor entrance, re-estimate the part/seat relationship, and retry from
   there with bounded xy correction and Rz compliant. Test small dither only
   after the part is unloaded or retracted; dithering at 5 N inside a
   ±1 mm corridor may tighten the jam. Never use a broad yaw search there.
   Measurement: fewer stalls at 5–15 mm depth, lower lateral force-time
   integral, and higher attempt-1 completion without increasing peak force
   or damaging parts.
   Risk: wrist force cannot reveal the exact contact state inside compliant
   fingers; a second sensor or visual check may be needed.
   Effort: 16–24 hours. UR force mode controls the selected axes in a task
   frame and its limits mean different things for compliant and rigid axes;
   verify the actual selection vectors and limits during replay.
   [UR force-mode documentation](https://www.universal-robots.com/manuals/EN/HTML/SW5_26/Content/prod-scriptmanual/all_scripts/force_mode_task_frame_selection_vector.htm)

6. Generalize execute-mate only after those errors are measured. Derive free
   approach, surface search, corridor passage, and seat verification from
   the mate trajectory, CAD contact geometry, clearance, and pose uncertainty.
   Keep global force and motion safety limits. Give execute-mate no part IDs,
   learned per-part offsets, or hard-coded stage heights.
   Measurement: the same stage generator handles one U, then line_green,
   then yellow without changing its code or settings; CAD replay flags
   collisions and clearance violations before physical trials.
   Risk: CAD misses pad deformation, friction, and real edge geometry.
   Effort: 24–40 hours.

7. Harden the cycle supervisor. Record every cycle, classify failed grasps,
   failed localization, wedges, and moved-base events, and route each to a
   bounded recovery. Start only when the workspace light is on. Judge progress
   from raw attempt-1 rate, contact-to-seat time, base-frame seat deviation,
   lateral force-time integral, and stalls, grouped by layout and part.
   Measurement: 30 consecutive single-part trials per condition as an
   initial screen, then at least 100 full generated-layout cycles for a
   reliability claim; report failures as well as recoveries.
   Risk: three clean cycles reveal little about rare failures.
   Effort: 12–20 hours.

I would put base-free coordinates and pose quality before the unified
primitive. The primitive would otherwise learn around a moving reference.
The CAD stage work can proceed offline in parallel, but it should not drive
the arm until the reference and grasp-error measurements pass their checks.

What seems missing or wrong

- “Wrist yaw matches camera base yaw” says nothing about part yaw in the
  fingers. The visible correction at contact and seat orientation deviation
  are stronger evidence of a held-part error.
- Agreement between marker yaw and depth-outline yaw is weak evidence if
  both use the same camera, calibration, and view. The 8 mm detection-set jump
  is larger than every tight clearance in this assembly.
- TCP-at-seat minus CAD aim is not necessarily part-at-seat minus CAD aim.
  The compliant grasp can absorb translation and rotation. Measure the
  seated part separately.
- The 24.5 mm mesh-height error shows CAD-derived stages need geometry
  validation before use. A generic algorithm fed wrong geometry will be
  generically wrong.
- There is no stated uncertainty or freshness gate on poses. At 3–4 Hz,
  a timestamp and an explicit “base stationary since this read” condition
  matter more than another casual read.
- The loose base may move under insertion load. Base-frame offsets alone
  cannot fix motion during contact.
- Attempt-1 success for U parts conceals a 20-fold spread in contact time
  and a roughly 60-fold spread in lateral force-time integral. “Seated” is
  too weak a quality criterion.

Three experiments first, one manipulated part at a time

1. One U, bias attribution: pick and reinsert it repeatedly at two
   deliberately different base yaw angles and positions. Keep the grasp
   method fixed. Log a close base read, held-part read, independent base
   feature check, first contact, and final part pose. A residual that rotates
   with the base points to a frame or offset error; one tied to the gripper
   points to grasp/slip; one tied to a pocket points to geometry or contact.

2. One U, held-pose A/B: alternate the new xy/yaw correction on and off
   across matched picks and layouts. Run enough trials to compare the
   distributions of first-contact error, rim time, and lateral force-time,
   not just successes. Reject a correction that improves the camera residual
   but leaves contact unchanged.

3. line_green only, corridor recovery: leave the two U parts seated and
   manipulate only the bar. Induce small, known initial xy/yaw errors within
   safe bounds. Compare current retries with “unload, retract above entry,
   re-aim, Rz compliant descent.” Record wedge depth and force trace. Try
   dither only if the retract strategy leaves a measured residual.

Layer boundary

1. PLANNER accepts the assembly graph, CAD, closing trajectories, symmetry
   classes, and available observations. It returns an ordered set of mates,
   accepted realizations, grasps, reorientations, intermediate placements,
   and layouts. It owns grasp and layout choices and the corrected CAD
   assets. It does not own force-mode axes, search patterns, or live contact
   recovery.

2. EXECUTE-MATE accepts one chosen mate trajectory, its CAD contact geometry
   and clearance, a measured base pose and held-part transform with
   uncertainty, and global safety limits. It derives stages and compliant
   axes, executes the mate, and returns a contact trace, measured correction,
   final-pose evidence, or a specific failure. It owns rim search and
   corridor-wedge handling. It must contain no U/bar/yellow constants or
   robot-frame learned seat offsets.

3. VISION returns timestamped, symmetry-aware base and part poses, marker-set
   identity, quality/uncertainty, and the held-part-to-gripper transform. It
   owns detection-set flips, fold-branch ambiguity, calibration, and the
   grasp-error measurement. It does not choose mates or infer insertion
   success from an image alone.

A thin supervisor outside those three interfaces owns the light gate,
cycle recording, retries, and stopping rules.
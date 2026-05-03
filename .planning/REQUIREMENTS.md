# Requirements: Compliant Insertion Studio

**Defined:** 2026-05-01
**Core Value:** Replace the failing `--insert` real-mode path with a force-compliant insert primitive that works reliably across every FMB1 part and is a one-config-file extension to any new part.

## Project Layout

All deliverables live under a single self-contained folder so the entire subsystem can be lifted into other robotics projects with a one-line edit at the host project's `translate_object` equivalent.

```
ros-mcp-server/
  compliant_insertion_studio/        ← self-contained subsystem
    wrapper/                         ← episode lifecycle + ROS deps
      compliant_insert.py
    dispatcher/                      ← config resolution, no ROS deps (unit-testable)
      compliant_insert_episode.py
    shared/
      insert_config.py               ← YAML deep-merge helper
      ft_smoke_test.py               ← F/T validity smoke-test primitive
    configs/
      defaults.yaml
      <object>.yaml ...
    analyzer/
      analyze_inserts.html           ← static dashboard
      assets/                        ← optional cached Plotly + PapaParse for offline
    logs/                            ← .gitignored
    docs/
      README.md
      tune_a_new_part.md             ← VAL-04 deliverable
      ft_calibration_sop.md          ← CAL deliverable
  primitives/translate_object.py     ← single integration touchpoint at :1085
```

## v1 Requirements

Requirements for FMB1 PoC release. Each maps to a roadmap phase.

**Track tags** (per `.planning/codebase/CONVENTIONS.md` two-track principle):
- `[N]` — **No-robot** — work this when away from the robot
- `[R]` — **Robot-required** — needs physical UR5e + ROS bringup connected
- `[N→R]` — **Hybrid** — code/scaffolding done no-robot, final verification at-robot

See `.planning/TRACKS.md` for the live "what's ready right now" list per track.

### Pre-Collection Setup (SETUP)

Hardware/software setup that must complete *before* demos are collected, otherwise data is contaminated.

- [ ] `[R]` **SETUP-01**: `ur_robot_driver` and `ur_msgs` upgraded to ≥ 2.13.0 (April 2026 release with F/T-frame bugfixes)
- [ ] `[N→R]` **SETUP-02**: `set_payload` is single-source-of-truth at robot bringup only (UR 5.4.x bug: mid-session re-call interacts with `zero_ftsensor`) — bringup-launch edit is `[N]`, verification is `[R]`
- [ ] `[N→R]` **SETUP-03**: Force-mode tuning baseline locked: `gain_scaling=0.5`, `damping_factor=0.7` (research consensus for low-force regime; current default `0.025` damping is too low) — code change is `[N]`, real-robot test is `[R]`
- [x] `[N]` **SETUP-04**: `pyproject.toml` numpy pin reconciled with runtime numpy 2.2.6 (lift the `<2` pin)
- [x] `[N]` **SETUP-05**: `.gitignore` excludes `compliant_insertion_studio/logs/insert_*.csv` and `*.meta.json`
- [x] `[N]` **SETUP-06**: `compliant_insertion_studio/` folder scaffolded with subdirectory layout shown above and a top-level `README.md` describing how to use it standalone (calibrate → wrap → analyze → configure → integrate)
- [x] `[N]` **SETUP-07**: Existing `primitives/compliant_insert.py` (the working scaffold validated this session — F/T zero, force_mode RPC, signal handlers, basic CSV) migrated to `compliant_insertion_studio/wrapper/compliant_insert.py` with import paths fixed. Phase 2 extends this file (adds HOVER, full schema, signal markers, hands-off gate); does **not** rewrite from scratch.

### F/T Calibration & Validity (CAL)

F/T calibration is **three layers** — see `.planning/CONVENTIONS.md` §8. CAL covers all three.

**Foundational (per-mount, one-time):** payload identification — recovers gripper+jig mass and CoG so UR's gravity compensation is correct. Without this, force mode produces orientation-dependent bias that no amount of `zero_ftsensor` can fix. Algorithm: linear least-squares per Kubus-Kröger-Wahl 2007, derived from `_references/repos/force_torque_tools/force_torque_sensor_calib/src/ft_calib.cpp` and documented in `_references/articles/ft_payload_calibration_math.md`.

**Session-level (per-session, smoke check):** quick "is the sensor still trustworthy?" probe in a known neutral pose. Catches drift/damage between sessions.

**Per-pose (`zero_ftsensor`):** runtime bias subtraction immediately before force mode — handled in WRAP, not CAL.

#### Foundational — payload calibration (one-time per gripper mount)

- [x] `[N]` **CAL-01**: New primitive `compliant_insertion_studio/shared/ft_calibration.py` — moves robot through ≥ 8 well-distributed poses (gripper-pointing direction varied across the sphere), settles + samples /wrench at each, computes payload mass + CoG + 6-axis bias via Kubus-2007 LSQ in NumPy. ~200 LOC. Algorithm derived from `_references/repos/force_torque_tools/`; written fresh for ROS2 + UR + no-accelerometer (gravity derived from EE orientation via `/tcp_pose_broadcaster/pose`).
- [x] `[N]` **CAL-02**: Output written to `compliant_insertion_studio/configs/ft_calibration_<gripper_id>_<YYYYMMDD>.yaml` with: `mass_kg`, `cog_xyz_m`, `bias_force_xyz_N`, `bias_torque_xyz_Nm`, residuals per axis, pose count, conditioning warnings, source poses logged. Versioned by gripper_id so multiple gripper assemblies can be calibrated independently.
- [ ] `[R]` **CAL-03**: Operator workflow: (1) attach gripper + jig in target configuration with no part, (2) launch standard ROS2 bringup, (3) `python3 compliant_insertion_studio/shared/ft_calibration.py --gripper-id <name>` → arm moves through ≥ 8 poses (~3 min total), (4) script prints recommended `set_target_payload(mass, cog)` line for paste into bringup launch, (5) operator pastes + commits the bringup change, (6) restarts ROS bringup once. After this, payload is correct for all subsequent sessions until gripper changes.
- [ ] `[N→R]` **CAL-04**: Pose sequence chosen for good LSQ conditioning (gravity vector in F/T frame must span at least 3 linearly independent directions; ≥ 8 poses recommended). Pose set parameterized in YAML (`compliant_insertion_studio/configs/calibration_poses.yaml`) so it can be tuned per workspace. — RViz fake-hardware preview is `[N]`, final at-bringup verification is `[R]`
- [x] `[N]` **CAL-05**: Calibration script self-validates: residual per axis < 0.5 N or 0.05 Nm, recovered mass within ±10% of expected (operator provides expected mass as `--expected-mass-kg` for sanity check). Fails loudly if poses are too similar (rank-deficient H matrix).

#### Session-level — F/T smoke test

- [x] `[N]` **CAL-06**: New primitive `compliant_insertion_studio/shared/ft_smoke_test.py` — assumes payload is foundationally correct, robot is in a steady neutral pose, calls `zero_ftsensor`, holds 5 s, samples /wrench at 100 Hz, reports residual bias per axis + drift rate
- [x] `[N]` **CAL-07**: Pass criteria: per-axis residual |F| < 2 N, per-axis residual |T| < 0.3 Nm, drift over 5 s window < 0.5 N/s in any axis. Documented in `docs/ft_calibration_sop.md`.
- [ ] `[N→R]` **CAL-08**: Smoke-test runnable standalone (`python ft_smoke_test.py`) and as PRE-phase precondition in the episode wrapper (skip ACTIVE if smoke fails) — standalone code is `[N]` (done), wrapper integration is Phase-2 `[N]`, real-robot pass-criteria validation is `[R]`

#### Documentation

- [x] `[N]` **CAL-09**: SOP doc `compliant_insertion_studio/docs/ft_calibration_sop.md` covers all three layers: (a) when to re-run foundational calibration (gripper change, jig change, sensor remount), (b) F/T sensor warm-up window (≥ 10–30 min before first session of the day), (c) when to re-run smoke (start of session, after a protective stop, after physical bumps), (d) what to do if smoke fails (re-run foundational; if foundational also fails, escalate — do not proceed)
- [x] `[N]` **CAL-10**: SOP also documents the relationship to UR's URCap "Measure" wizard (alternative to our calibration script — pendant-driven, 4 poses, less precise, but no ROS2 dependency). Operator may use either; outputs are interchangeable in `set_target_payload()`. See `_references/articles/ur_polyscope_payload_measure_wizard.md`.

### Episode Wrapper (WRAP)

The lifecycle owner: hover-zero-active-exit, signal handling, telemetry write. Lives in `compliant_insertion_studio/wrapper/compliant_insert.py`.

- [ ] `[N→R]` **WRAP-01**: `compliant_insertion_studio/wrapper/compliant_insert.py` owns full lifecycle: `PRE → HOVER → ZERO → ACTIVE → DONE/ABORT`
- [ ] `[N→R]` **WRAP-02**: PRE phase runs CAL smoke-test (or skips if `--skip-smoke` for repeated rapid attempts) and validates preconditions (gripper holding part, joint state available, target known)
- [ ] `[N→R]` **WRAP-03**: HOVER phase moves EE to base_xy + per-object_offset_xy, z = base_z + per-object hover offset, holding current object orientation (reuses `translate_for_target_real`)
- [ ] `[N→R]` **WRAP-04**: ZERO phase: 1.0 s post-controller-switch settle → `/io_and_status_controller/zero_ftsensor` CLI call → 0.5 s settle → fresh /wrench sample → warn if any axis bias > 2 N
- [ ] `[N→R]` **WRAP-05**: ZERO phase enforces "STEP BACK" hands-off window: operator-confirmed gate before zero (Y/N or signal), plus +1 s post-zero drift check (operator finger contaminates bias)
- [ ] `[N→R]` **WRAP-06**: ACTIVE phase enters `force_mode_controller` in `base_link` task frame, full 6-DOF compliance default, parameterized force/damping/gain from object's YAML config (or wrapper CLI args during collection)
- [ ] `[N→R]` **WRAP-07**: ACTIVE phase verifies `switch_controller` transition completed via `list_controllers` poll (RPC success ≠ transition complete)
- [ ] `[N→R]` **WRAP-08**: DONE/ABORT exit: `stop_force_mode` → `switch_controller` back to `scaled_joint_trajectory_controller` → **`move_to_safe_height` first** → then `move_home`. Direct home from inserted pose plans straight-line trajectory ignoring the inserted base.
- [ ] `[N→R]` **WRAP-09**: Idempotent cleanup: SIGTERM handler always reaches DONE exit, even if force_mode is already stopped or controller switch fails partway
- [ ] `[N→R]` **WRAP-10**: Signal interface: `SIGUSR1` toggles `event_marker` column ("I'm pushing" / "I let go"), `SIGUSR2` re-zeroes F/T mid-episode (logs `zero_event` row), `SIGTERM` ends as success, custom signal for abort
- [ ] `[N→R]` **WRAP-11**: HOVER pose joint-limit pre-check (skip ACTIVE if hover IK lands at a joint-limit edge that would force protective stop)

### Telemetry Schema (TELE)

Locked schema before any demos collected — re-collecting demos because schema was wrong is the most expensive failure mode.

- [ ] `[N]` **TELE-01**: CSV schema includes phase tag (PRE/HOVER/ZERO/ACTIVE/DONE/ABORT), event_marker, full TCP pose (xyz + quat), full target pose (xyz + quat from assembly config), per-axis errors (dx/dy/dz/droll/dpitch/dyaw recomputed per sample), wrench (Fxyz + Txyz), gripper width, commanded Fz, zero_event flag
- [ ] `[N]` **TELE-02**: Sidecar `<csv>.meta.json` per episode: object/base/grasp_id, assembly target world-frame, start/end ISO timestamps, outcome (success/abort/timeout), full force-mode params used, post-zero bias measured, hands-off window timestamps, free-text user notes (prompted at end), session F/T warm-up duration, smoke-test result
- [ ] `[N]` **TELE-03**: Path convention `compliant_insertion_studio/logs/insert_<object>_<YYYYMMDD_HHMMSS>.csv` with matching `.meta.json`
- [ ] `[N]` **TELE-04**: Logging rate fixed at 100 Hz (subsample every 5th /wrench message — F/T topic publishes at 500 Hz; full rate makes ~150 MB/episode CSVs unworkable)
- [ ] `[N]` **TELE-05**: Schema versioned (`schema_version: 1` in meta JSON) so dashboard can detect/handle future schema bumps
- [ ] `[N]` **TELE-06**: Aborts preserved equally to successes — never delete a CSV based on outcome (demo selection bias is top-3 LfD failure mode)

### Data Collection (DATA)

The 20-episode FMB1 dataset that informs Phase 4 algorithm derivation.

- [ ] `[R]` **DATA-01**: ≥ 5 guided demo episodes per FMB1 object × 4 objects (u_brown, u_orange, line_green, inverted_u_yellow) — ≥ 20-episode minimum
- [ ] `[R]` **DATA-02**: Each episode includes operator-narrated `user_notes` describing what they did to guide the part in
- [ ] `[R]` **DATA-03**: Mandatory failure-mode quota: ≥ 1 abort or intentional misalignment per object (populates failure-signature library)
- [ ] `[R]` **DATA-04**: All demos in real mode on physical UR5e + OnRobot RG2 + FMB1 base (no sim, no synthetic)
- [ ] `[R]` **DATA-05**: Hands-off window observed every episode (operator steps back during ZERO phase)

### Analyzer Dashboard (DASH)

Single static HTML — drop-in inspect, no build, no server. Lives in `compliant_insertion_studio/analyzer/`.

- [ ] `[N→R]` **DASH-01**: Single static `analyzer/analyze_inserts.html` opens directly in browser (Plotly.js 3.5.1 + PapaParse 5.5.3 from CDN, with optional cached copies in `analyzer/assets/` for offline use)
- [ ] `[N→R]` **DASH-02**: File picker (`<input type="file" multiple>`) loads CSVs + meta JSONs; auto-pairs by basename
- [ ] `[N→R]` **DASH-03**: Single-episode view: F vs t (3 traces) + T vs t (3 traces) + Z vs t with synced cursors; sidecar metadata panel; event-marker vertical lines; phase-band background coloring
- [ ] `[N→R]` **DASH-04**: F-vs-Z phase plot (the diagnostic gold standard — peg-in-hole signature surfaces here, not in F-vs-t)
- [ ] `[N→R]` **DASH-05**: 3D trajectory view with target marked; uses `scattergl` for ≥ 50-episode performance (Plotly SVG dies at scale)
- [ ] `[N→R]` **DASH-06**: Cross-episode overlay view: filter by object + outcome, overlay traces **time-aligned on first-contact event** (not absolute time — non-obvious gotcha that breaks the view if missed)
- [ ] `[N→R]` **DASH-07**: Per-object signature card: median Fz at success, |Tx|/|Ty| peak distributions, lateral travel during ACTIVE, descent duration — auto-computed from hands-off-window-restricted samples only
- [ ] `[N→R]` **DASH-08**: Decimation for traces > N samples (browser memory; per Plotly community guidance)
- [ ] `[N→R]` **DASH-09**: Functional UI only — no styling polish, no responsive design

### Algorithm Derivation (ALGO)

Per-object configs + the universal algorithm. This phase is the project's research deliverable — termination criterion isn't decided until data is in.

- [ ] `[N→R]` **ALGO-01**: One universal algorithm in `wrapper/compliant_insert.py` parameterized by per-object YAML configs in `configs/<object>.yaml`
- [ ] `[N→R]` **ALGO-02**: `configs/defaults.yaml` exists; per-object files override (deep-merge via `shared/insert_config.py`)
- [ ] `[N→R]` **ALGO-03**: Config schema covers: axis-wise compliance selection, Fz target, hover offset z, termination predicate (force-absorbed, motion-stopped, z-reached, snap-fit Fz_peak_then_drop, multi-peg torque-band — combinable with AND/OR), retry behavior (max 1–2 retract+re-approach), per-axis tolerance bands, hands-off window duration
- [ ] `[N→R]` **ALGO-04**: Termination criterion **derived from data per-object** — Phase outcome includes "decided what 'success' looks like for each FMB1 stack and recorded rationale in the YAML"
- [ ] `[N→R]` **ALGO-05**: Default termination = combined predicate (force-absorbed AND motion-stopped) with `must_agree=true`; chamfer-rest false-positive prevention
- [ ] `[N→R]` **ALGO-06**: Schema supports snap-fit `Fz_peak_then_drop` event detection (some FMB1 parts may be snap-fit — auto-detect from F-vs-Z phase plot loops)
- [ ] `[N→R]` **ALGO-07**: Schema supports multi-peg torque-band termination (uneven seating produces Tx/Ty rather than extra Fz)
- [ ] `[N→R]` **ALGO-08**: Rule-derivation approach used as primary; statistical classifier (k-means / decision tree on episode feature vectors) reserved as time-boxed (~1 day) research carve-out, only run after rule-derivation complete, must beat best heuristic by ≥ 15 percentage points on held-out-*object* split to ship
- [ ] `[N→R]` **ALGO-09**: Mid-insert re-classification explicitly disallowed (anti-pattern flagged in research)

### Dispatcher + Integration (DISP)

The integration step — done **last**, otherwise broken dispatcher silently breaks all `--insert` ablations.

- [ ] `[N→R]` **DISP-01**: `dispatcher/compliant_insert_episode.py`: config resolution, mode selection (collect vs runtime), unknown-object fallback. No ROS deps in dispatcher itself; unit-testable without robot.
- [ ] `[N→R]` **DISP-02**: Lazy config loading by name (a YAML parse error in one object's config does not break inserts of other objects)
- [ ] `[N→R]` **DISP-03**: Unknown-object behavior: drops into `MANUAL_GUIDED` fallback (full 6-DOF compliance, no auto-terminate, operator-terminated only). Output JSON includes `hint` field with exact `--collect` command for the LLM to surface to operator.
- [ ] `[N→R]` **DISP-04**: `primitives/translate_object.py:1085` `script_path` updated to point at `compliant_insertion_studio/dispatcher/compliant_insert_episode.py`. Existing arg passthrough at lines 1093–1107 already covers everything needed.
- [ ] `[N→R]` **DISP-05**: `server_p3.py` MCP tool surface unchanged (translate_object signature is preserved)
- [ ] `[N→R]` **DISP-06**: Existing `recording_dryrun_real_u_brown.yaml` and `recording_dryrun_real.yaml` ablation YAMLs complete the insert leg autonomously once configs are in place
- [ ] `[N→R]` **DISP-07**: `compliant_insertion_studio/README.md` documents standalone use: how to install into a *different* robotics project (copy folder, edit one path in host's translate_object equivalent, configure assembly_target source)

### Generalization Validation (VAL)

Strengthened gate: ≥ 5 consecutive autonomous successes per object, plus second-assembly proof.

- [ ] `[R]` **VAL-01**: Each FMB1 object passes ≥ 5 consecutive autonomous insert successes (no manual interventions, no parameter tweaks between attempts)
- [ ] `[R]` **VAL-02**: At least one part from a *second* assembly successfully inserted using only its YAML config — no algorithm changes; documents the "tuning a new part in 30 min" workflow
- [ ] `[R]` **VAL-03**: Second-assembly part passes ≥ 3 consecutive autonomous successes
- [ ] `[R]` **VAL-04**: "Tune a new part" SOP documented step-by-step in `docs/tune_a_new_part.md` (collect demos → open dashboard → write config → test) with timing budget
- [ ] `[R]` **VAL-05**: All FMB1 ablation YAMLs (`recording_dryrun_real_u_brown.yaml`, `recording_dryrun_real.yaml`) execute end-to-end without operator intervention on the insert leg

## v2 Requirements

Acknowledged but deferred — useful future work, not in PoC scope.

### Robustness (ROBUST)

- **ROBUST-01**: Per-session F/T temperature drift compensation (auto re-zero between objects)
- **ROBUST-02**: Adaptive retry: re-derive partial config on the fly if first attempt fails (online adaptation)
- **ROBUST-03**: Multi-strategy retry chains (tilt search, spiral search, alternative grasps)
- **ROBUST-04**: Auto-detect F/T sensor health degradation across sessions and prompt re-mount

### Hardware Coverage (HARD)

- **HARD-01**: Franka Panda port (different force-mode API)
- **HARD-02**: Vacuum gripper support (no width feedback)
- **HARD-03**: Bi-manual / dual-arm coordination

### Vision Integration (VIS)

- **VIS-01**: Mid-insert AprilTag re-emergence triggers vision-corrected position estimate
- **VIS-02**: External depth camera contact-point estimation
- **VIS-03**: Failure photo capture for offline review

### Learning (LEARN)

- **LEARN-01**: Online learning loop (auto-re-derive config from N most recent episodes)
- **LEARN-02**: Sequence model (transformer) over F/T+pose timeseries for direct force prediction
- **LEARN-03**: Cross-part transfer (use u_brown signatures to seed inverted_u_yellow config)

### Dashboard Polish (POLISH)

- **POLISH-01**: Publication-quality plot exports (LaTeX-ready figures)
- **POLISH-02**: Annotated narrative view (signature → rule mapping with explanation)
- **POLISH-03**: Comparison table view (side-by-side per-object signature stats)

## Out of Scope

Explicitly excluded from any version. Anti-features documented to prevent re-adding.

| Feature | Reason |
|---------|--------|
| Other robot arms (Franka/KUKA/etc.) | Algorithm uses UR `force_mode_controller` SetForceMode service surface and UR-specific F/T zero. Mechanical port deferred to dedicated milestone. |
| Other grippers (vacuum, soft, multi-finger) | Per-grasp width references and width-only `verify_grasp` depend on OnRobot RG2 conventions. |
| Vision-in-the-insert-loop | AprilTag is occluded by the gripper during ACTIVE phase; force-only feedback by design. Re-emergence of the tag mid-insert is ignored. The whole point of compliance is to absorb the sim-to-real perception gap. |
| Online learning / RL / policy gradients | Strictly offline analysis from logged demos. No runtime parameter updates. |
| Multi-strategy retry chains | Max 1–2 retries (retract 5 mm → re-approach → try again). No fallback ladders, no rule cascades, no escalation to a different family at runtime. |
| Vision-language model orchestration of the insert | The agent (LLM) calls the insert primitive as a black box. No mid-insert agent loop. |
| Real-mode work for non-FMB1 assemblies in PoC milestone | Second-assembly validation is one part, not full coverage. Full coverage is a follow-up milestone. |
| Mid-insert re-classification | Once a config is loaded, parameters are fixed for the episode. No runtime family-switching. |
| `prismatic_peg_insertion.py` from stash | Reference-only. Stale import paths, untracked, overcomplicated for this scope. Reuse RPC patterns; do not extend. |
| `--insertion-type {prismatic,legacy}` flag in translate_object | Old knob; deprecate in Phase 5/6 once dispatcher is the only insert path. |
| rosbag2 parallel logging | Operator workflow is "drop CSV in logs/, refresh browser" — rosbag inserts a deserialization step. CSV + JSON sidecar is the right contract. |
| Workspace-frame / camera calibration | Camera-to-base TF already calibrated; the perception gap is what compliance is built to absorb. Re-calibrating the camera is a separate ROS-side task, not a compliant-insert concern. |
| Gripper-width auto-calibration | Commanded 35 mm settling at 32.6 mm is an `onrobot_ros` driver issue (the driver isn't accounting for the right reference width). Separate fix in the gripper driver, not in this project. **Tracked as known issue, not blocking** — workaround is to command "open" before each `move_to_grasp`. |
| Camera/AprilTag re-calibration as part of this project | Calibration is best-effort; project explicitly designed around the residual perception gap. |

## Traceability

Every v1 requirement maps to exactly one phase. Coverage = 100% (62/62).

| Requirement | Phase | Status |
|-------------|-------|--------|
| SETUP-01 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| SETUP-02 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| SETUP-03 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| SETUP-04 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| SETUP-05 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| SETUP-06 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| SETUP-07 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-01 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-02 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-03 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-04 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-05 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-06 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-07 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-08 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-09 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-10 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| WRAP-01 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| WRAP-02 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| WRAP-03 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| WRAP-04 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| WRAP-05 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| WRAP-06 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| WRAP-07 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| WRAP-08 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| WRAP-09 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| WRAP-10 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| WRAP-11 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| TELE-01 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| TELE-02 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| TELE-03 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| TELE-04 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| TELE-05 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| TELE-06 | Phase 2: Episode Wrapper + Telemetry Schema | Pending |
| DATA-01 | Phase 3: 20-Episode FMB1 Data Collection | Pending |
| DATA-02 | Phase 3: 20-Episode FMB1 Data Collection | Pending |
| DATA-03 | Phase 3: 20-Episode FMB1 Data Collection | Pending |
| DATA-04 | Phase 3: 20-Episode FMB1 Data Collection | Pending |
| DATA-05 | Phase 3: 20-Episode FMB1 Data Collection | Pending |
| DASH-01 | Phase 4: Analyzer Dashboard | Pending |
| DASH-02 | Phase 4: Analyzer Dashboard | Pending |
| DASH-03 | Phase 4: Analyzer Dashboard | Pending |
| DASH-04 | Phase 4: Analyzer Dashboard | Pending |
| DASH-05 | Phase 4: Analyzer Dashboard | Pending |
| DASH-06 | Phase 4: Analyzer Dashboard | Pending |
| DASH-07 | Phase 4: Analyzer Dashboard | Pending |
| DASH-08 | Phase 4: Analyzer Dashboard | Pending |
| DASH-09 | Phase 4: Analyzer Dashboard | Pending |
| ALGO-01 | Phase 5: Algorithm Derivation + Per-Object Configs | Pending |
| ALGO-02 | Phase 5: Algorithm Derivation + Per-Object Configs | Pending |
| ALGO-03 | Phase 5: Algorithm Derivation + Per-Object Configs | Pending |
| ALGO-04 | Phase 5: Algorithm Derivation + Per-Object Configs | Pending |
| ALGO-05 | Phase 5: Algorithm Derivation + Per-Object Configs | Pending |
| ALGO-06 | Phase 5: Algorithm Derivation + Per-Object Configs | Pending |
| ALGO-07 | Phase 5: Algorithm Derivation + Per-Object Configs | Pending |
| ALGO-08 | Phase 5: Algorithm Derivation + Per-Object Configs | Pending |
| ALGO-09 | Phase 5: Algorithm Derivation + Per-Object Configs | Pending |
| DISP-01 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |
| DISP-02 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |
| DISP-03 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |
| DISP-04 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |
| DISP-05 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |
| DISP-06 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |
| DISP-07 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |
| VAL-01 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |
| VAL-02 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |
| VAL-03 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |
| VAL-04 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |
| VAL-05 | Phase 6: Dispatcher Integration + Generalization Validation | Pending |

**Coverage:**
- v1 requirements: **69 total** (7 SETUP + 10 CAL + 11 WRAP + 6 TELE + 5 DATA + 9 DASH + 9 ALGO + 7 DISP + 5 VAL)
- Mapped to phases: 69
- Unmapped: 0

**Per-phase totals:**
- Phase 1 (Foundation Setup + F/T Calibration): **17 reqs** (SETUP-01..07, CAL-01..10)
- Phase 2 (Episode Wrapper + Telemetry Schema): 17 reqs (WRAP-01..11, TELE-01..06)
- Phase 3 (20-Episode FMB1 Data Collection): 5 reqs (DATA-01..05)
- Phase 4 (Analyzer Dashboard): 9 reqs (DASH-01..09)
- Phase 5 (Algorithm Derivation + Per-Object Configs): 9 reqs (ALGO-01..09)
- Phase 6 (Dispatcher Integration + Generalization Validation): 12 reqs (DISP-01..07, VAL-01..05)

**Note on phase boundaries:** per `.planning/CONVENTIONS.md` §6, requirements may complete out of phase order when work is naturally coupled. Traceability is updated when this happens to record where each requirement *actually* completed.

---
*Requirements defined: 2026-05-01*
*Last updated: 2026-05-01 after roadmap creation (traceability populated, coverage 62/62)*

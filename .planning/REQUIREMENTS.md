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

### Pre-Collection Setup (SETUP)

Hardware/software setup that must complete *before* demos are collected, otherwise data is contaminated.

- [ ] **SETUP-01**: `ur_robot_driver` and `ur_msgs` upgraded to ≥ 2.13.0 (April 2026 release with F/T-frame bugfixes)
- [ ] **SETUP-02**: `set_payload` is single-source-of-truth at robot bringup only (UR 5.4.x bug: mid-session re-call interacts with `zero_ftsensor`)
- [ ] **SETUP-03**: Force-mode tuning baseline locked: `gain_scaling=0.5`, `damping_factor=0.7` (research consensus for low-force regime; current default `0.025` damping is too low)
- [ ] **SETUP-04**: `pyproject.toml` numpy pin reconciled with runtime numpy 2.2.6 (lift the `<2` pin)
- [ ] **SETUP-05**: `.gitignore` excludes `compliant_insertion_studio/logs/insert_*.csv` and `*.meta.json`
- [ ] **SETUP-06**: `compliant_insertion_studio/` folder scaffolded with subdirectory layout shown above and a top-level `README.md` describing how to use it standalone (calibrate → wrap → analyze → configure → integrate)
- [ ] **SETUP-07**: Existing `primitives/compliant_insert.py` (the working scaffold validated this session — F/T zero, force_mode RPC, signal handlers, basic CSV) migrated to `compliant_insertion_studio/wrapper/compliant_insert.py` with import paths fixed. Phase 2 extends this file (adds HOVER, full schema, signal markers, hands-off gate); does **not** rewrite from scratch.

### F/T Calibration & Validity (CAL)

F/T sensor calibration is one-time-per-mount, but a quick **smoke test** at session start (or when problems surface) confirms the sensor is still trustworthy. Not re-calibration — a 10-second confidence check.

- [ ] **CAL-01**: New primitive `compliant_insertion_studio/shared/ft_smoke_test.py` — moves arm to a known neutral pose, calls `zero_ftsensor`, holds 5 s, samples /wrench, reports residual bias per axis + drift rate
- [ ] **CAL-02**: Pass criteria documented in `docs/ft_calibration_sop.md`: per-axis residual |F| < 2 N, per-axis residual |T| < 0.3 Nm, drift over 5 s window < 0.5 N/s in any axis
- [ ] **CAL-03**: Smoke-test runnable standalone (`python ft_smoke_test.py`) and as a precondition wrapper integrated into the episode wrapper (skip ACTIVE if smoke fails)
- [ ] **CAL-04**: SOP doc `docs/ft_calibration_sop.md` covers: F/T sensor warm-up window (≥ 10–30 min powered on before first session of the day), when to re-run smoke (start of session, after a protective stop, after physical bumps to the sensor), what to do if smoke fails (escalate — do not proceed)

### Episode Wrapper (WRAP)

The lifecycle owner: hover-zero-active-exit, signal handling, telemetry write. Lives in `compliant_insertion_studio/wrapper/compliant_insert.py`.

- [ ] **WRAP-01**: `compliant_insertion_studio/wrapper/compliant_insert.py` owns full lifecycle: `PRE → HOVER → ZERO → ACTIVE → DONE/ABORT`
- [ ] **WRAP-02**: PRE phase runs CAL smoke-test (or skips if `--skip-smoke` for repeated rapid attempts) and validates preconditions (gripper holding part, joint state available, target known)
- [ ] **WRAP-03**: HOVER phase moves EE to base_xy + per-object_offset_xy, z = base_z + per-object hover offset, holding current object orientation (reuses `translate_for_target_real`)
- [ ] **WRAP-04**: ZERO phase: 1.0 s post-controller-switch settle → `/io_and_status_controller/zero_ftsensor` CLI call → 0.5 s settle → fresh /wrench sample → warn if any axis bias > 2 N
- [ ] **WRAP-05**: ZERO phase enforces "STEP BACK" hands-off window: operator-confirmed gate before zero (Y/N or signal), plus +1 s post-zero drift check (operator finger contaminates bias)
- [ ] **WRAP-06**: ACTIVE phase enters `force_mode_controller` in `base_link` task frame, full 6-DOF compliance default, parameterized force/damping/gain from object's YAML config (or wrapper CLI args during collection)
- [ ] **WRAP-07**: ACTIVE phase verifies `switch_controller` transition completed via `list_controllers` poll (RPC success ≠ transition complete)
- [ ] **WRAP-08**: DONE/ABORT exit: `stop_force_mode` → `switch_controller` back to `scaled_joint_trajectory_controller` → **`move_to_safe_height` first** → then `move_home`. Direct home from inserted pose plans straight-line trajectory ignoring the inserted base.
- [ ] **WRAP-09**: Idempotent cleanup: SIGTERM handler always reaches DONE exit, even if force_mode is already stopped or controller switch fails partway
- [ ] **WRAP-10**: Signal interface: `SIGUSR1` toggles `event_marker` column ("I'm pushing" / "I let go"), `SIGUSR2` re-zeroes F/T mid-episode (logs `zero_event` row), `SIGTERM` ends as success, custom signal for abort
- [ ] **WRAP-11**: HOVER pose joint-limit pre-check (skip ACTIVE if hover IK lands at a joint-limit edge that would force protective stop)

### Telemetry Schema (TELE)

Locked schema before any demos collected — re-collecting demos because schema was wrong is the most expensive failure mode.

- [ ] **TELE-01**: CSV schema includes phase tag (PRE/HOVER/ZERO/ACTIVE/DONE/ABORT), event_marker, full TCP pose (xyz + quat), full target pose (xyz + quat from assembly config), per-axis errors (dx/dy/dz/droll/dpitch/dyaw recomputed per sample), wrench (Fxyz + Txyz), gripper width, commanded Fz, zero_event flag
- [ ] **TELE-02**: Sidecar `<csv>.meta.json` per episode: object/base/grasp_id, assembly target world-frame, start/end ISO timestamps, outcome (success/abort/timeout), full force-mode params used, post-zero bias measured, hands-off window timestamps, free-text user notes (prompted at end), session F/T warm-up duration, smoke-test result
- [ ] **TELE-03**: Path convention `compliant_insertion_studio/logs/insert_<object>_<YYYYMMDD_HHMMSS>.csv` with matching `.meta.json`
- [ ] **TELE-04**: Logging rate fixed at 100 Hz (subsample every 5th /wrench message — F/T topic publishes at 500 Hz; full rate makes ~150 MB/episode CSVs unworkable)
- [ ] **TELE-05**: Schema versioned (`schema_version: 1` in meta JSON) so dashboard can detect/handle future schema bumps
- [ ] **TELE-06**: Aborts preserved equally to successes — never delete a CSV based on outcome (demo selection bias is top-3 LfD failure mode)

### Data Collection (DATA)

The 20-episode FMB1 dataset that informs Phase 4 algorithm derivation.

- [ ] **DATA-01**: ≥ 5 guided demo episodes per FMB1 object × 4 objects (u_brown, u_orange, line_green, inverted_u_yellow) — ≥ 20-episode minimum
- [ ] **DATA-02**: Each episode includes operator-narrated `user_notes` describing what they did to guide the part in
- [ ] **DATA-03**: Mandatory failure-mode quota: ≥ 1 abort or intentional misalignment per object (populates failure-signature library)
- [ ] **DATA-04**: All demos in real mode on physical UR5e + Robotiq + FMB1 base (no sim, no synthetic)
- [ ] **DATA-05**: Hands-off window observed every episode (operator steps back during ZERO phase)

### Analyzer Dashboard (DASH)

Single static HTML — drop-in inspect, no build, no server. Lives in `compliant_insertion_studio/analyzer/`.

- [ ] **DASH-01**: Single static `analyzer/analyze_inserts.html` opens directly in browser (Plotly.js 3.5.1 + PapaParse 5.5.3 from CDN, with optional cached copies in `analyzer/assets/` for offline use)
- [ ] **DASH-02**: File picker (`<input type="file" multiple>`) loads CSVs + meta JSONs; auto-pairs by basename
- [ ] **DASH-03**: Single-episode view: F vs t (3 traces) + T vs t (3 traces) + Z vs t with synced cursors; sidecar metadata panel; event-marker vertical lines; phase-band background coloring
- [ ] **DASH-04**: F-vs-Z phase plot (the diagnostic gold standard — peg-in-hole signature surfaces here, not in F-vs-t)
- [ ] **DASH-05**: 3D trajectory view with target marked; uses `scattergl` for ≥ 50-episode performance (Plotly SVG dies at scale)
- [ ] **DASH-06**: Cross-episode overlay view: filter by object + outcome, overlay traces **time-aligned on first-contact event** (not absolute time — non-obvious gotcha that breaks the view if missed)
- [ ] **DASH-07**: Per-object signature card: median Fz at success, |Tx|/|Ty| peak distributions, lateral travel during ACTIVE, descent duration — auto-computed from hands-off-window-restricted samples only
- [ ] **DASH-08**: Decimation for traces > N samples (browser memory; per Plotly community guidance)
- [ ] **DASH-09**: Functional UI only — no styling polish, no responsive design

### Algorithm Derivation (ALGO)

Per-object configs + the universal algorithm. This phase is the project's research deliverable — termination criterion isn't decided until data is in.

- [ ] **ALGO-01**: One universal algorithm in `wrapper/compliant_insert.py` parameterized by per-object YAML configs in `configs/<object>.yaml`
- [ ] **ALGO-02**: `configs/defaults.yaml` exists; per-object files override (deep-merge via `shared/insert_config.py`)
- [ ] **ALGO-03**: Config schema covers: axis-wise compliance selection, Fz target, hover offset z, termination predicate (force-absorbed, motion-stopped, z-reached, snap-fit Fz_peak_then_drop, multi-peg torque-band — combinable with AND/OR), retry behavior (max 1–2 retract+re-approach), per-axis tolerance bands, hands-off window duration
- [ ] **ALGO-04**: Termination criterion **derived from data per-object** — Phase outcome includes "decided what 'success' looks like for each FMB1 stack and recorded rationale in the YAML"
- [ ] **ALGO-05**: Default termination = combined predicate (force-absorbed AND motion-stopped) with `must_agree=true`; chamfer-rest false-positive prevention
- [ ] **ALGO-06**: Schema supports snap-fit `Fz_peak_then_drop` event detection (some FMB1 parts may be snap-fit — auto-detect from F-vs-Z phase plot loops)
- [ ] **ALGO-07**: Schema supports multi-peg torque-band termination (uneven seating produces Tx/Ty rather than extra Fz)
- [ ] **ALGO-08**: Rule-derivation approach used as primary; statistical classifier (k-means / decision tree on episode feature vectors) reserved as time-boxed (~1 day) research carve-out, only run after rule-derivation complete, must beat best heuristic by ≥ 15 percentage points on held-out-*object* split to ship
- [ ] **ALGO-09**: Mid-insert re-classification explicitly disallowed (anti-pattern flagged in research)

### Dispatcher + Integration (DISP)

The integration step — done **last**, otherwise broken dispatcher silently breaks all `--insert` ablations.

- [ ] **DISP-01**: `dispatcher/compliant_insert_episode.py`: config resolution, mode selection (collect vs runtime), unknown-object fallback. No ROS deps in dispatcher itself; unit-testable without robot.
- [ ] **DISP-02**: Lazy config loading by name (a YAML parse error in one object's config does not break inserts of other objects)
- [ ] **DISP-03**: Unknown-object behavior: drops into `MANUAL_GUIDED` fallback (full 6-DOF compliance, no auto-terminate, operator-terminated only). Output JSON includes `hint` field with exact `--collect` command for the LLM to surface to operator.
- [ ] **DISP-04**: `primitives/translate_object.py:1085` `script_path` updated to point at `compliant_insertion_studio/dispatcher/compliant_insert_episode.py`. Existing arg passthrough at lines 1093–1107 already covers everything needed.
- [ ] **DISP-05**: `server_p3.py` MCP tool surface unchanged (translate_object signature is preserved)
- [ ] **DISP-06**: Existing `recording_dryrun_real_u_brown.yaml` and `recording_dryrun_real.yaml` ablation YAMLs complete the insert leg autonomously once configs are in place
- [ ] **DISP-07**: `compliant_insertion_studio/README.md` documents standalone use: how to install into a *different* robotics project (copy folder, edit one path in host's translate_object equivalent, configure assembly_target source)

### Generalization Validation (VAL)

Strengthened gate: ≥ 5 consecutive autonomous successes per object, plus second-assembly proof.

- [ ] **VAL-01**: Each FMB1 object passes ≥ 5 consecutive autonomous insert successes (no manual interventions, no parameter tweaks between attempts)
- [ ] **VAL-02**: At least one part from a *second* assembly successfully inserted using only its YAML config — no algorithm changes; documents the "tuning a new part in 30 min" workflow
- [ ] **VAL-03**: Second-assembly part passes ≥ 3 consecutive autonomous successes
- [ ] **VAL-04**: "Tune a new part" SOP documented step-by-step in `docs/tune_a_new_part.md` (collect demos → open dashboard → write config → test) with timing budget
- [ ] **VAL-05**: All FMB1 ablation YAMLs (`recording_dryrun_real_u_brown.yaml`, `recording_dryrun_real.yaml`) execute end-to-end without operator intervention on the insert leg

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
| Other grippers (vacuum, soft, multi-finger) | Per-grasp width references and width-only `verify_grasp` depend on Robotiq 2F-85 conventions. |
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
| CAL-01 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-02 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-03 | Phase 1: Foundation Setup + F/T Calibration | Pending |
| CAL-04 | Phase 1: Foundation Setup + F/T Calibration | Pending |
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
- v1 requirements: 62 total (6 SETUP + 4 CAL + 11 WRAP + 6 TELE + 5 DATA + 9 DASH + 9 ALGO + 7 DISP + 5 VAL)
- Mapped to phases: 62
- Unmapped: 0

**Per-phase totals:**
- Phase 1 (Foundation Setup + F/T Calibration): 10 reqs (SETUP-01..06, CAL-01..04)
- Phase 2 (Episode Wrapper + Telemetry Schema): 17 reqs (WRAP-01..11, TELE-01..06)
- Phase 3 (20-Episode FMB1 Data Collection): 5 reqs (DATA-01..05)
- Phase 4 (Analyzer Dashboard): 9 reqs (DASH-01..09)
- Phase 5 (Algorithm Derivation + Per-Object Configs): 9 reqs (ALGO-01..09)
- Phase 6 (Dispatcher Integration + Generalization Validation): 12 reqs (DISP-01..07, VAL-01..05)

---
*Requirements defined: 2026-05-01*
*Last updated: 2026-05-01 after roadmap creation (traceability populated, coverage 62/62)*

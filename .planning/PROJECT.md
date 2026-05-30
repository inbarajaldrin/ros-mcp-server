# Compliant Insertion Studio

## What This Is

A data-collection wrapper, analyzer dashboard, and **parametric peg-in-hole policy** for force-compliant assembly inserts on a UR5e + OnRobot RG2, replacing the current broken `prismatic_peg_insertion.py` real-mode insert path. Operator runs guided demonstrations; the system records F/T + pose telemetry per episode; analysis surfaces per-object parameters (axis-wise compliance, force levels, termination criteria, retry behavior) for a single universal insert algorithm parameterized differently per part. Proof-of-concept target: FMB1 assembly (u_brown, u_orange, line_green, inverted_u_yellow); design must generalize to a second assembly without rework.

## Core Value

**Replace the failing `--insert` real-mode path with a force-compliant insert primitive that works reliably across every FMB1 part and is a one-config-file extension to any new part.** If everything else slips, the FMB1 inserts must complete autonomously end-to-end.

## Requirements

### Validated

(None yet — ship to validate)

### Active

#### Episode Wrapper (data collection infrastructure)
- [ ] New primitive `primitives/compliant_insert_episode.py` owns the full lifecycle: `PRE → HOVER → ZERO → ACTIVE → DONE/ABORT`
- [ ] HOVER phase moves EE to base_xy + per-object_offset_xy at z = base_z + per-object hover offset, holding current object orientation (reuses `translate_for_target_real`)
- [ ] ZERO phase settles 1 s after controller switch, calls `/io_and_status_controller/zero_ftsensor`, settles 0.5 s, verifies residual |F| < 2 N (warns if exceeded)
- [ ] ACTIVE phase enters force mode in `base_link`, full 6-DOF compliance default, gentle Fz commanded down, runs until SIGTERM/SIGUSR signals
- [ ] DONE/ABORT exit moves to safe-height **first** (not direct home — avoids inserted-base collision), then home; tags outcome
- [ ] Signal interface: `SIGUSR1` toggles `event_marker` column in CSV (operator's "I'm pushing now"/"I let go" annotation), `SIGUSR2` re-zeroes F/T mid-episode, `SIGTERM` ends as success, custom signal for abort

#### Enriched Telemetry
- [ ] CSV schema includes phase tag, event_marker, full TCP pose (xyz + quat), full target pose (xyz + quat from assembly config), per-axis errors (dx/dy/dz/droll/dpitch/dyaw recomputed per sample), wrench (Fxyz + Txyz), gripper width, commanded Fz
- [ ] Sidecar JSON metadata per episode: object/base/grasp_id, assembly target, start/end ISO timestamps, outcome, full force-mode params used, post-zero bias measured, free-text user notes prompted at end
- [ ] CSV path convention `logs/insert_<object>_<YYYYMMDD_HHMMSS>.csv` with matching `.meta.json`

#### Analyzer Dashboard
- [ ] Single static `tools/analyze_inserts.html` — no server, no build, opens directly in browser
- [ ] Drop CSVs + meta JSONs into `logs/`, dashboard auto-discovers and loads them
- [ ] Single-episode view: F vs t (3 traces), T vs t (3 traces), Z vs t synced cursors, F-vs-Z phase plot, 3D trajectory with target marked, event-marker vertical lines, sidecar metadata panel
- [ ] Cross-episode overlay view: filter by object + outcome, overlay matching F/T traces to spot signature shapes
- [ ] Per-object signature card: auto-computed stats (median Fz at success, |Tx|/|Ty| peak distributions, lateral travel during ACTIVE, descent duration) — these become the source for parameter values
- [ ] Functional UI only — no styling polish, no responsive design, screenshots-into-paper quality is fine

#### Data Collection (FMB1 PoC)
- [ ] 5 guided demo episodes per FMB1 object × 4 objects = 20-episode minimum dataset
- [ ] Each episode includes operator-narrated `user_notes` describing what they did to guide the part in
- [ ] Mix of clean inserts and intentional misalignments to populate failure-mode library
- [ ] All episodes in real mode on the physical UR5e + OnRobot RG2 + FMB1 base

#### Parametric Algorithm
- [ ] One universal `compliant_insert.py` algorithm, parameterized by per-object YAML configs in `primitives/insert_configs/<object>.yaml`
- [ ] Config schema covers: axis-wise compliance selection, Fz target, hover offset z, termination thresholds (force-absorbed, motion-stopped, z-reached — combination logic), retry behavior (max 1–2 retract+re-approach), per-axis tolerance bands (xy, z, yaw, roll, pitch)
- [ ] Termination criterion **derived from data, not assumed** — Phase outcome includes "decided what 'success' looks like for this stack and recorded it"
- [ ] Classification approach starts as rule-based heuristics over F/T+pose features extracted from the dashboard, with research carve-out for whether a statistical classifier (k-means / decision tree on episode feature vectors) gives better results than hand-derived rules at this dataset size — decision deferred to Phase that runs after data is collected
- [ ] Family/inheritance pattern: a `defaults.yaml` with universal values + per-object overrides — so adding a new part is a small diff, not a fresh file

#### Integration
- [ ] `translate_object.py:1085` real-mode insert path points at the new dispatcher (not at the broken `prismatic_peg_insertion.py`)
- [ ] Existing `--insert` flow in real mode works end-to-end via MCP server (`server.py`) and via direct primitive CLI
- [ ] Existing `recording_dryrun_real_u_brown.yaml` and `recording_dryrun_real.yaml` ablation runs complete the insert leg without manual intervention once configs are in place
- [ ] Behavior on unknown object: drops into manual-guidance fallback mode (force-compliant, no autonomous termination), operator runs ~5 demos, system populates a starter config — explicit calibration ramp, no silent default-config gambling

#### Generalization Validation
- [ ] Universal algorithm validated on at least one part from a *second* assembly (the user's other dataset) using only the new config — no algorithm changes
- [ ] Documented "tuning a new part in 30 min" workflow: collect demos → open dashboard → write config → test

### Out of Scope

- **Other robot arms (Franka/KUKA/etc.)** — Algorithm uses UR `force_mode_controller` SetForceMode service surface and UR-specific F/T zero. Porting is mechanical but explicitly deferred.
- **Other grippers (vacuum, soft, multi-finger)** — Per-grasp width references and width-only verify_grasp depend on OnRobot RG2 conventions.
- **Vision-in-the-insert-loop** — AprilTag is occluded by the gripper during ACTIVE phase; force-only feedback by design. Re-emergence of the tag mid-insert is ignored.
- **Online learning / RL / policy gradients** — Strictly offline analysis from logged demos. No runtime parameter updates.
- **Multi-strategy retry chains** — Max 1–2 retries (retract 5 mm → re-approach → try again). No fallback ladders, no rule cascades, no escalation to a different family at runtime.
- **Vision-language model orchestration of the insert** — The agent (LLM) calls the insert primitive as a black box. No mid-insert agent loop.
- **Real-mode work for non-FMB1 assemblies in PoC milestone** — Second-assembly validation is one part, not full coverage. Full coverage is a follow-up milestone.

## Context

**The repo (ros-mcp-server)** is a ROS2-Humble MCP server fronting a UR5e + OnRobot RG2 for sim-and-real robotic assembly. The primitives live in `primitives/`: `move_to_grasp`, `control_gripper`, `move_to_safe_height`, `rotate_object`, `translate_object` (with sim and real branches), and a stash of older real-mode insertion attempts in `primitives/_real_mode_stash/` (`prismatic_peg_insertion.py`, `peg_in_hole_insertion.py`, `urscript/peg_in_hole_insert.py`, `urscript/move_down.py`, `legacy/insert_compliance.py`, etc.). The stash is **untracked in git** and uses stale import paths (`primitives.utils.workspace_config` → should be `primitives.shared.config`); it is reference-only, not a foundation to build on.

**Recent (this session) progress:**
- Real-mode pipeline confirmed working through `move_to_grasp → close → safe_height → rotate → translate place_down` for u_brown after two patches:
  - `rotate_object.py:622` — removed `if self.mode != 'sim': return False` early-out so geometric posture filters (compact-config, EE-facing-robot, self-collision, etc.) run in real mode too. This was forcing real-mode IK into "wrist-low" configurations that subsequent primitives couldn't handle.
  - `core/move_to_clear_area.py:_verify_grasp` — skips pose-lookup in real mode (AprilTag is occluded once grasped); `translate_object.py:142` pre-verify uses `--width-only` to match.
- A first-cut `primitives/compliant_insert.py` (~250 LOC) is in place: switches to `force_mode_controller`, zeros F/T, starts force mode with full 6-DOF compliance + Fz=-3 N, logs basic CSV, supports SIGTERM/USR1/USR2 signals. Validated empirically: with proper F/T zero (post-zero bias < 0.5 N) the robot descends in the correct direction and yields freely to operator pushes. **This script is a placeholder for the wrapper to be built; not the final architecture.**
- The `recording_dryrun_real_u_brown.yaml` and `recording_dryrun_real.yaml` ablation YAMLs and `mcp_config.json` are wired to use `server.py` (state-threaded, auto-injects orientation/grasp_id between primitive calls) — so the agent doesn't need to write quaternions; the server tracks them. Real-mode insert is the only blocking primitive for those YAMLs to complete.

**Hardware/services available** (all confirmed live in the current ROS graph):
- `/objects_poses_real` (vision-tracked AprilTag poses, drops parts when occluded)
- `/grasp_points_real` (computed by `utils/grasp_points_publisher.py`)
- `/gripper_command`, `/gripper_width`, `/gripper_grasp_detected` (OnRobot RG2 driver, started by `ros2 run onrobot_ros gripper_control`)
- `/tcp_pose_broadcaster/pose`, `/force_torque_sensor_broadcaster/wrench`
- Controllers loaded: `scaled_joint_trajectory_controller` (active), `passthrough_trajectory_controller`, `force_mode_controller`
- Services: `/force_mode_controller/start_force_mode` (`ur_msgs/srv/SetForceMode`), `/stop_force_mode` (Trigger), `/io_and_status_controller/zero_ftsensor` (Trigger), `/controller_manager/switch_controller`

**Operator workflow constraint:** Pendant in Local mode (operator wants manual-control accessibility), so dashboard `--recover` calls fail; protective-stop recovery is manual. Compliant insert design must minimize protective-stop risk.

**Termination criterion is genuinely unknown.** Three candidates (force-absorbed, z-reached, motion-stopped) are all plausible. **Discovering which works — and whether it's a single rule or a combination — is itself a project deliverable**, not an upstream assumption.

## Constraints

- **Tech stack**: ROS2 Humble, Python 3.10, `rclpy`, `ur_robot_driver`, OnRobot RG2 driver. Force mode via `ur_msgs/srv/SetForceMode` only — no direct URScript injection.
- **Hardware**: One physical UR5e + OnRobot RG2 + workspace cameras. Single-instance, no parallel data collection.
- **Operator time**: ~5 demos per object × 4 objects = ~30–60 min collection sessions. Design must accommodate iterative collection across multiple sessions.
- **Pendant mode**: Local mode preferred (operator retains manual control). Dashboard service calls (`--recover`, etc.) cannot be automated.
- **Compliance**: Force-mode commanded wrench must stay gentle (≤ 5 N) by default — gear / part / fixture damage limits.
- **Safety**: Operator's hand near robot during demos. Must always be able to interrupt cleanly (SIGTERM cleanup must reliably switch back to position controller and stop force mode).
- **Existing API**: `translate_object.py --insert --mode real` flow must be preserved as the integration point. The new system replaces what runs *inside* that call, not the call signature.
- **Data location**: All logs to `logs/insert_*.csv` + `.meta.json` in repo root. Dashboard reads from there. Not committed (binary-ish, large) — `.gitignore` entry needed.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat termination criterion as a **deliverable**, not an input | Operator honestly cannot answer which signal (force-absorbed / z-reached / motion-stopped) is most reliable. The data will reveal it. | — Pending |
| Parametric universal algorithm + per-object YAML, not per-object scripts | All objects are peg-in-hole; differences are tolerances, force levels, axis choices. One algorithm with config table is cleaner than N near-duplicate scripts. | — Pending |
| Rule-based heuristics derived from dashboard observations as starting point; statistical classifier as research carve-out | RL/transformers are overkill for ~20-episode datasets. Hand-derived rules over interpretable features are the lowest-risk path; only escalate to classifiers if rules can't separate families. | — Pending |
| Single static HTML dashboard (no server, no framework) | Sole user is the operator. Functional plots > polish. Plotly.js CDN + FileReader is enough. | — Pending |
| Episode wrapper owns full lifecycle including safe-height-then-home exit | Direct `move_home` from inserted pose plans straight-line joint trajectory that ignores the inserted base — observed during initial test. Wrapper must add the safe-height bookend. | — Pending |
| F/T sensor zero with 1.0 s post-switch + 0.5 s post-zero settles, plus residual bias verification | Validated empirically this session: 0.3 s settles produced 4 N residual biases that drove wrong-direction drift. Stash's `move_down` uses 1.0 + 0.5 s settles for the same reason. | ✓ Good |
| Existing `_real_mode_stash/` is **reference only**, not foundation | Stash code uses stale import paths (`primitives.utils` instead of `primitives.shared`), is untracked in git, and is overcomplicated for this scope. Re-derive cleanly using stash's force-mode RPC patterns as guidance. | ✓ Good |
| Manual-guidance fallback for unknown parts (not "refuse to run", not "try a default") | Practical middle ground: operator runs a few demos, system learns parameters from this episode + any prior, then auto-runs subsequent inserts of that part. Avoids damaging-on-first-try while keeping the system usable. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-01 after initialization*

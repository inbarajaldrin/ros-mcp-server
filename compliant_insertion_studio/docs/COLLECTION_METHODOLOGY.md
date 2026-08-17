# Guided-Demo Collection Methodology

> **Portable convention** for collecting force-modulated peg-in-hole demos with a human operator + AI agent + robot, then turning them into a per-object parameterized insertion algorithm. Validated on UR5e + OnRobot RG2 + workspace camera (this repo, FMB1 assembly, 60 demos, 2026-05). Designed to drop onto **any** force-controlled robot arm with a 6-DOF F/T sensor and an adequate gripper.
>
> When you port this studio to a new robot, **read this doc first**, then audit the 4 portability seams in §10.

---

## Table of contents

1. [Three-actor model](#1-three-actor-model)
2. [The frontend: operator + parts + base](#2-the-frontend-operator--parts--base)
3. [The backend: orchestrator + wrapper + telemetry](#3-the-backend-orchestrator--wrapper--telemetry)
4. [The agent in the loop](#4-the-agent-in-the-loop)
5. [Schema design (data contract)](#5-schema-design-data-contract)
6. [Tagging taxonomy](#6-tagging-taxonomy)
7. [Re-pick batching for grasp variety](#7-re-pick-batching-for-grasp-variety)
8. [Force-mode tuning protocol](#8-force-mode-tuning-protocol)
9. [Cleanup discipline](#9-cleanup-discipline)
10. [Portability seams (what changes per robot)](#10-portability-seams-what-changes-per-robot)
11. [Critical anti-patterns](#11-critical-anti-patterns)
12. [Methodology validation: what we measured](#12-methodology-validation-what-we-measured)

---

## 1. Three-actor model

Demo collection is **three actors moving in lockstep**, not one operator commanding one robot:

```
   ┌──────────────────────────────────────────────────────────────┐
   │                        OPERATOR (human)                       │
   │  • places parts on workspace, says "ready"                    │
   │  • physically guides part during ACTIVE phase                 │
   │  • says "done" when insertion complete                        │
   │  • tags outcome (autonomous | assisted) at end of episode     │
   └──────────────┬───────────────────────────────┬───────────────┘
                  │                               │
                  │ verbal cue                    │ physical assist
                  ▼                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                  AGENT (Claude / orchestrator)                │
   │  • runs setup primitives in sequence                          │
   │  • launches wrapper as background subprocess                  │
   │  • sends SIGTERM on operator's "done"                         │
   │  • post-edits meta JSON with operator's tag                   │
   │  • verifies no zombie state after each demo                   │
   └──────────────┬───────────────────────────────┬───────────────┘
                  │                               │
                  │ subprocess + signals          │ ROS topics + services
                  ▼                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                          ROBOT (UR5e)                         │
   │  • bringup + force_mode_controller + scaled_joint_controller  │
   │  • OnRobot RG2 gripper bridge                                 │
   │  • aruco camera + grasp_points publisher                      │
   │  • writes 100 Hz CSV + per-episode meta JSON                  │
   └──────────────────────────────────────────────────────────────┘
```

**Why the agent is in the middle**: the operator can't watch a stdin prompt while guiding the part with both hands. The agent translates verbal "done" into SIGTERM, prompts for the assist tag after motion stops, and post-edits the meta JSON — all while the operator's hands stay near the robot.

---

## 2. The frontend: operator + parts + base

### Workspace layout

| Element | Required | Notes |
|---|---|---|
| Assembly base (e.g., `base1`) | yes | Must be camera-visible (aruco marker up). Define a `DEFAULT_BASE_POSITION` constant for fallback when camera is occluded. |
| Object set (≥4 parts for FMB1; can be ≥1 for proof-of-concept) | yes | Each must have a `<object>_grasp_points.json` defining grasp_id + width per grasp pose. |
| "Clear area" position (off-base, in-workspace) | yes | Used by `move_to_clear_area --move` for place-down between picks. Must be in-camera-view for re-grasp via live camera. |
| Workspace camera covering base + object positions | yes | aruco-detected pose published as `/objects_poses_real`. |
| HOME pose well-clear of workspace | yes | Robot returns here between demos so camera has unoccluded view. |

### Operator preconditions per session

1. **Pendant in Remote Control**, driver headless (`launch_robot.sh real --headless`) — agent commands robot via ROS, operator retains physical override at the pendant (E-stop, mode switch) and is the only one who can clear a protective stop. *(Updated 2026-08-16; collection demos through May 2026 ran with the pendant in Local and an External Control `.urp` played by hand.)*
2. **F/T payload calibration valid** — `set_target_payload(mass, cog)` in bringup, smoke test passes per-session before any demo.
3. **Hands-clear protocol** — for the 5-second STEP-BACK gate before each ZERO + force-mode entry, operator stands back. After zeroing, operator may approach.

### Operator's per-demo loop

```
1. Operator: place part on workspace, confirm "ready" to agent
2. Agent: runs setup (pick → rotate → place → regrasp → rotate)
3. Agent: launches wrapper (PRE → HOVER → ZERO → ACTIVE)
4. Agent: announces "ACTIVE PID=<n>"
5. Operator: watches robot descend, nudges part if needed
6. Operator: says "done with assist" / "done without assist" / "restart this run"
7. Agent: SIGTERMs wrapper → cleanup runs → meta JSON written
8. Agent: post-edits meta with operator's assist_level + a brief note
9. (loop)
```

**Verbal protocol** is intentionally minimal. Don't require the operator to type or click during a demo — both hands are at the robot.

---

## 3. The backend: orchestrator + wrapper + telemetry

### Three layers

```
┌────────────────────────────────────────────────────────────┐
│  Layer 3: ORCHESTRATOR (run_assembly_step.py)             │
│  • Implements canonical pick→rotate→place→regrasp→rotate  │
│  • --setup-only flag: stops before wrapper, prints HELD_QUAT │
│  • --already-held: skips pick/place/regrasp                │
│  • Per-object reusable: same CLI for any FMB1 part         │
└─────────────────────────┬──────────────────────────────────┘
                          │ subprocess
                          ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 2: WRAPPER (compliant_insert.py)                    │
│  • FSM: PRE → HOVER → ZERO → ACTIVE → DONE/ABORT          │
│  • Owns the data contract (CSV + meta JSON)                │
│  • SIGTERM-handled exit with idempotent cleanup            │
│  • --skip-home-on-done flag for fast tuning iteration      │
└─────────────────────────┬──────────────────────────────────┘
                          │ ROS services + topics
                          ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 1: PRIMITIVES (move_to_grasp, rotate_object, …)     │
│  • Single-purpose ROS nodes                                │
│  • Standard JSON output: {"result": "success"|"failure"}   │
│  • Each prints `current_object_orientation` for chaining   │
└────────────────────────────────────────────────────────────┘
```

### State chaining (no fresh camera reads when held)

The held-object orientation **MUST** chain through primitive outputs, not be re-read from the camera. The camera can't reliably see a part once the gripper is around it (occlusion). Each primitive that touches orientation prints a JSON like:

```json
{"result": "success", ..., "current_object_orientation": {"quat": {"x": ..., "y": ..., "z": ..., "w": ...}}}
```

The orchestrator extracts the quat and feeds it to the next primitive's `--current-object-orientation X Y Z W`. After ABORTs that move the robot through cleanup (which rotates the wrist), the chain is broken — re-establish by either (a) full re-pick from camera-detected pose, or (b) `--already-held` mode that runs a fresh `rotate_object` to canonical orientation regardless of input drift.

### The wrapper's FSM

| Phase | Purpose | Operator role | Exits to |
|---|---|---|---|
| `PRE` | F/T smoke test, sanity checks | hands clear | `HOVER` (or `ABORT`) |
| `HOVER` | Move held part above target hole (subprocess: `_run_hover`) | hands clear | `ZERO` (or `ABORT`) |
| `ZERO` | Switch to force_mode_controller, STEP-BACK gate, zero F/T | step back during 5s gate | `ACTIVE` (or `ABORT`) |
| `ACTIVE` | Force mode on, log telemetry until exit signal | guide part, say "done" | `DONE` (timeout, SIGTERM, or end-criterion when implemented) |
| `DONE` | Stop force mode, switch back, safe height, optional move_home | none | exit cleanly |
| `ABORT` | Same cleanup as DONE (idempotent) | none | exit with abort outcome |

**Key design choices:**

- **Cleanup is idempotent** — even if force mode is already stopped or controller switch fails partway, the path runs to completion. Operators learn to trust SIGTERM only when cleanup is reliable.
- **HOVER is a subprocess** — keeps the wrapper free of the heavy `translate_object` import surface, isolates pose-computation crashes from FSM state.
- **CSV writes are line-buffered** — flushes after every newline so a SIGKILL during ACTIVE preserves all samples up to the kill point.

---

## 4. The agent in the loop

The agent (Claude or any other orchestrator) sits between operator and robot. Its responsibilities and the rules that make it work reliably:

### Agent rules

1. **One Bash tool call per workflow step.** Chained bash commands with embedded heredocs have silent failure modes — one broken `extract_var=$(...)` causes the rest of the chain to skip with no output. Setup, launch, and cleanup are three separate Bash calls.

2. **Wait on subprocesses with `kill -0 $PID`, not `pgrep -f`.** The `pgrep -f "python3 -m mywrapper"` pattern matches the bash shell running pgrep (because the search string is in the eval body) → infinite wait. Use:
   ```bash
   while kill -0 $PID 2>/dev/null; do sleep 0.5; done
   ```
   This is reliable and per-PID.

3. **Run the wrapper as a backgrounded subprocess** so the agent retains a PID handle for SIGTERM:
   ```bash
   nohup bash -c '<source> ; python3 -m wrapper ...' > /tmp/wrap.log 2>&1 &
   disown
   PID=$(ps -ef | awk '$8 ~ /python3$/ && /wrapper/ {print $2; exit}')
   ```

4. **Verify ACTIVE entered before announcing "ready for done".** Poll the wrapper log until it contains the ACTIVE marker:
   ```bash
   until grep -q "ACTIVE: enter force mode" /tmp/wrap.log; do sleep 1; done
   ```

5. **Post-edit meta JSON inline** with a heredoc Python script — small enough to be a single Bash call, no orchestrator round-trip needed:
   ```bash
   python3 << 'EOF'
   import json
   p = "compliant_insertion_studio/logs/insert_<obj>_<ts>.meta.json"
   m = json.load(open(p))
   m['assist_level'] = 'assisted'
   m['user_notes'] = 'standard assist — ...'
   json.dump(m, open(p, 'w'), indent=2, sort_keys=False)
   EOF
   ```

6. **Audit after each block.** Don't trust that demos completed correctly without a count + tag-presence check at the end of each object batch. (Single chained-command silent failure can lose a meta update; only an audit catches it.)

7. **Skip the `_run_hover` subprocess only via --already-held** when the part is already held + oriented. Never fabricate a held quaternion — chain from the previous primitive's output.

### Verbal protocol the agent listens for

| Operator says | Agent does |
|---|---|
| "ready" / "the object is in the scene" | Start setup (pick + rotate + place + regrasp + rotate) |
| "done with assist" | SIGTERM wrapper → meta tag `assisted` |
| "done without assist" / "autonomous done" | SIGTERM wrapper → meta tag `autonomous` |
| "restart this run" / "discard this" | SIGTERM wrapper → delete CSV+meta → resetup → re-launch |
| "regrasp" / "fresh pick" | place_down + open + clear-area + re-pick + rotate |
| "the object fell" | move_home → re-detect → operator places back → re-pick |

The agent should not require the operator to use exact strings — listen for intent ("done", "asisst", "asist", typos all OK) and confirm via response.

---

## 5. Schema design (data contract)

The CSV+meta JSON is the **data contract** between collection (this phase) and every downstream consumer (dashboard, signature card extractor, algorithm derivation, validation harness). Schema bugs discovered downstream mean re-collection — the project's most expensive failure mode.

### Per-row CSV (one row per 100 Hz sample)

| Group | Columns | Why |
|---|---|---|
| Time + phase | `t_s`, `phase`, `event_marker`, `hands_off`, `zero_event` | FSM state at every sample |
| TCP pose | `tcp_x..z`, `tcp_qx..qw` | Robot end-effector world pose |
| Target pose | `target_x..z`, `target_qx..qw` | Per-object hover target (placeholder for actual hole in v1) |
| Per-axis error | `dx`, `dy`, `dz`, `droll`, `dpitch`, `dyaw` | TCP vs target deltas |
| Wrench | `fx`, `fy`, `fz`, `tx`, `ty`, `tz` | F/T sensor readings (transformed to base frame) |
| Wrench frame | `wrench_frame_id` | Which frame the wrench was sourced from (sanity check) |
| Gripper | `gripper_width` | Actual jaw gap (mm) |
| Command | `commanded_fz` | What the wrapper requested |
| Object pose | `obj_x..z`, `obj_qx..qw` | Estimate from `R_tcp × R_tcp_to_object_constant` |

**Why object pose is per-row** even though it's TCP-derived: Phase 5 may want to compare commanded vs estimated object orientation over time. Storing it per-row avoids a join at analysis time.

### Per-episode meta JSON

| Required field | Source | Why |
|---|---|---|
| `schema_version` | constant | Migration support |
| `object`, `base`, `grasp_id` | CLI | Identity |
| `outcome`, `outcome_reason` | wrapper | Was it a clean exit? |
| `assist_level` | **operator post-edit** | autonomous \| assisted |
| `user_notes` | **operator post-edit** | One-line description of what happened |
| `current_object_orientation_input` | CLI args | Traceability if pose was wrong |
| `tcp_pose_at_active_start` | wrapper snapshot | Time-zero anchor for downstream alignment |
| `tcp_to_object_transform` | derived at HOVER end | Used to compute per-row obj_q* in CSV |
| `hover_pose_world` | wrapper | Where the part was hovered (placeholder for actual hole pose) |
| `assembly_target_world` | wrapper | Base pose target |
| `force_mode_params` | wrapper | The exact wrench/selection/gain/damping used |
| `foundational_calibration` | wrapper | F/T payload calibration provenance |
| `smoke_test` | wrapper | Per-session F/T sanity check |
| `post_zero_bias` + `post_zero_drift_check` | wrapper | F/T baseline at ACTIVE start |
| `hands_off_window` | wrapper | When the operator was hands-clear |
| `mid_episode_zero_events` | wrapper | If F/T re-zero fired during ACTIVE |

**Schema versioning rule**: additions at the end are v1.1 → v1.2 (additive, backward-compatible). Renames or reorders are v2.

---

## 6. Tagging taxonomy

### Two-state assist_level (this milestone)

- **`autonomous`** — robot performed the insertion with zero operator physical intervention during ACTIVE
- **`assisted`** — operator nudged, guided, or otherwise touched the robot/part during ACTIVE

We rejected a 3rd `abort` tag — the operator's stated convention is **delete bad data; keep clean test data**. If a demo went wrong (camera lost the part, gripper slipped, robot took an unexpected path), the agent deletes the CSV+meta and the demo doesn't count.

### Why minimum viable tagging

The Sirius / "Robot Learning on the Job" framework (Liu et al. 2024) recognizes 4 categories: autonomous, pre-intervention, intervention, post-intervention. We intentionally chose binary tagging because:

1. **Operator cognitive load matters** — every additional tag option per demo slows collection
2. **Mid-episode markers can be derived from raw F/T data** post-hoc — no need to mark moments live
3. **Per-sample assist_level** is a Phase 5+ concern (training a per-sample classifier from telemetry); over-tagging now wastes operator time

If you need per-sample tagging later, derive it from F/T variance and the `event_marker` column without changing the data contract.

### user_notes discipline

Every meta JSON gets a one-line `user_notes` field. Templates we found useful:

- `"standard assist — nudged into hole, settled cleanly"`
- `"autonomous insertion, no intervention"`
- `"standard assist — re-pick batch (fresh grasp), nudged into hole, settled cleanly"`
- `"autonomous (re-pick batch) — tight fit, no intervention"`

Specifics that matter for Phase 5 analysis:
- Mention "re-pick batch" if the demo came after a `place_down + re-grasp` cycle
- Mention "tight fit" if friction was unusual
- Mention any anomaly (operator's hand made unintended contact, robot bounced at first contact, etc.)

---

## 7. Re-pick batching for grasp variety

For objects that need >10 demos, run **two batches of 10 with a place-down + fresh-pick in between**. The fresh pick captures grasp pose variance that would otherwise be absent from a single-grasp dataset.

### Why batching, not interleaving

- Single consistent grasp gives Phase 5 a clean baseline to derive per-object thresholds against
- Interleaved re-grasps would add a per-sample confounding variable (grasp pose drift on top of operator-assist drift)
- Two clear batches separate the variance sources

### Re-pick procedure

```
1. Final demo of batch N completes
2. translate_object --place-down --object-name X    (lateral move + lower)
3. control_gripper open
4. move_to_safe_height
5. control_gripper to grasp_width
6. move_to_clear_area --move --object-name X       (positions EE for camera view)
7. move_to_grasp --object-name X --grasp-id N      (LIVE camera, not hardcoded)
8. control_gripper close
9. move_to_safe_height
10. rotate_object (with returned current_object_orientation)
11. (resume demo loop with --already-held + new HELD_QUAT)
```

### Anti-pattern in re-pick: hardcoded XYZ + camera bypass

**Don't** use `move_to_clear_area + move_down + close + lift` for re-pick. That descends to a fixed XYZ regardless of where the part actually landed. If the part bounced 5cm off-target, the gripper closes on empty space and the next demo fails or holds nothing.

**Do** use `move_to_clear_area --move` (positions EE near the camera-clear pose, no descent) followed by `move_to_grasp` (live camera-detected pose, descends to the actual part). The first is for getting the EE out of the camera's view of the workspace; the second is for the actual pick.

---

## 8. Force-mode tuning protocol

### Defaults are intentionally too gentle

Out-of-the-box wrapper defaults (`fz=3 N`, `lin-speed=0.02 m/s`, `gain=0.5`, `damping=0.7`) are conservative enough that any robot won't damage anything on first run. That makes them too slow for iterative tuning sessions.

### Tuning sequence we converged on (UR5e + RG2)

| Round | Change | Result | Reason |
|---|---|---|---|
| 0 | defaults: `fz=3, lin=0.02, gain=0.5` | 30+ s descent, 120s ACTIVE | too slow for operator-in-loop |
| 1 | `lin-speed=0.06` (3×) | minimal change in descent feel | speed cap not the bottleneck at low gain |
| 2 | `lin-speed=0.18 + gain=1.0` | noticeably faster | gain limits accel too |
| 3 | `lin-speed=0.54 + gain=1.0` | fast enough but slow ramp-up | force limits accel |
| 4 | `fz=9 (override 5N cap) + lin-speed=0.54 + gain=1.0` | **locked**: 60-70s full demo cycle | acceptable to operator, gripper still safe |

### Rules for tuning a new robot

1. **Bump speed cap first** (cheap; safety floor still set by force compliance)
2. **Then bump gain** (faster force-mode response)
3. **Force last** — only if speed cap can't be reached because acceleration to it is force-limited
4. **Max force should respect the part's binding constraint** (gear damage, fixture limits, sensor saturation), not the wrapper's convention
5. **`--override-fz-cap` requires explicit operator awareness** — it's there to make raising the limit deliberate, not to be auto-set

### Locked-in tuning provenance

Every demo's meta JSON records `force_mode_params` so Phase 5 (and any future agent comparing across collection sessions) can see exactly what `fz`, `gain`, `damping`, `selection_vector` were used. **Don't** use a global config file for these — embed in each episode.

---

## 9. Cleanup discipline

### Per-demo

1. **SIGTERM is the clean exit signal**. Wrapper's DONE/ABORT path is idempotent.
2. **Wait for clean exit before next demo** — `kill -0 $PID` loop, NOT `pgrep -f`.
3. **Audit immediately**: read the meta JSON the wrapper wrote, confirm `outcome_reason`, post-edit `assist_level` + `user_notes`.
4. **Delete bad data** — if the operator says "discard this run", `rm` the CSV+meta. Per the no-`abort`-tag convention, only clean demos persist.

### Per-block (object switch)

1. **Audit count + tag presence** for the object's CSVs:
   ```python
   metas = sorted(Path("logs").glob(f"insert_{obj}_*.meta.json"))
   untagged = sum(1 for p in metas if json.load(open(p)).get('assist_level') is None)
   no_notes = sum(1 for p in metas if not json.load(open(p)).get('user_notes'))
   assert len(metas) == TARGET_COUNT and untagged == 0 and no_notes == 0
   ```
2. **Place-down + open + safe height + move_home** before swapping to next object
3. **Update STATE.md / TRACKS.md / HANDOFF.json** if pausing

### Per-session (end of day)

1. **Final audit across all 4 objects** — total demo count + tag breakdown + schema version
2. **Release any held part** before walking away (gripper hold-force can damage edges over hours)
3. **Optional**: close the ROS stack via `close_robot.sh` + `pkill` for camera/gripper/grasp_pub. If the next session is within 24 hours, leaving it running saves ~5 min cold-start.
4. **`/gsd-pause-work`** to write the structured handoff for next session

---

## 10. Portability seams (what changes per robot)

Move this studio to a different robot? **Audit these 4 seams**:

### Seam 1: ROS launch + driver

| File | What's robot-specific |
|---|---|
| `compliant_insertion_studio/launch/ur5e_with_rg2.launch.py` | UR-specific bringup, IP, controllers |
| `compliant_insertion_studio/scripts/launch_robot.sh` | Pendant cycle instructions specific to UR |
| `primitives/shared/config.py` | `HOME_JOINTS`, `HOME_POSE`, `TABLE_HEIGHT`, `DEFAULT_BASE_POSITION` |
| `primitives/shared/ik.py` | DH params for UR5e |
| Bringup parameter file | `set_target_payload(mass, cog)` for the new gripper |

For a new robot:
- Replace launch file with the new robot's bringup
- Update HOME pose + workspace constants
- Replace IK if not UR5e
- Run F/T calibration for the new gripper, paste new `set_target_payload`

### Seam 2: Gripper

| File | What's gripper-specific |
|---|---|
| `~/Desktop/ros2_ws/src/onrobot_ros/` (external) | OnRobot RG2 Modbus bridge |
| `primitives/control_gripper.py` | Mode flags, width semantics, settling check |
| `primitives/move_to_grasp.py` | Expected width tolerance (currently `expected - 5.0` to match RG2 grip overshoot) |
| `compliant_insertion_studio/urdf/onrobot_rg2/` | URDF for RViz + collision |

For a new gripper:
- Replace bridge with the new gripper's ROS driver
- Verify width-reporting semantics (raw vs offset; tolerance windows)
- Update grasp tolerance constants if firmware behavior differs
- Swap URDF

### Seam 3: Camera + object localization

| File | What's camera-specific |
|---|---|
| `~/Desktop/ros2_ws/src/aruco_camera_localizer/` (external) | Aruco-specific (not transferable to other markers without code) |
| `utils/grasp_points_publisher.py` | Reads `aruco-grasp-annotator` data dir |
| `~/Documents/aruco-grasp-annotator/data/grasp_points/` | Per-object grasp_id definitions |

For a different localization scheme (e.g., learned object detection):
- Replace the localizer node — must publish `tf2_msgs/TFMessage` on `/objects_poses_real` with `child_frame_id == object name`
- Keep the `<object>_grasp_points.json` schema (id, position-in-CAD-frame, validity widths)
- Update `data_path_finder.py` to find your new data dir

### Seam 4: Per-object data (NOT code)

| Path | Content |
|---|---|
| `<grasp_data_root>/grasp_points/<object>_grasp_points.json` | Per-grasp positions + valid gripper widths |
| `<grasp_data_root>/symmetry/` | Object symmetry data for `rotate_object` |
| `<grasp_data_root>/fmb_assembly1.json` | Base + object name registry |
| `ablations/ground_truth_resources/Assembly_*_results.json` | Canonical sequence per assembly (the orchestrator implements this) |

For a new assembly:
- Add the new objects to grasp_points data
- Add an Assembly JSON listing the canonical pick→rotate→place→regrasp→rotate→insert sequence
- Tune per-object grasp widths + grasp_id selections via the JSON survey snippet in `compliant_insertion_studio/docs/SETUP.md` §4.3

**Things that DON'T change per robot** (because they're the methodology, not the implementation):

- Wrapper FSM (PRE→HOVER→ZERO→ACTIVE→DONE)
- Schema v1.1 (CSV columns + meta JSON keys)
- Orchestrator pattern (setup-only → background wrapper → SIGTERM → meta post-edit)
- Re-pick batching for grasp variety
- Force-mode tuning protocol (the *values* may change, the *order* doesn't)
- Cleanup discipline + audit pattern
- The agent rules (kill -0, one Bash call per step, etc.)
- The assist_level / user_notes taxonomy

If your new robot has a non-force-mode controller (e.g., admittance instead of impedance), the wrapper's ZERO + ACTIVE phases need rework. Everything else is portable.

---

## 11. Critical anti-patterns

These hit us during this session. **Don't repeat them.**

| Anti-pattern | What it manifests as | Structural mitigation |
|---|---|---|
| **`pgrep -f "..."` in until-loop** | Loop never exits because pgrep matches the bash shell running pgrep | Use `while kill -0 $PID 2>/dev/null; do sleep 0.5; done` instead |
| **Chained bash commands across workflow steps** | Silent failure: one bad arg stops the chain, no output captured | One Bash tool call per step (cleanup, setup, launch, post-edit are separate) |
| **Hardcoded XYZ for re-pick** | Gripper descends to fixed location, misses if part bounced | `move_to_clear_area --move` (positions EE) + `move_to_grasp` (live camera) |
| **Reading held-object pose from camera** | Camera occluded by gripper → fails or returns stale pose | Chain `current_object_orientation` through primitive outputs |
| **ANSI codes in `ros2` CLI subprocess output** | Parser mismatches (e.g., `parts[-1] == "active"` never matches `"\x1b[0m"`) | Strip ANSI: `re.sub(r'\x1b\[[0-9;]*m', '', line)` |
| **Treating gripper firmware grip-mode as a position mode** | Commanded width ≠ settled width by 2-5mm | Tolerate the variance in downstream checks (already done in `move_to_grasp.py`) |
| **Silent `outcome=timeout` is treated as failure** | Phase 5 looks for clean success/failure tags but every demo says "timeout" | Wrapper exits on operator SIGTERM with `outcome_reason="operator_sigterm"`, treat that as the success signal until real end-criteria are derived in Phase 5 |
| **Trusting "looks done" without an audit** | Demo count drifts (lost meta updates) and goes undetected until end-of-day | Per-block audit script counts CSVs + checks every meta has tags |

Each one was discovered by hitting it. They're now in `CLAUDE.md` anti-patterns and in `.planning/.continue-here.md` blocking constraints.

---

## 12. Methodology validation: what we measured

This methodology produced a clean dataset of 60 demos in one ~5-hour at-robot session. Some empirical numbers worth recording:

| Metric | Value | Notes |
|---|---|---|
| Demos per hour (with operator-in-loop, single object) | ~10-12 | Including setup, ACTIVE, cleanup, meta edit |
| Demos per hour (re-pick batches) | ~8-10 | Slower due to release + clear-area + re-grasp |
| Setup time per demo (`--already-held` mode) | ~10 sec | Re-rotate is the only motion |
| Setup time per demo (`--already-held=false`, fresh pick) | ~60 sec | Full pick → rotate → place → regrasp → rotate |
| Wrapper ACTIVE phase | 53-100s | Depends on operator's nudging speed |
| Cleanup with `--skip-home-on-done` | ~3-5s | Force mode stop + controller switch + safe height |
| Cleanup with default (with move_home) | ~8-10s | Adds joint-space move_home subprocess |
| CSV size per demo | 2-3 MB | ~6500 samples × 41 cols × ~25 bytes each |
| Meta JSON size per demo | 3-4 KB | 24 required keys + 13 optional |
| Total dataset size (60 demos) | ~150 MB | All in `compliant_insertion_studio/logs/` |
| Schema migration cost (v1.0 → v1.1) | ~30 min | Bump constant, add cols to CSV header, update _FORMATTERS, wire wrapper to populate |
| Operator cognitive load per demo | one verbal cue at end + zero typing | "done with assist" or "done without assist" |

**What scales linearly**: number of demos per session.
**What's a one-time cost**: methodology setup, schema design, orchestrator wiring.
**What can be skipped on a 2nd robot**: nothing in §1-9 (they're the methodology). Everything in §10 needs work.

---

## See also

- `compliant_insertion_studio/docs/SETUP.md` — bring-up runbook (per-session)
- `compliant_insertion_studio/docs/SCHEMA.md` — v1.1 schema spec
- `compliant_insertion_studio/docs/ft_calibration_sop.md` — three-layer F/T calibration
- `compliant_insertion_studio/scripts/run_assembly_step.py` — the orchestrator code (lives next to this doc)
- `compliant_insertion_studio/wrapper/compliant_insert.py` — the FSM wrapper
- `compliant_insertion_studio/wrapper/schema_v1.py` — column constants in code
- `.planning/HANDOFF.json` — current project state (rolls forward across sessions)
- `CLAUDE.md` — auto-loaded for every Claude Code session in this repo (anti-patterns + per-robot quirks)

---

# Addendum: GUIDED-mode collection pipeline (v2, 2026-05-06)

This doc was authored 2026-05-03 and captured the foundational three-actor methodology. Through 2026-05-06 the operationalization evolved substantially: the wrapper now has a dedicated GUIDED state for operator-drag data collection, and the marker-finalization workflow produces autonomous predicates from the collected data. Document v2 below; original §1–§12 remain authoritative for first-principles methodology.

## 13. The four-marker FSM consolidation

The 5-state FSM described in older parts of the codebase (APPROACH / FIND_HOLE / ENTRY_SETTLE / WEDGE_RECOVERY / INSERT) has been reframed by the operator into a 3-state, 4-marker model:

```
States (3):  Inserting | Aligning | At Target
Markers (4):
  1. Contact     — Inserting → Aligning      (peg-bottom touches a surface)
  2. Found Hole  — Aligning → Inserting      (peg cleared rim, descending into slot)
  3. Contact    — Inserting → Aligning      (multi-contact loop re-entry)
  4. At Target   — Inserting → DONE          (peg fully seated)
```

The legacy FIND_HOLE / ENTRY_SETTLE split exists in the FSM code as algorithmic convenience but doesn't map to a distinct operator-observable behavior. For new objects, treat the 4-marker model as canonical.

**Marker status (validated 2026-05-06 on u_orange/base1/grasp_id=1):**

| # | Marker | Predicate | Validated |
|---|---|---|---|
| 1 | Contact | `fz_smoothed > 3N for 0.1s sustained`, after 1.0s grace | ✅ |
| 2 | Found Hole | tilt-relax + F_lat collapse + dz/dt onset (tool frame) | 🟡 data collected, predicate pending |
| 3 | Re-Contact | same as #1 | ✅ by reuse |
| 4 | At Target | `\|tcp_z - predicted_tcp_z\| < 5mm` + motion stopped + tilt low, sustained 1s | ✅ |

## 14. GUIDED state — operator labels Found Hole at runtime

The wrapper has a `--guided-mode` flag (`compliant_insertion_studio/wrapper/compliant_insert.py`). With it set, APPROACH-Contact routes to a new `GUIDED` state instead of FIND_HOLE.

**GUIDED state behavior:**
- `selection_vector = (T, T, F, F, F, F)` — XY compliant, Z LOCKED, rotation LOCKED
- `wrench = zero` — pure compliance, gain=2.0, damping=0.05 (operator-friendly)
- Robot becomes "gimbal-stabilized": operator drags EE laterally; height + orientation stay locked

**Mark transition mechanism:**
- Operator drags peg above slot
- Sends SIGUSR1 to wrapper PID (collection script does this on Enter)
- FSM `mark_hole()` captures `tcp_xy` as `meta.hole_observed_operator.xy_m`
- FSM transitions to `INSERT_DESCENT` (autonomous Z descent at marked xy)
- Global seat detector fires when At-Target predicate holds → DONE

**Key advantage**: each GUIDED demo is a *labeled* Found-Hole sample. SIGUSR1 timestamp = operator's ground-truth "hole reached" moment. No post-hoc segmentation required for marker derivation.

## 15. Updated collection harness

`compliant_insertion_studio/scripts/collect_regime_data.py` is the v2 harness. Replaces ad-hoc per-demo CLI invocation.

**Per-demo flow:**
1. Operator places part anywhere on the rim near the slot
2. Script launches wrapper with `--guided-mode --base-offset-xy DX DY`
3. Wrapper does PRE → HOVER → ZERO → APPROACH (pure-Z, XY locked) → Contact
4. FSM transitions to GUIDED — script detects "FSM → GUIDED" log line
5. Script prints prompt; operator drags peg laterally to above the slot
6. Operator presses Enter → SIGUSR1 forwarded through `run_assembly_step` → wrapper → FSM `mark_hole()`
7. INSERT_DESCENT runs autonomously; At-Target marker auto-fires within 1-2s of seat
8. Wrapper cleanup retracts to safe height + home
9. Per-demo `_verify_physical_seat()` runs against raw CSV before counting demo as valid

**Variations to collect (one object/base/grasp_id):**

| Variation | --base-offset-xy | Reps |
|---|---|---|
| A_pos_x_10mm | `0.010 0.000` | 3 |
| B_neg_x_10mm | `-0.010 0.000` | 3 |
| C_pos_y_10mm | `0.000 0.010` | 3 |
| D_neg_y_10mm | `0.000 -0.010` | 3 |
| E_diag_pxpy_7mm | `0.007 0.007` | 3 |

Total: 15 demos × ~70s/demo = ~17 min operator time.

## 16. Direction-invariance — what the variations actually test

**The 5 variations are NOT for "directional coverage" of a direction-dependent control law.** They're for testing **direction-invariance** of the Found Hole marker.

**Marker (direction-invariant by construction)**: tilt-relax, F_lat collapse, dz/dt onset are *local* sensor signatures of "peg now over chamfer." Should fire identically regardless of which side peg approached from.

**Director (direction-dependent, not derived from operator data)**: vector from current xy toward predicted slot xy. In autonomous mode, this comes from CAD chain refined by base calibration (§19), not from the operator's drag history.

**Validation test**: derive Found Hole predicate from collected demos. Verify it fires within ±300ms of operator's SIGUSR1 timestamp **across all collected variations**. If it fires consistently regardless of which variation: predicate is direction-invariant, ship it. If it's bimodal across variations, the predicate has a frame conversion bug (probably world-frame instead of tool-frame).

**3 directions × 3 reps suffices** if direction-invariance holds. Collecting all 5 directions is over-coverage, not under-coverage.

## 17. Frame discipline for marker analysis

When deriving any predicate from F/T or torque data, **default to tool frame** (`tool0_controller`). Convert to base/world only for visualization or for matching against world-frame TCP positions.

The `<basename>.wrench_raw.csv` sidecar records native sensor-frame values at 500Hz. Use that for marker derivation, not the base-frame values in the main CSV.

Anti-pattern: computing operator-drag-vs-r_cop alignment in world frame produces bimodal sign across variations because the absolute drag direction differs by quadrant. Recomputed in tool frame, the alignment is direction-invariant by construction. If the user-reported "anomaly across variations" appears in your analysis, suspect a frame issue first.

## 18. F/T bias + force-mode transient (settled, was missing)

Two corrections that were operationalized 2026-05-06 after specific phantom-contact bugs:

### Post-zero F/T bias subtraction
Wrapper samples residual bias in PRE/ZERO and stashes on `ep.post_zero_bias_baselink`. Subtracts from all wrench components before passing to FSM. Without this, residual bias > contact threshold caused phantom contact at hover. CSV records RAW (uncorrected) wrench; for post-hoc analysis, correct as `corrected_fz = csv.fz - meta.post_zero_bias.Fz`.

### APPROACH grace period
1.0s after ACTIVE start, contact detection ignored. Lets force_mode_controller startup transient (raw fz can oscillate ±5N for ~0.5s) settle before we look for real contact. Configurable via `approach_grace_period_s`.

## 19. Base position calibration (parallel free win)

Each GUIDED demo's `meta.hole_observed_operator.xy_m` is one measurement of the true base position projected through the known CAD chain. Aggregate across N demos → calibrated base xy with mm-scale stddev.

Spec at `compliant_insertion_studio/analysis/BASE_CALIBRATION_FROM_HOLE_OBSERVATIONS.md`. Output: `configs/base_calibration_<base>_<date>.yaml`. Use the calibrated value in `primitives/shared/config.py:DEFAULT_BASE_POSITION`.

After calibration, the autonomous wrapper's `predicted_tcp_xy` becomes mm-accurate. The 10-15mm CAD-prior bias drops to ~1-2mm perception noise — autonomous insertion can be wide-search-free.

## 20. End-to-end recipe for a new object/base

1. **Bring up robot**: `launch_robot.sh real` (auto-applies `set_payload`)
2. **Verify F/T health**: `ft_smoke_test.py` — residual bias < 2N
3. **Collect**: `collect_regime_data.py --object <obj> --base <base> --grasp-id <id>` — 5 variations × 3 reps = 15 demos, ~17 min
4. **Aggregate**: `30_segment_regimes.py` per demo + cross-demo features at SIGUSR1 timestamp (tool frame)
5. **Derive Found Hole predicate**: data-driven thresholds; validate fires within ±300ms of operator's SIGUSR1 across all variations
6. **Calibrate base**: `40_calibrate_base_from_observations.py` (Phase 2)
7. **Validate autonomous**: run wrapper without `--guided-mode` — should seat without operator intervention
8. **Lock in**: update `defaults.yaml` with derived thresholds; commit calibration YAML

Each step is the test of the previous. If autonomous validation in step 7 fails, the derivation in step 5 was incomplete.

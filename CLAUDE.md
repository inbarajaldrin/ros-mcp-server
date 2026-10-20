# ros-mcp-server — UR5e + RG2 FMB1 assembly

**Branch `ur5e-fmb1-demo`. This is the tree that runs the real assembly.**
Last verified **2026-08-16**: FMB1 seated 4/4 (u_brown, u_orange, line_green, inverted_u_yellow)
with one operator action — placing a part on the table.

> `main` **cannot** run the real assembly. Its June candidate-native migration made the grasp
> publisher emit composite ids (101…303) while the assembly JSONs still carry flat ids (1, 2),
> and the replacement `--grasp-candidate` flag refuses `--mode real` by design. Every pick fails
> with *"Grasp point 1 not found"*. Work here, or fix that first.

---

## 1. Bring up the cell

Total ~1 minute. Sequence matters: driver → gripper → camera → grasp points.
**One command does all of 1.1–1.5:** `bash compliant_insertion_studio/scripts/stack_up.sh`
(no arm motion; `stack_down.sh` reverses it). The steps below are what it runs.

**Cell PC (Dell OptiPlex 3060, i5-8500T)** differs from the A4500 these docs were written on:
- D415 must be on a **USB 3 port at 5000 Mbps** (`cat /sys/bus/usb/devices/*/speed`). Internal
  ports 7/8 are USB 2 only. On USB 2 the localizer hangs silently (no topics, no error).
- The localizer needs OpenCV ≥ 4.7 (`pip3 install --user opencv-python==4.11.0.86`; system 4.5.4
  lacks `aruco.ArucoDetector`). It runs ~3 Hz here (CPU-bound).
- A cold `launch_robot.sh` can outlast its 90 s wait: if it does, activate
  `scaled_joint_trajectory_controller` and call `set_payload` by hand (values in the newest
  `configs/ft_calibration_*.yaml`).
- Logging out of the desktop kills the whole stack — re-run `stack_up.sh` after any re-login.

### 1.0 Confirm the robot is powered and in Remote Control

Only the pendant needs to be booted. No program loaded, nothing played.

```bash
cd ~/Documents/prismatic-manipulation
python3 scripts/ur_dashboard.py --host 192.168.1.111 --real mode safety remote
# want:  Robotmode: RUNNING | Safetystatus: NORMAL | remote: true
```

If `remote: false`, flip the pendant to Remote Control (top-right in PolyScope) — headless
control cannot be established from Local. If mode is not RUNNING, `… --real power_up` powers on
and releases brakes.

### 1.1 Driver (headless — no pendant program)

```bash
bash compliant_insertion_studio/scripts/launch_robot.sh real --headless
```

Sets the F/T payload from `configs/ft_calibration_*.yaml` — **required**, force mode has no
gravity compensation without it. Wait for `Robot connected to reverse interface` in
`/tmp/ur_bringup_logs/real_bringup_*.log`.

Omit `--headless` only if you have an External Control `.urp` on the pendant and want the URCap
path; then you must press Play yourself.

### 1.2 Gripper

```bash
nohup bash -c 'source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash;
  ros2 run onrobot_ros gripper_control' > /tmp/gripper.log 2>&1 &
sleep 10; tail -3 /tmp/gripper.log      # want "Gripper initialized successfully"
```

### 1.3 Camera

```bash
bash compliant_insertion_studio/scripts/launch_camera.sh --background
```

### 1.4 Grasp points — **must run from THIS tree**

```bash
nohup bash -c 'source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash;
  cd '"$PWD"'; python3 -u utils/grasp_points_publisher.py --mode real' > /tmp/grasp_pub.log 2>&1 &
```

Running `main`'s copy publishes composite ids and every pick fails. Dies silently if the camera
localizer restarts — re-check before any pick.

### 1.5 Verify before moving the robot

```bash
ros2 topic echo --once /objects_poses_real | grep child_frame_id   # base1 + all 4 parts
ros2 topic hz /grasp_points_real                                   # ~5 Hz
ros2 topic hz /tcp_pose_broadcaster/pose                           # ~500 Hz
ros2 control list_controllers | grep -E "scaled_joint|force_torque"  # both active
ros2 topic echo --once /gripper_status                             # Safety/Circuit all False
```

### 1.6 Demo RViz (optional)

```bash
bash compliant_insertion_studio/scripts/launch_rviz_demo.sh      # robot | gripper camera
```

Robot on the left, `/annotated_stream` on the right, no Displays panel; fitted to whatever screen
it starts on. The view is saved in `compliant_insertion_studio/rviz/fmb1_demo.rviz` — to keep a
new one, frame it in RViz, Ctrl+S, and copy `Views` from `/tmp/fmb1_rviz/fmb1_demo_runtime.rviz`.
RViz's fixed frame `base_link` is UR `base` rotated 180° about z: the workspace is at **+y** there.

---

## 2. Run the assembly

One object at a time. The script returns **1** on failure — check it and stop rather than chaining.

```bash
J=ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json
python3 -u ablations/replay_real_assembly.py --assembly-json $J --only u_brown
python3 -u ablations/replay_real_assembly.py --assembly-json $J --only u_orange          --skip-startup
python3 -u ablations/replay_real_assembly.py --assembly-json $J --only line_green        --skip-startup
python3 -u ablations/replay_real_assembly.py --assembly-json $J --only inverted_u_yellow --skip-startup
```

Order matters — parts nest. Each is 8 or 15 steps (pick → rotate → place → regrasp → rotate →
insert → release); `line_green` skips the regrasp. Leaving out `--only` runs all four in order
and stops at the first failure.

**`--derive-sequence` is the default** (`--no-derive-sequence` to replay the recorded sequences): observe the part first and run 1 or 2 passes as
`compliant_insertion_studio/planning/sequence_planner.py` decides, instead of the recorded pass
count. Rule: a top-down grasp holds whichever face is up at rest; one pass works only if that
face is also up in the target (fold-aware, 30°). Every part is planned **once, up front**, right
after the startup (home + gripper open, all table parts in view) and before any assembly motion;
with `--skip-startup` it plans from wherever you put the arm. Falls back to the recorded sequence
for a part it can't observe. A part already on an insertable face gets 1 pass (pick → rotate → insert).

**Cell layout** — `primitives/shared/cell_layout.yaml` names the assembly region (where the run's
base sits; the startup base read flies over it), the **clear area** (where a part is set down and
regrasped to reorient, `primitives/core/move_to_clear_area.py`; (−0.320, −0.500) m in the active
layout) and the **staging region** (where parts wait outside the assembly, with its camera view).
Pick a layout with `active:` or `FMB_LAYOUT=<name>`; add a layout rather than editing a used one.
The replay takes the base and parts from the assembly JSON (FMB1 base1, FMB3 base3), and the
per-part tables (grip width, grasp depth, base size) come from CAD for any assembly. Before regrasping there, the
replay checks the landing with the planner's rule and, if the part toppled onto a face that
can't be inserted in one pass, runs another place-down pass (max 2).

**Camera away from the table.** The D415 sits on the tool's −Y side. For a part laid on its side,
the two wrist rolls that give the same object pose put the camera above or below the gripper,
and which one is reachable is fixed by the grasp wrist (yaw vs yaw+180°). So the replay passes
`--base-name` to `move_to_grasp`, which picks the wrist whose next reorient is camera-up
(`primitives/shared/camera_up.py`); `rotate_object` prefers camera-up within an error bucket.

### Part left in the staging region (recovery)

When a run stops with the part at the clear area — e.g. "Object not detected" on the regrasp:

```bash
python3 -u -m primitives.control_gripper open --mode real         # fingers hide a part directly below
python3 -u -m primitives.core.move_to_clear_area --mode real       # keeps orientation; arm must be top-down
#   (use --fix-orientation instead if the arm is NOT top-down, e.g. after a reorient)
python3 -u ablations/replay_real_assembly.py --assembly-json $J --only <part> --skip-startup
```

The camera sees only what is under the arm: from home it cannot see the clear area, and over the
clear area it cannot see the table parts. A part is only ever in one of those two places.

### Stopping a run

```bash
bash compliant_insertion_studio/scripts/stop_run.sh
```

**First halts motion with the UR dashboard `stop`** — a trajectory the controller has already
accepted keeps running after its client process exits, so killing processes alone does NOT stop
the arm (2026-10-07: it kept moving and the operator had to force-stop). Then stops the replay and
every primitive with SIGINT → SIGTERM (never SIGKILL), waits for the insert wrapper's force-mode
cleanup, **re-sends the external-control program**, re-activates the trajectory controller, and
reports pose + gripper width. If the dashboard is unreachable and the arm is moving: pendant /
e-stop. Don't `kill` by PID: the background `bash` wrapper is not the replay.

Two insertion paths, routed in `primitives/translate_object.py` by `object_name`:

| object | path | typical |
|---|---|---|
| u_brown, u_orange | `compliant_insert` FSM (spiral SEARCH) | seats in <10 s, or on touchdown |
| line_green, inverted_u_yellow | `prismatic_peg_insertion` (stash) | its own 3-attempt recovery ladder |

**Don't trust the FSM's success label.** Verify from the raw CSV: `|tcp_z − predicted_seat| < 5 mm`
and motion stopped. Good seats land within ±1.7 mm.

### Single insert, part already held

```bash
python3 -u -m compliant_insertion_studio.scripts.run_assembly_step \
  --object-name u_brown --base-name base1 --grasp-id 1 --mode real \
  --already-held --current-object-orientation QX QY QZ QW \
  --autonomous-search --fz -9.0 --override-fz-cap
```

Search force defaults differ by entry point: `translate_object` passes the **tuned** F_press=9 /
Fmax=8; the older per-step examples used 5/5, which is weaker and demonstrably worse on u_brown.

### Shutdown

```bash
bash compliant_insertion_studio/scripts/close_robot.sh
pkill -SIGINT -f aruco_camera_localizer
P=grasp_points; pkill -SIGTERM -f "${P}_publisher"
```

---

## 3. Hard constraints

- **Force mode ≤ 5 N default**, ≤ 9 N with `--override-fz-cap`. Lateral ≤ 6 N (§10 of the skill).
- **Rotation stays LOCKED in SEARCH** — `selection_vector = (T,T,T,F,F,F)`. The skill's all-True
  rule is wrong here; see anti-patterns.
- **Settle the arm before any force zero or baseline** (≥1.5 s after a Cartesian move).
- **SIGTERM cleanup must reach safe-state exit** even mid-force-mode.
- **Held-object pose chains, not reads.** The gripper occludes the camera; chain
  `current_object_orientation` from the previous primitive's `__RESULT_JSON__`.
- **Safe height before `move_home` while holding a part** — `move_home` plans straight lines
  that ignore seated parts.
- **Module mode only:** `python3 -m primitives.X`, never `python3 primitives/X.py`.

---

## 4. Anti-patterns that have actually bitten

- **Taking a force reference on a moving arm.** The trajectory controller reports "complete" at
  commanded position, not at rest. Found in 4 files on 2026-08-16; symptoms are plausible wrong
  numbers, never errors — a 15.6 N bad zero drove the TCP **116 mm upward**. 0.0 s settle → 15.6 N;
  5.36 s → 0.11 N.
- **Single-sample contact thresholds.** The wrench carries ~60 ms impulses (measured
  `+0.18, +19.32, +56.98, +20.56, +0.36 N` with nothing touched). Require a sustain window.
- **Unlocking rotation in SEARCH.** Lateral force then applies a moment about the grasp point:
  the part pivots in the jaws, the gripper translates, and TCP displacement stops meaning peg
  displacement — making every swept-area number fiction. Cost: 5 failed runs. Check the
  `cmd_wrench_raw` sidecars of runs that seated before trusting a written rule.
- **Trusting a wrapper exit code.** `python3 …; echo "EXIT=$?"` exits 0 regardless.
- **Tuning parameters before analysing data.** Record the raw trace first — three "fixes" were
  attributed to a force number before anyone logged the actual signal.
- **`pkill -f` self-matching its own shell** (exit 144). Build the pattern at runtime:
  `P=grasp_points; pkill -f "${P}_publisher"`.
- **Killing the gripper bridge with `pkill -f gripper_control`** — won't match. Use
  `pkill -f "socat.*ttyUR"`, then wait 5 s before respawning (PTY race).
- **Observing a part the camera can't see** (2026-10-07). Two ways, both silent "not detected":
  the gripper half-closed at its pre-grasp width (fingers hide a part directly below), or the
  gripper horizontal after a reorient (camera not looking at the table). Open fully / go
  top-down first. `move_to_grasp` now opens once by itself if it can't see the part before
  step 1, and restores the pre-grasp width before descending.
- **`primitives.core.move_xyz` for a small move near parts.** It solves IK from a FIXED seed
  (not the current joints) and sends a single-point joint-space trajectory: it can pick another
  arm configuration and swing the whole arm. Gripping a seated line_green it went the wrong way
  (operator force-stop, 2026-10-07). Small straight moves: `move_to_safe_height --lift-by M`
  (Cartesian waypoints seeded from the current joints). Never shorten a primitive's duration
  without knowing the joint distance. Test new motion code with nothing gripped first.
- **Hard-cancelling a trajectory** (`cancel_goal_async`): the controller holds position at once —
  a full-speed stop. The `move_to_grasp` retreat now runs to completion instead.
- **Committing a motion target from one camera sample — or from samples taken in motion.** An
  11 mm z blip in a single grasp-point sample made a regrasp hold u_brown 10.9 mm off; the insert
  then rejected a genuine seat and pressed on it until stopped. The first fix (median of the last
  7 samples, approach included) gripped line_green ~9.5 mm off-centre: approach-phase estimates
  sat ~9-10 mm from the close-range one and outvoted it; the bar seated wrong. Now step 2 uses
  the median of samples taken **while the arm is stationary** (≥3; no extra wait — fewer samples → latest one). The offset itself is unexplained (calibration?) — not chased yet.

---

## 5. Where the detail lives

| need | doc |
|---|---|
| session handoff, open work | `compliant_insertion_studio/docs/HANDOFF_NEXT_AGENT.md` |
| FSM architecture, SEARCH director | `compliant_insertion_studio/docs/AUTONOMOUS_INSERTION_METHODOLOGY.md` |
| **binding rules for any FSM change** | `compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md` |
| why a rejected path was rejected | `compliant_insertion_studio/docs/ITERATION_TRACE_2026-05-06.md` |
| cold-start runbook, troubleshooting | `compliant_insertion_studio/docs/SETUP.md` |
| stack versions, alternatives | `compliant_insertion_studio/docs/STACK.md` |
| predicate + director specs | `compliant_insertion_studio/analysis/CONTROL_LAW.md`, `SEARCH_CONTROL_LAW.md` |
| per-object grasp id / width | `ablations/eval_resources/fmb1_assembly.json` |
| queued fixes | `docs/QUEUED_FIXES.md` |
| **debugging one part** (per-part iteration, no tipping, where the data is) | `compliant_insertion_studio/docs/DEBUG_LOOP.md` |

RG2 firmware quirks (no precise positioning mode; `/gripper_width` vs `/gripper_width_offset`
differ by 9.2 mm; safety-circuit latch needs a Compute Box power-cycle) — `SETUP.md` §6–7.

**Gripper safety-circuit latch — proven fix (2026-10-07).** `/gripper_status` shows
`Circuit1:True` or `Circuit2:True` (e.g. after a force-stop / safety FAULT) and the gripper
ignores commands. An **arm power cycle from the dashboard** clears it on this cell:
```bash
cd ~/Documents/prismatic-manipulation
python3 scripts/ur_dashboard.py --host 192.168.1.111 --real raw "power off"   # brakes hold the arm
sleep 10
python3 scripts/ur_dashboard.py --host 192.168.1.111 --real power_up         # power on + brake release
# then: resend the program, activate the controller, re-set the payload
ros2 service call /io_and_status_controller/resend_robot_program std_srvs/srv/Trigger "{}"
ros2 control switch_controllers --activate scaled_joint_trajectory_controller
ros2 service call /io_and_status_controller/set_payload ur_msgs/srv/SetPayload \
  "{mass: 2.1109, center_of_gravity: {x: -0.0032, y: 0.0031, z: -0.0318}}"
ros2 topic echo --once /gripper_status    # want Circuit1:False Circuit2:False
```
`/restart_gripper_connection` only reconnects the serial link — it does not clear the latch.
This RG2 has no body safety switches.

Sim/real parity: `primitives/shared/config.py` reads `ROS_MCP_MODE` at import, injected per
subprocess. Real path unchanged when unset.

---

## 5a. Debugging a part: iterate per part, not per cycle

When one part misbehaves, do not iterate through `fmb1_loop.py --layout staging` (generated layouts
+ tipping are the evaluation, and add minutes and failure modes per try). Pull just that part
(`disassemble.py --only PART --to-staging`), run just that part
(`replay_real_assembly.py --only PART --skip-startup [--part-at-staging]`), read the raw CSVs and the
CAD contact candidates before changing code, then put it back in the loop. The full procedure and
where each piece of data lives: `compliant_insertion_studio/docs/DEBUG_LOOP.md`.

## 6. Conventions

- Research before non-trivial code; clone references to `_references/repos/`.
- Say when you don't know. Don't invent an SOP by extrapolation.
- Ask before: new top-level dependency, >200 LOC unchecked, **any robot motion**, editing
  primitives outside `compliant_insertion_studio/`, departing from a documented decision.
- Don't commit unless asked. No `Co-Authored-By` trailers.
- `_references/` and `compliant_insertion_studio/logs/` are gitignored — never commit.

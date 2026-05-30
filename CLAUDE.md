<!-- GSD:project-start source:PROJECT.md -->

## ⚡ Next Agent Start Here

**Read `compliant_insertion_studio/docs/HANDOFF_NEXT_AGENT.md` first.** It tells you where you are, what works, what's open, and which docs to read in what order.

State as of **2026-05-07** (tag `real-world-verified-2026-05-07`, commit 75a418d): **all four FMB1 objects insert end-to-end via the replay script** — u_brown, u_orange, line_green, inverted_u_yellow seat correctly with zero manual intervention. Two insertion code paths in production (unification queued for later):

| Object | Path | Why |
|---|---|---|
| u_brown, u_orange | `compliant_insert` wrapper (FSM autonomous SEARCH) | tight-fit pegs, XY-locked descent |
| line_green, inverted_u_yellow | `prismatic_peg_insertion` (stash) via `translate_object` line_green/yellow branch | wide-grip parts; Rx/Ry compliance + geometric exit needed for jaws-on-rim seating |

Routing happens in `primitives/translate_object.py` based on `object_name`. Per-object base offsets in `primitives/shared/config.py:PER_OBJECT_BASE_OFFSET_M` are applied to BOTH paths via `--final-base-pos`.

Working command (full FMB1 assembly, all 4 objects on table, full stack up):
```bash
python3 -u ablations/replay_real_assembly.py \
  --assembly-json ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json \
  --only <object_name>      # one at a time, fail-stop
```
Or single-object autonomous (older API, still works for u_brown/u_orange):
```bash
python3 -u -m compliant_insertion_studio.scripts.run_assembly_step \
  --object-name <name> --base-name base1 --grasp-id <N> \
  --autonomous-search --search-F-press-N 5.0 --search-Fmax-N 5.0 \
  --fz -9.0 --override-fz-cap --mode real
```

Multi-iteration regrasp test harnesses:
```bash
# u_brown / u_orange (compliant_insert):
bash compliant_insertion_studio/scripts/loop_autonomous_insert.sh 3 --object <name> --no-randomize --regrasp
# line_green (prismatic via translate_object):
bash compliant_insertion_studio/scripts/loop_line_green_prismatic.sh 3
```

Doc routing (in order):
1. `compliant_insertion_studio/docs/HANDOFF_NEXT_AGENT.md` — this handoff
2. `compliant_insertion_studio/docs/AUTONOMOUS_INSERTION_METHODOLOGY.md` — current working architecture (compliant_insert path)
3. `compliant_insertion_studio/docs/ITERATION_TRACE_2026-05-06.md` — full reasoning trace (avoid re-exploring rejected paths)
4. `compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md` — methodology rules (binding for FSM changes)
5. `compliant_insertion_studio/analysis/CONTROL_LAW.md` + `SEARCH_CONTROL_LAW.md` — predicate + director specs

Reference (consult on demand, not on every session):
- `compliant_insertion_studio/docs/STACK.md` — full tech stack tables, version compat, alternatives considered
- `compliant_insertion_studio/docs/RATIONALE.md` — detailed rationale + historical Phase 5 architecture (now superseded)
- `compliant_insertion_studio/docs/SETUP.md` — cold-start runbook (canonical bringup; see §1 for grasp_points_publisher requirement)

---

## Commands (the scripts you'll actually call)

### Bringup (start each session)

```bash
# 1. Start the robot stack (UR driver + controllers, ~15s; add --rviz to also launch RViz):
bash compliant_insertion_studio/scripts/launch_robot.sh real

# 2. Start the camera (aruco_camera_localizer publishing /objects_poses_real):
bash compliant_insertion_studio/scripts/launch_camera.sh --background

# 3. Start the grasp-points publisher (publishes /grasp_points_real, required by
#    move_to_grasp). DIES SILENTLY if the camera localizer restarts — recheck
#    `ros2 topic list | grep grasp_points_real` before any pickup. (mode=default
#    publishes both _real and _sim topics.)
nohup python3 -u utils/grasp_points_publisher.py --mode default > /tmp/grasp_pub.log 2>&1 &
```

After all three are up, **STOP then PLAY on the pendant** to (re)attach external control. Verify with `ros2 topic hz /tcp_pose_broadcaster/pose` (~500 Hz) and `ros2 topic hz /grasp_points_real` (~5 Hz).

### Shutdown

```bash
bash compliant_insertion_studio/scripts/close_robot.sh   # cleanly stops driver + RViz + grippers
pkill -SIGINT -f aruco_camera_localizer
pkill -SIGTERM -f grasp_points_publisher
```

### Autonomous insertion (production)

```bash
# Single run from object on table → pick → rotate → place → regrasp → rotate → insert:
python3 -u -m compliant_insertion_studio.scripts.run_assembly_step \
  --object-name <name> --base-name base1 --grasp-id <N> \
  --autonomous-search --search-F-press-N 5.0 --search-Fmax-N 5.0 \
  --fz -9.0 --override-fz-cap --mode real
# (--grasp-width auto-resolves from fmb1_assembly.json. --grasp-id auto-resolved by loop wrapper.)

# Multi-iteration test with fresh regrasps between runs:
bash compliant_insertion_studio/scripts/loop_autonomous_insert.sh 3 \
  --object <name> --no-randomize --regrasp

# Single autonomous run when peg is ALREADY held (chains from prior insert):
python3 -u -m compliant_insertion_studio.scripts.run_assembly_step \
  --object-name <name> --base-name base1 --grasp-id <N> \
  --already-held --current-object-orientation QX QY QZ QW \
  --autonomous-search --search-F-press-N 5.0 --search-Fmax-N 5.0 \
  --fz -9.0 --override-fz-cap --mode real

# Re-grasp a held part (release at clear area + camera-grasp again):
python3 -u -m compliant_insertion_studio.scripts.regrasp_held_object \
  --object-name <name> --grasp-id <N> --mode real --skip-camera-check
```

### Data collection (new objects / re-calibration)

```bash
# Single GUIDED demo for new object — operator drags peg to hole, hits Enter to mark:
python3 -u -m compliant_insertion_studio.scripts.collect_regime_data \
  --object <name> --base base1 --grasp-id <N> --grasp-width <W> \
  --fz 9.0 --step-back-seconds 5.0 \
  --held-quat QX QY QZ QW \
  --variations A_pos_x_10mm --reps 1
# Reads hole_observed_operator from meta to derive bias for DEFAULT_BASE_POSITION.
```

### Send SIGUSR1 to the wrapper (for GUIDED data collection mid-drag)

```bash
# Find PID, then signal:
kill -SIGUSR1 $(pgrep -f "compliant_insertion_studio.wrapper.compliant_insert" | tail -1)
# Or send SIGUSR2 to mid-episode re-zero F/T.
```

### Diagnostics / debugging

```bash
# Verify camera publishing (5s timeout):
timeout 5 ros2 topic echo --once /objects_poses_real

# Verify wrench rate (~500 Hz expected):
ros2 topic hz /force_torque_sensor_broadcaster/wrench

# Read current TCP pose:
ros2 topic echo --once /tcp_pose_broadcaster/pose

# Stop the robot stack cleanly (use this, not pkill):
pkill -SIGINT -f "ros2.*launch.*ur5e"
```

### Common abort/cleanup signals

```bash
# Send SIGTERM to autonomous run (wrapper triggers safe-state cleanup):
pkill -SIGTERM -f "compliant_insertion_studio.scripts.run_assembly_step"

# Hard-stop the gripper bridge (NOT pkill -f gripper_control — won't match):
pkill -f "socat.*ttyUR"
# Wait 5+ seconds before respawning to avoid PTY/termios race.
```

---

## Project

**Compliant Insertion Studio** — force-compliant peg-in-hole insert primitive on UR5e + OnRobot RG2. Replaces the broken `prismatic_peg_insertion.py` real-mode insert path. Per-object parameterized algorithm. Proof-of-concept: FMB1 assembly (u_brown, u_orange, line_green, inverted_u_yellow). Single-config-file extension to new parts.

**Stack** (full detail in `docs/STACK.md`): ROS2 Humble + Python 3.10 + `rclpy` + `ur_robot_driver` 2.12 + OnRobot RG2 driver. Force mode via `ur_msgs/srv/SetForceMode` only — no URScript injection.

### Hard constraints

- **Pendant in Local mode.** No `dashboard_client/recover` calls.
- **Force-mode wrench ≤ 5 N default.** Higher only with explicit operator approval + `--override-fz-cap`.
- **SIGTERM cleanup must reach safe-state exit** even if force_mode is partway down. Use `os.setsid` + process-group SIGTERM in subprocess chains.
- **Hands-off during F/T zero**: operator confirmation gate, +1 s post-zero drift check, no operator load during baseline windows.
- **Safe height before move_home when holding a part.** Direct `move_home` plans straight-line trajectories that ignore inserted bases.
- **Don't commit unless explicitly approved.** No `Co-Authored-By` lines.
- **`_references/` and `compliant_insertion_studio/logs/` are gitignored.** Never commit.
- **All project code under `compliant_insertion_studio/`.** Touch `primitives/` only for host-repo bug fixes.

### F/T calibration is three layers

| Layer | Frequency | Purpose | Trigger |
|---|---|---|---|
| **Foundational** payload calibration | Per gripper mount (one-time) | Mass + CoG + bias via `set_target_payload`; written into `launch_robot.sh` | New gripper / sensor remount / orientation-bias observed |
| **Session** F/T smoke test | Per session | Zero + 5s hold + bias verification in known neutral pose | Start of session / after protective stop / after physical bump |
| **Per-pose** `zero_ftsensor` | Immediately before force-mode | Single-pose bias subtraction | Inside wrapper's PRE phase |

`zero_ftsensor` does **not** substitute for foundational. `set_payload` from `launch_robot.sh` is what gives force_mode correct gravity comp — pendant payload setting does NOT propagate.

---

## Conventions

### Key rules

- **Research before code**: spend 5–15 min on WebSearch/WebFetch before any non-trivial deliverable. Clone references to `_references/repos/`, save articles to `_references/articles/`.
- **Honesty over confidence**: if you don't know, say so. Pause and research; don't invent SOPs from extrapolation.
- **Per-piece copy/modify/write-fresh decision** when borrowing from references. Credit sources in code comments.
- **Inline default**: do work in main conversation so operator can see/intervene. Subagents only for parallel + independent + artifact-producing work.
- **Phase boundaries are guidance, not gates**: finish coupled work together; update REQUIREMENTS.md traceability where it actually completed.
- **Held-object pose chains, not reads**: when gripper holds a part, NEVER read its pose from `/objects_poses_real` (camera occluded by gripper). Chain `current_object_orientation` from prior primitive's `__RESULT_JSON__` output.
- **Strip ANSI before parsing ROS2 CLI**: `ros2 control list_controllers` and others emit `\x1b[…m` color escapes even on pipe. Use `re.sub(r'\x1b\[[0-9;]*m', '', line)` before tokenizing.

### Ask the operator before

Adding a new top-level dependency, writing > 200 LOC without checkpoint, performing any robot motion, modifying primitives outside `compliant_insertion_studio/`, or departing from a documented decision in PROJECT/REQUIREMENTS/ROADMAP.

### Anti-patterns (don't)

- Tuning parameters before analyzing data. Data first, structural change second.
- Subagents for routine work.
- Treating `translate_object --insert` as the insert path. The new wrapper (`compliant_insert.py`) is the replacement.
- Launching primitives via script path: `python3 primitives/move_to_safe_height.py` fails with ModuleNotFoundError because primitives import siblings. Use module mode: `python3 -m primitives.move_to_safe_height`.
- Treating wrench data as `base_link` frame. The CSV `wrench_frame_id = tool0_controller`. Direction-aware features (r_cop, F_lat) MUST be computed in tool frame.
- Counter-residual direction for force corrections during wedge-breaking. When peg is wedged at corner X, wrist sensor reads OPPOSITE direction. Use CAD-derived TOWARD-target instead.
- Pushing harder downward to break peg-on-rim wedges. Empirically deepens the wedge. Right action: retract 0.5–1.5mm + drop Fz to -2 to -4N + spiral search at lower gains. (Sources in `RATIONALE.md`.)
- Detecting "stuck" from instantaneous v_z. Force-mode oscillation makes v_z dip momentarily even mid-wedge. Use net z-descent over 2s window with Fz smoothed over 0.5s.
- Killing the gripper bridge with `pkill -f gripper_control` (won't match — actual cmdline is `python3 /opt/ros/humble/bin/ros2 run …`). Use `pkill -f "socat.*ttyUR"` or `kill -9 <PID>` after `ps aux | grep gripper_control`.
- Restarting socat-using processes without 5+ second wait between kill and respawn (PTY/termios race produces `(22, 'Invalid argument')` on next pyserial open).
- **Self-matching `pgrep -f` in `until` loops.** A bash one-liner like `until ! pgrep -f "loop_iterate.*u_orange" >/dev/null; do sleep 2; done` (run via Bash-tool `eval`) **never exits** because the spawning bash itself contains the literal pattern in its cmdline — pgrep matches its own host shell. Fixes: (a) match python invocation specifically (`pgrep -f "python3.*loop_iterate"`), (b) wait on a known PID with `while kill -0 <PID> 2>/dev/null`, or (c) use the Monitor tool.

Plus the autonomous-SEARCH-specific anti-patterns in `docs/HANDOFF_NEXT_AGENT.md` "Anti-patterns the current session worked through (DO NOT repeat)".

### Decision matrix — copy / modify / write-fresh

| Decision | When |
|---|---|
| **Copy** (lift file, attribute source) | Same language + framework + license + fits architecture as-is |
| **Modify after copying** | Mostly fits, surface tweaks only (paths, message types) |
| **Write fresh from algorithm/pattern** | Different language/framework/era; translate the *idea*, not lines |
| **Skip** | Doesn't fit stack/scope (e.g., needs accelerometer we don't have) |

---

## OnRobot RG2 firmware quirks (verified 2026-05-03)

- **No precise positioning mode**: only modes 1 (grip), 8 (stop), 16 (grip_w_offset). Both 1 and 16 are GRIP commands — close past target by 1-5mm. Width-based grasp checks must tolerate ≥5mm error.
- **Safety circuit latch**: bits 3, 5 of status reg 268. Per OnRobot docs: "can only be reset by power cycling." Software path: Modbus write `unit=63 addr=0 value=2` triggers Compute Box power-cycle (~10s); requires pendant STOP+PLAY after.
- **Width topics differ by 9.2mm**: `/gripper_width` is RAW mechanism, `/gripper_width_offset` is jaw-tip-to-jaw-tip gap (raw − 2 × 4.6mm fingertip). For grasping, use `/gripper_width_offset`.

---

## Subprocess invocation rule

Inside the wrapper / orchestrator scripts, ALWAYS launch primitives via module mode:

```
python3 -m primitives.move_to_safe_height ...    # CORRECT
python3 primitives/move_to_safe_height.py ...    # WRONG (ModuleNotFoundError, often swallowed)
```

---

## Per-object lookup (auto-resolved)

`primitives/shared/config.py:get_gripper_width_mm(object, grasp_id)` and `get_grasp_id_for_assembly(object)` read from `ablations/eval_resources/fmb1_assembly.json`. `run_assembly_step.py`, `regrasp_held_object.py`, and `loop_autonomous_insert.sh` auto-resolve. Pass `--grasp-width N` or `--grasp-id N` to override.

---

`docs/RATIONALE.md` has the long-form WHY for these rules + the legacy Phase 5 architecture (Mode A/B, iterative-loop workflow, etc.) which has been superseded by the autonomous SEARCH director.

## 2026-05-28 — mode-aware sim/real config (parity)

`primitives/shared/config.py` reads `RUNTIME_MODE` from env `ROS_MCP_MODE` at import.
`server_remap.py::_run_primitive` injects `ROS_MCP_MODE` from the tool's `--mode`, so each
primitive subprocess gets the right values (primitives bind config constants at import time,
so the env must be set before the process starts).

Mode-aware values (sim | real):
- `ROBOT_BASE_Z`: 0.0 | 0.08 → drives TABLE_HEIGHT / SAFE_HEIGHT / DEFAULT_BASE_POSITION.
  Sim robot+floor at z=0 (Isaac convention); real robot on an 8 cm mount. Without this, sim
  `place_down` drove to world −0.08 → INTO the ground plane → gripper orientation wrecked →
  assembly failures. (6e120af)
- `GRIPPER_CENTER_TOOL_OFFSET`: 0.23 | 0.2286 (flange → gripper-center, tool Z). (6e120af)
- `get_object_base_offset_m`: returns (0,0,0) in sim — real-arm calibration corrections don't
  apply to the CAD-exact twin. (6e120af)
- `SAFE_HEIGHT_ABOVE_TABLE = 0.50` m (hover height above table). (d8f9118)

The real path is unchanged when `ROS_MCP_MODE` is unset/`real` — the `real-world-verified-2026-05-07`
tag behavior is intact. Verified 2026-05-28: ground-truth FMB1 replay seats 4/4 in sim.

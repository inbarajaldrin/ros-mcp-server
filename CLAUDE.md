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
insert → release); `line_green` skips the regrasp.

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

RG2 firmware quirks (no precise positioning mode; `/gripper_width` vs `/gripper_width_offset`
differ by 9.2 mm; safety-circuit latch needs a Compute Box power-cycle) — `SETUP.md` §6–7.

Sim/real parity: `primitives/shared/config.py` reads `ROS_MCP_MODE` at import, injected per
subprocess. Real path unchanged when unset.

---

## 6. Conventions

- Research before non-trivial code; clone references to `_references/repos/`.
- Say when you don't know. Don't invent an SOP by extrapolation.
- Ask before: new top-level dependency, >200 LOC unchecked, **any robot motion**, editing
  primitives outside `compliant_insertion_studio/`, departing from a documented decision.
- Don't commit unless asked. No `Co-Authored-By` trailers.
- `_references/` and `compliant_insertion_studio/logs/` are gitignored — never commit.

# Compliant Insertion Studio — Full Setup & Bring-Up Guide

End-to-end runbook for getting from cold-machine to ready-to-collect-data on
the UR5e + OnRobot RG2 + workspace-camera stack. **Read this top-to-bottom
before the first bring-up of a fresh session.**

> **Audience:** operator (or next-resume Claude agent) who needs to land in
> this project, get the whole stack running, and either verify a wrapper
> or collect Phase 3 demo data.
>
> **Latest update:** 2026-08-16 — bring-up is now **headless** (§3.1) and the real
> assembly runs only from branch `ur5e-fmb1-demo` (§0). Sections 4.3, 6 and 9 are
> **historical**, dated 2026-05-03, and are marked as such in place.
>
> **Fastest path on `ur5e-fmb1-demo`:** `bash compliant_insertion_studio/scripts/stack_up.sh`
> brings up driver → gripper → camera → grasp points in one command, verifying each stage;
> `stack_down.sh` tears it down. §3 below is the per-stage version of the same sequence, for
> when you need to bring one piece up by hand or debug a stage.

---

## 0. Which tree — read before running anything real

**Branch `ur5e-fmb1-demo` (working tree `~/Documents/ros-mcp-server-verified`) is the tree that
runs the real assembly.** `main` cannot: its grasp publisher emits composite marker ids
(101…303) after the June candidate-native migration, while the assembly JSONs still carry flat
ids (1, 2), so every pick fails with `"Grasp point 1 not found"`. The replacement
`--grasp-candidate` flag refuses `--mode real` by design. Use `ur5e-fmb1-demo`, or a worktree at
tag `real-world-verified-2026-05-07` — and start the grasp points publisher **from that same
tree** (§3.3).

---

## 1. Repos & data — what lives where

| Path | What | Owned by | Tracked? |
|---|---|---|---|
| `~/Documents/ros-mcp-server/` | **THIS repo** — primitives, wrapper, scripts, planning | Project | git |
| `~/Documents/ros-mcp-server/compliant_insertion_studio/` | **Project deliverables** — wrapper FSM, orchestrator, configs, telemetry, dashboard, analysis | Project | git |
| `~/Desktop/ros2_ws/src/onrobot_ros/` | OnRobot RG2 ROS2 bridge — **modified locally** (see §6 Bug Fixes) | External (vendor fork) | git (separate repo) |
| `~/Desktop/ros2_ws/src/aruco_camera_localizer/` | Camera localization node (publishes `/objects_poses_real`) | External | git (separate repo) |
| `~/Documents/aruco-grasp-annotator/data/` | **Data dependency**: grasp points + assembly definitions | External | data-only |
| `~/Documents/aruco-grasp-annotator/data/grasp_points/<obj>_grasp_points.json` | Per-object grasp IDs + validity widths | External | data |
| `~/Documents/aruco-grasp-annotator/data/fmb_assembly1.json` | Base+object names recognized by the system | External | data |
| `ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json` | **Canonical sequence** that `run_assembly_step.py` implements | Project | git |
| `compliant_insertion_studio/configs/ft_calibration_*.yaml` | Per-mount F/T calibration output | Project (gitignored) | local-only |
| `compliant_insertion_studio/logs/insert_*.csv` + `.meta.json` | Episode telemetry (Phase 3 deliverables) | Project (gitignored) | local-only |

**Hardware:** UR5e at `192.168.1.111` (operator's workstation at `192.168.1.10`), OnRobot RG2 attached at `tool0`, RealSense camera over the workspace.

**Pendant mode: REMOTE CONTROL** (top-right in PolyScope). Headless bring-up cannot be established from Local. Only the pendant needs to be booted — no program is loaded and nothing is played. With Remote set, `dashboard_client` power-on / brake-release / mode queries work; **clearing a protective stop is still a manual action at the pendant**, and wrapper code must not assume `dashboard_client/recover` works.

---

## 2. One-time setup (verify before first run)

### 2.1 ROS2 environment
```bash
# Verify ROS2 Humble is sourced and packages are installed
source /opt/ros/humble/setup.bash
ros2 pkg list | grep -E "ur_robot_driver|ur_msgs|controller_manager"   # all three required
ros2 pkg list | grep -E "aruco_camera_localizer|onrobot_ros"           # external pkgs
```

### 2.2 ros2_ws built (with local onrobot_ros modifications)
```bash
cd ~/Desktop/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select onrobot_ros aruco_camera_localizer
source ~/Desktop/ros2_ws/install/setup.bash
```

### 2.3 Confirm grasp points data is present
```bash
ls ~/Documents/aruco-grasp-annotator/data/grasp_points/
# must contain: u_brown_grasp_points.json, u_orange_..., line_green_..., inverted_u_yellow_..., etc.
```

### 2.4 Confirm F/T payload calibration is in bringup
**Payload setting is automatic** as of 2026-05-06. `launch_robot.sh real` reads the most-recent `compliant_insertion_studio/configs/ft_calibration_*.yaml` and calls `/io_and_status_controller/set_payload` with the calibrated mass + CoG after the controller manager comes up. Watch the bringup output for `[launch_robot] Setting driver payload from ft_calibration_*.yaml:` followed by `success=True` — if that line is missing, force_mode will have ~21N of unmodeled gravity and TCP will drift ~8mm during APPROACH instead of <1mm. **Current calibration**: `mass=2.1109 kg, CoG=[-0.0032, +0.0031, -0.0318]`.

> **Why this matters and why it bit us**: from project start through 2026-05-05, the docs claimed `set_target_payload(...)` was "pasted into the bringup launch file." It wasn't. The pendant payload setting (set via the teach pendant) does NOT propagate to the ROS2 driver's force_mode controller — they're separate. Without `set_payload` over the ROS service, force_mode used 0 kg for gravity comp, peg gravity created ~21N of unmodeled wrist load, controller yielded → 7-8mm of TCP drift during the 19s APPROACH descent. Empirical fix on 2026-05-06: calling `set_payload` reduced drift from 7.85mm → 0.83mm (9.5× tighter). `launch_robot.sh` now does this automatically, sourcing the values from the calibration YAML so updates flow through cleanly.

If you re-mounted the gripper or replaced fingertips:
```bash
cd ~/Documents/ros-mcp-server
PYTHONPATH=$(pwd):$PYTHONPATH python3 -m compliant_insertion_studio.shared.ft_calibration --gripper-id <name>
# Outputs new YAML + the set_target_payload(...) line to paste into bringup
```

---

## 3. Cold-start bring-up (every session)

Run these in order. Each step has its own check.

### 3.1 Real bringup + RViz (headless)
```bash
cd ~/Documents/ros-mcp-server-verified          # the ur5e-fmb1-demo tree — see §0
bash compliant_insertion_studio/scripts/launch_robot.sh real --headless --rviz
```
- Pings robot at `192.168.1.111` first (fails fast if unreachable)
- Launches `ur5e_with_rg2.launch.py` with real hardware
- `--headless`: the driver sends its own control script over the primary interface, so **no
  External Control `.urp` is loaded and nothing is played on the pendant**. There is no program
  named `external_control.urp` on this controller — don't go looking for one.
- Sets the F/T payload from `configs/ft_calibration_*.yaml` — **required** (§2.4)
- Activates `scaled_joint_trajectory_controller`
- Logs to `/tmp/ur_bringup_logs/real_bringup_<ts>.log`; wait for `Robot connected to reverse interface`

**On the pendant:** power on, release brakes, and make sure the mode selector reads **Remote
Control**. That is all — nothing to load, nothing to play.

**Verify:**
```bash
ros2 control list_controllers | grep -E "scaled_joint|force_torque"
# expect both ACTIVE
```

> **URCap path (only if you deliberately omit `--headless`)**
> Then the pendant must have an External Control `.urp` loaded and you press **Play** ▶ yourself,
> and after any bringup restart you must press **STOP** ⏹ then **PLAY** ▶ again — the URCap loses
> its link on restart and without the cycle ros2_control trajectories silently fail with
> "velocity limits exceeded" while `program_running` still reports true. Verify with
> `ros2 service call /dashboard_client/program_running ur_dashboard_msgs/srv/IsProgramRunning`.
> Headless bring-up has neither the step nor the failure mode.

### 3.2 Camera node (publishes `/objects_poses_real`)
```bash
bash compliant_insertion_studio/scripts/launch_camera.sh --background
# Logs to /tmp/aruco_logs/aruco_<ts>.log
```
**Verify:**
```bash
ros2 topic echo --once /objects_poses_real | head -10
# expect: transforms with child_frame_id like "base1", "u_brown", etc.
```

### 3.3 Grasp points publisher (publishes `/grasp_points_real`)

**Must run from the `ur5e-fmb1-demo` tree** (§0). `main`'s copy publishes composite marker ids and
every pick fails, no matter which tree the assembly script itself runs from.

```bash
nohup bash -c 'source /opt/ros/humble/setup.bash; \
    source ~/Desktop/ros2_ws/install/setup.bash; \
    cd ~/Documents/ros-mcp-server-verified; \
    python3 -u utils/grasp_points_publisher.py --mode real' \
    > /tmp/grasp_pub.log 2>&1 &
```
**Verify:**
```bash
ros2 topic list | grep grasp_points_real
ros2 topic hz /grasp_points_real    # expect ~5 Hz
```
This publisher **dies silently** if the camera localizer restarts. Re-check the topic before any
pick.

### 3.4 OnRobot RG2 bridge (publishes `/gripper_width`, accepts `/gripper_command`)
```bash
nohup bash -c 'source /opt/ros/humble/setup.bash; \
    source ~/Desktop/ros2_ws/install/setup.bash; \
    ros2 run onrobot_ros gripper_control' \
    > /tmp/gripper.log 2>&1 &
sleep 8
tail -5 /tmp/gripper.log    # expect "Gripper initialized successfully"
```
**Verify:**
```bash
ros2 topic echo --once /gripper_status
# expect: Motion:False, Grasp:..., Width:..., Safety1:False, Safety2:False, Circuit1:False, Circuit2:False
# (Circuit1/Circuit2 fields require the local patch — see §6)
```

### 3.5 Test gripper actuates
```bash
cd ~/Documents/ros-mcp-server
python3 primitives/control_gripper.py open --mode real    # should move to ~101mm
python3 primitives/control_gripper.py close --mode real   # should move to ~1mm
python3 primitives/control_gripper.py open --mode real
```
If `close` doesn't move and width stays the same: **see §7 Troubleshooting → Stuck gripper.**

---

## 4. Run an assembly step (Phase 3 entry point)

The orchestrator implements the **canonical sequence** from `ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json`:

```
control_gripper width → move_to_grasp → close → safe_height
  → rotate_object → translate_object --place-down
  → control_gripper width → safe_height → move_to_grasp (regrasp) → close → safe_height
  → rotate_object (second) → compliant_insert wrapper (PRE→HOVER→ZERO→ACTIVE→DONE)
```

The **wrapper** replaces the legacy `translate_object --insert` (step 13 of the canonical) and handles its own cleanup (stop force mode → switch to position controller → safe height → home) on success or failure.

### 4.1 Full sequence (object on the table)
```bash
cd ~/Documents/ros-mcp-server
PYTHONPATH=$(pwd):$PYTHONPATH python3 -m compliant_insertion_studio.scripts.run_assembly_step \
    --object-name u_brown \
    --base-name base1 \
    --grasp-id 1 \
    --fz 3.0 \
    --step-back auto \
    --step-back-seconds 5
```
Outputs: `compliant_insertion_studio/logs/insert_u_brown_<ts>.csv` + `.meta.json`

### 4.2 Already-held mode (object already in gripper)
```bash
PYTHONPATH=$(pwd):$PYTHONPATH python3 -m compliant_insertion_studio.scripts.run_assembly_step \
    --object-name u_brown \
    --base-name base1 \
    --grasp-id 1 \
    --already-held \
    --current-object-orientation -0.005 -0.7058 0.7083 -0.0045 \
    --fz 3.0
```
The orientation must come from the **previous primitive's output** (state-tracking through the chain), NOT a fresh camera read — the camera can't reliably see the held part through gripper occlusion.

### 4.3 Per-object grasp_id & grasp_width

> **CORRECTED 2026-08-16.** All four parts are confirmed and have been running in production
> since 2026-05-07. **Nothing needs verifying here.** `grasp_id` and `gripper_width_mm` are
> auto-resolved from `ablations/eval_resources/fmb1_assembly.json` via
> `primitives.shared.config.get_gripper_width_mm()` / `get_grasp_id_for_assembly()` —
> `run_assembly_step.py`, `regrasp_held_object.py` and `loop_autonomous_insert.sh` all do this.
> Pass `--grasp-width N` / `--grasp-id N` only to override. The original text and the manual
> lookup snippet are kept below for the case where you're adding a **new** part.

*(Historical, 2026-05-03.)* Confirmed: `u_brown`: `--grasp-id 1 --grasp-width 35`. Needed
verification at the time: `u_orange`, `line_green`, `inverted_u_yellow`. Manual lookup for a new
part:

```bash
for obj in u_orange line_green inverted_u_yellow; do
    echo "=== $obj ==="
    python3 -c "
import json
d = json.load(open('/home/aaugus11/Documents/aruco-grasp-annotator/data/grasp_points/${obj}_grasp_points.json'))
for g in d['grasp_points']:
    w = g.get('grasp_validity', {}).get('x_axis_gripper_width_mm', '?')
    print(f'  id={g[\"id\"]}  width={w}mm  pos={g[\"position\"]}')
"
done
```

---

## 5. Cleanup (end of session)

```bash
bash compliant_insertion_studio/scripts/close_robot.sh   # graceful shutdown of bringup + RViz
pkill -SIGTERM -f aruco_camera_localizer
pkill -SIGTERM -f grasp_points_publisher
pkill -9 -f gripper_control                              # bridge leaves stale socat otherwise
pkill -9 -f "socat.*ttyUR"
```
**Why `kill -9` for `gripper_control` and `socat`:** they're not X11-bearing processes, and the bridge's `kill_socat()` cleanup is unreliable across rapid restarts. SIGKILL is safe here. *Never* use `kill -9` on RViz / Gazebo / tkinter — see global CLAUDE.md.

**Pendant:** nothing to do under headless bring-up — no program is running. (URCap path only: stop the External Control program manually with `STOP` ⏹ before ending if you'll restart bringup later.)

On `ur5e-fmb1-demo`, `bash compliant_insertion_studio/scripts/stack_down.sh` does the whole teardown in one command.

---

## 6. Bug-fix inventory (historical — 2026-05-03)

> **Historical record.** All of these were committed long ago and are in the tree; the section is
> kept because §7's troubleshooting entries refer back to it. Do not read it as pending work.

These were the local modifications that made WRAP-VERIFY work end-to-end.

### 6.1 In THIS repo (`~/Documents/ros-mcp-server`)

| File | Change | Why |
|---|---|---|
| `compliant_insertion_studio/wrapper/compliant_insert.py` | Strip ANSI escapes in `_list_active_controllers` (`_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')`); `import re` added | **Critical silent bug** — `ros2 control list_controllers` emits ANSI codes even to pipes; `parts[-1]` was always `'\x1b[0m'`, never `'active'`. Every controller switch silently aborted the wrapper. |
| `compliant_insertion_studio/wrapper/compliant_insert.py` | `_await_controller_active` default + both call sites: timeout `2.0s → 5.0s` | Manual switch takes 1.2s. The 2s ceiling was on the edge and tripped false aborts. |
| `compliant_insertion_studio/wrapper/_run_hover.py` | `node = TranslateObject()` → `node = TranslateObject(mode="real")` | `TranslateObject.__init__` requires `mode`. HOVER subprocess crashed on TypeError without it. |
| `primitives/move_to_grasp.py:888` | `current < (expected - 2.0)` → `current < (expected - 5.0)` | OnRobot RG2 mode 16 has inherent ±3-5mm grip overshoot (firmware-level). 2mm tolerance false-rejected real-world grasps. See §8 Reference notes. |
| `compliant_insertion_studio/scripts/launch_camera.sh` (NEW) | Helper for launching `aruco_camera_localizer` | Captures the ad-hoc command operator was reciting. |
| `compliant_insertion_studio/scripts/run_assembly_step.py` (NEW) | Reusable orchestrator implementing canonical assembly sequence | Phase 3 collection entry point. One CLI per assembly step. |

### 6.2 In external repo (`~/Desktop/ros2_ws/src/onrobot_ros`)

| File | Change | Why |
|---|---|---|
| `onrobot_ros/rg_gripper.py` (`publish_gripper_status`) | `/gripper_status` formatter now appends `Circuit1:{...}, Circuit2:{...}` after `Safety1/Safety2` | The CIRCUIT bits (status reg 268 bits 3, 5) are the actual "won't move" signal when the gripper safety latches. The original formatter only showed SWITCH bits (2, 4) which made the long-standing stuck-gripper bug invisible. |

**Rebuild after any change to the external repo:**
```bash
cd ~/Desktop/ros2_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select onrobot_ros
# then restart bridge (§3.4)
```

---

## 7. Troubleshooting

### 7.1 Stuck gripper (won't move)

**Diagnosis** — read the patched `/gripper_status`:
```bash
ros2 topic echo --once /gripper_status
# Look for: Circuit1:True or Circuit2:True
# → Safety circuit is latched. Gripper firmware refuses motor commands.
```

**Software fix** (Modbus power-cycle — DOCUMENTED but NOT yet auto-wired):
1. Send Modbus write: `unit=63 (Compute Box), addr=0, value=2`
2. Gripper Compute Box power-cycles (~10s)
3. Re-establish the tool link: under **headless** bring-up, restart the driver (§3.1). Under the URCap path, press **STOP then PLAY** on the pendant.
4. Re-launch the bridge (§3.4)
5. Re-verify `Circuit1:False` `Circuit2:False`

Reference implementation in upstream Osaka lib: `restartPowerCycle()` in `onrobot_rg_modbus_tcp/comModbusTcp.py`. **Caveat:** the Modbus connection breaks during reboot (expected proof-of-cycle), and the tool link goes stale until step 3.

**Manual fallback:** physically cycle the safety switches on the RG2 body (push-release each). Or full robot power-cycle.

### 7.2 Force mode controller "did not become active"

**This was the show-stopper bug fixed this session** — see §6.1. If you see this error after the fix is applied, check:
- The wrapper file actually has the ANSI strip: `grep _ANSI_ESCAPE_RE compliant_insertion_studio/wrapper/compliant_insert.py`
- Manual switch works: `ros2 control switch_controllers --activate force_mode_controller --deactivate scaled_joint_trajectory_controller`

### 7.3 `move_to_grasp` rejects with "Gripper is not open"

**This was the 2mm-tolerance bug fixed this session** — see §6.1. If you see this after the fix, check:
- Tolerance: `grep "expected_gripper_width - 5.0" primitives/move_to_grasp.py` (should match)
- Actual gripper width when called: `ros2 topic echo --once /gripper_width_offset`
- Expected width per object's grasp validity in `<object>_grasp_points.json`

### 7.4 Bringup runs but trajectories rejected with "velocity or acceleration limits exceeded"

**URCap path only.** The URCap link is stale after a bringup restart — **Pendant: STOP then PLAY**. Under headless bring-up (§3.1) there is no URCap link to go stale and this failure mode does not occur; if you see it, check that `--headless` was actually passed and that the pendant is in Remote Control.

### 7.5 ANSI codes in script output

Scripts that parse `ros2 control list_controllers` or similar must strip ANSI escapes (`\x1b[XXm`). This bit us once already. If a new diagnostic script returns empty when it shouldn't, suspect ANSI.

---

## 8. Reference notes (research findings, this session)

### 8.1 OnRobot RG2 firmware control modes (per upstream takuya-ki/onrobot-rg + official docs)

| Mode | Name | Width semantics | Behavior |
|---|---|---|---|
| 1 | `grip` | RAW mechanism width | Close until target raw width OR force. ±2mm overshoot. |
| 8 | `stop` | (n/a) | Halt current motion. |
| 16 | `grip_w_offset` | JAW-GAP (offset-corrected) | Close until target jaw gap OR force. ±3-5mm overshoot. |

**There is NO precise positioning mode.** Both 1 and 16 are GRIP modes; the gripper closes/opens past the target by 1-5mm depending on motion direction. Other ROS drivers in the wild (`takuya-ki/onrobot-rg`, `tonydle/OnRobot_ROS2_Driver`) don't compensate — they accept the firmware behavior. We do too: the 5mm tolerance widening in §6.1 matches firmware reality.

Sources:
- https://github.com/takuya-ki/onrobot-rg
- https://github.com/tonydle/OnRobot_ROS2_Driver
- https://onrobot.com/sites/default/files/documents/RG2_User%20_Manual_enEN_V1.9.2.pdf

### 8.2 Width readings — three sources, one truth

| Topic / source | What it reports |
|---|---|
| `/gripper_width` | Raw mechanism width (jaw spacing including 4.6mm fingertip body each side) |
| `/gripper_width_offset` | Jaw-tip-to-jaw-tip gap (`raw - 2 × 4.6mm = raw - 9.2mm`) |
| `/gripper_status` "Width:" field | Same as `/gripper_width` (raw) |
| `control_gripper.py` "Final width" | Reads `/gripper_width_offset` (jaw gap) |

**For grasping**, `gripper_width_offset` is what you care about — it's the actual gap between fingertip surfaces.

### 8.3 Two-track convention reminder

- `[N]` no-robot tasks (Phase 4 dashboard, Phase 6 dispatcher code)
- `[R]` at-robot tasks (Phase 3 collection, validation runs)
- See `.planning/TRACKS.md` for live ready/blocked lists per track.

### 8.4 Where the "canonical sequence" comes from

`ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json` lists the exact tool sequence per assembly step (per object × per base). Every entry has the same shape:
1. set gripper width
2. move_to_grasp
3. close
4. safe_height
5. rotate_object
6. translate_object --place-down
7. set gripper width
8. safe_height
9. move_to_grasp (regrasp)
10. close
11. safe_height
12. rotate_object (second)
13. **translate_object --insert** ← `run_assembly_step.py` substitutes `compliant_insert wrapper` here
14. set gripper width (release)
15. safe_height

This 15-step sequence is what the orchestrator implements (steps 14-15 are handled inside the wrapper's DONE path).

> **CORRECTED 2026-08-16.** The sequence is **not** uniform across parts. Each object is **8 or
> 15 steps** — `line_green` **skips the regrasp** (steps 7–11), going straight from the first
> safe-height to the second rotate and the insert. Read the per-object entry in
> `Assembly_fmb_assembly_1_results.json` rather than assuming 15 steps, and note that order
> matters across objects because the parts nest.

---

## 9. What's next

**Phase 3 collection closed in May 2026.** For current state, open work and the running order of
the full assembly, read `docs/HANDOFF_NEXT_AGENT.md` — that is the live handoff. Open fixes from
the 2026-08-16 session are in `docs/QUEUED_FIXES.md` (`ur5e-fmb1-demo` only).

> *Historical — the Phase 3 starting line, 2026-05-03:*
> 1. Run §3 to bring up the stack
> 2. Run §4.3 to confirm grasp_ids for the 3 unverified objects
> 3. Place each FMB1 object in workspace one at a time, run §4.1 per object
> 4. Manually guide part during ACTIVE on at least 1 demo per object (DATA-02 quota)
> 5. After each demo: confirm CSV+meta JSON written to `compliant_insertion_studio/logs/`
> 6. After all 20 demos: run §5 cleanup
> 7. Phase 4 dashboard work can proceed at-away on the collected data

# Running FMB1 and FMB3 on the UR5e + RG2 cell

A guide for a person who has not run this cell before. It covers what to install, how to bring the cell up, how to
assemble FMB1 and FMB3, how to stop safely, and what goes wrong. Written 2026-11-16 from branch `dev`.

`CLAUDE.md` (repo root) is the denser reference the agents work from. Where this guide is short, read it there.

---

## 0. Where things stand

| assembly | parts | status |
|---|---|---|
| **FMB1** (base1) | u_brown, u_orange, line_green, inverted_u_yellow | Routine. 16 hands-off certified cycles in a row (2026-10-24/25, runs 93-108), each about 12 min. Tipped-part starts (phase 2) were being worked on when the camera mount broke. |
| **FMB3** (base3) | line_brown, line_red, fork_orange, fork_yellow | Assembled 4/4 on 2026-10-20 after a day of fixes; **not run since**, and a lot of shared code has changed. Treat the first FMB3 run as a test: one part at a time (§6). |

**The camera.** The D415 moved to a new 3D-printed mount (v3.1) on 2026-11-10. The calibration in the localizer config
(`robot_config.yaml`, §5) is for the OLD mount. Until a hand-eye calibration is run for the new mount, every pose the
camera gives is off by centimetres. See §5 before any pick.

---

## 1. The cell

| what | where | notes |
|---|---|---|
| UR5e | `192.168.1.111`, PolyScope 5.17.3 | Driven headless by the ROS 2 driver. The pendant must be in **Remote Control** (top right in PolyScope). The arm rides a Vention 7th-axis slider (not modelled; only read its state). |
| OnRobot RG2 | UR tool connector (RS-485 through the driver's tool communication) | Bridge: `onrobot_ros` (`/gripper_command`, width / grasp / status topics). Fixed 40 N grip. |
| Intel RealSense D415 | **USB 3 port at 5000 Mbps** on the PC | On USB 2 the localizer hangs with no error. Check: `cat /sys/bus/usb/devices/*/speed`. |
| Cell PC | Ubuntu 22.04, ROS 2 Humble, `192.168.1.97` on the robot subnet | A slow PC (i5-8500T): the localizer runs at about 3 Hz, and RViz during a run has dropped the robot connection. Don't run RViz or heavy GUIs during runs. |

Frames: everything in the code is in the UR **`base`** frame (the DH base). The workspace is at **-y**, the table top
is at z = -0.080 m (the robot base sits 80 mm above the table).

---

## 2. Install (once)

### 2.1 System packages

```bash
sudo apt install ros-humble-ur-robot-driver ros-humble-realsense2-camera ros-humble-cv-bridge ros-humble-tf2-ros socat
pip3 install --user opencv-python==4.11.0.86     # the localizer needs cv2.aruco.ArucoDetector (OpenCV >= 4.7); apt's 4.5 lacks it
```

Versions the cell runs: ur-robot-driver 2.14.0, realsense2-camera 4.58.3, numpy 1.21 / scipy 1.8 / PyYAML 5.4 (Ubuntu's).

### 2.2 The ROS 2 workspace

Four packages, built together. The scripts look for the workspace at `$FMB_ROS_WS` (default `~/Desktop/ros2_ws`).

```bash
export FMB_ROS_WS=~/Desktop/ros2_ws            # or wherever you want it; put this in ~/.bashrc
mkdir -p $FMB_ROS_WS/src && cd $FMB_ROS_WS/src
git clone git@github.com:inbarajaldrin/aruco_camera_localizer.git   # the marker + depth localizer (main >= c4e9836)
git clone git@github.com:inbarajaldrin/max_camera_msgs.git
git clone -b dev git@github.com:inbarajaldrin/ros-mcp-server.git ~/ros-mcp-server   # THIS repo (dev branch, never main)
cp -r ~/ros-mcp-server/ros2_pkgs/onrobot_ros ~/ros-mcp-server/ros2_pkgs/pa_msgs .
cd $FMB_ROS_WS && source /opt/ros/humble/setup.bash && colcon build --symlink-install
```

`ros2_pkgs/README.md` says why the gripper bridge and the message packages are copied here (local edits that exist
nowhere else). **Use branch `dev`**: `main` cannot run the real assembly (its grasp-point ids do not match the assembly
files; every pick fails with "Grasp point 1 not found").

### 2.3 What lives in this repo

| path | what |
|---|---|
| `assets/cad_data/` | every part's mesh (`models/`), ArUco layout (`aruco/`, with printable marker sheets in `aruco/pdfs/`), grasp points, symmetry, and the assembly files (`fmb_assembly1*.json`, `fmb_assembly3*.json`). `$FMB_CAD_DATA` overrides the location. |
| `primitives/` | robot primitives, run as modules: `python3 -m primitives.move_home --mode real` (never `python3 primitives/x.py`) |
| `ablations/replay_real_assembly.py` | the assembly runner: picks, reorients, inserts and checks each part from `ablations/ground_truth_resources/Assembly_fmb_assembly_N_results.json` |
| `ablations/disassemble.py` | takes parts out of the base and lays them out in the staging region (resets a cycle) |
| `compliant_insertion_studio/` | the insertion (force-mode search), bring-up / stop scripts, docs |
| `seams/` | the ROS action servers the FMB1 runs go through (drive-mate insertion, execute-mate certification, vision, grip) |
| `primitives/shared/cell_layout.yaml` | where the base, the clear area (regrasp spot) and the staging region are on the table, per layout |

---

## 3. Bring the cell up (each session, about 1 minute, no arm motion)

1. **Robot**: pendant booted, robot powered with brakes released, **Remote Control** on. Check:
   ```bash
   cd ~/ros-mcp-server
   python3 compliant_insertion_studio/scripts/ur_dashboard.py --host 192.168.1.111 --real mode safety remote
   # want: Robotmode: RUNNING | Safetystatus: NORMAL | remote: true
   # not RUNNING: ... --real power_up
   ```
2. **Everything else in one command** (driver -> gripper -> camera -> grasp points, each verified):
   ```bash
   bash compliant_insertion_studio/scripts/stack_up.sh
   ```
   Read its last block. "camera sees only N frames" just means the camera is not looking at parts from where the arm is.
3. **The seam servers** (FMB1 runs use them):
   ```bash
   source /opt/ros/humble/setup.bash && source $FMB_ROS_WS/install/setup.bash
   for s in vision_server grip_adapter drive_mate_server execute_mate_server; do
     nohup python3 -u -m seams.$s > /tmp/$s.log 2>&1 &
   done
   ```
   They don't read your shell's environment after start: restart them if you change `FMB_LAYOUT` or other settings.
4. **Look at the camera** before moving anything (it is the fastest way to catch a problem):
   ```bash
   python3 compliant_insertion_studio/scripts/mjpeg_view.py --bind 127.0.0.1 --port 8080 --fps 3 &
   # open http://127.0.0.1:8080 : the annotated stream (markers, part outlines, +-x/+-y compass), the depth view and raw colour
   ```
   Lights off? Set a long manual exposure (`ros2 param set /camera/camera rgb_camera.enable_auto_exposure false`,
   `... rgb_camera.exposure 8000`, `... rgb_camera.gain 128`). Lights back on: set auto exposure `true` and gain `64`
   again, or every frame is white.

Down again: `bash compliant_insertion_studio/scripts/stack_down.sh`. Logging out of the desktop kills the whole stack.

---

## 4. Safety: read before moving the robot

- **Stopping a run: `bash compliant_insertion_studio/scripts/stop_run.sh`.** It sends the UR dashboard `stop` first. A
  trajectory the controller already accepted keeps running after its client is killed, so killing processes does NOT
  stop the arm. If the dashboard does not answer and the arm is moving: the pendant / e-stop.
- **Protective stop** (a contact): `bash compliant_insertion_studio/scripts/recover_pstop.sh`, then look at why.
- **Safety FAULT** (power off): close the safety popup and restart safety (`ur_dashboard.py ... close_safety_popup
  restart_safety`), `power_up`, then re-run `stack_up.sh`. Look at the cell first: on 2026-10-25 a fault was the camera
  cable caught by a wrist turn.
- **Gripper ignores commands** with `Circuit1:True` / `Circuit2:True` on `/gripper_status`: an arm power cycle from the
  dashboard clears it (`ur_dashboard.py ... raw "power off"`, wait 10 s, `power_up`, then `stack_up.sh`).
  `/restart_gripper_connection` does not clear it.
- Forces: the insertion stays at or below 5 N by default (9 N with an explicit override), 6 N sideways.
- **wrist_3 stays within +-4 rad** (the camera cable wraps the wrist); the motion code refuses beyond it.
- With a part in the gripper, go to safe height (`python3 -m primitives.move_to_safe_height --mode real`) before
  `move_home`: home plans a straight joint move that ignores parts on the table.
- Never resume a run while the gripper still holds a part: set it down first (§7).
- Don't kill a primitive with `kill -9` while it is in force mode: it must reach its clean exit. Use `stop_run.sh`.

---

## 5. The camera calibration (hand-eye)

The camera pose in the gripper frame is in the localizer config,
`$FMB_ROS_WS/src/aruco_camera_localizer/config/robot_config.yaml` (`cam_offset_position`, `cam_offset_quat`; restart the
localizer after an edit). The localizer, the depth code and the "keep the camera up" wrist logic all read it.

- **Old mount (until 2026-11-10)**: the values in the file, from a 24-view survey on base1 (2026-10-20). The camera sat
  on the tool's -Y side looking straight down.
- **New mount v3.1 (since 2026-11-10)**: from CAD, not yet calibrated, tool0 -> colour optical frame:
  position (0, 0.0889, 0.0839) m, quaternion xyzw (0.258819, 0, 0, 0.965926). The camera is now on the **+Y** side,
  tilted 30 degrees toward the fingertips. Expect 1-2 mm and about 1 degree of error from printing and seating.

To calibrate: put the CAD values in the file as a start, then run the base1 survey (`ablations/handeye_icp_survey.py`
collects views of base1 and fits its CAD to depth; `ablations/handeye_from_survey.py <survey.json>` solves the mount)
and write the solved values. With the new tilt, the view poses that assume a straight-down camera (part views, the base
view at 0.56 m) also need checking against real frames before the first run.

---

## 6. Assemble

All commands from the repo root with ROS and the workspace sourced. **The runner returns 1 on any failure: check it and
stop; never chain the next command blindly.** Each part is pick -> reorient -> (set down at the clear area and regrasp,
if the part lies on the wrong face) -> insert -> check the seat -> release.

### 6.1 FMB1

Table: base1 in the assembly region, the four parts in the staging region, the clear area (taped square around UR
(-0.32, -0.50)) empty. Layout `fmb1_home_2026_10_21` (`primitives/shared/cell_layout.yaml`).

The settings every hands-off run used:

```bash
export FMB_LAYOUT=fmb1_home_2026_10_21 BASE_FROM=icp NO_AIM_BIAS=1 PRE_INSERT_READ=off FOLLOW_YAW=0
J=ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json

# the whole assembly (startup = home + gripper open + look at every part, then the four parts in order)
SEAMS_DRIVE=1 SEAMS_CERTIFY=1 python3 -u ablations/replay_real_assembly.py --assembly-json $J; echo "EXIT $?"

# or one part at a time (order matters: u_brown, u_orange, line_green, inverted_u_yellow)
SEAMS_DRIVE=1 SEAMS_CERTIFY=1 python3 -u ablations/replay_real_assembly.py --assembly-json $J --only u_brown
SEAMS_DRIVE=1 SEAMS_CERTIFY=1 python3 -u ablations/replay_real_assembly.py --assembly-json $J --only u_orange --skip-startup
```

Resume after a stop with earlier parts already seated: `--start-from <part>`.

Take it apart again (pull every part, lay them out in staging):

```bash
python3 -u ablations/disassemble.py --assembly-json $J --to-free-spot
```

`ablations/assembly_cycle.sh` runs reset + assembly as one recorded cycle.

### 6.2 FMB3

Not run since 2026-10-20: go one part at a time and watch the camera. The seams are FMB1-only (`seams/bindings.yaml`
binds only FMB1), so FMB3 runs **without** `SEAMS_DRIVE` / `SEAMS_CERTIFY`.

1. Offline check (no robot): `python3 ablations/onboard_check.py fmb3` must print READY.
2. Table: base3 where the layout says, its parts in staging. Layout `fmb3_in_clear_region_2026_10_20` puts base3's
   assembly region at UR (-0.389, -0.471) and the clear area at (-0.300, -0.270). If the table is arranged differently,
   **add a new layout** to `cell_layout.yaml` (copy one, edit the numbers, select it with `FMB_LAYOUT`); never edit a
   layout an earlier run used.
3. Run:
   ```bash
   export FMB_LAYOUT=fmb3_in_clear_region_2026_10_20 BASE_FROM=icp NO_AIM_BIAS=1
   J=ablations/ground_truth_resources/Assembly_fmb_assembly_3_results.json
   python3 -u ablations/replay_real_assembly.py --assembly-json $J --only line_brown; echo "EXIT $?"
   ```
   then line_red, fork_orange and fork_yellow in the order of the assembly file.

On base3 a 4-dof flat base fit is wrong (base3 sits about 1 degree pitched on the table); `BASE_FROM=icp` (a 6-dof fit of
the base CAD to depth) is the one that worked. The forks and bars use the prismatic insert; a bar seated high at one end
is fixed with taps, not pushes (§8).

### 6.3 Is it seated?

Don't trust the insertion's own "success" label. The run checks each seat with the depth camera right after the insert
(`[seat-check]` lines) and, for FMB1, `execute_mate` certifies it (`placed=True` in the log;
`ablations/logs/mate_journal.jsonl`). A good seat lands within +-1.7 mm of the predicted height. The raw force/position
trace of each insert is in `compliant_insertion_studio/logs/insert_<part>_<time>.csv`.

---

## 7. When a run stops part-way

- **Part left at the clear area** (e.g. "Object not detected" on the regrasp):
  ```bash
  python3 -u -m primitives.control_gripper open --mode real
  python3 -u -m primitives.core.move_to_clear_area --mode real          # arm top-down; --fix-orientation if it is not
  python3 -u ablations/replay_real_assembly.py --assembly-json $J --only <part> --skip-startup
  ```
- **Part still in the gripper**: `python3 -u ablations/disassemble.py --assembly-json $J --place-held <part>` sets it down
  in staging; then run that part normally.
- **Two parts touching / a part lying on another**: the run refuses to pick from a crowded spot. Move the part with
  `disassemble.py --relocate <part> --grasp-id auto`, or by hand.
- Debugging one misbehaving part: `compliant_insertion_studio/docs/DEBUG_LOOP.md` (iterate on that part alone, read the
  raw CSVs before changing anything).

---

## 8. Things that have bitten us

| symptom | cause | what to do |
|---|---|---|
| poses or picks off by 5-10 mm | the camera sampled while the arm was moving (D415 frames during motion are about 10 mm off) | the code takes poses only from a still arm; don't shorten its settle times |
| a finger lands on a part (high force, protective stop) | the part's pose came from a weak measurement | the pick uses a CAD fit to depth; if it can't trust one it stops instead of guessing. Look at the frame, fix the scene. |
| line_green seats 2-8 mm high at one end | it rests on a wedge of base1 | the run detects it and taps it in from the side, rechecking each time |
| base1 moves during a run | it is not fixed to the table; hard searches drag it | the run re-reads the base before every insert; on a free base the search force is 1.5 N |
| the robot program drops ("not running" mid-run) | reverse-interface drop | `stop_run.sh` restores it (stop, re-send program, controllers) |
| the arm swings wide on a small move | IK from a fixed seed picks another configuration | use Cartesian moves (`move_to_safe_height --lift-by`), not `primitives.core.move_xyz`, near parts |
| camera frame all white / all black | exposure set for the other lighting | §3 step 4 |
| localizer publishes nothing, no error | camera on a USB 2 port | move it to a 5000 Mbps port |

---

## 9. Shut down

```bash
bash compliant_insertion_studio/scripts/stop_run.sh        # if anything is running
bash compliant_insertion_studio/scripts/stack_down.sh
python3 compliant_insertion_studio/scripts/ur_dashboard.py --host 192.168.1.111 --real raw "power off"   # optional
```

Leave the pendant in Local mode if someone else will use it from the pendant.

## 10. Where to read more

| need | doc |
|---|---|
| the full reference (bring-up, constraints, anti-patterns) | `CLAUDE.md` |
| cold start and troubleshooting | `compliant_insertion_studio/docs/SETUP.md` |
| a new assembly (FMB2, a new base) | `compliant_insertion_studio/.claude/skills/onboard-assembly/SKILL.md` |
| the insertion's architecture | `compliant_insertion_studio/docs/AUTONOMOUS_INSERTION_METHODOLOGY.md`, `ARCHITECTURE_SEAMS.md` |
| debugging one part | `compliant_insertion_studio/docs/DEBUG_LOOP.md` |
| open problems and queued fixes | `compliant_insertion_studio/docs/HANDOFF_NEXT_AGENT.md`, `docs/QUEUED_FIXES.md` |

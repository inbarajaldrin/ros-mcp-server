# INTERFACE - what a planner on the real FMB1 cell calls (verified 2026-10-24)

Checked against the code in `repo/seams/`, the interface files in `~/Desktop/ros2_ws/src/pa_msgs/`, and the live graph
(`ros2 action list -t`, `ros2 service list -t`, `ros2 topic list`, read-only, 2026-10-24 ~09:00). `repo` =
`/home/aldrin/Documents/ros-mcp-server-verified`. The briefing `dimelab_env_elevated/INSTRUCTIONS_FOR_PLAN_ANYTHING_AGENT.md`
§4-6 was the starting map; every place it is wrong or stale is marked **CORRECTION**.

All names below are what the cell serves TODAY. The twin-surface names in `declaration.draft.yaml` (`/ur5e_0/...`,
`/objects/poses` in twin `world`) are the adapter's target and are not built (OWED.md).

Frames: every pose on this surface is in UR `base` (the workspace is at -y) or in `tool0` (the flange). Twin world = `(-x, -y, z + 0.080)`, yaw + 180 deg. Quaternions are xyzw.

---

## 0. What is live

| surface | name | type | server (repo) | live 2026-10-24 |
|---|---|---|---|---|
| arm motion | `/scaled_joint_trajectory_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | UR driver | yes (controller active) |
| arm state | `/joint_states` | `sensor_msgs/msg/JointState` | UR driver | yes, 474 Hz |
| flange pose | `/tcp_pose_broadcaster/pose` | `geometry_msgs/msg/PoseStamped` (frame `base`) | UR driver | yes, 471 Hz |
| wrench | `/force_torque_sensor_broadcaster/wrench` | `geometry_msgs/msg/WrenchStamped` (frame `tool0_controller`) | UR driver | yes, 475 Hz |
| grip | `/ur5e/rg2/grip` | `pa_gripper_msgs/action/Grip` | `seams/grip_adapter.py` | yes |
| grip events | `/ur5e/rg2/grip_events` | `std_msgs/msg/String` (json) | `seams/grip_adapter.py:56` | yes |
| insertion | `/ur5e/drive_mate` | `fmb_seam_msgs/action/DriveMate` | `seams/drive_mate_server.py` | yes |
| certification | `/ur5e/execute_mate` | `pa_assembly_msgs/action/ExecuteMate` | `seams/execute_mate_server.py` | yes |
| one pose, with sigma | `/vision/get_pose` | `fmb_seam_msgs/srv/GetVisionPose` | `seams/vision_server.py` | yes |
| seat measurement | `/vision/certify_seat` | `fmb_seam_msgs/srv/CertifySeat` | `seams/vision_server.py` | yes |
| object stream | `/objects/poses` (relay of `/objects_poses_real`) | `tf2_msgs/msg/TFMessage` | `seams/vision_server.py:52-54` | yes, 2.6 Hz |
| gripper state | `/gripper_width_offset`, `/gripper_width`, `/gripper_grasp_detected`, `/gripper_motion_ongoing`, `/gripper_status` | Float32 / Bool / String | onrobot_ros `rg_gripper` | yes, 10 Hz |
| clock | none | - | - | no `/clock`: wall time |

Packages (`~/Desktop/ros2_ws/src/pa_msgs/`, all v0.1.0): `fmb_seam_msgs` (ours), `pa_assembly_msgs`, `pa_gripper_msgs`
(copied from prompt2assembly, per `repo/compliant_insertion_studio/docs/ARCHITECTURE_SEAMS.md:77-78`; the copy's
byte-equality with prompt2assembly was not re-checked here).

Not exposed to planners: the UR dashboard (`/dashboard_client/*`), `/io_and_status_controller/*`, the force-mode and
other inactive controllers. Stopping a run is `bash repo/compliant_insertion_studio/scripts/stop_run.sh` (dashboard `stop`
first: an accepted trajectory keeps running after its client dies, `repo/CLAUDE.md:165-171`).

---

## 1. Arm: FollowJointTrajectory and `/joint_states`

- Action `/scaled_joint_trajectory_controller/follow_joint_trajectory` (`repo/primitives/core/move_joints.py:75`).
  Speed scaling currently 100 % (`/speed_scaling_state_broadcaster/speed_scaling`).
- Joint names are **unprefixed** (`shoulder_pan_joint`, ...) and arrive on `/joint_states` in the order
  `shoulder_pan, wrist_2, wrist_3, wrist_1, elbow, shoulder_lift` (live) - read by name, never by index.
  The twin uses `ur5e_0_`-prefixed names on `/ur5e_0/joint_states` (briefing §4.1; not re-verified here).
- The cell's own motion code caps joint speed at 0.80 rad/s and Cartesian speed at 0.0798 m/s with a 2.5 s minimum
  duration (`repo/primitives/shared/config.py:317-322`). These are the cell's policy, not the controller's limits.
- Rules the cell learned the hard way (`repo/CLAUDE.md:207-257`):
  - keep `|wrist_3| <= 4 rad` (camera and gripper cables; `repo/primitives/shared/ik.py:47`);
  - small moves near parts as Cartesian waypoints seeded from the current joints, never a fixed-seed IK solution;
  - never hard-cancel a trajectory (full-speed stop);
  - with a part in hand, go to safe height (table + 0.40 m = UR z 0.320) before a long move.
- Home: `[1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]` (UR joint order; `config.py:341`), flange at
  (0.0650, -0.3850, 0.4810) UR `base` (live).
- **Any real motion needs the operator's go-ahead** (`repo/CLAUDE.md` §6).

## 2. Grip: `pa_gripper_msgs/action/Grip` at `/ur5e/rg2/grip`

Goal (`Grip.action`): `float64 width_m`, `float64 force`, `string expect`, `float64 expect_span_m`.

| field | real-cell meaning (`seams/grip_adapter.py`) |
|---|---|
| `width_m` | tip-to-tip gap target, m. Refused (`verdict refused`, goal aborted) outside `[0, 0.105]` (:66-67). `>= 0.099` sends `open` (full open, reads 100.6 mm), `<= 0.001` sends `close`, else a width in mm (:68). |
| `force` | **recorded, not applied**. The RG2 always closes at 40 N (driver default 400 x 0.1 N; :16-20, :46). |
| `expect`, `expect_span_m` | `expect_span_m > 0` -> identity by `width_vs_span`: `|measured gap - span| <= 3 mm` (:83-85) |

Result (`Grip.Result`, filled at :63-89):

| field | real cell |
|---|---|
| `reached` | `|gap - width_m| <= 3 mm` (`REACH_TOL_M`, :43 - the RG2 has no precise positioning mode) |
| `holding` | only after a CLOSING move: the RG2 grip flag, or stalled > 2 mm short of the command (:82) |
| `width_measured_m` | `/gripper_width_offset` / 1000, read 0.3 s after the move (:72-74) |
| `width_commanded_m` | the goal width |
| `theta_measured_rad`, `effort_measured_nm`, `attach_drift_mm` | NaN (onrobot_ros reports no angle or effort) |
| `stall_margin_m` | `width_measured - width_commanded` |
| `attach_applied`, `attached_body`, `attach_refusal` | false / "" (no attach on a real cell) |
| `expected_part` / `discovered_part` | goal `expect` / "" |
| `identity_verified`, `identity_authority` | see above / `width_vs_span` or `none` |
| `verdict` | `holding`, `reached`, `refused` or `unreachable`. Never `slip`. |

Feedback: **none is published** (the adapter calls `gh.succeed()` with no `publish_feedback`). One json event per
result on `/ur5e/rg2/grip_events` (:90-95). Width semantics: `/gripper_width_offset` = plate spacing - 2 x 4.6 mm
(prompt2assembly's `fingertip_offset_mm`); `/gripper_width` reads 9.2 mm more (live 97.4 vs 88.2). The RG2 has **no
joint state** on this cell (`/rg2/joint_states` has 0 publishers, live). The grip flag is also true when the fingers
are closed on nothing (`execute_mate_server.py:118-120`): judge "holding" by flag **and** gap > 2 mm.

Measured widths: open 100.6 mm; closed empty 0.5-0.6 mm; closed on u_brown/u_orange 21.0-21.5, line_green 25.9-26.1,
inverted_u_yellow 41.7-42.9 mm (MEASURED_CELL `gripper.*`). Flange to the closed fingertip ends: 230.86 mm; pad centre
14.9 mm above the tips (`config.py:12-16`).

## 3. Perception

### 3.1 `/objects/poses` (and `/objects_poses_real`)

`tf2_msgs/msg/TFMessage`, one message per processed camera frame with every object in view. Verified live:
- `header.frame_id` = `"World"`, but the values are in **UR `base`** (`localizer_bridge.py:183`; live base1 at
  (0.135, -0.466, -0.0625)).
- children are cell names: `base1`, `u_brown`, `u_orange`, `line_green`, `inverted_u_yellow`.
- only objects in the wrist camera's view; ~2.6 Hz with the arm still (live); **nothing** while the TCP moves faster
  than 15 mm/s (`filter_config.yaml:45-47`).
- **z is snapped**, not measured: parts to the table-rest height of their resting face when within 10 mm, the base to
  table + 17.5 mm, base yaw-only (`filter_config.yaml:19-29`). Live 2026-10-24: u_brown and u_orange, seated in base1,
  were published at z -0.045 (the table-rest value). Orientation is snapped to one branch of the part's symmetry group.
- object frames are not on `/tf`; the camera is not on the robot's TF tree.
- The cell's TF `world` is `base_link` (identity), not the table plane (live `/tf_static`).

Use it to find candidates. Use `/vision/get_pose` for anything a motion is committed on.

### 3.2 `/vision/get_pose` - `fmb_seam_msgs/srv/GetVisionPose`

Request: `string name`, `string kind` (`base` | `part` | `held`), `float64 timeout_s` (<= 0 -> 15 s).
Response: `bool ok`, `string error`, `fmb_seam_msgs/VisionPose pose`.

**No motion**: the answer is read from wherever the camera is; the planner flies the view first. One request at a time
(mutually exclusive callback group, `vision_server.py:49`). Each request runs `seams.vision_ops` in a subprocess.

| kind | what it does (`seams/vision_ops.py`) |
|---|---|
| `base` | 6-dof CAD fit to aligned depth (`base_icp.read_icp`), seeded by the last saved base read if it is < 900 s old (`repo/ablations/logs/last_base_pose.json`), else by markers (:77-104). Needs the whole base in view. `source "depth_cad"`, `n 0`, sigma `[0.0018, 0.0018, NaN, NaN]` (:105-108: z and yaw repeatability not measured). |
| `part` | 5 still-camera marker samples (a sample counts only if the TCP was still for 0.8 s before it), >= 3 required; median position, mean quaternion; sigma = the samples' std in x, y, z and yaw (:110-124). `source "markers"`. |
| `held` | **NOT SERVED.** Returns `ok false, "held: not served yet - the planner takes the held read (seam open)"` (:74-76). |

**CORRECTION (briefing §4.3 step 2, and the `GetVisionPose.srv` comment):** there is no `get_pose(part, held)` read.
The reference planner builds the held pose itself from the chain - grasp id, the grasp depth `move_to_grasp` executed,
and the part's world orientation from the previous primitive - with `seams.drive_mate_server.held_pose()`
(`repo/ablations/replay_real_assembly.py:1355-1386`), because the gripper occludes a held part (`repo/CLAUDE.md:212-213`).

### 3.3 `VisionPose` (`fmb_seam_msgs/msg/VisionPose.msg`) as `/vision/get_pose` fills it

| field | .msg says | what the code puts there (`vision_ops.py:126-134`, `vision_server.py:61-71`) |
|---|---|---|
| `header.frame_id` | frame of the pose (`base` = UR base) | `"base"` always |
| `header.stamp` | when the camera saw it (frame arrival time) | **CORRECTION:** `time.time()` when the read FINISHED, not a frame time |
| `name` | as the assembly graph names it | the cell name you asked for (`base1`, `u_brown`, ...) - not `occurrence:0N` |
| `pose` | | UR `base`, xyzw. Part z is the localizer's snapped rest height. Base z = table + 17.5 mm. |
| `sigma[4]` | x, y, z (m), yaw (rad), 1 sd, from repeatability | base `[0.0018, 0.0018, NaN, NaN]`; part = the samples' spread (repeatability only, not bias: hand-eye roll ~+-2 mm, ~2 mm -y bias) |
| `stationary_since` | camera still from this time on | **CORRECTION:** `request time - 0.8 s` - a lower bound from the acceptance rule, not the measured start of stillness |
| `source` | markers / depth_cad / markers+depth_cad / chained | `markers` (part) or `depth_cad` (base); the other two are never produced |
| `view_z` | flange height of the view | the TCP z at the end of the read |
| `detection_set` | which markers / surfaces | always `""` |
| `n` | samples or points | number of still samples (part); 0 for base |
| `symmetry` | symmetry group | `<name>_symmetry` if that file exists, else `""` |

### 3.4 `/vision/certify_seat` - `fmb_seam_msgs/srv/CertifySeat`

Request `string mate_id` (`fmb1/mate:NN`; bare `mate:NN` accepted while one assembly is bound). Response:
`bool inside`, `float64[3] error_mate_frame_m`, `float64 tilt_deg`, `string method`, `string evidence_json`
(`vision_server.py:75-84`, `vision_ops.py:201-307`).

- One base read (depth CAD fit; on failure the last saved read) and one still depth capture.
- z: the depth seat check's height error against the **graph's** seat (our CAD seat minus `graph_seat_z_mm`
  difference, :238-245), judged within half-width 1 mm + 2 x 0.5 mm = **2.0 mm** (:267), centre and both end heights
  where the part can wedge; end heights from a fresh tap map when nothing has touched the part since (:272-278).
- x, y: marker position error, **reported, not judged** (no stated sigma). Tilt: reported only.
- `method`: `both` (depth + markers), `depth_seat_check`, `markers` or `none` (**CORRECTION:** the .srv comment omits
  `none`). `inside` is false unless the height was measured.
- No motion: fly a seat view first (`seat_check.view_pose`). An end reading more than the band BELOW the seat is
  rejected as a view artefact (:279-283).

## 4. Insertion: `fmb_seam_msgs/action/DriveMate` at `/ur5e/drive_mate`

Goal (`DriveMate.action`):

| field | real cell (`seams/drive_mate_server.py`) |
|---|---|
| `mate_id` | `fmb1/mate:NN`, resolved through `repo/seams/bindings.yaml` to mover + reference (:75-87). Bare `mate:NN` accepted while one assembly is bound. **CORRECTION:** the .action's example `"fmb_assembly_1/u_brown"` is stale. |
| `target` | `VisionPose` of the REFERENCE (base1) in frame `base`; any other frame -> refusal `target_frame` (:230-231). Send the **measured** base, not an aim: corrections ride separately. Its `sigma[0:2]` drives the sigma gate (:162-184); 0 or NaN = "not stated" -> bindings' 1.8 mm. (The CLI client sends zeros, `seams/client.py:39-43`.) |
| `held` | `VisionPose` of the mover in **`tool0`**; other frames -> `held_frame` (:232-233). The CAD grasp point within 6 mm of the tool axis is the grasp; its tool-z is the grasp depth (:61, :105-114); none -> `held_unmatched`. Build it with `held_pose(obj, grasp_id, held_quat_world, tcp_quat_world, grasp_depth_m)` (:96-102). |
| `corrections_json` | json list of `{name, dxy_m, dyaw_rad, about, dxy_base_m}` (`seams/aim.py:274-278`): names `follow`, `bias`, `yaw_exp`; applied by the server on the base its own alignment arrives at (`aim.py:281-300`). Bad json -> refusal `corrections`. |

What it does: on a free base (`BASE_FIXED` false) and if the sigma gate says the target's xy sigma exceeds the part's
CAD clearance, it first runs `seams.drive_align` (it **moves the arm** over the seat and may re-measure the base and the
grasp depth), then one insert attempt through `primitives.translate_object --insert` (routes by part name: U parts
compliant spiral/touch, lines prismatic - recorded debt, :29-32), seat z from learned certified inserts or CAD
(:252-260). One goal at a time: a second goal while busy is **rejected** (:143-144).

Result:

| field | real cell |
|---|---|
| `seated` | the arm at rest (TCP still < 0.5 mm over 0.5 s) and `|tcp_z - predicted seat z| < 5 mm` (`insert_outcome.py:32`, server :62, :126-141). Never the FSM label. |
| `stage_reached` | `seat`, `wedged` (still, 0 < dz <= 7 mm, within 5 mm in xy), `above`, `below`, `unsettled`, `no_trace` (`insert_outcome.py:15`). **CORRECTION:** the .action comment (`SEAT | BASE_RIM | OBJ:<part> | WEDGED | NONE`) is stale; the briefing's list is right. |
| `slide[2]` | end TCP xy - predicted seat xy (m, base) |
| `measured_correction[3]` | end TCP - prediction (m, base) |
| `trace_path` | the insert's raw CSV |
| `refusal` | `mate_unknown`, `assembly_unbound`, `assembly_ambiguous`, `target_frame`, `held_frame`, `held_unmatched`, `insert_failed_to_start`, **and** `robot_stopped` (:277, the align step saw a safety stop) and `corrections` (:293). **CORRECTION:** the briefing and the server docstring omit the last two. |
| `base_measured` | `VisionPose` (frame `base`, sigma NaN, `source "drive_align: <route>"`): the base the alignment measured - aim the NEXT insert from it (:297) |
| `align_json` | route, measured base, log, aim, corrections applied (:298-299) |

Feedback: `stage` (`aligning` or `inserting`), `z_above_seat` = **NaN always** (:198-202, :328-332).
Cancel: SIGINT to the insert's process group, SIGTERM after 30 s, never SIGKILL (:146-160); the client cancels on
Ctrl-C (`client.py:59-63`).
**After it returns the part is still gripped at the seat** (no gripper command in the server; `translate_object` passes
`--no-post-insert-move`, `primitives/translate_object.py:479-484`; a wedged bar is held where it stopped,
`PRISMATIC_HOLD_IF_WEDGED`, :317). The planner releases (Grip open) and retreats.

## 5. Certification: `pa_assembly_msgs/action/ExecuteMate` at `/ur5e/execute_mate`

Goal: `string mate`. Owns no motion. Five checks in order, each able only to refuse (`execute_mate_server.py:1-25,105-158`):

1. resolves through bindings: `mate_unknown` / `assembly_unbound` / `assembly_ambiguous`;
2. every earlier mate of the order certified in this episode (journal): `order_violated`;
3. the RG2 does not hold a part (flag AND gap > 2 mm): else `not_at_rest`;
4. every predecessor still passes NOW (one shared capture): else `predecessor_unseated`; a predecessor not measurable
   from this view (line_green under yellow) is carried on its journalled certification (recorded deviation);
5. this mate's seat: `engagement_invalid` (height not measured from this view, or vision did not answer) or
   `not_at_seat`.

**CORRECTION (order):** the code checks the grip (3) BEFORE re-reading the predecessors (4); the docstring lists them
the other way round. Success: `placed true, fastened false, refusal ""`, `mate_class insertion`, `fastening_class free`,
`ee_class parallel_jaw`, `order_position` = index in `bindings.yaml` order, `engagement_depth_m` = the z error,
`granted_on` = `both` / `depth_seat_check`, `engagement_json`, `next_move` (one sentence). Fields never set here:
`slip`, `coexists_grip`, `coexists_seat`, `weld_prim` (""), and no feedback is published. Journal: one line per call in
`repo/ablations/logs/mate_journal.jsonl`; certifying `mate:00` starts a new episode. **There is no
`/objects/fastening_events` topic** (the twin's journal stream). Fly a seat view first; the reference planner tries
views until one measures (`replay_real_assembly.py:1512-1569`).

## 6. The per-mate sequence the reference planner uses (verified)

`repo/ablations/replay_real_assembly.py` with `SEAMS_DRIVE=1`, `SEAMS_CERTIFY=1`; client CLI `python3 -m seams.client`
(`seams/client.py:1-12`).

0. Once, at start: home, gripper open, observe every table part and the base; plan each part's passes
   (`[derive-sequence]`: every logged start was 1 pass).
1. Base: fly the base view (flange z 0.56 m), read base1 (depth CAD fit). Re-read before every insert: it moves.
2. Pick: fly over the part, `get_pose(part)` (still samples), choose the pre-grasp opening for its stance, grasp
   (`move_to_grasp --pad-center`: pad centre on the grasp point; records the executed grasp depth).
3. Reorient (`rotate_object`), keeping the camera up where possible.
4. Build `held` from the chain (§3.2 correction). Call `drive_mate{fmb1/mate:NN, target = measured base, held,
   corrections}`.
5. Per-seat depth check right after the insert; on a fail the supervisor recovers (release + nudge, edge press,
   re-aim) - recovery is the planner's/supervisor's, not drive-mate's (mem `per-seat-check-and-nudge.md`).
6. Open the RG2 (Grip), lift to safe height.
7. Fly a seat view; `execute_mate{fmb1/mate:NN}`; on `not_at_seat`, recover and ask again.
8. Next mate: aim from `base_measured`.

## 7. Briefing claims checked

| briefing claim | status |
|---|---|
| §1.1 frames, `p_world = (-x, -y, z + 0.080)` | confirmed (`/tf_static` base_link->base = pi about z). New: the cell's own TF `world` = base_link. |
| §1.4 RG2 always 40 N, `Grip.force` recorded only | confirmed (`grip_adapter.py:16-20,46`) |
| §1.4 pre-grasp openings 35/35/39.8/56.7 mm | the RECORDED values; runs use 39 / 35-39 / 55.8-87.8 / 96.7 mm chosen per stance |
| §1.5 hand-eye and 910.13 intrinsics | confirmed as the LOCALIZER's; the driver's `camera_info` (used by depth) says fx 923.80, fy 923.63, cx 622.89, cy 351.84, no distortion |
| §1.5 localizer ~3 Hz | 2.6 Hz measured today (arm still, 3 objects in view) |
| §2.2 active layout uncertain (`fmb1_parked`) | **stale**: `cell_layout.yaml:18` now says `active: fmb1_home_2026_10_21` |
| §2.3 last base read (0.1467, -0.4877), yaw 19.9 | stale: now (0.13445, -0.46945), yaw 15.14 (run 79); live (0.13508, -0.46609), 15.38 |
| §2.4 "all parts lie on a side" | the logged stances are the assembled orientation up to yaw (U parts, yellow: rpy (-90, 0, yaw)); line_green lies on a long face; all planned 1 pass |
| §2.4 u_brown at the clear area | stale: the current disassembly puts u_brown in staging too (`staged_parts.json`, run 79) |
| §3.3 `p_twin = Rz(-90) p_ours` "inferred" | consistent with both U slot translations (the only z-up rotation that maps them); orientations still unchecked |
| §4.1 "no gripper joint state" | confirmed; `/rg2/joint_states` exists with 0 publishers |
| §4.2 VisionPose `header.stamp` = frame arrival time | **wrong for `/vision/get_pose`**: it is the read's completion time |
| §4.2 DriveMate refusals | add `robot_stopped`, `corrections` |
| §4.2 ExecuteMate refusals | confirmed; the grip check runs before the predecessor re-read |
| §4.3 step 2 `get_pose(part, held)` | **not served**; the held pose is chained |
| §4.3 step 3 "hover over the seat" before drive-mate | drive-mate's own `translate_object` (and `drive_align`) fly to the hover |

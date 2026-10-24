# OWED - what this pack does not settle (2026-10-24)

The pack: `MEASURED_CELL.yaml` (facts), `declaration.draft.yaml` (static-only `pa.environment-declaration/1`,
`fmb1_realcell`), `task.draft.yaml` + `layouts/fmb1_realcell_run78_start.json`, `INTERFACE.md`, `tools/` (the two
read-only probes that produced `evidence/`). Nothing was committed, nothing was written on dual-a4500, nothing moved.

## 1. The live half of /onboard-environment (OWED to a session with the simulator and the publisher)

Per onboard-environment SKILL.md §1 "STATIC-ONLY authoring": declared here, not done, not faked.

- **Part 2 GATE**: the static gate has not been run on `declaration.draft.yaml`. Expect refusals at least for the
  `null` `minimum_rate_hz` floors (§3 below) and possibly for the non-dimelab stream keys (`coverage`,
  `snapshot_grouping: one_message_per_snapshot`, the camera streams).
- **Sim-binding map** (`pa.sim-binding-map/0`): not authored - prim paths and `controller_exec_edge` need a live stage.
- **Part 3 LAUNCH / bringup** and **Part 4 VERIFY** (wire readback, physics readback, `sim_instrumentation.physics_config`):
  not done. `physics.scene/bodies/physics_scene` carry `status: UNMEASURED_owed_first_compose`.
- **Part 5 PUBLISH**: no `receipt.json`, no `gate.json`; the task's `environment.receipt_sha256` is `OWED`.
- **Review checkpoints E1/E2/E3** (Robo Render environment editor): not built.
- **Robot pins**: sidecar `5a2d9d59...`, composite `d91b5a68...` and the table asset receipt `a7ad4633...` are copied
  from dimelab and re-pin at publish. The robot record itself is owed changes for this cell: a wrist-camera link at the
  measured hand-eye, and a check of the composite's flange-to-fingertip against the measured 230.86 mm (the twin's
  quick-changer chain says 18.1 mm, robo-manipulate says 15.92 mm; neither was compared).
- **Task publication** (/author-task): `task.draft.yaml` is unpublished; its prompt pins and assembly receipt are
  copied from `fmb1_assemble`; the publisher-derived `physics` block is absent; a real-cell `task_layer` prompt is owed.

## 2. The adapter the operator ruled is ours (2026-10-23) - NOT BUILT

The declaration states the twin surface; the cell serves something else today. Each line is a piece of adapter:

| twin surface (declared) | real today | needed |
|---|---|---|
| `/ur5e_0/arm_controller/follow_joint_trajectory` | `/scaled_joint_trajectory_controller/follow_joint_trajectory` | action alias / proxy |
| `/ur5e_0/rg2/grip` | `/ur5e/rg2/grip` | rename (token `ur5e` -> `ur5e_0`) |
| `/ur5e_0/execute_mate` | `/ur5e/execute_mate` | rename |
| `/ur5e_0/rg2/grip_events` | `/ur5e/rg2/grip_events` | rename |
| `/ur5e_0/joint_states`, `ur5e_0_*` names | `/joint_states`, unprefixed, non-UR order | prefixed republish |
| RG2 joint `ur5e_0_rg2_gripper_joint` | none (`/rg2/joint_states` has 0 publishers) | width -> joint via the composite's coupling, or a declared absence |
| `/ur5e_0/arm_controller/controller_state`, `_action/status` | under `/scaled_joint_trajectory_controller/` | alias |
| `/objects/poses` in twin `world`, children `fmb1/occurrence:0N`, centroid frames, object frames on `/tf` | label "World", values in UR `base`, cell names, bbox-centre frames, no `/tf` | transform `(-x,-y,z+0.080)`, rename, re-frame (needs §4.3), publish on `/tf` too |
| TF `world` = table-top plane, `world -> base_link` z 0.080 | `/tf_static` `world -> base_link` = **identity** | correct the static transform (or rename the cell's frame) |
| `/clock`, `use_sim_time` | none, wall time | either a wall-time contract or a clock bridge - a platform decision |
| `/objects/fastening_events` (ticket 082 journal; referee refuses `NO_FASTENING_EVIDENCE` without it) | ExecuteMate journals to the FILE `ablations/logs/mate_journal.jsonl` | a topic emitter in `execute_mate_server.py` |
| `goal_echo` | no emitter | only if a scored real run needs the FJT goal record |
| Grip feedback | none published | publish width feedback, or declare its absence |

Plus two surfaces the schema has no slot for (the declaration lists them in comments): `/ur5e/drive_mate`
(`DriveMate`) and `/vision/get_pose`, `/vision/certify_seat`. **Schema question for the platform:** may
`robots[].actions` carry extra roles (`drive_mate`), and where do perception services belong? Until answered, a
planner learns them from `INTERFACE.md`, not from the declaration.

## 3. UNMEASURED values (and what would measure each)

| value | measure by |
|---|---|
| table extent and xy placement (the declaration pins dimelab's 1.89 x 0.72 table as a STAND-IN) | tape from the robot base to the edges, or a depth survey of the edges at >= 0.5 m; design extent from the Vention USD on dual-a4500 (`~/a4500-backup-20260828/Desktop/omniverse_Library_10thoc2025/Digital workspace/VentionAssembly_393251_v88.usd`, read-only) |
| table material, static/dynamic friction, restitution | identify the panel; tilt/pull test with an FMB1 part and with base1 |
| Vention slider e-stop state today (last read: engaged, 2026-10-15) | read-only `estop/status;` to MachineMotion :9999 |
| every stream's `minimum_rate_hz` (object poses, camera, joint states, TCP, wrench, gripper, /tf) | record a full run as mcap and derive floors per SKILL.md (#126 method); for `/objects/poses` judge only still segments (the stream is silent while the arm moves) |
| camera rates on the wire (a Python probe received color 7.4-9.1 Hz, aligned depth 5.2-5.9 Hz vs the 15 fps profile) | `ros2 bag record` stats or a C++ subscriber |
| base CAD-fit sigma in z and yaw (vision_ops.py:107 sends NaN) | repeat the fit on a still base over N views |
| marker xy sigma in certification (reported, not judged) | marker xy of a seated part vs a tap map over N views |
| tilt-angle repeatability (reported only) | same survey |
| base1's true z (code: table + 17.5 mm; depth fit median 16.9 mm world, n 520) and its +0.9 deg pitch (touch survey said 0.47 deg) | 4-touch height survey on the base top (the same-frame-aim method) |
| hand-eye roll about the optical axis (~+-2 mm at the gripper) and the ~2 mm -y seat bias | QUEUED_FIXES #12 |
| clear-area tape square size (task reorient region uses the clear point +- 56.4 mm instead) | measure the tape |
| referee at-rest bounds at the cell's 2.6 Hz | derive from still-part pose series recorded on the cell |
| twin flange-to-fingertip vs 230.86 mm; quick-changer 18.1 vs 15.92 mm | compute from the composite and compare |
| cell part frame -> graph occurrence frame for the PARTS (orientation and centroid offsets; only base1's translation map is checked) | compare the cell meshes with the graph's Body STEP/USD: centroid offset per part, axis permutation |

## 4. Where the real cell cannot satisfy the schema or the twin's assumptions

1. **No scene-truth pose stream.** Poses come from a wrist camera: in-view only, ~2.6 Hz, nothing while the TCP moves
   > 15 mm/s, z snapped to the table-rest height (also for SEATED parts - live 2026-10-24: seated U parts published at
   z -0.045), orientation snapped to a symmetry branch. The planner must fly views and use `/vision/get_pose` (still
   samples + sigma) for anything it commits a motion to. The twin planners assumed 30 Hz of every object.
2. **Held parts are not observable** (gripper occludes them). The held pose is chained from grasp id + executed grasp
   depth + orientation from the previous step; `/vision/get_pose kind=held` returns "not served".
3. **The base is free and moves** (untaped; pushed by the reset; carried by a hard spiral; 1-4 mm creep per part with
   the Vention e-stop engaged). The twin task held Body1 `static`; this task frees it. Re-measure it before every insert.
4. **The insert is a force-controlled contact search** (`/ur5e/drive_mate`, <= 5 N on parts by default, <= 9 N with
   override, lateral <= 6 N, 1.5 N search on a free base, rotation locked), not a position descent, and it may move the
   arm itself (align step). Recovery (nudge, edge press, re-aim) is outside drive-mate.
5. **Simulator-only referee inputs do not exist**: `support_events`, `contact_events`, `attach_events`,
   `object_state_events`, `robot_health`, `isaac_joint_states`. The task's `require_support` and `require_weld_absent`
   gates have no real input; a real-cell referee definition is owed.
6. **Certification is weaker than the graph's band**: z judged within 2.0 mm (1 mm + 2 sigma), xy and tilt reported
   only, seated bars carry a -3 deg roll bias, and line_green under yellow is carried on its journalled certification.
7. **Grip**: force not commandable (always 40 N), no joint angle/effort, verdict never `slip`, no attach.
8. **Time** is wall time; the twin's sim-clock stamping and staleness rules do not transfer.
9. **Regions overlap**: base1's measured footprint union reaches into the staging region (UR x up to 0.263 vs staging
   x >= 0.19). The cell keeps parts 25 mm away by measurement, not by disjoint regions.
10. **Recording**: raw RGB and aligned depth are excused (bandwidth on the CPU-bound cell PC); compressed RGB is recorded.
11. **Recorded debt in the driver**: drive-mate's insert routes by part name and adds a learned per-part seat z
    (ARCHITECTURE_SEAMS.md:90-91).

## 5. Inconsistencies found

| # | what | where |
|---|---|---|
| 1 | The cell's live TF `world` is `base_link` (identity), i.e. 0.080 m above the twin/pack `world`. | `/tf_static` live vs declaration base_pose |
| 2 | Two camera intrinsic sets: markers use fx 910.13 / fy 911.16 / cx 619.57 / cy 349.15 + distortion; depth uses the driver's fx 923.80 / fy 923.63 / cx 622.89 / cy 351.84, no distortion (1.5 % focal difference). | `robot_config.yaml:15-21` vs live `camera_info`; `ablations/seat_check.py:84` |
| 3 | `VisionPose.header.stamp` documented as frame arrival time; `/vision/get_pose` stamps the read's completion time. `stationary_since` is `request - 0.8 s`, not a measured stillness start. `detection_set` always "". | `VisionPose.msg` vs `vision_ops.py:129-134` |
| 4 | `GetVisionPose.srv` and the briefing (§4.3 step 2) offer `kind: held`; the server refuses it. | `vision_ops.py:74-76` |
| 5 | `DriveMate.action` comment: `stage_reached SEAT | BASE_RIM | OBJ:<part> | WEDGED | NONE`; server: `seat | wedged | above | below | unsettled | no_trace`. Its `mate_id` example `fmb_assembly_1/u_brown` is stale. | `DriveMate.action:5,16` vs `insert_outcome.py:15` |
| 6 | DriveMate refusals `robot_stopped` and `corrections` exist in code, not in the docstring or briefing. `z_above_seat` feedback is always NaN. | `drive_mate_server.py:277,293,200,330` |
| 7 | ExecuteMate docstring orders "predecessors now" before "at rest"; the code checks the grip first. | `execute_mate_server.py:6-14` vs `:125-150` |
| 8 | `CertifySeat.srv` lists `method` values without `none`. | `CertifySeat.srv:8` vs `vision_ops.py:284` |
| 9 | Briefing §2.2: active layout uncertain (`fmb1_parked`) - now `active: fmb1_home_2026_10_21`. §2.3 last base read and §2.4 starting spots (u_brown at the clear area) are stale. §2.4 "all parts lie on a side": logged stances are the assembled orientation up to yaw; all planned 1 pass. | `cell_layout.yaml:18`, `last_base_pose.json`, `staged_parts.json`, `[derive-sequence]` lines |
| 10 | The configured assembly region (`fmb1_home_2026_10_21`, UR x <= 0.152) does not contain base1's measured placements (centre x up to 0.1595, footprint to 0.263). | `cell_layout.yaml:25` vs evidence/run_log_poses.json |
| 11 | base1 z: code sets table + 17.5 mm; the depth CAD fit reads 0.6 mm lower (median -63.1 mm UR, sd 0.25, n 520) with a steady +0.9 deg pitch. | `config.py:121` vs `[base] CAD fit` log lines |
| 12 | Mount height quoted as 82 mm in one memory note; code and Vention design say 80 / 80.48 mm. | `mem staging-layout-tipping.md` |
| 13 | Hand-eye in `mem camera-calibration-2026-10-13.md` is the superseded 10-13 solve; the file holds the 10-20 survey values (the briefing has the right ones). | `robot_config.yaml:9-12` |
| 14 | Localizer default `table_z -0.0825` in code vs `-0.08` in the config that is loaded. | `aruco_camera_localizer/robot_config.py:36` vs `config/robot_config.yaml:44` |
| 15 | Localizer rate reported as ~3, ~3.6 and ~2.3 Hz in notes; 2.6 Hz measured today (arm still, 3 objects). | `CLAUDE.md:25`, mem optiplex-cell-quirks, `replay_real_assembly.py:561` |
| 16 | Twin-internal: dimelab declares `grip_action_feedback` as `pa_robot_msgs/action/Grip_FeedbackMessage`; the composite's Grip type is `pa_gripper_msgs/action/Grip`. | dimelab `declaration.yaml:491` vs `composite.yaml:256` |
| 17 | `CLAUDE.md` §3 states a 5 N force-mode default while `translate_object` passes the tuned 9 N / 8 N (CLAUDE.md:192-193 says so); the empty-base push guard is 12 N. Both documented, but a planner reading only §3 gets 5 N. | `CLAUDE.md:207` |
| 18 | The briefing's pre-grasp openings (35/35/39.8/56.7 mm) are the recorded ones; runs open wider per stance (up to 96.7 mm). | `[pick]` log lines |
| 19 | The loop that set base1's per-run targets (`/tmp/fmb1_loop.log`) has no script in the repo. | `/tmp/fmb1_loop.log` |
| 20 | The task asked for the staged record via `STAGED`; that file holds the commanded SET-DOWN pose (z = release height, e.g. line_green z -0.039 vs its resting -0.067), not the resting pose. The resting pose comes from the pick-time observations. | `disassemble.py:131-145` vs `[pick]` lines |

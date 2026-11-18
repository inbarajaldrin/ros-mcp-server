# Notes for pose6d: what the cell's perception is and what a replacement must keep

pose6d is parked (PLAN.md R2); the cell keeps `aruco_camera_localizer`. This file holds what pose6d will need when it replaces
it. First written 2026-11-16 for the pose6d session; kept current here. Unknowns are marked.

## The camera and the robot
- Intel RealSense D415 on the wrist (arm-mounted), USB 3. Driver `realsense2_camera` 4.58.3, launched by
  `compliant_insertion_studio/scripts/stack_up.sh`: colour 1280x720 @ 15, depth 1280x720 @ 15, `align_depth.enable:=true`, no IR.
- Mount: v3.1 printed saddle since 2026-11-10, CAD tool0 -> colour optical (0, 0.0889, 0.0839) m, quat xyzw
  (0.258819, 0, 0, 0.965926) (+Y side, tilted 30 deg), NOT yet calibrated. The old mount (until 11-10, -Y side, straight down)
  was calibrated 2026-10-13 with a ChArUco board (3 spots, 326 frames; held-out ~0.9 mm) and refit 2026-10-20 from a 24-view
  base1 survey (`ablations/handeye_icp_survey.py`, `ablations/handeye_from_survey.py`; spread 1.2 mm median, 3.5 mm max).
- Robot frame: UR `base` (DH base); workspace at -y; table at z -0.080 m. Twin world = (-x, -y, z + 0.080), yaw + 180 deg.

## The localizer as it runs
- `$FMB_ROS_WS/src/aruco_camera_localizer` (github.com/inbarajaldrin/aruco_camera_localizer, main >= c4e9836), launched by
  `compliant_insertion_studio/scripts/launch_camera.sh --headless --image-topic /camera/camera/color/image_raw`, which runs
  `ros2 run aruco_camera_localizer localize` with `FMB_CAD_DATA=<repo>/assets/cad_data`.
- Config: `config/robot_config.yaml` (hand-eye `cam_offset_position` / `cam_offset_quat` in the EE frame = `/tcp_pose_broadcaster/pose`;
  intrinsics `camera_matrix [910.13, 911.16, 619.57, 349.15]`, distortion `[0.130074, -0.439479, -0.001453, -0.000549, 0.369498]`;
  `table_z -0.08`), `config/filter_config.yaml` (combined multi-marker poses, boards yaw-only, z snapped to rest height,
  symmetry snap below 10 deg tilt, IPPE disambiguation toward table-resting, motion pause above 15 mm/s TCP, max reprojection 1 px,
  AprilTag corner refine), `config/sticker_calibration.json` (per-sticker dx/dy/dtheta + printed edge, pose6d-ios ADR 0010 model,
  solved 2026-10-14 from 72 frames of the seated assembly; stickers 6, 18, 24).
- Intrinsics: origin unknown (pre-existing); validated at 0.23-0.25 px per-view RMS on 2026-10-13. The depth code uses the driver's
  `camera_info` instead (fx 923.80, fy 923.63, cx 622.89, cy 351.84, no distortion): a 1.5 % focal difference, unresolved.
- Markers (`assets/cad_data/aruco/<part>_aruco.json`): every marker 0.021 m, border 0.05; per marker `aruco_id`, `face_type`,
  `surface_normal`, `T_object_to_marker`. Part frame = mesh bounding-box centre (`assets/cad_data/models/<part>.obj`).
  FMB1 `DICT_5X5_50`: base1 0-3, inverted_u_yellow 4-9, line_green 10-15, u_brown 16-21, u_orange 22-27.
  FMB3 `DICT_4X4_50`: fork_orange 0-5, fork_yellow 6-11, line_red 12-17, line_brown 18-23, base3 24-27.

## What it publishes
| topic | type | notes |
|---|---|---|
| `/objects_poses_real` | `tf2_msgs/TFMessage` | `child_frame_id` = part name; `header.frame_id` "World" but values in UR `base`; m, xyzw; in-view parts only; ~2.6-3 Hz still, nothing above 15 mm/s TCP; z snapped, symmetry-snapped. ~11 consumers (move_to_grasp, move_to_clear_area, replay_real_assembly, cell_io, grasp_points_publisher -> `/grasp_points_real`, scan_workspace, queries/*, seams/vision_server -> `/objects/poses`, `/vision/get_pose`) |
| `/objects_poses_real_raw` | `TFMessage` | before snapping (pick_from_bag, bag recorder) |
| `/annotated_stream` | `sensor_msgs/Image` | overlays + compass (operator page, RViz, bags) |
| `/camera_pose` | `PoseStamped` | bags, render_bag |

## How well it works, and the workarounds
- Marker reads at the low pre-grasp hover were 9-10 mm off; picks and base reads commit on CAD-to-depth fits seeded by the marker
  (`ablations/part_depth.py`, `ablations/base_icp.py`). Marker yaw runs a few degrees off where stickers sit turned.
- Moving camera ~10 mm off -> poses only from a still arm (>= 0.25 s after stopping; 0.8 s window in `seams/vision_ops.py`).
- At the 0.56 m base view the detection set flips and the base read steps ~8 mm -> 6-dof depth fit for the base.
- IPPE flips on side-facing markers (line_green 80-90 deg off) -> resting-face / 6-dof depth fits.
- Cross-dictionary phantoms with FMB1 + FMB3 on one table -> dedupe (localizer af6c3a2).
- Seated parts published at table-rest z (the snap) -> depth seat checks certify, not the localizer.
- Held parts occluded; half-closed fingers hide the part below; a horizontal gripper sees no table.
- Lighting: dark room needs manual exposure 8000 / gain 128; that setting in daylight gives white frames.
- 2.6-3 Hz, CPU-bound on the i5-8500T; USB 2 makes it hang silently.
- Open: optical-axis roll (+-2 mm at the gripper), a ~2 mm -y bias, boards solving ~4 mm above the table.

## What a replacement must provide to switch without breaking
- `/objects_poses_real` with the same semantics (names, UR `base`, bbox-centre part frames, xyzw, symmetry branch, z snap), or
  every consumer changed in the same step; `/objects_poses_real_raw`, `/annotated_stream`, `/camera_pose`.
- No output while the arm moves fast; capture-time stamps (today `/vision/get_pose` stamps the read's end); >= ~3 Hz.
- `robot_config.yaml` is read directly by the cell's Python (`primitives/shared/config.py LOCALIZER_CONFIG`: seat checks, depth
  fits, camera-up wrist choice, hand-eye tools): move it and that path together.
- Marker/part data readable from `$FMB_CAD_DATA` in the `*_aruco.json` format, or converted from `.pose6d` for those readers.
- `DICT_4X4_50` and `DICT_5X5_50` at once, 21 mm markers, cross-dictionary dedupe.
- The intended swap point is the vision seam: `seams/vision_server.py` / `seams/vision_ops.py` (`GetVisionPose`, `CertifySeat`,
  `fmb_seam_msgs/VisionPose` with a sigma). In DroidStack terms the perception node is an adapter publishing on TF (plan47, 2026-11-18).

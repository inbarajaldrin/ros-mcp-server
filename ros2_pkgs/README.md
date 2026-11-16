# ROS 2 packages the cell needs that have no other home

Copy (or symlink) both into your ROS 2 workspace `src/` and build them with the rest (see ONBOARDING.md §2).

| package | what | source |
|---|---|---|
| `onrobot_ros/` | the OnRobot RG2 bridge (`ros2 run onrobot_ros gripper_control`): `/gripper_command`, `/gripper_width`, `/gripper_width_offset`, `/gripper_grasp_detected`, `/gripper_motion_ongoing`, `/gripper_status`, `/restart_gripper_connection` | `github.com/karishmapatnaik/onrobot_ros` @ its HEAD **plus the cell's local edits** (the state-feedback topics above, `onrobot.py` and `rg_gripper.py`, +402 lines) that were never pushed upstream. Copied 2026-11-16 from the cell PC's `~/Desktop/ros2_ws/src/onrobot_ros`. |
| `pa_msgs/` | `fmb_seam_msgs` (ours: `VisionPose`, `DriveMate`, `GetVisionPose`, `CertifySeat`) and `pa_assembly_msgs` / `pa_gripper_msgs` (copied from prompt2assembly: `ExecuteMate`, `Grip`), used by `seams/` | the cell PC's `~/Desktop/ros2_ws/src/pa_msgs` (never in git), copied 2026-11-16 |

The other two packages come from GitHub: `aruco_camera_localizer` (github.com/inbarajaldrin/aruco_camera_localizer, main >= c4e9836)
and `max_camera_msgs` (github.com/inbarajaldrin/max_camera_msgs).

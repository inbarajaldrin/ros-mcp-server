# Checkpoint fmb1-2026-10-23-run48

A full FMB1 cycle (reset + assembly, every mate certified) ran on commit `41ff77e518` with the dependencies in
MANIFEST.json. Tag: `ckpt/fmb1-2026-10-23-run48`.

## Demo from this checkpoint (leaves the development branch alone)
```bash
cd ~/Documents/ros-mcp-server-verified
git worktree add ~/fmb1_demo_fmb1-2026-10-23-run48 ckpt/fmb1-2026-10-23-run48          # a frozen copy of the code at this checkpoint
cd ~/fmb1_demo_fmb1-2026-10-23-run48
bash compliant_insertion_studio/scripts/stack_up.sh      # driver, gripper, camera, grasp points (no arm motion)
# seam servers, from the checkpoint copy:
for s in vision_server grip_adapter drive_mate_server execute_mate_server; do nohup python3 -u -m seams.$s > /tmp/$s.log 2>&1 & done
export FMB_LAYOUT=fmb1_home_2026_10_21 BASE_FROM=icp NO_AIM_BIAS=1 PRE_INSERT_READ=off FOLLOW_YAW=0
J=ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json
# parts staged (u_brown at the clear area, the others on the table) -> one assembly:
SEAMS_DRIVE=1 SEAMS_CERTIFY=1 python3 -u ablations/replay_real_assembly.py --assembly-json $J --part-at-staging
```
The full cycle with the reset is in compliant_insertion_studio/docs/HANDOFF_NEXT_AGENT.md (2026-10-22/23 section).

## Restore the dependencies to this checkpoint
- ros2_ws packages: for each entry in MANIFEST.json `ros2_ws`, `git -C ~/Desktop/ros2_ws/src/<pkg> checkout <git>` and
  `git apply checkpoints/fmb1-2026-10-23-run48/<patch>` where a patch is listed; packages saved as a tarball: extract over src/.
  Then `colcon build` in ~/Desktop/ros2_ws.
- calibration: copy configs/localizer_*.yaml|json back to ~/Desktop/ros2_ws/src/aruco_camera_localizer/config/
  (drop the localizer_ prefix); the F/T calibration and cell layout are in the repo at this commit already.
- Python packages: pip_freeze.txt; ROS debs: ros_debs.txt.

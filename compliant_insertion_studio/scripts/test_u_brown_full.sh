#!/bin/bash
# Full pick→insert→home test for u_brown using the autonomous SEARCH director.
#
# Flow:
#   1. Release currently-held u_orange (move-to-clear-area + open + safe-height)
#   2. run_assembly_step on u_brown with --autonomous-search (full canonical
#      pick → rotate → place → regrasp → rotate → insert)
#   3. move_home
#
# Camera must already be up (launch_camera.sh --background).
#
# Usage:
#   bash compliant_insertion_studio/scripts/test_u_brown_full.sh

set -uo pipefail
cd /home/aaugus11/Documents/ros-mcp-server

step() { echo; echo "===== $* ====="; }

step "1. Release currently-held u_orange (move-to-clear-area + open + safe-height)"
python3 -m primitives.core.move_to_clear_area --mode real || exit 1
python3 -m primitives.control_gripper 35 --mode real || exit 1
sleep 1.5
python3 -m primitives.move_to_safe_height --mode real || exit 1

step "2. Full u_brown pick→rotate→place→regrasp→rotate→insert (autonomous SEARCH)"
python3 -u -m compliant_insertion_studio.scripts.run_assembly_step \
  --object-name u_brown --base-name base1 --grasp-id 1 \
  --grasp-width 35 \
  --mode real \
  --fz -9.0 --override-fz-cap \
  --step-back auto --step-back-seconds 1.0 \
  --autonomous-search \
  --search-F-press-N 7.0 \
  --search-Fmax-N 5.0 \
  --search-v-s-mm-s 5.0 \
  --search-max-duration-s 25.0 \
  || { echo "u_brown insert failed"; exit 1; }

step "3. move_home"
python3 -m primitives.move_home --mode real || exit 1

echo
echo "===== u_brown FULL PIPELINE COMPLETE ====="

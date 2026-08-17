#!/bin/bash
# stack_up.sh — bring up the whole cell for a real FMB1 assembly, in one command.
#
#   bash compliant_insertion_studio/scripts/stack_up.sh
#
# Order matters and is enforced here: driver -> gripper -> camera -> grasp points.
# Each stage is verified before the next starts, so a failure names the stage that
# broke instead of surfacing three steps later as a mystery.
#
# Run it from the tree you intend to run the assembly from. The grasp publisher is
# started from $PWD deliberately: main's copy publishes composite marker ids
# (101..303) while the assembly JSONs carry flat ids, so every pick fails with
# "Grasp point 1 not found". See the root CLAUDE.md.
#
# Preconditions this script checks rather than assumes:
#   - robot powered, brakes released, pendant in REMOTE CONTROL
#   - overlay workspace built (~/Desktop/ros2_ws)
#
# Idempotent: re-running skips whatever is already healthy.
#
# Exit codes: 0 all up and verified | non-zero = the stage that failed.

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_IP="${ROBOT_IP:-192.168.1.111}"
DASH="${DASH:-$HOME/Documents/prismatic-manipulation/scripts/ur_dashboard.py}"
cd "$REPO" || exit 1

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
die()  { printf '  \033[31m✗ %s\033[0m\n' "$1"; exit "${2:-1}"; }
step() { printf '\n\033[1m[%s] %s\033[0m\n' "$1" "$2"; }

# ROS setup scripts reference unbound vars internally, so `set -u` kills this
# script before its first line of output. Disable u only across the sourcing.
set +u
source /opt/ros/humble/setup.bash 2>/dev/null
source "$HOME/Desktop/ros2_ws/install/setup.bash" 2>/dev/null
set -u

echo "════════════════════════════════════════════════════════════"
echo "  stack_up — tree: $REPO"
echo "════════════════════════════════════════════════════════════"

# ---------------------------------------------------------------- 0. robot state
step 0/5 "Robot power + pendant mode"
timeout 3 ping -c1 "$ROBOT_IP" >/dev/null 2>&1 \
    || die "UR at $ROBOT_IP unreachable — check cables/network." 2
if [ -f "$DASH" ]; then
    DSTATE="$(timeout 20 python3 "$DASH" --host "$ROBOT_IP" --real --wait 15 mode remote 2>&1)"
    echo "$DSTATE" | grep -q "RUNNING" \
        || die "Robot not RUNNING. Power on + release brakes: python3 $DASH --host $ROBOT_IP --real power_up" 2
    echo "$DSTATE" | grep -q "remote: true" \
        || die "Pendant is in LOCAL mode. Headless control needs REMOTE CONTROL (top-right in PolyScope)." 2
    ok "robot RUNNING, pendant in Remote Control"
else
    warn "dashboard client not found at $DASH — skipping power/mode check"
fi

# ---------------------------------------------------------------- 1. UR driver
step 1/5 "UR driver (headless external control)"
if timeout 8 ros2 control list_controllers 2>/dev/null | grep -q "scaled_joint_trajectory_controller.*active"; then
    ok "driver already up"
else
    bash "$REPO/compliant_insertion_studio/scripts/launch_robot.sh" real --headless >/tmp/stack_up_driver.log 2>&1 \
        || { tail -15 /tmp/stack_up_driver.log; die "driver bringup failed (see /tmp/stack_up_driver.log)" 3; }
    grep -q "Payload has been set successfully" /tmp/stack_up_driver.log \
        && ok "F/T payload set from ft_calibration_*.yaml" \
        || warn "payload NOT confirmed — force mode will lack gravity comp"
    ok "driver up, controllers active"
fi

# ---------------------------------------------------------------- 2. gripper
step 2/5 "OnRobot RG2 bridge"
if timeout 5 ros2 topic list 2>/dev/null | grep -q "^/gripper_status$"; then
    ok "gripper bridge already up"
else
    nohup bash -c 'source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash;
        ros2 run onrobot_ros gripper_control' >/tmp/gripper.log 2>&1 &
    for _ in $(seq 1 20); do
        sleep 1; grep -q "Gripper node ready\|initialized successfully" /tmp/gripper.log && break
    done
    grep -q "Gripper node ready\|initialized successfully" /tmp/gripper.log \
        || { tail -8 /tmp/gripper.log; die "gripper bridge failed (see /tmp/gripper.log)" 4; }
    ok "gripper initialized"
fi
GS="$(timeout 8 ros2 topic echo --once /gripper_status 2>/dev/null | head -2 | tr -d '\n')"
case "$GS" in
    *Circuit1:True*|*Circuit2:True*) die "gripper SAFETY CIRCUIT LATCHED — needs a Compute Box power-cycle (SETUP.md §7.1)" 4 ;;
    *) ok "gripper safety circuits clear" ;;
esac

# ---------------------------------------------------------------- 3. camera
step 3/5 "Camera (aruco localizer)"
if timeout 5 ros2 topic list 2>/dev/null | grep -q "^/objects_poses_real$"; then
    ok "camera already publishing"
else
    bash "$REPO/compliant_insertion_studio/scripts/launch_camera.sh" --background >/dev/null 2>&1
    for _ in $(seq 1 20); do
        sleep 1; timeout 3 ros2 topic list 2>/dev/null | grep -q "^/objects_poses_real$" && break
    done
    timeout 5 ros2 topic list 2>/dev/null | grep -q "^/objects_poses_real$" \
        || die "camera did not start (see /tmp/aruco_logs/)" 5
    ok "camera up"
fi

# ---------------------------------------------------------------- 4. grasp points
step 4/5 "Grasp-points publisher (from THIS tree)"
P=grasp_points
if pgrep -f "python3.*${P}_publisher" >/dev/null 2>&1; then
    ok "grasp publisher already running"
    warn "if it was started from another tree, ids may be composite — stack_down first to be sure"
else
    nohup bash -c "source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash;
        cd '$REPO'; python3 -u utils/grasp_points_publisher.py --mode real" >/tmp/grasp_pub.log 2>&1 &
    for _ in $(seq 1 20); do
        sleep 1; timeout 3 ros2 topic list 2>/dev/null | grep -q "^/grasp_points_real$" && break
    done
    timeout 5 ros2 topic list 2>/dev/null | grep -q "^/grasp_points_real$" \
        || { tail -8 /tmp/grasp_pub.log; die "grasp publisher failed (see /tmp/grasp_pub.log)" 6; }
    ok "grasp publisher up"
fi

# ---------------------------------------------------------------- 5. verify
step 5/5 "Pre-flight"
FAIL=0
FRAMES="$(timeout 12 ros2 topic echo --once /objects_poses_real 2>/dev/null | grep -c child_frame_id)"
[ "${FRAMES:-0}" -ge 2 ] && ok "camera sees $FRAMES frames" || { warn "camera sees only ${FRAMES:-0} frames — are the parts in view?"; FAIL=1; }

HZ="$(timeout 8 ros2 topic hz /grasp_points_real 2>&1 | grep -oE 'average rate: [0-9.]+' | head -1)"
[ -n "$HZ" ] && ok "grasp_points_real ${HZ#average rate: } Hz (expect ~5)" || { warn "no /grasp_points_real rate"; FAIL=1; }

THZ="$(timeout 8 ros2 topic hz /tcp_pose_broadcaster/pose 2>&1 | grep -oE 'average rate: [0-9.]+' | head -1)"
[ -n "$THZ" ] && ok "tcp_pose ${THZ#average rate: } Hz (expect ~500)" || { warn "no /tcp_pose_broadcaster/pose"; FAIL=1; }

timeout 8 ros2 control list_controllers 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -q "force_torque_sensor_broadcaster.*active" \
    && ok "force_torque_sensor_broadcaster active" || { warn "F/T broadcaster not active"; FAIL=1; }

echo
if [ "$FAIL" -eq 0 ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "  STACK UP. Run the assembly one object at a time:"
    echo "    J=ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json"
    echo "    python3 -u ablations/replay_real_assembly.py --assembly-json \$J --only u_brown"
    echo "════════════════════════════════════════════════════════════"
else
    echo "  Stack is up but PRE-FLIGHT HAS WARNINGS — read them before moving the robot."
    exit 7
fi

#!/bin/bash
# stack_down.sh — stop the whole cell cleanly, in one command.
#
#   bash compliant_insertion_studio/scripts/stack_down.sh
#
# Reverse of stack_up.sh: grasp points -> camera -> gripper -> driver. Publishers
# go first so nothing is left commanding a robot whose driver is going away.
#
# Signal discipline (these are not stylistic — each has cost a session):
#   - RViz and other X11 owners get SIGTERM only. SIGKILL on an X11 window can
#     cascade into KWin BadWindow errors and take down the display session.
#   - The gripper bridge does not match `pkill -f gripper_control` (its real
#     cmdline is `python3 /opt/ros/humble/bin/ros2 run ...`). Match the socat it
#     spawns, and leave >=5s before any respawn or the next pyserial open fails
#     with (22, 'Invalid argument') on the PTY.
#   - pkill patterns are built at runtime ($P) so the pattern text in this
#     script's own cmdline cannot self-match and kill the calling shell (exit 144).
#
# Usage: stack_down.sh [-v]     -v : list surviving processes at the end
# Exit:  0 clean | 1 something survived (listed)

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERBOSE="${1:-}"

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
step() { printf '\n\033[1m[%s] %s\033[0m\n' "$1" "$2"; }

# See note in stack_up.sh: `set -u` + ROS setup sourcing exits before any output.
set +u
source /opt/ros/humble/setup.bash 2>/dev/null
set -u

echo "════════════════════════════════════════════════════════════"
echo "  stack_down"
echo "════════════════════════════════════════════════════════════"

# ------------------------------------------------------- 1. grasp publisher
step 1/4 "Grasp-points publisher"
P=grasp_points
if pgrep -f "python3.*${P}_publisher" >/dev/null 2>&1; then
    pkill -SIGTERM -f "python3.*${P}_publisher"
    sleep 2
    pgrep -f "python3.*${P}_publisher" >/dev/null 2>&1 \
        && { pkill -SIGKILL -f "python3.*${P}_publisher"; warn "needed SIGKILL"; } \
        || ok "stopped"
else
    ok "not running"
fi

# ------------------------------------------------------- 2. camera
step 2/4 "Camera (aruco localizer)"
A=aruco_camera
if pgrep -f "${A}_localizer" >/dev/null 2>&1; then
    # SIGINT so rclpy runs its shutdown hooks and releases the device cleanly.
    pkill -SIGINT -f "${A}_localizer"
    sleep 3
    pgrep -f "${A}_localizer" >/dev/null 2>&1 \
        && { pkill -SIGTERM -f "${A}_localizer"; sleep 2; warn "needed SIGTERM"; } \
        || ok "stopped"
else
    ok "not running"
fi

# ------------------------------------------------------- 3. gripper bridge
step 3/4 "OnRobot RG2 bridge"
G=gripper_control
S=socat
if pgrep -f "$G" >/dev/null 2>&1 || pgrep -f "${S}.*ttyUR" >/dev/null 2>&1; then
    pkill -SIGTERM -f "$G" 2>/dev/null
    pkill -SIGTERM -f "${S}.*ttyUR" 2>/dev/null
    sleep 3
    if pgrep -f "${S}.*ttyUR" >/dev/null 2>&1; then
        pkill -SIGKILL -f "${S}.*ttyUR" 2>/dev/null
        warn "socat needed SIGKILL"
    fi
    ok "stopped (wait >=5s before restarting — PTY/termios race)"
else
    ok "not running"
fi

# ------------------------------------------------------- 4. driver + RViz
step 4/4 "UR driver + RViz"
if [ -x "$REPO/compliant_insertion_studio/scripts/close_robot.sh" ] \
   || [ -f "$REPO/compliant_insertion_studio/scripts/close_robot.sh" ]; then
    bash "$REPO/compliant_insertion_studio/scripts/close_robot.sh" >/tmp/stack_down_driver.log 2>&1
    grep -q "CLEAN" /tmp/stack_down_driver.log && ok "driver stopped cleanly" \
        || warn "close_robot.sh reported leftovers (see /tmp/stack_down_driver.log)"
else
    warn "close_robot.sh missing — driver not stopped"
fi
# close_robot.sh has been observed to leave the `ros2 launch` parent alive after
# its children are gone. SIGTERM first; it owns no X11 window.
L=ur5e_with_rg2
for pid in $(pgrep -f "[r]os2.*launch.*${L}" 2>/dev/null); do
    kill -SIGTERM "$pid" 2>/dev/null; sleep 3
    kill -0 "$pid" 2>/dev/null && kill -SIGKILL "$pid" 2>/dev/null
    ok "stopped lingering launch parent (pid $pid)"
done

# RG2 helper nodes spawned by ur5e_with_rg2.launch.py. close_robot.sh does NOT
# kill these, so they accumulate one pair per driver launch — observed 2026-08-16
# with TWO rg2_joint_state_publishers and TWO rg2_command_bridges alive after a
# "clean" shutdown, both publishing to the same topics. ros2 node list warns
# "nodes in the graph that share an exact name"; two joint_state_publishers on one
# topic is a real fault, not cosmetic.
# distinct names: A is already taken by the camera step above
RG=rg2; RJ=_joint_state_publisher; RB=_command_bridge
if pgrep -f "${RG}${RJ}" >/dev/null 2>&1 || pgrep -f "${RG}${RB}" >/dev/null 2>&1; then
    pkill -SIGTERM -f "${RG}${RJ}" 2>/dev/null
    pkill -SIGTERM -f "${RG}${RB}" 2>/dev/null
    sleep 2
    ok "RG2 helper nodes stopped"
else
    ok "RG2 helper nodes not running"
fi

ros2 daemon stop >/dev/null 2>&1

# ------------------------------------------------------- report
echo
LEFT="$(pgrep -af "ur_ros2_control|[u]r5e_with_rg2|[a]ruco_camera_localizer|[g]rasp_points_publisher|[s]ocat.*ttyUR|${RG}${RJ}|${RG}${RB}" 2>/dev/null | grep -v "bash -c" || true)"
if [ -z "$LEFT" ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "  STACK DOWN. Nothing left running."
    echo "════════════════════════════════════════════════════════════"
    exit 0
else
    echo "  Survivors:"; echo "$LEFT" | sed 's/^/    /'
    [ "$VERBOSE" = "-v" ] && { echo; echo "  All ROS nodes:"; timeout 5 ros2 node list 2>/dev/null | sed 's/^/    /'; }
    exit 1
fi

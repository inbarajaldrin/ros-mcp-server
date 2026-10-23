#!/bin/bash
# launch_rviz_demo.sh — RViz for the FMB1 demo: robot | gripper camera, side by side.
#
#   bash compliant_insertion_studio/scripts/launch_rviz_demo.sh            # background
#   bash compliant_insertion_studio/scripts/launch_rviz_demo.sh --fg       # foreground
#   ... --raw            camera panel shows the raw RealSense image (/camera/camera/color/image_raw)
#   ... --topic T        camera panel shows image topic T
#   ... --split 0.4      camera gets 40% of the width (default 0.5)
#   ... --camera-left    camera on the left, robot on the right
#   ... --depth          a depth panel under the camera: /depth_view (height above the table in colour,
#                        compliant_insertion_studio/scripts/depth_view.py - started here if it is not running)
#
# Left half: UR5e + RG2 models from the live driver, plus /grasp_points_real.
# Right half: the D415 image (marker detections drawn by aruco_camera_localizer).
# No Displays/Views panels — reopen one from RViz's Panels menu if needed.
#
# Run AFTER stack_up.sh: it needs the driver (/robot_description, /rg2/...) and the
# camera. It does not move the robot and does not start anything but RViz.
#
# The split is fitted to the screen it starts on (make_split_layout.py), so it works
# unchanged on a projector or a different monitor.
#
# Stop: stack_down.sh / close_robot.sh already SIGTERM `rviz2 -d`. Never SIGKILL an
# X11 window (see stack_down.sh).

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RVIZ_DIR="$REPO/compliant_insertion_studio/rviz"
RUN_DIR=/tmp/fmb1_rviz
mkdir -p "$RUN_DIR"

FG=false; TOPIC=/annotated_stream; SPLIT=0.5; SIDE=right; DEPTH=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fg) FG=true; shift ;;
        --raw) TOPIC=/camera/camera/color/image_raw; shift ;;   # realsense2_camera (was /intel_camera_rgb_raw)
        --topic) TOPIC="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --camera-left) SIDE=left; shift ;;
        --depth) DEPTH=true; shift ;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
die()  { printf '  \033[31m✗ %s\033[0m\n' "$1"; exit "${2:-1}"; }

set +u
source /opt/ros/humble/setup.bash 2>/dev/null
source "$HOME/Desktop/ros2_ws/install/setup.bash" 2>/dev/null
set -u

[[ -n "${DISPLAY:-}" ]] || die "no DISPLAY — run this from the desktop session" 2

R=rviz2
if pgrep -f "${R} -d .*fmb1_demo" >/dev/null 2>&1; then
    warn "demo RViz already running (pid $(pgrep -f "${R} -d .*fmb1_demo" | tr '\n' ' '))"
    exit 0
fi

echo "[rviz_demo] pre-flight"
TOPICS="$(timeout 8 ros2 topic list 2>/dev/null)"
for t in /robot_description /rg2/robot_description; do
    grep -qx "$t" <<<"$TOPICS" && ok "$t" \
        || warn "$t missing — robot model will be blank until the driver is up (stack_up.sh)"
done
if grep -qx "$TOPIC" <<<"$TOPICS"; then
    HZ="$(timeout 6 ros2 topic hz "$TOPIC" 2>&1 | grep -m1 -oE 'average rate: [0-9.]+')"
    [[ -n "$HZ" ]] && ok "$TOPIC ${HZ#average rate: } Hz" \
        || warn "$TOPIC exists but no frames in 6 s — camera panel will be blank"
else
    warn "$TOPIC missing — camera panel will be blank until launch_camera.sh is up"
fi

CFG="$RUN_DIR/fmb1_demo_runtime.rviz"
sed "s|Value: /annotated_stream|Value: $TOPIC|" "$RVIZ_DIR/fmb1_demo.rviz" > "$RUN_DIR/fmb1_demo_topic.rviz"
SECOND=()
if [[ "$DEPTH" == true ]]; then
    D=depth_view
    if ! pgrep -f "${D}.py" >/dev/null 2>&1; then
        nohup python3 -u "$REPO/compliant_insertion_studio/scripts/depth_view.py" >/tmp/depth_view.log 2>&1 &
        ok "started depth_view.py (log /tmp/depth_view.log)"
    fi
    python3 - "$RUN_DIR/fmb1_demo_topic.rviz" <<'PY'
import sys, yaml
p = sys.argv[1]; cfg = yaml.safe_load(open(p))
cam = next(d for d in cfg['Visualization Manager']['Displays'] if d.get('Name') == 'Camera')
dep = dict(cam, Name='Depth', Topic=dict(cam['Topic'], Value='/depth_view'))
cfg['Visualization Manager']['Displays'].append(dep)
yaml.safe_dump(cfg, open(p, 'w'), sort_keys=False, width=1000)
PY
    SECOND=(--second-dock Depth)
fi
LAYOUT="$(python3 "$RVIZ_DIR/make_split_layout.py" "$RUN_DIR/fmb1_demo_topic.rviz" "$CFG" \
        --image-dock Camera --split "$SPLIT" --camera-side "$SIDE" "${SECOND[@]}" 2>"$RUN_DIR/layout.err")" \
    || { cat "$RUN_DIR/layout.err" >&2; die "could not build the split layout" 3; }
ok "layout ${LAYOUT#*: }, camera topic $TOPIC"

if [[ "$FG" == true ]]; then
    exec rviz2 -d "$CFG"
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="$RUN_DIR/rviz_$TS.log"
nohup rviz2 -d "$CFG" >"$LOG" 2>&1 &
PID=$!
sleep 4
kill -0 "$PID" 2>/dev/null || { tail -15 "$LOG" >&2; die "rviz2 exited during startup (log: $LOG)" 4; }
renice -n 10 -p "$PID" >/dev/null 2>&1   # rendering must not starve the UR driver (stack_up.sh)
ok "RViz up (pid $PID, log $LOG)"

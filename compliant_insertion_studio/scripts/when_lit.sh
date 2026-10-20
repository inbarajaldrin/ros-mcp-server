#!/bin/bash
# Wait for the workspace light (colour frame mean > 80 and base1 + all 4 parts detected, held for ~60 s),
# then run the given command. The loop runs only when the light is on (operator, 2026-10-19); with the
# light off the localizer sees nothing and every verify fails closed.
#
#   bash compliant_insertion_studio/scripts/when_lit.sh 'bash ablations/part_loop.sh u_brown 3'
#   bash compliant_insertion_studio/scripts/when_lit.sh 'python3 -u ablations/fmb1_loop.py --cycles 3 --layout spots'
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash
CMD=${1:?command to run once lit}; NEED=${WHEN_LIT_PARTS:-5}
probe() { timeout 20 python3 - <<'EOF'
import rclpy, numpy as np, time
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
rclpy.init(); n=rclpy.create_node('lightprobe'); st={}; seen=set()
n.create_subscription(Image,'/camera/camera/color/image_raw',lambda m: st.__setitem__('c',m),1)
n.create_subscription(TFMessage,'/objects_poses_real',lambda m: seen.update(t.child_frame_id for t in m.transforms),10)
t=time.time()
while time.time()-t<5: rclpy.spin_once(n,timeout_sec=0.1)
m=st.get('c'); b=np.frombuffer(m.data,np.uint8).mean() if m else 0
print(f"{b:.0f} {len(seen & {'base1','u_brown','u_orange','line_green','inverted_u_yellow'})}")
EOF
}
lit=0
while true; do
  read B N <<< "$(probe)"
  if [ "${B:-0}" -gt 80 ] && [ "${N:-0}" -ge "$NEED" ]; then lit=$((lit+1)); else lit=0; fi
  [ $lit -ge 8 ] && break
  [ $(( $(date +%s) % 300 )) -lt 8 ] && echo "$(date +%T) waiting for light: brightness ${B:-?}, detected ${N:-?}/$NEED"
  sleep 2
done
echo "$(date +%T) LIT (brightness $B, $N parts seen for ~60 s) - running: $CMD"
bash -c "$CMD"
echo "$(date +%T) command exited $?"

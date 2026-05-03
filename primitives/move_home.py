import sys
import os
import json
import argparse

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from primitives.core.move_xyz import run_ik, output_result
from primitives.shared.config import HOME_POSE, HOME_JOINTS, GRIPPER_CENTER_TOOL_OFFSET

GRASP_DISTANCE_THRESHOLD = 0.06  # 60mm — same as move_to_clear_area / translate_object


GRIPPER_HOLDING_MAX_WIDTH_MM = 30.0  # Above this, gripper is not holding (closed ~2-20mm, half-open ~35mm, open ~97.5mm)


def check_holding_object_sim():
    """Check if the gripper is closed around an object near the gripper center.

    Requires both conditions:
    1. Gripper width < GRIPPER_HOLDING_MAX_WIDTH_MM (gripper is closed/narrow)
    2. Any sim object is within GRASP_DISTANCE_THRESHOLD of the gripper center

    Returns (holding, object_name) or (False, None) if not holding or topics unavailable.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from tf2_msgs.msg import TFMessage
    from std_msgs.msg import Float64
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    import time

    rclpy.init()
    node = Node('move_home_grasp_check')

    ee_data = {}
    object_poses = {}
    gripper_data = {}

    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE, depth=10)

    def ee_cb(msg):
        ee_data['pos'] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        ee_data['quat'] = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                     msg.pose.orientation.z, msg.pose.orientation.w])

    def obj_cb(msg):
        for tf in msg.transforms:
            object_poses[tf.child_frame_id] = tf

    def width_cb(msg):
        gripper_data['width'] = msg.data

    node.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', ee_cb, qos)
    node.create_subscription(TFMessage, '/objects_poses_sim', obj_cb, 10)
    node.create_subscription(Float64, '/gripper_width_sim', width_cb, 10)

    # Spin briefly to collect data
    start = time.time()
    while time.time() - start < 2.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if 'pos' in ee_data and object_poses and 'width' in gripper_data:
            break

    holding = False
    held_name = None

    if 'width' in gripper_data and gripper_data['width'] >= GRIPPER_HOLDING_MAX_WIDTH_MM:
        # Gripper is open — not holding anything regardless of proximity
        pass
    elif 'pos' in ee_data and object_poses:
        ee_rot = R.from_quat(ee_data['quat']).as_matrix()
        gripper_center = ee_data['pos'] + ee_rot @ GRIPPER_CENTER_TOOL_OFFSET

        for name, tf in object_poses.items():
            obj_pos = np.array([tf.transform.translation.x,
                                tf.transform.translation.y,
                                tf.transform.translation.z])
            dist = np.linalg.norm(obj_pos - gripper_center)
            if dist <= GRASP_DISTANCE_THRESHOLD:
                holding = True
                held_name = name
                break

    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass

    return holding, held_name


def _move_home_joint_space(duration_s: float = 5.0) -> tuple:
    """Drive joints directly to HOME_JOINTS via primitives/core/move_joints.py.

    Subprocess pattern (rather than import) — move_joints.py uses sys.exit and
    has its own rclpy lifecycle that doesn't compose cleanly with this caller.
    Returns (success, error_or_none).
    """
    import subprocess
    move_joints_script = os.path.join(_project_root, "primitives", "core", "move_joints.py")
    cmd = [
        sys.executable, move_joints_script, "send",
        "--positions", *(str(j) for j in HOME_JOINTS),
        "--duration", str(duration_s),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_s + 30)
    except subprocess.TimeoutExpired:
        return False, f"move_joints subprocess timeout (> {duration_s + 30}s)"
    if proc.returncode != 0:
        return False, f"move_joints failed: {proc.stderr.strip()[:300]}"
    return True, None


def main(args=None):
    parser = argparse.ArgumentParser(description='Move robot to home position')
    parser.add_argument('--mode', type=str, default='real', choices=['sim', 'real'])
    parser.add_argument('--joint-space', action='store_true',
                        help="Use HOME_JOINTS joint-space target instead of HOME_POSE Cartesian. "
                             "Predictable joint config — useful when the Cartesian IK picks an "
                             "awkward seed (e.g. before F/T calibration).")
    parser.add_argument('--duration', type=float, default=5.0,
                        help="Joint-space mode only: trajectory duration in seconds (default 5.0)")
    parsed = parser.parse_args()

    # In sim mode, check if the robot is holding an object before moving home
    if parsed.mode == 'sim':
        holding, held_name = check_holding_object_sim()
        if holding:
            result = {"result": "failure",
                      "error": f"Cannot move home while holding '{held_name}'. Release the object first."}
            output_result(result)
            sys.exit(1)

    if parsed.joint_space:
        success, error = _move_home_joint_space(duration_s=parsed.duration)
    else:
        target_position = HOME_POSE[0:3]
        target_rpy = HOME_POSE[3:6]
        success, error = run_ik(target_position, target_rpy,
                                reference_frame='ee')

    result = {"result": "success" if success else "failure"}
    if not success and error:
        result["error"] = error

    output_result(result)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

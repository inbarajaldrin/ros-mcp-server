#!/usr/bin/env python3
"""
Scan Workspace - Scans workspace at fixed height to locate object

The algorithm:
1. Only works in real mode (raises error if sim mode is used)
2. Defines a predefined scanning path across x,y at a fixed z height
3. Subscribes to /objects_poses_real to detect object
4. Moves along the path, checking for the object at each waypoint
5. Stops as soon as the object is detected
6. Exits
"""

import sys
import os
import json

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import numpy as np
import argparse
import time
from primitives.shared.ik import IKSolver, IKSolverConfig, rpy_to_matrix
from primitives.shared.velocity_profiles import single_point, compute_duration

# Configuration
OBJECT_TOPIC_REAL = "/objects_poses_real"
ACTION_SERVER = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# Scanning parameters
SCAN_HEIGHT = 0.35  # Fixed Z height for scanning (in meters)
SCAN_X_MIN = -0.4   # Minimum X position
SCAN_X_MAX = 0.4    # Maximum X position
SCAN_Y_MIN = -0.5   # Minimum Y position
SCAN_Y_MAX = -0.3   # Maximum Y position
SCAN_STEP = 0.1     # Step size for scanning grid (in meters)
MOVEMENT_DURATION = 3.0  # Duration for each movement (in seconds)

# Fixed orientation (face down gripper)
FIXED_ROLL = 0.0
FIXED_PITCH = 180.0
FIXED_YAW = 0.0


def generate_scan_path(current_x, current_y):
    """
    Generate a Boustrophedon scanning path starting from current position.
    Scans with fixed X first (moves in Y direction), then moves to next X column.
    Uses zigzag pattern for Y movement.
    
    Args:
        current_x: Current X position of the robot
        current_y: Current Y position of the robot
    
    Returns:
        List of [x, y, z] positions to scan
    """
    path = []
    # np.arange is exclusive of stop, so add tiny epsilon to include the max value
    epsilon = 1e-9
    x_positions = np.arange(SCAN_X_MIN, SCAN_X_MAX + epsilon, SCAN_STEP)
    y_positions = np.arange(SCAN_Y_MIN, SCAN_Y_MAX + epsilon, SCAN_STEP)
    
    # Find the closest X position to start from
    start_x_idx = np.argmin(np.abs(x_positions - current_x))
    start_x = x_positions[start_x_idx]
    
    # Find the closest Y position to start from
    start_y_idx = np.argmin(np.abs(y_positions - current_y))
    start_y = y_positions[start_y_idx]
    
    # Reorder X positions to start from closest to current position
    x_positions_ordered = list(x_positions[start_x_idx:]) + list(x_positions[:start_x_idx])
    
    # Boustrophedon pattern: for each X (column), scan in Y direction (zigzag)
    for i, x in enumerate(x_positions_ordered):
        # Determine Y direction for this column (zigzag: alternate up/down)
        if i % 2 == 0:
            # Even columns: scan from Y_MIN to Y_MAX
            y_ordered = list(y_positions)
        else:
            # Odd columns: scan from Y_MAX to Y_MIN
            y_ordered = list(reversed(y_positions))
        
        # For the first column (starting column), start from current Y position
        if x == start_x:
            # Find where current_y is in the ordered list
            try:
                current_y_idx = y_ordered.index(start_y)
                # Reorder to start from current Y
                y_ordered = y_ordered[current_y_idx:] + y_ordered[:current_y_idx]
            except ValueError:
                # If current_y not exactly in list, use closest
                closest_y_idx = np.argmin(np.abs(np.array(y_ordered) - current_y))
                y_ordered = y_ordered[closest_y_idx:] + y_ordered[:closest_y_idx]
        
        # Add all Y positions for this X column
        for y in y_ordered:
            path.append([x, y, SCAN_HEIGHT])
    
    return path


class ScanWorkspace(Node):
    def __init__(self, object_name):
        super().__init__('scan_workspace')

        self.object_name = object_name
        self.object_found = False
        self.current_poses = {}
        self.current_ee_position = None
        self.ee_pose_received = False
        self.scan_path = []
        self.current_waypoint_index = 0
        self.trajectory_completed = False
        self.trajectory_success = False
        self.should_exit = False  # Flag to signal main loop to exit
        self.error_message = None  # Track error for JSON output
        
        # Joint state tracking
        self.current_joint_angles = None
        self.joint_angles_received = False

        # Action client for trajectory execution
        self.action_client = ActionClient(self, FollowJointTrajectory, ACTION_SERVER)
        
        # Subscribe to current EE pose to get starting position
        ee_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,  # Match UR driver publisher QoS
            depth=10
        )
        self.ee_pose_sub = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback,
            ee_qos
        )
        
        # Subscribe to joint states for duration computation
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Subscribe to object poses (real mode only)
        self.object_sub = self.create_subscription(
            TFMessage,
            OBJECT_TOPIC_REAL,
            self.object_callback,
            10
        )
        
        # Timer to check for object and move to next waypoint
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.movement_in_progress = False
        
        self.get_logger().info(f"Scanning workspace for object: {object_name}")
        self.get_logger().info(f"Waiting for current EE pose to generate scan path...")
        
        # Start with first waypoint
        self.trajectory_completed = True  # Allow first movement to start
    
    def joint_state_callback(self, msg):
        """Callback for joint state data"""
        if len(msg.name) >= 6 and len(msg.position) >= 6:
            joint_dict = dict(zip(msg.name, msg.position))
            positions = []
            for name in JOINT_NAMES:
                if name in joint_dict:
                    positions.append(joint_dict[name])
                else:
                    return
            self.current_joint_angles = np.array(positions)
            self.joint_angles_received = True

    def ee_pose_callback(self, msg):
        """Callback for current end-effector pose"""
        if not self.ee_pose_received:
            self.current_ee_position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            self.ee_pose_received = True
            # Generate scan path based on current position
            self.scan_path = generate_scan_path(
                self.current_ee_position[0],
                self.current_ee_position[1]
            )
            self.get_logger().info(f"Current EE position: [{self.current_ee_position[0]:.3f}, {self.current_ee_position[1]:.3f}, {self.current_ee_position[2]:.3f}]")
            self.get_logger().info(f"Scan path contains {len(self.scan_path)} waypoints")
            self.get_logger().info(f"Scanning at fixed height: {SCAN_HEIGHT}m")
            self.get_logger().info(f"Using Boustrophedon pattern: fixed X first, zigzag in Y direction")
    
    def object_callback(self, msg):
        """Callback for object poses - check if target object is detected"""
        for transform in msg.transforms:
            frame_id = transform.child_frame_id
            self.current_poses[frame_id] = transform
            
            if frame_id == self.object_name:
                if not self.object_found:
                    self.object_found = True
                    self.get_logger().info(f"Object '{self.object_name}' detected at waypoint {self.current_waypoint_index + 1}/{len(self.scan_path)}")
                    # Stop moving - cancel any ongoing trajectory
                    self.cancel_current_movement()
    
    def cancel_current_movement(self):
        """Cancel any ongoing trajectory movement"""
        # Note: ActionClient doesn't have a direct cancel method in rclpy
        # The movement will complete naturally, but we won't move to next waypoint
        pass
    
    def execute_trajectory(self, position):
        """
        Execute a trajectory to move to the specified position.
        
        Args:
            position: [x, y, z] target position
            
        Returns:
            True if trajectory was sent successfully, False otherwise
        """
        # Compute IK for target position
        rpy = [FIXED_ROLL, FIXED_PITCH, FIXED_YAW]
        target_pose = np.eye(4)
        target_pose[:3, 3] = np.array(position)
        target_pose[:3, :3] = rpy_to_matrix(rpy)
        q6 = -(np.mod(rpy[2] + 180, 360) - 180)
        q_guess = np.radians([85, -80, 90, -90, -90, q6])
        solver = IKSolver(IKSolverConfig())
        joint_angles = solver.solve(seeds=[q_guess], target_pose=target_pose, perturbations=5, dx=0.001)
        
        if joint_angles is None:
            self.get_logger().warn(f"Motion planning failed: scan waypoint at {position} is outside the reachable workspace, skipping")
            return False

        # Compute duration dynamically based on joint distance
        duration = MOVEMENT_DURATION  # fallback
        if self.current_joint_angles is not None:
            joint_dist = float(np.max(np.abs(np.array(joint_angles) - self.current_joint_angles)))
            duration = compute_duration(joint_distance=joint_dist, profile='s_curve')
            self.get_logger().info(f"Duration: {duration:.2f}s (joint={joint_dist:.2f}rad)")

        # Create trajectory point via single_point profile
        profile = single_point(joint_angles, duration)
        positions, velocities, t = profile[0]

        # Create trajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = JOINT_NAMES
        traj_msg.points = [JointTrajectoryPoint(
            positions=positions,
            velocities=velocities,
            time_from_start=Duration(sec=int(t))
        )]
        
        # Create goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj_msg
        goal.goal_time_tolerance = Duration(sec=1)
        
        # Reset flags
        self.trajectory_completed = False
        self.trajectory_success = False
        self.movement_in_progress = True
        
        # Send goal
        self.get_logger().info(f"Moving to waypoint {self.current_waypoint_index + 1}/{len(self.scan_path)}: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        send_goal_future = self.action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)
        
        return True
    
    def goal_response_callback(self, future):
        """Callback when goal is accepted/rejected"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            self.trajectory_completed = True
            self.trajectory_success = False
            return
        
        self.get_logger().debug("Trajectory goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        """Callback when trajectory execution completes"""
        result = future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.trajectory_success = True
            self.get_logger().debug("Trajectory completed successfully")
        else:
            self.trajectory_success = False
            self.get_logger().warn(f"Trajectory completed with error code: {result.error_code}")
        
        self.trajectory_completed = True
    
    def timer_callback(self):
        """Timer callback to check for object and move to next waypoint"""
        # Wait for path to be generated
        if not self.ee_pose_received or len(self.scan_path) == 0:
            return
        
        # If object found, exit
        if self.object_found:
            self.get_logger().info("Object found! Stopping scan and exiting.")
            self.timer.cancel()
            self.should_exit = True
            return
        
        # If currently moving, wait for trajectory to complete
        if self.movement_in_progress:
            if not self.trajectory_completed:
                return
            # Movement just completed
            self.movement_in_progress = False
        
        # Check if we've completed all waypoints
        if self.current_waypoint_index >= len(self.scan_path):
            self.get_logger().warn(f"Scan complete: Object '{self.object_name}' not found after scanning {len(self.scan_path)} waypoints")
            self.timer.cancel()
            self.should_exit = True
            return
        
        # Move to next waypoint
        waypoint = self.scan_path[self.current_waypoint_index]
        if self.execute_trajectory(waypoint):
            self.movement_in_progress = True
            self.current_waypoint_index += 1
        else:
            # Motion planning failed, skip this waypoint and try next
            self.get_logger().warn(f"Skipping waypoint {self.current_waypoint_index + 1} — position is outside the reachable workspace")
            self.current_waypoint_index += 1
            # Don't set movement_in_progress, allow next waypoint to be tried immediately
    
    def run(self):
        """Run the scanning process"""
        # Wait for action server
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.error_message = "UR robot driver isn't running"
            self.get_logger().error("Action server not available. Exiting.")
            return False

        # Start scanning
        self.get_logger().info("Starting workspace scan...")
        # The timer will handle movement to waypoints

        return True

    def output_result_json(self, movement_type="scan_workspace"):
        """Output result in JSON format for MCP server"""
        if self.error_message:
            # Failure format
            result = {
                "result": "failure",
                "mode": "real",
                "object_name": self.object_name,
                "error": self.error_message
            }
        else:
            # Success format
            result = {
                "result": "success",
                "mode": "real",
                "object_name": self.object_name
            }

        output_result(result)


# ---------------------------------------------------------------------------
# Survey mode (2026-10-13): view one FIXED target (base1) from many poses and record what
# the camera reports from each. A perfect camera/hand-eye gives the same base1 pose from
# every viewpoint; the spread is the error. The dataset also feeds a hand-eye fit.
#
# Motion is NOT the scan path above: that uses fixed-seed IK + a single joint-space point,
# which can pick another arm configuration (it swung the arm on 2026-10-07 via move_xyz).
# Survey moves are Cartesian waypoints seeded from the CURRENT joints (as move_to_safe_height),
# flange never below SAFE_HEIGHT, tool axis aimed at the target, tilt <= SURVEY_MAX_TILT_DEG.
# ---------------------------------------------------------------------------
SURVEY_HEIGHTS = (0.34, 0.42, 0.50)          # flange z (m); SAFE_HEIGHT is 0.32
SURVEY_RING = {0.42: 0.10, 0.50: 0.15}       # flange xy offset from the target per height (m)
SURVEY_YAWS_DEG = (-40.0, 40.0)              # extra wrist yaws at the middle height, centre
SURVEY_MAX_TILT_DEG = 20.0
SURVEY_MAX_JOINT_STEP = 1.5                  # rad, any joint, per move — refuse bigger
SURVEY_RECORD_S = 3.0
SURVEY_MIN_RADIUS = 0.25                     # m, flange distance from the base z axis — no self-collision
                                             # check in the Cartesian planner, so stay clear of the shoulder


SURVEY_WIDE_RING = {0.34: 0.06, 0.42: 0.10, 0.50: 0.15}
SURVEY_WIDE_YAWS_DEG = (-50.0, 0.0, 50.0)    # ordered so consecutive wrist steps stay < 1 rad
# Close pass (operator-approved 2026-10-13, flat board + <5 cm objects only): grasp-range views.
# Below SAFE_HEIGHT on purpose; its own hard floor. Gripper centre ~91 mm above the table at 0.24.
SURVEY_CLOSE_HEIGHTS = (0.28, 0.24)
SURVEY_CLOSE_RING = {0.28: 0.04, 0.24: 0.03}
SURVEY_CLOSE_FLOOR = 0.235


def survey_viewpoints(target, R_start, wide=False, close=False):
    """[(flange_pos, R_flange, label)] aiming the tool z axis at `target`.
    wide=True (calibration target): wrist yaws at every height + rings at every height,
    and a +45 deg yaw copy of the middle ring — rotation diversity for a hand-eye solve."""
    from scipy.spatial.transform import Rotation as Rot
    x_ref = R_start[:, 0]
    out = []
    heights = SURVEY_CLOSE_HEIGHTS if close else SURVEY_HEIGHTS
    rings = SURVEY_CLOSE_RING if close else (SURVEY_WIDE_RING if wide else SURVEY_RING)
    for h in heights:
        if close and h < SURVEY_CLOSE_FLOOR:
            raise ValueError(f"close survey height {h} below floor {SURVEY_CLOSE_FLOOR}")
        offs = [(0.0, 0.0)]
        r = rings.get(h)
        if r:
            offs += [(r, 0.0), (0.0, r), (-r, 0.0), (0.0, -r)]
        for dx, dy in offs:
            p = np.array([target[0] + dx, target[1] + dy, h])
            z = target - p
            z /= np.linalg.norm(z)
            x = x_ref - z * (x_ref @ z)
            x /= np.linalg.norm(x)
            R = np.column_stack([x, np.cross(z, x), z])
            if close:
                yaws = SURVEY_WIDE_YAWS_DEG if dx == dy == 0.0 else (0.0,)
            elif wide:
                yaws = SURVEY_WIDE_YAWS_DEG if dx == dy == 0.0 else ((0.0, 45.0) if h == SURVEY_HEIGHTS[1] else (0.0,))
            else:
                yaws = (0.0,) + (SURVEY_YAWS_DEG if (h == SURVEY_HEIGHTS[1] and dx == dy == 0.0) else ())
            for yaw in yaws:
                Ry = R @ Rot.from_euler('z', yaw, degrees=True).as_matrix()
                out.append((p, Ry, f"h{h:.2f}_dx{dx:+.2f}_dy{dy:+.2f}_yaw{yaw:+.0f}"))
    return out


class Survey(Node):
    def __init__(self, target_name, image_topic=None):
        super().__init__('scan_workspace_survey')
        self.target_name = target_name
        self.joints = None
        self.tcp = None
        self.frames = {}
        self.rec = None
        self.action_client = ActionClient(self, FollowJointTrajectory, ACTION_SERVER)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self._tcp_cb,
                                 QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.VOLATILE, depth=10))
        self.create_subscription(TFMessage, OBJECT_TOPIC_REAL, self._obj_cb, 10)
        # pre-snap poses (localizer 2026-10-13); absent on older localizers -> just empty
        self.create_subscription(TFMessage, OBJECT_TOPIC_REAL + '_raw', self._raw_cb, 10)
        self.image = None
        if image_topic:
            from sensor_msgs.msg import Image
            self.create_subscription(Image, image_topic, lambda m: setattr(self, 'image', (time.time(), m)), 2)

    def _joint_cb(self, msg):
        d = dict(zip(msg.name, msg.position))
        if all(n in d for n in JOINT_NAMES):
            self.joints = np.array([d[n] for n in JOINT_NAMES])

    def _tcp_cb(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.tcp = (time.time(), [p.x, p.y, p.z, o.x, o.y, o.z, o.w])
        if self.rec is not None:
            self.rec['tcp'].append(self.tcp[1])

    def _obj_cb(self, msg):
        now = time.time()
        for t in msg.transforms:
            tr, q = t.transform.translation, t.transform.rotation
            row = [now, tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w]
            self.frames[t.child_frame_id] = row
            if self.rec is not None:
                self.rec['objects'].setdefault(t.child_frame_id, []).append(row)

    def _raw_cb(self, msg):
        if self.rec is None:
            return
        now = time.time()
        for t in msg.transforms:
            tr, q = t.transform.translation, t.transform.rotation
            self.rec['objects_raw'].setdefault(t.child_frame_id, []).append(
                [now, tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w])

    def spin_for(self, s):
        end = time.time() + s
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_still(self, window=0.5, tol=0.0002, timeout=6.0):
        hist, end = [], time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.tcp is None:
                continue
            now = time.time()
            hist = [h for h in hist + [(now, np.array(self.tcp[1][:3]))] if now - h[0] <= window]
            if now - hist[0][0] >= 0.95 * window:
                pts = np.array([h[1] for h in hist])
                if np.max(np.linalg.norm(pts - pts.mean(0), axis=1)) < tol:
                    return True
        return False

    def move_to(self, pos, R):
        """Cartesian move from the current joints. Returns (ok, reason)."""
        from primitives.shared.ik import compute_cartesian_waypoints_ik, forward_kinematics, dh_params
        from primitives.shared.velocity_profiles import s_curve_profile
        q0 = self.joints.copy()
        wps = compute_cartesian_waypoints_ik(q0, pos[2], num_waypoints=60, target_pos=pos, target_orientation=R)
        if wps is None:
            return False, "no Cartesian IK path"
        T = forward_kinematics(dh_params, wps[-1])
        if np.linalg.norm(T[:3, 3] - pos) > 0.002:
            return False, f"IK end error {1000 * np.linalg.norm(T[:3, 3] - pos):.1f} mm"
        step = float(np.max(np.abs(np.array(wps[-1]) - q0)))
        if step > SURVEY_MAX_JOINT_STEP:
            return False, f"joint step {step:.2f} rad > {SURVEY_MAX_JOINT_STEP}"
        cart = float(np.linalg.norm(pos - forward_kinematics(dh_params, q0)[:3, 3]))
        dur = compute_duration(joint_distance=step, cartesian_distance=cart, profile='s_curve')
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        for p, v, t in s_curve_profile([q0] + list(wps), dur):
            traj.points.append(JointTrajectoryPoint(positions=p, velocities=v, time_from_start=Duration(
                sec=int(t), nanosec=int((t - int(t)) * 1e9))))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance = Duration(sec=1)
        fut = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return False, "goal rejected"
        res = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res, timeout_sec=dur + 15.0)   # never cancel: runs to completion
        if res.result() is None or res.result().result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            return False, "trajectory did not complete"
        return True, f"{dur:.1f}s"


def run_survey(target_name, out_path, target_xyz=None, image_dir=None, wide=False, close=False):
    from primitives.shared.ik import forward_kinematics, dh_params
    rclpy.init()
    node = Survey(target_name, os.environ.get('SURVEY_IMAGE_TOPIC', '/intel_camera_rgb_raw') if image_dir else None)
    if image_dir:
        import cv2
        from cv_bridge import CvBridge
        bridge = CvBridge()
        os.makedirs(image_dir, exist_ok=True)
    result = {"result": "failure", "mode": "real", "movement_type": "survey", "target": target_name}
    try:
        if not node.action_client.wait_for_server(timeout_sec=10.0):
            result["error"] = "UR robot driver isn't running"
            return result
        node.spin_for(2.0)
        if node.joints is None or (target_xyz is None and target_name not in node.frames):
            result["error"] = f"no joints or '{target_name}' not visible from the start pose"
            return result
        target = np.array(target_xyz if target_xyz is not None else node.frames[target_name][1:4], dtype=float)
        T0 = forward_kinematics(dh_params, node.joints)
        views = survey_viewpoints(target, T0[:3, :3], wide=wide, close=close)
        node.get_logger().info(f"Survey of {target_name} at {np.round(target, 4).tolist()}: {len(views)} viewpoints")
        poses = []
        for i, (pos, R, label) in enumerate(views):
            tilt = float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1))))
            if np.hypot(pos[0], pos[1]) < SURVEY_MIN_RADIUS:
                node.get_logger().warn(f"[{i + 1}/{len(views)}] {label}: flange {np.hypot(pos[0], pos[1]):.3f} m "
                                       f"from the base axis < {SURVEY_MIN_RADIUS}, skipped")
                continue
            if tilt > SURVEY_MAX_TILT_DEG + 0.5:
                node.get_logger().warn(f"[{i + 1}/{len(views)}] {label}: tilt {tilt:.1f} deg > limit, skipped")
                continue
            ok, why = node.move_to(pos, R)
            node.get_logger().info(f"[{i + 1}/{len(views)}] {label} tilt {tilt:.1f} deg: {'moved ' + why if ok else 'SKIPPED ' + why}")
            if not ok:
                if why in ("goal rejected", "trajectory did not complete"):
                    result["error"] = f"motion failed at {label}: {why}"
                    break               # fail closed: stop the survey where the arm is
                continue
            still = node.wait_still()
            t_settled = time.time()
            node.rec = {'tcp': [], 'objects': {}, 'objects_raw': {}}
            frames = []
            for k in range(3):                       # 3 raw frames across the record window
                node.spin_for(SURVEY_RECORD_S / 3)
                # Only frames RECEIVED after the arm settled: a frame captured before the move ended
                # (the localizer republishes at ~3 Hz) put a 22.8 px outlier in the 2026-10-13 data.
                if image_dir and node.image is not None and node.tcp is not None and node.image[0] > t_settled + 0.4:
                    fn = f"{image_dir}/{i:02d}_{label}_{k}.png"
                    cv2.imwrite(fn, bridge.imgmsg_to_cv2(node.image[1], 'bgr8'))
                    frames.append({"file": fn, "tcp": node.tcp[1], "joints": node.joints.tolist()})
            rec, node.rec = node.rec, None
            tgt = np.array(rec['objects'].get(target_name, []))
            poses.append({"label": label, "tilt_deg": tilt, "still": still, "joints": node.joints.tolist(), "frames": frames,
                          "flange_cmd": pos.tolist(), "tcp_mean": np.mean(rec['tcp'], 0).tolist() if rec['tcp'] else None,
                          "n": int(len(tgt)), "target_median": np.median(tgt[:, 1:4], 0).tolist() if len(tgt) else None,
                          "objects": rec['objects'], "objects_raw": rec['objects_raw']})
            raw = np.array(rec['objects_raw'].get(target_name, []))
            node.get_logger().info(f"    {target_name}: n={len(tgt)}" + (
                f" median {np.round(1000 * np.median(tgt[:, 1:4], 0), 1).tolist()} mm" if len(tgt) else " NOT SEEN")
                + (f" | raw n={len(raw)} median {np.round(1000 * np.median(raw[:, 1:4], 0), 1).tolist()} mm" if len(raw) else ""))
        seen = [p for p in poses if p["target_median"]]
        if seen:
            m = np.array([p["target_median"] for p in seen])
            spread = 1000 * np.linalg.norm(m - np.median(m, 0), axis=1)
            result.update(n_poses=len(poses), n_seen=len(seen), spread_max_mm=float(spread.max()),
                          spread_median_mm=float(np.median(spread)))
        with open(out_path, 'w') as f:
            json.dump({"target": target_name, "target_start": target.tolist(), "poses": poses}, f)
        result["dataset"] = out_path
        if "error" not in result:
            result["result"] = "success"
        return result
    finally:
        node.action_client.destroy()
        node.destroy_node()
        rclpy.shutdown()


def output_result(result):
    """Output JSON result with markers"""
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


def main(args=None):
    parser = argparse.ArgumentParser(description='Scan Workspace - Locate object by scanning at fixed height')
    parser.add_argument('--object-name', type=str, required=True, help='Name of the object to locate')
    parser.add_argument('--mode', type=str, default='real', choices=['real'],
                       help='Mode: only "real" is supported (default: "real")')
    parser.add_argument('--survey', action='store_true',
                        help='Survey mode: view the (fixed, already visible) object from many poses '
                             'and record what the camera reports; writes a JSON dataset.')
    parser.add_argument('--out', type=str, default=None, help='Survey dataset path (JSON)')
    parser.add_argument('--plan-only', action='store_true', help='Survey: print the viewpoints, do not move')
    parser.add_argument('--target-xyz', type=float, nargs=3, default=None,
                        help='Survey: aim at this point (robot base frame, m) instead of the object\'s pose')
    parser.add_argument('--image-dir', type=str, default=None, help='Survey: save 3 raw frames per pose here')
    parser.add_argument('--wide', action='store_true', help='Survey: calibration viewpoint set (more wrist yaws)')
    parser.add_argument('--close', action='store_true',
                        help='Survey: grasp-range pass, flange 0.28/0.24 m (BELOW safe height; flat target, clear table only)')

    args = parser.parse_args()

    if args.survey:
        if args.plan_only:
            from primitives.shared.config import DEFAULT_BASE_POSITION
            tgt = np.array(args.target_xyz or DEFAULT_BASE_POSITION, dtype=float)
            R0 = rpy_to_matrix([FIXED_ROLL, FIXED_PITCH, FIXED_YAW])
            for pos, R, label in survey_viewpoints(tgt, R0, wide=args.wide, close=args.close):
                tilt = np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))
                print(f"{label:32s} flange {np.round(pos, 3).tolist()} tilt {tilt:4.1f} deg")
            sys.exit(0)
        out = args.out or time.strftime(f"/tmp/survey_{args.object_name}_%Y%m%d_%H%M%S.json")
        result = run_survey(args.object_name, out, args.target_xyz, args.image_dir, args.wide, args.close)
        output_result(result)
        sys.exit(0 if result["result"] == "success" else 1)

    # Enforce real mode only
    if args.mode != 'real':
        # Create minimal node for JSON output before exiting
        result = {
            "result": "failure",
            "mode": "real",
            "object_name": args.object_name,
            "error": "Only real mode is supported for workspace scanning"
        }
        output_result(result)
        sys.exit(1)

    rclpy.init()
    node = ScanWorkspace(object_name=args.object_name)

    try:
        if not node.run():
            output_result_json_from_node(node)
            node.action_client.destroy()
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

        # Spin until object found or scan complete
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)

        # Check result
        if node.object_found:
            node.get_logger().info("Scan workspace: SUCCESS - Object located")
            output_result_json_from_node(node)
            node.action_client.destroy()
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(0)
        else:
            node.error_message = "Object not found after scanning workspace"
            node.get_logger().error("Scan workspace: FAILED - Object not found")
            output_result_json_from_node(node)
            node.action_client.destroy()
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
        if rclpy.ok():
            node.action_client.destroy()
            node.destroy_node()
            rclpy.shutdown()
        sys.exit(1)
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
        node.error_message = f"Unexpected error: {e}"
        import traceback
        traceback.print_exc()
        try:
            output_result_json_from_node(node)
        except:
            pass
        if rclpy.ok():
            node.action_client.destroy()
            node.destroy_node()
            rclpy.shutdown()
        sys.exit(1)
    finally:
        if rclpy.ok():
            try:
                node.action_client.destroy()
                node.destroy_node()
                rclpy.shutdown()
            except:
                pass


def output_result_json_from_node(node):
    """Helper function to call output_result_json on the node"""
    try:
        node.output_result_json()
    except:
        pass


if __name__ == '__main__':
    main()


"""
Snap EE to closest cardinal orientation in each face direction.

For each of the 8 face directions, finds the roll variant closest to the
current EE orientation, solves IK keeping position fixed, and optionally
executes the selected one.

Usage:
    python3 primitives/utils/snap_to_cardinal.py
"""

import sys
import os
import json
import time

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from primitives.rotate_object import ExtendedCardinalOrientations
from primitives.shared.ik import IKSolver, IKSolverConfig, forward_kinematics, dh_params
from primitives.shared.velocity_profiles import single_point
from primitives.shared.config import TABLE_HEIGHT, TABLE_COLLISION_MARGIN_SIDEWAYS, TABLE_COLLISION_MARGIN_FACEDOWN
from primitives.shared.collision import check_collision_with_table, check_self_collision, check_trajectory_collision


class SnapToCardinal(Node):
    def __init__(self):
        super().__init__('snap_to_cardinal')
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        self.action_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.ee_pose_received = False
        self.ee_position = None
        self.ee_quat = None
        self.current_joint_angles = None
        self.joint_angles_received = False
        self.operation_success = False
        self.operation_complete = False

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.create_subscription(
            PoseStamped, '/tcp_pose_broadcaster/pose',
            self.ee_pose_callback, qos_profile
        )
        self.create_subscription(
            JointState, '/joint_states',
            self.joint_state_callback, 10
        )

        self.action_client.wait_for_server()
        self.run()

    def ee_pose_callback(self, msg):
        self.ee_position = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        ])
        self.ee_quat = np.array([
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w
        ])
        self.ee_pose_received = True

    def joint_state_callback(self, msg):
        if len(msg.name) == 6 and len(msg.position) == 6:
            joint_dict = dict(zip(msg.name, msg.position))
            ordered = [joint_dict.get(n) for n in self.joint_names]
            if all(v is not None for v in ordered):
                self.current_joint_angles = np.array(ordered)
                self.joint_angles_received = True

    def wait_for_data(self):
        while rclpy.ok() and (not self.ee_pose_received or not self.joint_angles_received):
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.ee_pose_received and self.joint_angles_received

    # --- Collision checks ---

    def check_table_collision(self, joint_angles, margin=TABLE_COLLISION_MARGIN_SIDEWAYS):
        return check_collision_with_table(joint_angles, z_threshold=TABLE_HEIGHT - margin)

    def collision_checker(self, joint_angles, margin=TABLE_COLLISION_MARGIN_SIDEWAYS):
        return self.check_table_collision(joint_angles, margin=margin) or check_self_collision(joint_angles)

    # --- World-frame direction checks ---

    def check_ee_facing_robot(self, joint_angles, y_threshold=0.5):
        """Returns True if tool Z points toward +Y world (bad)."""
        T = forward_kinematics(dh_params, joint_angles)
        tool_z = T[:3, :3][:, 2]
        return tool_z[1] > y_threshold

    def check_camera_direction(self, joint_angles, x_threshold=-0.5):
        """Returns True if tool Z points toward -X world (camera body extends toward robot)."""
        T = forward_kinematics(dh_params, joint_angles)
        tool_z = T[:3, :3][:, 2]
        return tool_z[0] < x_threshold

    # --- Main logic ---

    def run(self):
        if not self.wait_for_data():
            self.get_logger().error("Failed to get EE pose / joint states")
            self.operation_complete = True
            return

        R_current = Rot.from_quat(self.ee_quat).as_matrix()
        ee_pos = self.ee_position.copy()
        seed_joints = self.current_joint_angles.copy()

        self.get_logger().info(
            f"Current EE position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]"
        )

        # Get all 24 cardinals and group by face direction
        all_cardinals = ExtendedCardinalOrientations.get_all_extended_cardinals()

        # Group by face: extract face name (everything before _rollN)
        face_groups = {}
        for name, quat in all_cardinals.items():
            parts = name.rsplit('_roll', 1)
            face = parts[0]  # e.g. "face_down"
            R_card = Rot.from_quat(quat).as_matrix()
            dist = ExtendedCardinalOrientations.rotation_matrix_distance(R_current, R_card)
            if face not in face_groups or dist < face_groups[face][1]:
                face_groups[face] = (name, dist, R_card)

        # Solve IK for each face direction's best roll variant
        results = []
        for face, (card_name, rot_dist, R_card) in face_groups.items():
            margin = TABLE_COLLISION_MARGIN_FACEDOWN if face == 'face_down' else TABLE_COLLISION_MARGIN_SIDEWAYS
            target_pose = np.eye(4)
            target_pose[:3, 3] = ee_pos
            target_pose[:3, :3] = R_card

            joint_bounds = [
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-2*np.pi, 2*np.pi),
            ]
            solver = IKSolver(IKSolverConfig(joint_bounds=joint_bounds))
            solution = solver.solve(
                seeds=[seed_joints.copy()],
                target_pose=target_pose,
                collision_checker=lambda ja, m=margin: self.collision_checker(ja, margin=m),
                perturbations=3,
                dx=0.001,
            )

            if solution is None:
                results.append({
                    'name': card_name, 'dist': rot_dist,
                    'status': 'IK_FAIL', 'cost': None, 'joints': None,
                    'camera': None, 'ee_facing': None,
                })
                continue

            ik_cost = solver._best_result.cost
            camera_bad = self.check_camera_direction(solution)
            ee_bad = self.check_ee_facing_robot(solution)
            traj_collision = check_trajectory_collision(seed_joints, solution, z_threshold=TABLE_HEIGHT - margin)

            if camera_bad:
                status = 'REJECTED (tool Z toward -X world)'
            elif ee_bad:
                status = 'REJECTED (tool Z toward +Y world)'
            elif traj_collision:
                status = 'REJECTED (trajectory collision)'
            else:
                status = 'PASS'

            results.append({
                'name': card_name, 'dist': rot_dist, 'status': status,
                'cost': ik_cost, 'joints': solution,
                'camera': 'FAIL' if camera_bad else 'ok',
                'ee_facing': 'FAIL' if ee_bad else 'ok',
            })

        # Sort by rotation distance
        results.sort(key=lambda r: r['dist'])

        # Print table
        print()
        print(f" {'#':>2}  {'Cardinal':<30} {'Dist':>6}  {'IK Cost':>8}  {'Camera':>6}  {'EE':>4}  Status")
        print(f" {'':->2}  {'':->30} {'':->6}  {'':->8}  {'':->6}  {'':->4}  {'':->30}")
        selectable = []
        for i, r in enumerate(results):
            num = i + 1
            cost_str = f"{r['cost']:.4f}" if r['cost'] is not None else "  N/A "
            cam_str = r['camera'] if r['camera'] else " N/A "
            ee_str = r['ee_facing'] if r['ee_facing'] else "N/A"
            print(f" {num:>2}  {r['name']:<30} {r['dist']:5.1f}°  {cost_str:>8}  {cam_str:>6}  {ee_str:>4}  {r['status']}")
            if r['status'] == 'PASS':
                selectable.append((num, r))
        print()

        if not selectable:
            self.get_logger().error("No PASS candidates to execute.")
            self.operation_complete = True
            return

        # Interactive selection
        print(f"Selectable: {', '.join(str(n) for n, _ in selectable)}")
        try:
            choice = int(input("Pick a number to execute (0 to cancel): "))
        except (ValueError, EOFError):
            choice = 0

        if choice == 0:
            self.get_logger().info("Cancelled.")
            self.operation_success = True
            self.operation_complete = True
            return

        # Find selected result
        selected = None
        for num, r in selectable:
            if num == choice:
                selected = r
                break

        if selected is None:
            self.get_logger().error(f"Invalid selection: {choice}")
            self.operation_complete = True
            return

        self.get_logger().info(f"Executing: {selected['name']} (dist={selected['dist']:.1f}°)")
        self.execute_trajectory(selected['joints'])

    def execute_trajectory(self, target_joints):
        profile = single_point(target_joints, 5.0)
        positions, velocities, t = profile[0]

        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        traj.points = [JointTrajectoryPoint(
            positions=positions,
            velocities=velocities,
            time_from_start=Duration(sec=int(t)),
        )]
        goal.trajectory = traj
        goal.goal_time_tolerance = Duration(sec=1)

        self.get_logger().info("Sending trajectory...")
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response)

    def goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            self.operation_complete = True
            return
        self.get_logger().info("Goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        result = future.result()
        if result.status == 4:
            self.get_logger().info("Movement completed successfully")
            self.operation_success = True
        else:
            self.get_logger().error(f"Movement failed: status={result.status}")
        self.operation_complete = True


def main():
    rclpy.init()
    node = SnapToCardinal()
    try:
        start_time = time.time()
        while rclpy.ok() and not node.operation_complete:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - start_time > 30.0:
                node.get_logger().error("Timeout")
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass

    sys.exit(0 if node.operation_success else 1)


if __name__ == '__main__':
    main()

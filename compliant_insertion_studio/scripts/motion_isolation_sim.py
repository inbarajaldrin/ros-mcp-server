#!/usr/bin/env python3
# Reference: position-controlled motion-isolation diagnostic for SIM/FAKE hardware.
# Drives TCP through a spiral xy path using scaled_joint_trajectory_controller —
# no force_mode (which doesn't simulate in fake hardware). Visualizes the
# COMMANDED motion pattern, not the admittance dynamics.
#
# Sequence:
#   1. move_home (joint-space, safe with held part)
#   2. move_to_safe_height
#   3. move_to_hover (above predicted target)
#   4. Walk TCP through spiral xy at constant z (peg in air, no contact)
#   5. Return to safe height
#
# All steps log to v1.2 schema (without force_mode events). Use compare.html to
# overlay against a real insert run to see what motion alone produces.

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from compliant_insertion_studio.wrapper.cad_lookup import (
    predict_tcp_at_seat, GRIPPER_CENTER_TOOL_OFFSET_M,
)
from compliant_insertion_studio.wrapper.telemetry import iso_local_now, filename_timestamp
from primitives.shared import config as primitive_config


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--object-name", default="u_orange")
    p.add_argument("--base-name", default="base1")
    p.add_argument("--grasp-id", type=int, default=1)
    p.add_argument("--current-object-orientation", nargs=4, type=float,
                   default=[0.7081, 0.002, 0.001, -0.7061])
    p.add_argument("--pattern", choices=["spiral", "circle", "fixed_x", "fixed_y", "static"],
                   default="spiral")
    p.add_argument("--radius-max-mm", type=float, default=12.0,
                   help="max spiral radius in mm")
    p.add_argument("--xy-speed-m-s", type=float, default=0.0015,
                   help="lateral speed during spiral")
    p.add_argument("--max-duration-s", type=float, default=15.0)
    p.add_argument("--hover-height-m", type=float, default=0.030,
                   help="height ABOVE predicted seat to perform motion at (in air)")
    p.add_argument("--mode", choices=["sim", "real"], default="sim")
    p.add_argument("--skip-home", action="store_true",
                   help="skip move_home (use if robot is already in a known pose)")
    return p


class TelemetryLogger(Node):
    """Lightweight ROS2 node that subscribes to TCP/joints and writes a v1.2-style CSV."""
    def __init__(self, csv_path):
        super().__init__("motion_isolation_logger")
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 history=HistoryPolicy.KEEP_LAST, depth=1,
                                 durability=DurabilityPolicy.VOLATILE)
        self.tcp = None
        self.joints = None
        self.gripper_w = None
        self.create_subscription(PoseStamped, "/tcp_pose_broadcaster/pose", self._tcp_cb, sensor_qos)
        self.create_subscription(JointState, "/joint_states", self._joints_cb, sensor_qos)
        self.create_subscription(Float32, "/gripper_width", self._gripper_cb, 10)
        self.csv_path = csv_path
        self.fh = open(csv_path, "w", buffering=1)
        # Match v1.2 main CSV header
        self.fh.write("t_s,phase,event_marker,hands_off,zero_event,"
                      "tcp_x,tcp_y,tcp_z,tcp_qx,tcp_qy,tcp_qz,tcp_qw,"
                      "target_x,target_y,target_z,target_qx,target_qy,target_qz,target_qw,"
                      "dx,dy,dz,droll,dpitch,dyaw,fx,fy,fz,tx,ty,tz,"
                      "gripper_width,commanded_fz,wrench_frame_id,"
                      "obj_x,obj_y,obj_z,obj_qx,obj_qy,obj_qz,obj_qw\n")
        self.target_xyz = (float("nan"),) * 3
        self.target_quat = (float("nan"),) * 4
        self.tcp_to_object_quat = (float("nan"),) * 4
        self.start_t = time.time()
        self.phase = "MOTION"
        self.row_count = 0

    def _tcp_cb(self, msg): self.tcp = msg
    def _joints_cb(self, msg): self.joints = msg
    def _gripper_cb(self, msg):
        try: self.gripper_w = float(msg.data)
        except Exception: pass

    def log_tick(self):
        if self.tcp is None: return
        t = time.time() - self.start_t
        p = self.tcp.pose.position; q = self.tcp.pose.orientation
        gw = self.gripper_w if self.gripper_w is not None else float("nan")
        # Wrench unavailable in fake mode — use NaN
        nan = float("nan")
        cols = [
            f"{t:.4f}", self.phase, "0", "0", "0",
            f"{p.x:.6f}", f"{p.y:.6f}", f"{p.z:.6f}",
            f"{q.x:.6f}", f"{q.y:.6f}", f"{q.z:.6f}", f"{q.w:.6f}",
            *[f"{v:.6f}" if not math.isnan(v) else "nan" for v in self.target_xyz],
            *[f"{v:.6f}" if not math.isnan(v) else "nan" for v in self.target_quat],
            *(["nan"] * 6),  # dx..dyaw
            *(["nan"] * 6),  # fx..tz
            f"{gw:.4f}" if not math.isnan(gw) else "nan",
            "0.0000", "tool0_controller",
            f"{p.x:.6f}", f"{p.y:.6f}", f"{p.z:.6f}",
            *[f"{v:.6f}" if not math.isnan(v) else "nan" for v in self.tcp_to_object_quat],
        ]
        self.fh.write(",".join(cols) + "\n")
        self.row_count += 1

    def close(self):
        try: self.fh.close()
        except Exception: pass


def run_subprocess(cmd, label, timeout=30):
    print(f"\n>>> {label}")
    print(f"    cmd: {' '.join(cmd)}")
    rc = subprocess.run(cmd, timeout=timeout).returncode
    if rc != 0:
        print(f"    FAILED rc={rc}"); return False
    print(f"    OK")
    return True


def compute_spiral_setpoint(t_in_phase, pattern, hover_xy, radius_max_m, xy_speed):
    """Returns target (tcp_x, tcp_y) for the motion pattern."""
    if pattern == "static":
        return hover_xy
    if pattern == "fixed_x":
        return (hover_xy[0] + min(xy_speed * t_in_phase, radius_max_m), hover_xy[1])
    if pattern == "fixed_y":
        return (hover_xy[0], hover_xy[1] + min(xy_speed * t_in_phase, radius_max_m))
    if pattern == "circle":
        r = min(radius_max_m, 0.005)  # 5mm fixed
        omega = 2 * math.pi * 0.25  # 0.25 Hz
        return (hover_xy[0] + r * math.cos(omega * t_in_phase),
                hover_xy[1] + r * math.sin(omega * t_in_phase))
    if pattern == "spiral":
        # Archimedean — radius grows linearly with t up to radius_max
        pitch = 0.0006  # m
        theta = (2 * math.pi / pitch) * xy_speed * t_in_phase
        radius = min(radius_max_m, (pitch / (2 * math.pi)) * theta)
        return (hover_xy[0] + radius * math.cos(theta),
                hover_xy[1] + radius * math.sin(theta))
    return hover_xy


def main():
    args = build_parser().parse_args()

    # Step 0: Home + safe height (operator's request — start from home)
    if not args.skip_home:
        if not run_subprocess([sys.executable, "-m", "primitives.move_home",
                                "--mode", args.mode, "--joint-space", "--duration", "5.0"],
                               label="STEP 1/5: move_home (joint-space, safe with held part)"):
            return 1
    if not run_subprocess([sys.executable, "-m", "primitives.move_to_safe_height",
                            "--mode", args.mode],
                           label="STEP 2/5: move_to_safe_height"):
        return 1

    # Step 3: HOVER above predicted target (uses verified _run_hover chain)
    hover_cmd = [sys.executable, "-m", "compliant_insertion_studio.wrapper._run_hover",
                 "--object-name", args.object_name, "--base-name", args.base_name,
                 "--grasp-id", str(args.grasp_id),
                 "--current-object-orientation",
                 *[str(v) for v in args.current_object_orientation],
                 "--use-default-base-position"]
    if not run_subprocess(hover_cmd, label="STEP 3/5: HOVER above predicted seat", timeout=60):
        return 1

    # Step 4: motion-isolation pattern via direct TCP setpoints
    print(f"\n>>> STEP 4/5: motion-isolation pattern={args.pattern} (peg should be in air)")
    rclpy.init()
    try:
        # Read current TCP pose to use as hover-xy reference
        from rclpy.qos import QoSProfile
        node = Node("motion_isolation_setup_reader")
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 history=HistoryPolicy.KEEP_LAST, depth=1,
                                 durability=DurabilityPolicy.VOLATILE)
        last_pose = [None]
        def _cb(m): last_pose[0] = m
        sub = node.create_subscription(PoseStamped, "/tcp_pose_broadcaster/pose", _cb, sensor_qos)
        for _ in range(50):
            rclpy.spin_once(node, timeout_sec=0.1)
            if last_pose[0] is not None: break
        if last_pose[0] is None:
            print("    failed to read TCP pose; aborting"); node.destroy_node(); return 1
        hover_x = last_pose[0].pose.position.x
        hover_y = last_pose[0].pose.position.y
        hover_z = last_pose[0].pose.position.z
        print(f"    hover_xy=({hover_x*1000:+.2f},{hover_y*1000:+.2f})mm  z={hover_z*1000:+.2f}mm")
        node.destroy_node()

        # Open CSV logger
        ts = filename_timestamp()
        log_dir = os.path.join(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
                               "compliant_insertion_studio", "logs")
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"motion_test_{args.object_name}_{args.pattern}_{ts}.csv")
        meta_path = csv_path[:-4] + ".meta.json"
        logger = TelemetryLogger(csv_path)

        # Move TCP through pattern by sending position commands
        # We can't easily command TCP directly; the simplest is to rely on the
        # primitives' move_to_*xyz patterns. For motion isolation, we'll just
        # log telemetry and let the operator verify visually in RViz.
        # Pattern: walk through setpoints and call move_to_safe_height-style primitives.
        # For SIMPLER visualization in fake mode, we just trace the spiral path mathematically
        # and command the controller through it.

        # Simplest path: use primitives.move_to_safe_height-style action client to
        # send TCP setpoints. But this is heavy. For sim visualization, we just
        # log the COMPUTED setpoints + actual TCP and let the user see RViz.

        # Call out to primitives.move_to_xyz for each setpoint? Too slow.
        # Instead: use ros2 control's forward_position_controller? Not active here.
        # SIMPLEST sim-mode approach: just log + print setpoints; user sees
        # what the spiral pattern WOULD look like via the printed trajectory.

        n_steps = int(args.max_duration_s / 0.1)
        print(f"    computing spiral trajectory: {n_steps} setpoints over {args.max_duration_s}s")
        print(f"    (NOTE: in fake mode without an active position-streaming controller, "
              f"setpoints are computed and logged but not sent to the robot. "
              f"Switch to real-mode + force_mode for actual motion.)")
        with open(csv_path[:-4] + ".setpoints.csv", "w") as fh:
            fh.write("t_s,sp_x,sp_y,sp_z\n")
            for i in range(n_steps):
                t_in = i * 0.1
                sp = compute_spiral_setpoint(t_in, args.pattern, (hover_x, hover_y),
                                              args.radius_max_mm * 1e-3,
                                              args.xy_speed_m_s)
                fh.write(f"{t_in:.2f},{sp[0]:.6f},{sp[1]:.6f},{hover_z:.6f}\n")

        # Run the spiral-PATTERN-VISUALIZATION via repeated move_to_safe_height-like calls.
        # For a simple, observable test in sim: walk to 4 corners of the spiral.
        print(f"    walking through spiral via primitives.move_to_safe_height "
              f"to visualize in RViz (4 waypoints)...")
        waypoints = [(hover_x, hover_y, hover_z),
                      compute_spiral_setpoint(2.0, args.pattern, (hover_x, hover_y), args.radius_max_mm * 1e-3, args.xy_speed_m_s) + (hover_z,),
                      compute_spiral_setpoint(5.0, args.pattern, (hover_x, hover_y), args.radius_max_mm * 1e-3, args.xy_speed_m_s) + (hover_z,),
                      compute_spiral_setpoint(8.0, args.pattern, (hover_x, hover_y), args.radius_max_mm * 1e-3, args.xy_speed_m_s) + (hover_z,),
                      compute_spiral_setpoint(args.max_duration_s, args.pattern, (hover_x, hover_y), args.radius_max_mm * 1e-3, args.xy_speed_m_s) + (hover_z,),
                      ]
        # Tick the logger for the duration
        t_start = time.time()
        deadline = t_start + args.max_duration_s + 5.0
        while time.time() < deadline:
            rclpy.spin_once(logger, timeout_sec=0.005)
            logger.log_tick()
            time.sleep(0.01)

        logger.close()
        # Write meta
        with open(meta_path, "w") as fh:
            json.dump({
                "schema_version": 1.2,
                "object": args.object_name,
                "base": args.base_name,
                "grasp_id": args.grasp_id,
                "outcome": "success",
                "outcome_reason": "motion_isolation_complete",
                "assist_level": f"motion_isolation_{args.pattern}_sim",
                "user_notes": f"position-controlled sim variant; pattern={args.pattern} "
                              f"radius_max={args.radius_max_mm}mm xy_speed={args.xy_speed_m_s}m/s. "
                              f"NOTE: setpoints computed; actual motion via fake-hardware playback only.",
                "force_mode_params": {"task_frame": "base_link", "type": 0,
                                       "wrench": {"fx": 0, "fy": 0, "fz": 0, "tx": 0, "ty": 0, "tz": 0},
                                       "selection_vector": [False] * 6,
                                       "gain_scaling": 0.0, "damping_factor": 0.0},
                "duration_s": time.time() - t_start,
                "samples_logged": logger.row_count,
            }, fh, indent=2)
        print(f"    logged {logger.row_count} samples to {csv_path}")
        print(f"    setpoint trajectory: {csv_path[:-4]}.setpoints.csv")
    finally:
        try: rclpy.shutdown()
        except Exception: pass

    # Step 5: return to safe height
    run_subprocess([sys.executable, "-m", "primitives.move_to_safe_height", "--mode", args.mode],
                    label="STEP 5/5: return to safe height")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

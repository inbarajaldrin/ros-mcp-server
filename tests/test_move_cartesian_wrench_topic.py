"""move_cartesian, executed on a node-less Mover (stand-in ROS modules; nothing joins a graph)."""
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import tests.ros_stubs as ros_stubs

ros_stubs.install()
import primitives.move_cartesian as mc  # noqa: E402
from primitives.shared.ik import dh_params, forward_kinematics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HOME = np.array([1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
# top-down at home, then turned -60 deg about world x: the camera side 60 deg below horizontal
CAMERA_DOWN = np.array([1.571218, -1.473045, 2.11277, -3.25773, -1.573413, -0.00103])


@pytest.mark.parametrize("flag,topic", [(None, "/force_torque_sensor_broadcaster/wrench"),
                                        ("0", "/force_torque_sensor_broadcaster/wrench"),
                                        ("1", "/force_torque_sensor_broadcaster/wrench_sim")])
def test_the_force_guard_reads_the_real_wrench_unless_droidstack_sim(flag, topic):
    """Only a DroidStack sim run may read the simulator's wrench_sim topic."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROS_MCP_")}
    if flag is not None:
        env["ROS_MCP_DROIDSTACK_SIM"] = flag
    code = ("import tests.ros_stubs as s; s.install()\n"
            "import primitives.move_cartesian as mc\n"
            "seen = []\n"
            "mc.Mover.create_subscription = lambda self, typ, name, *a: seen.append(name)\n"
            "mc.Mover()\n"
            "print(seen[1])\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True).stdout.split()[-1]
    assert out == topic


class _Sent(Exception):
    pass


def _plan(q0, xyz, orient, **kw):
    class ActionClient:
        def wait_for_server(self, timeout_sec=None):
            return True

        def send_goal_async(self, goal):
            raise _Sent()

    node = types.SimpleNamespace(q=None, ac=ActionClient(), F=[])
    q0 = np.array(q0, float)
    real_spin = mc.rclpy.spin_once
    mc.rclpy.spin_once = lambda n, timeout_sec=None: setattr(n, "q", q0.copy())
    try:
        return mc.plan_move(node, xyz, orient, **kw).get("error")
    except _Sent:
        return "SENT"
    finally:
        mc.rclpy.spin_once = real_spin


def test_a_top_down_descent_is_sent():
    p = forward_kinematics(dh_params, HOME)[:3, 3]
    assert _plan(HOME, [p[0], p[1], p[2] - 0.35], ("keep", None), min_z=-1.0) == "SENT"


def test_the_final_stance_is_checked_with_the_camera_rule():
    """Review P2-3: a landing that ends with the camera toward the table, or the mount within 30 mm of it, is
    refused where the move ends - not only at the stance move_down checked above it."""
    p = forward_kinematics(dh_params, HOME)[:3, 3]
    assert _plan(CAMERA_DOWN, [p[0], p[1], 0.20], ("keep", None), min_z=-1.0) == "SENT"
    why = _plan(CAMERA_DOWN, [p[0], p[1], 0.15], ("keep", None), min_z=-1.0)
    assert why.startswith("CAMERA_TOWARD_TABLE") and why.endswith("(at the end of the Cartesian move)")
    # the clearance is measured on the mount/D415's enclosing geometry (padded fitted boxes, then the exact CAD): at
    # z 0.10 the real mount is 37.9 mm up (the old coarse envelopes read lower), at 0.08 it is 23 mm
    why = _plan(CAMERA_DOWN, [p[0], p[1], 0.08], ("keep", None), min_z=-1.0)
    assert why.startswith("CAMERA_CLEARANCE")


def test_a_reorienting_move_is_checked_at_every_waypoint():
    T = forward_kinematics(dh_params, HOME)
    p = T[:3, 3]
    Rt = R.from_euler("x", -60, degrees=True).as_matrix() @ T[:3, :3]
    why = _plan(HOME, [p[0], p[1], p[2] - 0.3], ("quat", R.from_matrix(Rt).as_quat().tolist()), min_z=-1.0,
                max_joint_step=3.0)
    assert why.startswith("CAMERA_TOWARD_TABLE") and "at waypoint" in why

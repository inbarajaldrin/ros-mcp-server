#!/usr/bin/env python3
"""
HOVER subprocess driver — one-shot.

Spawned by compliant_insert.py to navigate the robot to the per-object HOVER
pose, then exit. Wraps the host's existing TranslateObject.translate_for_target_real
so the wrapper inherits the production HOVER motion (DH-based IK, fold symmetry,
collision-checked Jacobian waypoints, closed-loop correction) without re-deriving it.

Why subprocess (not import-into-wrapper):
- Sidesteps the dual-rclpy-Node coupling between the wrapper and TranslateObject
- Matches existing primitive-orchestration pattern (see translate_object.py:1085 -> prismatic_peg_insertion subprocess)
- Lets the wrapper subprocess this with a clean rclpy lifecycle that can't leak

Output: __RESULT_JSON__ block on stdout (parsed by parent wrapper).
Exit code: 0 on success, 1 on failure.
"""

import argparse
import json
import os
import sys

# Make the host repo importable regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import rclpy  # noqa: E402

from primitives.translate_object import TranslateObject  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="HOVER one-shot subprocess for compliant_insert")
    ap.add_argument("--object-name", required=True)
    ap.add_argument("--base-name", required=True)
    ap.add_argument("--grasp-id", type=int, required=True)
    ap.add_argument("--current-object-orientation", nargs=4, type=float, required=True,
                    help="quaternion x y z w of the held object's CURRENT orientation")
    ap.add_argument("--final-base-pos", nargs=3, type=float, default=None,
                    help="x y z of base position in world frame")
    ap.add_argument("--final-base-orientation", nargs=4, type=float, default=None,
                    help="quaternion x y z w of base orientation in world frame")
    ap.add_argument("--use-default-base-position", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = None
    result = {"result": "failure", "error": "unknown"}
    try:
        node = TranslateObject(mode="real")

        # Wait for joint state + current EE pose subscriptions to populate.
        import time as _t
        t0 = _t.time()
        while _t.time() - t0 < 5.0:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.current_ee_pose is not None and node.current_joint_angles is not None:
                break
        if node.current_ee_pose is None:
            result = {"result": "failure", "error": "EE pose subscription did not populate within 5s"}
            return
        if node.current_joint_angles is None:
            # joint state may also be lazy-loaded; try the explicit reader
            node.read_current_joint_angles()
            if node.current_joint_angles is None:
                result = {"result": "failure", "error": "joint state subscription did not populate within 5s"}
                return

        success = node.translate_for_target_real(
            args.object_name,
            args.base_name,
            final_base_pos=args.final_base_pos,
            final_base_orientation=args.final_base_orientation,
            use_default_base=args.use_default_base_position,
            grasp_id=args.grasp_id,
            object_orientation=args.current_object_orientation,
        )

        if success:
            # Capture the post-HOVER EE pose for the wrapper's meta JSON
            ee = node.current_ee_pose
            result = {
                "result": "success",
                "object_name": args.object_name,
                "base_name": args.base_name,
                "grasp_id": args.grasp_id,
                "ee_pose_at_hover": {
                    "xyz_m": [ee.pose.position.x, ee.pose.position.y, ee.pose.position.z],
                    "quat_xyzw": [ee.pose.orientation.x, ee.pose.orientation.y,
                                  ee.pose.orientation.z, ee.pose.orientation.w],
                },
            }
        else:
            result = {"result": "failure", "error": node.error_message or "translate_for_target_real returned False"}

    except Exception as e:
        result = {"result": "failure", "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")
    sys.exit(0 if result.get("result") == "success" else 1)


if __name__ == "__main__":
    main()

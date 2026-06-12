import sys
import os
import json
import argparse

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from primitives.core.move_xyz import run_ik, output_result
from primitives.shared.config import HOME_POSE, HOME_JOINTS


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

    # NOTE: the refuse-while-holding guard lives server-side now (_assert_not_holding in
    # server_core.py — the centralized verify_grasp width-only predicate, sim AND real).
    # The bespoke sim-only check that used to run here was its inline duplicate.

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

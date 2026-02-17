import sys
import os
import json

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from primitives.core.move_xyz import run_ik, output_result
from primitives.shared.config import HOME_POSE

HOME_MOVEMENT_DURATION = 5.0


def main(args=None):
    target_position = HOME_POSE[0:3]
    target_rpy = HOME_POSE[3:6]

    success, error = run_ik(target_position, target_rpy,
                            duration=HOME_MOVEMENT_DURATION,
                            reference_frame='ee')

    result = {"result": "success" if success else "failure"}
    if not success and error:
        result["error"] = error

    output_result(result)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

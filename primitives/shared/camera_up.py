"""Keep the gripper camera away from the table when reorienting a held part.

The D415 sits on the tool's -Y side (aruco_camera_localizer robot_config.yaml:
cam_offset_position [-0.015, -0.037, 0.07] in the EE frame). When a part is laid on its
side the gripper is horizontal, and the two wrist rolls that give the same object
orientation differ by 180 deg about the tool axis: one has the camera above the gripper,
the other below it, facing the table.

2026-10-07, u_brown, same start pose and same commanded placement:
  grip 0 deg relative to the part   -> reorient face_right_roll270, camera up  -> seated
  grip 180 deg relative to the part -> reorient face_right_roll90,  camera down -> toppled
                                                                        at the staging drop

Which roll is available is decided at GRASP time: the two top-down wrist yaws differ by
180 deg about the tool axis, which flips the camera side of every downstream reorient.
So move_to_grasp picks its wrist with reorient_camera_z(), and rotate_object filters its
candidates with camera_z().

Only the world-z component is used, so it is the same in UR's `base` and `base_link`
frames (they differ by a rotation about z).
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation as R

from primitives.shared.config import DEFAULT_BASE_ORIENTATION
from primitives.shared.fold_symmetry import equivalent_orientations, load_symmetry_data
from utils.data_path_finder import (find_assembly_json_by_base_name, get_assembly_data_dir,
                                    get_symmetry_dir)

CAMERA_SIDE_TOOL = np.array([0.0, -1.0, 0.0])


def camera_z(R_EE) -> float:
    """World-z of the camera side. > 0: camera above the gripper, away from the table."""
    return float((np.asarray(R_EE) @ CAMERA_SIDE_TOOL)[2])


def assembly_targets(object_name: str, base_name: str):
    """Fold-equivalent world orientations the object must reach in `base_name`.

    Mirrors rotate_object.reorient_for_target in real mode: R_base is
    DEFAULT_BASE_ORIENTATION, the target is the JSON quaternion relative to the base.
    Returns None when the assembly config has no entry for the object.
    """
    json_file = find_assembly_json_by_base_name(base_name, str(get_assembly_data_dir()))
    if not json_file:
        return None
    with open(json_file) as fh:
        config = json.load(fh)
    q = None
    for comp in config.get('components', []):
        if comp.get('name') == object_name:
            qd = comp.get('rotation', {}).get('quaternion', {})
            q = [qd.get('x', 0.0), qd.get('y', 0.0), qd.get('z', 0.0), qd.get('w', 1.0)]
            break
    if q is None:
        return None
    R_target = R.from_quat(DEFAULT_BASE_ORIENTATION).as_matrix() @ R.from_quat(q).as_matrix()
    fold_data = load_symmetry_data(object_name, str(get_symmetry_dir()))
    return list(equivalent_orientations(R_target, fold_data))


def reorient_camera_z(R_EE_grasp, R_object, targets) -> float:
    """Best camera_z among the reorients that would carry the object, held with this
    grasp, onto any fold-equivalent target. Same grasp model as rotate_object:
    R_grasp = R_EE^T @ R_object, and the reorient EE is target @ R_grasp^T.
    """
    R_grasp = np.asarray(R_EE_grasp).T @ np.asarray(R_object)
    return max(camera_z(np.asarray(T) @ R_grasp.T) for T in targets)

"""Keep the gripper camera away from the table when reorienting a held part.

The D415 sits on one side of the tool (the old mount: -Y, cam_offset_position [-0.015, -0.037, 0.07]; the v3.1 mount since
2026-11-10: +Y, [-0.0008, 0.0848, 0.0927]); CAMERA_SIDE_TOOL is read from the localizer's robot_config.yaml so a mount change
flips it with the calibration. When a part is laid on its
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

def _camera_side_tool():
    """Unit vector in the tool's XY plane toward the camera, from the hand-eye mount in the localizer config."""
    import yaml
    from primitives.shared.config import LOCALIZER_CONFIG
    try:
        t = np.asarray(yaml.safe_load(open(LOCALIZER_CONFIG))['cam_offset_position'], dtype=float)
        v = np.array([t[0], t[1], 0.0])
        if np.linalg.norm(v) > 0.01:
            return v / np.linalg.norm(v)
    except Exception:
        pass
    raise RuntimeError(f"camera_up: no camera side in {LOCALIZER_CONFIG} (cam_offset_position)")


CAMERA_SIDE_TOOL = _camera_side_tool()


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


# --- Hard rule (lead, 2026-11-19, after the operator saw r48c's u_orange place-down camera-side-down) ---
# The camera and its mount are never put on the table side near the table, and never brought near it.
NEAR_TABLE_M = 0.150          # gripper centre within this of the table: the side rule applies (pick, place-down,
                              # regrasp, insert - and every reorient that leads to them)
SIDE_RULE_CAMERA_Z_MIN = -0.05  # camera side may not point below horizontal (3 deg tolerance for numerics/tilt)
TOP_DOWN_COS = 0.966          # tool z within 15 deg of straight down: the camera side is horizontal by construction,
                              # only the clearance floor applies
CAMERA_CLEARANCE_M = 0.030    # mount / D415 envelope stays at least this far above the table at every route sample
# Axis-aligned envelopes of the v3.1 mount and D415 CAD meshes in the tool0 frame (min, max; m), from the lab
# model's visual meshes mount_lower / mount_upper / d415 on rg2_base_link (fmb-cell-lab, 2026-11-19).
CAMERA_GEOMETRY_TOOL = {
    "mount_lower": ((-0.044, -0.024, 0.0721), (0.044, -0.0007, 0.1041)),
    "mount_upper": ((-0.044, 0.0008, 0.0583), (0.087, 0.1141, 0.1041)),
    "d415": ((-0.0145, 0.076, 0.0711), (0.0845, 0.1022, 0.097)),
}


def _camera_corners_tool():
    corners = []
    for lo, hi in CAMERA_GEOMETRY_TOOL.values():
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    corners.append((x, y, z))
    return np.array(corners)


_CORNERS = _camera_corners_tool()


def camera_stance_refusal(T_flange, near_table=None):
    """None, or a typed reason why this flange pose puts the camera toward / near the table.

    T_flange: 4x4 flange pose in the robot base frame.  near_table: force the side rule (a reorient or
    place-down stance that will be carried to the table); None decides by the gripper centre's height."""
    from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET, TABLE_HEIGHT
    T_flange = np.asarray(T_flange, dtype=float)
    Rf, pf = T_flange[:3, :3], T_flange[:3, 3]
    lowest = float(np.min((_CORNERS @ Rf.T + pf)[:, 2]) - TABLE_HEIGHT)
    if lowest < CAMERA_CLEARANCE_M:
        return (f"CAMERA_CLEARANCE: mount/D415 {1000 * lowest:.0f} mm above the table "
                f"(needs {1000 * CAMERA_CLEARANCE_M:.0f})")
    if near_table is None:
        near_table = float((Rf @ GRIPPER_CENTER_TOOL_OFFSET + pf)[2] - TABLE_HEIGHT) < NEAR_TABLE_M
    top_down = float(-Rf[2, 2]) > TOP_DOWN_COS
    cz = camera_z(Rf)
    if near_table and not top_down and cz < SIDE_RULE_CAMERA_Z_MIN:
        return (f"CAMERA_TOWARD_TABLE: camera side points {np.degrees(np.arcsin(min(1.0, -cz))):.0f} deg below "
                f"horizontal at a near-table stance")
    return None


CAMERA_MOUNT_ARM_MARGIN_M = 0.010   # the self-collision check's 1 cm margin (collision.check_self_collision)


def camera_mount_arm_refusal(joints):
    """None, or a typed reason: the v3.1 mount/D415 (fitted CAD boxes, faces sampled at <= 5 mm, the sampling
    gap subtracted) within the margin of the arm: collision.PHYSICAL_LINK_CAPSULES (proven to enclose the UR5e's
    real surface, shoulder to wrist_1) and the base cylinder (lead ruling 2026-11-19: the DH capsules are a
    mis-placed approximation built for self-collision, not used here).  r48c (MuJoCo, 2026-11-19): a camera-up
    yellow reorient drove the mount into the wrist_1 zone; the arm-only self-collision check could not see it."""
    from primitives.shared.camera_geometry import mount_box_spheres_tool, mount_surface_tool
    from primitives.shared.collision import base_cylinder_distance, physical_link_capsules
    from primitives.shared.ik import dh_params, forward_kinematics
    q = np.asarray(joints, float)
    T = forward_kinematics(dh_params, q)
    surface, gap, owner = mount_surface_tool()
    centres, radii = mount_box_spheres_tool()
    C = centres @ T[:3, :3].T + T[:3, 3]

    def axis_distance(P, a, ab):
        t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)

    for a, b, r, label in physical_link_capsules(q):
        ab = b - a
        # skip only the boxes whose bounding sphere clears the margin plus the sampling gap (their samples would
        # too); sample the faces of the others
        near = np.flatnonzero(axis_distance(C, a, ab) - radii - gap - r < CAMERA_MOUNT_ARM_MARGIN_M)
        if near.size == 0:
            continue
        W = surface[np.isin(owner, near)] @ T[:3, :3].T + T[:3, 3]
        # the nearest surface sample, less the sampling gap: a lower bound on the true box-to-capsule clearance
        clearance = float(np.min(axis_distance(W, a, ab))) - gap - r
        if clearance < CAMERA_MOUNT_ARM_MARGIN_M:
            return (f"CAMERA_MOUNT_ARM_COLLISION: mount/D415 {1000 * clearance:.1f} mm from the {label} "
                    f"(needs {1000 * CAMERA_MOUNT_ARM_MARGIN_M:.0f})")
    clearance = float(np.min(base_cylinder_distance(surface @ T[:3, :3].T + T[:3, 3]))) - gap
    if clearance < CAMERA_MOUNT_ARM_MARGIN_M:
        return (f"CAMERA_MOUNT_ARM_COLLISION: mount/D415 {1000 * clearance:.1f} mm from the base "
                f"(needs {1000 * CAMERA_MOUNT_ARM_MARGIN_M:.0f})")
    return None


def camera_route_refusal(start_joints, target_joints, near_table_end=False, num_samples=40):
    """The rule at every joint-linear sample of a route: table side and clearance, and the mount against the arm;
    the side rule at the end when the route leads to the table."""
    from primitives.shared.ik import dh_params, forward_kinematics
    start, target = np.asarray(start_joints, float), np.asarray(target_joints, float)
    for i in range(num_samples + 1):
        alpha = i / num_samples
        q = start + alpha * (target - start)
        reason = (camera_stance_refusal(forward_kinematics(dh_params, q),
                                        near_table=(True if (near_table_end and i == num_samples) else None))
                  or camera_mount_arm_refusal(q))
        if reason:
            return f"{reason} (alpha={alpha:.3f})"
    return None


def hover_route_refusal(start_joints, waypoints):
    """None, or a typed reason to refuse a fix-orientation move: the recovery from a non-top-down stance to a face-down
    hover at safe height (move_to_grasp.fix_orientation_inprocess, move_to_clear_area --fix-orientation), sent as
    joint-linear legs start -> waypoints[0] (the lift, if any) -> ... -> waypoints[-1] (the hover).

    Every leg gets the camera rule at every sample (table side, mount/D415 clearance, mount against the arm) and the
    cable caps (joint travel <= one turn, |wrist_3| <= WRIST3_LIMIT_RAD; an empty-gripper refusal reads ROUTE_*).  The
    last leg's final sample is the hover stance itself.  The hover is fixed-seed IK, so nothing chose it for the camera:
    with the camera on tool +Y a recovery swing is where the October stances put the mount into the table (lead,
    2026-11-20, review r5 P2-3).  Fail closed: a start stance that already breaks the rule refuses the move."""
    from primitives.shared.held_part_collision import joint_travel_refusal
    legs = [np.asarray(start_joints, float)] + [np.asarray(w, float) for w in waypoints]
    for i, (a, b) in enumerate(zip(legs[:-1], legs[1:])):
        name = "hover" if i == len(legs) - 2 else f"leg {i + 1}"
        reason = joint_travel_refusal(a, b, prefix="ROUTE") or camera_route_refusal(a, b)
        if reason:
            return f"{reason} [fix-orientation {name}]"
    return None

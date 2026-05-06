import numpy as np

ROBOT_BASE_Z = 0.08         # Robot base origin Z in world/table frame (meters)
TABLE_HEIGHT_WORLD = 0.0    # Table surface Z in world frame (meters)
TABLE_HEIGHT = TABLE_HEIGHT_WORLD - ROBOT_BASE_Z  # Table surface Z in robot base frame (meters)
TABLE_COLLISION_MARGIN_SIDEWAYS = 0.08  # Safety margin for sideways/horizontal EE orientations
TABLE_COLLISION_MARGIN_FACEDOWN = 0.08  # Safety margin for face_down EE orientation 
GRIPPER_CENTER_TOOL_OFFSET = np.array([0.0, 0.0, 0.2286])  # 3D offset from EE flange to gripper center in tool frame (meters)
SAFE_HEIGHT_ABOVE_TABLE = 0.4      # Safe/hover height above table surface (meters)
SAFE_HEIGHT = TABLE_HEIGHT + SAFE_HEIGHT_ABOVE_TABLE  # Absolute safe height Z in robot base frame
ROTATE_ABOUT_GRIPPER_CENTER = False # If True, rotate about gripper center; if False, rotate about flange
DEFAULT_BASE_ORIENTATION = [0.0, 0.0, 0.0, 1.0]  # Base object orientation [x, y, z, w] quaternion (identity = no rotation)
# Calibrated 2026-05-06: empirical bias (+1.52, -3.78) mm derived from a
# centered-grasp u_brown GUIDED demo. The (0.0, -0.4) starting value put CAD
# predictions ~5mm off the physical fixture; this corrected value puts CAD
# directly on actual slot positions and removes the need for FSM-side bias
# correction in autonomous SEARCH.
DEFAULT_BASE_POSITION = [0.00152, -0.40378, TABLE_HEIGHT + 0.0175]  # Default base object position [x, y, z] in robot base frame (meters)

# UR5e motion limits — two independent constraints, compute_duration takes the max.
# Cartesian limits govern translation speed (reference: safe_height ↔ table = 0.32m → 5.0s).
# Joint limits govern rotation speed independently (2 rad rotation → 5.0s).
JOINT_VEL_LIMIT = 0.80      # rad/s  (hardware max: pi rad/s)
JOINT_ACCEL_LIMIT = 0.80    # rad/s²
CART_VEL_LIMIT = 0.0798     # m/s    (hardware max: 1.0 m/s)
CART_ACCEL_LIMIT = 0.0798   # m/s²
MIN_DURATION = 2.5          # seconds — floor to avoid jerky micro-moves
MAX_DURATION = 10.0         # seconds — cap for very long moves

HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0]  # EE tool frame
PICK_STATION_POSE = [-0.330, -0.385, 0.404, 0, 180, 0]  # EE tool frame

# Joint-space "tidy" home — operator-confirmed 2026-05-03 on the real arm
# inside the rectangular workspace (robot base mounted at the long-side
# center). Tidy = arm geometry that matches the F/T calibration starting
# pose (face_down_canonical, with shoulder_pan rotated to face the
# workspace) so going from this home into the calibration sequence
# requires zero arm reconfiguration. Use this when HOME_POSE Cartesian
# IK picks an unfortunate joint config (e.g. shoulder_pan landing at
# +79° instead of +90° due to mount-orientation drift).
#   shoulder_pan = +90°  (EE faces into the workspace rectangle)
#   shoulder_lift = -90°  upper arm horizontal pointing away from base
#   elbow = +90°          forearm vertical, pointing down (elbow-up)
#   wrist_1 = -90°        wrist neutral so gripper hangs aligned
#   wrist_2 = -90°        gripper opening axis aligned with workspace
#   wrist_3 = 0°          no final rotation
HOME_JOINTS = [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

# UR5e DH parameters: (theta_offset, d, a, alpha)
DH_PARAMS = [
    (0,  0.1625,  0,     np.pi/2),
    (0,  0,      -0.425,  0),
    (0,  0,      -0.3922, 0),
    (0,  0.1333,  0,     np.pi/2),
    (0,  0.0997,  0,    -np.pi/2),
    (0,  0.0996,  0,     0)
]


# --- Gripper width lookup from fmb1_assembly.json ---
# Source of truth for per-object/per-grasp gripper-width-before-pick.
# Loaded lazily on first call. Used by:
#   - compliant_insertion_studio/scripts/run_assembly_step.py
#   - compliant_insertion_studio/scripts/regrasp_held_object.py
#   - compliant_insertion_studio/scripts/loop_autonomous_insert.sh (via above)
#
# If the file or entry is missing, callers get None and fall back to their
# own default (35 mm).
import json as _json
import os as _os

_ASSEMBLY_JSON_PATHS = [
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
                  "ablations", "eval_resources", "fmb1_assembly.json"),
]
_GRASP_WIDTH_CACHE: dict | None = None  # populated on first lookup


def _load_grasp_width_table() -> dict:
    """Returns {(object_name, grasp_id): gripper_width_mm} from fmb1_assembly.json.
    Empty dict if file not found / parse failed."""
    global _GRASP_WIDTH_CACHE
    if _GRASP_WIDTH_CACHE is not None:
        return _GRASP_WIDTH_CACHE
    table: dict = {}
    for path in _ASSEMBLY_JSON_PATHS:
        if not _os.path.exists(path):
            continue
        try:
            d = _json.load(open(path))
            for step in d.get("assembly_order", []):
                obj = step.get("object_name")
                gid = step.get("grasp_id")
                w = step.get("gripper_width_mm")
                if obj is not None and gid is not None and w is not None:
                    table[(obj, int(gid))] = float(w)
        except Exception:
            pass
    _GRASP_WIDTH_CACHE = table
    return table


def get_gripper_width_mm(object_name: str, grasp_id: int,
                         default_mm: float | None = None) -> float | None:
    """Lookup gripper_width_mm for (object_name, grasp_id) from fmb1_assembly.json.
    Returns default_mm (caller-supplied) if entry not found, or None if the
    caller didn't provide a default and the entry is missing."""
    return _load_grasp_width_table().get((object_name, int(grasp_id)), default_mm)


def get_grasp_id_for_assembly(object_name: str) -> int | None:
    """Returns the grasp_id used in the FMB1 assembly for `object_name`, or
    None if the object isn't in the assembly. Used by loop scripts so
    callers don't need to remember per-object grasp_ids."""
    for (obj, gid) in _load_grasp_width_table().keys():
        if obj == object_name:
            return int(gid)
    return None


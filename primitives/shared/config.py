import numpy as np
import os as _os

# Runtime mode, read once at import. Set by ros-mcp-server's _run_primitive from
# the tool's --mode arg (env var ROS_MCP_MODE), since primitives bind config
# constants at import time. Defaults to 'real' so the tagged real path is
# unaffected when the env var is absent.
RUNTIME_MODE = _os.environ.get("ROS_MCP_MODE", "real").strip().lower()

ROBOT_BASE_Z = (0.0 if RUNTIME_MODE == "sim" else 0.08)  # world frame. Sim: robot+floor at z=0 (Isaac convention); Real: 8cm mount. Mode-aware so TABLE_HEIGHT/DEFAULT_BASE_POSITION/place_down map to the SAME world Z in both modes (sim place_down was driving into the ground plane at world -0.08).
TABLE_HEIGHT_WORLD = 0.0    # Table surface Z in world frame (meters)
TABLE_HEIGHT = TABLE_HEIGHT_WORLD - ROBOT_BASE_Z  # Table surface Z in robot base frame (meters)
TABLE_COLLISION_MARGIN_SIDEWAYS = 0.08  # Safety margin for sideways/horizontal EE orientations
TABLE_COLLISION_MARGIN_FACEDOWN = 0.08  # Safety margin for face_down EE orientation 
# Mode-aware EE-flange -> gripper-center offset (tool frame, meters). The sim
# (Isaac ur5e-dt) twin and the real RG2+coupling differ in flange->grasp-frame
# distance, so the value must track the mode. Real=0.2286 (real-world-verified
# 2026-05-07), Sim=0.23 (historical sim value, lost when commit 2b42e33 collapsed
# the SIM/REAL toggle to a single hardcoded real value — this restores it as a
# proper mode switch instead of a manual comment toggle).
GRIPPER_CENTER_TOOL_OFFSET_REAL = np.array([0.0, 0.0, 0.2286])  # real RG2 + coupling
GRIPPER_CENTER_TOOL_OFFSET_SIM  = np.array([0.0, 0.0, 0.23])    # Isaac ur5e-dt twin
GRIPPER_CENTER_TOOL_OFFSET = (
    GRIPPER_CENTER_TOOL_OFFSET_SIM if RUNTIME_MODE == "sim"
    else GRIPPER_CENTER_TOOL_OFFSET_REAL
)  # 3D offset from EE flange to gripper center in tool frame (meters)
SAFE_HEIGHT_ABOVE_TABLE = 0.4      # Safe/hover height above table surface (meters)
SAFE_HEIGHT = TABLE_HEIGHT + SAFE_HEIGHT_ABOVE_TABLE  # Absolute safe height Z in robot base frame
ROTATE_ABOUT_GRIPPER_CENTER = False # If True, rotate about gripper center; if False, rotate about flange
DEFAULT_BASE_ORIENTATION = [0.0, 0.0, 0.0, 1.0]  # Base object orientation [x, y, z, w] quaternion (identity = no rotation)
# 2026-05-07: operator marked base at (+55, -480) mm; refined by u_brown
# autonomous insert at 09:57. Observed actual seat TCP=(+33.95, -519.60) mm
# vs predicted (+30.70, -527.50) mm → camera/calibration delta = (+3.25,
# +7.90) mm in XY. Applied to DEFAULT_BASE_POSITION below. Z left unchanged
# (Δz=-2.23mm, but operator chose not to apply).
DEFAULT_BASE_POSITION = [0.05825, -0.47210, TABLE_HEIGHT + 0.0175]  # Default base object position [x, y, z] in robot base frame (meters)

# Per-object base-position offset (meters in robot base frame). Added on top
# of DEFAULT_BASE_POSITION when inserting this specific object — i.e. when
# CAD predicted_tcp_at_seat is computed, the base is treated as
# (DEFAULT_BASE_POSITION + PER_OBJECT_BASE_OFFSET_M[object_name]).
#
# Why per-object: DEFAULT_BASE_POSITION was calibrated from a u_brown
# centered-grasp GUIDED demo (2026-05-06). u_brown's slot is at y=-47.5mm
# off-center in base1; small base-tilt or fixture rotation gets baked into
# the calibration but shifts differently at other slot locations. For slots
# centered on the base (line_green, inverted_u_yellow), the same calibration
# bias points slightly off the actual hole center.
#
# Calibrated 2026-05-07. Three seat-z readings collected during tuning of
# inverted_u_yellow on base1 with line_green seated underneath:
#   without line_green (operator-seat):  TCP_z = +212.00 mm  Δz = -7.90  mm
#   with line_green (operator-seat):     TCP_z = +207.68 mm  Δz = -12.22 mm
#   with line_green (autonomous-seat):   TCP_z = +212.78 mm  Δz = -7.12  mm
# The operator could press harder than autonomous Fz=-9N and bottom the
# part out at +207.68mm; the autonomous chain can only reach +212.78mm
# under the project's 9N cap. We anchor predicted_seat to the autonomous-
# achievable depth (-7.12mm) so the global seat detector fires correctly
# without needing to widen its tolerance — landing at 212.78 is exactly
# at predicted_seat and falls inside the 5mm SEAT classifier band.
# X and Y stay calibrated to the operator-with-line_green seat (the most
# precise reading available for those axes).
PER_OBJECT_BASE_OFFSET_M: dict = {
    # 2026-05-07 v4: u_brown calibrated from full-assembly run actual seat
    # TCP=(+82.35, -523.02, +199.07)mm vs predicted (+84.93, -519.50, +200.80)mm
    #   → Δ=(-2.59, -3.52, -1.74)mm.
    "u_brown":  (-0.00259, -0.00352, -0.00174),
    # 2026-05-07 v4: u_orange calibrated from full-assembly run actual seat
    # TCP=(+31.77, -427.29, +199.75)mm vs predicted (+33.96, -423.69, +200.80)mm
    #   → Δ=(-2.19, -3.60, -1.06)mm.
    "u_orange": (-0.00219, -0.00360, -0.00106),
    # 2026-05-07 v3: yellow-specific calibration. Started with line_green's
    # offset, observed actual seat TCP=(+60.48, -470.91, +212.98)mm vs
    # predicted (+60.27, -470.19, +224.66)mm → Δ=(+0.21, -0.72, -11.68)mm.
    # Yellow's prong geometry seats 11.7mm deeper than line_green's CAD
    # predicts; XY nearly identical.
    # Final offset = line_green offset (+1.13, +1.88, +4.76) + Δ
    #              = (+1.34, +1.16, -6.92) mm
    "inverted_u_yellow": (+0.00134, +0.00116, -0.00692),
    # 2026-05-07: line_green calibrated from operator-manual-seat (with
    # u_brown + u_orange already seated as flanking obstacles):
    #   TCP_actual=(+3.31, -399.38, +191.73) mm
    #   TCP_predicted (no offset)=(+2.71, -403.85, +190.49) mm
    #   Δ = (+0.60, +4.46, +1.24) mm
    # Note: line_green also has a known body-on-adjacent-part collision
    # issue (gripper jaws holding line_green at center hit u_brown/u_orange
    # tops during descent). This offset alone won't solve that — a true
    # fix needs lateral-approach descent or different grasp_id. Captured
    # here so the autonomous chain can at least classify SEAT correctly
    # if/when descent does complete.
    # 2026-05-07 v2: re-calibrated against operator-positioned line_green
    # in slot. TCP_actual=(+60.57, -470.31, +195.25)mm vs
    # TCP_predicted_no_offset=(+59.44, -472.19, +190.49)mm  →  Δ=(+1.13, +1.88, +4.76)mm.
    # The previous +4.46mm Y offset was overshooting (compliance was drifting
    # the bar away from the true slot in -Y); the actual slot is +1.88mm in Y.
    "line_green": (+0.00113, +0.00188, +0.00476),
}


def get_object_base_offset_m(object_name: str) -> tuple[float, float, float]:
    """Per-object base-position offset (meters), REAL mode only.

    These are real-arm calibration corrections (compliance drift, camera delta)
    measured against observed seat TCPs. The sim twin is CAD-exact and reads the
    base pose from /objects_poses_sim, so applying these in sim would inject
    real-world error. Returns (0,0,0) in sim mode or if no entry. (The compliant
    insert path that consumes this is already real-gated; this guard makes the
    sim/real split explicit at the source.)"""
    if RUNTIME_MODE == "sim":
        return (0.0, 0.0, 0.0)
    return tuple(PER_OBJECT_BASE_OFFSET_M.get(object_name, (0.0, 0.0, 0.0)))


# Per-object SEAT classifier tolerance (meters). Used by both the contact
# classifier ("is this contact at SEAT?") and the global seat detector
# ("is the peg now at the predicted seated position?").  Default 5mm is
# loose enough that small CAD prediction noise + force-mode compliance
# don't prevent SEAT detection on single-stage parts (u_brown/u_orange/
# inverted_u_yellow). line_green needs tighter (2mm) because the
# autonomous can plausibly stop 3mm above true seat without being
# physically seated, and we need the global seat detector to NOT fire
# prematurely so the multi-stage GUIDED re-entry can hand back to the
# operator for a second nudge.
PER_OBJECT_SEAT_TOL_M: dict = {
    "line_green": 0.002,
}


def get_object_seat_tolerance_m(object_name: str, default: float = 0.005) -> float:
    """Per-object SEAT classifier + global-seat-detector tolerance (m)."""
    return float(PER_OBJECT_SEAT_TOL_M.get(object_name, default))


# Per-object INSERT_DESCENT compliance mode. Default is XY-locked, Z-only
# compliance (selection vector (F,F,T,F,F,F)) — appropriate for u_brown /
# u_orange / inverted_u_yellow where the spiral SEARCH already aligned the
# peg before INSERT_DESCENT begins.
#
# For line_green the slot is the same width as the bar (~26mm), so the
# gripper jaws holding the bar at its center don't fit through the slot
# alongside the bar. The descent has to find a yaw that minimises clearance
# resistance against the slot walls — operator-suggested 2026-05-07: enable
# yaw compliance during INSERT_DESCENT so the part naturally settles at the
# correct yaw under torque feedback from the slot walls. Selection vector
# (F,F,T,F,F,T) — Z compliant + Tz (yaw) compliant; XY+roll+pitch locked.
PER_OBJECT_FREE_YAW_INSERT: dict = {
    "line_green": True,
}


def is_free_yaw_insert(object_name: str) -> bool:
    """Whether this object uses yaw-compliant INSERT_DESCENT."""
    return bool(PER_OBJECT_FREE_YAW_INSERT.get(object_name, False))


# Per-object force overrides for the insertion pipeline. Default is 9N
# downforce throughout (calibrated to break stiction on u_brown/u_orange/
# inverted_u_yellow rim friction). For line_green the gripper jaws holding
# the bar at center collide with base1's outer surface — pushing 9N drives
# the gripper hard into the base, which is both unsafe and prevents any
# yaw-compliance from settling the part. Drop to 4N approach + 4N search
# press + 3N insert descent so the part can yield to slot-wall feedback
# without slamming.
PER_OBJECT_INSERT_FORCES_N: dict = {
    # 2026-05-07 v3: line_green sat fully blocked at -4N (Fz reaction = +4.15N
    # mean, Z dropped 2mm in 30s while Y slid 7.7mm). Friction at gripper-base
    # contact dominated. Cut forces in HALF again to 2.0N and lower; the
    # gripper-base contact must be light enough that compliance can reposition.
    "line_green": {
        "fz_approach":    2.0,   # default 9.0
        "search_F_press": 1.5,   # default 9.0  (commanded Fz during compliant_descent)
        "search_Fmax":    1.5,   # default 8.0
        "insert_fz":      1.5,   # default 8.0  (fsm_cfg insert_fz_N during INSERT_DESCENT)
    },
}


def get_object_insert_forces(object_name: str) -> dict:
    """Per-object force overrides; returns empty dict if not configured."""
    return dict(PER_OBJECT_INSERT_FORCES_N.get(object_name, {}))


# 2026-05-07: per-object SEARCH mode.
#   "spiral"            (default): Archimedean PD spiral via SearchDirector.
#                       Drives commanded Fx/Fy to sweep an outward spiral.
#                       Works for u_brown / u_orange / inverted_u_yellow where
#                       the slot is point-like and the gripper clears the base.
#   "compliant_descent" (line_green): pure passive XYZ compliance under -Fz,
#                       no commanded lateral force. Slot rim physically
#                       guides the peg laterally as Z descends. If Z stalls,
#                       escalate selection vector to free Rx/Ry/Rz so slot
#                       walls can tilt + rotate the peg into alignment.
#                       Borrowed from primitives/_real_mode_stash/legacy/
#                       passive_compliance_move.py (Tier 0 → Tier 1 escalation).
PER_OBJECT_SEARCH_MODE: dict = {
    "line_green": "compliant_descent",
}


def get_object_search_mode(object_name: str) -> str:
    """Per-object SEARCH-state mode. Default 'spiral'."""
    return str(PER_OBJECT_SEARCH_MODE.get(object_name, "spiral"))


# 2026-05-07: per-object yaw bias torque applied during free-yaw INSERT_DESCENT
# and during compliant_descent tiers with rotational compliance enabled. Sign
# determines rotation direction, magnitude is in Nm. Applied in base_link.
# line_green: observed rotating wrong direction on impact under pure compliance,
# operator requested sign reverse — start with negative bias.
PER_OBJECT_YAW_BIAS_NM: dict = {
    "line_green": +0.5,   # 2026-05-07: -0.5 was wrong direction → flipped
}


def get_object_yaw_bias_nm(object_name: str) -> float:
    """Per-object active Tz bias (Nm) for free-yaw INSERT_DESCENT. 0 = pure compliance."""
    return float(PER_OBJECT_YAW_BIAS_NM.get(object_name, 0.0))

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


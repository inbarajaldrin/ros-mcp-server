"""
Telemetry Schema v1 — column constants and meta JSON keys.

Single source of truth for the wrapper's CSV header order and meta JSON shape.
The dashboard depends on the EXACT column names and order defined here.

Spec: ../docs/SCHEMA.md
"""

SCHEMA_VERSION = 1

# CSV header — ORDER IS PART OF THE CONTRACT.
# Adding a column at the end is non-breaking (dashboard reads by name).
# Reordering or renaming is a v2 schema bump (see SCHEMA.md).
CSV_COLUMNS: tuple[str, ...] = (
    "t_s", "phase", "event_marker", "hands_off", "zero_event",
    "tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw",
    "target_x", "target_y", "target_z", "target_qx", "target_qy", "target_qz", "target_qw",
    "dx", "dy", "dz", "droll", "dpitch", "dyaw",
    "fx", "fy", "fz", "tx", "ty", "tz",
    "gripper_width", "commanded_fz",
)

# Phase FSM values. EXACT strings written to the CSV `phase` column.
PHASE_PRE = "PRE"
PHASE_HOVER = "HOVER"
PHASE_ZERO = "ZERO"
PHASE_ACTIVE = "ACTIVE"
PHASE_DONE = "DONE"
PHASE_ABORT = "ABORT"
PHASE_VALUES: frozenset[str] = frozenset({
    PHASE_PRE, PHASE_HOVER, PHASE_ZERO, PHASE_ACTIVE, PHASE_DONE, PHASE_ABORT,
})

# Outcome enum (meta JSON `outcome` key). `crashed_no_meta` is dashboard-only.
OUTCOME_SUCCESS = "success"
OUTCOME_ABORT = "abort"
OUTCOME_TIMEOUT = "timeout"
WRAPPER_OUTCOMES: frozenset[str] = frozenset({
    OUTCOME_SUCCESS, OUTCOME_ABORT, OUTCOME_TIMEOUT,
})

# Euler convention for droll/dpitch/dyaw. Passed to scipy.spatial.transform.Rotation.as_euler.
# DO NOT change without bumping schema_version.
EULER_CONVENTION = "xyz"   # extrinsic XYZ (scipy default for lowercase)

# Float formatting — applied per column when writing CSV rows.
# Keep in sync with SCHEMA.md "Float formatting" section.
FMT_TIME = "{:.4f}"
FMT_POSITION = "{:.6f}"
FMT_QUAT = "{:.6f}"
FMT_ERROR_LIN = "{:.6f}"
FMT_ERROR_ANG = "{:.6f}"
FMT_FORCE = "{:.4f}"
FMT_TORQUE = "{:.4f}"
FMT_GRIPPER = "{:.4f}"
FMT_COMMAND = "{:.4f}"

# Required meta JSON keys (top-level). Wrapper MUST populate these or the
# dashboard treats the episode as malformed.
META_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "object", "base", "grasp_id", "wrapper_version",
    "start_iso", "end_iso", "duration_s",
    "outcome", "outcome_reason",
    "assembly_target_world",
    "force_mode_params",
    "foundational_calibration",
    "smoke_test",
    "post_zero_bias",
    "post_zero_drift_check",
    "hands_off_window",
    "mid_episode_zero_events",
    "user_notes",
)

# Optional / recommended meta JSON keys. Dashboard uses if present, ignores if missing.
META_OPTIONAL_KEYS: tuple[str, ...] = (
    "session_warmup_minutes",
    "ros_distro", "ur_driver_version", "ur_msgs_version",
    "controller_at_start", "controllers_loaded",
    "joint_state_at_active_start",
    "hover_pose_world",
    "ik_pre_check",
)


def empty_meta_template() -> dict:
    """Return a meta JSON skeleton with all required keys present and sensible defaults.

    Wrapper fills fields in as the episode progresses. Anything still null at write time
    indicates a phase that didn't run (e.g. crashed before ZERO completed).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "object": None,
        "base": None,
        "grasp_id": None,
        "wrapper_version": None,
        "start_iso": None,
        "end_iso": None,
        "duration_s": None,
        "outcome": None,
        "outcome_reason": None,
        "assembly_target_world": {"xyz_m": None, "quat_xyzw": None},
        "force_mode_params": None,
        "foundational_calibration": None,
        "smoke_test": None,
        "post_zero_bias": None,
        "post_zero_drift_check": None,
        "hands_off_window": None,
        "mid_episode_zero_events": [],
        "user_notes": "",
    }


def csv_path_for(object_name: str, timestamp_str: str, log_dir) -> str:
    """Return the CSV path for an episode following the TELE-03 naming convention.

    `timestamp_str` should be `YYYYMMDD_HHMMSS` (caller formats — wrapper uses local time).
    `log_dir` is a pathlib.Path or str.
    """
    return f"{log_dir}/insert_{object_name}_{timestamp_str}.csv"


def meta_path_for(csv_path: str) -> str:
    """Return the .meta.json sidecar path for a given CSV path."""
    if not csv_path.endswith(".csv"):
        raise ValueError(f"Expected .csv path, got: {csv_path}")
    return csv_path[:-4] + ".meta.json"

"""
Telemetry Schema v1 — column constants and meta JSON keys.

Single source of truth for the wrapper's CSV header order and meta JSON shape.
The dashboard depends on the EXACT column names and order defined here.

Spec: ../docs/SCHEMA.md
"""

SCHEMA_VERSION = 1.2  # 2026-05-05: adds 4 sidecar files (joints_raw, wrench_raw, cmd_wrench_raw, fm_events) — see G001-G004 in analysis/STATE.json. Main CSV unchanged.

# CSV header — ORDER IS PART OF THE CONTRACT.
# Adding a column at the end is non-breaking (dashboard reads by name).
# Reordering or renaming is a v2 schema bump (see SCHEMA.md).
#
# v1.1 (2026-05-03): added wrench_frame_id (str) + obj_x..obj_qw (object pose
# estimate, computed per-row as TCP × tcp_to_object_transform). Both append
# at end → backward-compatible for dashboards that read by header name.
CSV_COLUMNS: tuple[str, ...] = (
    "t_s", "phase", "event_marker", "hands_off", "zero_event",
    "tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw",
    "target_x", "target_y", "target_z", "target_qx", "target_qy", "target_qz", "target_qw",
    "dx", "dy", "dz", "droll", "dpitch", "dyaw",
    "fx", "fy", "fz", "tx", "ty", "tz",
    "gripper_width", "commanded_fz",
    # --- v1.1 additions ---
    "wrench_frame_id",
    "obj_x", "obj_y", "obj_z", "obj_qx", "obj_qy", "obj_qz", "obj_qw",
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
    # --- v1.1 additions (required) ---
    "assist_level",                      # "autonomous" | "assisted" — operator tag at end-of-episode
    "current_object_orientation_input",  # quat the wrapper was launched with (CLI arg)
    "tcp_pose_at_active_start",          # snapshot of TCP world pose at the moment ACTIVE began
    "hover_pose_world",                  # promoted from optional in v1.0
    "tcp_to_object_transform",           # 4x4 (or {pos,quat}) used to derive per-row obj_* in CSV
)

# Optional / recommended meta JSON keys. Dashboard uses if present, ignores if missing.
# NOTE: also includes v1.1 'required' keys so they're settable via set_optional()
# without a separate setter API. The REQUIRED list is documentation-only;
# OPTIONAL is the runtime validation set for the setter.
META_OPTIONAL_KEYS: tuple[str, ...] = (
    "session_warmup_minutes",
    "ros_distro", "ur_driver_version", "ur_msgs_version",
    "controller_at_start", "controllers_loaded",
    "joint_state_at_active_start",
    "ik_pre_check",
    # --- v1.1 (also in META_REQUIRED_KEYS — documented as required for Phase 3) ---
    "assist_level",
    "current_object_orientation_input",
    "tcp_pose_at_active_start",
    "hover_pose_world",
    "tcp_to_object_transform",
    # --- v1.2 (Phase 5) ---
    "termination_config",     # which YAML drove termination + the predicate dict
    "termination_fire_debug", # actual values at the moment the predicate fired
    "cad_prediction",         # CAD-derived predicted_tcp_at_seat + provenance
    "hole_observed",          # tcp_xy at the moment z-descent spike detected
                              # during spiral search (peg found chamfer/slot)
    # --- v1.3 (Phase 5 v3 — ContactSearchFSM, 2026-05-04 PM) ---
    "fsm_config",             # FSM tunables loaded from defaults.yaml
    "fsm_result",             # surface_z, hole_xy, hole_z, descent_post_hole_m, etc.
    # --- v1.4 (GUIDED mode for regime-decoding data collection, 2026-05-06) ---
    "hole_observed_operator", # tcp_xy at SIGUSR1 (operator's hand at the hole)
    # --- v1.5 (v4 Found Hole predicate validation, 2026-05-06) ---
    "hole_observed_v4_predicate", # tcp_xy at the tick the v4 |fz| state-transition
                                  # predicate fired (analysis/CONTROL_LAW.md). Logged
                                  # in stage 3a (parallel to SIGUSR1); also triggers
                                  # the GUIDED→INSERT_DESCENT transition in stage 3b.
    # --- v1.6 (CAD-derived contact classifier, 2026-05-07) ---
    "contact_classification", # {label, best_match_label, best_match_residual_mm,
                              # tcp_z_at_contact_m, candidates_m, in_tolerance, t_s}
                              # populated by ContactSearchFSM at first contact via
                              # cad_geometry.build_contact_candidates(). label ∈
                              # {SEAT, BASE_RIM, OBJ:<name>, UNKNOWN}. SEAT in-tol
                              # short-circuits APPROACH→DONE; others fall through
                              # to today's SEARCH/GUIDED/FIND_HOLE dispatch.
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
        # --- v1.1 additions ---
        "assist_level": None,
        "current_object_orientation_input": None,
        "tcp_pose_at_active_start": None,
        "hover_pose_world": None,
        "tcp_to_object_transform": None,
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

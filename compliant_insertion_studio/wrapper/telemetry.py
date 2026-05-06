"""
Telemetry writers — CSV row writer + meta JSON builder.

Both classes are ROS-free so they can be smoke-tested without rclpy
(see test_telemetry.py).

Spec: ../docs/SCHEMA.md
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import schema_v1 as s


# ---------------------------------------------------------------------------
# Per-axis error computation
# ---------------------------------------------------------------------------

def compute_per_axis_errors(
    tcp_xyz: tuple[float, float, float],
    tcp_quat_xyzw: tuple[float, float, float, float],
    target_xyz: tuple[float, float, float],
    target_quat_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Return (dx, dy, dz, droll, dpitch, dyaw).

    Linear: tcp - target, sign-preserving.
    Angular: relative rotation `target^-1 . tcp` decomposed as scipy 'xyz' Euler
    (extrinsic). Convention is locked in schema_v1.EULER_CONVENTION; do not
    change without bumping schema_version.
    """
    dx = tcp_xyz[0] - target_xyz[0]
    dy = tcp_xyz[1] - target_xyz[1]
    dz = tcp_xyz[2] - target_xyz[2]

    R_tcp = Rotation.from_quat(tcp_quat_xyzw)
    R_target = Rotation.from_quat(target_quat_xyzw)
    R_rel = R_target.inv() * R_tcp
    droll, dpitch, dyaw = R_rel.as_euler(s.EULER_CONVENTION)

    return dx, dy, dz, float(droll), float(dpitch), float(dyaw)


# ---------------------------------------------------------------------------
# ISO timestamp helper
# ---------------------------------------------------------------------------

def iso_local_now() -> str:
    """Return current local time as ISO8601 with offset (e.g. 2026-05-02T20:14:33.521+02:00)."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def filename_timestamp() -> str:
    """Return YYYYMMDD_HHMMSS for log-file naming (TELE-03)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _fmt(value, formatter: str) -> str:
    """Format a number per the schema's FMT_* string. NaN -> 'nan'."""
    if value is None:
        return "nan"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "nan"
    except (TypeError, ValueError):
        pass
    return formatter.format(value)


# Mapping of column name -> formatter (None means "as-is, no formatting").
_FORMATTERS: dict[str, str | None] = {
    "t_s": s.FMT_TIME,
    "phase": None,
    "event_marker": None,
    "hands_off": None,
    "zero_event": None,
    "tcp_x": s.FMT_POSITION, "tcp_y": s.FMT_POSITION, "tcp_z": s.FMT_POSITION,
    "tcp_qx": s.FMT_QUAT, "tcp_qy": s.FMT_QUAT, "tcp_qz": s.FMT_QUAT, "tcp_qw": s.FMT_QUAT,
    "target_x": s.FMT_POSITION, "target_y": s.FMT_POSITION, "target_z": s.FMT_POSITION,
    "target_qx": s.FMT_QUAT, "target_qy": s.FMT_QUAT, "target_qz": s.FMT_QUAT, "target_qw": s.FMT_QUAT,
    "dx": s.FMT_ERROR_LIN, "dy": s.FMT_ERROR_LIN, "dz": s.FMT_ERROR_LIN,
    "droll": s.FMT_ERROR_ANG, "dpitch": s.FMT_ERROR_ANG, "dyaw": s.FMT_ERROR_ANG,
    "fx": s.FMT_FORCE, "fy": s.FMT_FORCE, "fz": s.FMT_FORCE,
    "tx": s.FMT_TORQUE, "ty": s.FMT_TORQUE, "tz": s.FMT_TORQUE,
    "gripper_width": s.FMT_GRIPPER,
    "commanded_fz": s.FMT_COMMAND,
    # --- v1.1 ---
    "wrench_frame_id": None,            # string, no numeric format
    "obj_x": s.FMT_POSITION, "obj_y": s.FMT_POSITION, "obj_z": s.FMT_POSITION,
    "obj_qx": s.FMT_QUAT, "obj_qy": s.FMT_QUAT, "obj_qz": s.FMT_QUAT, "obj_qw": s.FMT_QUAT,
}


class CSVWriter:
    """Per-row CSV writer for telemetry. Line-buffered for crash-safety."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        # buffering=1 = line-buffered; flushes after every newline
        self._fh = open(self.path, "w", buffering=1, newline="")
        self._fh.write(",".join(s.CSV_COLUMNS) + "\n")
        self._row_count = 0

    def write(self, row: dict) -> None:
        """Write one telemetry row.

        `row` must have all keys in schema_v1.CSV_COLUMNS. Phase string is
        validated. Numeric fields are formatted per schema_v1.FMT_*.
        """
        missing = [c for c in s.CSV_COLUMNS if c not in row]
        if missing:
            raise ValueError(f"CSV row missing required columns: {missing}")

        if row["phase"] not in s.PHASE_VALUES:
            raise ValueError(f"Invalid phase {row['phase']!r}, expected one of {sorted(s.PHASE_VALUES)}")

        cells = []
        for col in s.CSV_COLUMNS:
            fmt = _FORMATTERS[col]
            v = row[col]
            cells.append(_fmt(v, fmt) if fmt is not None else str(v))
        self._fh.write(",".join(cells) + "\n")
        self._row_count += 1

    @property
    def row_count(self) -> int:
        return self._row_count

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.flush()
            self._fh.close()


# ---------------------------------------------------------------------------
# Meta JSON builder
# ---------------------------------------------------------------------------

class MetaJSONBuilder:
    """Accumulate per-episode metadata across phases; write once at end."""

    def __init__(self):
        self._meta = s.empty_meta_template()
        self._mid_episode_zero_events = []

    # --- identity ----------------------------------------------------------
    def set_identity(self, *, object_name: str, base: str | None, grasp_id: int | None,
                     wrapper_version: str) -> None:
        self._meta["object"] = object_name
        self._meta["base"] = base
        self._meta["grasp_id"] = grasp_id
        self._meta["wrapper_version"] = wrapper_version

    # --- timing ------------------------------------------------------------
    def set_start(self, iso: str) -> None:
        self._meta["start_iso"] = iso

    def set_end(self, iso: str, duration_s: float) -> None:
        self._meta["end_iso"] = iso
        self._meta["duration_s"] = round(duration_s, 4)

    # --- outcome -----------------------------------------------------------
    def set_outcome(self, outcome: str, reason: str) -> None:
        if outcome not in s.WRAPPER_OUTCOMES:
            raise ValueError(f"Invalid outcome {outcome!r}, expected one of {sorted(s.WRAPPER_OUTCOMES)}")
        self._meta["outcome"] = outcome
        self._meta["outcome_reason"] = reason

    # --- target ------------------------------------------------------------
    def set_assembly_target(self, xyz_m: list[float], quat_xyzw: list[float]) -> None:
        self._meta["assembly_target_world"] = {
            "xyz_m": [float(v) for v in xyz_m],
            "quat_xyzw": [float(v) for v in quat_xyzw],
        }

    # --- force-mode params -------------------------------------------------
    def set_force_mode_params(self, params: dict) -> None:
        self._meta["force_mode_params"] = params

    # --- calibration provenance --------------------------------------------
    def set_foundational_calibration(self, cal: dict | None) -> None:
        self._meta["foundational_calibration"] = cal

    def set_smoke_test(self, st: dict | None) -> None:
        self._meta["smoke_test"] = st

    # --- zero / drift ------------------------------------------------------
    def set_post_zero_bias(self, bias: dict) -> None:
        self._meta["post_zero_bias"] = bias

    def set_post_zero_drift_check(self, drift: dict) -> None:
        self._meta["post_zero_drift_check"] = drift

    def add_mid_episode_zero(self, t_s: float, post_zero_bias: dict) -> None:
        self._mid_episode_zero_events.append({
            "t_s": round(float(t_s), 4),
            "post_zero_bias": post_zero_bias,
        })
        self._meta["mid_episode_zero_events"] = self._mid_episode_zero_events

    # --- hands-off window --------------------------------------------------
    def set_hands_off_window(self, *, start_iso: str, end_iso: str,
                             duration_s: float, trigger: str) -> None:
        self._meta["hands_off_window"] = {
            "start_iso": start_iso,
            "end_iso": end_iso,
            "duration_s": round(float(duration_s), 4),
            "trigger": trigger,
        }

    # --- user notes --------------------------------------------------------
    def set_user_notes(self, notes: str) -> None:
        self._meta["user_notes"] = notes

    # --- optional / provenance ---------------------------------------------
    def set_optional(self, key: str, value) -> None:
        if key not in s.META_OPTIONAL_KEYS:
            raise ValueError(f"{key!r} is not a recognized optional meta key. "
                             f"Known: {sorted(s.META_OPTIONAL_KEYS)}")
        self._meta[key] = value

    # --- write -------------------------------------------------------------
    def to_dict(self) -> dict:
        return dict(self._meta)

    def write(self, path: str | Path) -> None:
        with open(str(path), "w") as f:
            json.dump(self._meta, f, indent=2, sort_keys=False)
            f.write("\n")

#!/usr/bin/env python3
"""
Synthetic-CSV smoke test — no ROS, no robot.

Validates that the telemetry writers (CSVWriter, MetaJSONBuilder) produce
schema-v1-compliant output. Run away from the robot to catch schema regressions
before the operator returns.

Usage:
  python3 -m compliant_insertion_studio.wrapper.test_telemetry

Exits 0 if all checks pass, 1 if anything fails. Prints a short summary line per check.
"""

import csv as _csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

# Allow `python3 path/to/test_telemetry.py` invocation too
if __package__ is None or __package__ == "":
    _PKG_PARENT = Path(__file__).resolve().parents[2]
    if str(_PKG_PARENT) not in sys.path:
        sys.path.insert(0, str(_PKG_PARENT))
    from compliant_insertion_studio.wrapper import schema_v1 as s
    from compliant_insertion_studio.wrapper.telemetry import (
        CSVWriter, MetaJSONBuilder, compute_per_axis_errors,
        iso_local_now, filename_timestamp,
    )
else:
    from . import schema_v1 as s
    from .telemetry import (
        CSVWriter, MetaJSONBuilder, compute_per_axis_errors,
        iso_local_now, filename_timestamp,
    )


# ---------------------------------------------------------------------------
# Reusable test machinery
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self):
        self.failures: list[str] = []
        self.checks_run = 0

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        self.checks_run += 1
        marker = "OK  " if condition else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  {marker}  {label}{suffix}")
        if not condition:
            self.failures.append(label)

    def summary_and_exit(self) -> None:
        print(f"\n{self.checks_run} checks run, {len(self.failures)} failure(s).")
        if self.failures:
            print("FAIL: " + "; ".join(self.failures))
            sys.exit(1)
        print("PASS")
        sys.exit(0)


def _synth_row(phase: str, t_s: float, *,
               event_marker: int = 0, hands_off: int = 0, zero_event: int = 0,
               commanded_fz: float = 0.0) -> dict:
    """Build a fully-populated synthetic row."""
    return {
        "t_s": t_s, "phase": phase,
        "event_marker": event_marker, "hands_off": hands_off, "zero_event": zero_event,
        "tcp_x": 0.5, "tcp_y": 0.0, "tcp_z": 0.1 - 0.001 * t_s,   # descend over time
        "tcp_qx": 0.0, "tcp_qy": 0.0, "tcp_qz": 0.0, "tcp_qw": 1.0,
        "target_x": 0.5, "target_y": 0.0, "target_z": 0.05,
        "target_qx": 0.0, "target_qy": 0.0, "target_qz": 0.0, "target_qw": 1.0,
        "dx": 0.0, "dy": 0.0, "dz": (0.1 - 0.001 * t_s) - 0.05,
        "droll": 0.0, "dpitch": 0.0, "dyaw": 0.0,
        "fx": 0.05, "fy": -0.02, "fz": 1.5 + 0.05 * t_s,            # rising contact
        "tx": 0.001, "ty": -0.002, "tz": 0.0,
        "gripper_width": 0.043, "commanded_fz": commanded_fz,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compute_per_axis_errors(r: _Result) -> None:
    print("\n[compute_per_axis_errors]")
    dx, dy, dz, droll, dpitch, dyaw = compute_per_axis_errors(
        (0.5, 0.1, 0.2), (0, 0, 0, 1),
        (0.4, -0.1, 0.05), (0, 0, 0, 1),
    )
    r.check("dx == 0.1", math.isclose(dx, 0.1, abs_tol=1e-9), f"got {dx}")
    r.check("dy == 0.2", math.isclose(dy, 0.2, abs_tol=1e-9), f"got {dy}")
    r.check("dz == 0.15", math.isclose(dz, 0.15, abs_tol=1e-9), f"got {dz}")
    r.check("zero rotation gives zero euler", all(abs(v) < 1e-9 for v in (droll, dpitch, dyaw)))

    # 90-deg rotation about Z (target identity, tcp = quat for 90-deg yaw)
    import math as _m
    s2 = _m.sin(_m.pi / 4)
    c2 = _m.cos(_m.pi / 4)
    _, _, _, _, _, dyaw90 = compute_per_axis_errors(
        (0, 0, 0), (0, 0, s2, c2),    # 90-deg yaw
        (0, 0, 0), (0, 0, 0, 1),
    )
    r.check("90deg yaw -> dyaw ~= pi/2",
            math.isclose(dyaw90, _m.pi / 2, abs_tol=1e-6),
            f"got {dyaw90}")


def test_csv_writer_schema_compliance(r: _Result) -> None:
    print("\n[CSVWriter schema compliance]")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "insert_synth_20260101_000000.csv")
        w = CSVWriter(path)

        # Drive synthetic episode through all 6 phases
        sequence = [
            (s.PHASE_PRE, 100, {}),
            (s.PHASE_HOVER, 50, {}),
            (s.PHASE_ZERO, 100, {"hands_off": 1, "zero_event": 0}),
            (s.PHASE_ACTIVE, 3000, {"hands_off": 1, "commanded_fz": -3.0}),
            (s.PHASE_DONE, 50, {}),
            (s.PHASE_ABORT, 0, {}),       # zero-row phase is fine
        ]
        t = 0.0
        for phase, n, overrides in sequence:
            for i in range(n):
                row = _synth_row(phase, t, **overrides)
                w.write(row)
                t += 0.01
        w.close()

        # Validate: header order, row count, parseable as floats/ints
        with open(path) as f:
            reader = _csv.DictReader(f)
            r.check("header matches schema_v1.CSV_COLUMNS exactly",
                    list(reader.fieldnames or []) == list(s.CSV_COLUMNS),
                    f"got {reader.fieldnames}")
            rows = list(reader)
        expected_n = sum(n for _, n, _ in sequence)
        r.check(f"row count == {expected_n}", len(rows) == expected_n, f"got {len(rows)}")
        r.check("CSVWriter.row_count tracks correctly", w.row_count == expected_n,
                f"got {w.row_count}")

        # Spot-check field types
        if rows:
            row0 = rows[0]
            r.check("t_s parses as float", _is_float(row0["t_s"]))
            r.check("phase is one of PHASE_VALUES", row0["phase"] in s.PHASE_VALUES)
            r.check("event_marker parses as int", _is_int(row0["event_marker"]))
            r.check("hands_off parses as int", _is_int(row0["hands_off"]))
            r.check("zero_event parses as int", _is_int(row0["zero_event"]))
            r.check("tcp_x has 6 decimals",
                    _decimal_count(row0["tcp_x"]) == 6, f"got {row0['tcp_x']}")
            r.check("fx has 4 decimals",
                    _decimal_count(row0["fx"]) == 4, f"got {row0['fx']}")

        # Phase distribution
        phase_counts = {}
        for row in rows:
            phase_counts[row["phase"]] = phase_counts.get(row["phase"], 0) + 1
        r.check("PRE rows present", phase_counts.get(s.PHASE_PRE, 0) > 0)
        r.check("ACTIVE rows present", phase_counts.get(s.PHASE_ACTIVE, 0) > 0)
        r.check("DONE rows present", phase_counts.get(s.PHASE_DONE, 0) > 0)


def test_csv_writer_rejects_bad_input(r: _Result) -> None:
    print("\n[CSVWriter input validation]")
    with tempfile.TemporaryDirectory() as td:
        # Bad phase
        path1 = os.path.join(td, "bad_phase.csv")
        w = CSVWriter(path1)
        try:
            w.write({**_synth_row(s.PHASE_PRE, 0.0), "phase": "BOGUS"})
            r.check("rejects invalid phase string", False, "no exception raised")
        except ValueError:
            r.check("rejects invalid phase string", True)
        finally:
            w.close()

        # Missing column
        path2 = os.path.join(td, "missing_col.csv")
        w2 = CSVWriter(path2)
        bad = _synth_row(s.PHASE_PRE, 0.0)
        del bad["fz"]
        try:
            w2.write(bad)
            r.check("rejects row missing columns", False, "no exception raised")
        except ValueError:
            r.check("rejects row missing columns", True)
        finally:
            w2.close()


def test_csv_crash_safety(r: _Result) -> None:
    print("\n[CSVWriter crash safety (line-buffered)]")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "linebuf.csv")
        w = CSVWriter(path)
        w.write(_synth_row(s.PHASE_PRE, 0.0))
        # Don't call close() — simulate crash. The line-buffered fd should still
        # have flushed the row to disk.
        with open(path) as f:
            content = f.read()
        # We expect: header line + 1 data line, both terminated
        r.check("header present in unflushed file", content.startswith(",".join(s.CSV_COLUMNS)))
        r.check("at least 2 newlines (header + 1 row)", content.count("\n") >= 2,
                f"got {content.count(chr(10))} newlines")


def test_meta_json_required_keys(r: _Result) -> None:
    print("\n[MetaJSONBuilder required keys]")
    m = MetaJSONBuilder()
    m.set_identity(object_name="u_brown", base="fmb1_base", grasp_id=0,
                   wrapper_version="test@deadbeef")
    m.set_start(iso_local_now())
    m.set_end(iso_local_now(), 24.586)
    m.set_outcome(s.OUTCOME_SUCCESS, "operator_sigterm")
    m.set_assembly_target([0.4, 0.0, 0.05], [0, 0, 0, 1])
    m.set_force_mode_params({"task_frame": "base_link", "fz": -3.0})
    m.set_foundational_calibration({"mass_kg": 1.0823, "cog_xyz_m": [0.0014, -0.0025, 0.0521]})
    m.set_smoke_test({"result": "pass", "bias": {"Fx": 0.31}})
    m.set_post_zero_bias({"Fx": 0.18, "Fy": -0.09, "Fz": 0.21,
                           "Tx": 0.005, "Ty": -0.012, "Tz": 0.001})
    m.set_post_zero_drift_check({"delta_t_s": 1.0, "max_axis_drift_n": 0.04})
    m.set_hands_off_window(start_iso=iso_local_now(), end_iso=iso_local_now(),
                            duration_s=20.0, trigger="operator_step_back_confirmed")
    m.set_user_notes("synthetic test")

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "meta.json")
        m.write(p)
        with open(p) as f:
            d = json.load(f)

    r.check("schema_version == 1", d.get("schema_version") == 1, f"got {d.get('schema_version')}")
    missing = [k for k in s.META_REQUIRED_KEYS if k not in d]
    r.check("all META_REQUIRED_KEYS present", not missing, f"missing: {missing}")
    r.check("outcome in enum", d.get("outcome") in s.WRAPPER_OUTCOMES, f"got {d.get('outcome')}")
    r.check("mid_episode_zero_events is a list", isinstance(d.get("mid_episode_zero_events"), list))


def test_meta_json_outcome_enum(r: _Result) -> None:
    print("\n[MetaJSONBuilder outcome enum]")
    m = MetaJSONBuilder()
    try:
        m.set_outcome("not_an_outcome", "reason")
        r.check("rejects unknown outcome", False, "no exception raised")
    except ValueError:
        r.check("rejects unknown outcome", True)


def test_meta_json_optional_keys(r: _Result) -> None:
    print("\n[MetaJSONBuilder optional keys]")
    m = MetaJSONBuilder()
    m.set_optional("ros_distro", "humble")
    r.check("recognized optional accepted", m.to_dict().get("ros_distro") == "humble")
    try:
        m.set_optional("unknown_key", "value")
        r.check("rejects unknown optional key", False)
    except ValueError:
        r.check("rejects unknown optional key", True)


def test_path_helpers(r: _Result) -> None:
    print("\n[path helpers]")
    p = s.csv_path_for("u_brown", "20260502_201433", "logs")
    r.check("csv_path follows TELE-03 convention",
            p == "logs/insert_u_brown_20260502_201433.csv", f"got {p}")
    mp = s.meta_path_for(p)
    r.check("meta_path replaces .csv with .meta.json",
            mp == "logs/insert_u_brown_20260502_201433.meta.json", f"got {mp}")
    try:
        s.meta_path_for("logs/foo.txt")
        r.check("meta_path_for rejects non-csv input", False)
    except ValueError:
        r.check("meta_path_for rejects non-csv input", True)


def test_filename_timestamp_format(r: _Result) -> None:
    print("\n[filename_timestamp format]")
    ts = filename_timestamp()
    r.check("YYYYMMDD_HHMMSS format", len(ts) == 15 and ts[8] == "_", f"got {ts}")
    r.check("all digits except underscore", ts.replace("_", "").isdigit())


# ---------------------------------------------------------------------------
# Helpers used by tests
# ---------------------------------------------------------------------------

def _is_float(v: str) -> bool:
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


def _is_int(v: str) -> bool:
    try:
        int(v); return True
    except (TypeError, ValueError):
        return False


def _decimal_count(v: str) -> int:
    if "." not in v:
        return 0
    return len(v.split(".", 1)[1])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"=== Telemetry schema-v{s.SCHEMA_VERSION} synthetic smoke test ===")
    print(f"CSV columns: {len(s.CSV_COLUMNS)}; phases: {len(s.PHASE_VALUES)}; "
          f"required meta keys: {len(s.META_REQUIRED_KEYS)}")

    r = _Result()
    test_compute_per_axis_errors(r)
    test_csv_writer_schema_compliance(r)
    test_csv_writer_rejects_bad_input(r)
    test_csv_crash_safety(r)
    test_meta_json_required_keys(r)
    test_meta_json_outcome_enum(r)
    test_meta_json_optional_keys(r)
    test_path_helpers(r)
    test_filename_timestamp_format(r)
    r.summary_and_exit()


if __name__ == "__main__":
    main()

# Telemetry Schema v1 — Compliant Insertion Studio

**Status:** Locked for Phase 2 build. Bumps to v2+ require a milestone-level decision.
**Authority:** This file. Wrapper writes to it; dashboard, signature card, and config derivation read from it. Any disagreement is fixed here first, then propagated.

---

## Why this matters

The CSV + meta JSON pair is the **data contract** between the episode wrapper (Phase 2) and every downstream consumer:

- **Phase 3** collects 20 episodes against this schema
- **Phase 4** dashboard reads only this schema (no other source of truth)
- **Phase 5** algorithm derivation runs against signatures auto-computed from this schema
- **Phase 6** dispatcher/validation reuses the wrapper, so still writes this schema

A schema bug discovered in Phase 4 means **re-collecting all 20 episodes** — the project's most expensive failure mode. Get it right here.

Per-row fields are wide (33 columns) by design: rather than join tables in the dashboard, we duplicate fixed-per-episode data (target pose) into every row so the browser-side analysis is a single PapaParse stream + no joins. Cost is ~120 KB per episode in redundant target-pose bytes; acceptable for ~30 s × 100 Hz episodes (~30 MB total per episode CSV).

---

## File layout (TELE-03)

```
compliant_insertion_studio/logs/
  insert_<object>_<YYYYMMDD_HHMMSS>.csv         <- per-sample telemetry
  insert_<object>_<YYYYMMDD_HHMMSS>.meta.json   <- per-episode metadata
```

- `<object>`: object name (`u_brown`, `u_orange`, `line_green`, `inverted_u_yellow`, or any future part)
- `<YYYYMMDD_HHMMSS>`: episode start timestamp, local time (matches file mtime within ~1 s)
- The CSV and meta JSON share the same basename (no extension before `.csv` / `.meta.json`)
- Logs directory is `.gitignore`d

**Aborts preserved (TELE-06):** the wrapper NEVER deletes a CSV based on outcome. Both files exist for every started episode. The meta JSON's `outcome` field distinguishes success/abort/timeout. Demo-selection bias is a top-3 LfD failure mode; preservation is mandatory.

---

## CSV schema (TELE-01)

**Logging rate:** 100 Hz fixed (TELE-04). The `/force_torque_sensor_broadcaster/wrench` topic publishes at ~500 Hz on the operator's UR5e (live-verified). Wrapper subsamples every 5th message at write time. Full-rate logging produces ~150 MB CSVs that are unworkable in the browser dashboard.

**Header row** (exact order, exact names — column order is part of the contract):

```
t_s,phase,event_marker,hands_off,zero_event,
tcp_x,tcp_y,tcp_z,tcp_qx,tcp_qy,tcp_qz,tcp_qw,
target_x,target_y,target_z,target_qx,target_qy,target_qz,target_qw,
dx,dy,dz,droll,dpitch,dyaw,
fx,fy,fz,tx,ty,tz,
gripper_width,commanded_fz
```

(33 columns; line breaks above are presentational only — the actual header is one line.)

### Column dictionary

| # | Column | Type | Units | Phase scope | Notes |
|---|---|---|---|---|---|
| 1 | `t_s` | float | seconds | all | Sample time relative to episode start (`PRE` opens the clock at 0.0). Monotonic. |
| 2 | `phase` | string | enum | all | One of `PRE`, `HOVER`, `ZERO`, `ACTIVE`, `DONE`, `ABORT`. Exactly the wrapper's current FSM state at sample time. |
| 3 | `event_marker` | int | counter | all | Operator-toggled annotation counter. Increments by 1 each `SIGUSR1`. Starts at 0. Use for "I'm pushing" / "I let go" boundaries — increment-on-edge means the dashboard can split traces into N regions. |
| 4 | `hands_off` | int | 0/1 | all | 1 during the operator's hands-off window (after STEP-BACK gate confirmed in `ZERO`, until end of `ACTIVE`). 0 elsewhere. Phase 4 signature card filters on this column. |
| 5 | `zero_event` | int | 0/1 | mostly 0 | 1 on rows that are zero-baseline samples (the post-zero +1 s drift check, and any mid-episode `SIGUSR2` re-zero). 0 elsewhere. Always paired with the row's normal F/T sample. |
| 6–8 | `tcp_x`, `tcp_y`, `tcp_z` | float | meters | all | TCP position in `base_link` frame from `/tcp_pose_broadcaster/pose`. |
| 9–12 | `tcp_qx`, `tcp_qy`, `tcp_qz`, `tcp_qw` | float | unit quat | all | TCP orientation in `base_link` frame, scalar-last (matches ROS convention). |
| 13–15 | `target_x`, `target_y`, `target_z` | float | meters | all | Assembly target hole position in `base_link` frame. Fixed for the episode (loaded from assembly config + `translate_for_target_real` resolution). Repeated per row by design — keeps dashboard analysis JSON-join-free. |
| 16–19 | `target_qx`, `target_qy`, `target_qz`, `target_qw` | float | unit quat | all | Assembly target orientation. Same fixed-per-episode logic as position. |
| 20–22 | `dx`, `dy`, `dz` | float | meters | all | `tcp - target` linear error per axis. Sign-preserving. |
| 23–25 | `droll`, `dpitch`, `dyaw` | float | radians | all | Relative rotation `target⁻¹ · tcp` decomposed as XYZ extrinsic Euler via `scipy.spatial.transform.Rotation.from_quat(...).as_euler('xyz')`. Convention is fixed in `schema_v1.py`. |
| 26–28 | `fx`, `fy`, `fz` | float | newtons | all | Wrench force in `base_link` frame from `/force_torque_sensor_broadcaster/wrench`. Sign convention: positive Z is up (= robot is pulling away from a downward push), so contact during a downward `commanded_fz` shows up as **positive** `fz` magnitude on contact. |
| 29–31 | `tx`, `ty`, `tz` | float | newton-meters | all | Wrench torque, same frame and convention. |
| 32 | `gripper_width` | float | meters | all | Latest gripper width from `/gripper_width`. NaN if topic not yet seen. |
| 33 | `commanded_fz` | float | newtons | mostly 0 | The Fz value commanded into `force_mode_controller` at this sample. 0 outside `ACTIVE`. Negative = pushing down. Useful for distinguishing dwell rows from push rows in the dashboard. |

### Sign conventions cheat-sheet (frequent foot-gun)

- **Z-up** in `base_link` (UR convention)
- **Pushing down** = negative `commanded_fz`
- **Felt contact pushing back** = positive `fz` (resists the downward push)
- **Quaternion**: scalar-LAST (`x, y, z, w`), matching `geometry_msgs/Quaternion` and scipy's default
- **Euler decomposition**: XYZ extrinsic (the scipy default for `as_euler('xyz')`). A rotation about world X is `droll`, about world Y is `dpitch`, about world Z is `dyaw`.

### Float formatting

- Position/error: 6 decimal places (`%.6f`) — sub-µm precision, well past sensor capability, keeps the dashboard's plotted points from quantizing visibly
- Quaternion: 6 decimals
- Force/torque: 4 decimals (sensor noise floor is ~10 mN; 4 decimals captures it without false precision)
- `t_s`: 4 decimals (0.1 ms — well past 100 Hz period)
- Integers (`event_marker`, `hands_off`, `zero_event`): no decimals
- Gripper width: 4 decimals
- NaN: literal `nan` string (PapaParse handles)

### Crash-safety

CSV writer flushes after every row (line-buffered via `open(..., 'w', buffering=1)`). A SIGKILL or power loss mid-episode leaves a partially-written-but-parseable CSV. The corresponding meta JSON is written **once at episode end** — a partial CSV without a meta JSON is a crashed episode and the dashboard treats it as `outcome=crashed_no_meta`.

---

## Meta JSON schema (TELE-02)

**Schema version (TELE-05):** every meta JSON's first key is `"schema_version": 1`. The dashboard checks this; mismatched versions surface a warning but do not silently mis-render.

### Required keys (every episode must have these)

```jsonc
{
  "schema_version": 1,

  // Episode identity
  "object": "u_brown",                          // matches CSV filename and config file
  "base": "fmb1_base",                          // assembly base name
  "grasp_id": 0,                                // grasp point index used
  "wrapper_version": "compliant_insert.py@<git-sha>",  // for traceability

  // Timing
  "start_iso": "2026-05-02T20:14:33.521+02:00", // local ISO8601 with offset
  "end_iso":   "2026-05-02T20:14:58.107+02:00", // local ISO8601 with offset
  "duration_s": 24.586,                         // end - start, seconds

  // Outcome
  "outcome": "success",                         // one of: success | abort | timeout | crashed_no_meta
                                                // (crashed_no_meta is dashboard-assigned, never wrapper-written)
  "outcome_reason": "operator_sigterm",         // free-text short tag (e.g. "operator_sigterm",
                                                //   "timeout_reached", "operator_abort_signal",
                                                //   "ik_joint_limit_pre_check_failed")

  // Assembly target (fixed for episode, mirrored in every CSV row)
  "assembly_target_world": {
    "xyz_m": [0.4123, -0.0856, 0.0421],
    "quat_xyzw": [0.0, 0.0, 0.0, 1.0]
  },

  // Force-mode parameters used in ACTIVE (exactly what the wrapper sent to SetForceMode)
  "force_mode_params": {
    "task_frame": "base_link",
    "type": 2,                                  // NO_TRANSFORM
    "selection_vector": [true, true, true, true, true, true],   // [x, y, z, rx, ry, rz]
    "wrench": {                                 // commanded wrench, base_link frame
      "fx": 0.0, "fy": 0.0, "fz": -3.0,
      "tx": 0.0, "ty": 0.0, "tz": 0.0
    },
    "speed_limits": {
      "linear_xyz_m_s":  [0.02, 0.02, 0.02],
      "angular_xyz_r_s": [0.20, 0.20, 0.20]
    },
    "gain_scaling": 0.5,
    "damping_factor": 0.7
  },

  // Calibration provenance
  "foundational_calibration": {
    "yaml_path": "compliant_insertion_studio/configs/ft_calibration_onrobot_rg2_with_camera_20260501.yaml",
    "mass_kg": 1.0823,
    "cog_xyz_m": [0.0014, -0.0025, 0.0521],
    "calibration_age_days": 4
  },
  "smoke_test": {
    "result": "pass",                           // pass | fail | skipped
    "bias": {"Fx": 0.31, "Fy": -0.18, "Fz": 0.42, "Tx": 0.011, "Ty": -0.024, "Tz": 0.003},
    "drift_per_s": {"Fx_per_s": 0.012, "Fy_per_s": -0.008, "Fz_per_s": 0.041},
    "iso": "2026-05-02T20:13:55.012+02:00"
  },

  // Per-pose zero (PRE/ZERO phase result, before ACTIVE)
  "post_zero_bias": {
    "Fx": 0.18, "Fy": -0.09, "Fz": 0.21,
    "Tx": 0.005, "Ty": -0.012, "Tz": 0.001
  },
  "post_zero_drift_check": {                    // +1 s sample after zero (WRAP-04)
    "delta_t_s": 1.0,
    "Fx": 0.21, "Fy": -0.10, "Fz": 0.18,
    "max_axis_drift_n": 0.04                    // max |this - post_zero_bias|
  },

  // Hands-off window (WRAP-05 + DASH-07 signature card filter)
  "hands_off_window": {
    "start_iso": "2026-05-02T20:14:38.103+02:00",
    "end_iso":   "2026-05-02T20:14:58.107+02:00",
    "duration_s": 20.004,
    "trigger": "operator_step_back_confirmed"   // operator_step_back_confirmed | signal | auto
  },

  // Mid-episode re-zero events (SIGUSR2 — WRAP-10)
  "mid_episode_zero_events": [
    // {"t_s": 12.34, "post_zero_bias": {"Fx": ..., ...}}, ...
  ],

  // Operator narration (TELE-02, prompted at end-of-episode)
  "user_notes": "Started slightly off-axis, leaned into the part with my left hand to align."
}
```

### Optional / provenance keys (recommended but not required)

```jsonc
{
  "session_warmup_minutes": 18,                 // sensor warm-up before this episode
  "ros_distro": "humble",
  "ur_driver_version": "2.13.0",
  "ur_msgs_version": "2.4.0",
  "controller_at_start": "scaled_joint_trajectory_controller",
  "controllers_loaded": [
    "scaled_joint_trajectory_controller",
    "force_mode_controller",
    "passthrough_trajectory_controller",
    "io_and_status_controller",
    "force_torque_sensor_broadcaster",
    "tcp_pose_broadcaster"
  ],
  "joint_state_at_active_start": {
    "names": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
    "positions_rad": [0.12, -1.57, 1.23, -1.31, -1.57, 0.0]
  },
  "hover_pose_world": {                          // for IK-preflight forensics (WRAP-11)
    "xyz_m": [...],
    "quat_xyzw": [...]
  },
  "ik_pre_check": {
    "passed": true,
    "min_joint_margin_rad": 0.42                // smallest distance to any limit
  }
}
```

### Outcome state machine

| outcome | When the wrapper writes it |
|---|---|
| `success` | Wrapper reached `DONE` cleanly (timeout reached, or `SIGTERM` received) |
| `abort` | Operator sent the abort signal, OR a precondition failed (smoke test, IK pre-check, switch-controller verification) |
| `timeout` | Wrapper hit `--timeout` BEFORE `SIGTERM`. (Distinguished from `success` to surface "the algorithm did not autonomously terminate" — a key signal for Phase 5 termination-criterion derivation.) |
| `crashed_no_meta` | **Dashboard-assigned only**, never wrapper-written. CSV exists without matching meta JSON. |

---

## Schema versioning policy (TELE-05)

- v1 is locked at this milestone
- Adding optional keys to the meta JSON is NOT a version bump (dashboards must tolerate unknown keys)
- Adding CSV columns at the END of the row is NOT a version bump (PapaParse with header-mode is column-name-keyed; old dashboards just ignore new columns)
- ANY of the following IS a version bump (write `schema_version: 2` and update this file):
  - Removing or renaming a CSV column
  - Reordering CSV columns
  - Removing or renaming a meta JSON required key
  - Changing a column's units, frame, or sign convention
  - Changing the Euler decomposition convention (currently XYZ extrinsic)
  - Changing the float-formatting rules in a way that breaks parseFloat round-trips
  - Changing log file naming
- Bumps require a milestone decision and an entry in the "Schema history" section below.

## Schema history

| Version | Date | Author | Reason |
|---|---|---|---|
| v1 | 2026-05-02 | Phase 2 build | Initial lock. |

---

## Cross-references

- Wrapper code: `compliant_insertion_studio/wrapper/compliant_insert.py` (Phase 2 deliverable; produces this schema)
- Constants module: `compliant_insertion_studio/wrapper/schema_v1.py` (column names, phase enum, default formatters)
- Dashboard reader: `compliant_insertion_studio/analyzer/analyze_inserts.html` (Phase 4; consumes this schema)
- Calibration SOP: `compliant_insertion_studio/docs/ft_calibration_sop.md` (defines `foundational_calibration` and `smoke_test` blocks)
- Conventions: `.planning/codebase/CONVENTIONS.md` (calibration hierarchy, hands-off window, force ≤ 5 N defaults)
- Requirements traceability: `.planning/REQUIREMENTS.md` (TELE-01..06, WRAP-01..11)

"""
Stage A: episode preprocessor for the analyzer dashboard.

Inputs:  compliant_insertion_studio/logs/insert_<object>_<ts>.csv + .meta.json
Outputs: compliant_insertion_studio/logs/insert_<object>_<ts>.signals.json

For every episode pair this script computes:
  - logical_object grouping (u_brown + u_orange -> u_shape)
  - per-shape clean baseline (autonomous + bootstrap-from-quietest assisted)
  - 3 phase-segmentation methods (M1 force-domain, M2 motion-domain, M3 drift-from-ideal)
  - drift measurement (TCP_xy at DONE - target_xy + object rotation drift)

The HTML viewer (Stage B) reads CSV + signals.json and renders the 3 method
overlays so the operator can pick the best segmentation visually.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


LOGICAL_OBJECT_MAP = {
    "u_brown":           "u_shape",
    "u_orange":          "u_shape",
    "line_green":        "line_green",
    "inverted_u_yellow": "inverted_u_yellow",
}

# Channels used for nudge detection in M1 + baseline distribution.
# `power_lat` = |F_lat * v_lat| sample product, used by M3 — pooled directly
# (not as product-of-percentiles, which under-counts because peaks don't coincide).
M1_CHANNELS = ("F_lat", "T_lat", "Tz_abs", "v_lat")
M3_POWER_CHANNEL = "power_lat"
SAMPLE_HZ = 100  # logging rate (TELE-04)
SAMPLE_DT = 1.0 / SAMPLE_HZ


# ---------------------------------------------------------------------------
# Episode loading + derived columns
# ---------------------------------------------------------------------------

def load_episode(csv_path: Path):
    csv_path = Path(csv_path)
    meta_path = csv_path.with_name(csv_path.stem + ".meta.json")
    df = pd.read_csv(csv_path)
    df["hands_off"] = pd.to_numeric(df["hands_off"], errors="coerce").fillna(0).astype(int)
    df["event_marker"] = pd.to_numeric(df["event_marker"], errors="coerce").fillna(0).astype(int)
    with open(meta_path) as f:
        meta = json.load(f)
    return df, meta


def _quat_angular_velocity(quats: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Approximate body angular velocity (3 components, rad/s) from a sequence
    of quaternions sampled at non-uniform dt. Uses dq/dt ≈ 0.5 * w_quat * q.
    Returns Nx3 array; first row is zeros.
    """
    n = len(quats)
    if n < 2:
        return np.zeros((n, 3))
    out = np.zeros((n, 3))
    for i in range(1, n):
        q0 = quats[i - 1]
        q1 = quats[i]
        if np.dot(q0, q1) < 0:
            q1 = -q1
        # Relative rotation q_rel = q1 * inv(q0); axis-angle gives omega
        try:
            r_rel = R.from_quat(q1) * R.from_quat(q0).inv()
            rotvec = r_rel.as_rotvec()
        except Exception:
            rotvec = np.zeros(3)
        out[i] = rotvec / max(dt[i], 1e-4)
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Wrench is logged in tool0_controller (wrist) frame
    per SCHEMA.md; TCP velocities are in base_link. Computing F·v as a true
    physical power requires both vectors in the same frame, so we transform
    the wrench to base_link per row using the TCP quaternion.
    """
    df = df.copy()
    # Time derivatives (base_link frame)
    dt = df["t_s"].diff().fillna(SAMPLE_DT).clip(lower=1e-4)
    df["v_x"] = df["tcp_x"].diff().fillna(0) / dt
    df["v_y"] = df["tcp_y"].diff().fillna(0) / dt
    df["v_z"] = df["tcp_z"].diff().fillna(0) / dt
    df["v_lat"] = np.sqrt(df["v_x"] ** 2 + df["v_y"] ** 2)

    # Tool-frame magnitudes (kept — useful as direct readouts)
    df["F_lat"] = np.sqrt(df["fx"] ** 2 + df["fy"] ** 2)
    df["T_lat"] = np.sqrt(df["tx"] ** 2 + df["ty"] ** 2)
    df["Tz_abs"] = df["tz"].abs()

    # Transform wrench from tool0_controller -> base_link per row.
    # R_world_tool from TCP quat, then F_world = R_world_tool @ F_tool.
    quats = df[["tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]].to_numpy()
    forces_tool = df[["fx", "fy", "fz"]].to_numpy()
    torques_tool = df[["tx", "ty", "tz"]].to_numpy()
    rots = R.from_quat(quats)
    forces_base = rots.apply(forces_tool)
    torques_base = rots.apply(torques_tool)
    df["fx_base"] = forces_base[:, 0]
    df["fy_base"] = forces_base[:, 1]
    df["fz_base"] = forces_base[:, 2]
    df["tx_base"] = torques_base[:, 0]
    df["ty_base"] = torques_base[:, 1]
    df["tz_base"] = torques_base[:, 2]
    df["F_lat_base"] = np.sqrt(df["fx_base"] ** 2 + df["fy_base"] ** 2)

    # M3 power: signed dot product F_base · v in the lateral plane = real
    # external mechanical power going into the lateral DOFs (W).
    # Positive = force aligned with motion (operator pushing in direction of motion).
    df["power_lat_signed"] = df["fx_base"] * df["v_x"] + df["fy_base"] * df["v_y"]
    # Magnitude version is always-positive "lateral activity" (legacy / floor).
    df["power_lat"] = np.abs(df["power_lat_signed"])

    # Angular velocity from TCP quaternion derivatives (used by M5).
    omega = _quat_angular_velocity(quats, dt.to_numpy())
    df["omega_x"] = omega[:, 0]
    df["omega_y"] = omega[:, 1]
    df["omega_z"] = omega[:, 2]
    # M5 signal: torque × angular velocity coupling magnitude (rotational power).
    # Torque published in base frame is more meaningful here.
    df["rot_power_mag"] = np.abs(df["tx_base"] * df["omega_x"]
                                  + df["ty_base"] * df["omega_y"]
                                  + df["tz_base"] * df["omega_z"])

    # M7 signal: object-pose change rate vs TCP change rate.
    # If the object is rigidly held, obj_pose follows tcp_pose exactly. A
    # divergence implies grasp slip OR an external force above the wrist sensor
    # that shifts the object without the TCP responding. Computed as the L2
    # norm of (Δobj_xy − Δtcp_xy) per dt.
    if all(c in df.columns for c in ("obj_x", "obj_y")):
        dobj_x = df["obj_x"].diff().fillna(0)
        dobj_y = df["obj_y"].diff().fillna(0)
        dtcp_x = df["tcp_x"].diff().fillna(0)
        dtcp_y = df["tcp_y"].diff().fillna(0)
        slip_x = (dobj_x - dtcp_x) / dt
        slip_y = (dobj_y - dtcp_y) / dt
        df["slip_lat"] = np.sqrt(slip_x ** 2 + slip_y ** 2)
    else:
        df["slip_lat"] = 0.0
    return df


def is_autonomous(meta: dict) -> bool:
    return meta.get("assist_level") == "autonomous"


def shape_for(meta: dict) -> str:
    return LOGICAL_OBJECT_MAP.get(meta.get("object", ""), meta.get("object", "unknown"))


def active_handsoff(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["phase"] == "ACTIVE") & (df["hands_off"] == 1)].copy()


# ---------------------------------------------------------------------------
# Per-shape baseline (clean reference distribution)
# ---------------------------------------------------------------------------

def _pool_samples_from_dfs(dfs_with_masks):
    """Pool per-channel samples from a list of (df_active, sample_keep_mask) tuples.

    sample_keep_mask: boolean numpy array same length as df_active. Samples
    where the mask is False are excluded from the baseline pool.
    """
    pooled: dict[str, list] = {ch: [] for ch in M1_CHANNELS}
    pooled["fz"] = []
    pooled[M3_POWER_CHANNEL] = []
    for df_a, keep in dfs_with_masks:
        if df_a.empty:
            continue
        if keep is None:
            keep = np.ones(len(df_a), dtype=bool)
        for ch in M1_CHANNELS:
            if ch in df_a.columns:
                pooled[ch].extend(df_a[ch].to_numpy()[keep].tolist())
        pooled["fz"].extend(df_a["fz"].to_numpy()[keep].tolist())
        if M3_POWER_CHANNEL in df_a.columns:
            pooled[M3_POWER_CHANNEL].extend(
                df_a[M3_POWER_CHANNEL].to_numpy()[keep].tolist()
            )
    return {ch: np.array(v) for ch, v in pooled.items()}


def _channel_thresholds(channels: dict[str, np.ndarray]) -> dict:
    out: dict[str, dict] = {}
    for ch, samples in channels.items():
        if len(samples) == 0:
            out[ch] = {"median": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0}
            continue
        out[ch] = {
            "median": float(np.median(samples)),
            "p95":    float(np.percentile(samples, 95)),
            "p99":    float(np.percentile(samples, 99)),
            "p999":   float(np.percentile(samples, 99.9)),
        }
    return out


def compute_shape_baselines(episodes, refine_iters: int = 2) -> dict:
    """
    Build per-shape clean baselines by mining QUIET WINDOWS across ALL episodes,
    not whole episodes.

    Pass 0 (seed): pool ACTIVE+hands_off samples from autonomous traces only.
                   Compute initial M1 thresholds.
    Pass i+1:    Run the M1 force-domain detector on every episode of the shape
                   (auto + assisted). Mark all samples within any nudge interval
                   as "noisy"; keep the rest as quiet samples. Re-pool all quiet
                   samples across all episodes. Recompute thresholds.

    Iterates `refine_iters` times. Quiet-window pooling typically grows the
    baseline pool 10-30× larger than the autonomous-only seed, so percentile
    estimates converge to the real noise floor of robot-alone behavior.

    Returns dict[shape] -> {channels, thresholds, provenance}.
    """
    by_shape: dict[str, dict] = {}
    for csv_path, df, meta in episodes:
        shape = shape_for(meta)
        bucket = by_shape.setdefault(shape, {"autonomous": [], "assisted": []})
        df_a = add_derived(active_handsoff(df))
        bucket["autonomous" if is_autonomous(meta) else "assisted"].append(
            (csv_path, df_a, meta)
        )

    baselines: dict[str, dict] = {}
    for shape, traces in by_shape.items():
        # Seed: autonomous traces only, full samples (no mask)
        seed = [(d, None) for _, d, _ in traces["autonomous"]]
        channels = _pool_samples_from_dfs(seed) if seed else {ch: np.array([]) for ch in (*M1_CHANNELS, "fz", M3_POWER_CHANNEL)}
        thr = _channel_thresholds(channels)
        provenance = {
            "seed_n_autonomous": len(traces["autonomous"]),
            "seed_n_samples": int(sum(len(d) for _, d, _ in traces["autonomous"])),
            "iterations": [],
        }

        # If shape has no autonomous traces (shouldn't happen now, but defensive),
        # fall back to all assisted as the seed.
        if not seed and traces["assisted"]:
            seed = [(d, None) for _, d, _ in traces["assisted"]]
            channels = _pool_samples_from_dfs(seed)
            thr = _channel_thresholds(channels)
            provenance["seed_n_autonomous"] = 0
            provenance["seed_fallback_to_assisted"] = True

        all_traces = traces["autonomous"] + traces["assisted"]

        # Iterative refinement: mask out M1-triggered windows, repool
        for it in range(refine_iters):
            keep_masks = []
            n_total = 0
            n_kept = 0
            for csv_path, df_a, meta in all_traces:
                if df_a.empty:
                    keep_masks.append((df_a, np.array([], dtype=bool)))
                    continue
                runs = _m1_runs_against_thr(df_a, thr)
                mask = np.ones(len(df_a), dtype=bool)
                for a, b in runs:
                    mask[a:b + 1] = False
                keep_masks.append((df_a, mask))
                n_total += len(df_a)
                n_kept += int(mask.sum())

            channels = _pool_samples_from_dfs(keep_masks)
            new_thr = _channel_thresholds(channels)
            provenance["iterations"].append({
                "iter": it + 1,
                "n_quiet_samples": n_kept,
                "n_total_samples": n_total,
                "quiet_fraction": float(n_kept / max(n_total, 1)),
                "F_lat_p99": new_thr.get("F_lat", {}).get("p99", 0.0),
                "v_lat_p99": new_thr.get("v_lat", {}).get("p99", 0.0),
                "power_lat_p99": new_thr.get(M3_POWER_CHANNEL, {}).get("p99", 0.0),
            })
            thr = new_thr

        baselines[shape] = {
            "channels": channels,
            "thresholds": thr,
            "n_baseline_traces": len(all_traces),
            "n_autonomous": len(traces["autonomous"]),
            "n_assisted": len(traces["assisted"]),
            "provenance": provenance,
        }
    return baselines


def _m1_runs_against_thr(df_a: pd.DataFrame, thr: dict) -> list:
    """Return list of [start, end] sample-index pairs where any M1 channel
    exceeds its baseline p99 threshold (sustained ≥ 50 ms, i.e. ≥ 5 samples
    at 100 Hz). Used by quiet-window baseline refinement.
    """
    if df_a.empty:
        return []
    nudge_thr = {
        "F_lat":  max(thr.get("F_lat",  {}).get("p99", 1e9), 1.0),
        "T_lat":  max(thr.get("T_lat",  {}).get("p99", 1e9), 0.05),
        "Tz_abs": max(thr.get("Tz_abs", {}).get("p99", 1e9), 0.05),
        "v_lat":  max(thr.get("v_lat",  {}).get("p99", 1e9), 0.001),
    }
    above = np.zeros(len(df_a), dtype=bool)
    for ch, t in nudge_thr.items():
        if ch in df_a.columns:
            above |= (df_a[ch].to_numpy() > t)
    runs = _runs(above, min_len=max(1, int(0.05 * SAMPLE_HZ)))
    runs = _dilate_merge(runs, dilate=10, merge_gap=20, n=len(df_a))
    return runs


def baseline_thresholds(baseline: dict) -> dict:
    """Return the precomputed thresholds dict (set by compute_shape_baselines).
    Kept as a thin accessor for callers that previously computed thresholds lazily.
    """
    return baseline.get("thresholds", _channel_thresholds(baseline.get("channels", {})))


# ---------------------------------------------------------------------------
# Phase boundary helpers
# ---------------------------------------------------------------------------

def first_phase_index(df: pd.DataFrame, phase: str):
    rows = df.index[df["phase"] == phase]
    return int(rows[0]) if len(rows) else None


def find_start_phase_idx(df):
    return first_phase_index(df, "ACTIVE")


def find_termination(df: pd.DataFrame):
    """First DONE row, else last ACTIVE row."""
    done = first_phase_index(df, "DONE")
    if done is not None:
        return done
    active = df.index[df["phase"] == "ACTIVE"]
    return int(active[-1]) if len(active) else None


def _runs(mask: np.ndarray, min_len: int = 1):
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                runs.append([i, j - 1])
            i = j
        else:
            i += 1
    return runs


def _dilate_merge(intervals, dilate=10, merge_gap=20, n=0):
    if not intervals:
        return []
    expanded = [[max(0, a - dilate), min(n - 1, b + dilate)] for a, b in intervals]
    merged = [expanded[0]]
    for a, b in expanded[1:]:
        la, lb = merged[-1]
        if a - lb <= merge_gap:
            merged[-1] = [la, max(lb, b)]
        else:
            merged.append([a, b])
    return merged


def _idx_to_t(df, idx):
    if idx is None or idx < 0 or idx >= len(df):
        return None
    return float(df["t_s"].iloc[idx])


# ---------------------------------------------------------------------------
# Method 1 — Force-domain thresholds vs clean baseline
# ---------------------------------------------------------------------------

def method1_force(df: pd.DataFrame, thr: dict) -> dict:
    df = add_derived(df)
    start = find_start_phase_idx(df)
    term = find_termination(df)
    if start is None:
        return {"method": "M1_force", "error": "no ACTIVE phase"}

    sustain_short = max(1, int(0.05 * SAMPLE_HZ))   # 50 ms — for nudge runs
    sustain_contact = max(1, int(0.10 * SAMPLE_HZ))  # 100 ms — for first contact

    # First contact = smoothed fz crosses baseline_p99 sustained 100ms
    # NB fz in the data is positive when peg pushes back (tool frame)
    fz_thr = max(thr.get("fz", {}).get("p99", 1.0), 1.0)
    fz_smoothed = df["fz"].rolling(5, min_periods=1, center=True).mean()
    first_contact = None
    for i in range(start, len(df) - sustain_contact + 1):
        window = fz_smoothed.iloc[i:i + sustain_contact]
        if (window > fz_thr).all():
            first_contact = i
            break

    # Nudge mask: ANY of {F_lat, T_lat, Tz_abs, v_lat} above its baseline p99
    nudge_thr = {
        "F_lat":  max(thr.get("F_lat",  {}).get("p99", 1e9), 1.0),
        "T_lat":  max(thr.get("T_lat",  {}).get("p99", 1e9), 0.05),
        "Tz_abs": max(thr.get("Tz_abs", {}).get("p99", 1e9), 0.05),
        "v_lat":  max(thr.get("v_lat",  {}).get("p99", 1e9), 0.001),
    }
    in_active = (df["phase"] == "ACTIVE").values
    above = np.zeros(len(df), dtype=bool)
    for ch, t in nudge_thr.items():
        if ch in df:
            above |= (df[ch].values > t) & in_active

    runs = _runs(above, min_len=sustain_short)
    runs = _dilate_merge(runs, dilate=10, merge_gap=20, n=len(df))

    return {
        "method": "M1_force",
        "label":  "M1 — Force-domain thresholds (vs baseline p99)",
        "start_idx":         start,
        "first_contact_idx": first_contact,
        "termination_idx":   term,
        "start_t":           _idx_to_t(df, start),
        "first_contact_t":   _idx_to_t(df, first_contact),
        "termination_t":     _idx_to_t(df, term),
        "nudge_intervals":   runs,
        "nudge_intervals_t": [[_idx_to_t(df, a), _idx_to_t(df, b)] for a, b in runs],
        "thresholds_used":   nudge_thr,
        "fz_contact_threshold": float(fz_thr),
    }


# ---------------------------------------------------------------------------
# Method 2 — Kinematic (motion derivatives, force-blind)
#
# Lens: ignore F/T entirely. Look only at how the TCP moves. Nudges show up as
# unexpected lateral velocity peaks; first-contact shows up as deceleration in
# the descent rate.
# ---------------------------------------------------------------------------

def method2_motion(df: pd.DataFrame, thr: dict) -> dict:
    df = add_derived(df)
    active_idx = find_start_phase_idx(df)
    term = find_termination(df)
    if active_idx is None:
        return {"method": "M2_motion", "error": "no ACTIVE phase"}

    win = 11
    v_z_s = df["v_z"].rolling(win, min_periods=1, center=True).mean()
    v_lat_s = df["v_lat"].rolling(win, min_periods=1, center=True).mean()

    DESCENT_VEL = 0.005  # m/s — descent kinetic threshold (>5 mm/s downward)
    start = None
    for i in range(active_idx, len(df)):
        if v_z_s.iloc[i] < -DESCENT_VEL:
            start = i
            break

    # First contact: descent rate drops to ≤ 30% of the free-descent rate observed
    # in the first ~250 ms (25 samples at 100 Hz) after the start.
    first_contact = None
    initial_rate = None
    if start is not None:
        free_desc = v_z_s.iloc[start:start + 25].abs()
        if len(free_desc) > 5 and free_desc.median() > 1e-4:
            initial_rate = float(free_desc.median())
            for i in range(start + win, len(df)):
                if abs(v_z_s.iloc[i]) < initial_rate * 0.30:
                    first_contact = i
                    break

    # Nudge: lateral velocity exceeds 1.5× shape baseline v_lat p99
    # (with absolute floor of 4 mm/s — autonomous traces never exceed this).
    # Force-blind — we never look at fx/fy here.
    base_vlat_p99 = thr.get("v_lat", {}).get("p99", 0.003)
    v_lat_thr = max(base_vlat_p99 * 1.5, 0.004)
    above = (v_lat_s.values > v_lat_thr) & (df["phase"].values == "ACTIVE")
    runs = _runs(above, min_len=max(1, int(0.05 * SAMPLE_HZ)))
    runs = _dilate_merge(runs, dilate=10, merge_gap=20, n=len(df))

    return {
        "method": "M2_motion",
        "label":  "M2 — Kinematic (motion derivatives, force-blind)",
        "start_idx":         start,
        "first_contact_idx": first_contact,
        "termination_idx":   term,
        "start_t":           _idx_to_t(df, start),
        "first_contact_t":   _idx_to_t(df, first_contact),
        "termination_t":     _idx_to_t(df, term),
        "nudge_intervals":   runs,
        "nudge_intervals_t": [[_idx_to_t(df, a), _idx_to_t(df, b)] for a, b in runs],
        "descent_velocity_thr_m_s": DESCENT_VEL,
        "free_descent_rate_m_s":    initial_rate,
        "first_contact_v_z_factor": 0.30,
        "v_lat_threshold_m_s":      float(v_lat_thr),
    }


# ---------------------------------------------------------------------------
# Method 3 — Energetic (external mechanical work via F_lat · v_lat)
#
# Lens: the force_mode_controller commands fz only; commanded fx, fy, tx, ty,
# tz are zero. So `F_lat · v_lat` is, modulo friction transients, the rate of
# external work being injected into the lateral degrees of freedom. Sustained
# positive external power = an external agent (operator) is doing work on the
# part. Friction between peg + chamfer also contributes; we filter that out
# by requiring sustained windows.
# ---------------------------------------------------------------------------

def method3_energy(df: pd.DataFrame, thr: dict) -> dict:
    df = add_derived(df)
    active_idx = find_start_phase_idx(df)
    term = find_termination(df)
    if active_idx is None:
        return {"method": "M3_energy", "error": "no ACTIVE phase"}

    # Instantaneous |F_lat * v_lat| — magnitude of lateral mechanical power.
    # We use magnitude (not signed) because friction reaction can momentarily
    # invert the sign without changing the fact that energy is being dissipated
    # in the lateral DOF — we want both directions counted as "lateral activity".
    p_lat_inst = df["F_lat"].values * df["v_lat"].values
    p_lat_smoothed = pd.Series(p_lat_inst).rolling(11, min_periods=1, center=True).mean()

    # Threshold: 3× shape baseline power_lat p99 (computed on autonomous +
    # bootstrap-quietest pool). Floor at 5 mW. Empirically autonomous noise
    # floor sits ~3-6 mW; assists range 8-50 mW.
    base_p_p99 = thr.get("power_lat", {}).get("p99", 0.003)
    p_thr_active = max(base_p_p99 * 3.0, 0.005)

    above = (p_lat_smoothed.values > p_thr_active) & (df["phase"].values == "ACTIVE")
    runs = _runs(above, min_len=max(1, int(0.05 * SAMPLE_HZ)))
    runs = _dilate_merge(runs, dilate=10, merge_gap=20, n=len(df))

    # Cumulative lateral mechanical work over ACTIVE (J) — useful per-episode scalar
    in_active = (df["phase"].values == "ACTIVE")
    dt_s = df["t_s"].diff().fillna(SAMPLE_DT).values
    work_increments = np.where(in_active, p_lat_smoothed.values * dt_s, 0.0)
    cumulative_work_active_J = float(np.sum(work_increments))

    # Start = first row in ACTIVE where commanded_fz transitions to non-zero
    start = None
    for i in range(active_idx, len(df)):
        if abs(df["commanded_fz"].iloc[i]) > 0.01:
            start = i
            break

    # First contact = first row in ACTIVE where smoothed fz crosses the
    # baseline fz p99 (peg is pushing back). Independent of nudge intervals
    # so M3 still flags first-contact even on a perfectly autonomous trace.
    first_contact = None
    fz_smoothed = df["fz"].rolling(5, min_periods=1, center=True).mean().values
    fz_thr = max(thr.get("fz", {}).get("p99", 1.0), 1.0)
    sustain = max(1, int(0.10 * SAMPLE_HZ))
    for i in range(active_idx, len(df) - sustain + 1):
        if (fz_smoothed[i:i + sustain] > fz_thr).all():
            first_contact = i
            break

    return {
        "method": "M3_energy",
        "label":  "M3 — Energetic (external lateral work F_lat · v_lat)",
        "start_idx":         start,
        "first_contact_idx": first_contact,
        "termination_idx":   term,
        "start_t":           _idx_to_t(df, start),
        "first_contact_t":   _idx_to_t(df, first_contact),
        "termination_t":     _idx_to_t(df, term),
        "nudge_intervals":   runs,
        "nudge_intervals_t": [[_idx_to_t(df, a), _idx_to_t(df, b)] for a, b in runs],
        "p_ext_threshold_W":            float(p_thr_active),
        "cumulative_external_work_J":   cumulative_work_active_J,
    }


# ---------------------------------------------------------------------------
# Method 5 — Torque–motion coupling (rotational power)
#
# Operator nudges sometimes manifest as twists/rotations rather than lateral
# pushes. Tx, Ty, Tz × angular velocity captures rotational mechanical power.
# ---------------------------------------------------------------------------

def method5_torque_motion(df: pd.DataFrame) -> dict:
    df = add_derived(df)
    active_idx = find_start_phase_idx(df)
    term = find_termination(df)
    if active_idx is None:
        return {"method": "M5_torque", "error": "no ACTIVE phase"}

    rp = df["rot_power_mag"].rolling(11, min_periods=1, center=True).mean()
    in_active = (df["phase"].values == "ACTIVE")
    rp_active = rp.values[in_active] if in_active.any() else rp.values
    if len(rp_active) > 5:
        # Robust threshold: median + 5 × MAD (median absolute deviation)
        med = float(np.median(rp_active))
        mad = float(np.median(np.abs(rp_active - med))) or 1e-6
        rp_thr = max(med + 5.0 * 1.4826 * mad, 0.005)
    else:
        rp_thr = 0.01

    above = (rp.values > rp_thr) & in_active
    runs = _runs(above, min_len=max(1, int(0.05 * SAMPLE_HZ)))
    runs = _dilate_merge(runs, dilate=10, merge_gap=20, n=len(df))

    return {
        "method": "M5_torque",
        "label":  "M5 — Torque–motion coupling (rotational power)",
        "start_idx":         find_start_phase_idx(df),
        "first_contact_idx": None,
        "termination_idx":   term,
        "start_t":           _idx_to_t(df, find_start_phase_idx(df)),
        "first_contact_t":   None,
        "termination_t":     _idx_to_t(df, term),
        "nudge_intervals":   runs,
        "nudge_intervals_t": [[_idx_to_t(df, a), _idx_to_t(df, b)] for a, b in runs],
        "rot_power_threshold_W": float(rp_thr),
    }


# ---------------------------------------------------------------------------
# Method 7 — Object/TCP relative slip
#
# If an external force is applied above the wrist (operator pushing on the
# robot's link), the F/T sensor at the wrist won't see it but the object's
# pose will shift relative to the TCP. M7 fires on object xy-displacement
# rate that exceeds the TCP xy-displacement rate.
# ---------------------------------------------------------------------------

def method7_object_slip(df: pd.DataFrame) -> dict:
    df = add_derived(df)
    active_idx = find_start_phase_idx(df)
    term = find_termination(df)
    if active_idx is None or "slip_lat" not in df.columns:
        return {"method": "M7_slip", "error": "no ACTIVE or no obj_*"}

    slip = df["slip_lat"].rolling(11, min_periods=1, center=True).mean()
    in_active = (df["phase"].values == "ACTIVE")
    slip_active = slip.values[in_active] if in_active.any() else slip.values
    if len(slip_active) > 5:
        med = float(np.median(slip_active))
        mad = float(np.median(np.abs(slip_active - med))) or 1e-6
        slip_thr = max(med + 5.0 * 1.4826 * mad, 0.002)
    else:
        slip_thr = 0.005

    above = (slip.values > slip_thr) & in_active
    runs = _runs(above, min_len=max(1, int(0.05 * SAMPLE_HZ)))
    runs = _dilate_merge(runs, dilate=10, merge_gap=20, n=len(df))

    return {
        "method": "M7_slip",
        "label":  "M7 — Object/TCP relative slip (above-wrist + grasp slip)",
        "start_idx":         active_idx,
        "first_contact_idx": None,
        "termination_idx":   term,
        "start_t":           _idx_to_t(df, active_idx),
        "first_contact_t":   None,
        "termination_t":     _idx_to_t(df, term),
        "nudge_intervals":   runs,
        "nudge_intervals_t": [[_idx_to_t(df, a), _idx_to_t(df, b)] for a, b in runs],
        "slip_threshold_m_s": float(slip_thr),
    }


# ---------------------------------------------------------------------------
# Per-episode feature summary (used by the cross-episode scatter view)
# ---------------------------------------------------------------------------

def compute_episode_features(df: pd.DataFrame, methods_out: list) -> dict:
    df = add_derived(df)
    active = df[df["phase"] == "ACTIVE"]
    if active.empty:
        return {}
    feat = {
        "n_active_samples":      int(len(active)),
        "active_duration_s":     float(active["t_s"].iloc[-1] - active["t_s"].iloc[0]),
        "max_F_lat":             float(active["F_lat"].max()),
        "max_T_lat":             float(active["T_lat"].max()),
        "max_v_lat":             float(active["v_lat"].max()),
        "max_power_lat":         float(active["power_lat"].max()),
        "max_rot_power":         float(active.get("rot_power_mag", pd.Series([0])).max()),
        "max_slip_lat":          float(active.get("slip_lat", pd.Series([0])).max()),
        "median_fz_active":      float(active["fz"].median()),
        "z_descent_m":           float(active["tcp_z"].iloc[0] - active["tcp_z"].iloc[-1]),
        "xy_travel_m":           float(np.sum(np.sqrt(active["tcp_x"].diff().fillna(0)**2 + active["tcp_y"].diff().fillna(0)**2))),
    }
    # Pull cumulative_external_work from M3 if present
    for m in methods_out:
        if m.get("method") == "M3_energy":
            feat["total_external_work_J"] = float(m.get("cumulative_external_work_J", 0.0))
    return feat


# ---------------------------------------------------------------------------
# Drift / grasp-error measurement at termination
# ---------------------------------------------------------------------------

def compute_drift_signature(df: pd.DataFrame, meta: dict) -> dict:
    term = find_termination(df)
    if term is None:
        return {}

    last = df.iloc[term]
    target_xyz = np.array(meta["assembly_target_world"]["xyz_m"])
    target_quat = np.array(meta["assembly_target_world"]["quat_xyzw"])

    tcp_xyz = np.array([last["tcp_x"], last["tcp_y"], last["tcp_z"]])
    tcp_quat = np.array([last["tcp_qx"], last["tcp_qy"], last["tcp_qz"], last["tcp_qw"]])

    dxy = (tcp_xyz - target_xyz)[:2]
    lateral_xy_drift_m = float(np.linalg.norm(dxy))

    obj_quat = None
    if all(c in df.columns for c in ("obj_qx", "obj_qy", "obj_qz", "obj_qw")):
        obj_quat = np.array([last["obj_qx"], last["obj_qy"],
                             last["obj_qz"], last["obj_qw"]])

    obj_rpy_drift_deg = None
    if obj_quat is not None and not np.isnan(obj_quat).any():
        try:
            q_err = (R.from_quat(target_quat).inv()
                     * R.from_quat(obj_quat)).as_euler("xyz", degrees=True)
            obj_rpy_drift_deg = q_err.tolist()
        except Exception:
            obj_rpy_drift_deg = None

    return {
        "lateral_xy_drift_m":     lateral_xy_drift_m,
        "tcp_minus_target_xyz_m": (tcp_xyz - target_xyz).tolist(),
        "obj_rpy_drift_deg":      obj_rpy_drift_deg,
        "tcp_at_done":            {"xyz_m": tcp_xyz.tolist(), "quat_xyzw": tcp_quat.tolist()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="compliant_insertion_studio/logs")
    parser.add_argument("--refine-iters", type=int, default=2,
                        help="quiet-window baseline refinement iterations per shape")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    csv_files = sorted(logs_dir.glob("insert_*.csv"))
    print(f"Found {len(csv_files)} CSVs in {logs_dir}")

    episodes = []
    for csv_path in csv_files:
        try:
            df, meta = load_episode(csv_path)
            episodes.append((csv_path, df, meta))
        except Exception as e:
            print(f"  ! load failed: {csv_path.name}: {e}")
    print(f"Loaded {len(episodes)} episode pairs")

    baselines = compute_shape_baselines(episodes, refine_iters=args.refine_iters)
    print("\n=== Per-shape baselines (quiet-window pooled) ===")
    for shape, b in baselines.items():
        thr = baseline_thresholds(b)
        prov = b.get("provenance", {})
        last_iter = prov.get("iterations", [{}])[-1] if prov.get("iterations") else {}
        print(f"  {shape}: traces={b['n_baseline_traces']} (auto={b['n_autonomous']}, assisted={b['n_assisted']})")
        print(f"    seed: {prov.get('seed_n_autonomous',0)} autonomous trace(s), {prov.get('seed_n_samples',0)} samples")
        for it in prov.get("iterations", []):
            print(f"    iter {it['iter']}: kept {it['n_quiet_samples']}/{it['n_total_samples']} samples ({100*it['quiet_fraction']:.1f}% quiet)  F_lat_p99={it['F_lat_p99']:.3f}  power_lat_p99={it['power_lat_p99']:.5f}")
        for ch in ("F_lat", "T_lat", "Tz_abs", "v_lat", "fz", "power_lat"):
            t = thr.get(ch, {})
            print(f"    {ch:>9}: median={t.get('median',0):.5f}  p95={t.get('p95',0):.5f}  p99={t.get('p99',0):.5f}  p999={t.get('p999',0):.5f}")

    summary = []
    print("\n=== Writing signals.json sidecars ===")
    for csv_path, df, meta in episodes:
        shape = shape_for(meta)
        thr = baseline_thresholds(baselines[shape])

        m1 = method1_force(df, thr)
        m2 = method2_motion(df, thr)
        m3 = method3_energy(df, thr)
        m5 = method5_torque_motion(df)
        m7 = method7_object_slip(df)
        methods_out = [m1, m2, m3, m5, m7]
        drift = compute_drift_signature(df, meta)
        features = compute_episode_features(df, methods_out)

        out = {
            "schema_version": 1,
            "episode": {
                "object":          meta.get("object"),
                "logical_shape":   shape,
                "assist_level":    meta.get("assist_level"),
                "outcome":         meta.get("outcome"),
                "duration_s":      meta.get("duration_s"),
                "user_notes":      meta.get("user_notes"),
                "csv_basename":    csv_path.name,
            },
            "baseline_used": {
                "shape":              shape,
                "n_baseline_traces":  baselines[shape]["n_baseline_traces"],
                "n_autonomous":       baselines[shape]["n_autonomous"],
                "n_assisted":         baselines[shape]["n_assisted"],
                "thresholds":         thr,
                "provenance":         baselines[shape].get("provenance", {}),
            },
            "methods": methods_out,
            "drift":    drift,
            "features": features,
        }
        sidecar = csv_path.with_name(csv_path.stem + ".signals.json")
        with open(sidecar, "w") as f:
            json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)

        summary.append({
            "csv":     csv_path.name,
            "shape":   shape,
            "assist":  meta.get("assist_level", "?"),
            "m1_n":    len(m1.get("nudge_intervals", [])),
            "m2_n":    len(m2.get("nudge_intervals", [])),
            "m3_n":    len(m3.get("nudge_intervals", [])),
            "m5_n":    len(m5.get("nudge_intervals", [])),
            "m7_n":    len(m7.get("nudge_intervals", [])),
            "lat_mm":  drift.get("lateral_xy_drift_m", 0) * 1000,
        })

    print(f"\n=== Per-episode summary ({len(summary)} episodes) ===")
    print(f"{'csv':<50}{'shape':<22}{'assist':<12}{'M1':>4}{'M2':>4}{'M3':>4}{'M5':>4}{'M7':>4}{'drift_mm':>10}")
    for r in summary:
        print(f"{r['csv']:<50}{r['shape']:<22}{r['assist']:<12}{r['m1_n']:>4}{r['m2_n']:>4}{r['m3_n']:>4}{r['m5_n']:>4}{r['m7_n']:>4}{r['lat_mm']:>10.2f}")

    print(f"\n[OK] wrote {len(summary)} signals.json sidecars")


if __name__ == "__main__":
    main()

"""
Per-demo feature extractor over the GUIDED segment.

Slices each insert_u_orange_*.csv into:
  - GUIDED:        [t_contact, t_sigusr1)        — operator drags peg on rim
  - INSERT_DESCENT:[t_sigusr1, t_end]            — pure Z descent at hole_xy

Computes candidate F/T-derived signals to feed Phase C (Found Hole detector)
and Phase D (search director regression). Bias is subtracted from the wrench
using meta.post_zero_bias before any feature is computed.

Output: <bn>.features.json per demo + manifest.json across demos.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from _paths import LOG_DIR, DATA_DIR  # noqa: E402

OUT_DIR = DATA_DIR / "guided_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTACT_THRESHOLD_N = 3.0
CONTACT_SUSTAIN_SAMPLES = 10  # 0.1s at 100Hz
APPROACH_GRACE_S = 1.0


def _iso_to_epoch(s: str) -> float:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def _smooth(x: np.ndarray, window_s: float, dt: float) -> np.ndarray:
    n = max(1, int(round(window_s / dt)))
    if n <= 1 or len(x) < n:
        return x.copy()
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")


def _segment_indices(t_s: np.ndarray, fz_abs: np.ndarray, t_sigusr1_rel: float):
    """
    Returns (i_contact, i_sigusr1) within the ACTIVE-row arrays.

    Contact: first index where fz_abs > 3N for 10 consecutive samples,
             after the grace period from the start of ACTIVE.
    SIGUSR1: nearest index to t_sigusr1_rel.
    """
    t0 = t_s[0]
    grace_end = t0 + APPROACH_GRACE_S
    i_contact = None
    for i in range(len(fz_abs) - CONTACT_SUSTAIN_SAMPLES):
        if t_s[i] < grace_end:
            continue
        if all(fz_abs[i + j] > CONTACT_THRESHOLD_N for j in range(CONTACT_SUSTAIN_SAMPLES)):
            i_contact = i
            break
    if i_contact is None:
        return None, None
    i_sigusr1 = int(np.argmin(np.abs(t_s - t_sigusr1_rel)))
    if i_sigusr1 <= i_contact:
        return None, None
    return i_contact, i_sigusr1


def _ee_z_in_world(quats: np.ndarray) -> np.ndarray:
    """quats shape (N, 4) in (x,y,z,w). Returns world-frame EE +Z axis (N, 3)."""
    rots = R.from_quat(quats)
    # The EE-Z axis is the third column of the rotation matrix.
    mats = rots.as_matrix()  # (N, 3, 3)
    return mats[:, :, 2]


def _tilt_features(ee_z_world: np.ndarray):
    """
    tilt_deg: angle between EE-Z and world +Z (rim hits peg from below → EE-Z is roughly +world_Z when face-down).
    tilt_xy: signed projection onto world XY plane (direction the peg is cocked TOWARDS).
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    cos_t = np.clip(ee_z_world @ z_axis, -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(cos_t))
    # Signed components: project EE-Z onto world XY (sign-preserving).
    # For a face-down EE, EE-Z ≈ +world_Z. If the peg cocks +X, EE-Z ≈ (sin_eps, 0, cos_eps)
    # and tilt_x = +sin_eps. The peg is leaning IN that XY direction.
    tilt_x = ee_z_world[:, 0]
    tilt_y = ee_z_world[:, 1]
    return tilt_deg, tilt_x, tilt_y


def _rotate_vec_tool_to_world(quats: np.ndarray, vec_tool: np.ndarray) -> np.ndarray:
    """quats (N,4) xyzw; vec_tool (N,3) in tool frame; returns vec_world (N,3)."""
    rots = R.from_quat(quats)
    return rots.apply(vec_tool)


def _process_demo(meta_path: str) -> dict | None:
    meta = json.load(open(meta_path))
    hole = meta.get("hole_observed_operator")
    if not hole:
        return None
    bn = os.path.basename(meta_path).replace(".meta.json", "")
    csv_path = meta_path.replace(".meta.json", ".csv")
    if not os.path.exists(csv_path):
        return None

    bias = meta.get("post_zero_bias") or {}
    bx, by, bz = bias.get("Fx", 0.0), bias.get("Fy", 0.0), bias.get("Fz", 0.0)
    btx, bty, btz = bias.get("Tx", 0.0), bias.get("Ty", 0.0), bias.get("Tz", 0.0)

    raw_ts = float(hole["t_s"])
    if raw_ts > 1e9:
        # wall-time epoch (source = fsm_guided_sigusr1)
        t_start_wall = _iso_to_epoch(meta["start_iso"])
        t_sigusr1_rel = raw_ts - t_start_wall
    else:
        # already CSV-relative (source = fsm_guided_sigusr1_backfill_from_csv)
        t_sigusr1_rel = raw_ts

    # Read ACTIVE rows
    rows = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r_ in rdr:
            if r_.get("phase") == "ACTIVE":
                rows.append(r_)
    if len(rows) < 100:
        return None

    def col(key):
        return np.array([float(r_[key]) for r_ in rows], dtype=float)

    t_s = col("t_s")
    tcp_x = col("tcp_x"); tcp_y = col("tcp_y"); tcp_z = col("tcp_z")
    qx = col("tcp_qx"); qy = col("tcp_qy"); qz = col("tcp_qz"); qw = col("tcp_qw")
    fx = col("fx") - bx; fy = col("fy") - by; fz = col("fz") - bz
    tx = col("tx") - btx; ty = col("ty") - bty; tz_t = col("tz") - btz

    quats = np.column_stack([qx, qy, qz, qw])

    dt = float(np.median(np.diff(t_s)))
    if not (0.005 < dt < 0.05):
        return None  # weird sample rate

    # Segment
    fz_abs = np.abs(fz)
    fz_smooth = _smooth(fz_abs, 0.1, dt)
    i_c, i_s = _segment_indices(t_s, fz_smooth, t_sigusr1_rel)
    if i_c is None:
        return None

    n = len(rows)

    # Velocities (centered diff, smoothed)
    def cdiff(x):
        v = np.zeros_like(x)
        v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
        v[0] = (x[1] - x[0]) / dt
        v[-1] = (x[-1] - x[-2]) / dt
        return v

    vx = _smooth(cdiff(tcp_x), 0.2, dt)
    vy = _smooth(cdiff(tcp_y), 0.2, dt)
    vz = _smooth(cdiff(tcp_z), 0.1, dt)

    drag_speed = np.hypot(vx, vy)
    # Drag direction unit vector (world XY); zero when speed too low to be reliable.
    eps = 1e-6
    safe = drag_speed > 5e-4  # 0.5 mm/s
    drag_ux = np.where(safe, vx / np.maximum(drag_speed, eps), 0.0)
    drag_uy = np.where(safe, vy / np.maximum(drag_speed, eps), 0.0)

    # Lateral wrench in tool frame
    F_lat_tool = np.hypot(fx, fy)
    # Smoothed components for use in derivatives
    fx_s = _smooth(fx, 0.05, dt)
    fy_s = _smooth(fy, 0.05, dt)
    fz_s = _smooth(fz, 0.1, dt)

    # r_cop in tool frame (mm). Guarded against tiny |fz| via 0.5N floor.
    fz_for_cop = np.where(np.abs(fz_s) < 0.5, np.copysign(0.5, fz_s + 1e-9), fz_s)
    rcop_x_tool = (-ty) / fz_for_cop  # m
    rcop_y_tool = (tx) / fz_for_cop   # m
    rcop_mag = np.hypot(rcop_x_tool, rcop_y_tool)
    rcop_arg = np.arctan2(rcop_y_tool, rcop_x_tool)

    # Rotate (-r_cop) tool-frame vector to world XY → "direction toward hole"-hypothesis vector
    minus_rcop_tool = np.column_stack([-rcop_x_tool, -rcop_y_tool, np.zeros(n)])
    minus_rcop_world = _rotate_vec_tool_to_world(quats, minus_rcop_tool)
    minus_rcop_xy_norm = np.hypot(minus_rcop_world[:, 0], minus_rcop_world[:, 1])
    rcop_world_ux = np.where(minus_rcop_xy_norm > eps,
                             minus_rcop_world[:, 0] / np.maximum(minus_rcop_xy_norm, eps), 0.0)
    rcop_world_uy = np.where(minus_rcop_xy_norm > eps,
                             minus_rcop_world[:, 1] / np.maximum(minus_rcop_xy_norm, eps), 0.0)

    # Rotate F_lat tool vector to world (also a candidate gradient: peg pushes back along contact normal)
    f_lat_tool_3 = np.column_stack([fx_s, fy_s, np.zeros(n)])
    f_lat_world = _rotate_vec_tool_to_world(quats, f_lat_tool_3)
    f_lat_world_x = f_lat_world[:, 0]
    f_lat_world_y = f_lat_world[:, 1]
    F_lat_world_mag = np.hypot(f_lat_world_x, f_lat_world_y)
    # "Drag-with-resistance" hypothesis: operator drags AGAINST the contact resistance (i.e., -F_lat_world dir = direction into the rim, hole side)
    # Ambiguous a priori — let regression decide sign.

    # Tilt features
    ee_z_world = _ee_z_in_world(quats)
    tilt_deg, tilt_x, tilt_y = _tilt_features(ee_z_world)
    tilt_xy_norm = np.hypot(tilt_x, tilt_y)
    tilt_ux = np.where(tilt_xy_norm > eps, tilt_x / np.maximum(tilt_xy_norm, eps), 0.0)
    tilt_uy = np.where(tilt_xy_norm > eps, tilt_y / np.maximum(tilt_xy_norm, eps), 0.0)

    # Drag-vs-(-r_cop) alignment per tick (dot product, world XY)
    dot_drag_minus_rcop = drag_ux * rcop_world_ux + drag_uy * rcop_world_uy
    # Drag-vs-tilt alignment (does the operator drag IN the direction of tilt? AGAINST it?)
    dot_drag_tilt = drag_ux * tilt_ux + drag_uy * tilt_uy
    # Drag-vs-F_lat alignment (signed)
    fl_norm = np.maximum(F_lat_world_mag, eps)
    dot_drag_flat = (drag_ux * f_lat_world_x + drag_uy * f_lat_world_y) / fl_norm

    # Path length cumulative during GUIDED
    seg_len = np.hypot(np.diff(tcp_x), np.diff(tcp_y))
    seg_len = np.concatenate([[0.0], seg_len])
    path_cum = np.cumsum(seg_len)

    # SIGUSR1 row anchor → distances back-from-hole during GUIDED
    hole_xy = np.array(hole["xy_m"], dtype=float)
    dist_to_hole = np.hypot(tcp_x - hole_xy[0], tcp_y - hole_xy[1])

    # Slice GUIDED + INSERT_DESCENT segments
    guided_slice = slice(i_c, i_s)
    descent_slice = slice(i_s, n)

    def pack(sl):
        def ll(arr):
            return arr[sl].tolist()
        return {
            "t_s": ll(t_s),
            "tcp_x": ll(tcp_x), "tcp_y": ll(tcp_y), "tcp_z": ll(tcp_z),
            "tcp_qx": ll(qx), "tcp_qy": ll(qy), "tcp_qz": ll(qz), "tcp_qw": ll(qw),
            "fx": ll(fx), "fy": ll(fy), "fz": ll(fz),
            "tx": ll(tx), "ty": ll(ty), "tz": ll(tz_t),
            "fz_smooth": ll(fz_s),
            "F_lat_tool": ll(F_lat_tool),
            "F_lat_world_mag": ll(F_lat_world_mag),
            "f_lat_world_x": ll(f_lat_world_x),
            "f_lat_world_y": ll(f_lat_world_y),
            "rcop_x_tool_m": ll(rcop_x_tool),
            "rcop_y_tool_m": ll(rcop_y_tool),
            "rcop_mag_m": ll(rcop_mag),
            "rcop_arg_rad": ll(rcop_arg),
            "minus_rcop_world_ux": ll(rcop_world_ux),
            "minus_rcop_world_uy": ll(rcop_world_uy),
            "tilt_deg": ll(tilt_deg),
            "tilt_x": ll(tilt_x), "tilt_y": ll(tilt_y),
            "tilt_ux": ll(tilt_ux), "tilt_uy": ll(tilt_uy),
            "vx": ll(vx), "vy": ll(vy), "vz": ll(vz),
            "drag_speed": ll(drag_speed),
            "drag_ux": ll(drag_ux), "drag_uy": ll(drag_uy),
            "drag_safe": [bool(x) for x in safe[sl]],
            "dot_drag_minus_rcop": ll(dot_drag_minus_rcop),
            "dot_drag_tilt": ll(dot_drag_tilt),
            "dot_drag_flat": ll(dot_drag_flat),
            "path_cum_m": ll(path_cum - path_cum[sl][0] if len(path_cum[sl]) else path_cum),
            "dist_to_hole_m": ll(dist_to_hole),
        }

    summary = {
        "i_contact": int(i_c),
        "i_sigusr1": int(i_s),
        "n_active": n,
        "dt_s": dt,
        "t_contact_s": float(t_s[i_c]),
        "t_sigusr1_s": float(t_s[i_s]),
        "guided_dur_s": float(t_s[i_s] - t_s[i_c]),
        "descent_dur_s": float(t_s[-1] - t_s[i_s]),
        "guided_dxy_m": [float(tcp_x[i_s] - tcp_x[i_c]), float(tcp_y[i_s] - tcp_y[i_c])],
        "guided_dxy_path_m": float(path_cum[i_s] - path_cum[i_c]),
        "guided_dz_m": float(tcp_z[i_s] - tcp_z[i_c]),
        "descent_dxy_m": [float(tcp_x[-1] - tcp_x[i_s]), float(tcp_y[-1] - tcp_y[i_s])],
        "descent_dz_m": float(tcp_z[-1] - tcp_z[i_s]),
        "drag_align_minus_rcop_mean_guided": float(
            np.mean(dot_drag_minus_rcop[i_c:i_s][safe[i_c:i_s]])
            if np.any(safe[i_c:i_s]) else 0.0),
        "drag_align_tilt_mean_guided": float(
            np.mean(dot_drag_tilt[i_c:i_s][safe[i_c:i_s]])
            if np.any(safe[i_c:i_s]) else 0.0),
        "drag_align_flat_mean_guided": float(
            np.mean(dot_drag_flat[i_c:i_s][safe[i_c:i_s]])
            if np.any(safe[i_c:i_s]) else 0.0),
        "fz_at_sigusr1_smooth": float(fz_s[i_s]),
        "fz_at_sigusr1_minus_pre_window_mean_300ms": float(
            fz_s[i_s] - np.mean(fz_s[max(0, i_s - 30):i_s])),
        "tilt_at_sigusr1_deg": float(tilt_deg[i_s]),
        "tilt_peak_pre_300ms_deg": float(np.max(tilt_deg[max(0, i_s - 30):i_s])
                                          if i_s > 0 else 0.0),
        "rcop_mag_at_sigusr1_mm": float(1000 * rcop_mag[i_s]),
        "rcop_mag_pre_window_mean_300ms_mm": float(
            1000 * np.mean(rcop_mag[max(0, i_s - 30):i_s])),
        "F_lat_at_sigusr1_N": float(F_lat_world_mag[i_s]),
        "F_lat_pre_window_mean_300ms_N": float(
            np.mean(F_lat_world_mag[max(0, i_s - 30):i_s])),
        "vz_at_sigusr1_m_s": float(vz[i_s]),
        "vz_post_300ms_m_s": float(np.mean(vz[i_s:min(n, i_s + 30)])),
        "hole_observed_xy_m": list(hole["xy_m"]),
        "wrench_frame_id": rows[0].get("wrench_frame_id"),
    }

    return {
        "basename": bn,
        "summary": summary,
        "guided": pack(guided_slice),
        "descent": pack(descent_slice),
    }


def main():
    meta_paths = sorted(glob.glob(str(LOG_DIR / "insert_u_orange_2026050[5-6]*.meta.json")),
                        key=os.path.getmtime)
    manifest = {"demos": []}
    ok = 0
    for mp in meta_paths:
        try:
            out = _process_demo(mp)
        except Exception as e:
            out = None
            print(f"  FAIL {os.path.basename(mp)}: {e}", file=sys.stderr)
        if out is None:
            continue
        path = OUT_DIR / f"{out['basename']}.features.json"
        with open(path, "w") as f:
            json.dump(out, f)
        manifest["demos"].append({
            "basename": out["basename"],
            "summary": out["summary"],
            "path": str(path.relative_to(DATA_DIR.parent.parent)),
        })
        ok += 1
        s = out["summary"]
        print(f"  OK {out['basename']}: guided={s['guided_dur_s']:.2f}s "
              f"path={1000*s['guided_dxy_path_m']:.1f}mm "
              f"align(-rcop)={s['drag_align_minus_rcop_mean_guided']:+.2f} "
              f"align(tilt)={s['drag_align_tilt_mean_guided']:+.2f} "
              f"align(flat)={s['drag_align_flat_mean_guided']:+.2f}")

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{ok}/{len(meta_paths)} demos extracted → {OUT_DIR}")


if __name__ == "__main__":
    main()

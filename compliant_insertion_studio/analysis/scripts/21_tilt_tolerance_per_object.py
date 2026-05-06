#!/usr/bin/env python3
# Reference: A1 — derive per-object TCP tilt-tolerance envelope from GOLD operator demos.
#
# Used by the iter-10 control law (insertion-control-law-derivation skill, §13):
#   - tilt > tolerance → override seat-prior steering with tilt-direction steering
#   - tilt relaxation event = chamfer engaged → latch new seat estimate
#
# For each GOLD demo, compute:
#   - tilt_at_contact (baseline reference)
#   - tilt_max during ACTIVE (peak — typically chamfer engagement moment)
#   - tilt_at_end / tilt_at_seat (resting state after seating)
#   - relaxation_magnitude = tilt_max - tilt_at_seat
#   - relaxation_time = time from peak to "back to within 1° of final"
#
# Aggregate per object → emit a YAML-compatible recommendation:
#   tilt_tolerance_deg: p95 of tilt time series during normal engagement
#   tilt_relaxation_min_deg: minimum drop magnitude that signals chamfer engagement
#   tilt_relaxation_window_s: typical sustain time for the drop

import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _paths import LOG_DIR, DATA_DIR


def to_f(s, d=float("nan")):
    try: return float(s)
    except: return d


def col(rows, k, dt=float):
    return np.array([to_f(r.get(k)) for r in rows], dtype=dt)


def quat_to_tilt_deg(qx, qy, qz, qw):
    """Tilt of EE Z-axis from world -Z (canonical face-down)."""
    z_world_z = -(1 - 2 * (qx*qx + qy*qy))
    z_world_z = max(-1.0, min(1.0, z_world_z))
    return math.degrees(math.acos(z_world_z))


def find_contact_idx(fz_arr, threshold=5.0, sustain=10):
    sm = np.convolve(np.abs(fz_arr), np.ones(5)/5, mode="same")
    above = sm > threshold
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= sustain: return i - run + 1
    return None


def analyze_episode(csv_path, meta_path):
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    meta = json.load(open(meta_path))
    obj = meta.get("object", "?")
    outcome = meta.get("outcome", "?")
    assist = meta.get("assist_level", None)

    active = [r for r in rows if r.get("phase") == "ACTIVE"]
    if not active or len(active) < 100:
        return None

    t = col(active, "t_s")
    fs = 1.0 / np.median(np.diff(t[:200])) if len(t) > 1 else 100.0
    qx = col(active, "tcp_qx"); qy = col(active, "tcp_qy"); qz = col(active, "tcp_qz"); qw = col(active, "tcp_qw")
    fz_t = col(active, "fz")
    pz = col(active, "tcp_z")

    ci = find_contact_idx(fz_t)
    if ci is None:
        return None

    tilt = np.array([quat_to_tilt_deg(qx[i], qy[i], qz[i], qw[i]) for i in range(len(active))])

    # Limit to post-contact window — that's where engagement-relevant tilt happens
    post = tilt[ci:]
    pz_post = pz[ci:]
    n = len(post)
    if n < 50: return None

    tilt_at_contact = float(post[0])
    tilt_max = float(np.nanmax(post))
    tilt_max_idx = int(np.nanargmax(post))
    t_at_max = tilt_max_idx / fs

    # tilt_at_seat: take last 1 s if available (peg should be settled by then)
    tail_n = max(1, int(fs * 1.0))
    tilt_at_seat = float(np.nanmedian(post[-tail_n:])) if n > tail_n else float(post[-1])

    # final z drop — proxy for "did peg actually seat"
    z_drop_final_mm = float((pz_post[0] - np.nanmin(pz_post)) * 1000)

    # relaxation: drop from peak to "back within 1° of seat value"
    relax_mag = max(0.0, tilt_max - tilt_at_seat)
    # find first index after peak where tilt drops to within 1° of seat
    tilt_after_peak = post[tilt_max_idx:]
    target_band = tilt_at_seat + 1.0
    relax_idx = None
    for i, v in enumerate(tilt_after_peak):
        if not math.isnan(v) and v <= target_band:
            relax_idx = i; break
    relax_time_s = (relax_idx / fs) if relax_idx is not None else None

    # tilt_p95 for "normal engagement" envelope (excluding the brief chamfer-engagement spike)
    tilt_p95 = float(np.nanpercentile(post, 95))

    return {
        "object": obj,
        "csv": os.path.basename(csv_path),
        "outcome": outcome,
        "assist": assist,
        "tilt_at_contact_deg": tilt_at_contact,
        "tilt_max_deg": tilt_max,
        "tilt_max_t_s": t_at_max,
        "tilt_at_seat_deg": tilt_at_seat,
        "tilt_p95_deg": tilt_p95,
        "relax_magnitude_deg": relax_mag,
        "relax_time_s": relax_time_s,
        "z_drop_final_mm": z_drop_final_mm,
        "n_post_contact_samples": n,
    }


def main():
    out_per_episode = []
    log_dir = str(LOG_DIR)
    csvs = sorted([f for f in os.listdir(log_dir)
                   if f.startswith("insert_") and f.endswith(".csv")
                   and not any(s in f for s in ["raw", "fm_events"])])
    for fn in csvs:
        csv_path = os.path.join(log_dir, fn)
        meta_path = csv_path[:-4] + ".meta.json"
        if not os.path.exists(meta_path): continue
        try:
            r = analyze_episode(csv_path, meta_path)
            if r: out_per_episode.append(r)
        except Exception as e:
            print(f"FAIL on {fn}: {e}")
            continue

    print(f"\nAnalyzed {len(out_per_episode)} episodes\n")

    # Aggregate per object — separate GOLD (success) from FAIL
    by_obj = {}
    for r in out_per_episode:
        key = (r["object"], r["outcome"] == "success")
        by_obj.setdefault(key, []).append(r)

    print(f"{'object':22s} {'class':10s} {'n':>4s}  {'tilt@contact':>14s} {'tilt_p95':>10s} {'tilt_max':>10s} {'tilt@seat':>11s} {'relax_mag':>11s} {'relax_t':>9s} {'z_drop':>9s}")
    print("-" * 120)
    summary = {}
    for (obj, is_success), rs in sorted(by_obj.items()):
        if not rs: continue
        cls = "GOLD" if is_success else "FAIL"
        def med(k): return np.nanmedian([r[k] for r in rs if r[k] is not None])
        def p95(k): return np.nanpercentile([r[k] for r in rs if r[k] is not None], 95)
        # Filter to runs that actually had meaningful descent (z_drop > 5mm) for GOLD aggregation
        if is_success:
            rs_seated = [r for r in rs if r["z_drop_final_mm"] > 5.0]
            if not rs_seated:
                rs_seated = rs
        else:
            rs_seated = rs
        n = len(rs_seated)
        med_at_contact = float(np.nanmedian([r["tilt_at_contact_deg"] for r in rs_seated]))
        med_p95 = float(np.nanmedian([r["tilt_p95_deg"] for r in rs_seated]))
        med_max = float(np.nanmedian([r["tilt_max_deg"] for r in rs_seated]))
        med_seat = float(np.nanmedian([r["tilt_at_seat_deg"] for r in rs_seated]))
        med_relax = float(np.nanmedian([r["relax_magnitude_deg"] for r in rs_seated]))
        relax_ts = [r["relax_time_s"] for r in rs_seated if r["relax_time_s"] is not None]
        med_relax_t = float(np.nanmedian(relax_ts)) if relax_ts else float("nan")
        med_zdrop = float(np.nanmedian([r["z_drop_final_mm"] for r in rs_seated]))
        print(f"{obj:22s} {cls:10s} {n:>4d}  {med_at_contact:>11.2f}°  {med_p95:>7.2f}°  {med_max:>7.2f}°  {med_seat:>8.2f}°  {med_relax:>8.2f}°  {med_relax_t:>6.2f}s  {med_zdrop:>6.1f}mm")

        if is_success and n >= 3:
            summary[obj] = {
                "n_demos_seated": n,
                "tilt_at_contact_med_deg": med_at_contact,
                "tilt_p95_med_deg": med_p95,
                "tilt_max_med_deg": med_max,
                "tilt_at_seat_med_deg": med_seat,
                "relax_magnitude_med_deg": med_relax,
                "relax_time_med_s": med_relax_t,
                "z_drop_med_mm": med_zdrop,
                # Recommended config values:
                "config_tilt_tolerance_deg": float(round(med_p95 + 1.0, 2)),  # p95 + 1° margin
                "config_relax_threshold_deg": float(round(max(0.5, 0.5 * med_relax), 2)),  # half the typical relax magnitude, min 0.5°
                "config_relax_sustain_s": float(round(0.3, 2)),  # typical chamfer engagement window is 0.3-0.5s
            }

    print(f"\n=== Recommended per-object config (for fsm.find_hole_tilt_*) ===\n")
    for obj, cfg in summary.items():
        print(f"  {obj}:")
        print(f"    find_hole_tilt_tolerance_deg:   {cfg['config_tilt_tolerance_deg']:.2f}   # p95 of tilt during GOLD ACTIVE + 1° margin")
        print(f"    find_hole_tilt_relax_min_deg:   {cfg['config_relax_threshold_deg']:.2f}   # half of GOLD's median peak→seat relaxation magnitude")
        print(f"    find_hole_tilt_relax_sustain_s: {cfg['config_relax_sustain_s']:.2f}")

    # Write to data/
    out_path = os.path.join(str(DATA_DIR), "tilt_tolerance_per_object.json")
    json.dump({"per_episode": out_per_episode, "per_object_summary": summary},
              open(out_path, "w"), indent=2, default=str)
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()

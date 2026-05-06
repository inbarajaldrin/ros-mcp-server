"""
Phase 5 — event extraction per episode.

For every CSV in compliant_insertion_studio/logs/, find the four canonical events:
  1. t_first_contact   — |Fz_smoothed| > 9 N sustained 100 ms after ACTIVE start
  2. t_hole_found      — first sustained negative v_z spike post-contact
                         (peg drops into hole; v_z << contact-phase mean)
  3. interventions     — windows in [contact, hole_found] OR [contact, term]
                         where lateral residual + lateral velocity simultaneously
                         exceeded baseline (operator-help equivalent), with
                         direction (atan2 of mean fx_base, fy_base in window),
                         magnitude (max |F_lat_base|), and z-after (did z drop
                         within 1 s post-window? = "the push worked")
  4. t_termination     — phase=DONE entry (or last ACTIVE row)

Outputs:
  - .planning/phases/05-algorithm-derivation/events.json   — per-episode + per-shape
  - stdout summary table operator can review
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

LOGS = Path("compliant_insertion_studio/logs")
OUT = Path(".planning/phases/05-algorithm-derivation/events.json")
SHAPE_MAP = {
    "u_brown":           "u_shape",
    "u_orange":          "u_shape",
    "line_green":        "line_green",
    "inverted_u_yellow": "inverted_u_yellow",
}
SAMPLE_HZ = 100
SAMPLE_DT = 1.0 / SAMPLE_HZ


# ---------------------------------------------------------------------------
# Per-episode event extraction
# ---------------------------------------------------------------------------

def _smooth(s: pd.Series, window: int = 11) -> pd.Series:
    return s.rolling(window, min_periods=1, center=True).mean()


def _wrench_in_base(df: pd.DataFrame) -> pd.DataFrame:
    """Transform wrench from tool0_controller to base_link per row using TCP quat."""
    quats = df[["tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]].to_numpy()
    forces_tool = df[["fx", "fy", "fz"]].to_numpy()
    rots = R.from_quat(quats)
    forces_base = rots.apply(forces_tool)
    df = df.copy()
    df["fx_base"] = forces_base[:, 0]
    df["fy_base"] = forces_base[:, 1]
    df["fz_base"] = forces_base[:, 2]
    return df


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = _wrench_in_base(df)
    dt = df["t_s"].diff().fillna(SAMPLE_DT).clip(lower=1e-4)
    df["v_x"] = df["tcp_x"].diff().fillna(0) / dt
    df["v_y"] = df["tcp_y"].diff().fillna(0) / dt
    df["v_z"] = df["tcp_z"].diff().fillna(0) / dt
    df["v_lat"] = np.sqrt(df["v_x"] ** 2 + df["v_y"] ** 2)
    df["F_lat_base"] = np.sqrt(df["fx_base"] ** 2 + df["fy_base"] ** 2)
    df["T_lat"] = np.sqrt(df["tx"] ** 2 + df["ty"] ** 2)
    df["fz_smoothed"]    = _smooth(df["fz"])
    df["v_z_smoothed"]   = _smooth(df["v_z"], window=15)
    df["v_lat_smoothed"] = _smooth(df["v_lat"], window=15)
    df["F_lat_smoothed"] = _smooth(df["F_lat_base"], window=11)
    df["T_lat_smoothed"] = _smooth(df["T_lat"], window=11)
    return df


def _find_first_contact(df: pd.DataFrame, active_start_idx: int,
                         fz_threshold: float = 9.0,
                         sustain_samples: int = 10) -> int | None:
    """First sustained |fz_smoothed| > threshold after ACTIVE start."""
    fz = df["fz_smoothed"].to_numpy()
    n = len(fz)
    for i in range(active_start_idx, n - sustain_samples + 1):
        window = fz[i:i + sustain_samples]
        if (window > fz_threshold).all():
            return i
    return None


def _find_hole_found(df: pd.DataFrame, contact_idx: int, term_idx: int) -> int | None:
    """Detect the "peg drops into hole" moment in [contact, term].

    Physics: post-contact, the peg sits on the chamfer/rim — descent rate is low
    or near-zero. When the peg finds the hole entry, it drops in — descent rate
    spikes briefly. We detect this as the most-negative v_z sample in the
    [contact + 250 ms, term − 100 ms] window, gated by a relative threshold:
    the spike must be at least 2× the median |v_z| of the post-contact window.
    """
    if contact_idx is None or term_idx is None or contact_idx >= term_idx:
        return None
    sl_start = contact_idx + 25  # 250 ms after contact
    sl_end   = term_idx - 10     # 100 ms before termination
    if sl_end - sl_start < 10:
        return None
    v_z = df["v_z_smoothed"].to_numpy()
    window = v_z[sl_start:sl_end]
    median_abs = np.median(np.abs(window)) or 1e-4
    rel_thr = -2.0 * median_abs       # 2× faster than typical post-contact descent
    abs_thr = -0.005                  # absolute floor: ≥ 5 mm/s downward
    threshold = min(rel_thr, abs_thr)
    # Find the most-negative point that crosses threshold
    drops = np.where(window < threshold)[0]
    if len(drops) == 0:
        return None
    most_negative_offset = int(drops[np.argmin(window[drops])])
    return sl_start + most_negative_offset


def _find_interventions(df: pd.DataFrame, contact_idx: int, term_idx: int,
                         f_lat_thr: float, t_lat_thr: float, v_lat_thr: float,
                         min_samples: int = 5,
                         dilate: int = 10, merge_gap: int = 20) -> list[dict]:
    """Identify operator-equivalent intervention windows in [contact, term].

    Detection: a sample is "intervention" if force is anomalous (F_lat or T_lat
    above baseline), AND there is corresponding motion either at the same
    sample OR within a ±500 ms temporal envelope. The temporal coupling
    catches "stuck push" events where the operator pushes against a jam (force
    rises but motion stays zero) until the jam releases (motion appears 200-500
    ms later). Pure friction = force without motion in the entire ±500 ms
    envelope, which we exclude.
    """
    if contact_idx is None or term_idx is None:
        return []
    sl = slice(contact_idx, term_idx)
    f_lat = df["F_lat_smoothed"].to_numpy()[sl]
    t_lat = df["T_lat_smoothed"].to_numpy()[sl]
    v_lat = df["v_lat_smoothed"].to_numpy()[sl]
    fx_b = df["fx_base"].to_numpy()[sl]
    fy_b = df["fy_base"].to_numpy()[sl]
    tcp_z = df["tcp_z"].to_numpy()
    t_s = df["t_s"].to_numpy()

    force_anomaly = (f_lat > f_lat_thr) | (t_lat > t_lat_thr)
    motion_raw = v_lat > v_lat_thr
    # Temporal envelope: motion within ±500 ms of any sample counts
    envelope_samples = 50  # 500 ms at 100 Hz
    n = len(motion_raw)
    motion_envelope = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - envelope_samples)
        hi = min(n, i + envelope_samples + 1)
        motion_envelope[i] = motion_raw[lo:hi].any()
    above = force_anomaly & motion_envelope

    # Find contiguous runs ≥ min_samples
    runs = []
    n = len(above)
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= min_samples:
                runs.append([i, j - 1])
            i = j
        else:
            i += 1

    # Dilate + merge
    if not runs:
        return []
    expanded = [[max(0, a - dilate), min(n - 1, b + dilate)] for a, b in runs]
    merged = [expanded[0]]
    for a, b in expanded[1:]:
        la, lb = merged[-1]
        if a - lb <= merge_gap:
            merged[-1] = [la, max(lb, b)]
        else:
            merged.append([a, b])

    # Convert each merged run to a labelled intervention dict
    interventions = []
    for a_rel, b_rel in merged:
        a_abs = contact_idx + a_rel
        b_abs = contact_idx + b_rel
        # Direction: average force vector across the window (in base frame).
        # Convert to angle (rad) and magnitude (N).
        fx_win = fx_b[a_rel:b_rel + 1]
        fy_win = fy_b[a_rel:b_rel + 1]
        mean_fx = float(np.mean(fx_win))
        mean_fy = float(np.mean(fy_win))
        direction_rad = float(np.arctan2(mean_fy, mean_fx))
        magnitude_N = float(np.max(np.sqrt(fx_win ** 2 + fy_win ** 2)))

        # "Did the push work?" — z displacement in 1s after window end
        end_t = t_s[b_abs]
        post_window_end = end_t + 1.0
        post_idx = b_abs + 100  # 1s @ 100Hz
        if post_idx >= len(tcp_z):
            post_idx = len(tcp_z) - 1
        z_at_end = tcp_z[b_abs]
        z_1s_after = tcp_z[post_idx]
        z_drop_after_mm = (z_at_end - z_1s_after) * 1000.0  # positive = descended

        interventions.append({
            "t_start_s":    float(t_s[a_abs]),
            "t_end_s":      float(t_s[b_abs]),
            "duration_s":   float(t_s[b_abs] - t_s[a_abs]),
            "direction_rad": direction_rad,
            "direction_deg": math.degrees(direction_rad),
            "mag_N":        magnitude_N,
            "z_drop_after_1s_mm": z_drop_after_mm,
            "n_samples":    int(b_rel - a_rel + 1),
        })

    return interventions


def extract_events_for(csv_path: Path, baselines_per_shape: dict) -> dict:
    meta_path = Path(str(csv_path).replace(".csv", ".meta.json"))
    if not meta_path.exists():
        return {"basename": csv_path.stem, "error": "no meta.json"}
    meta = json.load(open(meta_path))
    df = pd.read_csv(csv_path)
    df["hands_off"] = pd.to_numeric(df["hands_off"], errors="coerce").fillna(0).astype(int)
    df = _add_derived(df)

    obj = meta.get("object")
    shape = SHAPE_MAP.get(obj, obj)
    bl = baselines_per_shape.get(shape, {})

    # Phase indices
    active_idx = int(df.index[df["phase"] == "ACTIVE"][0]) if (df["phase"] == "ACTIVE").any() else None
    done_idx = int(df.index[df["phase"] == "DONE"][0]) if (df["phase"] == "DONE").any() else None
    term_idx = done_idx if done_idx is not None else (active_idx and int(df.index[df["phase"] == "ACTIVE"][-1]))

    if active_idx is None or term_idx is None:
        return {"basename": csv_path.stem, "shape": shape, "error": "no ACTIVE phase"}

    contact_idx = _find_first_contact(df, active_idx)
    hole_idx = _find_hole_found(df, contact_idx, term_idx) if contact_idx else None

    # Intervention thresholds: use shape baseline p99 with safety floors
    f_lat_thr = max(bl.get("F_lat_p99", 1.0) * 1.0, 5.0)   # min 5 N
    t_lat_thr = max(bl.get("T_lat_p99", 0.05) * 1.0, 0.30) # min 0.3 Nm
    v_lat_thr = max(bl.get("v_lat_p99", 0.003) * 1.5, 0.004)  # min 4 mm/s

    # Search interventions in [contact, term] (covers both pre-hole and post-hole)
    interventions = _find_interventions(df, contact_idx, term_idx,
                                         f_lat_thr=f_lat_thr,
                                         t_lat_thr=t_lat_thr,
                                         v_lat_thr=v_lat_thr)

    def t_at(idx):
        return float(df["t_s"].iloc[idx]) if idx is not None else None

    return {
        "basename":    csv_path.stem,
        "shape":       shape,
        "object":      obj,
        "assist":      meta.get("assist_level"),
        "outcome":     meta.get("outcome"),
        "duration_s":  meta.get("duration_s"),
        "user_notes":  meta.get("user_notes", ""),
        "events": {
            "t_active_start_s":   t_at(active_idx),
            "t_first_contact_s":  t_at(contact_idx),
            "t_hole_found_s":     t_at(hole_idx),
            "t_termination_s":    t_at(term_idx),
            "active_to_contact_s":   None if contact_idx is None else t_at(contact_idx) - t_at(active_idx),
            "contact_to_hole_s":     None if (contact_idx is None or hole_idx is None) else t_at(hole_idx) - t_at(contact_idx),
            "hole_to_termination_s": None if (hole_idx is None) else t_at(term_idx) - t_at(hole_idx),
        },
        "thresholds_used": {
            "f_lat_thr_N": f_lat_thr,
            "t_lat_thr_Nm": t_lat_thr,
            "v_lat_thr_m_s": v_lat_thr,
        },
        "n_interventions":   len(interventions),
        "interventions":     interventions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Pull the per-shape baselines that preprocess.py wrote into signals.json
    baselines_per_shape: dict = {}
    for sig_path in LOGS.glob("insert_*.signals.json"):
        s = json.load(open(sig_path))
        shape = s.get("episode", {}).get("logical_shape")
        thr = s.get("baseline_used", {}).get("thresholds", {})
        if shape and shape not in baselines_per_shape:
            baselines_per_shape[shape] = {
                "F_lat_p99":     thr.get("F_lat", {}).get("p99", 0),
                "T_lat_p99":     thr.get("T_lat", {}).get("p99", 0),
                "v_lat_p99":     thr.get("v_lat", {}).get("p99", 0),
                "power_lat_p99": thr.get("power_lat", {}).get("p99", 0),
            }

    print("=== Per-shape baselines (from preprocess.py output) ===")
    for shape, b in baselines_per_shape.items():
        print(f"  {shape}: F_lat_p99={b['F_lat_p99']:.2f}  T_lat_p99={b['T_lat_p99']:.3f}  "
              f"v_lat_p99={b['v_lat_p99']:.4f}  power_lat_p99={b['power_lat_p99']:.4f}")

    episodes = []
    for csv_path in sorted(LOGS.glob("insert_*.csv")):
        ep = extract_events_for(csv_path, baselines_per_shape)
        episodes.append(ep)

    # Per-shape aggregate stats
    by_shape: dict = defaultdict(lambda: {
        "n_episodes": 0,
        "active_to_contact_s": [],
        "contact_to_hole_s": [],
        "hole_to_termination_s": [],
        "n_interventions_per_episode": [],
        "intervention_mags_N": [],
        "intervention_directions_deg": [],
        "intervention_durations_s": [],
        "intervention_z_drops_after_mm": [],
        "n_episodes_no_intervention": 0,
        "n_episodes_no_hole_found": 0,
    })
    for ep in episodes:
        if ep.get("error"):
            continue
        shape = ep["shape"]
        bs = by_shape[shape]
        bs["n_episodes"] += 1
        evt = ep["events"]
        for k in ("active_to_contact_s", "contact_to_hole_s", "hole_to_termination_s"):
            if evt.get(k) is not None:
                bs[k].append(evt[k])
        bs["n_interventions_per_episode"].append(ep["n_interventions"])
        if ep["n_interventions"] == 0:
            bs["n_episodes_no_intervention"] += 1
        if evt.get("t_hole_found_s") is None:
            bs["n_episodes_no_hole_found"] += 1
        for iv in ep["interventions"]:
            bs["intervention_mags_N"].append(iv["mag_N"])
            bs["intervention_directions_deg"].append(iv["direction_deg"])
            bs["intervention_durations_s"].append(iv["duration_s"])
            bs["intervention_z_drops_after_mm"].append(iv["z_drop_after_1s_mm"])

    def stats(vals):
        a = np.array(vals)
        if len(a) == 0:
            return None
        return {
            "n":     int(len(a)),
            "mean":  float(a.mean()),
            "std":   float(a.std()),
            "p5":    float(np.percentile(a, 5)),
            "p50":   float(np.median(a)),
            "p95":   float(np.percentile(a, 95)),
            "min":   float(a.min()),
            "max":   float(a.max()),
        }

    summary = {}
    for shape, bs in by_shape.items():
        summary[shape] = {
            "n_episodes": bs["n_episodes"],
            "n_episodes_no_intervention": bs["n_episodes_no_intervention"],
            "n_episodes_no_hole_found": bs["n_episodes_no_hole_found"],
            "active_to_contact_s":     stats(bs["active_to_contact_s"]),
            "contact_to_hole_s":       stats(bs["contact_to_hole_s"]),
            "hole_to_termination_s":   stats(bs["hole_to_termination_s"]),
            "n_interventions_per_episode": stats(bs["n_interventions_per_episode"]),
            "intervention_mag_N":           stats(bs["intervention_mags_N"]),
            "intervention_direction_deg":   stats(bs["intervention_directions_deg"]),
            "intervention_duration_s":      stats(bs["intervention_durations_s"]),
            "intervention_z_drop_after_mm": stats(bs["intervention_z_drops_after_mm"]),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"baselines_per_shape": baselines_per_shape,
                   "summary_per_shape": summary,
                   "episodes": episodes}, f, indent=2)
    print(f"\nWrote {OUT} ({len(episodes)} episodes)")

    # Print summary table
    print("\n" + "=" * 86)
    print("PER-SHAPE EVENT TIMELINE + INTERVENTION STATISTICS")
    print("=" * 86)
    for shape, s in summary.items():
        print(f"\n--- {shape} (n={s['n_episodes']}) ---")
        print(f"  active→contact:    n={s['active_to_contact_s']['n'] if s['active_to_contact_s'] else 0:2d}  "
              f"median={s['active_to_contact_s']['p50']:.2f} s  p5={s['active_to_contact_s']['p5']:.2f}  p95={s['active_to_contact_s']['p95']:.2f}" if s['active_to_contact_s'] else "  active→contact:    NEVER DETECTED")
        if s["contact_to_hole_s"]:
            print(f"  contact→hole:      n={s['contact_to_hole_s']['n']:2d}  median={s['contact_to_hole_s']['p50']:.2f} s  "
                  f"p5={s['contact_to_hole_s']['p5']:.2f}  p95={s['contact_to_hole_s']['p95']:.2f}")
        else:
            print(f"  contact→hole:      NO HOLE-FOUND DETECTED in any episode")
        print(f"  no_hole_found:     {s['n_episodes_no_hole_found']}/{s['n_episodes']} episodes")
        if s["hole_to_termination_s"]:
            print(f"  hole→termination:  n={s['hole_to_termination_s']['n']:2d}  median={s['hole_to_termination_s']['p50']:.2f} s  "
                  f"p5={s['hole_to_termination_s']['p5']:.2f}  p95={s['hole_to_termination_s']['p95']:.2f}")
        intv = s["n_interventions_per_episode"]
        print(f"  interventions/ep:  median={intv['p50']:.0f}  max={intv['max']:.0f}  "
              f"no_intervention={s['n_episodes_no_intervention']}/{s['n_episodes']} episodes")
        if s["intervention_mag_N"]:
            print(f"  intv magnitude:    n={s['intervention_mag_N']['n']:2d}  median={s['intervention_mag_N']['p50']:.2f} N  "
                  f"p5={s['intervention_mag_N']['p5']:.2f}  p95={s['intervention_mag_N']['p95']:.2f}")
            print(f"  intv duration:     median={s['intervention_duration_s']['p50']:.2f} s  "
                  f"p95={s['intervention_duration_s']['p95']:.2f}")
            print(f"  intv direction:    n={s['intervention_direction_deg']['n']:2d}  "
                  f"min={s['intervention_direction_deg']['min']:.0f}°  max={s['intervention_direction_deg']['max']:.0f}°  "
                  f"std={s['intervention_direction_deg']['std']:.0f}° "
                  f"(non-uniform std → systematic bias direction)")
            print(f"  z-drop after 1s:   median={s['intervention_z_drop_after_mm']['p50']:.2f} mm  "
                  f"p95={s['intervention_z_drop_after_mm']['p95']:.2f}  "
                  f"(positive = the push 'worked' → robot descended after)")


if __name__ == "__main__":
    main()

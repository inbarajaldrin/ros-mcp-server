"""Regenerate analyzer/manifest.json from the CSV+signals.json sidecars in logs/.

Run via analyzer/serve.sh, or standalone after preprocess.py.
"""
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent  # compliant_insertion_studio/
LOGS = ROOT / "logs"
OUT = ROOT / "analyzer" / "manifest.json"

episodes = []
for csv in sorted(LOGS.glob("insert_*.csv")):
    sig = csv.with_name(csv.stem + ".signals.json")
    meta = csv.with_name(csv.stem + ".meta.json")
    if not sig.exists() or not meta.exists():
        continue
    s = json.load(open(sig))
    ep = s["episode"]
    drift_mm = (s.get("drift", {}).get("lateral_xy_drift_m", 0) or 0) * 1000
    counts = {m["method"]: len(m.get("nudge_intervals", [])) for m in s.get("methods", [])}
    feats = s.get("features", {}) or {}
    episodes.append({
        "basename":      csv.stem,
        "csv":           f"logs/{csv.name}",
        "meta":          f"logs/{meta.name}",
        "signals":       f"logs/{sig.name}",
        "object":        ep.get("object"),
        "logical_shape": ep.get("logical_shape"),
        "assist_level":  ep.get("assist_level"),
        "duration_s":    ep.get("duration_s"),
        "drift_mm":      drift_mm,
        "user_notes":    ep.get("user_notes"),
        "m1_nudges":     counts.get("M1_force", 0),
        "m2_nudges":     counts.get("M2_motion", 0),
        "m3_nudges":     counts.get("M3_energy", 0),
        "m5_nudges":     counts.get("M5_torque", 0),
        "m7_nudges":     counts.get("M7_slip", 0),
        # Per-episode features for cross-episode scatter view
        "max_F_lat":     feats.get("max_F_lat"),
        "max_T_lat":     feats.get("max_T_lat"),
        "max_v_lat":     feats.get("max_v_lat"),
        "max_power_lat": feats.get("max_power_lat"),
        "max_rot_power": feats.get("max_rot_power"),
        "max_slip_lat":  feats.get("max_slip_lat"),
        "median_fz":     feats.get("median_fz_active"),
        "z_descent_m":   feats.get("z_descent_m"),
        "xy_travel_m":   feats.get("xy_travel_m"),
        "active_dur_s":  feats.get("active_duration_s"),
        "total_work_J":  feats.get("total_external_work_J"),
    })

episodes.sort(key=lambda e: e["basename"])
OUT.write_text(json.dumps({
    "schema_version": 1,
    "generated_at":   datetime.now().isoformat(),
    "n_episodes":     len(episodes),
    "episodes":       episodes,
}, indent=2))
print(f"wrote {OUT.relative_to(ROOT)} ({len(episodes)} episodes)")

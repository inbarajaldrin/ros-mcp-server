"""
Found Hole predicate v4 — "rim-contact released" state-transition detector.

Replaces v3 (which required simultaneous fz drop + low F_lat at one sample —
brittle because F_lat lags fz collapse by ~250ms).

State machine, runnable real-time:
  ON_RIM    : |fz_smooth| > rim_high_thresh
  OFF_RIM   : |fz_smooth| < rim_low_thresh AND F_lat <= flat_max
  FIRE when: just been ON_RIM in last `recent_window_s`, AND
             has been OFF_RIM continuously for `off_sustain_s`.

This fires once at the first stable rim-cross. If the operator wobbles back
onto the rim, the predicate re-arms.

Acceptance:
  G1. Fires in [Contact, SIGUSR1] on every demo (lead_s > 0).
  G2. At firing time, peg xy within `r_max_to_hole` m of hole_observed_xy.
  G3. (G1 + G2) hold across all 3 variations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _paths import DATA_DIR  # noqa: E402

SIG_PATH = DATA_DIR / "found_hole_signature.json"
FEATURES_DIR = DATA_DIR / "guided_features"

PARAMS = {
    "rim_high_thresh":   4.0,    # N — |fz| above this counts as on-rim
    "rim_low_thresh":    3.0,    # N — |fz| below this counts as off-rim (hysteresis)
    "flat_max":          1e9,    # N — disabled; the state is purely |fz|-based
    "off_sustain_s":     0.30,   # s — how long off-rim before firing
    "recent_window_s":   2.5,    # s — must have been on-rim in this window prior
    "r_max_to_hole_m":   0.005,  # 5mm tolerance for G2
}


def _scan(feats: dict, p: dict) -> dict:
    g = feats["guided"]
    s = feats["summary"]
    dt = s["dt_s"]
    t_s = np.array(g["t_s"])
    fz = np.array(g["fz_smooth"])
    flat = np.array(g["F_lat_tool"])
    tcp_x = np.array(g["tcp_x"])
    tcp_y = np.array(g["tcp_y"])
    hole_xy = np.array(s["hole_observed_xy_m"])

    abs_fz = np.abs(fz)
    on_rim = abs_fz > p["rim_high_thresh"]
    off_rim = (abs_fz < p["rim_low_thresh"]) & (flat <= p["flat_max"])

    n_off_sustain = max(1, int(round(p["off_sustain_s"] / dt)))
    n_recent      = max(1, int(round(p["recent_window_s"] / dt)))

    fire_idx = None
    off_run = 0
    for i in range(len(fz)):
        if off_rim[i]:
            off_run += 1
        else:
            off_run = 0
        if off_run >= n_off_sustain:
            # Verify ON_RIM happened in [i-n_recent, i-n_off_sustain]
            lo = max(0, i - n_recent)
            hi = max(0, i - n_off_sustain)
            if hi > lo and np.any(on_rim[lo:hi]):
                fire_idx = i - n_off_sustain + 1
                break

    if fire_idx is None:
        return {"fire_idx": None, "fire_t_s": None}

    dist = float(np.hypot(tcp_x[fire_idx] - hole_xy[0],
                          tcp_y[fire_idx] - hole_xy[1]))
    return {
        "fire_idx": int(fire_idx),
        "fire_t_s": float(t_s[fire_idx]),
        "fz_at_fire": float(fz[fire_idx]),
        "abs_fz_at_fire": float(abs_fz[fire_idx]),
        "F_lat_at_fire": float(flat[fire_idx]),
        "tcp_xy_at_fire": [float(tcp_x[fire_idx]), float(tcp_y[fire_idx])],
        "dist_to_hole_at_fire_m": dist,
    }


def main():
    sig = json.load(open(SIG_PATH))
    clean_demos = sig["clean_demos"]
    variations = sig["variations"]

    print("Predicate v4 (state-transition rim-cross detector):")
    for k, v in PARAMS.items():
        print(f"  {k}: {v}")

    print()
    print(f"{'demo':<15s}{'var':<8s} {'sigusr1_t':>10s} {'fire_t':>9s} "
          f"{'lead_s':>7s} {'|fz|@fire':>10s} {'Flat@fire':>10s} "
          f"{'dist→hole':>11s} {'G1':>4s} {'G2_5mm':>7s}")

    pass_g1 = 0
    pass_g2 = 0
    rows = []
    for bn, var in zip(clean_demos, variations):
        feats = json.load(open(FEATURES_DIR / f"{bn}.features.json"))
        s = feats["summary"]
        t_event = s["t_sigusr1_s"]
        r = _scan(feats, PARAMS)

        if r["fire_idx"] is None:
            print(f"{bn[-15:]:<15s}{var.split('_')[0]:<8s} {t_event:>10.2f} "
                  f"{'-':>9s} {'-':>7s} {'-':>10s} {'-':>10s} {'-':>11s} "
                  f"{'NO':>4s} {'NO':>7s}")
            rows.append({"basename": bn, "variation": var, "result": r,
                         "g1": False, "g2": False})
            continue

        lead = t_event - r["fire_t_s"]
        g1 = lead >= 0
        g2 = r["dist_to_hole_at_fire_m"] <= PARAMS["r_max_to_hole_m"]
        if g1: pass_g1 += 1
        if g2: pass_g2 += 1
        print(f"{bn[-15:]:<15s}{var.split('_')[0]:<8s} {t_event:>10.2f} "
              f"{r['fire_t_s']:>9.2f} {lead:>+7.2f} "
              f"{r['abs_fz_at_fire']:>10.2f} {r['F_lat_at_fire']:>10.2f} "
              f"{1000*r['dist_to_hole_at_fire_m']:>10.2f}mm "
              f"{('YES' if g1 else 'NO'):>4s} {('YES' if g2 else 'NO'):>7s}")
        rows.append({"basename": bn, "variation": var, "lead_s": lead,
                     "result": r, "g1": g1, "g2": g2})

    print(f"\n  G1 (fires before SIGUSR1):           {pass_g1}/{len(clean_demos)}")
    print(f"  G2 (peg within 5mm of hole at fire):  {pass_g2}/{len(clean_demos)}")

    print(f"\n  G3 per variation:")
    for v in sorted(set(variations)):
        vd = [r for r in rows if r["variation"] == v]
        ok1 = sum(1 for r in vd if r["g1"])
        ok2 = sum(1 for r in vd if r["g2"])
        print(f"    {v:<22s} G1={ok1}/{len(vd)} G2={ok2}/{len(vd)}")

    out = {"params": PARAMS, "results": rows,
           "g1": pass_g1, "g2": pass_g2, "n": len(clean_demos)}
    (DATA_DIR / "found_hole_v4_validation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

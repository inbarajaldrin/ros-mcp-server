"""
Found Hole predicate v3 — rim-cross transient detector.

Reframed target: the "Found Hole" marker should fire at the rim-to-chamfer
transition (the actual local geometric event), not at the operator's SIGUSR1
(which fires 2-3 seconds later after the operator fine-tunes peg xy).

Predicate (tool-frame, derivative-based):
  Fires when ALL hold for at least sustain_s seconds:
    a) fz dropped by >= fz_drop_N over the past drop_window_s seconds  (rim contact released)
    b) F_lat <= flat_post_thresh                                       (peg no longer resisted laterally)
    c) |fz_smooth| <= fz_zero_band_post                                 (post-drop fz near zero)
    d) (optional) vz <= vz_onset_thresh                                  (peg starts descending)

Acceptance gates (revised):
  G1. Fires exactly once between Contact and SIGUSR1 inclusive (no early fire = no fire before
      Contact, can't happen since GUIDED only starts at Contact; no late fire = fires before
      operator's mark).
  G2. At firing time, the peg's xy position is within R_max of the operator's eventual
      hole_observed_xy (proves the predicate identifies the actual hole region — what an
      autonomous controller needs).
  G3. Holds across all 3 variations (A_pos_x / C_pos_y / D_neg_y) → direction-invariant.
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
    "fz_drop_N":           5.0,    # require fz dropped by >= 5N in the window
    "drop_window_s":       0.30,   # 300ms window
    "flat_post_thresh":    2.0,    # N
    "fz_zero_band_post":   2.0,    # N (|fz| <= this AFTER drop)
    "vz_onset_thresh":     0.0001, # m/s (require non-positive vz, i.e. descending or steady)
    "sustain_s":           0.10,   # 100ms post-drop sustain
    "use_vz_onset":        False,
    "r_max_to_hole_m":     0.005,  # 5mm — at firing time, peg must be within this of hole_observed_xy
}


def _load_features(bn: str) -> dict:
    return json.load(open(FEATURES_DIR / f"{bn}.features.json"))


def _scan(feats: dict, p: dict) -> dict:
    g = feats["guided"]
    s = feats["summary"]
    dt = s["dt_s"]
    t_s = np.array(g["t_s"])
    fz_s = np.array(g["fz_smooth"])
    flat = np.array(g["F_lat_tool"])
    vz = np.array(g["vz"])
    tcp_x = np.array(g["tcp_x"])
    tcp_y = np.array(g["tcp_y"])
    hole_xy = np.array(s["hole_observed_xy_m"])

    n_drop = max(1, int(round(p["drop_window_s"] / dt)))
    n_sustain = max(1, int(round(p["sustain_s"] / dt)))

    # fz_drop[i] = max(fz_smooth[i-n_drop:i]) - fz_smooth[i]
    fz_max_pre = np.array([
        np.max(fz_s[max(0, i - n_drop):i + 1]) for i in range(len(fz_s))
    ])
    fz_drop = fz_max_pre - fz_s

    cond_a = fz_drop >= p["fz_drop_N"]
    cond_b = flat <= p["flat_post_thresh"]
    cond_c = np.abs(fz_s) <= p["fz_zero_band_post"]
    cond = cond_a & cond_b & cond_c
    if p["use_vz_onset"]:
        cond_d = vz <= p["vz_onset_thresh"]
        cond &= cond_d

    fire_idx = None
    run = 0
    for i, c in enumerate(cond):
        if c:
            run += 1
            if run >= n_sustain:
                fire_idx = i - n_sustain + 1
                break
        else:
            run = 0

    if fire_idx is None:
        return {"fire_idx": None, "fire_t_s": None, "n_fires_total": int(np.sum(cond))}

    dist_to_hole_at_fire = float(np.hypot(tcp_x[fire_idx] - hole_xy[0],
                                          tcp_y[fire_idx] - hole_xy[1]))

    return {
        "fire_idx": int(fire_idx),
        "fire_t_s": float(t_s[fire_idx]),
        "fz_smooth_at_fire": float(fz_s[fire_idx]),
        "fz_drop_at_fire": float(fz_drop[fire_idx]),
        "F_lat_at_fire": float(flat[fire_idx]),
        "vz_at_fire": float(vz[fire_idx]),
        "tcp_xy_at_fire": [float(tcp_x[fire_idx]), float(tcp_y[fire_idx])],
        "dist_to_hole_at_fire_m": dist_to_hole_at_fire,
        "n_fires_total": int(np.sum(cond)),
    }


def main():
    sig = json.load(open(SIG_PATH))
    clean_demos = sig["clean_demos"]
    variations = sig["variations"]

    print("Predicate v3 (rim-cross transient detector):")
    for k, v in PARAMS.items():
        print(f"  {k}: {v}")

    print()
    print(f"{'demo':<15s}{'var':<8s} {'sigusr1_t':>10s} {'fire_t':>9s} "
          f"{'lead_s':>7s} {'fz_drop':>8s} {'fz@fire':>8s} {'Flat@fire':>10s} "
          f"{'vz@fire':>9s} {'r_to_hole':>10s} {'within_5mm':>11s}")
    rows = []
    pass_g1 = 0
    pass_g2 = 0
    for bn, var in zip(clean_demos, variations):
        feats = _load_features(bn)
        s = feats["summary"]
        t_event = s["t_sigusr1_s"]
        r = _scan(feats, PARAMS)
        if r["fire_idx"] is None:
            print(f"{bn[-15:]:<15s}{var.split('_')[0]:<8s} {t_event:>10.2f} "
                  f"{'-':>9s} {'-':>7s} {'-':>8s} {'-':>8s} {'-':>10s} {'-':>9s} {'-':>10s} {'NO FIRE':>11s}")
            rows.append({"basename": bn, "variation": var, "result": r,
                         "g1_fires_before_event": False, "g2_within_5mm": False})
            continue
        lead_s = t_event - r["fire_t_s"]
        # G1: fires BEFORE SIGUSR1 (lead_s > 0)
        g1 = lead_s > 0
        # G2: peg position at fire is within R_max of hole_observed
        g2 = r["dist_to_hole_at_fire_m"] <= PARAMS["r_max_to_hole_m"]
        if g1: pass_g1 += 1
        if g2: pass_g2 += 1

        print(f"{bn[-15:]:<15s}{var.split('_')[0]:<8s} {t_event:>10.2f} "
              f"{r['fire_t_s']:>9.2f} {lead_s:>7.2f} "
              f"{r['fz_drop_at_fire']:>8.2f} {r['fz_smooth_at_fire']:>+8.2f} "
              f"{r['F_lat_at_fire']:>10.2f} {1000*r['vz_at_fire']:>+8.2f} "
              f"{1000*r['dist_to_hole_at_fire_m']:>9.2f} "
              f"{('YES' if g2 else 'NO'):>11s}")

        rows.append({"basename": bn, "variation": var, "result": r,
                     "lead_s": lead_s, "g1_fires_before_event": g1, "g2_within_5mm": g2})

    print(f"\n  G1 (fires before SIGUSR1):           {pass_g1}/{len(clean_demos)}")
    print(f"  G2 (peg xy within 5mm of hole at fire): {pass_g2}/{len(clean_demos)}")

    # Per-variation breakdown for G3 (direction-invariance)
    print(f"\n  G3 (direction-invariance) per variation:")
    for v in sorted(set(variations)):
        vdemos = [r for r in rows if r["variation"] == v]
        ok1 = sum(1 for r in vdemos if r.get("g1_fires_before_event"))
        ok2 = sum(1 for r in vdemos if r.get("g2_within_5mm"))
        print(f"    {v:<22s} G1={ok1}/{len(vdemos)} G2={ok2}/{len(vdemos)}")

    out = {"params": PARAMS, "results": rows}
    (DATA_DIR / "found_hole_v3_validation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

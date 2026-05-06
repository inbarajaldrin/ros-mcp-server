"""
Found Hole predicate validator.

Builds a tool-frame, direction-invariant predicate from the signature analysis
in `32_found_hole_signature` and tests it on each of the 10 clean demos. The
acceptance gates:
  G1. The predicate fires WITHIN ±300 ms of the operator's SIGUSR1 on every
      demo.
  G2. The predicate does NOT fire earlier than (SIGUSR1 - 300 ms) in any demo
      (zero false positives during the rim-search portion of GUIDED).
  G3. The predicate is invariant across A_pos_x / C_pos_y / D_neg_y variations
      (it uses no world-frame quantities — invariance is by construction; this
      gate just confirms (G1) holds across all variations equally).

Predicate v1 (derived from AUC analysis on 10 demos):
  Fires when ALL hold for at least `sustain_s` seconds:
    a) vz <= vz_thresh                             (peg descending)
    b) F_lat_tool <= flat_thresh                    (rim resistance gone)
    c) |fz_smooth| <= fz_zero_band                  (vertical rim load released)

Optional refinement v2 (added if v1 fails G2):
    d) (vz - rolling_min_1s_vz) <= 0.0002 m/s       (vz at its 1s-window minimum)
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

# Tunable predicate parameters (initialize from the signature distributions)
VZ_THRESH         = -0.0003   # m/s (-0.3 mm/s); p95 of at-event = -0.0001
FLAT_THRESH       = 2.0        # N
FZ_ZERO_BAND      = 1.0        # N (|fz_smooth| <= this)
SUSTAIN_S         = 0.10       # s

VZ_MIN_DELTA      = 0.0003     # m/s — for v2 refinement


def _load_features(bn: str) -> dict:
    return json.load(open(FEATURES_DIR / f"{bn}.features.json"))


def _scan_predicate(feats: dict, vz_thresh, flat_thresh, fz_band, sustain_s,
                    use_vz_min_refinement=False, vz_min_delta=VZ_MIN_DELTA):
    g = feats["guided"]
    s = feats["summary"]
    dt = s["dt_s"]
    t_s = np.array(g["t_s"])
    vz = np.array(g["vz"])
    flat = np.array(g["F_lat_tool"])
    fz_s = np.array(g["fz_smooth"])

    n_sustain = max(1, int(round(sustain_s / dt)))
    n_1s = max(1, int(round(1.0 / dt)))

    cond_a = vz <= vz_thresh
    cond_b = flat <= flat_thresh
    cond_c = np.abs(fz_s) <= fz_band

    if use_vz_min_refinement:
        rolling_min = np.array(
            [np.min(vz[max(0, i - n_1s):i + 1]) for i in range(len(vz))]
        )
        cond_d = (vz - rolling_min) <= vz_min_delta
        cond = cond_a & cond_b & cond_c & cond_d
    else:
        cond = cond_a & cond_b & cond_c

    # Find first sustained-fire index (n_sustain consecutive True)
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

    return {
        "fire_idx_in_guided": int(fire_idx) if fire_idx is not None else None,
        "fire_t_s": float(t_s[fire_idx]) if fire_idx is not None else None,
        "n_total": int(np.sum(cond)),
        "guided_dur_s": s["guided_dur_s"],
        "t_sigusr1_s": s["t_sigusr1_s"],
    }


def _validate(predicate_fn, label: str):
    sig = json.load(open(SIG_PATH))
    clean_demos = sig["clean_demos"]
    variations = sig["variations"]

    results = []
    print(f"\n=== {label} ===")
    print(f"{'demo':<15s}{'var':<8s} {'sigusr1_t':>10s} {'fire_t':>9s} {'Δt(ms)':>8s} {'within±300ms':>14s} {'fires_total':>11s}")
    pass_g1 = 0
    pass_g2 = 0
    for bn, var in zip(clean_demos, variations):
        feats = _load_features(bn)
        r = predicate_fn(feats)
        t_event = r["t_sigusr1_s"]
        fire_t = r["fire_t_s"]
        if fire_t is None:
            within = False
            dt_ms = float("nan")
        else:
            dt_ms = 1000 * (fire_t - t_event)
            within = abs(dt_ms) <= 300
        # G2: predicate should not fire earlier than (event - 300ms)
        no_early_fp = fire_t is None or fire_t >= (t_event - 0.300)

        if within:
            pass_g1 += 1
        if no_early_fp:
            pass_g2 += 1

        results.append({
            "basename": bn, "variation": var,
            "fire_t_s": fire_t, "t_sigusr1_s": t_event,
            "delta_ms": dt_ms,
            "within_pm_300ms": within,
            "no_early_false_positive": no_early_fp,
            "fires_total": r["n_total"],
        })
        print(f"{bn[-15:]:<15s}{var.split('_')[0]:<8s} {t_event:>10.2f} "
              f"{('-' if fire_t is None else f'{fire_t:.2f}'):>9s} "
              f"{('NaN' if fire_t is None else f'{dt_ms:+.0f}'):>8s} "
              f"{('YES' if within else '— '):>14s} {r['n_total']:>11d}")

    print(f"\n  G1 (fires within ±300ms): {pass_g1}/{len(clean_demos)}")
    print(f"  G2 (no early fire before event-300ms): {pass_g2}/{len(clean_demos)}")
    return results, pass_g1, pass_g2


def main():
    print(f"Predicate v1 parameters:")
    print(f"  vz       <= {VZ_THRESH*1000:.2f} mm/s")
    print(f"  F_lat    <= {FLAT_THRESH:.2f} N")
    print(f"  |fz|     <= {FZ_ZERO_BAND:.2f} N")
    print(f"  sustain    {SUSTAIN_S*1000:.0f} ms")

    res_v1, g1_v1, g2_v1 = _validate(
        lambda f: _scan_predicate(f, VZ_THRESH, FLAT_THRESH, FZ_ZERO_BAND, SUSTAIN_S),
        "Predicate v1: vz↓ AND F_lat↓ AND |fz|↓"
    )

    res_v2, g1_v2, g2_v2 = _validate(
        lambda f: _scan_predicate(f, VZ_THRESH, FLAT_THRESH, FZ_ZERO_BAND, SUSTAIN_S,
                                   use_vz_min_refinement=True),
        "Predicate v2: v1 AND vz at 1s-window minimum"
    )

    out = {
        "v1": {"params": {"vz_thresh": VZ_THRESH, "flat_thresh": FLAT_THRESH,
                          "fz_band": FZ_ZERO_BAND, "sustain_s": SUSTAIN_S},
               "results": res_v1, "g1": g1_v1, "g2": g2_v1},
        "v2": {"params": {"vz_thresh": VZ_THRESH, "flat_thresh": FLAT_THRESH,
                          "fz_band": FZ_ZERO_BAND, "sustain_s": SUSTAIN_S,
                          "vz_min_delta": VZ_MIN_DELTA},
               "results": res_v2, "g1": g1_v2, "g2": g2_v2},
    }
    (DATA_DIR / "found_hole_validation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

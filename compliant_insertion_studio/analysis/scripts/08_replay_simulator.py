#!/usr/bin/env python3
# Reference: counterfactual replay of a proposed wrench-command function against the 132 May-4 FAIL trajectories.
# Produces a predicted TCP trajectory under the new command, compared against the actual TCP trajectory and
# the GOLD-class reference. Used by 09_score_staged_patch.py to compute evidence_score.
#
# Model (linear admittance approximation):
#   v_observed = K @ (F_sensed - F_cmd_actual)
#   For May-4 attempts, F_cmd_actual_lateral was non-trivial but unlogged. We approximate it with the FSM
#   config from the loop_attempt log when available, else assume zero (conservative).
#
#   Under the proposed command:
#   v_predicted = v_observed + K @ (F_cmd_actual - F_cmd_proposed)
#                = v_observed - K @ delta_cmd
#
# This is a poor man's CI: filters obviously-wrong patches before consuming robot time.
# Limitations:
#   - Ignores rim-contact nonlinearity (sensor reading depends on TCP pose; a different pose would
#     produce a different F_sensed)
#   - Treats admittance gain K as a scalar derived from operator demos
#   - Single-step prediction (no Fz_t collapse modeling under counterfactual contact)
#
# Use:  python3 08_replay_simulator.py <staged_dir>
# Reads <staged_dir>/cmd_function.py, writes <staged_dir>/REPLAY.md.

import importlib.util, json, os, sys, math, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _paths import DATA_DIR, ITERATIONS_DIR

# Empirical admittance gain estimated from operator demos: 1 N -> ~0.5 mm/s lateral velocity steady state
# (from F_lat_pre = 2.4 N producing ~1.5 mm displacement over 1 s in u_orange GOLD).
# Scalar diagonal — full 3x3 not needed for the lateral-only filter we're building.
K_LIN_M_PER_NS = 0.0005   # m/(N*s) -> 0.5 mm/s per N
K_ANG_RAD_PER_NMS = 0.05  # rad/(N*m*s)


def load_cmd_function(staged_dir):
    path = os.path.join(staged_dir, "cmd_function.py")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — staged dir must contain cmd_function.py with cmd_wrench(state) -> 6-tuple")
    spec = importlib.util.spec_from_file_location("cmd_function", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if not hasattr(mod, "cmd_wrench"):
        raise SystemExit("cmd_function.py must define cmd_wrench(state) -> 6-tuple")
    return mod.cmd_wrench


def build_state(ps, i, contact_idx):
    """State dict passed to cmd_wrench at each timepoint."""
    return {
        "t_s": ps["t_s"][i],
        "t_since_contact_s": ps["t_s"][i] - ps["t_s"][contact_idx],
        "tcp_xy_m": (ps["tcp_x"][i], ps["tcp_y"][i]),
        "tcp_z_m": ps["tcp_z"][i],
        "fz_t_N": ps["fz_t"][i],
        "F_lat_base_N": ps["F_lat_base"][i],
        "z_drop_mm": ps["z_drop_mm"][i],
        "tilt_deg": ps["tilt_deg"][i],
        "phase_idx": i - contact_idx,
    }


def replay_one(ps, contact_idx, cmd_fn, hole_xy_m=None):
    n = len(ps["t_s"])
    if contact_idx is None or contact_idx >= n - 10: return None

    # Per-sample proposed command (lateral + Tz only — Fz held to actual cmd_fz)
    state_template = {"hole_xy_m": hole_xy_m}
    delta_fx = np.zeros(n); delta_fy = np.zeros(n)
    for i in range(contact_idx, n):
        s = build_state(ps, i, contact_idx); s.update(state_template)
        try: cmd = cmd_fn(s)
        except Exception: cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # cmd_actual_lateral assumed 0 for May-4 baseline replay (we lack per-sample log)
        delta_fx[i] = cmd[0] - 0.0
        delta_fy[i] = cmd[1] - 0.0

    # Predicted lateral velocity perturbation
    dvx = -K_LIN_M_PER_NS * delta_fx
    dvy = -K_LIN_M_PER_NS * delta_fy
    # Integrate to get predicted xy displacement perturbation
    dt = np.diff(np.array(ps["t_s"]))
    dt = np.concatenate([[0.0], dt])
    dx_pred = np.cumsum(dvx * dt)
    dy_pred = np.cumsum(dvy * dt)

    # Counterfactual TCP xy
    tcp_x_actual = np.array(ps["tcp_x"], dtype=float)
    tcp_y_actual = np.array(ps["tcp_y"], dtype=float)
    tcp_x_pred = tcp_x_actual + dx_pred
    tcp_y_pred = tcp_y_actual + dy_pred

    # Distance to hole_xy at end (closer = better) and within search window (1s post-contact)
    if hole_xy_m is not None:
        hx, hy = hole_xy_m
        i_end = min(n - 1, contact_idx + int(1.0 / max(np.median(dt), 1e-3)))
        dist_actual = math.hypot(tcp_x_actual[i_end] - hx, tcp_y_actual[i_end] - hy)
        dist_pred = math.hypot(tcp_x_pred[i_end] - hx, tcp_y_pred[i_end] - hy)
        dist_delta_mm = (dist_pred - dist_actual) * 1000  # negative = closer = better
    else:
        dist_delta_mm = float("nan")

    # xy displacement magnitude in 1s post-contact (vs operator GOLD 1.2-1.7mm)
    i_end = min(n - 1, contact_idx + int(1.0 / max(np.median(dt), 1e-3)))
    disp_actual_mm = math.hypot(tcp_x_actual[i_end] - tcp_x_actual[contact_idx],
                                tcp_y_actual[i_end] - tcp_y_actual[contact_idx]) * 1000
    disp_pred_mm = math.hypot(tcp_x_pred[i_end] - tcp_x_actual[contact_idx],
                              tcp_y_pred[i_end] - tcp_y_actual[contact_idx]) * 1000

    return {
        "n_samples_post_contact": n - contact_idx,
        "disp_actual_mm_at_1s": disp_actual_mm,
        "disp_predicted_mm_at_1s": disp_pred_mm,
        "dist_to_hole_delta_mm": dist_delta_mm,
        "predicted_in_gold_band": 1.2 <= disp_pred_mm <= 1.7,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("staged_dir")
    p.add_argument("--object", default="u_orange")
    args = p.parse_args()
    staged_dir = os.path.abspath(args.staged_dir)
    cmd_fn = load_cmd_function(staged_dir)

    summaries = json.load(open(os.path.join(str(DATA_DIR), "summaries.json")))
    targets = [r for r in summaries
               if r["object"] == args.object and r["outcome"] == "abort"
               and "20260504" in r["csv_path"] and r["final_z_drop_mm"] < 5.0]

    # default hole_xy_m from STATE.json:current_target if not given by cmd_function
    state_path = os.path.join(os.path.dirname(staged_dir), "..", "..", "STATE.json")
    state_path = os.path.normpath(state_path)
    if os.path.exists(state_path):
        st = json.load(open(state_path))
        hole_xy_m = tuple(st.get("current_target", {}).get("predicted_seat_xy_m", [0.0341, -0.3635]))
    else:
        hole_xy_m = (0.0341, -0.3635)

    results = []
    for r in targets:
        ps_path = os.path.join(str(DATA_DIR), "per_sample",
                               os.path.basename(r["csv_path"]).replace(".csv", ".per_sample.json"))
        if not os.path.exists(ps_path): continue
        ps = json.load(open(ps_path))
        out = replay_one(ps, r["contact_idx_active"], cmd_fn, hole_xy_m)
        if out is None: continue
        out["csv"] = os.path.basename(r["csv_path"])
        results.append(out)

    n = len(results)
    if n == 0:
        print("no replay-able failures"); sys.exit(2)
    in_band = sum(1 for r in results if r["predicted_in_gold_band"])
    closer = sum(1 for r in results if r["dist_to_hole_delta_mm"] == r["dist_to_hole_delta_mm"] and r["dist_to_hole_delta_mm"] < 0)
    p25 = lambda key: float(np.percentile([r[key] for r in results], 25))
    p50 = lambda key: float(np.percentile([r[key] for r in results], 50))
    p75 = lambda key: float(np.percentile([r[key] for r in results], 75))

    summary = {
        "n_failures_replayed": n,
        "fraction_in_gold_disp_band": in_band / n,
        "fraction_closer_to_hole": closer / n,
        "predicted_disp_mm_at_1s": {"p25": p25("disp_predicted_mm_at_1s"),
                                    "p50": p50("disp_predicted_mm_at_1s"),
                                    "p75": p75("disp_predicted_mm_at_1s")},
        "actual_disp_mm_at_1s": {"p25": p25("disp_actual_mm_at_1s"),
                                 "p50": p50("disp_actual_mm_at_1s"),
                                 "p75": p75("disp_actual_mm_at_1s")},
        "dist_to_hole_delta_mm": {"p25": p25("dist_to_hole_delta_mm"),
                                  "p50": p50("dist_to_hole_delta_mm"),
                                  "p75": p75("dist_to_hole_delta_mm")},
        "hole_xy_m": list(hole_xy_m),
        "object": args.object,
    }
    json.dump({"summary": summary, "per_episode": results},
              open(os.path.join(staged_dir, "replay_results.json"), "w"), indent=2)

    md = ["# Replay simulation — counterfactual on 132 May-4 FAIL traces\n",
          f"**Object:** {args.object}",
          f"**Episodes replayed:** {n}",
          f"**Hole xy used:** ({hole_xy_m[0]*1000:.1f}, {hole_xy_m[1]*1000:.1f}) mm\n",
          "## Headline",
          f"- {in_band}/{n} = **{in_band/n*100:.1f}%** of failures predicted to land in GOLD displacement band [1.2, 1.7] mm @ 1 s post-contact",
          f"- {closer}/{n} = **{closer/n*100:.1f}%** of failures predicted to end CLOSER to the hole vs actual\n",
          "## Predicted vs actual displacement at t = 1 s post-contact (mm)",
          "| | p25 | p50 | p75 | GOLD median |",
          "|---|---|---|---|---|",
          f"| actual    | {summary['actual_disp_mm_at_1s']['p25']:.2f} | {summary['actual_disp_mm_at_1s']['p50']:.2f} | {summary['actual_disp_mm_at_1s']['p75']:.2f} | 1.2-1.7 |",
          f"| predicted | {summary['predicted_disp_mm_at_1s']['p25']:.2f} | {summary['predicted_disp_mm_at_1s']['p50']:.2f} | {summary['predicted_disp_mm_at_1s']['p75']:.2f} | 1.2-1.7 |\n",
          "## Caveats",
          "- Linear admittance approximation; ignores rim-contact nonlinearity",
          "- Single-step prediction (no Fz collapse cascade modeled)",
          "- Treats cmd_fx/cmd_fy actual as 0 (conservative — May-4 FSM did command non-trivial lateral, so true delta is smaller than computed)",
          "- Use as a **wrong-move filter**, NOT a success predictor"]
    with open(os.path.join(staged_dir, "REPLAY.md"), "w") as fh:
        fh.write("\n".join(md))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

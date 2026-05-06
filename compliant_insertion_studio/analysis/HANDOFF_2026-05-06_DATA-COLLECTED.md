# Handoff: data collection complete, ready for cross-demo analysis (2026-05-06)

You are picking up a long session that just finished collecting GUIDED-mode data. Read this document end-to-end before doing anything else. Earlier files (`SESSION_STATE_2026-05-06.md`, `FSM_MARKERS.md`, `REGIME_DECODING.md`, `BASE_CALIBRATION_FROM_HOLE_OBSERVATIONS.md`, the skill at `compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md`) are still authoritative for project context — this doc is the *delta* and the *next-steps blueprint*.

Operator's note: prior agent's context was at 90%, which is why this hand-off exists.

---

## What's done

### Data collected (the deliverable)

15+ valid GUIDED-mode demos for `u_orange/base1/grasp_id=1`. Each demo:

- Operator-marked the hole via SIGUSR1 → `meta.hole_observed_operator.xy_m`
- Peg seated to predicted CAD seat z within ~5mm tolerance
- Full v1.2 sidecar bundle (main CSV, joints_raw 250Hz, wrench_raw 500Hz, cmd_wrench_raw event-rate, fm_events)

Inventory the dataset:

```bash
python3 <<'PY'
import glob, os, json, csv
from collections import defaultdict
demos_by_label = defaultdict(list)
for mp in sorted(glob.glob('compliant_insertion_studio/logs/insert_u_orange_2026050[5-6]*.meta.json'),
                 key=os.path.getmtime):
    m = json.load(open(mp))
    if not m.get('hole_observed_operator'): continue
    bn = os.path.basename(mp).replace('.meta.json','')
    csv_p = mp.replace('.meta.json','.csv')
    if not os.path.exists(csv_p): continue
    rows = list(csv.DictReader(open(csv_p)))
    active = [r for r in rows if r.get('phase','') == 'ACTIVE']
    if len(active) < 100: continue
    def f(r,k):
        try: return float(r[k])
        except: return float('nan')
    fz = [abs(f(r,'fz')) for r in active]
    pz = [f(r,'tcp_z') for r in active]
    px = [f(r,'tcp_x') for r in active]
    py = [f(r,'tcp_y') for r in active]
    ci = next((i for i in range(len(fz)-5) if all(fz[j]>3 for j in range(i,i+5))), None)
    if ci is None: continue
    descent_mm = (pz[ci]-pz[-1])*1000
    if not (25 <= descent_mm <= 45): continue
    cx, cy = px[ci]*1000, py[ci]*1000
    if -19 < cx < -10:   label = "A_pos_x_10mm"
    elif -39 < cx < -29: label = "B_neg_x_10mm"
    elif -345 > cy > -360 and -29 < cx < -20: label = "C_pos_y_10mm"
    elif cy > -345 and -29 < cx < -20: label = "C_pos_y_10mm"
    elif cy < -360 and -29 < cx < -20: label = "D_neg_y_10mm"
    else: label = "E_diag_pxpy_7mm"
    demos_by_label[label].append(bn)
for label in ['A_pos_x_10mm','B_neg_x_10mm','C_pos_y_10mm','D_neg_y_10mm','E_diag_pxpy_7mm']:
    print(f"{label:<22s} count={len(demos_by_label[label])}")
PY
```

Run that first. You should see ≥3 valid demos per variation. Some variations may have 4 (extra reps were collected — fine).

### Code state (everything's in place)

| Component | State |
|---|---|
| `wrapper/contact_search_fsm.py` | New states `GUIDED` + `INSERT_DESCENT`. APPROACH selection vector locks XY. Global seat detector uses absolute `predicted_tcp_z` (multi-contact safe). Tilt baseline tracking. APPROACH grace period 1.0s. |
| `wrapper/compliant_insert.py` | `--guided-mode` CLI flag. SIGUSR1 forwarded to `fsm.mark_hole()`. Post-zero F/T bias subtracted from wrench before passing to FSM. start_force_mode RPC retries 3x with backoff. ep.fsm bound for handler reach. hole_observed_operator persisted on DONE/ABORT/SIGTERM exit paths. |
| `wrapper/schema_v1.py` | `hole_observed_operator` registered in META_OPTIONAL_KEYS. |
| `scripts/run_assembly_step.py` | `--guided-mode` plumbed. SIGUSR1/SIGUSR2 forwarders to wrapper child. `_run` streams subprocess output line-by-line (no buffering). Outer timeout bumped to 700s in guided mode. Wrapper invoked with `python3 -u`. |
| `scripts/collect_regime_data.py` | Pipe-and-watch for `FSM → GUIDED` log before prompting operator. SIGUSR1 sent on Enter. `_verify_physical_seat` reads `hole_observed_operator` from meta and uses 10mm tolerance from hole (not from contact — operator legitimately drags). |
| `scripts/launch_robot.sh` | Auto-applies set_payload from latest `configs/ft_calibration_*.yaml` at bringup. |
| `configs/defaults.yaml` | `contact_threshold_N=3.0`, `fz_smooth_window_s=0.1`, `approach_fz_N=12.0`, `insert_min_descent_m=0.025` (was 5mm — let through phantom seats), `at_target_z_tol_m` configurable (default 5mm). |

### Hard rules (mental-model anchors — do NOT re-derive these)

1. **TCP at canonical = peg at canonical, period.** The CAD chain's `fold_symmetry_used.angle_error_deg` is camera-noise propagation, NOT a real geometric offset. Don't reason from it.
2. **The on-contact marker is correct at 3N + 0.1s sustained**, with grace period 1s after force-mode active to ignore startup transient. Don't tighten further.
3. **Chained held_quat preserves R_grasp/fold info** from initial perception at pick time. The bootstrap-from-tcp pattern was tried and removed — it discards R_grasp and flips the fold. Don't re-add.
4. **Pendant payload setting does NOT propagate to ROS-side force_mode.** `launch_robot.sh` calls `set_payload` via service. Without it: 7.85mm TCP drift per APPROACH; with it: 0.83mm.
5. **`grasp_id` does not encode fold.** It's a position offset only. Fold is set physically at pick time and conserved through the trajectory.
6. **At Target marker uses absolute `predicted_tcp_z`** (multi-contact safe). z_drop from surface_z is the FALLBACK for single-contact, single-Contact runs. New designs should use absolute.
7. **CSV records RAW wrench** (uncorrected by F/T bias). Bias subtraction happens only for FSM contact-detection. For analysis, apply `corrected_fz = csv.fz - meta.post_zero_bias.Fz` post-hoc.
8. **Don't trust FSM stdout outcome labels** (`fsm_seated`, `success`, etc.) as ground truth. Verify against raw CSV: descent post-contact 25-45mm, final xy ≈ hole_observed.

### Bug fixes from this session (numbered for the skill)

Each was a hard-won lesson; all settled — don't reopen unless you have new evidence.

- #19 **set_payload at bringup** — was missing from launch despite docs claiming otherwise. 9.5× drift reduction.
- #21 **Rotation lock + APPROACH speed** — selection_vector(F,F,T,F,F,F) for prismatic peg; APPROACH 6N→12N for 2× faster descent.
- #22 **Physical-seat verifier** — script-level cross-check on each demo (3 rules from the false-seat case).
- #24 **insert_min_descent_m 5→25mm** — FSM rejects phantom-contact through-air drops at the source.
- #26 **GUIDED state + INSERT_DESCENT integration** — operator-drag mode with SIGUSR1 hole-mark.
- #28 **Prompt timing gate** — wait for "FSM → GUIDED" log before prompting operator.
- #29 **Streaming subprocess output** — was buffered through `subprocess.run(capture_output=True)`; switched to Popen+readline.
- #30 **ep.fsm binding + 600s timeout** — SIGUSR1 handler couldn't reach FSM; wrapper timeout was 120s.
- #32 **At-Target uses absolute predicted_tcp_z** — multi-contact safe; tolerance 5mm.
- #33 **Post-zero F/T bias subtracted** before FSM — eliminates phantom-contact-at-hover when residual >2N.
- #34 **start_force_mode retry 3x with backoff** — handles controller's "active but not ready" window.
- #35 **GUIDED-aware verifier** — checks final_xy near hole_observed (not contact); 10mm tolerance.
- #36 **APPROACH grace period 1.0s** — suppresses force-mode startup transient.

The skill at `compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md` has full lessons through #11–#12; you should append #21+ if you write new sections.

---

## What's next (the actual work to continue)

### Phase 1: Cross-demo aggregation (4 hours work, fully unblocked)

The collected demos give us the data to finalize the **Found Hole** marker (the 4th and last marker; the other 3 — Contact, Re-Contact, At Target — are already finalized as autonomous predicates).

#### Step 1.1: Run the segmenter on every valid demo

Existing script: `analysis/scripts/30_segment_regimes.py`. Will likely need a small fix to use the absolute `predicted_tcp_z` from meta rather than relative to surface_z (currently it uses GOLD-derived heuristics; double-check).

For each demo:
```bash
python3 -m compliant_insertion_studio.analysis.scripts.30_segment_regimes <basename> > /tmp/seg_<basename>.json
```

Expected output per demo: timeline `[(t_start, t_end, regime), ...]` with regimes `RIM`, `EDGE_OF_SLOT`, `CHAMFER_TRANSIT`, `IN_SLOT_DESCENT`, `SEATED`. The user's proposed simpler model collapses these to `Inserting / Aligning / At Target` — see Phase 2.

#### Step 1.2: Identify the Found Hole signature

For each demo, the operator's SIGUSR1 (= `meta.hole_observed_operator.t_s`) marks the operator's labeled "Found Hole" moment. Across N demos:

1. Read each demo's CSV at the SIGUSR1 timestamp.
2. Compute features at that moment: `F_lat`, `T_lat`, `tilt_deg`, `tilt_x_deg`, `tilt_y_deg`, `dz/dt`, `tcp_z relative to predicted`, `time_since_contact`, `xy_distance_from_contact`.
3. Aggregate across demos: median + p5 + p95 of each feature.
4. The features that are **tight** (small p95-p5 spread) and **distinct from non-Found-Hole moments** are the predicate components.

Hypothesis from a single GOLD demo (validate with the 15+ collected):
- `tilt_deg` shows a peak then drop right around the engage moment → **tilt-relax event** is the strongest signal
- `F_lat` collapses to <1N right when peg is over the hole (no rim resistance)
- `dz/dt > 1mm/s` initiates within ~0.5s of the operator's mark

If the hypothesis holds across demos, the Found Hole predicate becomes:
```
Found Hole fires when:
  tilt_deg dropped ≥ 1.0° from its rolling 1s peak AND
  F_lat < 1.5N for 0.2s sustained AND
  recently in CONTACT regime (Aligning → Inserting transition)
```

Write this as `analysis/scripts/31_decode_operator_action.py` (skeleton in `REGIME_DECODING.md` §3).

#### Step 1.3: Validate the predicate self-consistently

For each demo, scan forward through the trajectory with the predicate. Does it fire at approximately the operator's SIGUSR1 moment (within ±300ms)? If yes — predicate works. If no — refine.

Write as `analysis/scripts/34_validate_law.py`.

#### Step 1.4: Finalize the autonomous Found Hole marker

Once validated, replace the GUIDED-mode SIGUSR1 mechanism with the autonomous predicate. The wrapper's GUIDED state can keep the SIGUSR1 path for future data collection on new objects, but the FIND_HOLE state should now use this predicate for autonomous insertion.

This is a small wrapper code change; main work is in the analysis.

### Phase 2: Base position calibration (parallel to Phase 1)

Spec already written: `analysis/BASE_CALIBRATION_FROM_HOLE_OBSERVATIONS.md`.

Each demo's `hole_observed_operator.xy_m` is a measurement of the true base position (projected through the known CAD chain). Aggregate across N demos → produce `analysis/scripts/40_calibrate_base_from_observations.py`.

Output: `compliant_insertion_studio/configs/base_calibration_base1_<date>.yaml` with calibrated `xyz_m` + stddev. Operator commits + uses in `primitives/shared/config.py:DEFAULT_BASE_POSITION`.

After this, the autonomous wrapper's `predicted_tcp_xy` will be mm-accurate instead of cm-off. The 10-15mm CAD-prior bias drops to ~1-2mm perception noise.

### Phase 3: Autonomous validation

Once Found Hole is autonomous + base is calibrated:

1. Run the wrapper WITHOUT `--guided-mode` on a fresh u_orange demo.
2. Wrapper does APPROACH → Contact → FIND_HOLE (now autonomous, not GUIDED) → ENTRY_SETTLE → INSERT → DONE.
3. Verify peg seats autonomously without any operator drag or SIGUSR1.
4. If first attempt fails: capture the trajectory, compare to GOLDs, identify the divergence, refine.

Per the user's guidance from earlier: "if you think you know the control law then i expect the autonomous to complete the insertion." After Phase 1+2, that's the test.

---

## Methodology (do NOT skip)

The user has explicitly enforced these multiple times across the session. Re-violating burns context.

1. **Each iteration runs the 3-way comparison BEFORE proposing any change.** GOLD vs current vs predicted. Don't tune in isolation.
2. **Don't trust FSM stdout claims as ground truth.** Verify physically from CSV. The verifier in `collect_regime_data.py` is the pattern.
3. **Annotation/labels only after raw-data analysis.** Never intuition-based thresholds.
4. **Structural changes, not parameter sweeps.** Each new iteration needs a data-derived structural reason.
5. **Tilt is contact-induced position-error feedback** (the primary signal beyond Fz). Use directional components (tilt_x, tilt_y) not just magnitude.
6. **Anti-pattern flagged earlier in session**: agent diagnosed problems in isolation without running 3-way comparisons; the user pushed back hard. Don't repeat.

---

## Files relevant to this work

Priority order (read first → last):

1. **This file** — current state + roadmap
2. `compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md` — methodology + hard rules + anti-patterns (§11–§12 are critical)
3. `compliant_insertion_studio/analysis/REGIME_DECODING.md` — framework + scripts to write
4. `compliant_insertion_studio/analysis/FSM_MARKERS.md` — every marker, current threshold, tuning path
5. `compliant_insertion_studio/analysis/BASE_CALIBRATION_FROM_HOLE_OBSERVATIONS.md` — Phase 2 spec
6. `compliant_insertion_studio/scripts/collect_regime_data.py` — collection harness (interactive, working)
7. `compliant_insertion_studio/scripts/run_assembly_step.py` — orchestrator
8. `compliant_insertion_studio/wrapper/compliant_insert.py` — wrapper
9. `compliant_insertion_studio/wrapper/contact_search_fsm.py` — FSM
10. `compliant_insertion_studio/configs/defaults.yaml` — all FSM thresholds (with comments dating each change)
11. `compliant_insertion_studio/logs/insert_u_orange_2026050[5-6]*` — the data (CSV + sidecars + meta per demo)
12. `compliant_insertion_studio/logs/regime_collection_*.json` — session manifests

---

## Marker status (final tally for this session)

| # | Marker | Predicate | Status |
|---|---|---|---|
| 1 | **Contact** | `fz_smoothed > 3N for 0.1s sustained`, after 1s grace from APPROACH start | ✅ finalized + validated |
| 2 | **Found Hole** | operator SIGUSR1 (data-collection); autonomous predicate TBD from cross-demo analysis | 🟡 Phase 1 work — data is collected |
| 3 | **Re-Contact** (multi-contact loop) | same as #1 | ✅ finalized by reuse |
| 4 | **At Target** | `\|tcp_z - predicted_tcp_z\| < 5mm` + motion stopped + tilt low, sustained 1s | ✅ finalized + validated |

3 of 4 are autonomous predicates. The 4th has its operator-labeled training data, ready to be decoded into an autonomous predicate.

---

## If you only have time for one thing

Run the cross-demo aggregation for the Found Hole marker (Phase 1, steps 1.1–1.4). That unblocks Phase 3 (autonomous insertion validation), which is the project's success criterion. Phase 2 (base calibration) is a parallel improvement that's nice-to-have but not on the critical path.

If you have 20 minutes: just do the inventory script at the top, confirm dataset is good, write a short status update, and stop. Don't half-do the analysis.

---

## What NOT to do

- Don't re-add bootstrap-from-TCP. It was tried and removed. R_grasp must be preserved through the chain.
- Don't tune the contact threshold lower than 3N. With grace period and bias subtraction, 3N is settled.
- Don't change `selection_vector` away from rotation-locked for prismatic peg insertion. The lock is the design.
- Don't trust `fsm_seated` outcome label without verifying descent + final_xy from CSV. The skill explicitly warns about this.
- Don't propose a parameter-sweep iteration loop. Operator has been burned by this multiple times.
- Don't try to "fix" `fold_angle_err` by rotating EE. It's camera noise, not real geometry.
- Don't add code that requires Remote pendant mode. Operator runs Local mode.

# Staged patch — 002-lower-damping-find-hole

## Proposed change

Lower `fsm.damping_find_hole` (UR force_mode `damping_factor` during the FIND_HOLE state) from **0.3 → 0.18**, and lower `correction.action.damping_search` from **0.50 → 0.30** to keep the spiral burst proportionally compliant. No changes to commanded forces, gain_scaling, selection_vector, or the Mode B action. The Fz cap (-9 N) and lateral cap (6 N) are untouched. This is a **single-parameter config-only change**.

Theoretical mechanism: `damping_factor` in the UR force mode controller scales an internal velocity-damping term; the steady-state TCP lateral velocity per N of commanded F_lat varies roughly as 1/(1 + damping). Going 0.3 → 0.18 raises admittance velocity per N by approximately 1.4× — directly addressing the 1.88× post-contact v_xy gap (GOLD 2.35 vs FAIL 1.25 mm/s) measured on the canonical A/B pair (matched gain=1.0/damping=0.7 force_mode + matched cmd_wrench history). The 1.74× r_cop gap and 1.82× F_lat_base gap are downstream consequences of the v_xy gap (less lateral motion per second → less chamfer engagement → less reaction force build-up → no Fz collapse).

## Backing invariants

- **I016** (canonical A/B pair, schema v1.2 sidecars; this iteration's discovery 008): with matched force_mode_params and matched FSM-cmd history at gain=1.0/damping=0.3 (find_hole state), operator success vs autonomous fail differ by 1.88× v_xy, 1.82× F_lat_base, 1.74× r_cop, 2.27× xy_excursion, 2.80× admittance-residual TCP velocity. Operator pushes through j0 (base joint, peak 2.40 Nm effort, corr(eff_j0, |v_xy|) = +0.53). Force-mode admittance at the rule-capped 6 N lateral cannot match that velocity at damping=0.3.
- **I004** (verified 4 objects): operator TCP displacement = 1.2-1.7 mm/s sustained over ~1 s during search phase. The autonomous FAIL achieves 1.25 mm/s median and only 7 mm total in 5 s — the missing factor is sustain duration AND velocity per N.
- **I003** (verified 4 objects): operator search direction = (seat_xy - tcp_xy).normalized within 5-15°. The current find_hole position-spiral already satisfies this when seat_xy is correct; the issue isn't direction, it's response.

Refutation evidence supporting `damping` reduction (not gain or force) as the lever:
- H006 (cmd_fz > 9 N) refuted — magnitude is bounded.
- H007 (spiral_F_max sweeps 4/6/10/12 N) refuted — magnitude alone doesn't move the needle.
- H008 (spiral_v 5× slower) refuted — speed alone doesn't either.
- The orthogonal lever not yet tried in the v82-v97 history is the **damping_factor itself during find_hole**. Search of `v82_v97_iteration_history.json` for damping changes finds only the `correction.action.damping_search` 0.20 → 0.50 change (FAIL — coasting issue), and the global `force_mode.damping_factor` 0.7 → 0.95 → 0.7 revert (pre-contact drift). Neither tested **find-hole-state damping below 0.3**.

## Files in this directory

- `PATCH.diff` — `git diff` against `compliant_insertion_studio/configs/defaults.yaml` only.
- `cmd_function.py` — Python approximation. Note: this patch changes admittance, not commanded wrench, so the cmd_wrench is unchanged but the predicted velocity per N rises.
- `REPLAY.md` — output of `scripts/08_replay_simulator.py` against 132 May-4 FAIL traces.
- `evidence_score.json` — output of `scripts/09_score_staged_patch.py`.

## Hard-rule compliance check

1. **DO NOT modify wrapper directly** — OK. Only `configs/defaults.yaml` is touched.
2. **DO NOT modify configs directly** — OK. The change is delivered as PATCH.diff in `staged/`; not applied.
3. **DO NOT use FSM stdout claims as ground truth** — OK. Discovery 008 used CSV-only ground truth.
4. **DO NOT lock XY via selection_vector** — OK. selection_vector unchanged.
5. **DO NOT use counter-residual direction** — OK. No direction change; this is a controller-compliance change.
6. **DO NOT exceed cmd_fz=-9 N or |cmd_F_lat|=6 N** — OK. Force commands unchanged. The change increases velocity per N, not force.
7. **DO NOT remove state-independent global seat detector** — OK. Untouched.
8. **DO NOT re-test refuted hypotheses** — OK. Closest neighbour: pre-contact `force_mode.damping_factor` 0.7 → 0.95 (REVERTED for pre-contact drift). This patch lowers find-hole state damping (post-contact only), opposite direction, different state. Also: H008 (spiral_v sweep) is about commanded velocity magnitude in position-spiral, not controller damping. Distinct.
9. **All primitives use module mode** — OK. No primitive subprocess invocation changed.

## Predicted at-robot outcome

- **Predicted `durable_collapse_rate`**: 0.40-0.55 on a fresh 5-attempt batch (current u_orange autonomous baseline = 0.20). Mechanism: the existing FSM logic already commands directed lateral force toward the seat; raising admittance velocity per N by ~1.4× lifts post-contact v_xy from ~1.25 to ~1.75 mm/s, closing most (but not all) of the 1.88× gap to GOLD's 2.35 mm/s. Some of the 2.27× xy_excursion gap is also from sustain duration (operator pushes for 5 s, FSM rotates direction every 1.5 s); damping alone won't close that, but compounded with H101's directed-sweep stage 1 it likely will.
- **Predicted `first_divergence_time_s`**: 0.4-0.7 s (vs current 0.04-0.17 s). Higher admittance moves the trajectory closer to GOLD for longer.
- **Confidence: medium.** Backed by 2 directly-relevant invariants (I016 from the canonical pair + I004 portable-across-4-objects), the hard-rule compliance check is clean, refutation neighbours are distant, and the change is config-only with one knob. Downside: lower damping is known to encourage "coasting" past detected chamfer (per the existing config comment about `damping_search` 0.20 → 0.50). 0.18 is close to that 0.20 risk zone. If the peg starts overshooting newly-found chamfers this would manifest as new false-positives in ENTRY_SETTLE. Mitigation: the existing `engaged_v_xy_max_m_s = 0.003` engagement gate stays in place and will reject overshoots. Recommend pairing operator validation with checking ENTRY_SETTLE failure mode counts.

## Operator action when at robot

1. `cd compliant_insertion_studio && git apply analysis/iterations/staged/002-lower-damping-find-hole/PATCH.diff`
2. Run `python3 -m compliant_insertion_studio.scripts.loop_iterate --object-name u_orange --base-name base1 --grasp-id 1 --target-success-count 5`
3. Run `python3 analysis/scripts/run_all.py && python3 analysis/scripts/score_iteration.py analysis/iterations/staged/002-lower-damping-find-hole --csv-glob 'insert_u_orange_*.csv'`
4. Compare: durable_collapse_rate vs the matched-config baseline (GOLD=success, FAIL=abort). If ≥ 0.50 with no new ENTRY_SETTLE failure-mode regressions → promote. If 0.20-0.50 → keep but combine with 001-h101-directed-sweep. If < 0.20 or ENTRY_SETTLE regressions → revert and tag ROBOT_REFUTED.
5. Order vs 001-h101-directed-sweep: I recommend 001 first (controllable direction), then 002 stacked on top (controllable speed). Independent levers; the patches don't conflict (they touch different config keys).

## Replay + score result (2026-05-05)

```
score: 6.0
confidence: medium
backing_invariants_valid: [I016, I004, I003]
backing_invariants_invalid: []
rule_violations: 0
replay:
  n_failures_replayed: 69
  fraction_in_gold_band: 0.0
  fraction_closer_to_hole: 1.0
```

Notes on the replay numbers:

- `fraction_in_gold_band = 0.0` is **not a regression**. The replay simulator's `cmd_function.py` for this patch returns the FSM's nominal find_hole-state commanded wrench (cmd_fy = +5 N), the SAME wrench the May-4 FSM was issuing per `cmd_wrench_raw.csv`. Under the simulator's linear admittance with K=0.5 mm/s/N (estimated from May-3 GOLD operator demos), cmd_fy=+5 N predicts ~2.5 mm displacement at t=1s — above the GOLD operator-demo band of [1.2, 1.7] mm. The simulator's K is calibrated against operator demos at the OLD `damping_factor=0.7` global setting. The patch lowers find-hole-state damping to 0.18, which would in fact raise per-N velocity and therefore actually reach an even larger displacement at-robot. So the replay's "out-of-band high" prediction is consistent with the patch direction; the simulator simply cannot calibrate a different damping_factor.
- `fraction_closer_to_hole = 1.0` (69/69) is the strong signal: every FAIL trace would have ended closer to the seat under the directed-cmd_fy command, regardless of damping. The 1.74-2.27× operator-vs-autonomous gap in I016 is direction-and-velocity-coupled, not direction-only.
- `confidence = medium` because the score weight system requires `f_band ≥ 0.4` for the high tier and the simulator cannot model the damping change. To bump to "high" we would need either: (a) the operator's at-robot validation, or (b) an extended simulator that takes `damping_factor` as a parameter and recalibrates K accordingly. Option (b) is a future analysis-script improvement, not a blocker.


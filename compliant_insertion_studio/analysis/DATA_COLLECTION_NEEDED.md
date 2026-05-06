# Data collection needed to advance regime-decoding

This is the immediate-next collection action that follows from REGIME_DECODING.md. Read that document first for the full framework. This file is the concrete to-do for the operator at the robot.

---

## What we have

- 1 v1.2 GOLD demo: `insert_u_orange_20260504_113809.csv` (5-file sidecar bundle present)
- ~60 v1.1 GOLD demos: schema doesn't include native-rate wrench / cmd_wrench / fm_events sidecars. Useful for sanity but NOT sufficient for decoding the law.
- 4 v1.2 May-5 episodes: a mix of GOLD and FAIL.

## What's blocking us

The segmenter (analysis/scripts/30_segment_regimes.py) successfully partitions one GOLD demo into regimes. But:

- We can't tune the detector thresholds (TH dict in the script) against a single trajectory — we'd be overfitting.
- We can't derive per-regime *operator direction rules* (pushing direction, force magnitude, commitment duration) from a single example because we'd be reading off N=1.
- We can't characterize *transition triggers* without seeing how operator behavior varies across initial conditions.

## What to collect

### Priority 1 — same-condition repeatability (need first)

**5 v1.2 GOLD demos at u_orange / base1 / grasp_id=1 with the part placed at the *same* starting xy.** This establishes the operator's intrinsic variability vs. signal repeatability. Without this baseline, we can't separate "operator did differently on purpose" from "noise."

CLI per demo:
```bash
python3 -m compliant_insertion_studio.scripts.run_assembly_step \
  --object-name u_orange --base-name base1 --grasp-id 1 \
  --grasp-width 35 --mode real
```

Check after each demo:
```bash
ls -la compliant_insertion_studio/logs/insert_u_orange_*.{csv,joints_raw.csv,wrench_raw.csv,cmd_wrench_raw.csv,fm_events.csv,meta.json}
```

All 5 sidecars must exist for each demo. If any sidecar is missing, the wrapper isn't writing v1.2 output — check schema_v1.py.

### Priority 2 — varied initial xy offset, ALL directions (need next)

The point of this variation is NOT to test the same correction repeatedly with longer slides. It's to capture **which-side-to-correct-from** behavior. The operator's RIM direction depends on which side the peg lands relative to the slot. With samples in only one quadrant, we'd over-fit a single correction direction; with all 4 sides + diagonals, we can decode `direction(sensor_state) → correction_xy_unit` as a function and validate it generalizes.

**Place the part at varied starting xy positions BY DIRECTION, then by magnitude.** What matters is which side of the slot the peg makes first contact — operator's correction direction is the diagnostic signal.

| Starting peg position relative to slot opening | N demos |
|---|---|
| **+X side** at ~5mm and ~10mm offset | 3 |
| **-X side** at ~5mm and ~10mm offset | 3 |
| **+Y side** at ~5mm and ~10mm offset | 3 |
| **-Y side** at ~5mm and ~10mm offset | 3 |
| **diagonal (+X+Y, +X-Y, -X+Y, -X-Y)** at ~7mm offset | 4 (1 per quadrant) |

Total: 16 demos at varied directions. Combined with Priority 1 (5 same-condition), that's **21 demos total** for the first complete pass.

Operator process per demo:
1. Place part at the intended starting offset (visual estimate of which side, mm offset)
2. Run the standard CLI:
   ```bash
   python3 -m compliant_insertion_studio.scripts.run_assembly_step \
     --object-name u_orange --base-name base1 --grasp-id 1 \
     --grasp-width 35 --mode real
   ```
3. Note in the log: `+X / -X / +Y / -Y / +X+Y / +X-Y / -X+Y / -X-Y`, approximate offset mm, demo basename

The DECODER's job is to verify: across these demos, does the operator's RIM-direction unit vector point reliably from `peg_first_contact_xy` toward `slot_xy`? If yes, we have a generalizable rule. If no (e.g., operator always pushes -X regardless of which side), the behavior is more complex than direction-toward-slot.

### Priority 3 — other objects (after u_orange is decoded)

5 demos each for `u_brown`, `line_green`, `inverted_u_yellow` at grasp_id=1 to test that the law's structure (regimes + per-regime rules) generalizes. Per-object thresholds may differ; the regime *count* and *order* should not.

### Total for first complete pass

- 5 same-condition + 15 varied-offset = **20 v1.2 GOLD u_orange demos**
- ~40 minutes operator time at the robot

This unblocks tasks 10, 11, 12 in the regime-decoding pipeline (per-regime direction, transition triggers, synthesized law).

---

## What to record per demo (operator note)

Quick text note (not sensor data) per demo, in `analysis/data_collection_log.txt`:

```
2026-05-06 14:23   insert_u_orange_20260506_142345
  initial_offset_mm: ~8 (visual)
  difficulty (1-5): 2
  notes: clean slide, peg dropped on first chamfer engagement
```

---

## Sanity checks for the operator session

Before each batch:

1. F/T sensor smoke test: `bash compliant_insertion_studio/scripts/ft_smoke_test.sh` (if it exists; otherwise just zero F/T and visually verify ~0 baseline at face-down EE)
2. Confirm aruco perception is publishing both topics (per SETUP.md §3.2-3.3)
3. Confirm payload is set to 2.11 kg in the pendant (CLAUDE.md note)

After each demo:

1. Check sidecars exist (5 files per basename)
2. Check the meta.json's `cad_prediction.predicted_tcp_at_seat.xyz_m` field is populated
3. Run the segmenter to verify the segmentation is sensible:
   ```bash
   python3 compliant_insertion_studio/analysis/scripts/30_segment_regimes.py <basename>
   ```
   Expect: APPROACH → RIM → ... → SEATED. If many UNKNOWN segments dominate, the detector thresholds need adjusting after the dataset is collected.

---

## What I'll do once the data exists

1. Run the segmenter on all collected demos.
2. Cross-demo aggregation: per-regime stats (median direction unit vector, force magnitude, commitment duration, tilt range) with stddev across demos.
3. Tune detector thresholds to maximize segmentation consistency (% of samples that classify into a regime, not UNKNOWN).
4. Decode regime-transition triggers (which signals reliably cross which thresholds at boundaries).
5. Output the synthesized control law as `derived_law.yaml` per REGIME_DECODING.md §5.
6. Validate against a held-out demo not used for tuning.
7. Translate the law into FSM code changes.

None of this requires more code from me right now; the pipeline scaffolding is in place. The blocker is data.
